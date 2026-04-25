#!/usr/bin/env python3
"""
Interface Analysis Module - extracts and quantifies interface properties from predicted protein-protein complexes.
Designed as both a standalone CLI tool and an importable module for integration into the batch processing pipeline.

Features (structural):
    - Interface contact identification (CB-CB distance threshold)
    - Interface geometry: contact count, fractions, symmetry, density
    - Interface-specific pLDDT vs bulk pLDDT comparison
    - Paradox complex detection (high interface confidence + global disorder)

Additional features (PAE-aware, requires PKL data):
    - Interface PAE mapping and statistics (bidirectional max over PAE[A,B]/PAE[B,A])
    - PAE-only confident contacts (PAE < 5Å) and strict confident contacts (PAE < 5Å AND both pLDDT >= 70)
    - Computational hot-spot residue extraction

Composite interface confidence score (heuristic, not calibrated):
    score = WEIGHT_PLDDT * f(interface_plddt_combined)
          + WEIGHT_PAE   * strict_confident_contact_fraction
          + WEIGHT_SYMMETRY * interface_symmetry
          + WEIGHT_DENSITY  * min(contacts_per_interface_residue / DENSITY_NORMALIZATION, 1.0)

    The composite is a screening heuristic, NOT a calibrated estimate of interface correctness.
    Weights are expert-chosen, partially informed by the 9,573-complex dataset distribution,
    but have NOT been fitted against DockQ, pDockQ2, or any benchmarked ground truth.

    - pLDDT and strict-confident-contact fraction are the confidence-bearing components.
    - Interface symmetry is a geometric plausibility feature, not a confidence feature.
      Asymmetric interfaces can be biologically real (enzyme-substrate, peptide-domain,
      antibody-antigen, hub proteins binding short linear motifs), so low symmetry
      should not be read as low confidence.
    - Contact density is a packing plausibility feature, not a confidence feature.
      Dense interfaces can still be misdocked; sparse interfaces can be biologically real
      for transient or motif-mediated binding.

Usage (standalone):
    python interface_analysis.py --pdb structure.pdb
    python interface_analysis.py --pdb structure.pdb --json output.json
    python interface_analysis.py --pdb structure.pdb --pkl result.pkl

Usage (as importable module):
    from interface_analysis import analyse_interface, analyse_interface_with_pae
    result = analyse_interface("structure.pdb", threshold=8.0)
    result_pae = analyse_interface_with_pae("structure.pdb", pae_matrix, chain_lengths)
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Optional, Union
import numpy as np

# Import from refactored pdockq module
from pdockq import (
    read_pdb_Edited as read_pdb,
    read_pdb_with_chain_info_New as read_pdb_with_chain_info,
    calc_pdockq_and_contacts_New as calc_pdockq_and_contacts,
    compute_pae_chain_offsets_New as compute_pae_chain_offsets,
    find_best_chain_pair_New as find_best_chain_pair,
    ContactResult_New as ContactResult,
    DEFAULT_CONTACT_THRESHOLD_New as DEFAULT_CONTACT_THRESHOLD,
    PairContactResult,
    compute_all_chain_pairs,
    calc_pdockq_whole_complex,
    ChainInfo_New,
)

from dataclasses import dataclass, field, asdict

#----------------------------Constants-------------------------------------

# Interface pLDDT thresholds
INTERFACE_PLDDT_HIGH = 70       # Residues above this are confidently placed
INTERFACE_PLDDT_VERY_HIGH = 90  # Very high confidence

# PAE confidence threshold (Ångströms) - literature-grounded (PMC11956457)
PAE_CONFIDENT_THRESHOLD = 5.0

# Geometry thresholds for flagging
MIN_INTERFACE_CONTACTS = 5          # Below this, interface is negligible

# Calibrated from 9,573-complex dataset to flag unusually sparse interfaces for contacts_per_interface_residue below this value (20th percentile density = 1.15 among contacts >= 5)
SPARSE_INTERFACE_DENSITY = 1.15

ASYMMETRIC_INTERFACE_RATIO = 0.5    # interface_symmetry below this is flagged as asymmetric

# Disorder thresholds
SUBSTANTIAL_DISORDER_FRACTION = 0.3  # >30% residues below pLDDT 50
PLDDT_DISORDER_THRESHOLD = 50

# Composite score normalisation (calibrated from 9,573-complex dataset)
# Density is a packing plausibility feature, not a confidence feature:
# dense interfaces can still be misdocked and sparse interfaces can be biologically real.
DENSITY_NORMALIZATION = 2.0          # density / 2.0, capped at 1.0 (95th percentile density = 1.70; rounded up to 2.0)

# Composite score weights - HEURISTIC, not fitted against DockQ/pDockQ2 or any benchmarked
# ground truth. Ordering is expert-chosen, partially informed by the 9,573-complex dataset:
#   - interface pLDDT and strict confident-contact fraction are treated as the
#     confidence-bearing components
#   - interface symmetry is a geometric plausibility feature (asymmetric interfaces can be
#     biologically real - e.g. enzyme-substrate, peptide-domain, antibody-antigen, hub-motif)
#   - contacts-per-interface-residue density is a packing plausibility feature
# Any claim about calibrated correctness would require benchmarking against DockQ, pDockQ2,
# or known positive/negative PPIs and is out of scope for this toolkit.
WEIGHT_PLDDT = 0.35
WEIGHT_PAE = 0.35
WEIGHT_SYMMETRY = 0.15
WEIGHT_DENSITY = 0.15

# Paradox detection thresholds (High-quality + substantial disorder)
PARADOX_IPTM_THRESHOLD = 0.75
PARADOX_PDOCKQ_THRESHOLD = 0.5
PARADOX_CONFIDENT_CONTACT_GENUINE = 0.73  # above -> likely genuine binding (calibrated from 9,573-complex dataset: median of 138 paradox complexes)
PARADOX_CONFIDENT_CONTACT_ARTEFACT = 0.50 # below -> likely artefactual (calibrated from 9,573-complex dataset: 25th percentile of 138 paradox complexes)

# Metric disagreement threshold (ipTM vs pDockQ)
# All disagreement cases in 9,573-complex dataset are ipTM >> pDockQ, confirming pDockQ is systematically more stringent - often penalising genuine interfaces in disordered complexes.
METRIC_DISAGREEMENT_THRESHOLD = 0.52  # 90th percentile of |iptm - pdockq| distribution

# Bidirectional PAE handling
# PAE is a directional quantity: pae[i, j] is the expected position error at residue i when the
# predicted and actual structures are aligned on residue j. For an inter-chain contact (a, b)
# we take the elementwise max of pae[a, b] and pae[b, a] so that a contact is only considered
# confidently placed if BOTH directional alignments agree.
USE_BIDIRECTIONAL_PAE = True

# pLDDT component floor for the composite score
# Prevents complexes with very-low-pLDDT interfaces from collapsing the pLDDT term to 0,
# which would hide information carried by the PAE / symmetry / density components.
PLDDT_COMPONENT_FLOOR = 0.05


def normalise_interface_plddt(plddt: Optional[float]) -> Optional[float]:
    """Map interface pLDDT to a 0-1 confidence component for the composite score.

    Continuous, band-aware transform anchored on AlphaFold confidence conventions:
        - pLDDT <= 50      -> floor (PLDDT_COMPONENT_FLOOR = 0.05); AlphaFold's "very low
                              confidence" region is compressed to the floor, not given
                              proportional credit.
        - 50 <  pLDDT < 90 -> linear ramp from floor up to 1.0 across the low/high bands.
        - pLDDT >= 90      -> 1.0; "very high confidence" region saturates.

    The function is continuous in pLDDT (pLDDT itself is continuous, so the normaliser
    should be too - a residue moving from pLDDT 49.9 to 50.1 should not produce a score jump).

    Shape reference:
        pLDDT = 50  -> 0.05 (floor)
        pLDDT = 52  -> 0.05 (still floored; raw = 0.05 clips to floor)
        pLDDT = 60  -> 0.25
        pLDDT = 70  -> 0.50
        pLDDT = 80  -> 0.75
        pLDDT = 90  -> 1.00
        pLDDT = 100 -> 1.00 (capped)

    Args:
        plddt: Interface pLDDT on the AlphaFold 0-100 scale. None is passed through.

    Returns:
        A float in [PLDDT_COMPONENT_FLOOR, 1.0], or None if plddt is None.
    """
    if plddt is None:
        return None
    raw = (plddt - 50.0) / 40.0
    return round(float(np.clip(raw, PLDDT_COMPONENT_FLOOR, 1.0)), 4)


#-----------------Interface Contact Identification----------------------------

def identify_interface_contacts(chain_coords: dict[str, np.ndarray], chain_plddt: dict[str, np.ndarray], threshold: float = DEFAULT_CONTACT_THRESHOLD) -> ContactResult:
    """Identify all inter-chain contacts and compute pDockQ in one pass.
    Wraps calc_pdockq_and_contacts() to provide the unified contact extraction used by all downstream interface analysis functions.
    Args:
        chain_coords: Dict mapping chain IDs to (N, 3) coordinate arrays.
        chain_plddt: Dict mapping chain IDs to (N,) pLDDT arrays.
        threshold: Distance cutoff in Ångströms.
    Returns:
        ContactResult with contacts, distances, interface sets, and pDockQ.
    """
    if len(chain_coords) < 2:
        return ContactResult()
    return calc_pdockq_and_contacts(chain_coords, chain_plddt, t=threshold)

#-----------Interface Geometry Features--------------------------------------

def compute_interface_geometry(contact_result: ContactResult) -> dict:
    """Compute geometric properties of the protein-protein interface.
    Produces metrics that characterise interface size, shape, and symmetry.
    These features help distinguish genuine biological interfaces (typically large, symmetric, dense) from crystal-packing artefacts or prediction errors (small, asymmetric, sparse).
    Args:
        contact_result: ContactResult from identify_interface_contacts().
    Returns:
        Dictionary with keys:
            n_interface_contacts: Total CB-CB contact pairs across chains.
            n_interface_residues_a: Unique interface residues on chain A.
            n_interface_residues_b: Unique interface residues on chain B.
            n_interface_residues_total: Combined unique interface residues.
            interface_fraction_a: Fraction of chain A residues at interface.
            interface_fraction_b: Fraction of chain B residues at interface.
            interface_symmetry: min(frac_a, frac_b) / max(frac_a, frac_b) [1.0 is perfectly symmetric and 0.0 is one-sided].
            contacts_per_interface_residue: Average contacts per interface residue (density measure).
            mean_contact_distance: Mean distance of interface contacts (Å).
            min_contact_distance: Closest inter-chain contact distance (Å).
    """
    n_contacts = contact_result.n_interface_contacts
    n_if_a = len(contact_result.interface_residues_a)
    n_if_b = len(contact_result.interface_residues_b)
    n_if_total = n_if_a + n_if_b
    n_res_a = contact_result.n_residues_a
    n_res_b = contact_result.n_residues_b

    # Interface fractions (guard against zero-length chains)
    frac_a = n_if_a / n_res_a if n_res_a > 0 else 0.0
    frac_b = n_if_b / n_res_b if n_res_b > 0 else 0.0

    # Symmetry: ratio of smaller to larger interface fraction
    max_frac = max(frac_a, frac_b)
    symmetry = min(frac_a, frac_b) / max_frac if max_frac > 0 else 0.0

    # Contact density
    density = n_contacts / n_if_total if n_if_total > 0 else 0.0

    # Distance statistics
    if len(contact_result.contact_distances) > 0:
        mean_dist = float(np.mean(contact_result.contact_distances))
        min_dist = float(np.min(contact_result.contact_distances))
    else:
        mean_dist = None
        min_dist = None

    return {
        'n_interface_contacts': n_contacts,
        'n_interface_residues_a': n_if_a,
        'n_interface_residues_b': n_if_b,
        'n_interface_residues_total': n_if_total,
        'interface_fraction_a': round(frac_a, 4),
        'interface_fraction_b': round(frac_b, 4),
        'interface_symmetry': round(symmetry, 4),
        'contacts_per_interface_residue': round(density, 4),
        'mean_contact_distance': round(mean_dist, 2) if mean_dist is not None else None,
        'min_contact_distance': round(min_dist, 2) if min_dist is not None else None,
    }

#----------------------Interface-Specific pLDDT-------------------------------------

def compute_interface_plddt(contact_result: ContactResult) -> dict:
    """Compare pLDDT confidence at the interface vs the bulk of each chain.
    This is the computational replacement for PyMOL eyeballing.  
    A positive interface_vs_bulk_delta indicates the interface is MORE confident than the overall structure - characteristic of genuine interactions and the "paradox complexes" where disordered proteins fold upon binding.
    Args:
        contact_result: ContactResult from identify_interface_contacts().
    Returns:
        Dictionary with keys:
            interface_plddt_a: Mean pLDDT of chain A interface residues.
            interface_plddt_b: Mean pLDDT of chain B interface residues.
            interface_plddt_combined: Mean pLDDT across all interface residues.
            bulk_plddt_a: Mean pLDDT of chain A non-interface residues.
            bulk_plddt_b: Mean pLDDT of chain B non-interface residues.
            bulk_plddt_combined: Mean pLDDT of all non-interface residues.
            interface_vs_bulk_delta: interface_combined - bulk_combined.
            interface_plddt_high_fraction: Fraction of interface residues with pLDDT >= 70.
    """
    if contact_result.n_interface_contacts == 0:
        return {
            'interface_plddt_a': None,
            'interface_plddt_b': None,
            'interface_plddt_combined': None,
            'bulk_plddt_a': None,
            'bulk_plddt_b': None,
            'bulk_plddt_combined': None,
            'interface_vs_bulk_delta': None,
            'interface_plddt_high_fraction': None,
        }

    plddt_a = contact_result.plddt_a
    plddt_b = contact_result.plddt_b
    if_indices_a = sorted(contact_result.interface_residues_a)
    if_indices_b = sorted(contact_result.interface_residues_b)

    # Interface pLDDT per chain
    if_plddt_a = plddt_a[if_indices_a]
    if_plddt_b = plddt_b[if_indices_b]
    if_plddt_all = np.concatenate([if_plddt_a, if_plddt_b])

    # Bulk (non-interface) pLDDT per chain
    bulk_mask_a = np.ones(len(plddt_a), dtype=bool)
    bulk_mask_a[if_indices_a] = False
    bulk_mask_b = np.ones(len(plddt_b), dtype=bool)
    bulk_mask_b[if_indices_b] = False

    bulk_plddt_a = plddt_a[bulk_mask_a]
    bulk_plddt_b = plddt_b[bulk_mask_b]
    bulk_plddt_all = np.concatenate([bulk_plddt_a, bulk_plddt_b])

    # Compute means (guard against empty bulk arrays for tiny chains)
    mean_if_a = float(np.mean(if_plddt_a))
    mean_if_b = float(np.mean(if_plddt_b))
    mean_if_combined = float(np.mean(if_plddt_all))

    mean_bulk_a = float(np.mean(bulk_plddt_a)) if len(bulk_plddt_a) > 0 else None
    mean_bulk_b = float(np.mean(bulk_plddt_b)) if len(bulk_plddt_b) > 0 else None
    mean_bulk_combined = float(np.mean(bulk_plddt_all)) if len(bulk_plddt_all) > 0 else None

    # Delta: how much better the interface is than the bulk
    if mean_bulk_combined is not None:
        delta = mean_if_combined - mean_bulk_combined
    else:
        delta = None

    # Fraction of interface residues with high confidence
    high_fraction = float(np.sum(if_plddt_all >= INTERFACE_PLDDT_HIGH) / len(if_plddt_all))

    return {
        'interface_plddt_a': round(mean_if_a, 2),
        'interface_plddt_b': round(mean_if_b, 2),
        'interface_plddt_combined': round(mean_if_combined, 2),
        'bulk_plddt_a': round(mean_bulk_a, 2) if mean_bulk_a is not None else None,
        'bulk_plddt_b': round(mean_bulk_b, 2) if mean_bulk_b is not None else None,
        'bulk_plddt_combined': round(mean_bulk_combined, 2) if mean_bulk_combined is not None else None,
        'interface_vs_bulk_delta': round(delta, 2) if delta is not None else None,
        'interface_plddt_high_fraction': round(high_fraction, 4),
    }

#-----------------Interface PAE Mapping------------------------------------------

def _compute_interface_pae_indices(contact_result: ContactResult, pae_matrix: np.ndarray, chain_lengths: Optional[tuple[int, int]] = None, *,
    chain_offsets: Optional[tuple[int, int]] = None,
    cb_to_ca_maps: Optional[tuple[list[int], list[int]]] = None) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Map interface contacts to (row, col) indices in the PAE matrix.

    Internal helper shared by extract_interface_pae() and compute_interface_pae_features()
    so that bidirectional PAE lookup does not duplicate the index-mapping logic.

    Returns (pae_row_indices, pae_col_indices) both as 1D int arrays, or None if the
    PAE matrix dimensions don't match or mapping fails or there are no contacts.
    """
    if contact_result.n_interface_contacts == 0:
        return None

    # Resolve chain offsets
    if chain_offsets is not None:
        offset_a, offset_b = chain_offsets
    elif chain_lengths is not None:
        # Legacy dimer path: chain A starts at 0, chain B starts at len_A
        len_a, len_b = chain_lengths
        expected_total = len_a + len_b
        if pae_matrix.shape[0] != expected_total:
            return None
        offset_a, offset_b = 0, len_a
    else:
        return None

    # Map CB contact indices -> PAE matrix indices
    contacts = contact_result.contacts

    if cb_to_ca_maps is not None:
        map_a, map_b = cb_to_ca_maps
        try:
            pae_row_indices = np.array([offset_a + map_a[idx] for idx in contacts[:, 0]])
            pae_col_indices = np.array([offset_b + map_b[idx] for idx in contacts[:, 1]])
        except (IndexError, KeyError):
            return None
    else:
        # Direct mapping: CB index == CA index (no mismatch)
        # Validate dimensions for the direct case
        if chain_lengths is not None:
            len_a, len_b = chain_lengths
            if (contact_result.n_residues_a != len_a
                    or contact_result.n_residues_b != len_b):
                return None
        pae_row_indices = contacts[:, 0] + offset_a
        pae_col_indices = contacts[:, 1] + offset_b

    # Bounds check against PAE matrix
    max_idx = pae_matrix.shape[0]
    if (np.any(pae_row_indices >= max_idx) or np.any(pae_col_indices >= max_idx)
            or np.any(pae_row_indices < 0) or np.any(pae_col_indices < 0)):
        return None

    return pae_row_indices, pae_col_indices


def extract_interface_pae(contact_result: ContactResult, pae_matrix: np.ndarray, chain_lengths: Optional[tuple[int, int]] = None, *,
    chain_offsets: Optional[tuple[int, int]] = None,
    cb_to_ca_maps: Optional[tuple[list[int], list[int]]] = None) -> Optional[np.ndarray]:
    """Map interface contacts to the PAE matrix and extract inter-chain PAE values.

    PAE is directional: pae[i, j] is the expected position error at residue i when the
    predicted and actual structures are aligned on residue j. When USE_BIDIRECTIONAL_PAE is
    True (default), this function returns the elementwise maximum of pae[a, b] and pae[b, a]
    so that a contact is only considered confidently placed if BOTH directional alignments
    agree. Use compute_interface_pae_features() if you need the forward and reverse arrays
    separately.

    The PAE matrix from AlphaFold2 has shape (N_total, N_total) where N_total is the sum of
    residues across ALL chains. This function handles:
      - Multi-chain complexes: chain_offsets specify where each chain starts in the PAE matrix.
      - CB mismatch: cb_to_ca_maps translate CB-based contact indices into full-residue (CA-based)
        indices matching the PAE matrix.

    Args:
        contact_result: ContactResult with interface contact indices.
        pae_matrix: Full PAE matrix from the PKL file, shape (N, N).
        chain_lengths: DEPRECATED for multi-chain use. Tuple of (n_residues_chain_A, n_residues_chain_B) for the dimer case. Used only as fallback when chain_offsets is not provided.
        chain_offsets: Tuple of (offset_A, offset_B) - the starting row/column index of each chain in the PAE matrix. For a dimer this is (0, len_A); for multi-chain complexes, computed by compute_pae_chain_offsets().
        cb_to_ca_maps: Tuple of (map_A, map_B) where map_A[i] gives the full-residue (CA) index of CB-array position i for chain A. When None, direct (identity) mapping is assumed.
    Returns:
        1D array of PAE values at each interface contact (bidirectional max when
        USE_BIDIRECTIONAL_PAE is True, otherwise forward only), or None if the PAE matrix
        dimensions don't match or mapping fails.
    """
    indices = _compute_interface_pae_indices(contact_result, pae_matrix, chain_lengths,
                                             chain_offsets=chain_offsets, cb_to_ca_maps=cb_to_ca_maps)
    if indices is None:
        return None
    pae_row_indices, pae_col_indices = indices
    forward = pae_matrix[pae_row_indices, pae_col_indices]
    if not USE_BIDIRECTIONAL_PAE:
        return forward
    reverse = pae_matrix[pae_col_indices, pae_row_indices]
    return np.maximum(forward, reverse)

def compute_interface_pae_features(contact_result: ContactResult, pae_matrix: np.ndarray, chain_lengths: Optional[tuple[int, int]] = None, *,
    chain_offsets: Optional[tuple[int, int]] = None,
    cb_to_ca_maps: Optional[tuple[list[int], list[int]]] = None) -> dict:
    """Compute PAE-derived interface quality features.

    These features complement pLDDT-based analysis by assessing the predicted relative
    positioning between residue pairs, not just per-residue confidence.

    Two confident-contact fractions are returned:
      - pae_confident_contact_fraction: PAE < 5A only (fraction of contacts whose inter-chain
        positioning is confident). Historical metric; used by paradox-detection thresholds
        that were calibrated against this definition.
      - strict_confident_contact_fraction: PAE < 5A AND both residue pLDDT >= 70. Stricter
        definition used by the composite interface confidence score.

    Directional diagnostics (not part of the composite score; reported for the methods
    section): forward_mean is the mean of pae[a_idx, b_idx], reverse_mean is the mean of
    pae[b_idx, a_idx], directional_delta_mean / _max summarise abs(forward - reverse).

    Args:
        contact_result: ContactResult from identify_interface_contacts().
        pae_matrix: Full PAE matrix from the PKL file.
        chain_lengths: Tuple of (n_residues_chain_A, n_residues_chain_B). Used as fallback when chain_offsets is not provided.
        chain_offsets: Tuple of (offset_A, offset_B) in the PAE matrix.
        cb_to_ca_maps: Tuple of (map_A, map_B) for CB->CA index translation.
    Returns:
        Dictionary with keys:
            interface_pae_mean, interface_pae_median, interface_pae_min, interface_pae_max:
                Stats over the combined (bidirectional-max) per-contact PAE array.
            n_pae_confident_contacts, pae_confident_contact_fraction:
                PAE-only confident contacts (combined PAE < 5A).
            n_strict_confident_contacts, strict_confident_contact_fraction:
                Strict confident contacts (combined PAE < 5A AND both pLDDT >= 70).
            cross_chain_pae_mean: Mean of the full cross-chain PAE block (not just contacts -
                captures overall chain-chain relative positioning).
            interface_pae_forward_mean, interface_pae_reverse_mean: Mean PAE in each direction.
            interface_pae_directional_delta_mean, interface_pae_directional_delta_max:
                abs(forward - reverse) stats - diagnostic for directional disagreement.
    """
    empty = {
        'interface_pae_mean': None,
        'interface_pae_median': None,
        'interface_pae_min': None,
        'interface_pae_max': None,
        'n_pae_confident_contacts': None,
        'pae_confident_contact_fraction': None,
        'n_strict_confident_contacts': None,
        'strict_confident_contact_fraction': None,
        'cross_chain_pae_mean': None,
        'interface_pae_forward_mean': None,
        'interface_pae_reverse_mean': None,
        'interface_pae_directional_delta_mean': None,
        'interface_pae_directional_delta_max': None,
    }

    indices = _compute_interface_pae_indices(contact_result, pae_matrix, chain_lengths,
                                             chain_offsets=chain_offsets, cb_to_ca_maps=cb_to_ca_maps)
    if indices is None:
        return empty
    pae_row_indices, pae_col_indices = indices
    if len(pae_row_indices) == 0:
        return empty

    # Directional PAE lookups
    forward = pae_matrix[pae_row_indices, pae_col_indices]
    reverse = pae_matrix[pae_col_indices, pae_row_indices]
    combined = np.maximum(forward, reverse) if USE_BIDIRECTIONAL_PAE else forward

    # Determine offsets for cross-chain PAE block extraction
    if chain_offsets is not None:
        off_a, off_b = chain_offsets
    elif chain_lengths is not None:
        off_a, off_b = 0, chain_lengths[0]
    else:
        return empty

    # Determine chain extents in the PAE matrix for cross-chain block
    if cb_to_ca_maps is not None:
        ca_len_a = max(cb_to_ca_maps[0]) + 1 if cb_to_ca_maps[0] else contact_result.n_residues_a
        ca_len_b = max(cb_to_ca_maps[1]) + 1 if cb_to_ca_maps[1] else contact_result.n_residues_b
    elif chain_lengths is not None:
        ca_len_a, ca_len_b = chain_lengths
    else:
        ca_len_a = contact_result.n_residues_a
        ca_len_b = contact_result.n_residues_b

    # Cross-chain PAE block (all A-vs-B pairs, not just contacts)
    cross_chain_block = pae_matrix[off_a:off_a + ca_len_a, off_b:off_b + ca_len_b]
    cross_chain_mean = float(np.mean(cross_chain_block))

    # PAE-only confident contacts (uses combined PAE)
    n_confident = int(np.sum(combined < PAE_CONFIDENT_THRESHOLD))
    n_total = len(combined)

    # Strict confident contacts (combined PAE AND both pLDDT >= threshold)
    contacts = contact_result.contacts
    plddt_a_vals = contact_result.plddt_a[contacts[:, 0]]
    plddt_b_vals = contact_result.plddt_b[contacts[:, 1]]
    strict_mask = (
        (combined < PAE_CONFIDENT_THRESHOLD)
        & (plddt_a_vals >= INTERFACE_PLDDT_HIGH)
        & (plddt_b_vals >= INTERFACE_PLDDT_HIGH)
    )
    n_strict = int(np.sum(strict_mask))

    # Directional disagreement diagnostics
    directional_abs_delta = np.abs(forward - reverse)

    return {
        'interface_pae_mean': round(float(np.mean(combined)), 2),
        'interface_pae_median': round(float(np.median(combined)), 2),
        'interface_pae_min': round(float(np.min(combined)), 2),
        'interface_pae_max': round(float(np.max(combined)), 2),
        'n_pae_confident_contacts': n_confident,
        'pae_confident_contact_fraction': round(n_confident / n_total, 4),
        'n_strict_confident_contacts': n_strict,
        'strict_confident_contact_fraction': round(n_strict / n_total, 4),
        'cross_chain_pae_mean': round(cross_chain_mean, 2),
        'interface_pae_forward_mean': round(float(np.mean(forward)), 2),
        'interface_pae_reverse_mean': round(float(np.mean(reverse)), 2),
        'interface_pae_directional_delta_mean': round(float(np.mean(directional_abs_delta)), 2),
        'interface_pae_directional_delta_max': round(float(np.max(directional_abs_delta)), 2),
    }

#-----------------Confident Interface Residues (Computational Hot Spots)------------

def identify_confident_interface_residues(contact_result: ContactResult, pae_matrix: np.ndarray, chain_lengths: Optional[tuple[int, int]] = None,
    chain_residue_numbers: Optional[dict[str, list[int]]] = None,
    pae_threshold: float = PAE_CONFIDENT_THRESHOLD,
    plddt_threshold: float = INTERFACE_PLDDT_HIGH,
    *,
    chain_offsets: Optional[tuple[int, int]] = None,
    cb_to_ca_maps: Optional[tuple[list[int], list[int]]] = None) -> dict:
    """Identify interface residues that pass both PAE and pLDDT confidence filters.
    These are the "computational hot spots" - residue pairs at the interface where AlphaFold2 is confident about both the local structure (pLDDT) and the relative positioning between chains (PAE).  
    These are the primary drug-discovery-relevant output.
    Args:
        contact_result: ContactResult from identify_interface_contacts().
        pae_matrix: Full PAE matrix from the PKL file.
        chain_lengths: Tuple of (n_residues_chain_A, n_residues_chain_B). Legacy parameter; use chain_offsets for multi-chain.
        chain_residue_numbers: Optional dict of chain_id -> list of PDB residue numbers, for mapping back to biological numbering.
        pae_threshold: Maximum PAE for a confident contact (default 5.0 Å).
        plddt_threshold: Minimum pLDDT for a confident residue (default 70).
        chain_offsets: Tuple of (offset_A, offset_B) in the PAE matrix.
        cb_to_ca_maps: Tuple of (map_A, map_B) for CB->CA index translation.
    Returns:
        Dictionary with keys:
            n_confident_residues_a: Confident unique residues on chain A.
            n_confident_residues_b: Confident unique residues on chain B.
            confident_residue_indices_a: List of array indices for chain A.
            confident_residue_indices_b: List of array indices for chain B.
            confident_residue_numbers_a: PDB residue numbers (if provided).
            confident_residue_numbers_b: PDB residue numbers (if provided).
            confident_contacts: List of (idx_a, idx_b, pae, plddt_a, plddt_b) tuples.
    """
    empty = {
        'n_confident_residues_a': 0,
        'n_confident_residues_b': 0,
        'confident_residue_indices_a': [],
        'confident_residue_indices_b': [],
        'confident_residue_numbers_a': [],
        'confident_residue_numbers_b': [],
        'confident_contacts': [],
    }

    if contact_result.n_interface_contacts == 0:
        return empty

    interface_pae = extract_interface_pae(contact_result, pae_matrix, chain_lengths, chain_offsets=chain_offsets, cb_to_ca_maps=cb_to_ca_maps)
    if interface_pae is None:
        return empty

    plddt_a = contact_result.plddt_a
    plddt_b = contact_result.plddt_b
    contacts = contact_result.contacts

    # Identify contacts where AlphaFold is confident about both the individual residue structures (pLDDT >= threshold) and their relative positioning (PAE < threshold). 
    # Contacts failing either criterion may reflect prediction uncertainty rather than true physical proximity.
    confident_mask = (
        (interface_pae < pae_threshold)
        & (plddt_a[contacts[:, 0]] >= plddt_threshold)
        & (plddt_b[contacts[:, 1]] >= plddt_threshold)
    )

    confident_contacts_arr = contacts[confident_mask]

    if len(confident_contacts_arr) == 0:
        return empty

    confident_idx_a = sorted(set(confident_contacts_arr[:, 0].tolist()))
    confident_idx_b = sorted(set(confident_contacts_arr[:, 1].tolist()))

    # Map to PDB residue numbers if available
    chain_ids = contact_result.chain_ids
    if chain_residue_numbers and chain_ids[0] in chain_residue_numbers:
        res_nums_a = [chain_residue_numbers[chain_ids[0]][i] for i in confident_idx_a]
        res_nums_b = [chain_residue_numbers[chain_ids[1]][i] for i in confident_idx_b]
    else:
        res_nums_a = []
        res_nums_b = []

    # Build detailed contact list
    confident_contact_details = []
    for i in range(len(confident_contacts_arr)):
        idx_a = int(confident_contacts_arr[i, 0])
        idx_b = int(confident_contacts_arr[i, 1])
        confident_contact_details.append({
            'idx_a': idx_a,
            'idx_b': idx_b,
            'pae': round(float(interface_pae[confident_mask][i]), 2),
            'plddt_a': round(float(plddt_a[idx_a]), 2),
            'plddt_b': round(float(plddt_b[idx_b]), 2),
        })

    return {
        'n_confident_residues_a': len(confident_idx_a),
        'n_confident_residues_b': len(confident_idx_b),
        'confident_residue_indices_a': confident_idx_a,
        'confident_residue_indices_b': confident_idx_b,
        'confident_residue_numbers_a': res_nums_a,
        'confident_residue_numbers_b': res_nums_b,
        'confident_contacts': confident_contact_details,
    }

#---------------All-Pairs Metrics (Multimer Refactor Phase 3)---------------------

@dataclass
class PairMetricRecord:
    """JSON-serialisable per-pair record — geometry + PAE + symmetry combined.

    Produced by ``compute_all_pair_metrics``; one record per unique chain pair
    (including pairs with zero inter-chain contacts). This is what lands in the
    ``pair_metrics`` CSV column via ``asdict`` + ``json.dumps``.

    PAE fields distinguish "PAE unavailable" (None) from "PAE available, no
    confident contacts" (0.0). Interface residue sets are converted to sorted
    lists so the JSON round-trip is stable.
    """
    chain_i: str
    chain_j: str
    accession_i: str
    accession_j: str
    n_contacts: int
    interface_plddt: Optional[float]
    pdockq: float
    ppv: Optional[float]
    symmetry: Optional[float]
    pae_confident_fraction: Optional[float]
    strict_confident_fraction: Optional[float]
    interface_residues_i: list = field(default_factory=list)
    interface_residues_j: list = field(default_factory=list)


def _compute_pair_symmetry(pair: PairContactResult) -> Optional[float]:
    """Compute interface symmetry for a single pair (min/max of interface fractions).

    Returns None for zero-contact pairs: symmetry is undefined without an interface.
    Matches the dimer formula in ``compute_interface_geometry`` so N=2 metric
    identity holds.
    """
    raw = pair._raw
    if raw is None or raw.n_interface_contacts == 0:
        return None
    n_if_a = len(raw.interface_residues_a)
    n_if_b = len(raw.interface_residues_b)
    n_res_a = raw.n_residues_a
    n_res_b = raw.n_residues_b
    frac_a = n_if_a / n_res_a if n_res_a > 0 else 0.0
    frac_b = n_if_b / n_res_b if n_res_b > 0 else 0.0
    max_frac = max(frac_a, frac_b)
    if max_frac == 0.0:
        return None
    return round(min(frac_a, frac_b) / max_frac, 4)


def _compute_pair_pae_fractions(
    pair: PairContactResult,
    pae_matrix: Optional[np.ndarray],
    chain_offsets: Optional[dict[str, int]],
    cb_to_ca_map: Optional[dict[str, list[int]]],
) -> tuple[Optional[float], Optional[float]]:
    """Compute (pae_confident_fraction, strict_confident_fraction) for one pair.

    Returns (None, None) when the PAE matrix is unavailable — distinct from
    (0.0, 0.0) which would mean "PAE available, no confident contacts". The
    caller in ``aggregate_pair_metrics`` relies on this distinction for
    contact-weighted means.
    """
    if pae_matrix is None or chain_offsets is None:
        return None, None
    raw = pair._raw
    if raw is None or raw.n_interface_contacts == 0:
        return (0.0, 0.0) if pae_matrix is not None else (None, None)

    off_i = chain_offsets.get(pair.chain_i)
    off_j = chain_offsets.get(pair.chain_j)
    if off_i is None or off_j is None:
        return None, None

    cb_maps = None
    if cb_to_ca_map is not None:
        map_i = cb_to_ca_map.get(pair.chain_i)
        map_j = cb_to_ca_map.get(pair.chain_j)
        if map_i and map_j:
            cb_maps = (map_i, map_j)

    indices = _compute_interface_pae_indices(
        raw, pae_matrix, chain_lengths=None,
        chain_offsets=(off_i, off_j), cb_to_ca_maps=cb_maps,
    )
    if indices is None:
        return None, None
    rows, cols = indices
    if len(rows) == 0:
        return 0.0, 0.0

    forward = pae_matrix[rows, cols]
    reverse = pae_matrix[cols, rows]
    combined = np.maximum(forward, reverse) if USE_BIDIRECTIONAL_PAE else forward

    contacts = raw.contacts
    plddt_a = raw.plddt_a[contacts[:, 0]]
    plddt_b = raw.plddt_b[contacts[:, 1]]
    strict_mask = (
        (combined < PAE_CONFIDENT_THRESHOLD)
        & (plddt_a >= INTERFACE_PLDDT_HIGH)
        & (plddt_b >= INTERFACE_PLDDT_HIGH)
    )

    n_total = len(combined)
    n_confident = int(np.sum(combined < PAE_CONFIDENT_THRESHOLD))
    n_strict = int(np.sum(strict_mask))
    return round(n_confident / n_total, 4), round(n_strict / n_total, 4)


def compute_all_pair_metrics(
    pair_results: list[PairContactResult],
    pae_matrix: Optional[np.ndarray] = None,
    chain_offsets: Optional[dict[str, int]] = None,
    cb_to_ca_map: Optional[dict[str, list[int]]] = None,
) -> list[PairMetricRecord]:
    """Combine per-pair geometry, PAE, and symmetry into JSON-safe records.

    Args:
        pair_results: Output of ``pdockq.compute_all_chain_pairs``.
        pae_matrix: Full PAE matrix; None when PKL is unavailable.
        chain_offsets: Chain -> starting PAE row/column offset (dict, not tuple,
            because we have N>2 chains). When None, PAE fields are None.
        cb_to_ca_map: Chain -> CB->CA index list (from ``ChainInfo_New``).

    Returns:
        One ``PairMetricRecord`` per input pair, preserving order. Zero-contact
        pairs emit the record with null score-bearing fields so
        ``aggregate_pair_metrics`` can surface dangling chains via ``pdockq_min``.
    """
    records: list[PairMetricRecord] = []
    for pair in pair_results:
        symmetry = _compute_pair_symmetry(pair)
        pae_conf, strict_conf = _compute_pair_pae_fractions(
            pair, pae_matrix, chain_offsets, cb_to_ca_map,
        )
        records.append(PairMetricRecord(
            chain_i=pair.chain_i,
            chain_j=pair.chain_j,
            accession_i=pair.accession_i,
            accession_j=pair.accession_j,
            n_contacts=pair.n_contacts,
            interface_plddt=round(pair.interface_plddt, 2) if pair.interface_plddt is not None else None,
            pdockq=round(pair.pdockq, 4),
            ppv=round(pair.ppv, 4) if pair.ppv is not None else None,
            symmetry=symmetry,
            pae_confident_fraction=pae_conf,
            strict_confident_fraction=strict_conf,
            interface_residues_i=sorted(int(r) for r in pair.interface_residues_i),
            interface_residues_j=sorted(int(r) for r in pair.interface_residues_j),
        ))
    return records


def aggregate_pair_metrics(records: list[PairMetricRecord]) -> dict:
    """Roll per-pair records up into the per-complex aggregate scalars.

    Aggregation policy (from the dissertation-safe plan):
        pdockq_mean                    unweighted mean across all pairs (zero-contact pairs contribute 0.0)
        pdockq_min                     min across all pairs (includes zero-contact)
        contact_count_total            sum across pairs
        symmetry_mean                  contact-weighted; skip zero-contact pairs
        symmetry_min                   min across pairs with at least one contact
        interface_plddt_mean           contact-weighted; skip zero-contact pairs
        pae_confident_fraction_mean    contact-weighted; None if PAE unavailable on any contact-bearing pair
        strict_confident_fraction_mean contact-weighted; None if PAE unavailable on any contact-bearing pair

    None vs 0.0: a contact-bearing pair with ``pae_confident_fraction is None``
    means PAE was not available. In that case the aggregate is None, not a partial
    average, because silently dropping pairs would understate PAE unavailability.
    Pairs with ``n_contacts == 0`` are excluded from PAE aggregation entirely
    (no contacts → no fraction to weight).
    """
    n = len(records)
    if n == 0:
        return {
            'pdockq_mean': None,
            'pdockq_min': None,
            'contact_count_total': 0,
            'symmetry_mean': None,
            'symmetry_min': None,
            'interface_plddt_mean': None,
            'pae_confident_fraction_mean': None,
            'strict_confident_fraction_mean': None,
        }

    pdockqs = [r.pdockq for r in records]
    pdockq_mean = round(float(np.mean(pdockqs)), 4)
    pdockq_min = round(float(np.min(pdockqs)), 4)
    contact_count_total = int(sum(r.n_contacts for r in records))

    with_contacts = [r for r in records if r.n_contacts > 0]

    # Contact-weighted mean over pairs that have contacts; None if all pairs zero.
    def _weighted_mean(attr: str) -> Optional[float]:
        num = 0.0
        denom = 0
        for r in with_contacts:
            val = getattr(r, attr)
            if val is None:
                continue
            num += val * r.n_contacts
            denom += r.n_contacts
        if denom == 0:
            return None
        return round(num / denom, 4)

    symmetry_mean = _weighted_mean('symmetry')
    interface_plddt_mean = _weighted_mean('interface_plddt')
    symmetries_nonzero = [r.symmetry for r in with_contacts if r.symmetry is not None]
    symmetry_min = round(float(np.min(symmetries_nonzero)), 4) if symmetries_nonzero else None

    # PAE aggregates: None if any contact-bearing pair reports None (i.e. PAE
    # genuinely unavailable). 0.0 on a pair is valid data and included.
    def _pae_weighted_mean(attr: str) -> Optional[float]:
        if not with_contacts:
            return None
        num = 0.0
        denom = 0
        for r in with_contacts:
            val = getattr(r, attr)
            if val is None:
                return None
            num += val * r.n_contacts
            denom += r.n_contacts
        if denom == 0:
            return None
        return round(num / denom, 4)

    pae_confident_fraction_mean = _pae_weighted_mean('pae_confident_fraction')
    strict_confident_fraction_mean = _pae_weighted_mean('strict_confident_fraction')

    return {
        'pdockq_mean': pdockq_mean,
        'pdockq_min': pdockq_min,
        'contact_count_total': contact_count_total,
        'symmetry_mean': symmetry_mean,
        'symmetry_min': symmetry_min,
        'interface_plddt_mean': interface_plddt_mean,
        'pae_confident_fraction_mean': pae_confident_fraction_mean,
        'strict_confident_fraction_mean': strict_confident_fraction_mean,
    }


def serialise_pair_metrics(records: list[PairMetricRecord]) -> str:
    """Serialise pair records to a compact JSON string for the CSV column."""
    return json.dumps([asdict(r) for r in records], separators=(',', ':'))


#------------------Main Analysis Functions (Module Interface)-------------------------------------

def analyse_interface(pdb_path: Union[str, Path], threshold: float = DEFAULT_CONTACT_THRESHOLD) -> dict:
    """Extract all interface features from a PDB file.
    This is the main importable function for PDB-only analysis. All output values are JSON-serialisable (no raw numpy types).
    Args:
        pdb_path: Path to an AlphaFold2 PDB file.
        threshold: Contact distance threshold in Ångströms.
    Returns:
        Dictionary combining:
            - pDockQ and PPV scores
            - Interface geometry features
            - Interface pLDDT features
            - Basic quality flags
    """
    chain_coords, chain_plddt = read_pdb(str(pdb_path))

    if len(chain_coords) < 2:
        return {'error': 'fewer_than_two_chains', 'pdb_path': str(pdb_path)}

    # Core contact identification (also computes pDockQ)
    contact_result = identify_interface_contacts(chain_coords, chain_plddt, threshold)

    # Compute features
    geometry = compute_interface_geometry(contact_result)
    plddt_features = compute_interface_plddt(contact_result)

    # Assemble result
    result = {
        'pdockq': round(contact_result.pdockq, 4),
        'ppv': round(contact_result.ppv, 4),
        'avg_interface_plddt': (
            round(contact_result.avg_if_plddt, 2)
            if contact_result.avg_if_plddt is not None else None
        ),
    }
    result.update(geometry)
    result.update(plddt_features)

    # Quality flags
    result['flags'] = _compute_flags(result)

    # Composite score (None without PAE, but included for consistency)
    result['interface_confidence_score'] = compute_interface_confidence(result)

    return result

def analyse_interface_with_pae(pdb_path: Union[str, Path], pae_matrix: np.ndarray, chain_lengths: Optional[tuple[int, int]] = None, threshold: float = DEFAULT_CONTACT_THRESHOLD) -> dict:
    """PDB features plus PAE-derived interface features.
    Uses read_pdb_with_chain_info() for proper multi-chain support and CB->CA mapping. The chain_lengths parameter is accepted for backward compatibility but is no longer required.
    Args:
        pdb_path: Path to an AlphaFold2 PDB file.
        pae_matrix: PAE matrix from the PKL file, shape (N_total, N_total).
        chain_lengths: DEPRECATED. Previously required but now computed automatically. Ignored if provided.
        threshold: Contact distance threshold in Ångströms.
    Returns:
        Dictionary combining features with:
            - Interface PAE statistics
            - Confident contact fraction
            - Confident interface residue lists
    """
    # Use the chain-info reader for full multi-chain + CB-aware support
    chain_info = read_pdb_with_chain_info(str(pdb_path))

    if len(chain_info.chain_ids) < 2:
        return {'error': 'fewer_than_two_chains', 'pdb_path': str(pdb_path)}

    # Find the best interacting chain pair
    ch_a, ch_b, contact_result = find_best_chain_pair(chain_info, t=threshold)

    geometry = compute_interface_geometry(contact_result)
    plddt_features = compute_interface_plddt(contact_result)

    result = {
        'pdockq': round(contact_result.pdockq, 4),
        'ppv': round(contact_result.ppv, 4),
        'avg_interface_plddt': (
            round(contact_result.avg_if_plddt, 2)
            if contact_result.avg_if_plddt is not None else None
        ),
    }
    result.update(geometry)
    result.update(plddt_features)

    # Compute PAE mapping parameters
    offsets = compute_pae_chain_offsets(chain_info)
    pae_chain_offsets = (offsets[ch_a], offsets[ch_b])

    cb_maps_a = chain_info.cb_to_ca_map.get(ch_a, [])
    cb_maps_b = chain_info.cb_to_ca_map.get(ch_b, [])
    cb_maps = (cb_maps_a, cb_maps_b) if cb_maps_a and cb_maps_b else None

    # PAE features
    pae_features = compute_interface_pae_features(
        contact_result, pae_matrix, chain_lengths=None,
        chain_offsets=pae_chain_offsets,
        cb_to_ca_maps=cb_maps,
    )
    result.update(pae_features)

    # Confident residues
    confident = identify_confident_interface_residues(contact_result, pae_matrix, chain_lengths=None, chain_residue_numbers=chain_info.chain_res_numbers, chain_offsets=pae_chain_offsets, cb_to_ca_maps=cb_maps)
    result['n_confident_residues_a'] = confident['n_confident_residues_a']
    result['n_confident_residues_b'] = confident['n_confident_residues_b']
    result['confident_residue_numbers_a'] = confident['confident_residue_numbers_a']
    result['confident_residue_numbers_b'] = confident['confident_residue_numbers_b']

    # Quality flags
    result['flags'] = _compute_flags(result)

    # Composite interface confidence score
    result['interface_confidence_score'] = compute_interface_confidence(result)
    return result

def analyse_interface_from_contact_result(contact_result: ContactResult, pae_matrix: Optional[np.ndarray] = None, chain_lengths: Optional[tuple[int, int]] = None,
    chain_residue_numbers: Optional[dict[str, list[int]]] = None,
    *,
    chain_offsets: Optional[tuple[int, int]] = None,
    cb_to_ca_maps: Optional[tuple[list[int], list[int]]] = None) -> dict:
    """Compute interface features from a pre-computed ContactResult.
    Use this when the batch pipeline has already called calc_pdockq_and_contacts() and you want to avoid re-reading the PDB.
    Args:
        contact_result: Pre-computed ContactResult.
        pae_matrix: Optional PAE matrix for Phase 2 features.
        chain_lengths: Legacy dimer chain lengths; prefer chain_offsets.
        chain_residue_numbers: Optional for PDB residue number mapping.
        chain_offsets: Tuple of (offset_A, offset_B) in the PAE matrix. Required for multi-chain complexes.
        cb_to_ca_maps: Tuple of (map_A, map_B) for CB->CA index translation. Required when CB count ≠ CA count.
    Returns:
        Dictionary of all computed interface features.
    """
    geometry = compute_interface_geometry(contact_result)
    plddt_features = compute_interface_plddt(contact_result)
    result = {
        'pdockq': round(contact_result.pdockq, 4),
        'ppv': round(contact_result.ppv, 4),
        'avg_interface_plddt': (
            round(contact_result.avg_if_plddt, 2)
            if contact_result.avg_if_plddt is not None else None
        ),
    }
    result.update(geometry)
    result.update(plddt_features)

    # if PAE available
    if pae_matrix is not None and (chain_lengths is not None or chain_offsets is not None):
        pae_features = compute_interface_pae_features(
            contact_result, pae_matrix, chain_lengths,
            chain_offsets=chain_offsets,
            cb_to_ca_maps=cb_to_ca_maps,
        )
        result.update(pae_features)
        confident = identify_confident_interface_residues(
            contact_result, pae_matrix, chain_lengths,
            chain_residue_numbers=chain_residue_numbers,
            chain_offsets=chain_offsets,
            cb_to_ca_maps=cb_to_ca_maps,
        )
        result['n_confident_residues_a'] = confident['n_confident_residues_a']
        result['n_confident_residues_b'] = confident['n_confident_residues_b']
        result['confident_residue_numbers_a'] = confident['confident_residue_numbers_a']
        result['confident_residue_numbers_b'] = confident['confident_residue_numbers_b']

    result['flags'] = _compute_flags(result)

    # Composite interface confidence score
    result['interface_confidence_score'] = compute_interface_confidence(result)
    return result

#---------------Flag Computation-------------------------------------------------

def _compute_flags(features: dict) -> list[str]:
    """Evaluate quality flags based on computed interface features.
    Args:
        features: Dictionary of interface features from analyse_interface().
    Returns:
        List of flag strings (empty if no issues detected).
    """
    flags = []
    n_contacts = features.get('n_interface_contacts', 0)
    density = features.get('contacts_per_interface_residue', 0)
    symmetry = features.get('interface_symmetry', 0)
    delta = features.get('interface_vs_bulk_delta')

    # Geometry flags
    if n_contacts < MIN_INTERFACE_CONTACTS:
        flags.append('small_interface')
    if n_contacts >= MIN_INTERFACE_CONTACTS and density < SPARSE_INTERFACE_DENSITY:
        flags.append('sparse_interface')
    if n_contacts >= MIN_INTERFACE_CONTACTS and symmetry < ASYMMETRIC_INTERFACE_RATIO:
        flags.append('asymmetric_interface')

    # pLDDT flags
    if delta is not None and delta > 10:
        flags.append('interface_better_than_bulk')

    # PAE flags
    # Keep pointing at the PAE-only fraction (pae_confident_contact_fraction), not the strict
    # fraction: the 0.2 threshold was calibrated against the PAE-only definition. Any
    # recalibration should happen alongside the paradox and quality-tier thresholds.
    confident_fraction = features.get('pae_confident_contact_fraction')
    if confident_fraction is not None:
        if confident_fraction < 0.2:
            flags.append('low_interface_confidence')

    return flags


#---------------Composite Interface Confidence Score-----------------------

def compute_interface_confidence(metrics: dict) -> Optional[float]:
    """Combine interface-level quality indicators into a single heuristic screening score.

    The score is a WEIGHTED AGGREGATION of four normalised components, NOT a calibrated
    estimate of interface correctness:
        1. Interface pLDDT, transformed by normalise_interface_plddt() (local structural
           confidence on AlphaFold's 0-100 scale, band-aware mapping to [0.05, 1.0])
        2. Strict confident contact fraction (PAE < 5A AND both residue pLDDT >= 70) - the
           strictest available confidence-bearing feature for inter-chain positioning
        3. Interface symmetry (geometric plausibility, not confidence - asymmetric
           interfaces can be biologically real)
        4. Contact density (packing plausibility, not confidence - sparse interfaces can be
           biologically real for transient/motif-mediated binding)

    Components 1 and 2 carry confidence information; components 3 and 4 are plausibility
    features retained because they correlate with typical binding geometries but should not
    be read as confidence in their own right. Weights are expert-chosen, partially informed
    by the 9,573-complex distribution, and have NOT been fitted against DockQ, pDockQ2, or
    any benchmarked ground truth. Treat the composite as a screening/aggregation heuristic.

    Requires PAE data (strict_confident_contact_fraction). Returns None when PAE features
    are unavailable.

    Args:
        metrics: Dictionary containing at minimum - interface_plddt_combined,
            strict_confident_contact_fraction, interface_symmetry and
            contacts_per_interface_residue.
    Returns:
        Score in range 0.0 to 1.0 or None if required metrics are missing.
    """
    if_plddt = metrics.get('interface_plddt_combined')
    conf_frac = metrics.get('strict_confident_contact_fraction')
    symmetry = metrics.get('interface_symmetry')
    density = metrics.get('contacts_per_interface_residue')

    # Require all four components for a meaningful composite score
    if any(v is None for v in [if_plddt, conf_frac, symmetry, density]):
        return None

    plddt_component = normalise_interface_plddt(if_plddt)
    pae_component = conf_frac                                   # already 0-1
    symmetry_component = symmetry                               # already 0-1
    density_component = min(density / DENSITY_NORMALIZATION, 1.0)

    score = (
        WEIGHT_PLDDT * plddt_component
        + WEIGHT_PAE * pae_component
        + WEIGHT_SYMMETRY * symmetry_component
        + WEIGHT_DENSITY * density_component
    )
    return round(score, 4)

def compute_extended_flags(interface_features: dict, iptm: Optional[float] = None, pdockq: Optional[float] = None, disorder_fraction: Optional[float] = None) -> list[str]:
    """Evaluate quality flags combining interface features with global metrics.
    Extends the basic structural flags from _compute_flags() with paradox detection flags that require ipTM, pDockQ, and disorder information from the batch pipeline.
    Args:
        interface_features: Dictionary of interface features.
        iptm: Interface pTM score (from PKL).
        pdockq: pDockQ score.
        disorder_fraction: Fraction of residues with pLDDT < 50.
    Returns:
        List of flag strings (empty if no issues detected).
    """
    # Start with structural flags
    flags = _compute_flags(interface_features)

    # Paradox detection (requires global metrics)
    if (iptm is not None and pdockq is not None and disorder_fraction is not None
            and iptm >= PARADOX_IPTM_THRESHOLD
            and pdockq >= PARADOX_PDOCKQ_THRESHOLD
            and disorder_fraction > SUBSTANTIAL_DISORDER_FRACTION):

        # Keep pointing at the PAE-only fraction: the GENUINE (0.73) and ARTEFACT (0.50)
        # thresholds were calibrated against 138 paradox complexes scored with the
        # PAE-only definition. Any recalibration to the strict definition is deferred
        # to a future pass alongside quality_tier_v2 threshold recalibration.
        conf_frac = interface_features.get('pae_confident_contact_fraction')

        if conf_frac is not None and conf_frac > PARADOX_CONFIDENT_CONTACT_GENUINE:
            flags.append('paradox_confident_disorder')
        elif conf_frac is not None and conf_frac < PARADOX_CONFIDENT_CONTACT_ARTEFACT:
            flags.append('paradox_artefactual')

    # Metric disagreement: large gap between ipTM and pDockQ
    # Systematically ipTM >> pDockQ in the 9,573-complex dataset, confirming pDockQ penalises genuine interfaces in disordered complexes.
    if (iptm is not None and pdockq is not None
            and abs(iptm - pdockq) > METRIC_DISAGREEMENT_THRESHOLD):
        flags.append('metric_disagreement')

    return flags

#---------------------Interface Export Record-------------------------------

def build_interface_export_record(complex_name: str, protein_a: str, protein_b: str, quality_tier_v2: str,
    interface_confidence_score: Optional[float],
    confident_residue_numbers_a: list[int],
    confident_residue_numbers_b: list[int],
    flags: list[str],
    iptm: Optional[float] = None,
    pdockq: Optional[float] = None,
    n_interface_contacts: Optional[int] = None,
    pae_confident_contact_fraction: Optional[float] = None,
    strict_confident_contact_fraction: Optional[float] = None,
    interface_plddt_combined: Optional[float] = None) -> dict:
    """Build a structured record for interface residue export.
    Produces a JSON-serialisable dictionary suitable for JSONL output.
    Each record describes a complex's confident interface residues - the computationally identified binding hot-spots that pass both PAE and pLDDT confidence filters.
    These records feed into downstream analyses: pathway mapping, genetic variant analysis, and drug target identification.
    Args:
        complex_name: Parsed complex identifier (e.g. "P12345_Q67890").
        protein_a: UniProt ID for chain A.
        protein_b: UniProt ID for chain B.
        quality_tier_v2: V2 quality classification ("High"/"Medium"/"Low").
        interface_confidence_score: Composite score (0.0-1.0) or None.
        confident_residue_numbers_a: PDB residue numbers for chain A confident interface residues.
        confident_residue_numbers_b: PDB residue numbers for chain B confident interface residues.
        flags: List of quality flag strings.
        iptm: Interface pTM score (optional, for context).
        pdockq: pDockQ score (optional, for context).
        n_interface_contacts: Total interface contacts (optional).
        pae_confident_contact_fraction: Fraction of contacts with PAE < 5A only (optional).
        strict_confident_contact_fraction: Fraction of contacts with PAE < 5A AND both
            pLDDT >= 70 - the fraction consumed by the composite score (optional).
        interface_plddt_combined: Mean interface pLDDT (optional).
    Returns:
        Dictionary with all fields needed for the JSONL export.
    """
    return {
        'complex_name': complex_name,
        'protein_a': protein_a,
        'protein_b': protein_b,
        'quality_tier_v2': quality_tier_v2,
        'interface_confidence_score': interface_confidence_score,
        'iptm': iptm,
        'pdockq': pdockq,
        'n_interface_contacts': n_interface_contacts,
        'pae_confident_contact_fraction': pae_confident_contact_fraction,
        'strict_confident_contact_fraction': strict_confident_contact_fraction,
        'interface_plddt_combined': interface_plddt_combined,
        'confident_interface_residues_a': confident_residue_numbers_a,
        'confident_interface_residues_b': confident_residue_numbers_b,
        'n_confident_residues_a': len(confident_residue_numbers_a),
        'n_confident_residues_b': len(confident_residue_numbers_b),
        'flags': flags,
    }

#-------------------CLI Entry Point-----------------------------------------

def main() -> None:
    """CLI entry point for standalone interface analysis."""
    parser = argparse.ArgumentParser(
        description="Analyse protein-protein interface from AlphaFold2 predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage (standalone):
    python interface_analysis.py --pdb structure.pdb
    python interface_analysis.py --pdb structure.pdb --json output.json
    python interface_analysis.py --pdb structure.pdb --pkl result.pkl

Usage (as importable module):
    from interface_analysis import analyse_interface, analyse_interface_with_pae
    result = analyse_interface("structure.pdb", threshold=8.0)
    result_pae = analyse_interface_with_pae("structure.pdb", pae_matrix, chain_lengths)
""",
)
    parser.add_argument("--pdb", required=True, help="Path to PDB file")
    parser.add_argument("--pkl", help="Path to PKL file (enables PAE analysis)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_CONTACT_THRESHOLD, help=f"Contact distance threshold in Å (default: {DEFAULT_CONTACT_THRESHOLD})")
    parser.add_argument("--json", help="Save results to JSON file")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")

    args = parser.parse_args()

    if args.pkl:
        # With PAE
        try:
            from read_af2_nojax import load_pkl_without_jax
            prediction = load_pkl_without_jax(args.pkl)
            pae_matrix = np.asarray(prediction.get('predicted_aligned_error'))
            if pae_matrix is None:
                print("Warning: No PAE matrix in PKL, falling back to Phase 1 only", file=sys.stderr)
                result = analyse_interface(args.pdb, threshold=args.threshold)
            else:
                result = analyse_interface_with_pae(
                    args.pdb, pae_matrix, threshold=args.threshold
                )
        except Exception as e:
            print(f"Warning: PAE analysis failed ({e}), falling back to Phase 1", file=sys.stderr)
            result = analyse_interface(args.pdb, threshold=args.threshold)
    else:
        result = analyse_interface(args.pdb, threshold=args.threshold)

    if not args.quiet:
        print(f"\n{'=' * 60}")
        print(f"Interface Analysis: {Path(args.pdb).name}")
        print(f"{'=' * 60}")

        # pDockQ
        print(f"\n  pDockQ:  {result.get('pdockq', 'N/A')}")
        print(f"  PPV:     {result.get('ppv', 'N/A')}")

        # Geometry
        print(f"\n  Interface Geometry:")
        print(f"    Contacts:       {result.get('n_interface_contacts', 0)}")
        print(f"    Residues (A/B): {result.get('n_interface_residues_a', 0)} / "
              f"{result.get('n_interface_residues_b', 0)}")
        print(f"    Fraction (A/B): {result.get('interface_fraction_a', 0):.3f} / "
              f"{result.get('interface_fraction_b', 0):.3f}")
        print(f"    Symmetry:       {result.get('interface_symmetry', 0):.3f}")
        print(f"    Density:        {result.get('contacts_per_interface_residue', 0):.2f} contacts/residue")

        # pLDDT
        print(f"\n  Interface pLDDT:")
        print(f"    Interface (combined): {result.get('interface_plddt_combined', 'N/A')}")
        print(f"    Bulk (combined):      {result.get('bulk_plddt_combined', 'N/A')}")
        print(f"    Delta (if - bulk):    {result.get('interface_vs_bulk_delta', 'N/A')}")
        print(f"    High-conf fraction:   {result.get('interface_plddt_high_fraction', 'N/A')}")

        # PAE
        if result.get('interface_pae_mean') is not None:
            print(f"\n  Interface PAE:")
            print(f"    Mean (bidirectional max): {result['interface_pae_mean']}")
            print(f"    PAE-only confident:   {result.get('n_pae_confident_contacts', 0)} / "
                  f"{result.get('n_interface_contacts', 0)} "
                  f"({result.get('pae_confident_contact_fraction', 0):.1%})")
            print(f"    Strict confident:     {result.get('n_strict_confident_contacts', 0)} / "
                  f"{result.get('n_interface_contacts', 0)} "
                  f"({result.get('strict_confident_contact_fraction', 0):.1%})")
            print(f"    Confident residues: A={result.get('n_confident_residues_a', 0)}, "
                  f"B={result.get('n_confident_residues_b', 0)}")

        # Composite score
        score = result.get('interface_confidence_score')
        if score is not None:
            print(f"\n  Composite Interface Confidence: {score:.4f}")

        # Flags
        flags = result.get('flags', [])
        if flags:
            print(f"\n  Flags: {', '.join(flags)}")
        else:
            print(f"\n  Flags: none")

    if args.json:
        # Remove numpy arrays for JSON serialisation
        json_result = {k: v for k, v in result.items()
                       if not isinstance(v, np.ndarray)}
        with open(args.json, 'w') as f:
            json.dump(json_result, f, indent=2)
        if not args.quiet:
            print(f"\n  Saved: {args.json}")


if __name__ == "__main__":
    main()