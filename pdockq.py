#!/usr/bin/env python3
"""
pDockQ - Predicted DockQ Score Calculator

Calculates predicted DockQ scores for AlphaFold2-predicted protein complexes.
Uses the FoldDock parameterisation which normalises by the average pLDDT value
at the interface.

Based on: pdockq = L / (1 + exp(-k*(x-x0))) + b
where L=0.724, x0=152.611, k=0.052, b=0.018, and x = avg_interface_plddt * log10(n_contacts).

Usage (standalone):
    python pdockq.py --pdbfile structure.pdb

Usage (as importable module):
    from pdockq import read_pdb_Edited, calc_pdockq_Edited, calc_pdockq_and_contacts_New
    chain_coords, chain_plddt = read_pdb_Edited("structure.pdb")
    pdockq, ppv = calc_pdockq_Edited(chain_coords, chain_plddt, t=8)

    # Extended version with full contact details (for interface analysis):
    result = calc_pdockq_and_contacts_New(chain_coords, chain_plddt, t=8)
    # result.contacts, result.avg_if_plddt, result.chain_ids, etc.
"""

import sys
import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np


# ── Constants (_New: extracted to module-level named constants) ───────

DEFAULT_CONTACT_THRESHOLD_New = 8  # Ångströms

# pDockQ sigmoid parameters (FoldDock calibration)
PDOCKQ_L_New = 0.724
PDOCKQ_X0_New = 152.611
PDOCKQ_K_New = 0.052
PDOCKQ_B_New = 0.018

# PPV lookup table - maps pDockQ thresholds to positive predictive values
PPV_VALUES_New = np.array([
    0.98128027, 0.96322524, 0.95333044, 0.94001920,
    0.93172991, 0.92420274, 0.91629946, 0.90952562, 0.90043139,
    0.89195530, 0.88570037, 0.87822061, 0.87116417, 0.86040801,
    0.85453785, 0.84294946, 0.83367787, 0.82238224, 0.81190228,
    0.80223507, 0.78549007, 0.77766077, 0.75941223, 0.74006263,
    0.73044282, 0.71391784, 0.70615739, 0.68635536, 0.66728511,
    0.63555449, 0.55890174,
])

PDOCKQ_THRESHOLDS_New = np.array([
    0.67333079, 0.65666073, 0.63254566, 0.62604391,
    0.60150931, 0.58313803, 0.56473810, 0.54122438, 0.52314392,
    0.49659878, 0.47746760, 0.44661346, 0.42628389, 0.39990988,
    0.38479715, 0.36493930, 0.34526004, 0.32625890, 0.31475668,
    0.29750023, 0.26673725, 0.24561247, 0.21882689, 0.19651314,
    0.17606258, 0.15398168, 0.13927677, 0.12024131, 0.09996019,
    0.06968505, 0.02946438,
])


# ── Data Structures (_New: entirely new additions) ───────────────────

@dataclass
class ContactResult_New:
    """Extended result from pDockQ calculation with full contact details.
    
    Attributes:
        pdockq: Predicted DockQ score (0 to ~0.74).
        ppv: Positive predictive value from calibration lookup.
        chain_ids: Tuple of (chain_A_id, chain_B_id).
        contacts: Nx2 array of contact indices - contacts[:,0] are chain A
                  residue indices, contacts[:,1] are chain B residue indices.
                  Empty (0,2) array if no contacts found.
        contact_distances: 1D array of distances for each contact pair.
        n_interface_contacts: Total number of inter-chain contacts.
        interface_residues_a: Set of chain A residue indices at the interface.
        interface_residues_b: Set of chain B residue indices at the interface.
        avg_if_plddt: Average pLDDT at the interface (None if no contacts).
        n_residues_a: Total residues in chain A.
        n_residues_b: Total residues in chain B.
        plddt_a: Per-residue pLDDT array for chain A.
        plddt_b: Per-residue pLDDT array for chain B.
    """
    pdockq: float = 0.0
    ppv: float = 0.0
    chain_ids: tuple[str, str] = ('', '')
    contacts: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=int))
    contact_distances: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    n_interface_contacts: int = 0
    interface_residues_a: set[int] = field(default_factory=set)
    interface_residues_b: set[int] = field(default_factory=set)
    avg_if_plddt: Optional[float] = None
    n_residues_a: int = 0
    n_residues_b: int = 0
    plddt_a: Optional[np.ndarray] = None
    plddt_b: Optional[np.ndarray] = None


# ── PDB Parsing ──────────────────────────────────────────────────────

def parse_atm_record_Edited(line: str) -> dict:
    """
    Parse a single ATOM record from a PDB file into a dictionary.

    _Edited: Changed from defaultdict() to plain dict literal; added type hint.

    Args:
        line: A single line from a PDB file starting with 'ATOM'.

    Returns:
        Dictionary with keys: name, atm_no, atm_name, atm_alt, res_name,
        chain, res_no, insert, resid, x, y, z, occ, B.
    """
    return {
        'name': line[0:6].strip(),
        'atm_no': int(line[6:11]),
        'atm_name': line[12:16].strip(),
        'atm_alt': line[17],
        'res_name': line[17:20].strip(),
        'chain': line[21],
        'res_no': int(line[22:26]),
        'insert': line[26].strip(),
        'resid': line[22:29],
        'x': float(line[30:38]),
        'y': float(line[38:46]),
        'z': float(line[46:54]),
        'occ': float(line[54:60]),
        'B': float(line[60:66]),
    }


def read_pdb_Edited(pdbfile: Union[str, Path]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Read an AlphaFold2 PDB file and extract CB coordinates and pLDDT per chain.

    _Edited: Added type hints, encoding='utf-8' with errors='replace' for
    Windows compatibility, uses parse_atm_record_Edited.

    Uses CB atoms (CA for glycine) as the representative atom per residue,
    following the standard approach for contact-based analysis.

    Args:
        pdbfile: Path to an AlphaFold2 PDB file with pLDDT in the B-factor column.

    Returns:
        Tuple of (chain_coords, chain_plddt) where each is a dict mapping
        chain ID to a numpy array. chain_coords values have shape (N, 3),
        chain_plddt values have shape (N,).
    """
    chain_coords: dict[str, list] = {}
    chain_plddt: dict[str, list] = {}

    with open(pdbfile, 'r', encoding='utf-8', errors='replace') as file:
        for line in file:
            if not line.startswith('ATOM'):
                continue
            record = parse_atm_record_Edited(line)

            # CB for all residues, CA for glycine (which has no CB)
            if record['atm_name'] == 'CB' or (record['atm_name'] == 'CA' and record['res_name'] == 'GLY'):
                chain_id = record['chain']
                if chain_id in chain_coords:
                    chain_coords[chain_id].append([record['x'], record['y'], record['z']])
                    chain_plddt[chain_id].append(record['B'])
                else:
                    chain_coords[chain_id] = [[record['x'], record['y'], record['z']]]
                    chain_plddt[chain_id] = [record['B']]

    # Convert lists to numpy arrays
    for chain_id in chain_coords:
        chain_coords[chain_id] = np.array(chain_coords[chain_id])
        chain_plddt[chain_id] = np.array(chain_plddt[chain_id])

    return chain_coords, chain_plddt


def read_pdb_with_residue_ids_New(pdbfile: Union[str, Path]) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, list[int]],
    dict[str, list[str]],
]:
    """
    Extended PDB reader that also returns residue numbers and names per chain.

    _New: Entirely new function for interface residue export.

    Args:
        pdbfile: Path to an AlphaFold2 PDB file.

    Returns:
        Tuple of (chain_coords, chain_plddt, chain_residue_numbers, chain_residue_names).
        chain_residue_numbers maps chain ID to list of PDB residue numbers (int).
        chain_residue_names maps chain ID to list of 3-letter residue names (str).
    """
    chain_coords: dict[str, list] = {}
    chain_plddt: dict[str, list] = {}
    chain_res_numbers: dict[str, list[int]] = {}
    chain_res_names: dict[str, list[str]] = {}

    with open(pdbfile, 'r', encoding='utf-8', errors='replace') as file:
        for line in file:
            if not line.startswith('ATOM'):
                continue
            record = parse_atm_record_Edited(line)

            if record['atm_name'] == 'CB' or (record['atm_name'] == 'CA' and record['res_name'] == 'GLY'):
                chain_id = record['chain']
                coord = [record['x'], record['y'], record['z']]

                if chain_id in chain_coords:
                    chain_coords[chain_id].append(coord)
                    chain_plddt[chain_id].append(record['B'])
                    chain_res_numbers[chain_id].append(record['res_no'])
                    chain_res_names[chain_id].append(record['res_name'])
                else:
                    chain_coords[chain_id] = [coord]
                    chain_plddt[chain_id] = [record['B']]
                    chain_res_numbers[chain_id] = [record['res_no']]
                    chain_res_names[chain_id] = [record['res_name']]

    for chain_id in chain_coords:
        chain_coords[chain_id] = np.array(chain_coords[chain_id])
        chain_plddt[chain_id] = np.array(chain_plddt[chain_id])

    return chain_coords, chain_plddt, chain_res_numbers, chain_res_names


# ── Multi-Chain & CB-Aware PDB Reading (_New) ─────────────────────────

@dataclass
class ChainInfo_New:
    """Full per-chain information for multi-chain and CB-aware PAE mapping.

    _New: Entirely new dataclass for multi-chain support.

    Provides the data needed to correctly map CB-based contact indices
    into the full PAE matrix, even when:
      - The complex has 3+ chains (multi-chain offset calculation)
      - Some residues lack CB atoms (CB->CA index mapping)

    Attributes:
        chain_ids: Ordered list of chain identifiers found in the PDB.
        cb_coords: Dict mapping chain ID to (N_cb, 3) CB coordinate arrays.
        cb_plddt: Dict mapping chain ID to (N_cb,) pLDDT arrays (CB residues).
        ca_counts: Dict mapping chain ID to total CA residue count.
                   This matches the PKL per-residue indexing.
        cb_to_ca_map: Dict mapping chain ID to list where cb_to_ca_map[chain][i]
                      gives the CA (full-residue) index for CB array position i.
        chain_res_numbers: Dict mapping chain ID to list of PDB residue numbers
                           (one per CB atom, for interface export).
        chain_res_names: Dict mapping chain ID to list of 3-letter residue names
                         (one per CB atom).
    """
    chain_ids: list[str] = field(default_factory=list)
    cb_coords: dict[str, np.ndarray] = field(default_factory=dict)
    cb_plddt: dict[str, np.ndarray] = field(default_factory=dict)
    ca_counts: dict[str, int] = field(default_factory=dict)
    cb_to_ca_map: dict[str, list[int]] = field(default_factory=dict)
    chain_res_numbers: dict[str, list[int]] = field(default_factory=dict)
    chain_res_names: dict[str, list[str]] = field(default_factory=dict)


def read_pdb_with_chain_info_New(pdbfile: Union[str, Path]) -> ChainInfo_New:
    """
    Read an AlphaFold2 PDB file and extract full chain information.

    _New: Entirely new two-pass reader for multi-chain PAE support.

    This reader solves three problems that the basic readers cannot:
      1. It counts CA atoms per chain (matching PKL residue counts) separately
         from CB atoms (used for contact analysis), resolving the CB mismatch.
      2. It builds a CB->CA index mapping per chain, allowing correct PAE matrix
         lookup even when some residues lack CB atoms.
      3. It preserves chain ordering and per-chain counts for multi-chain
         PAE offset computation.

    Args:
        pdbfile: Path to an AlphaFold2 PDB file with pLDDT in the B-factor column.

    Returns:
        ChainInfo_New with all per-chain data needed for multi-chain-aware,
        CB-mapped interface analysis.
    """
    # First pass: collect CA residues per chain (ordered by residue number).
    # This establishes the full residue indexing that matches the PKL/PAE matrix.
    # Second pass: collect CB atoms and build the CB->CA mapping.
    #
    # We do this in two passes for clarity, though a single-pass approach
    # is possible. At PDB-parsing scale, the overhead is negligible.

    # -- Pass 1: CA atoms (all residues) --
    # Track unique (chain, resid) to handle multi-atom residues.
    chain_ca_residues: dict[str, list[str]] = defaultdict(list)  # chain -> [resid, ...]
    seen_ca: set[tuple[str, str]] = set()

    with open(pdbfile, 'r', encoding='utf-8', errors='replace') as file:
        for line in file:
            if not line.startswith('ATOM'):
                continue
            atom_name = line[12:16].strip()
            if atom_name != 'CA':
                continue
            chain_id = line[21]
            resid = line[22:29]  # residue number + insertion code
            key = (chain_id, resid)
            if key not in seen_ca:
                seen_ca.add(key)
                chain_ca_residues[chain_id].append(resid)

    # -- Pass 2: CB atoms (contact representatives) --
    # For each CB atom, record its position in the CA array (the CB->CA map).
    chain_cb_coords: dict[str, list[list[float]]] = defaultdict(list)
    chain_cb_plddt: dict[str, list[float]] = defaultdict(list)
    chain_cb_to_ca: dict[str, list[int]] = defaultdict(list)
    chain_cb_res_numbers: dict[str, list[int]] = defaultdict(list)
    chain_cb_res_names: dict[str, list[str]] = defaultdict(list)
    seen_cb: set[tuple[str, str]] = set()

    with open(pdbfile, 'r', encoding='utf-8', errors='replace') as file:
        for line in file:
            if not line.startswith('ATOM'):
                continue
            record = parse_atm_record_Edited(line)

            if record['atm_name'] == 'CB' or (record['atm_name'] == 'CA' and record['res_name'] == 'GLY'):
                chain_id = record['chain']
                resid = line[22:29]
                key = (chain_id, resid)
                if key in seen_cb:
                    continue
                seen_cb.add(key)

                chain_cb_coords[chain_id].append([record['x'], record['y'], record['z']])
                chain_cb_plddt[chain_id].append(record['B'])
                chain_cb_res_numbers[chain_id].append(record['res_no'])
                chain_cb_res_names[chain_id].append(record['res_name'])

                # Find this residue's position in the CA array
                try:
                    ca_index = chain_ca_residues[chain_id].index(resid)
                    chain_cb_to_ca[chain_id].append(ca_index)
                except ValueError:
                    # CB atom without a matching CA - shouldn't happen but
                    # use the CB count as a fallback (direct 1:1 mapping).
                    chain_cb_to_ca[chain_id].append(len(chain_cb_to_ca[chain_id]))

    # Build result with stable chain ordering (preserves PDB order)
    ordered_chains = list(chain_ca_residues.keys())

    info = ChainInfo_New(chain_ids=ordered_chains)
    for ch in ordered_chains:
        info.ca_counts[ch] = len(chain_ca_residues[ch])
        if ch in chain_cb_coords:
            info.cb_coords[ch] = np.array(chain_cb_coords[ch])
            info.cb_plddt[ch] = np.array(chain_cb_plddt[ch])
            info.cb_to_ca_map[ch] = chain_cb_to_ca[ch]
            info.chain_res_numbers[ch] = chain_cb_res_numbers[ch]
            info.chain_res_names[ch] = chain_cb_res_names[ch]

    return info


def compute_pae_chain_offsets_New(
    chain_info: ChainInfo_New,
) -> dict[str, int]:
    """
    Compute the starting offset of each chain in the full PAE matrix.

    _New: Entirely new function for multi-chain PAE support.

    The PAE matrix rows/columns correspond to all residues across all chains,
    ordered by chain appearance.  Chain A occupies rows [0, ca_count_A),
    chain B occupies [ca_count_A, ca_count_A + ca_count_B), etc.

    Args:
        chain_info: ChainInfo_New from read_pdb_with_chain_info_New().

    Returns:
        Dict mapping chain ID to its starting row/column offset in the PAE matrix.
    """
    offsets: dict[str, int] = {}
    cumulative = 0
    for ch in chain_info.chain_ids:
        offsets[ch] = cumulative
        cumulative += chain_info.ca_counts[ch]
    return offsets


def find_best_chain_pair_New(
    chain_info: ChainInfo_New,
    t: float = DEFAULT_CONTACT_THRESHOLD_New,
) -> tuple[str, str, 'ContactResult_New']:
    """
    Find the chain pair with the most inter-chain contacts.

    _New: Entirely new function for multi-chain complex support.

    For multi-chain complexes, the first two chains alphabetically may not
    be the interacting pair.  This function tests all unique pairs and
    returns the one with the highest contact count (strongest interface).

    For standard dimers (2 chains), this simply returns the only pair.

    Args:
        chain_info: ChainInfo_New from read_pdb_with_chain_info_New().
        t: Contact distance threshold in Ångströms.

    Returns:
        Tuple of (chain_id_A, chain_id_B, ContactResult_New) for the best pair.
        If no pair has any contacts, returns the pair with the two longest chains.
    """
    chains_with_coords = [ch for ch in chain_info.chain_ids if ch in chain_info.cb_coords]

    if len(chains_with_coords) < 2:
        raise ValueError(f"Fewer than 2 chains with coordinates: {chains_with_coords}")

    if len(chains_with_coords) == 2:
        # Standard dimer - no search needed
        ch1, ch2 = chains_with_coords
        pair_coords = {ch1: chain_info.cb_coords[ch1], ch2: chain_info.cb_coords[ch2]}
        pair_plddt = {ch1: chain_info.cb_plddt[ch1], ch2: chain_info.cb_plddt[ch2]}
        result = calc_pdockq_and_contacts_New(pair_coords, pair_plddt, t=t)
        return ch1, ch2, result

    # Multi-chain: try all pairs
    from itertools import combinations

    best_pair = (chains_with_coords[0], chains_with_coords[1])
    best_result = None
    best_n_contacts = -1

    for ch_a, ch_b in combinations(chains_with_coords, 2):
        pair_coords = {ch_a: chain_info.cb_coords[ch_a], ch_b: chain_info.cb_coords[ch_b]}
        pair_plddt = {ch_a: chain_info.cb_plddt[ch_a], ch_b: chain_info.cb_plddt[ch_b]}
        result = calc_pdockq_and_contacts_New(pair_coords, pair_plddt, t=t)

        if result.n_interface_contacts > best_n_contacts:
            best_n_contacts = result.n_interface_contacts
            best_pair = (ch_a, ch_b)
            best_result = result

    # Fallback: if no contacts found in any pair, pick the two longest chains
    if best_result is None:
        ch_a, ch_b = chains_with_coords[0], chains_with_coords[1]
        pair_coords = {ch_a: chain_info.cb_coords[ch_a], ch_b: chain_info.cb_coords[ch_b]}
        pair_plddt = {ch_a: chain_info.cb_plddt[ch_a], ch_b: chain_info.cb_plddt[ch_b]}
        best_result = calc_pdockq_and_contacts_New(pair_coords, pair_plddt, t=t)
        best_pair = (ch_a, ch_b)

    return best_pair[0], best_pair[1], best_result


# ── pDockQ Calculation ───────────────────────────────────────────────

def _lookup_ppv_New(pdockq_score: float) -> float:
    """
    Look up the positive predictive value for a given pDockQ score.

    _New: Extracted from inline logic in the original calc_pdockq.

    Args:
        pdockq_score: The computed pDockQ score.

    Returns:
        PPV value from the calibration table.
    """
    indices = np.argwhere(PDOCKQ_THRESHOLDS_New >= pdockq_score)
    if len(indices) > 0:
        return float(PPV_VALUES_New[indices[-1]][0])
    return float(PPV_VALUES_New[0])


def calc_pdockq_Edited(
    chain_coords: dict[str, np.ndarray],
    chain_plddt: dict[str, np.ndarray],
    t: float = DEFAULT_CONTACT_THRESHOLD_New,
) -> tuple[float, float]:
    """
    Calculate pDockQ and PPV scores for a two-chain complex.

    _Edited: Uses named constants instead of magic numbers; added type hints;
    delegates PPV lookup to _lookup_ppv_New; returns explicit floats.

    This is the original interface - returns just (pdockq, ppv).

    Args:
        chain_coords: Dict mapping chain IDs to (N, 3) coordinate arrays.
        chain_plddt: Dict mapping chain IDs to (N,) pLDDT arrays.
        t: Distance threshold in Ångströms for defining contacts.

    Returns:
        Tuple of (pdockq_score, positive_predictive_value).
    """
    chain_keys = list(chain_coords.keys())[:2]
    ch1, ch2 = chain_keys[0], chain_keys[1]
    coords1, coords2 = chain_coords[ch1], chain_coords[ch2]
    plddt1, plddt2 = chain_plddt[ch1], chain_plddt[ch2]

    # Pairwise distance matrix between all residues
    mat = np.append(coords1, coords2, axis=0)
    a_min_b = mat[:, np.newaxis, :] - mat[np.newaxis, :, :]
    dists = np.sqrt(np.sum(a_min_b.T ** 2, axis=0)).T
    l1 = len(coords1)
    contact_dists = dists[:l1, l1:]
    contacts = np.argwhere(contact_dists <= t)

    if contacts.shape[0] < 1:
        return 0.0, 0.0

    avg_if_plddt = np.average(np.concatenate([
        plddt1[np.unique(contacts[:, 0])],
        plddt2[np.unique(contacts[:, 1])],
    ]))
    n_if_contacts = contacts.shape[0]
    x = avg_if_plddt * np.log10(n_if_contacts)
    pdockq = PDOCKQ_L_New / (1 + np.exp(-PDOCKQ_K_New * (x - PDOCKQ_X0_New))) + PDOCKQ_B_New
    ppv = _lookup_ppv_New(pdockq)

    return float(pdockq), float(ppv)


def calc_pdockq_and_contacts_New(
    chain_coords: dict[str, np.ndarray],
    chain_plddt: dict[str, np.ndarray],
    t: float = DEFAULT_CONTACT_THRESHOLD_New,
) -> ContactResult_New:
    """
    Calculate pDockQ with full contact details for interface analysis.

    _New: Entirely new extended version of calc_pdockq.

    Same core calculation as calc_pdockq_Edited(), but returns a
    ContactResult_New dataclass with all intermediate data needed by
    interface_analysis.py.

    Args:
        chain_coords: Dict mapping chain IDs to (N, 3) coordinate arrays.
        chain_plddt: Dict mapping chain IDs to (N,) pLDDT arrays.
        t: Distance threshold in Ångströms for defining contacts.

    Returns:
        ContactResult_New with pDockQ, PPV, contacts, distances, and interface sets.
    """
    chain_keys = list(chain_coords.keys())[:2]
    ch1, ch2 = chain_keys[0], chain_keys[1]
    coords1, coords2 = chain_coords[ch1], chain_coords[ch2]
    plddt1, plddt2 = chain_plddt[ch1], chain_plddt[ch2]

    result = ContactResult_New(
        chain_ids=(ch1, ch2),
        n_residues_a=len(coords1),
        n_residues_b=len(coords2),
        plddt_a=plddt1,
        plddt_b=plddt2,
    )

    # Pairwise distance matrix
    mat = np.append(coords1, coords2, axis=0)
    a_min_b = mat[:, np.newaxis, :] - mat[np.newaxis, :, :]
    dists = np.sqrt(np.sum(a_min_b.T ** 2, axis=0)).T
    l1 = len(coords1)
    contact_dists = dists[:l1, l1:]
    contacts = np.argwhere(contact_dists <= t)

    if contacts.shape[0] < 1:
        return result

    # Store contact details
    result.contacts = contacts
    result.contact_distances = contact_dists[contacts[:, 0], contacts[:, 1]]
    result.n_interface_contacts = contacts.shape[0]
    result.interface_residues_a = set(np.unique(contacts[:, 0]).tolist())
    result.interface_residues_b = set(np.unique(contacts[:, 1]).tolist())

    # Interface pLDDT
    avg_if_plddt = np.average(np.concatenate([
        plddt1[np.unique(contacts[:, 0])],
        plddt2[np.unique(contacts[:, 1])],
    ]))
    result.avg_if_plddt = float(avg_if_plddt)

    # pDockQ calculation
    n_if_contacts = contacts.shape[0]
    x = avg_if_plddt * np.log10(n_if_contacts)
    pdockq = PDOCKQ_L_New / (1 + np.exp(-PDOCKQ_K_New * (x - PDOCKQ_X0_New))) + PDOCKQ_B_New
    result.pdockq = float(pdockq)
    result.ppv = _lookup_ppv_New(pdockq)

    return result


# ── CLI Entry Point ──────────────────────────────────────────────────

def main_New() -> None:
    """CLI entry point - parse arguments, read PDB, calculate and print pDockQ.

    _New: Wrapped original module-level CLI code into a function.
    """
    parser = argparse.ArgumentParser(
        description="Calculate a predicted DockQ score for a predicted structure."
    )
    parser.add_argument(
        '--pdbfile', nargs=1, type=str, default=sys.stdin,
        help='Path to PDB file. B-factor column must contain pLDDT scores from AlphaFold.',
    )
    args = parser.parse_args()

    chain_coords, chain_plddt = read_pdb_Edited(args.pdbfile[0])

    if len(chain_coords.keys()) < 2:
        print('Only one chain in pdbfile', args.pdbfile[0])
        sys.exit()

    pdockq, ppv = calc_pdockq_Edited(chain_coords, chain_plddt, t=DEFAULT_CONTACT_THRESHOLD_New)
    print('pDockQ =', np.round(pdockq, 3), 'for', args.pdbfile[0])
    print('This corresponds to a PPV of at least', ppv)


if __name__ == "__main__":
    main_New()
