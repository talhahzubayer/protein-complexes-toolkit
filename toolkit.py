#!/usr/bin/env python3
"""
Batch Processor for AlphaFold2 Protein Complex Quality Assessment - processes multiple AlphaFold2 predictions by directly importing analysis functions.

Integrated analysis modules:
    - read_af2_nojax     -> PKL metric extraction (JAX-free)
    - pdockq             -> Interface quality scoring (pDockQ/PPV)
    - interface_analysis -> Interface geometry, pLDDT, PAE features, and export
    - id_mapper          -> Gene symbols, protein names, Ensembl IDs (via --enrich)
    - database_loaders   -> Database source tagging and evidence types (via --databases)

Scalability features:
    - Multiprocessing    -> --workers N for parallel processing via ProcessPoolExecutor
    - Progress tracking  -> tqdm progress bar (auto-fallback to print if not installed)
    - Checkpointing      -> --checkpoint saves progress every 50 complexes
    - Resume capability  -> --resume skips already-processed complexes

Enrichment features:
    - --enrich           -> Adds gene symbols, protein names, Ensembl IDs, amino acid sequences via id_mapper.py (requires STRING aliases file)
    - --databases        -> Tags each complex with source databases (STRING, BioGRID, HuRI, HuMAP) and evidence types via database_loaders.py
    - Base output is 46 columns and enriched output is up to 60 columns

Usage:
    # Basic (sequential, no checkpointing)
    python toolkit.py --dir "D:\\ProteinComplexes" --output results.csv
    python toolkit.py --dir "D:\\ProteinComplexes" --output results.csv --interface --pae

    # Full analysis with parallel workers and checkpointing
    python toolkit.py --dir "D:\\ProteinComplexes" --output results.csv --interface --pae -w 4 --checkpoint
    python toolkit.py --dir "D:\\ProteinComplexes" --output results.csv --interface --pae --export-interfaces interfaces.jsonl -w 4 --checkpoint

    # With enrichment (gene symbols, protein names, sequences)
    python toolkit.py --dir "D:\\ProteinComplexes" --output results.csv --interface --pae --enrich "C:\\Users\\Talhah Zubayer\\Documents\\protein-complexes-toolkit\\data\\ppi\\9606.protein.aliases.v12.0.txt"

    # With enrichment + database source tagging
    python toolkit.py --dir "D:\\ProteinComplexes" --output results.csv --interface --pae --enrich "C:\\Users\\Talhah Zubayer\\Documents\\protein-complexes-toolkit\\data\\ppi\\9606.protein.aliases.v12.0.txt" --databases "C:\\Users\\Talhah Zubayer\\Documents\\protein-complexes-toolkit\\data\\ppi"

    # Resume an interrupted run
    python toolkit.py --dir "D:\\ProteinComplexes" --output results.csv --interface --pae -w 4 --resume

    # Verbose (sequential only - verbose is suppressed with -w > 1)
    python toolkit.py --dir "D:\\ProteinComplexes" --output results.csv --interface --pae -v
"""

import os
import sys
import argparse
import csv
import json
import logging
import statistics
import time
import re
import numpy as np
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

# tqdm for displaying progress bar - fallback if not installed
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Checkpoint Constants
CHECKPOINT_INTERVAL = 50   # Save checkpoint every N complexes
CHECKPOINT_SUFFIX = '.checkpoint.jsonl'

# Direct module imports for core analysis functions - replaces subprocess calls and temp JSON files
# JAX mocking happens once when read_af2_nojax is first imported
from read_af2_nojax import load_pkl_without_jax, extract_metrics
from pdockq import (
    read_pdb_with_chain_info_New as read_pdb_with_chain_info,
    calc_pdockq_and_contacts_New as calc_pdockq_and_contacts,
    compute_pae_chain_offsets_New as compute_pae_chain_offsets,
    find_best_chain_pair_New as find_best_chain_pair,
)
from interface_analysis import (
    analyse_interface_from_contact_result,
    compute_extended_flags,
    build_interface_export_record,
)


#------Constants----------------------------------------------------
IPTM_HIGH_THRESHOLD = 0.75
IPTM_MEDIUM_THRESHOLD = 0.5

# pDockQ thresholds anchored to PPV calibration
PDOCKQ_HIGH_THRESHOLD = 0.5     # PPV ≈ 0.90
PDOCKQ_MEDIUM_THRESHOLD = 0.23  # PPV ≈ 0.76

# pLDDT disorder thresholds
PLDDT_POOR_THRESHOLD = 50
PLDDT_LOW_THRESHOLD = 70
SUBSTANTIAL_DISORDER_FRACTION = 0.3

# PAE threshold
PAE_CONFIDENT_THRESHOLD = 5.0

# CSV base columns that are always present
CSV_FIELDNAMES_BASE = [
    'complex_name', 'protein_a', 'protein_b', 'complex_type',
    'n_chains', 'best_chain_pair',
    'iptm', 'ptm', 'ranking_confidence',
    'plddt_mean', 'plddt_median', 'plddt_min', 'plddt_max',
    'plddt_below50_fraction', 'plddt_below70_fraction',
    'num_residues', 'pae_mean',
    'pdockq', 'ppv', 'quality_tier',
    'has_pdb', 'has_pkl', 'plddt_source',
    'species', 'structure_source',
]

# Enrichment columns added when --enrich is used
CSV_FIELDNAMES_ENRICHMENT = [
    'gene_symbol_a', 'gene_symbol_b',
    'protein_name_a', 'protein_name_b',
    'ensembl_id_a', 'ensembl_id_b',
    'secondary_accessions_a', 'secondary_accessions_b',  # pipe-separated alternate UniProt accessions (e.g. merged or TrEMBL entries)
    'database_source',
    'evidence_types',
    'sequence_a', 'sequence_b',
]

# Standard amino acid three-letter to one-letter code mapping
THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}

# Interface geometry columns added when --interface is used
CSV_FIELDNAMES_INTERFACE = [
    'n_interface_contacts', 'n_interface_residues_a', 'n_interface_residues_b',
    'interface_fraction_a', 'interface_fraction_b', 'interface_symmetry',
    'contacts_per_interface_residue',
    'interface_plddt_a', 'interface_plddt_b', 'interface_plddt_combined',
    'bulk_plddt_combined', 'interface_vs_bulk_delta',
    'interface_plddt_high_fraction',
]

# Interface PAE columns added when --interface --pae is used
CSV_FIELDNAMES_INTERFACE_PAE = [
    'interface_pae_mean', 'interface_pae_median',
    'n_confident_contacts', 'confident_contact_fraction',
    'cross_chain_pae_mean',
    'n_confident_residues_a', 'n_confident_residues_b',
    'interface_confidence_score',
    'quality_tier_v2',
]

# Flags column
CSV_FIELDNAMES_FLAGS = ['interface_flags']


#---------PDB B-Factor (pLDDT) Extraction------------------------------------

def extract_plddt_from_pdb(pdb_path: Path) -> Optional[dict]:
    """Extract per-residue pLDDT from the b-factor column of an AF2 PDB file.
    Uses Cα atoms to get one pLDDT per residue, then computes summary statistics and disorder fractions.
    Args:
        pdb_path: Path to an AlphaFold2 .pdb file.
    Returns:
        Dictionary of pLDDT statistics, or None if parsing fails.
    """
    plddt_values: list[float] = []
    try:
        with open(pdb_path, 'r', encoding='utf-8', errors='replace') as pdb_file:
            for line in pdb_file:
                if not line.startswith('ATOM'):
                    continue
                atom_name = line[12:16].strip()
                if atom_name != 'CA':
                    continue
                try:
                    bfactor = float(line[60:66].strip())
                    plddt_values.append(bfactor)
                except (ValueError, IndexError):
                    continue
        if not plddt_values:
            return None
        total_residues = len(plddt_values)
        below_50_count = sum(1 for val in plddt_values if val < PLDDT_POOR_THRESHOLD) # Yields the number 1 every time the if condition is true and adds all 1s to get total count
        below_70_count = sum(1 for val in plddt_values if val < PLDDT_LOW_THRESHOLD) 
        return {
            'plddt_mean': statistics.mean(plddt_values),
            'plddt_median': statistics.median(plddt_values),
            'plddt_min': min(plddt_values),
            'plddt_max': max(plddt_values),
            'num_residues': total_residues,
            'plddt_below50_fraction': round(below_50_count / total_residues, 4),
            'plddt_below70_fraction': round(below_70_count / total_residues, 4),
        }
    except Exception as error:
        print(f"  Warning: Could not parse b-factors from {pdb_path}: {error}", file=sys.stderr)
        return None

#--------File Discovery & Parsing-------------------------------------------------

def parse_complex_name(filename: str) -> tuple[str, str, str, str]:
    """Parse protein IDs and complex type from an AlphaFold2 output filename.
    Args:
        filename: The filename without directory path to parse.
    Returns:
        Tuple of complex_name, protein_a_id, protein_b_id, complex_type
    """
    clean_name = filename

    # Strip file extensions (.pdb, .pkl) if present
    for ext in ('.pdb', '.pkl'):
        if clean_name.endswith(ext):
            clean_name = clean_name[:-len(ext)]

    # Strip AlphaFold2 model suffixes - handles both naming conventions:
    # PKL files use:  _result_model_X_multimer_v3_pred_Y
    # PDB files use:  _relaxed_model_X_multimer_v3_pred_Y
    AF2_SUFFIX_PATTERN = re.compile(
        r'_(result|relaxed)_model_\d+_multimer_v\d+_pred_\d+$'
    )
    clean_name = AF2_SUFFIX_PATTERN.sub('', clean_name)

    # Handles legacy _results / .results suffixes from older datasets
    for suffix in ('.results', '_results'):
        if clean_name.endswith(suffix):
            clean_name = clean_name[:-len(suffix)]

    # Handle doubled-name format: A_B_A_B -> A_B
    name_parts = clean_name.split('_')
    if len(name_parts) >= 4:
        midpoint = len(name_parts) // 2
        first_half = '_'.join(name_parts[:midpoint])
        second_half = '_'.join(name_parts[midpoint:])
        if first_half == second_half:
            clean_name = first_half
            name_parts = clean_name.split('_')

    if len(name_parts) >= 2:
        protein_a_id = name_parts[0]
        protein_b_id = name_parts[1]
    else:
        protein_a_id = clean_name
        protein_b_id = clean_name

    complex_type = 'Homodimer' if protein_a_id == protein_b_id else 'Heterodimer'
    return clean_name, protein_a_id, protein_b_id, complex_type


def find_paired_data_files(directory: str) -> dict[str, dict[str, Path]]:
    """Find matching PDB and PKL files in a directory.
    Args:
        directory: Path to the directory containing PDB/PKL files.
    Returns:
        Dictionary mapping complex names to dicts with 'pdb' and/or 'pkl' paths.
    """
    data_directory = Path(directory)
    complexes: dict[str, dict[str, Path]] = defaultdict(dict)

    for file_path in data_directory.iterdir():
        if not file_path.is_file():
            continue
        complex_name, _, _, _ = parse_complex_name(file_path.name)
        if file_path.suffix == '.pdb':
            complexes[complex_name]['pdb'] = file_path
        elif file_path.suffix == '.pkl':
            complexes[complex_name]['pkl'] = file_path

    return dict(complexes)

#------Quality Classification------------------------------------------------------

def classify_prediction_quality(iptm_score: Optional[float], pdockq_score: Optional[float]) -> str:
    """Classify a prediction into a quality tier based on ipTM and pDockQ - Version 1 (2-metric classification).
    Args:
        iptm_score: Interface pTM score or None if unavailable.
        pdockq_score: pDockQ docking quality score or None if unavailable.
    Returns:
        Quality tier string: 'High', 'Medium', or 'Low'
    """
    safe_iptm = iptm_score or 0
    safe_pdockq = pdockq_score or 0

    if safe_iptm >= IPTM_HIGH_THRESHOLD and safe_pdockq >= PDOCKQ_HIGH_THRESHOLD:
        return 'High'
    elif safe_iptm >= IPTM_MEDIUM_THRESHOLD and safe_pdockq >= PDOCKQ_MEDIUM_THRESHOLD:
        return 'Medium'
    else:
        return 'Low'

# Interface confidence thresholds for tier reclassification - calibrated from the 9,573-complex dataset
UPGRADE_LOW_THRESHOLD = 0.64     # Low    -> High when composite score >= 0.64 (90th percentile of Low-tier scores)
UPGRADE_MEDIUM_THRESHOLD = 0.80  # Medium -> High when composite score >= 0.80 (90th percentile of Medium-tier scores)
DOWNGRADE_HIGH_THRESHOLD = 0.65  # High   -> Medium when composite score <= 0.65 (10th percentile of High-tier scores)

def classify_prediction_quality_v2(iptm_score: Optional[float], pdockq_score: Optional[float], interface_confidence: Optional[float] = None) -> str:
    """Enhanced quality classification incorporating interface confidence.
    Starts from the original 2-metric tier and adjusts based on the composite interface confidence score.
    This catches:
      - False negatives: Low/Medium tier with excellent interface evidence (pDockQ is size-sensitive and can penalise small genuine interfaces)
      - False positives: High tier where interface metrics are poor (headline scores mask a weak binding site)
    Falls back to original v1 classification when the composite score is unavailable (no PAE data).
    Args:
        iptm_score: Interface pTM score (or None if unavailable).
        pdockq_score: pDockQ docking quality score (or None if unavailable).
        interface_confidence: Composite interface confidence score from compute_interface_confidence() [0.0-1.0] or None if unavailable.
    Returns:
        Quality tier string: 'High', 'Medium', or 'Low'.
    """
    base_tier = classify_prediction_quality(iptm_score, pdockq_score)

    if interface_confidence is None:
        return base_tier

    # Upgrade: strong interface evidence overrides weak headline metrics
    if base_tier == 'Low' and interface_confidence >= UPGRADE_LOW_THRESHOLD:
        return 'High'
    if base_tier == 'Medium' and interface_confidence >= UPGRADE_MEDIUM_THRESHOLD:
        return 'High'

    # Downgrade: poor interface despite good headline metrics
    if base_tier == 'High' and interface_confidence <= DOWNGRADE_HIGH_THRESHOLD:
        return 'Medium'
    
    return base_tier

#------------------------------------------------------Core Processing---------------------------------------------------------------------------------

def _extract_pkl_metrics(file_paths: dict[str, Path], row: dict, *, run_interface_pae: bool, verbose: bool) -> Optional[np.ndarray]:
    """Extract ipTM, pTM, pLDDT metrics from a PKL file and optionally retain the PAE matrix.
    Args:
        file_paths: Dict with optional 'pdb' and 'pkl' Path entries.
        row: Result dict to update in-place with PKL metrics.
        run_interface_pae: Whether to retain the PAE matrix for downstream interface analysis.
        verbose: Whether to print per-step progress.
    Returns:
        PAE matrix as a numpy array if available and requested - otherwise None.
    """
    pae_matrix = None
    if 'pkl' not in file_paths:
        return pae_matrix
    
    try:
        prediction_result = load_pkl_without_jax(file_paths['pkl'])
        pkl_metrics = extract_metrics(prediction_result)
        row.update(pkl_metrics)
        row['plddt_source'] = 'pkl'

        # Keep PAE matrix in memory for interface analysis - discard after use
        if run_interface_pae and 'predicted_aligned_error' in prediction_result:
            pae_matrix = np.asarray(prediction_result['predicted_aligned_error'])
        if verbose:
            print(f"  PKL -> ipTM={pkl_metrics.get('iptm', 'N/A')}")

    except Exception as error:
        print(f"  Warning: PKL extraction failed for {file_paths['pkl']}: {error}", file=sys.stderr)

    return pae_matrix

def _extract_pdb_plddt(file_paths: dict[str, Path], row: dict, *, verbose: bool) -> None:
    """Extract per-residue pLDDT from PDB b-factors as a fallback when PKL is unavailable.
    Args:
        file_paths: Dict with optional 'pdb' and 'pkl' Path entries.
        row: Result dict to update in-place with pLDDT statistics.
        verbose: Whether to print per-step progress.
    """
    if 'pdb' not in file_paths:
        return

    pdb_plddt = extract_plddt_from_pdb(file_paths['pdb'])
    if pdb_plddt:
        row['plddt_below50_fraction'] = pdb_plddt['plddt_below50_fraction']
        row['plddt_below70_fraction'] = pdb_plddt['plddt_below70_fraction']
        if row.get('plddt_mean') is None:
            row['plddt_mean'] = pdb_plddt['plddt_mean']
            row['plddt_median'] = pdb_plddt['plddt_median']
            row['plddt_min'] = pdb_plddt['plddt_min']
            row['plddt_max'] = pdb_plddt['plddt_max']
            row['num_residues'] = pdb_plddt['num_residues']
            row['plddt_source'] = 'pdb'
            if verbose:
                print(f"  PDB -> pLDDT fallback: mean={pdb_plddt['plddt_mean']:.1f}")

def _compute_pdockq_and_chain_info(
    file_paths: dict[str, Path],
    row: dict,
    pae_matrix: Optional[np.ndarray],
    *,
    run_interface_pae: bool,
    verbose: bool,
) -> tuple[Optional[object], Optional[object], Optional[tuple], Optional[tuple]]:
    """Read PDB chain structure, find the best interacting chain pair, and compute pDockQ.
    Also pre-computes PAE chain offsets and CB-to-CA maps needed for downstream interface analysis.
    Args:
        file_paths: Dict with optional 'pdb' and 'pkl' Path entries.
        row: Result dict to update in-place with pDockQ, chain pair, and sequence data.
        pae_matrix: PAE matrix from PKL or None if unavailable.
        run_interface_pae: Whether to compute PAE offsets and CB-to-CA maps.
        verbose: Whether to print per-step progress.
    Returns:
        Tuple of contact_result, chain_info, pae_chain_offsets, cb_to_ca_maps.
        Any element may be None if the corresponding step was skipped or failed.
    """
    contact_result = None
    chain_info = None
    pae_chain_offsets = None
    cb_to_ca_maps = None

    # Default chain count: parse_complex_name() infers a dimer pair from the filename,
    # so 2 is the fallback when no PDB is available.
    row['n_chains'] = 2

    if 'pdb' not in file_paths:
        return contact_result, chain_info, pae_chain_offsets, cb_to_ca_maps

    try:
        chain_info = read_pdb_with_chain_info(str(file_paths['pdb']))
        if len(chain_info.chain_ids) >= 2:
            # Find the best interacting chain pair - also handles multi-chain 
            ch_a, ch_b, contact_result = find_best_chain_pair(chain_info, t=8)
            row['n_chains'] = len(chain_info.chain_ids)
            row['best_chain_pair'] = f'{ch_a}-{ch_b}'
            row['pdockq'] = round(contact_result.pdockq, 4)
            row['ppv'] = round(contact_result.ppv, 4)
            if len(chain_info.chain_ids) > 2 and verbose:
                print(f"  Multi-chain ({len(chain_info.chain_ids)} chains): "
                      f"best pair = {ch_a}-{ch_b}")

            # Pre-compute PAE mapping parameters for interface analysis
            if run_interface_pae and pae_matrix is not None:
                offsets = compute_pae_chain_offsets(chain_info)
                pae_chain_offsets = (offsets[ch_a], offsets[ch_b])

                # Build CB->CA maps for the selected chain pair
                map_a = chain_info.cb_to_ca_map.get(ch_a, [])
                map_b = chain_info.cb_to_ca_map.get(ch_b, [])
                if map_a and map_b:
                    # Only use maps when there's actually a mismatch
                    ca_a = chain_info.ca_counts.get(ch_a, 0)
                    ca_b = chain_info.ca_counts.get(ch_b, 0)
                    cb_a = contact_result.n_residues_a
                    cb_b = contact_result.n_residues_b
                    if ca_a != cb_a or ca_b != cb_b:
                        cb_to_ca_maps = (map_a, map_b)
                        if verbose:
                            print(f"  CB->CA mapping active: "
                                  f"chain {ch_a} CB={cb_a}/CA={ca_a}, "
                                  f"chain {ch_b} CB={cb_b}/CA={ca_b}")

            # Extract amino acid sequences from chain residue names
            if hasattr(chain_info, 'chain_res_names'):
                res_a = chain_info.chain_res_names.get(ch_a, [])
                res_b = chain_info.chain_res_names.get(ch_b, [])
                row['sequence_a'] = ''.join(THREE_TO_ONE.get(r, 'X') for r in res_a)
                row['sequence_b'] = ''.join(THREE_TO_ONE.get(r, 'X') for r in res_b)

            if verbose and row.get('pdockq') is not None:
                print(f"  PDB -> pDockQ={row['pdockq']}")
        else:
            row['n_chains'] = len(chain_info.chain_ids)
            print(f"  Warning: <2 chains in {file_paths['pdb']}", file=sys.stderr)

    except Exception as error:
        print(f"  Warning: pDockQ failed for {file_paths['pdb']}: {error}", file=sys.stderr)
        contact_result = None
        chain_info = None

    return contact_result, chain_info, pae_chain_offsets, cb_to_ca_maps


def _compute_interface_features(
    complex_name: str,
    row: dict,
    contact_result: Optional[object],
    chain_info: Optional[object],
    pae_matrix: Optional[np.ndarray],
    pae_chain_offsets: Optional[tuple],
    cb_to_ca_maps: Optional[tuple],
    *,
    run_interface_pae: bool,
    export_interfaces: bool,
    verbose: bool,
) -> None:
    """Run interface geometry, pLDDT, PAE, and flag analysis on a contact result.
    Args:
        complex_name: Parsed complex identifier (for error messages).
        row: Result dict to update in-place with interface features and flags.
        contact_result: pDockQ contact result from find_best_chain_pair (or None).
        chain_info: Chain info from read_pdb_with_chain_info (or None).
        pae_matrix: PAE matrix from PKL (or None).
        pae_chain_offsets: Tuple of (offset_a, offset_b) for PAE indexing (or None).
        cb_to_ca_maps: Tuple of (map_a, map_b) for CB-to-CA index correction (or None).
        run_interface_pae: Whether to include PAE in the interface analysis.
        export_interfaces: Whether to capture confident residue data for JSONL export.
        verbose: Whether to print per-step progress.
    """
    try:
        if contact_result is None or contact_result.n_interface_contacts < 0:
            return

        # Prepare chain_residue_numbers for export if needed
        export_res_numbers = None
        if export_interfaces and chain_info is not None:
            export_res_numbers = chain_info.chain_res_numbers

        # chain_lengths=None - superseded by chain_offsets
        interface_features = analyse_interface_from_contact_result(
            contact_result,
            pae_matrix=pae_matrix if run_interface_pae else None,
            chain_lengths=None,
            chain_residue_numbers=export_res_numbers,
            chain_offsets=pae_chain_offsets,
            cb_to_ca_maps=cb_to_ca_maps,
        )

        # Flatten interface features into the row - skip pdockq/ppv since we already set them above
        skip_keys = {'pdockq', 'ppv', 'avg_interface_plddt', 'flags', 'confident_contacts'}

        # Only skip residue number lists from CSV - they go to JSONL export
        if not export_interfaces:
            skip_keys.update({'confident_residue_numbers_a', 'confident_residue_numbers_b', 'confident_residue_indices_a', 'confident_residue_indices_b'})
        else:
            # Still skip the raw indices - only keep PDB residue numbers
            skip_keys.update({'confident_residue_indices_a', 'confident_residue_indices_b'})
        for key, value in interface_features.items():
            if key not in skip_keys:
                row[key] = value

        # Extended flags: structural + paradox detection
        flags = compute_extended_flags(interface_features, iptm=row.get('iptm'), pdockq=row.get('pdockq'), disorder_fraction=row.get('plddt_below50_fraction'))
        row['interface_flags'] = ','.join(flags) if flags else ''

        if verbose and interface_features.get('n_interface_contacts', 0) > 0:
            n_contacts = interface_features['n_interface_contacts']
            if_plddt = interface_features.get('interface_plddt_combined', 'N/A')
            delta = interface_features.get('interface_vs_bulk_delta', 'N/A')
            print(f"  Interface -> {n_contacts} contacts, "
                  f"pLDDT={if_plddt}, delta={delta}")
            if run_interface_pae and interface_features.get('confident_contact_fraction') is not None:
                print(f"  Interface PAE -> confident={interface_features['confident_contact_fraction']:.1%}")
            score = interface_features.get('interface_confidence_score')
            if score is not None:
                print(f"  Composite score: {score:.4f}")

    except Exception as error:
        print(f"  Warning: Interface analysis failed for {complex_name}: {error}", file=sys.stderr)


def process_single_complex(complex_name: str, file_paths: dict[str, Path], *, run_interface: bool = False, run_interface_pae: bool = False, export_interfaces: bool = False, verbose: bool = False) -> dict:
    """Run all analysis steps on a single protein complex.
    Args:
        complex_name: Parsed complex identifier.
        file_paths: Dict with optional 'pdb' and 'pkl' Path entries.
        run_interface: Whether to compute interface geometry + pLDDT features.
        run_interface_pae: Whether to also compute PAE-based interface features (requires both PDB and PKL - implies run_interface=True).
        export_interfaces: Whether to capture confident interface residue data for JSONL export (requires --interface --pae).
        verbose: Whether to print per-step progress.
    Returns:
        Dictionary of results for this complex (one CSV row).
    """
    _, protein_a_id, protein_b_id, complex_type = parse_complex_name(complex_name)

    row: dict = {'complex_name': complex_name,
                 'protein_a': protein_a_id,
                 'protein_b': protein_b_id,
                 'complex_type': complex_type,
                 'has_pdb': 'pdb' in file_paths,
                 'has_pkl': 'pkl' in file_paths,
                 'species': 'Homo sapiens (9606)',
                 'structure_source': 'AlphaFold2_prediction',
                 }

    pae_matrix = _extract_pkl_metrics(file_paths, row, run_interface_pae=run_interface_pae, verbose=verbose)
    _extract_pdb_plddt(file_paths, row, verbose=verbose)
    contact_result, chain_info, pae_chain_offsets, cb_to_ca_maps = _compute_pdockq_and_chain_info(file_paths, row, pae_matrix, run_interface_pae=run_interface_pae, verbose=verbose)

    if run_interface and 'pdb' in file_paths:
        _compute_interface_features(
            complex_name, row, contact_result, chain_info,
            pae_matrix, pae_chain_offsets, cb_to_ca_maps,
            run_interface_pae=run_interface_pae, export_interfaces=export_interfaces, verbose=verbose,
        )

    # Quality tier classification
    row['quality_tier'] = classify_prediction_quality(row.get('iptm'), row.get('pdockq'))
    row['quality_tier_v2'] = classify_prediction_quality_v2(row.get('iptm'), row.get('pdockq'), row.get('interface_confidence_score'))

    return row

#----------------------------Results Output-------------------------------------

def get_csv_fieldnames(include_interface: bool = False, include_pae: bool = False, include_enrichment: bool = False) -> list[str]:
    """Build the CSV column list based on enabled features."""
    fieldnames = list(CSV_FIELDNAMES_BASE)
    if include_enrichment:
        fieldnames.extend(CSV_FIELDNAMES_ENRICHMENT)
    if include_interface:
        fieldnames.extend(CSV_FIELDNAMES_INTERFACE)
        fieldnames.extend(CSV_FIELDNAMES_FLAGS)
    if include_pae:
        fieldnames.extend(CSV_FIELDNAMES_INTERFACE_PAE)
    return fieldnames

def write_results_csv(results: list[dict], output_path: str, include_interface: bool = False, include_pae: bool = False, include_enrichment: bool = False) -> None:
    """Write batch analysis results to a CSV file.
    Args:
        results: List of per-complex result dictionaries.
        output_path: File path for the output CSV.
        include_interface: Whether to include interface columns.
        include_pae: Whether to include PAE interface columns.
        include_enrichment: Whether to include enrichment columns (gene symbols, names, database sources, sequences).
    """
    fieldnames = get_csv_fieldnames(include_interface, include_pae, include_enrichment)

    with open(output_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)

def write_interface_exports(results: list[dict], output_path: str, min_tier: str = 'Medium') -> int:
    """Export confident interface residue data to a JSONL file.
    Each line is a self-contained JSON record describing one complex's confident interface residues - the computationally identified binding hot-spots that pass both PAE and pLDDT confidence filters.
    Only exports complexes that meet the quality tier threshold and have confident residue data available.
    Args:
        results: List of per-complex result dictionaries from batch processing.
        output_path: File path for the output JSONL file.
        min_tier: Minimum v2 quality tier to export. 'High' exports only High-tier complexes. 'Medium' exports High and Medium tiers. 'Low' exports all tiers.
    Returns:
        Number of complexes exported.
    """
    tier_rank = {'High': 3, 'Medium': 2, 'Low': 1}
    min_rank = tier_rank.get(min_tier, 2)
    exported_count = 0

    with open(output_path, 'w', encoding='utf-8') as jsonl_file:
        for row in results:
            v2_tier = row.get('quality_tier_v2', 'Low')

            # Skip complexes below quality threshold
            if tier_rank.get(v2_tier, 0) < min_rank:
                continue

            # Skip complexes without confident residue data
            conf_res_a = row.get('confident_residue_numbers_a', [])
            conf_res_b = row.get('confident_residue_numbers_b', [])
            if not conf_res_a and not conf_res_b:
                continue

            # Parse flags back from comma-separated string
            flags_str = row.get('interface_flags', '')
            flags_list = [f.strip() for f in flags_str.split(',') if f.strip()] if flags_str else []
            record = build_interface_export_record(
                complex_name=row['complex_name'],
                protein_a=row['protein_a'],
                protein_b=row['protein_b'],
                quality_tier_v2=v2_tier,
                interface_confidence_score=row.get('interface_confidence_score'),
                confident_residue_numbers_a=conf_res_a,
                confident_residue_numbers_b=conf_res_b,
                flags=flags_list,
                iptm=row.get('iptm'),
                pdockq=row.get('pdockq'),
                n_interface_contacts=row.get('n_interface_contacts'),
                confident_contact_fraction=row.get('confident_contact_fraction'),
                interface_plddt_combined=row.get('interface_plddt_combined'),
            )
            jsonl_file.write(json.dumps(record) + '\n')
            exported_count += 1

    return exported_count

#------------------Enrichment (gene symbols, database sources)-----------------------

def enrich_results(results: list[dict], lookup: dict[str, dict], database_pair_sets: Optional[dict[str, set]] = None, database_evidence: Optional[dict[str, set]] = None) -> None:
    """Enrich result rows with gene symbols, protein names, and database sources. Modifies the result dictionary in-place.
    Args:
        results: List of per-complex result dicts from batch processing.
        lookup: UniProt-keyed lookup dict from build_uniprot_lookup().
        database_pair_sets: Optional dict of {db_name: set of normalised UniProt pairs} for "source of complex" tagging.
        database_evidence: Optional dict of {db_name: set of evidence type strings}, pre-computed once to avoid scanning large DataFrames per complex.
    """
    from overlap_analysis import normalise_pair

    for row in results:
        prot_a = row.get('protein_a', '')
        prot_b = row.get('protein_b', '')
        info_a = lookup.get(prot_a, {})
        info_b = lookup.get(prot_b, {})
        row['gene_symbol_a'] = info_a.get('gene_symbol', '')
        row['gene_symbol_b'] = info_b.get('gene_symbol', '')
        row['protein_name_a'] = info_a.get('protein_name', '')
        row['protein_name_b'] = info_b.get('protein_name', '')
        row['ensembl_id_a'] = info_a.get('ensembl_protein_id', '')
        row['ensembl_id_b'] = info_b.get('ensembl_protein_id', '')

        # Secondary accessions - pipe-separated alternate UniProt accessions.
        # A single ENSP in STRING aliases can map to multiple UniProt IDs (e.g. reviewed Swiss-Prot + unreviewed TrEMBL, or merged/legacy accessions). 
        # The first (alphabetically sorted) is the primary accession used as protein_a/protein_b; the rest are joined with '|' here.  Example: "P38398|Q6IN68" for BRCA1.
        ensp_a = info_a.get('ensembl_protein_id', '')
        ensp_b = info_b.get('ensembl_protein_id', '')
        row['secondary_accessions_a'] = info_a.get('secondary_accessions', '')
        row['secondary_accessions_b'] = info_b.get('secondary_accessions', '')

        # Database source tagging
        if database_pair_sets:
            pair = normalise_pair(prot_a, prot_b)
            sources = [
                name for name, pair_set in sorted(database_pair_sets.items())
                if pair in pair_set
            ]
            row['database_source'] = '|'.join(sources)

            # Collect evidence types from matched databases
            if database_evidence:
                evidence_set: set[str] = set()
                for db_name in sources:
                    ev = database_evidence.get(db_name)
                    if ev:
                        evidence_set.update(ev)
                row['evidence_types'] = '|'.join(sorted(evidence_set))
            else:
                row['evidence_types'] = ''
        else:
            row['database_source'] = ''
            row['evidence_types'] = ''

def _checkpoint_path(output_path: str) -> Path:
    """Derive a checkpoint filepath from the output CSV path."""
    return Path(output_path).with_suffix(CHECKPOINT_SUFFIX)

def load_checkpoint(output_path: str) -> dict[str, dict]:
    """Load previously completed results from a checkpoint file.
    Args:
        output_path: The main output CSV path (checkpoint path derived from it).
    Returns:
        Dictionary mapping complex_name -> result dict for already-processed complexes. 
        Returns an empty dict if no checkpoint file exists.
    """
    ckpt = _checkpoint_path(output_path)
    if not ckpt.exists():
        logger.info("No checkpoint file found at %s", ckpt)
        return {}

    completed: dict[str, dict] = {}
    with open(ckpt, 'r', encoding='utf-8') as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                name = row.get('complex_name')
                if name:
                    completed[name] = row
            except json.JSONDecodeError as error:
                logger.debug("Corrupt checkpoint line %d: %s", line_number, error)
                print(f"  Warning: Skipping corrupt checkpoint line {line_number}: {error}",
                      file=sys.stderr)
    logger.info("Checkpoint loaded: %d complexes from %s", len(completed), ckpt)
    return completed

def save_checkpoint(results: list[dict], output_path: str) -> None:
    """Write all completed results to the checkpoint file (atomic overwrite).
    Args:
        results: List of per-complex result dictionaries completed so far.
        output_path: The main output CSV path (checkpoint path derived from it).
    """
    ckpt = _checkpoint_path(output_path)
    tmp = ckpt.with_suffix('.tmp')
    with open(tmp, 'w', encoding='utf-8') as fh:
        for row in results:
            fh.write(json.dumps(row, default=str) + '\n')
    tmp.replace(ckpt)
    logger.info("Checkpoint saved: %d complexes -> %s", len(results), ckpt)

def remove_checkpoint(output_path: str) -> None:
    """Remove the checkpoint file after successful completion."""
    ckpt = _checkpoint_path(output_path)
    if ckpt.exists():
        ckpt.unlink()
        logger.info("Checkpoint removed: %s", ckpt)

def _make_progress_bar(total: int, desc: str = "Processing"):
    """Create a tqdm progress bar, or a simple fallback counter.
    Returns:
        A context-manager-compatible object with an ``update()`` method and a ``set_postfix_str()`` method (no-op on fallback).
    """
    if tqdm is not None:
        return tqdm(total=total, desc=desc, unit="complex", ncols=100, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}")

    # Minimal fallback when tqdm is not installed
    class _FallbackBar:
        """Print-based fallback when tqdm is unavailable."""
        def __init__(self, total: int):
            self.n = 0
            self.total = total

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def update(self, n: int = 1) -> None:
            self.n += n
            if self.n % 25 == 0 or self.n == self.total:
                print(f"  [{self.n}/{self.total}] complexes processed")

        def set_postfix_str(self, s: str, refresh: bool = True) -> None:
            pass

    return _FallbackBar(total)

#-------------------------Worker wrapper for multiprocessing-----------------------------------------

def _worker_initializer():
    """Run once per worker process on startup.
    Forces the module import chain to execute in the worker so that any import failure raises BrokenProcessPool immediately rather than causing a silent hang.
    """
    # Import is intentionally unused - we only need the side effect of loading the module. "noqa: F401" suppresses the flake8 "imported but unused" warning.
    import toolkit  # noqa: F401

def _worker_process_complex(args_tuple: tuple) -> dict:
    """Top-level wrapper for process_single_complex that unpacks a tuple.
    ProcessPoolExecutor requires a picklable callable with a single argument. This unpacks the argument tuple and forwards to the real function.
    """
    complex_name, file_paths, kwargs = args_tuple
    return process_single_complex(complex_name, file_paths, **kwargs)

def run_batch_parallel(
    sorted_complexes: list[tuple[str, dict[str, Path]]],
    *,
    run_interface: bool,
    run_interface_pae: bool,
    export_interfaces: bool,
    verbose: bool,
    workers: int,
    output_path: str,
    enable_checkpoint: bool,
    resumed_results: list[dict],
) -> list[dict]:
    """Process complexes in parallel with progress tracking and checkpointing.
    Args:
        sorted_complexes: List of (complex_name, file_paths) tuples to process.
        run_interface: Whether to compute interface features.
        run_interface_pae: Whether to compute PAE features.
        export_interfaces: Whether to capture residue data for JSONL export.
        verbose: Whether to print verbose per-complex output.
        workers: Number of parallel workers (1 = sequential).
        output_path: Output CSV path (used to derive checkpoint path).
        enable_checkpoint: Whether to save periodic checkpoints.
        resumed_results: Already-completed results loaded from a checkpoint.
    Returns:
        Complete list of result dictionaries (resumed + newly processed) - sorted by complex name.
    """
    total = len(sorted_complexes) + len(resumed_results)
    to_process = len(sorted_complexes)

    if resumed_results:
        print(f"Resumed {len(resumed_results)} complexes from checkpoint, "
              f"{to_process} remaining\n")

    # Shared kwargs for every worker call
    shared_kwargs = dict(
        run_interface=run_interface,
        run_interface_pae=run_interface_pae,
        export_interfaces=export_interfaces,
        verbose=verbose and workers == 1,  # Verbose only meaningful in sequential mode
    )

    # Build work items
    work_items = [
        (name, paths, shared_kwargs)
        for name, paths in sorted_complexes
    ]

    results = list(resumed_results)  # Accumulate into a mutable list
    newly_processed = 0
    start_time = time.monotonic()

    with _make_progress_bar(total, desc="Analysing complexes") as pbar:
        # Account for already-completed work
        if resumed_results:
            pbar.update(len(resumed_results))

        if workers == 1:
            #-------------------Sequential mode (preserves verbose output)---------------------
            for complex_name, file_paths, kwargs in work_items:
                row = process_single_complex(complex_name, file_paths, **kwargs)
                results.append(row)
                newly_processed += 1
                tier = row.get('quality_tier', '?')
                pbar.set_postfix_str(f"{complex_name} -> {tier}")
                pbar.update(1)

                if enable_checkpoint and newly_processed % CHECKPOINT_INTERVAL == 0:
                    save_checkpoint(results, output_path)

        else:
            #----------------------Parallel mode------------------------------------------------
            # verbose per-complex output is suppressed in parallel mode because interleaved prints from multiple workers are unreadable
            # Instead we show the most recent complex and its quality tier in the progress bar
            print(f"Starting {workers} worker processes...", flush=True)
            with ProcessPoolExecutor(max_workers=workers, initializer=_worker_initializer) as executor:
                future_to_name = {executor.submit(_worker_process_complex, item): item[0] for item in work_items}
                for future in as_completed(future_to_name):
                    complex_name = future_to_name[future]
                    try:
                        row = future.result(timeout=300)
                        results.append(row)
                        newly_processed += 1
                        tier = row.get('quality_tier', '?')
                        pbar.set_postfix_str(f"{complex_name} -> {tier}")
                    except Exception as error:
                        print(f"\n  Error processing {complex_name}: {error}",
                              file=sys.stderr)
                        # Create a minimal error row so we don't lose track
                        results.append({
                            'complex_name': complex_name,
                            'quality_tier': 'Error',
                            '_error': str(error),
                        })
                    pbar.update(1)
                    if enable_checkpoint and newly_processed % CHECKPOINT_INTERVAL == 0:
                        save_checkpoint(results, output_path)

    elapsed = time.monotonic() - start_time
    rate = to_process / elapsed if elapsed > 0 else 0
    print(f"\nProcessed {to_process} complexes in {elapsed:.1f}s "
          f"({rate:.1f} complexes/s, {workers} worker{'s' if workers > 1 else ''})")

    # Final checkpoint before CSV write
    if enable_checkpoint and newly_processed > 0:
        save_checkpoint(results, output_path)

    # Sort by complex name for deterministic output order
    results.sort(key=lambda r: r.get('complex_name', ''))
    return results


def _aggregate_summary_statistics(results: list[dict], include_interface: bool = False) -> dict:
    """Compute summary statistics from batch results for display.
    Args:
        results: List of per-complex result dictionaries.
        include_interface: Whether to include interface statistics.
    Returns:
        Dictionary of aggregated statistics keyed by section name.
    """
    total = len(results)
    stats: dict = {
        'total_complexes': total,
        'homodimer_count': sum(1 for row in results if row['complex_type'] == 'Homodimer'),
        'heterodimer_count': sum(1 for row in results if row['complex_type'] == 'Heterodimer'),
        'quality_high': sum(1 for row in results if row.get('quality_tier') == 'High'),
        'quality_medium': sum(1 for row in results if row.get('quality_tier') == 'Medium'),
        'quality_low': sum(1 for row in results if row.get('quality_tier') == 'Low'),
        'iptm_values': [row['iptm'] for row in results if row.get('iptm')],
        'pdockq_values': [row['pdockq'] for row in results if row.get('pdockq')],
        'below50_values': [row['plddt_below50_fraction'] for row in results if row.get('plddt_below50_fraction') is not None],
        'below70_values': [row['plddt_below70_fraction'] for row in results if row.get('plddt_below70_fraction') is not None],
        'pkl_source_count': sum(1 for row in results if row.get('plddt_source') == 'pkl'),
        'pdb_fallback_count': sum(1 for row in results if row.get('plddt_source') == 'pdb'),
        'no_plddt_count': sum(1 for row in results if row.get('plddt_source') is None),
    }

    if include_interface:
        stats['contact_counts'] = [row['n_interface_contacts'] for row in results if row.get('n_interface_contacts') is not None]
        stats['if_plddt_values'] = [row['interface_plddt_combined'] for row in results if row.get('interface_plddt_combined') is not None]
        stats['delta_values'] = [row['interface_vs_bulk_delta'] for row in results if row.get('interface_vs_bulk_delta') is not None]
        all_flags: dict[str, int] = defaultdict(int)
        for row in results:
            flags_str = row.get('interface_flags', '')
            if flags_str:
                for flag in flags_str.split(','):
                    all_flags[flag.strip()] += 1
        stats['all_flags'] = dict(all_flags)
        stats['confident_fractions'] = [row['confident_contact_fraction'] for row in results if row.get('confident_contact_fraction') is not None]
        stats['composite_scores'] = [row['interface_confidence_score'] for row in results if row.get('interface_confidence_score') is not None]

        v2_tiers = [row.get('quality_tier_v2') for row in results if row.get('quality_tier_v2') is not None]
        if v2_tiers:
            stats['v2_high'] = sum(1 for t in v2_tiers if t == 'High')
            stats['v2_medium'] = sum(1 for t in v2_tiers if t == 'Medium')
            stats['v2_low'] = sum(1 for t in v2_tiers if t == 'Low')
            stats['v2_upgrades'] = sum(1 for r in results if r.get('quality_tier') != r.get('quality_tier_v2') and r.get('quality_tier_v2') == 'High')
            stats['v2_downgrades'] = sum(1 for r in results if r.get('quality_tier') == 'High' and r.get('quality_tier_v2') != 'High')

    return stats


def print_summary(results: list[dict], include_interface: bool = False) -> None:
    """Print a human-readable summary of the batch analysis results.
    Args:
        results: List of per-complex result dictionaries.
        include_interface: Whether to include interface statistics.
    """
    stats = _aggregate_summary_statistics(results, include_interface)
    total = stats['total_complexes']

    print(f"\nDataset Summary:")
    print(f"  Total complexes: {total}")
    print(f"  Homodimers:      {stats['homodimer_count']} ({100 * stats['homodimer_count'] / total:.1f}%)")
    print(f"  Heterodimers:    {stats['heterodimer_count']} ({100 * stats['heterodimer_count'] / total:.1f}%)")

    print(f"\nQuality Distribution:")
    print(f"  High:   {stats['quality_high']} ({100 * stats['quality_high'] / total:.1f}%)"
          f" - ipTM≥{IPTM_HIGH_THRESHOLD} & pDockQ≥{PDOCKQ_HIGH_THRESHOLD}")
    print(f"  Medium: {stats['quality_medium']} ({100 * stats['quality_medium'] / total:.1f}%)"
          f" - ipTM≥{IPTM_MEDIUM_THRESHOLD} & pDockQ≥{PDOCKQ_MEDIUM_THRESHOLD}")
    print(f"  Low:    {stats['quality_low']} ({100 * stats['quality_low'] / total:.1f}%)")

    if stats['iptm_values']:
        vals = stats['iptm_values']
        print(f"\nipTM: mean={statistics.mean(vals):.4f}, "
              f"min={min(vals):.4f}, max={max(vals):.4f}")

    if stats['pdockq_values']:
        vals = stats['pdockq_values']
        print(f"pDockQ: mean={statistics.mean(vals):.4f}, "
              f"min={min(vals):.4f}, max={max(vals):.4f}")

    # pLDDT disorder summary
    if stats['below50_values']:
        poorly_predicted_count = sum(1 for val in stats['below50_values'] if val > SUBSTANTIAL_DISORDER_FRACTION)
        print(f"\npLDDT Disorder Analysis (from PDB b-factors):")
        print(f"  Mean fraction below 50 (poorly predicted): {statistics.mean(stats['below50_values']):.3f}")
        print(f"  Mean fraction below 70 (low confidence):   {statistics.mean(stats['below70_values']):.3f}")
        print(f"  Complexes with >{SUBSTANTIAL_DISORDER_FRACTION:.0%} residues below 50:     "
              f"{poorly_predicted_count} ({100 * poorly_predicted_count / total:.1f}%)")

    # pLDDT source tracking
    if stats['pdb_fallback_count'] > 0:
        print(f"\npLDDT Source:")
        print(f"  From PKL:            {stats['pkl_source_count']}")
        print(f"  From PDB (fallback): {stats['pdb_fallback_count']}")
        if stats['no_plddt_count'] > 0:
            print(f"  No pLDDT available:  {stats['no_plddt_count']}")

    # Interface analysis summary
    if not include_interface:
        return

    contact_counts = stats.get('contact_counts', [])
    if contact_counts:
        print(f"\nInterface Analysis:")
        print(f"  Mean contacts: {statistics.mean(contact_counts):.1f}")
        print(f"  Zero-contact complexes: "
              f"{sum(1 for c in contact_counts if c == 0)}")
        if stats['if_plddt_values']:
            print(f"  Mean interface pLDDT: {statistics.mean(stats['if_plddt_values']):.1f}")
        if stats['delta_values']:
            positive_delta = sum(1 for d in stats['delta_values'] if d > 0)
            print(f"  Interface > bulk (positive delta): "
                  f"{positive_delta} ({100 * positive_delta / len(stats['delta_values']):.1f}%)")

        # Flag summary
        if stats['all_flags']:
            print(f"\n  Interface Flags:")
            for flag, count in sorted(stats['all_flags'].items(), key=lambda x: -x[1]):
                print(f"    {flag}: {count} ({100 * count / total:.1f}%)")

    # PAE-specific summary
    if stats['confident_fractions']:
        print(f"\nInterface PAE:")
        print(f"  Mean confident contact fraction: {statistics.mean(stats['confident_fractions']):.3f}")
        high_conf = sum(1 for f in stats['confident_fractions'] if f > 0.5)
        print(f"  Complexes with >50% confident contacts: "
              f"{high_conf} ({100 * high_conf / len(stats['confident_fractions']):.1f}%)")

    # Composite interface confidence score
    if stats['composite_scores']:
        scores = stats['composite_scores']
        print(f"\nComposite Interface Confidence (Phase 4):")
        print(f"  Mean: {statistics.mean(scores):.3f}, "
              f"Median: {statistics.median(scores):.3f}, "
              f"Min: {min(scores):.3f}, Max: {max(scores):.3f}")

    # Quality tier v2 reclassification summary
    if 'v2_high' in stats:
        print(f"\nQuality Tier v2 (interface-aware reclassification):")
        print(f"  High:   {stats['v2_high']} ({100 * stats['v2_high'] / total:.1f}%)")
        print(f"  Medium: {stats['v2_medium']} ({100 * stats['v2_medium'] / total:.1f}%)")
        print(f"  Low:    {stats['v2_low']} ({100 * stats['v2_low'] / total:.1f}%)")
        print(f"  Reclassified: {stats['v2_upgrades']} upgraded to High, "
              f"{stats['v2_downgrades']} downgraded from High")


#-------------------CLI Entry Point-----------------------------------------

def build_argument_parser() -> argparse.ArgumentParser:
    """Create and return the argument parser for the batch processor."""
    parser = argparse.ArgumentParser(
        description="Batch process AlphaFold2 predictions - direct imports, no subprocesses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic (sequential, no checkpointing)
    python toolkit.py --dir "D:\\ProteinComplexes" --output results.csv
    python toolkit.py --dir "D:\\ProteinComplexes" --output results.csv --interface --pae

    # Full analysis with parallel workers and checkpointing
    python toolkit.py --dir "D:\\ProteinComplexes" --output results.csv --interface --pae -w 4 --checkpoint
    python toolkit.py --dir "D:\\ProteinComplexes" --output results.csv --interface --pae --export-interfaces interfaces.jsonl -w 4 --checkpoint

    # With enrichment (gene symbols, protein names, sequences)
    python toolkit.py --dir "D:\\ProteinComplexes" --output results.csv --interface --pae --enrich "D:\\protein-complexes-toolkit\\data\\ppi\\9606.protein.aliases.v12.0.txt"

    # With enrichment + database source tagging
    python toolkit.py --dir "D:\\ProteinComplexes" --output results.csv --interface --pae --enrich "D:\\protein-complexes-toolkit\\data\\ppi\\9606.protein.aliases.v12.0.txt" --databases "D:\\protein-complexes-toolkit\\data\\ppi"

    # Resume an interrupted run
    python toolkit.py --dir "D:\\ProteinComplexes" --output results.csv --interface --pae -w 4 --resume

    # Verbose (sequential only - verbose is suppressed with -w > 1)
    python toolkit.py --dir "D:\\ProteinComplexes" --output results.csv --interface --pae -v
        """,
    )

    parser.add_argument("--dir", required=True, help="Directory containing PDB/PKL files")
    parser.add_argument("--output", default="batch_results.csv", help="Output CSV file")
    parser.add_argument("--interface", action="store_true", help="Compute interface geometry and pLDDT features")
    parser.add_argument("--pae", action="store_true", help="Compute PAE-based interface features (requires --interface and PKL files)")
    parser.add_argument("--export-interfaces", metavar="PATH",
                        help="Export confident interface residues to a JSONL file "
                             "(one JSON record per line). Requires --interface --pae. "
                             "Only exports complexes with High or Medium v2 tier.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--workers", "-w", type=int, default=1,
                        help="Number of parallel workers (default: 1 = sequential). "
                             "Values >1 enable multiprocessing via ProcessPoolExecutor.")
    parser.add_argument("--checkpoint", action="store_true",
                        help="Enable periodic checkpointing (saves progress every "
                             f"{CHECKPOINT_INTERVAL} complexes to <output>{CHECKPOINT_SUFFIX})")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from a previous checkpoint. Implies --checkpoint. "
                             "Already-processed complexes are skipped.")
    parser.add_argument("--enrich", metavar="ALIASES_PATH",
                        help="Enrich output with gene symbols, protein names, and "
                             "cross-references using a STRING aliases file.")
    parser.add_argument("--databases", metavar="DATA_DIR",
                        help="Tag each complex with its database source(s) by checking "
                             "against STRING, BioGRID, HuRI, and HuMAP. Requires --enrich.")
    parser.add_argument("--string-min-score", type=int, default=700,
                        help="Minimum STRING confidence score for database source matching "
                             "(default: 700). Only used with --databases.")
    return parser


def main() -> None:
    """Run the batch processing pipeline."""
    parser = build_argument_parser()
    args = parser.parse_args()

    # Validate flags
    if args.pae and not args.interface:
        print("Note: --pae implies --interface, enabling interface analysis.", file=sys.stderr)
        args.interface = True

    if args.export_interfaces:
        if not args.pae:
            print("Note: --export-interfaces implies --interface --pae, enabling both.",
                  file=sys.stderr)
            args.interface = True
            args.pae = True

    if args.databases and not args.enrich:
        print("Error: --databases requires --enrich", file=sys.stderr)
        sys.exit(1)

    if args.resume:
        args.checkpoint = True

    if args.workers < 1:
        print("Error: --workers must be >= 1", file=sys.stderr)
        sys.exit(1)

    if args.workers > 1 and args.verbose:
        print("Note: Verbose per-complex output is suppressed in parallel mode.",
              file=sys.stderr)

    # Discover data files
    print(f"Scanning data directory: {args.dir}")
    complexes = find_paired_data_files(args.dir)
    print(f"Found {len(complexes)} unique complexes")

    if len(complexes) == 0:
        print("No PDB/PKL files found!")
        sys.exit(1)

    # Resume from checkpoint if requested
    resumed_results: list[dict] = []
    completed_names: set[str] = set()

    if args.resume:
        checkpoint_data = load_checkpoint(args.output)
        if checkpoint_data:
            completed_names = set(checkpoint_data.keys())
            resumed_results = list(checkpoint_data.values())
            print(f"Checkpoint loaded: {len(resumed_results)} complexes already complete")
        else:
            print("No checkpoint found, starting from scratch")

    # Filter out already-completed complexes
    sorted_complexes = [
        (name, paths)
        for name, paths in sorted(complexes.items())
        if name not in completed_names
    ]

    if not sorted_complexes and resumed_results:
        print("All complexes already processed - writing final output.")
        results = list(resumed_results)
        results.sort(key=lambda r: r.get('complex_name', ''))
    else:
        # Process complexes (sequential or parallel)
        results = run_batch_parallel(
            sorted_complexes,
            run_interface=args.interface,
            run_interface_pae=args.pae,
            export_interfaces=bool(args.export_interfaces),
            verbose=args.verbose,
            workers=args.workers,
            output_path=args.output,
            enable_checkpoint=args.checkpoint,
            resumed_results=resumed_results,
        )

    # Enrichment (gene symbols, database sources)
    include_enrichment = False
    if args.enrich:
        enrich_start = time.time()
        from id_mapper import IDMapper, build_uniprot_lookup
        print(f"Loading ID mapper from: {args.enrich}", file=sys.stderr)
        mapper = IDMapper(args.enrich, verbose=True)
        lookup = build_uniprot_lookup(mapper)
        print(f"  Lookup table: {len(lookup):,} UniProt entries", file=sys.stderr)

        db_pair_sets = None
        db_evidence = None

        if args.databases:
            from database_loaders import load_all_databases
            from id_mapper import map_dataframe_to_uniprot
            from overlap_analysis import extract_pair_set

            print(f"Loading databases from: {args.databases}", file=sys.stderr)
            dbs = load_all_databases(
                args.databases,
                string_min_score=args.string_min_score,
                verbose=True,
            )

            # Map STRING and HuRI to UniProt for pair matching
            print("Mapping STRING IDs to UniProt...", file=sys.stderr)
            dbs['STRING'] = map_dataframe_to_uniprot(dbs['STRING'], mapper, verbose=True)
            print("Mapping HuRI IDs to UniProt...", file=sys.stderr)
            dbs['HuRI'] = map_dataframe_to_uniprot(dbs['HuRI'], mapper, verbose=True)

            # Build pair sets for each database
            print("Building pair sets...", file=sys.stderr)
            db_pair_sets = {}
            for name, df in dbs.items():
                if 'uniprot_a' in df.columns:
                    db_pair_sets[name] = extract_pair_set(
                        df, col_a='uniprot_a', col_b='uniprot_b'
                    )
                else:
                    db_pair_sets[name] = extract_pair_set(df)
                print(f"  {name}: {len(db_pair_sets[name]):,} unique pairs",
                      file=sys.stderr)

            # Pre-compute evidence types per database (avoids scanning millions of rows per complex inside enrich_results)
            db_evidence = {}
            for name, df in dbs.items():
                if 'evidence_type' in df.columns:
                    db_evidence[name] = set(
                        str(e) for e in df['evidence_type'].dropna().unique()
                    )
                else:
                    db_evidence[name] = set()

            total_pairs = sum(len(s) for s in db_pair_sets.values())
            print(f"  Total: {total_pairs:,} pairs across "
                  f"{len(db_pair_sets)} databases", file=sys.stderr)

        print(f"Enriching {len(results):,} complexes...", file=sys.stderr)
        enrich_results(results, lookup, db_pair_sets, db_evidence)
        include_enrichment = True
        enrich_elapsed = time.time() - enrich_start
        print(f"Enrichment complete: {len(results)} complexes annotated "
              f"in {enrich_elapsed:.1f}s", file=sys.stderr)

    # Write CSV output
    print(f"Writing CSV to {args.output}...", file=sys.stderr)
    write_results_csv(
        results, args.output,
        include_interface=args.interface,
        include_pae=args.pae,
        include_enrichment=include_enrichment,
    )

    print(f"\n{'=' * 60}")
    print(f"Results saved to: {args.output}")

    # Write JSONL interface export if requested
    if args.export_interfaces:
        exported_count = write_interface_exports(results, args.export_interfaces)
        print(f"Interface export: {args.export_interfaces} ({exported_count} complexes)")

    print(f"{'=' * 60}")

    # Clean up checkpoint on successful completion
    if args.checkpoint:
        remove_checkpoint(args.output)
        print("Checkpoint cleared (run completed successfully)")

    print_summary(results, include_interface=args.interface)

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()