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
    - Base output is 28 columns (incl. species_a/b/status) and enriched output is up to 124 columns with all features

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

import gc
import gzip
import math
import os
import pickle
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
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import dataclass
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
from file_io import decompressed_pdb_view, open_text_maybe_compressed
from read_af2_nojax import load_pkl_without_jax, extract_metrics
from pdockq import (
    read_pdb_with_chain_info_New as read_pdb_with_chain_info,
    calc_pdockq_and_contacts_New as calc_pdockq_and_contacts,
    compute_pae_chain_offsets_New as compute_pae_chain_offsets,
    find_best_chain_pair_New as find_best_chain_pair,
    compute_all_chain_pairs,
    calc_pdockq_whole_complex,
)
from interface_analysis import (
    analyse_interface_from_contact_result,
    compute_extended_flags,
    build_interface_export_record,
    compute_all_pair_metrics,
    aggregate_pair_metrics,
    serialise_pair_metrics,
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

# STRING get_string_ids batch size for the missing-protein pre-resolve in
# enrich_results. 200 stays well under the API's documented per-request
# identifier ceiling and keeps each request body small for retry safety.
ENRICH_API_BATCH_SIZE = 200

# Schema version marker - bump when column semantics change.
# Readers must treat a missing schema_version column as "legacy" (pre-refactor).
SCHEMA_VERSION = "multimer_v1"

# Diagnostic reasons for the `partial_reason` column.
# Empty string is the "fully calibrated, nothing wrong" sentinel — chosen for
# CSV friendliness over None (csv.DictWriter renders None as empty anyway, but
# downstream pandas reads NaN; a literal "" round-trips as "" without coercion).
#
# The vocabulary is a controlled single-column failure taxonomy. Any non-empty
# value excludes the row from recoverable / calibrated analyses. Values are
# stamped via `_stamp_partial_reason` (priority-aware) so that the dominant
# failure reason wins when several apply to the same row.
PARTIAL_REASON_NONE = ""

# PDB / structure input
PARTIAL_REASON_PDB_IO_ERROR             = "pdb_io_error"
PARTIAL_REASON_PDB_DECOMPRESSION_ERROR  = "pdb_decompression_error"
PARTIAL_REASON_PDB_PARSE_ERROR          = "pdb_parse_error"
PARTIAL_REASON_PDB_NO_CHAINS            = "pdb_no_chains"

# PKL / AF2 confidence input
PARTIAL_REASON_PKL_IO_ERROR             = "pkl_io_error"
PARTIAL_REASON_PKL_DECOMPRESSION_ERROR  = "pkl_decompression_error"
PARTIAL_REASON_PKL_UNPICKLE_ERROR       = "pkl_unpickle_error"
PARTIAL_REASON_PKL_LOADED_MISSING_IPTM  = "pkl_loaded_missing_iptm"
PARTIAL_REASON_PKL_LOADED_MISSING_PAE   = "pkl_loaded_missing_pae"

# Interface / composite computation
PARTIAL_REASON_ZERO_CONTACTS            = "no_positive_interface_contacts"
PARTIAL_REASON_MISSING_COMPOSITE        = "missing_required_composite_inputs"

# Sentinel / catch-all
PARTIAL_REASON_WORKER_EXCEPTION         = "worker_exception"

# Backward-compat aliases (legacy fallbacks for genuinely unclassified Exception
# cases inside _extract_pkl_metrics / _compute_pdockq_and_chain_info).
PARTIAL_REASON_UNREADABLE_PDB           = "unreadable_pdb_or_structure_input"
PARTIAL_REASON_UNREADABLE_PKL           = "missing_pkl_or_pkl_unreadable"

# Pre-existing, untouched
PARTIAL_REASON_INCOMPLETE               = "incomplete_input"


# Priority map for `_stamp_partial_reason`. Higher number wins.
# Existing precedence (PDB > PKL > zero contacts > composite-missing) is
# preserved by the relative ordering.
PARTIAL_REASON_PRIORITY = {
    PARTIAL_REASON_WORKER_EXCEPTION:         100,
    PARTIAL_REASON_PDB_DECOMPRESSION_ERROR:   90,
    PARTIAL_REASON_PDB_IO_ERROR:              85,
    PARTIAL_REASON_PDB_PARSE_ERROR:           80,
    PARTIAL_REASON_PDB_NO_CHAINS:             75,
    PARTIAL_REASON_UNREADABLE_PDB:            72,  # legacy fallback
    PARTIAL_REASON_PKL_DECOMPRESSION_ERROR:   70,
    PARTIAL_REASON_PKL_IO_ERROR:              65,
    PARTIAL_REASON_PKL_UNPICKLE_ERROR:        60,
    PARTIAL_REASON_UNREADABLE_PKL:            58,  # legacy fallback
    PARTIAL_REASON_ZERO_CONTACTS:             40,
    PARTIAL_REASON_PKL_LOADED_MISSING_IPTM:   35,
    PARTIAL_REASON_PKL_LOADED_MISSING_PAE:    30,
    PARTIAL_REASON_MISSING_COMPOSITE:         20,
    PARTIAL_REASON_INCOMPLETE:                10,
    PARTIAL_REASON_NONE:                       0,
}


def _stamp_partial_reason(row: dict, reason: Optional[str]) -> None:
    """Stamp a non-empty partial reason if it strictly outranks the current one.

    Production code should call this helper rather than assigning
    ``row['partial_reason']`` directly. The helper:
      * ignores empty / None reasons,
      * preserves the existing precedence (PDB > PKL > zero contacts > composite),
      * uses strict ``>`` so equal-priority overwrites do not churn.
    """
    if not reason:
        return
    current = row.get("partial_reason") or PARTIAL_REASON_NONE
    current_priority = PARTIAL_REASON_PRIORITY.get(current, 0)
    new_priority = PARTIAL_REASON_PRIORITY.get(reason, 0)
    if current == PARTIAL_REASON_NONE or new_priority > current_priority:
        row["partial_reason"] = reason


def _has_value(value) -> bool:
    """Return True iff `value` is a real, present datum.

    Treats None, "", and NaN (Python float or numpy float) as missing. Used by
    the calibration predicate so a downstream NaN-from-CSV can't accidentally
    mark a row as calibrated.
    """
    if value is None:
        return False
    if value == "":
        return False
    try:
        return not math.isnan(float(value))
    except (TypeError, ValueError, OverflowError):
        # Non-numeric truthy values (e.g. populated strings, dicts) count as present.
        return True


def _coerce_bool(value) -> bool:
    """Robust truthy-cast that handles CSV-loaded string sentinels.

    bool("False") is True in Python; this helper treats "False"/"0"/""/None as
    False so the screening function and the calibration-flag predicate cannot
    be silently fooled by a string-typed CSV column.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}

# CSV base columns that are always present
CSV_FIELDNAMES_BASE = [
    'schema_version',
    'complex_name', 'protein_a', 'protein_b', 'complex_type',
    'n_chains', 'best_chain_pair',
    'iptm', 'ptm', 'ranking_confidence',
    'plddt_mean', 'plddt_median', 'plddt_min', 'plddt_max',
    'plddt_below50_fraction', 'plddt_below70_fraction',
    'num_residues', 'pae_mean',
    'pdockq', 'ppv', 'quality_tier',
    'has_pdb', 'has_pkl', 'geometry_available', 'plddt_source',
    'species', 'structure_source',
    'species_a', 'species_b', 'species_status',
    # Multimer-safe identity columns (Phase 2 - multimer_v1 schema)
    'stoichiometry', 'is_homomeric',
    'unique_accessions', 'chain_ids', 'accession_chain_map',
    'tier_scope',
    'filename_n_chains', 'pdb_n_chains', 'chain_count_consistency',
    'complex_identity_json',
    # Diagnostic for HPC failure modes (empty string when the row is normal /
    # fully calibrated). See PARTIAL_REASON_* constants for the value vocabulary.
    'partial_reason',
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

# Interface geometry columns added when --interface is used.
#
# Best-pair vs whole-complex (multimer_v1 schema):
#   For N=2, best-pair values and whole-complex aggregates are metric-identical.
#   For N>2, `n_interface_contacts`, `interface_plddt_*`, `interface_symmetry`,
#   `interface_fraction_*` etc. describe the BEST pair (max contacts). The
#   `pair_metrics` JSON + `*_mean` / `*_min` / `pdockq_whole_complex` /
#   `contact_count_total` aggregates describe the full complex and are required
#   to avoid the pre-refactor order-statistic bias of `max(all pairs)`.
CSV_FIELDNAMES_INTERFACE = [
    'n_interface_contacts', 'n_interface_residues_a', 'n_interface_residues_b',
    'interface_residues_a', 'interface_residues_b',
    'interface_fraction_a', 'interface_fraction_b', 'interface_symmetry',
    'contacts_per_interface_residue',
    'interface_plddt_a', 'interface_plddt_b', 'interface_plddt_combined',
    'bulk_plddt_combined', 'interface_vs_bulk_delta',
    'interface_plddt_high_fraction',
    # Multimer_v1: per-pair snapshot + non-PAE aggregates
    'pair_metrics',
    'pdockq_mean', 'pdockq_min', 'pdockq_whole_complex',
    'contact_count_total',
    'interface_plddt_mean', 'symmetry_mean', 'symmetry_min',
]

# Interface PAE columns added when --interface --pae is used
# Two confident-contact fractions are exported:
#   - pae_confident_contact_fraction: PAE < 5A only (historical metric; paradox thresholds)
#   - strict_confident_contact_fraction: PAE < 5A AND both pLDDT >= 70 (consumed by composite)
# Directional diagnostics (*_forward_mean, *_reverse_mean, *_directional_delta_*) are
# exposed for the methods section - they quantify how often pae[i,j] disagrees with
# pae[j,i] across the dataset (not part of the composite score).
CSV_FIELDNAMES_INTERFACE_PAE = [
    'interface_pae_mean', 'interface_pae_median',
    'n_pae_confident_contacts', 'pae_confident_contact_fraction',
    'n_strict_confident_contacts', 'strict_confident_contact_fraction',
    'cross_chain_pae_mean',
    'interface_pae_forward_mean', 'interface_pae_reverse_mean',
    'interface_pae_directional_delta_mean', 'interface_pae_directional_delta_max',
    'n_confident_residues_a', 'n_confident_residues_b',
    'interface_confidence_score',
    'quality_tier_v2',
    # Composite screening interpretation (Decision #43): strong / moderate / weak
    # / unavailable, derived from interface_confidence_score + calibration flag +
    # headline metrics. Decoupled from quality_tier_v2 - this is a prioritisation
    # label, not a tier and not a probability.
    'composite_screen_status',
    # Multimer_v1: contact-weighted PAE aggregates across all pairs
    'pae_confident_fraction_mean', 'strict_confident_fraction_mean',
    # Multimer_v1 scope flag. True only for N=2 - the existing composite uses
    # best-pair inputs (strict fraction, symmetry, density) which are order-
    # statistic biased for N>2. A calibrated multimer composite is deferred
    # (optional Phase 8) - downstream code should treat N>2 composites as
    # uncalibrated and filter to tier_scope == "dimer_validated" for any
    # headline confidence claim.
    'composite_is_calibrated',
]

# Flags column
CSV_FIELDNAMES_FLAGS = ['interface_flags']


# Central registry of every fieldname constant in the toolkit's own CSV output.
# `test_schema_integrity.py` guards this mapping: it fails CI if any constant drifts
# (new columns appended without a registry update, or duplicates across groups).
# External modules (protein_clustering, variant_mapper, stability_scorer,
# protvar_client, disease_annotations, pathway_network) own their own CSV_FIELDNAMES_*
# constants; those are validated by their individual test modules.
CSV_FIELDNAME_REGISTRY: dict[str, list[str]] = {
    'base': CSV_FIELDNAMES_BASE,
    'enrichment': CSV_FIELDNAMES_ENRICHMENT,
    'interface': CSV_FIELDNAMES_INTERFACE,
    'interface_pae': CSV_FIELDNAMES_INTERFACE_PAE,
    'flags': CSV_FIELDNAMES_FLAGS,
}


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
        with open_text_maybe_compressed(pdb_path) as pdb_file:
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

# AlphaFold2 model suffixes - handles both naming conventions:
#   PKL files use:  _result_model_X_multimer_v3_pred_Y
#   PDB files use:  _relaxed_model_X_multimer_v3_pred_Y
AF2_SUFFIX_PATTERN = re.compile(
    r'_(result|relaxed)_model_\d+_multimer_v\d+_pred_\d+$'
)


@dataclass(frozen=True)
class ComplexIdentity:
    """Structural identity of a complex derived from its filename (and optionally PDB).

    Replaces the lossy `parse_complex_name` contract: every chain token is preserved,
    stoichiometry is canonicalised, and filename/PDB chain counts are cross-checked.
    """
    complex_id: str
    accessions: tuple[str, ...]
    unique_accessions: tuple[str, ...]
    n_chains: int
    stoichiometry: str
    is_homomeric: bool
    accession_chain_map: dict
    chain_ids: tuple[str, ...]
    legacy_complex_type: str
    filename_n_chains: int
    pdb_n_chains: Optional[int]
    chain_count_consistency: str


def _strip_filename_suffixes(filename: str) -> str:
    """Strip file extensions and AF2 model suffixes to leave just the accession tokens."""
    clean_name = filename
    for ext in ('.pdb', '.pkl'):
        if clean_name.endswith(ext):
            clean_name = clean_name[:-len(ext)]
    clean_name = AF2_SUFFIX_PATTERN.sub('', clean_name)
    for suffix in ('.results', '_results'):
        if clean_name.endswith(suffix):
            clean_name = clean_name[:-len(suffix)]
    return clean_name


def _chain_label(index: int) -> str:
    """Return chain label for index 0 -> 'A', 25 -> 'Z', 26 -> 'AA', etc."""
    if index < 26:
        return chr(ord('A') + index)
    # Extended labels for rare N>26 complexes
    first = chr(ord('A') + (index // 26) - 1)
    second = chr(ord('A') + (index % 26))
    return first + second


def _canonical_stoichiometry(accessions: tuple[str, ...]) -> str:
    """Canonical stoichiometry label from accession counts.

    Sort unique accessions by count descending, then alphabetical; assign A, B, C,...
    AFTER sorting so token-order variants (A_A_B vs B_A_A) yield the same label.
    """
    counts = Counter(accessions)
    sorted_unique = sorted(counts.keys(), key=lambda acc: (-counts[acc], acc))
    parts = []
    for i, acc in enumerate(sorted_unique):
        label = _chain_label(i)
        count = counts[acc]
        parts.append(label if count == 1 else f"{label}{count}")
    return ''.join(parts)


def parse_complex_identity(
    filename: str,
    pdb_n_chains: Optional[int] = None,
) -> ComplexIdentity:
    """Parse a filename into a full ComplexIdentity.

    Args:
        filename: Filename, optionally with .pdb/.pkl and AF2 model suffixes.
        pdb_n_chains: Structural chain count if a PDB is available. When provided
            and disagreeing with the filename token count, the row is flagged as
            `chain_count_consistency == "mismatch"` and PDB is authoritative for
            `n_chains` (stoichiometry remains filename-derived).
    """
    complex_id = _strip_filename_suffixes(filename)
    accessions = tuple(complex_id.split('_'))
    filename_n_chains = len(accessions)

    # unique_accessions preserves first-seen order (not canonical-sorted order)
    seen: set[str] = set()
    unique: list[str] = []
    for acc in accessions:
        if acc not in seen:
            seen.add(acc)
            unique.append(acc)
    unique_accessions = tuple(unique)

    stoichiometry = _canonical_stoichiometry(accessions)

    # Chain IDs follow raw token order: chain A = first token, B = second, ...
    chain_ids = tuple(_chain_label(i) for i in range(filename_n_chains))
    accession_chain_map = {ch: accessions[i] for i, ch in enumerate(chain_ids)}

    is_homomeric = len(unique_accessions) == 1

    # Chain-count consistency
    if pdb_n_chains is None:
        n_chains = filename_n_chains
        chain_count_consistency = "filename_only"
    elif filename_n_chains == pdb_n_chains:
        n_chains = filename_n_chains
        chain_count_consistency = "match"
    else:
        n_chains = pdb_n_chains
        chain_count_consistency = "mismatch"

    # legacy_complex_type is the coarse category retained for backward compatibility.
    # New code should prefer `stoichiometry` and `tier_scope`.
    if filename_n_chains == 2:
        legacy_complex_type = "Homodimer" if is_homomeric else "Heterodimer"
    else:
        legacy_complex_type = "Multi-chain"

    return ComplexIdentity(
        complex_id=complex_id,
        accessions=accessions,
        unique_accessions=unique_accessions,
        n_chains=n_chains,
        stoichiometry=stoichiometry,
        is_homomeric=is_homomeric,
        accession_chain_map=accession_chain_map,
        chain_ids=chain_ids,
        legacy_complex_type=legacy_complex_type,
        filename_n_chains=filename_n_chains,
        pdb_n_chains=pdb_n_chains,
        chain_count_consistency=chain_count_consistency,
    )


def parse_complex_name(filename: str) -> tuple[str, str, str, str]:
    """Parse protein IDs and coarse complex type from an AF2 output filename.

    Thin backward-compat wrapper over `parse_complex_identity`. For N>2 complexes
    this returns the first two accessions and `legacy_complex_type == "Multi-chain"`;
    callers that need full stoichiometry should use `parse_complex_identity` directly.

    Returns:
        (complex_name, protein_a_id, protein_b_id, complex_type)
    """
    ident = parse_complex_identity(filename)
    if ident.filename_n_chains >= 2:
        protein_a_id = ident.accessions[0]
        protein_b_id = ident.accessions[1]
    else:
        protein_a_id = ident.complex_id
        protein_b_id = ident.complex_id
    return ident.complex_id, protein_a_id, protein_b_id, ident.legacy_complex_type


_LOOSE_SUFFIXES = (
    '.pdb', '.pdb.bz2', '.pdb.gz',
    '.pkl', '.pkl.bz2', '.pkl.gz',
    '.results.pkl', '.results.pkl.bz2',
)


def _has_loose_files(directory: Path) -> bool:
    for f in directory.iterdir():
        if not f.is_file():
            continue
        name = f.name
        for suffix in _LOOSE_SUFFIXES:
            if name.endswith(suffix):
                return True
    return False


def find_paired_data_files(
    directory: str,
    *,
    purpose: str = "baseline",
    return_audit: bool = False,
):
    """Find paired PDB/PKL files in a loose, flat-dir, or sharded layout.

    Three layouts supported:

    - **Loose**: files directly in root, e.g. ``Test_Data/X_Y.pdb`` (legacy
      local layout). Complex names parsed from filenames.
    - **Flat-dir**: each child of root is a complex directory containing
      ``{name}.pdb`` and ``{name}.pkl`` (or compressed variants).
    - **Sharded**: ``{root}/{shard}/{complex_name}/...`` (HPC layout).

    Hierarchical layouts (flat-dir + sharded) delegate to
    ``complex_resolver.find_complexes`` which also writes a forensic
    manifest. The loose layout is handled in-place to preserve the original
    Test_Data semantics.

    Parameters
    ----------
    purpose
        Forwarded to the resolver to drive the auto run-id suffix
        (``"baseline"`` or ``"incremental"``). Ignored for the loose layout.
    return_audit
        When ``False`` (default), returns the legacy bare dict for backward
        compatibility. When ``True``, returns a
        :class:`complex_resolver.DiscoveryResult` carrying the run id and
        run-dir paths the resolver just wrote. The loose layout returns a
        synthetic ``DiscoveryResult`` with ``run_id=""`` and ``run_dir=None``.

    Returns
    -------
    dict | DiscoveryResult
        See ``return_audit``.
    """
    data_directory = Path(directory)

    if _has_loose_files(data_directory):
        complexes: dict[str, dict[str, Path]] = defaultdict(dict)
        for file_path in data_directory.iterdir():
            if not file_path.is_file():
                continue
            # parse_complex_name doesn't know about .bz2/.gz; strip the
            # compression suffix before grouping so X_Y.pdb.bz2 and
            # X_Y.pkl.bz2 share the same complex_name key.
            name_for_parsing = file_path.name
            for ext in ('.bz2', '.gz'):
                if name_for_parsing.lower().endswith(ext):
                    name_for_parsing = name_for_parsing[: -len(ext)]
                    break
            complex_name, _, _, _ = parse_complex_name(name_for_parsing)
            name_lower = file_path.name.lower()
            if name_lower.endswith('.pdb') or name_lower.endswith('.pdb.bz2') \
                    or name_lower.endswith('.pdb.gz'):
                complexes[complex_name].setdefault('pdb', file_path)
            elif name_lower.endswith('.pkl') or name_lower.endswith('.pkl.bz2') \
                    or name_lower.endswith('.pkl.gz'):
                complexes[complex_name].setdefault('pkl', file_path)
        loose_pairs = {
            name: paths
            for name, paths in complexes.items()
            if 'pdb' in paths and 'pkl' in paths
        }
        if return_audit:
            from complex_resolver import DiscoveryResult
            return DiscoveryResult(
                complexes=loose_pairs,
                run_id="",
                run_dir=None,
                manifest_path=None,
            )
        return loose_pairs

    from complex_resolver import find_complexes
    return find_complexes(
        root=data_directory,
        purpose=purpose,
        return_audit=return_audit,
    )

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

# Interface confidence thresholds for tier reclassification.
# The constants now reflect Decision #38 adoption after the 9K / 41K / 60K
# recalibration comparison. UPGRADE_LOW is held from the pre-recalibration set;
# UPGRADE_MEDIUM and DOWNGRADE_HIGH are the post-recalibration values.
UPGRADE_LOW_THRESHOLD = 0.64     # Low    -> High when composite score >= 0.64 (Decision #38 adopted; held from pre-recalibration)
UPGRADE_MEDIUM_THRESHOLD = 0.85  # Medium -> High when composite score >= 0.85 (Decision #38 adopted; was 0.80)
DOWNGRADE_HIGH_THRESHOLD = 0.63  # High   -> Medium when composite score <= 0.63 (Decision #38 adopted; was 0.65)

# Composite screening thresholds - interpretation of interface_confidence_score
# as a prioritisation signal, NOT a quality tier and NOT a probability.
# A row with composite >= 0.85 is a "strong" screening candidate; 0.63-0.85 is
# "moderate"; below 0.63 is "weak". Decoupled from quality_tier_v2 so the two
# layers can be reasoned about separately (Roadmap Decisions #38, #43).
COMPOSITE_SCREEN_STRONG_THRESHOLD = 0.85
COMPOSITE_SCREEN_MODERATE_THRESHOLD = 0.63

COMPOSITE_SCREEN_STATUS_STRONG = "strong_screen_candidate"
COMPOSITE_SCREEN_STATUS_MODERATE = "moderate_screen_candidate"
COMPOSITE_SCREEN_STATUS_WEAK = "weak_screen_candidate"
COMPOSITE_SCREEN_STATUS_UNAVAILABLE = "unavailable"

def classify_prediction_quality_v2(iptm_score: Optional[float], pdockq_score: Optional[float], interface_confidence: Optional[float] = None) -> str:
    """Primary fused V2 quality classification.

    quality_tier_v2 is a composite-informed reclassification of the V1
    quality_tier. V1 (ipTM + pDockQ) remains the global-confidence prior;
    interface_confidence_score (composite) acts as interface-specific
    evidence that can either rescue a V1 Low row, promote a V1 Medium row,
    or downgrade a V1 High row.

    Transitions (`UPGRADE_LOW_THRESHOLD = 0.64`,
    `UPGRADE_MEDIUM_THRESHOLD = 0.85`, `DOWNGRADE_HIGH_THRESHOLD = 0.63`):
      - V1 Low + composite < 0.64                   -> Low
      - V1 Low + 0.64 <= composite < 0.85           -> Medium  (rescue)
      - V1 Low + composite >= 0.85                  -> High    (strong-composite rescue)
      - V1 Medium + composite < 0.85                -> Medium
      - V1 Medium + composite >= 0.85               -> High    (promote)
      - V1 High + composite <= 0.63                 -> Medium  (downgrade)
      - V1 High + composite > 0.63                  -> High

    V1 Medium is never downgraded to Low under this policy: a Medium V1 row
    has already passed both medium global-confidence gates, so weak
    composite means "not enough interface evidence to promote", not
    "interface evidence overrides global".

    The rationale for the rescue-to-Medium band is to remove the rhetorical
    contradiction in the earlier V2 policy: a V1 Low row with moderate
    composite (0.64-0.85) was promoted directly to V2 High while the
    parallel composite_screen_status called it only a moderate screen
    candidate. Under primary fused, V1 Low needs strong composite evidence
    (>= 0.85) to reach High.

    Missing or non-finite composite (None, NaN, +/-inf, unparseable)
    preserves the V1 tier unchanged. If classify_prediction_quality returns
    anything other than 'Low'/'Medium'/'High', that value is also returned
    unchanged - the current V1 always returns one of the three tiers in
    practice, so the fallthrough is defensive only.

    Args:
        iptm_score: Interface pTM score (or None if unavailable).
        pdockq_score: pDockQ docking quality score (or None if unavailable).
        interface_confidence: Composite interface confidence score from
            compute_interface_confidence() [0.0-1.0] or None if unavailable.

    Returns:
        Quality tier string: 'High', 'Medium', or 'Low' under normal
        operation; the upstream V1 tier string verbatim if V1 returns an
        unrecognised sentinel.
    """
    base_tier = classify_prediction_quality(iptm_score, pdockq_score)

    try:
        score = float(interface_confidence)
    except (TypeError, ValueError):
        return base_tier
    if not math.isfinite(score):
        return base_tier

    if base_tier == 'Low':
        if score >= UPGRADE_MEDIUM_THRESHOLD:
            return 'High'
        if score >= UPGRADE_LOW_THRESHOLD:
            return 'Medium'
        return 'Low'
    if base_tier == 'Medium':
        if score >= UPGRADE_MEDIUM_THRESHOLD:
            return 'High'
        return 'Medium'
    if base_tier == 'High':
        if score <= DOWNGRADE_HIGH_THRESHOLD:
            return 'Medium'
        return 'High'
    return base_tier


def classify_composite_screen_status(
    *,
    composite_is_calibrated,
    interface_confidence_score,
    iptm,
    pdockq,
) -> str:
    """Screening-status label for a row's composite interface confidence.

    This is a prioritisation / screening label, NOT a quality tier and NOT a
    probability. Returns "unavailable" when the row is outside the calibrated
    dimer scope, when the composite is unavailable, or when either headline
    metric required for apples-to-apples interpretation (ipTM, pDockQ) is
    missing. Accepts both Python booleans and CSV-loaded string sentinels for
    composite_is_calibrated via _coerce_bool, so the same function is safe to
    call from the toolkit in-process row path AND from the backfill / audit
    CSV-row paths.
    """
    if not _coerce_bool(composite_is_calibrated):
        return COMPOSITE_SCREEN_STATUS_UNAVAILABLE
    if not _has_value(interface_confidence_score):
        return COMPOSITE_SCREEN_STATUS_UNAVAILABLE
    if not _has_value(iptm) or not _has_value(pdockq):
        return COMPOSITE_SCREEN_STATUS_UNAVAILABLE
    score = float(interface_confidence_score)
    if score >= COMPOSITE_SCREEN_STRONG_THRESHOLD:
        return COMPOSITE_SCREEN_STATUS_STRONG
    if score >= COMPOSITE_SCREEN_MODERATE_THRESHOLD:
        return COMPOSITE_SCREEN_STATUS_MODERATE
    return COMPOSITE_SCREEN_STATUS_WEAK


#------------------------------------------------------Core Processing---------------------------------------------------------------------------------

def _extract_pkl_metrics(file_paths: dict[str, Path], row: dict, *, run_interface_pae: bool, verbose: bool) -> tuple[Optional[np.ndarray], Optional[str]]:
    """Extract ipTM, pTM, pLDDT metrics from a PKL file and optionally retain the PAE matrix.
    Args:
        file_paths: Dict with optional 'pdb' and 'pkl' Path entries.
        row: Result dict to update in-place with PKL metrics.
        run_interface_pae: Whether to retain the PAE matrix for downstream interface analysis.
        verbose: Whether to print per-step progress.
    Returns:
        Tuple of (pae_matrix or None, partial_reason).
        ``partial_reason`` is ``None`` on full success. On failure or
        post-load incompleteness it is the granular ``PARTIAL_REASON_*``
        constant the caller should stamp via ``_stamp_partial_reason``.
        Distinguishes "loader raised" from "PKL absent" from "PKL readable
        but missing required fields" using stdlib exception types and the
        compressed-file suffix.
    """
    pae_matrix = None
    if 'pkl' not in file_paths:
        return pae_matrix, None

    pkl_path = file_paths['pkl']
    suffixes = Path(pkl_path).suffixes
    is_compressed = ".bz2" in suffixes or ".gz" in suffixes

    try:
        prediction_result = load_pkl_without_jax(pkl_path)
        pkl_metrics = extract_metrics(prediction_result)
        row.update(pkl_metrics)
        row['plddt_source'] = 'pkl'

        # Keep PAE matrix in memory for interface analysis - discard after use
        if run_interface_pae and 'predicted_aligned_error' in prediction_result:
            pae_matrix = np.asarray(prediction_result['predicted_aligned_error'])
        if verbose:
            print(f"  PKL -> ipTM={pkl_metrics.get('iptm', 'N/A')}")

        # Post-load completeness diagnostics. Distinguish between "loader raised"
        # (handled below) and "loader returned but the row would be uncalibratable"
        # because a required field is absent. Pin the cause at the source rather
        # than letting `_finalise_calibration_flag` see only an opaque "composite
        # missing".
        if not _has_value(pkl_metrics.get('iptm')):
            return pae_matrix, PARTIAL_REASON_PKL_LOADED_MISSING_IPTM
        if run_interface_pae and pae_matrix is None:
            return pae_matrix, PARTIAL_REASON_PKL_LOADED_MISSING_PAE
        return pae_matrix, None

    except PermissionError as error:
        print(f"  Warning: PKL permission error for {pkl_path}: {error}", file=sys.stderr)
        return pae_matrix, PARTIAL_REASON_PKL_IO_ERROR
    except (EOFError, pickle.UnpicklingError) as error:
        # EOFError on a compressed file usually indicates truncated decompression;
        # on an uncompressed file it indicates truncated pickle. Disambiguate by suffix.
        print(f"  Warning: PKL truncated/unpickle error for {pkl_path}: {error}", file=sys.stderr)
        return pae_matrix, (PARTIAL_REASON_PKL_DECOMPRESSION_ERROR
                            if is_compressed
                            else PARTIAL_REASON_PKL_UNPICKLE_ERROR)
    except gzip.BadGzipFile as error:
        print(f"  Warning: PKL gzip decompression failed for {pkl_path}: {error}", file=sys.stderr)
        return pae_matrix, PARTIAL_REASON_PKL_DECOMPRESSION_ERROR
    except OSError as error:
        # bz2 corrupt streams surface as OSError (`bz2.BZ2File` is a class, not
        # an exception). Disambiguate by suffix.
        print(f"  Warning: PKL OSError for {pkl_path}: {error}", file=sys.stderr)
        return pae_matrix, (PARTIAL_REASON_PKL_DECOMPRESSION_ERROR
                            if is_compressed
                            else PARTIAL_REASON_PKL_IO_ERROR)
    except Exception as error:
        # Genuinely unclassified — fall back to the legacy alias so downstream
        # callers still see a non-empty reason.
        print(f"  Warning: PKL extraction failed for {pkl_path}: {error}", file=sys.stderr)
        return pae_matrix, PARTIAL_REASON_UNREADABLE_PKL

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
) -> tuple[Optional[object], Optional[object], Optional[tuple], Optional[tuple], Optional[list], Optional[dict], Optional[str]]:
    """Read PDB chain structure, find the best interacting chain pair, and compute pDockQ.
    Also pre-computes PAE chain offsets and CB-to-CA maps needed for downstream interface analysis.
    Args:
        file_paths: Dict with optional 'pdb' and 'pkl' Path entries.
        row: Result dict to update in-place with pDockQ, chain pair, and sequence data.
        pae_matrix: PAE matrix from PKL or None if unavailable.
        run_interface_pae: Whether to compute PAE offsets and CB-to-CA maps.
        verbose: Whether to print per-step progress.
    Returns:
        Tuple of (contact_result, chain_info, pae_chain_offsets, cb_to_ca_maps,
        pair_results, all_chain_offsets, partial_reason). Any of the first six
        elements may be ``None`` if the corresponding step was skipped or failed.
        ``partial_reason`` is the granular PARTIAL_REASON_* constant the caller
        should stamp on classification failure (or ``None`` on success / when
        no PDB was supplied).
    """
    contact_result = None
    chain_info = None
    pae_chain_offsets = None
    cb_to_ca_maps = None
    pair_results: Optional[list] = None
    all_chain_offsets: Optional[dict] = None
    failure_reason: Optional[str] = None

    # n_chains is authoritatively set by parse_complex_identity / _populate_identity_fields
    # (filename_n_chains when no PDB, else PDB chain count, with mismatch flagging).

    if 'pdb' not in file_paths:
        return contact_result, chain_info, pae_chain_offsets, cb_to_ca_maps, pair_results, all_chain_offsets, failure_reason

    try:
        chain_info = read_pdb_with_chain_info(str(file_paths['pdb']))
        if len(chain_info.chain_ids) == 0:
            # Parsed but empty — surface as no_chains rather than parse_error.
            failure_reason = PARTIAL_REASON_PDB_NO_CHAINS
            chain_info = None
            return contact_result, chain_info, pae_chain_offsets, cb_to_ca_maps, pair_results, all_chain_offsets, failure_reason
        if len(chain_info.chain_ids) >= 2:
            # Find the best interacting chain pair - also handles multi-chain
            ch_a, ch_b, contact_result = find_best_chain_pair(chain_info, t=8)
            row['best_chain_pair'] = f'{ch_a}_{ch_b}'
            row['pdockq'] = round(contact_result.pdockq, 4)
            row['ppv'] = round(contact_result.ppv, 4)

            # Multimer_v1: enumerate every inter-chain pair (includes zero-contact pairs)
            # and compute the whole-complex pDockQ. For N=2 whole-complex exactly equals
            # best-pair pDockQ (protected by Phase 6 dimer regression test).
            acc_map_raw = row.get('accession_chain_map')
            try:
                acc_map = json.loads(acc_map_raw) if isinstance(acc_map_raw, str) else (acc_map_raw or {})
            except (json.JSONDecodeError, TypeError):
                acc_map = {}
            pair_results = compute_all_chain_pairs(chain_info, accession_chain_map=acc_map, t=8)
            row['pdockq_whole_complex'] = round(calc_pdockq_whole_complex(chain_info, t=8), 4)

            # All-chain PAE offsets (for per-pair PAE aggregation across the complex).
            if run_interface_pae and pae_matrix is not None:
                all_chain_offsets = compute_pae_chain_offsets(chain_info)
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
            print(f"  Warning: <2 chains in {file_paths['pdb']}", file=sys.stderr)
            # Single-chain parse counts as "no usable chains for interface analysis".
            failure_reason = PARTIAL_REASON_PDB_NO_CHAINS

    except FileNotFoundError as error:
        print(f"  Warning: PDB not found {file_paths['pdb']}: {error}", file=sys.stderr)
        contact_result = None
        chain_info = None
        pair_results = None
        all_chain_offsets = None
        failure_reason = PARTIAL_REASON_PDB_IO_ERROR
    except PermissionError as error:
        print(f"  Warning: PDB permission error for {file_paths['pdb']}: {error}", file=sys.stderr)
        contact_result = None
        chain_info = None
        pair_results = None
        all_chain_offsets = None
        failure_reason = PARTIAL_REASON_PDB_IO_ERROR
    except (ValueError, IndexError, KeyError) as error:
        contact_result = None
        chain_info = None
        pair_results = None
        all_chain_offsets = None
        msg = str(error).lower()
        if "no chains" in msg or "empty" in msg:
            failure_reason = PARTIAL_REASON_PDB_NO_CHAINS
        else:
            failure_reason = PARTIAL_REASON_PDB_PARSE_ERROR
        print(f"  Warning: PDB parse error for {file_paths['pdb']}: {error}", file=sys.stderr)
    except OSError as error:
        # By the time this helper runs the PDB has already been decompressed
        # (Section 4 of the plan), so OSError here is genuinely an I/O failure
        # against the on-disk decompressed file rather than a corrupt stream.
        print(f"  Warning: PDB OSError for {file_paths['pdb']}: {error}", file=sys.stderr)
        contact_result = None
        chain_info = None
        pair_results = None
        all_chain_offsets = None
        failure_reason = PARTIAL_REASON_PDB_IO_ERROR
    except Exception as error:
        # Genuinely unclassified — fall back to the legacy alias.
        print(f"  Warning: pDockQ failed for {file_paths['pdb']}: {error}", file=sys.stderr)
        contact_result = None
        chain_info = None
        pair_results = None
        all_chain_offsets = None
        failure_reason = PARTIAL_REASON_UNREADABLE_PDB

    return contact_result, chain_info, pae_chain_offsets, cb_to_ca_maps, pair_results, all_chain_offsets, failure_reason


def _compute_interface_features(
    complex_name: str,
    row: dict,
    contact_result: Optional[object],
    chain_info: Optional[object],
    pae_matrix: Optional[np.ndarray],
    pae_chain_offsets: Optional[tuple],
    cb_to_ca_maps: Optional[tuple],
    pair_results: Optional[list] = None,
    all_chain_offsets: Optional[dict] = None,
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

        # Multimer_v1 all-pairs aggregates. Populates pair_metrics JSON +
        # pdockq_mean/_min, contact_count_total, interface_plddt_mean,
        # symmetry_mean/_min, and (when PAE is available) PAE-fraction means.
        # For N=2 these coexist with the best-pair columns and must be metric-
        # identical on pdockq_mean / pdockq_min / contact_count_total.
        if pair_results is not None:
            cb_map_dict = None
            if chain_info is not None and hasattr(chain_info, 'cb_to_ca_map'):
                cb_map_dict = chain_info.cb_to_ca_map
            pair_records = compute_all_pair_metrics(
                pair_results,
                pae_matrix=pae_matrix if run_interface_pae else None,
                chain_offsets=all_chain_offsets,
                cb_to_ca_map=cb_map_dict,
            )
            row['pair_metrics'] = serialise_pair_metrics(pair_records)
            aggregates = aggregate_pair_metrics(pair_records)
            for agg_key, agg_val in aggregates.items():
                row[agg_key] = agg_val

        # Extended flags: structural + paradox detection
        flags = compute_extended_flags(interface_features, iptm=row.get('iptm'), pdockq=row.get('pdockq'), disorder_fraction=row.get('plddt_below50_fraction'))
        row['interface_flags'] = ','.join(flags) if flags else ''

        if verbose and interface_features.get('n_interface_contacts', 0) > 0:
            n_contacts = interface_features['n_interface_contacts']
            if_plddt = interface_features.get('interface_plddt_combined', 'N/A')
            delta = interface_features.get('interface_vs_bulk_delta', 'N/A')
            print(f"  Interface -> {n_contacts} contacts, "
                  f"pLDDT={if_plddt}, delta={delta}")
            if run_interface_pae and interface_features.get('pae_confident_contact_fraction') is not None:
                pae_frac = interface_features['pae_confident_contact_fraction']
                strict_frac = interface_features.get('strict_confident_contact_fraction')
                strict_str = f"{strict_frac:.1%}" if strict_frac is not None else "N/A"
                print(f"  Interface PAE -> pae_confident={pae_frac:.1%}, strict_confident={strict_str}")
            score = interface_features.get('interface_confidence_score')
            if score is not None:
                print(f"  Composite score: {score:.4f}")

    except Exception as error:
        print(f"  Warning: Interface analysis failed for {complex_name}: {error}", file=sys.stderr)


def _populate_identity_fields(row: dict, identity: ComplexIdentity) -> None:
    """Write a ComplexIdentity into the CSV row dict.

    Sets the multimer_v1 identity columns plus the legacy fields (`complex_name`,
    `protein_a`, `protein_b`, `complex_type`, `n_chains`) so callers see a
    consistent view. For N=2 this preserves the pre-refactor legacy semantics.
    """
    row['schema_version'] = SCHEMA_VERSION
    row['complex_name'] = identity.complex_id
    # Legacy two-slot view: first two tokens when N>=2, same token when N==1
    if identity.filename_n_chains >= 2:
        row['protein_a'] = identity.accessions[0]
        row['protein_b'] = identity.accessions[1]
    else:
        row['protein_a'] = identity.complex_id
        row['protein_b'] = identity.complex_id
    row['complex_type'] = identity.legacy_complex_type
    row['n_chains'] = identity.n_chains
    row['stoichiometry'] = identity.stoichiometry
    row['is_homomeric'] = identity.is_homomeric
    row['unique_accessions'] = ','.join(identity.unique_accessions)
    row['chain_ids'] = ','.join(identity.chain_ids)
    row['accession_chain_map'] = json.dumps(identity.accession_chain_map)
    row['filename_n_chains'] = identity.filename_n_chains
    row['pdb_n_chains'] = identity.pdb_n_chains
    row['chain_count_consistency'] = identity.chain_count_consistency
    row['tier_scope'] = "dimer_validated" if identity.n_chains == 2 else "multimer_provisional"
    row['complex_identity_json'] = json.dumps({
        'complex_id': identity.complex_id,
        'accessions': list(identity.accessions),
        'unique_accessions': list(identity.unique_accessions),
        'n_chains': identity.n_chains,
        'stoichiometry': identity.stoichiometry,
        'is_homomeric': identity.is_homomeric,
        'accession_chain_map': identity.accession_chain_map,
        'chain_ids': list(identity.chain_ids),
        'legacy_complex_type': identity.legacy_complex_type,
        'filename_n_chains': identity.filename_n_chains,
        'pdb_n_chains': identity.pdb_n_chains,
        'chain_count_consistency': identity.chain_count_consistency,
    })


def _populate_scope_flags(row: dict) -> None:
    """Identity-flag wiring hook (kept for symmetry with _populate_identity_fields).

    The strict `composite_is_calibrated` assignment used to live here, but it
    fired before composite inputs were known — so any dimer was wrongly tagged
    True even when ipTM/PAE/contacts were missing. Final assignment now
    happens once at the end of `process_single_complex` via
    `_finalise_calibration_flag`. This hook is intentionally minimal: it only
    needs to ensure `composite_is_calibrated` exists on the row.
    """
    row.setdefault('composite_is_calibrated', False)


def _finalise_calibration_flag(row: dict, *, run_interface_pae: bool) -> None:
    """Set the strict `composite_is_calibrated` flag and stamp the
    `MISSING_COMPOSITE` diagnostic when a dimer-validated row was expected to
    be calibrated but is not.

    A row is calibrated only when:
      * tier_scope == 'dimer_validated' (existing scope rule), AND
      * no upstream failure has stamped `partial_reason`, AND
      * every input the composite depends on is actually present.

    Multimer-provisional rows and basic-mode runs (no `--interface --pae`) are
    deliberately uncalibrated by design and must NOT be tagged partial.
    """
    composite_inputs_present = (
        _has_value(row.get('iptm'))
        and _has_value(row.get('interface_confidence_score'))
        and _has_value(row.get('strict_confident_contact_fraction'))
        and _has_value(row.get('interface_pae_mean'))
        and _has_value(row.get('n_interface_contacts'))
        and int(row.get('n_interface_contacts') or 0) > 0
    )
    row['composite_is_calibrated'] = bool(
        row.get('tier_scope') == 'dimer_validated'
        and not row.get('partial_reason')
        and composite_inputs_present
    )

    # Stamp MISSING_COMPOSITE only when interface+PAE was attempted and the
    # row should have been calibrated but isn't. Skip basic-mode and N>2.
    # _stamp_partial_reason's priority map enforces the existing gating: any
    # prior reason outranks MISSING_COMPOSITE so the stamp is a no-op when an
    # upstream failure has already been recorded.
    if (
        run_interface_pae
        and row.get('tier_scope') == 'dimer_validated'
        and not row['composite_is_calibrated']
    ):
        _stamp_partial_reason(row, PARTIAL_REASON_MISSING_COMPOSITE)


def process_single_complex(complex_name: str, file_paths: dict[str, Path], *, run_interface: bool = False, run_interface_pae: bool = False, export_interfaces: bool = False, stash_variant_data: bool = False, verbose: bool = False) -> dict:
    """Run all analysis steps on a single protein complex.
    Args:
        complex_name: Parsed complex identifier.
        file_paths: Dict with optional 'pdb' and 'pkl' Path entries.
        run_interface: Whether to compute interface geometry + pLDDT features.
        run_interface_pae: Whether to also compute PAE-based interface features (requires both PDB and PKL - implies run_interface=True).
        export_interfaces: Whether to capture confident interface residue data for JSONL export (requires --interface --pae).
        stash_variant_data: Whether to stash _chain_info, _pdb_path, and confident residue numbers for variant mapping (requires --interface --pae).
        verbose: Whether to print per-step progress.
    Returns:
        Dictionary of results for this complex (one CSV row).
    """
    # Decompress .pdb.bz2 / .pdb.gz once per complex into a per-call tempfile so
    # the four downstream PDB readers (extract_plddt_from_pdb, the three passes
    # inside read_pdb_with_chain_info_New, and the SASA parser) all consume
    # plain-disk text instead of re-decompressing a slow sequential codec each
    # time. Plain .pdb inputs pass through unchanged.
    has_pdb_originally = 'pdb' in file_paths
    pdb_decompression_failure: Optional[str] = None

    with ExitStack() as _pdb_stack:
        if has_pdb_originally:
            original_pdb_path = file_paths['pdb']
            suffixes = Path(original_pdb_path).suffixes
            is_compressed = ".bz2" in suffixes or ".gz" in suffixes
            try:
                effective_pdb = _pdb_stack.enter_context(
                    decompressed_pdb_view(original_pdb_path)
                )
                file_paths = {**file_paths, 'pdb': effective_pdb}
            except (gzip.BadGzipFile, EOFError, OSError) as error:
                # Corrupt compressed stream or uncompressed I/O failure during
                # decompression. Record the failure, drop 'pdb' from file_paths
                # so downstream readers don't try the corrupt original, but keep
                # has_pdb=True (the resolver did discover a PDB file).
                print(
                    f"  Warning: PDB decompression failed for {original_pdb_path}: {error}",
                    file=sys.stderr,
                )
                pdb_decompression_failure = (
                    PARTIAL_REASON_PDB_DECOMPRESSION_ERROR if is_compressed
                    else PARTIAL_REASON_PDB_IO_ERROR
                )
                file_paths = {k: v for k, v in file_paths.items() if k != 'pdb'}

        # Phase 2: parse full ComplexIdentity from filename. pdb_n_chains is unknown
        # at this point; populate initial identity fields so downstream steps always
        # see stoichiometry/tier_scope. After the PDB is read we re-parse with
        # pdb_n_chains and overwrite - this is what surfaces filename/PDB mismatches.
        initial_identity = parse_complex_identity(complex_name, pdb_n_chains=None)

        row: dict = {
            # has_pdb reflects discovery, not usability — `composite_is_calibrated`
            # is the computability guarantee, and `partial_reason` records why a
            # discovered file was unusable.
            'has_pdb': has_pdb_originally,
            'has_pkl': 'pkl' in file_paths,
            'geometry_available': False,
            'species': 'Homo sapiens (9606)',
            'structure_source': 'AlphaFold2_prediction',
            # Always-present diagnostic; populated below on any failure path.
            'partial_reason': PARTIAL_REASON_NONE,
            'composite_is_calibrated': False,
            # Default screening status; the finalisation block at the end of this
            # function overwrites this with the real classification.
            'composite_screen_status': COMPOSITE_SCREEN_STATUS_UNAVAILABLE,
        }
        _populate_identity_fields(row, initial_identity)
        _populate_scope_flags(row)

        # Stamp PDB decompression failure (if any) before downstream stamping —
        # priority ordering ensures it remains the dominant reason.
        if pdb_decompression_failure:
            _stamp_partial_reason(row, pdb_decompression_failure)

        pae_matrix, pkl_failure_reason = _extract_pkl_metrics(
            file_paths, row, run_interface_pae=run_interface_pae, verbose=verbose,
        )
        _stamp_partial_reason(row, pkl_failure_reason)

        _extract_pdb_plddt(file_paths, row, verbose=verbose)
        (
            contact_result,
            chain_info,
            pae_chain_offsets,
            cb_to_ca_maps,
            pair_results,
            all_chain_offsets,
            pdb_failure_reason,
        ) = _compute_pdockq_and_chain_info(
            file_paths, row, pae_matrix,
            run_interface_pae=run_interface_pae, verbose=verbose,
        )
        _stamp_partial_reason(row, pdb_failure_reason)

        # Geometry availability: True iff every chain pair was successfully enumerated.
        # Zero-contact pairs are valid geometry (still count toward the expected pair
        # count); the flag only goes False on actual enumeration failure (no PDB,
        # unparseable PDB, chain extraction failure).
        n_chains_for_pairs = len(chain_info.chain_ids) if chain_info is not None else 0
        expected_pair_count = n_chains_for_pairs * (n_chains_for_pairs - 1) // 2
        row['geometry_available'] = bool(
            chain_info is not None
            and pair_results is not None
            and len(pair_results) == expected_pair_count
        )

        # Finalise identity now that PDB chain count is known. When filename and PDB
        # disagree, n_chains follows PDB (authoritative for structural metrics) and
        # the row is flagged via chain_count_consistency == "mismatch".
        if chain_info is not None:
            pdb_n_chains = len(chain_info.chain_ids)
            final_identity = parse_complex_identity(complex_name, pdb_n_chains=pdb_n_chains)
            _populate_identity_fields(row, final_identity)
            _populate_scope_flags(row)

        if run_interface and 'pdb' in file_paths:
            # Keep confident residue numbers if exporting interfaces OR stashing for variant mapping
            _compute_interface_features(
                complex_name, row, contact_result, chain_info,
                pae_matrix, pae_chain_offsets, cb_to_ca_maps,
                pair_results, all_chain_offsets,
                run_interface_pae=run_interface_pae,
                export_interfaces=export_interfaces or stash_variant_data,
                verbose=verbose,
            )

            # Stash raw interface residue PDB numbers for PyMOL script generation
            # (public keys - written to CSV so batch CLI can also skip re-reading PDBs)
            if contact_result is not None and chain_info is not None:
                _bp = row.get('best_chain_pair', '')
                if _bp and '_' in _bp:
                    _ia, _ib = _bp.split('_', 1)
                else:
                    _ia, _ib = 'A', 'B'
                row['interface_residues_a'] = '|'.join(
                    str(chain_info.chain_res_numbers[_ia][i])
                    for i in sorted(contact_result.interface_residues_a)
                )
                row['interface_residues_b'] = '|'.join(
                    str(chain_info.chain_res_numbers[_ib][i])
                    for i in sorted(contact_result.interface_residues_b)
                )

            # Stash structural data for variant mapping (private keys, stripped before CSV write)
            # SASA is computed here inside the worker to avoid pickling heavy ChainInfo_New
            # objects (numpy arrays) across the process boundary - only lightweight dicts are returned.
            if stash_variant_data and chain_info is not None:
                from variant_mapper import compute_residue_sasa_both_chains
                best_pair = row.get('best_chain_pair', '')
                if best_pair and '_' in best_pair:
                    _va, _vb = best_pair.split('_', 1)
                else:
                    _va, _vb = 'A', 'B'
                try:
                    sasa_a, sasa_b = compute_residue_sasa_both_chains(
                        file_paths['pdb'], _va, _vb)
                except Exception:
                    sasa_a, sasa_b = {}, {}

                row['_sasa_a'] = sasa_a                    # dict[int, float] - lightweight
                row['_sasa_b'] = sasa_b
                row['_chain_res_numbers_a'] = chain_info.chain_res_numbers.get(_va, [])
                row['_chain_res_numbers_b'] = chain_info.chain_res_numbers.get(_vb, [])
                row['_cb_coords_a'] = chain_info.cb_coords.get(_va, np.empty((0, 3))).tolist()
                row['_cb_coords_b'] = chain_info.cb_coords.get(_vb, np.empty((0, 3))).tolist()
                row['_confident_residue_numbers_a'] = row.get('confident_residue_numbers_a', [])
                row['_confident_residue_numbers_b'] = row.get('confident_residue_numbers_b', [])

        # Zero-contact diagnostic — ill-defined composite (log(0) in pDockQ).
        # _stamp_partial_reason's priority map preserves the precedence rule
        # (any earlier PDB/PKL failure outranks this).
        n_contacts = row.get('n_interface_contacts')
        if _has_value(n_contacts) and int(n_contacts) == 0:
            _stamp_partial_reason(row, PARTIAL_REASON_ZERO_CONTACTS)

        # Final strict calibration flag — replaces the optimistic per-identity
        # assignment that used to fire from `_populate_scope_flags` before
        # composite inputs were known.
        _finalise_calibration_flag(row, run_interface_pae=run_interface_pae)

        # Quality tier classification
        row['quality_tier'] = classify_prediction_quality(row.get('iptm'), row.get('pdockq'))
        row['quality_tier_v2'] = classify_prediction_quality_v2(row.get('iptm'), row.get('pdockq'), row.get('interface_confidence_score'))

        # Composite screening interpretation (Decision #43). Runs after the v2
        # tier and after _finalise_calibration_flag has set composite_is_calibrated,
        # so the inputs the classifier reads are all final.
        row['composite_screen_status'] = classify_composite_screen_status(
            composite_is_calibrated=row.get('composite_is_calibrated'),
            interface_confidence_score=row.get('interface_confidence_score'),
            iptm=row.get('iptm'),
            pdockq=row.get('pdockq'),
        )

        return row

#----------------------------Results Output-------------------------------------

def get_csv_fieldnames(include_interface: bool = False, include_pae: bool = False, include_enrichment: bool = False, include_clustering: bool = False, include_variants: bool = False, include_stability: bool = False, include_protvar: bool = False, include_disease: bool = False, include_pathways: bool = False) -> list[str]:
    """Build the CSV column list based on enabled features."""
    fieldnames = list(CSV_FIELDNAMES_BASE)
    if include_enrichment:
        fieldnames.extend(CSV_FIELDNAMES_ENRICHMENT)
    if include_interface:
        fieldnames.extend(CSV_FIELDNAMES_INTERFACE)
        fieldnames.extend(CSV_FIELDNAMES_FLAGS)
    if include_pae:
        fieldnames.extend(CSV_FIELDNAMES_INTERFACE_PAE)
    if include_clustering:
        from protein_clustering import CSV_FIELDNAMES_CLUSTERING
        fieldnames.extend(CSV_FIELDNAMES_CLUSTERING)
    if include_variants:
        from variant_mapper import CSV_FIELDNAMES_VARIANTS
        fieldnames.extend(CSV_FIELDNAMES_VARIANTS)
    if include_stability:
        from stability_scorer import CSV_FIELDNAMES_STABILITY
        fieldnames.extend(CSV_FIELDNAMES_STABILITY)
    if include_protvar:
        from protvar_client import CSV_FIELDNAMES_PROTVAR
        fieldnames.extend(CSV_FIELDNAMES_PROTVAR)
    if include_disease:
        from disease_annotations import CSV_FIELDNAMES_DISEASE
        fieldnames.extend(CSV_FIELDNAMES_DISEASE)
    if include_pathways:
        from pathway_network import CSV_FIELDNAMES_PATHWAYS
        fieldnames.extend(CSV_FIELDNAMES_PATHWAYS)
    return fieldnames

def write_results_csv(results: list[dict], output_path: str, include_interface: bool = False, include_pae: bool = False, include_enrichment: bool = False, include_clustering: bool = False, include_variants: bool = False, include_stability: bool = False, include_protvar: bool = False, include_disease: bool = False, include_pathways: bool = False) -> None:
    """Write batch analysis results to a CSV file.
    Args:
        results: List of per-complex result dictionaries.
        output_path: File path for the output CSV.
        include_interface: Whether to include interface columns.
        include_pae: Whether to include PAE interface columns.
        include_enrichment: Whether to include enrichment columns (gene symbols, names, database sources, sequences).
        include_clustering: Whether to include clustering columns (cluster IDs, homologous pairs).
        include_variants: Whether to include variant mapping columns.
        include_stability: Whether to include stability scoring columns (EVE scores).
        include_protvar: Whether to include ProtVar API cross-validation columns.
        include_disease: Whether to include disease annotation columns (disease, PTM, GO, drug target).
        include_pathways: Whether to include pathway and network columns.
    """
    fieldnames = get_csv_fieldnames(include_interface, include_pae, include_enrichment, include_clustering, include_variants, include_stability, include_protvar, include_disease, include_pathways)

    with open(output_path, 'w', newline='', encoding='utf-8-sig') as csv_file:
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
                pae_confident_contact_fraction=row.get('pae_confident_contact_fraction'),
                strict_confident_contact_fraction=row.get('strict_confident_contact_fraction'),
                interface_plddt_combined=row.get('interface_plddt_combined'),
            )
            jsonl_file.write(json.dumps(record) + '\n')
            exported_count += 1

    return exported_count

#------------------Species Annotation & Filtering-------------------------------------

def annotate_species(results: list[dict], classifier) -> None:
    """Populate species_a, species_b, species_status for each result row in-place.
    Called before enrichment so downstream modules can gate species-dependent
    work. Values: reviewed_human / trembl_human / non_human.
    Args:
        results: List of per-complex result dicts from batch processing.
        classifier: SpeciesClassifier instance (from id_mapper).
    """
    from id_mapper import combine_species_statuses
    for row in results:
        sa = classifier.classify(row.get('protein_a', ''))
        sb = classifier.classify(row.get('protein_b', ''))
        row['species_a'] = sa
        row['species_b'] = sb
        row['species_status'] = combine_species_statuses(sa, sb)


def get_human_mask(df) -> 'pd.Series':
    """DataFrame mask for figure / Venn overlap filtering: reviewed_human only.
    Used where dissertation numbers must compare like-with-like reviewed entries.
    """
    return df['species_status'] == 'reviewed_human'


def is_annotatable(row: dict) -> bool:
    """Per-row guard for annotation loops: skip only non_human rows.
    TrEMBL-human rows still get variant/disease/cluster annotation because
    UniProt has records for them. Rows without species_status default to
    annotatable (back-compat for callers on pre-species CSVs).
    """
    status = row.get('species_status')
    if status is None:
        return True
    return status != 'non_human'


#------------------Enrichment (gene symbols, database sources)-----------------------

def _resolve_missing_proteins_batched(
    identifiers: list[str],
    mapper,
    batch_size: int = ENRICH_API_BATCH_SIZE,
) -> dict[str, str]:
    """Resolve identifiers to ENSP IDs via STRING API, with cache-aware batching.

    Replaces the per-row ``mapper.resolve_id`` calls that used to dominate
    ``enrich_results``. The fast path is a per-protein cache lookup that
    matches the cache-key shape of the legacy ``get_string_ids([id])`` calls,
    so caches written by previous toolkit runs (Run 2 / Run 3 single-protein
    keys) are fully reused. Cache misses are batched into one API call per
    ``batch_size`` identifiers; each batch response is then decomposed and
    written back as per-protein cache entries so subsequent runs skip the
    network entirely.

    Args:
        identifiers: Identifiers (UniProt accessions, gene names, etc.) that
            missed the local lookup table.
        mapper: ``IDMapper`` instance — only ``mapper._api_available`` is
            consulted (and latched off on ``StringAPIError``). Pass ``None``
            to disable API resolution entirely.
        batch_size: Number of identifiers per batched API call.

    Returns:
        Dict ``{input_id: ensp}`` for resolved proteins. Identifiers absent
        from the dict either have no STRING mapping (true negative) or were
        skipped because the API latch was off.
    """
    if not identifiers or mapper is None:
        return {}
    if not getattr(mapper, "_api_available", True):
        return {}

    from string_api import (
        _cache_key,
        _read_cache,
        _resolve_cache_dir,
        _write_cache,
        get_string_ids,
        StringAPIError,
        STRING_API_SPECIES,
    )
    from id_mapper import STRING_TAXONOMY_PREFIX

    cache_dir = _resolve_cache_dir(None)
    resolved: dict[str, str] = {}
    to_batch: list[str] = []

    # Step 1: per-protein cache lookup. Single-id key shape matches what
    # `get_string_ids([id])` would write, so any prior single-call cache
    # hits here without a network round-trip.
    for prot in identifiers:
        if not cache_dir:
            to_batch.append(prot)
            continue
        single_params = {
            "identifiers": prot,
            "species": STRING_API_SPECIES,
            "echo_query": 1,
        }
        key = _cache_key("get_string_ids", single_params)
        cached = _read_cache(cache_dir, key)
        if cached is None:
            to_batch.append(prot)
            continue
        # Cache hit. Empty list = known miss; leave unresolved.
        if cached:
            row = cached[0]
            ensp = str(row.get("stringId", "")).removeprefix(
                STRING_TAXONOMY_PREFIX
            )
            if ensp:
                resolved[prot] = ensp

    # Step 2: batch the residual.
    for i in range(0, len(to_batch), batch_size):
        chunk = to_batch[i:i + batch_size]
        try:
            df = get_string_ids(chunk)
        except StringAPIError as e:
            # HTTP 404 from STRING means "no matches in this batch" — common
            # for batches of TrEMBL-only accessions, since STRING doesn't
            # recognise any of them and returns a single 404 instead of an
            # empty 200. Treat as an empty response, cache the IDs as known
            # misses, and continue with the next chunk. Other errors (5xx,
            # timeouts, malformed responses) latch off the API entirely.
            #
            # The unpatched per-row code rarely hit this because single-ID
            # calls return 200+[] for unknown IDs; only multi-ID batches
            # where every member is unknown trigger the 404 path.
            if "HTTP 404" in str(e):
                if cache_dir:
                    for prot in chunk:
                        single_params = {
                            "identifiers": prot,
                            "species": STRING_API_SPECIES,
                            "echo_query": 1,
                        }
                        key = _cache_key("get_string_ids", single_params)
                        try:
                            _write_cache(
                                cache_dir, key, "get_string_ids", []
                            )
                        except OSError:
                            pass
                continue
            mapper._api_available = False
            break

        records = (
            df.to_dict(orient="records")
            if df is not None and not df.empty
            else []
        )

        per_prot: dict[str, list[dict]] = {}
        for r in records:
            query = str(r.get("queryItem", ""))
            if not query:
                continue
            per_prot.setdefault(query, []).append(r)
            ensp = str(r.get("stringId", "")).removeprefix(
                STRING_TAXONOMY_PREFIX
            )
            if ensp and query not in resolved:
                resolved[query] = ensp

        # Step 3: write per-protein cache entries so future runs skip step 2.
        if cache_dir:
            for prot in chunk:
                single_params = {
                    "identifiers": prot,
                    "species": STRING_API_SPECIES,
                    "echo_query": 1,
                }
                key = _cache_key("get_string_ids", single_params)
                rows_for_prot = per_prot.get(prot, [])
                try:
                    _write_cache(
                        cache_dir, key, "get_string_ids", rows_for_prot
                    )
                except OSError:
                    pass

    return resolved


def enrich_results(results: list[dict], lookup: dict[str, dict], database_pair_sets: Optional[dict[str, set]] = None, database_evidence: Optional[dict[str, set]] = None, mapper=None) -> None:
    """Enrich result rows with gene symbols, protein names, and database sources.
    Modifies the result dictionary in-place.

    Performance design (post-Run-3 retune): three pre-passes hoist the work
    that used to fire per-row out of the hot loop.

      * Precomputed ``ensp_to_info`` reverse index replaces the
        ``for acc, info in lookup.items()`` scan that used to fire on every
        API-resolved miss (~72,000 dict items walked per scan over the 144K
        lookup table).
      * ``isoform_base`` precomputes ``split_isoform`` for every distinct
        missing protein so the hot loop hits a base-accession entry in
        ``lookup`` without re-parsing the isoform suffix per row. This
        replaces the ``mapper._resolve_id_local → uniprot_to_ensembl``
        local-resolution path that the unpatched code triggered per row.
      * ``_resolve_missing_proteins_batched`` collects only the *truly*
        missing identifiers (no lookup hit, no isoform-base hit) and
        resolves them in chunks of ``ENRICH_API_BATCH_SIZE`` in one
        pre-pass, replacing the per-row ``mapper.resolve_id`` HTTP
        round-trips. On a 9.4K dataset with 18.6% miss rate this collapses
        ~3,500 serial calls (~38 minutes wall-clock) into a handful of
        batched calls.

    Args:
        results: Per-complex result dicts from batch processing.
        lookup: UniProt-keyed lookup dict from ``build_uniprot_lookup()``.
            Already indexes by both isoform-suffixed and base accession.
        database_pair_sets: Optional ``{db_name: set of normalised UniProt
            pairs}`` for source-of-complex tagging.
        database_evidence: Optional ``{db_name: set of evidence type strings}``
            pre-computed once upstream to avoid scanning large DataFrames
            per row.
        mapper: Optional ``IDMapper`` for API-backed resolution of proteins
            absent from both the lookup and its isoform-base index.
    """
    from overlap_analysis import normalise_pair
    from id_mapper import split_isoform

    # Pre-pass 1: ENSP -> info reverse index (one-time O(N)).
    ensp_to_info: dict[str, dict] = {}
    for info in lookup.values():
        ensp = info.get("ensembl_protein_id")
        if ensp:
            ensp_to_info.setdefault(ensp, info)

    # Pre-pass 2: classify every distinct missing protein. Isoforms whose
    # base accession is already in ``lookup`` are recoverable in-memory; only
    # the residual goes to the API.
    isoform_base: dict[str, str] = {}
    truly_missing: set[str] = set()
    seen: set[str] = set()
    for row in results:
        for prot in (row.get("protein_a", ""), row.get("protein_b", "")):
            if not prot or prot in seen or prot in lookup:
                continue
            seen.add(prot)
            base, _ = split_isoform(prot)
            if base != prot and base in lookup:
                isoform_base[prot] = base
            else:
                truly_missing.add(prot)

    # Pre-pass 3: batch-resolve the truly-missing residual via STRING API.
    api_resolved: dict[str, str] = {}
    if (
        mapper is not None
        and getattr(mapper, "_api_available", True)
        and truly_missing
    ):
        t0 = time.time()
        api_resolved = _resolve_missing_proteins_batched(
            sorted(truly_missing), mapper
        )
        print(
            f"  Resolved {len(isoform_base):,} isoforms locally; "
            f"batch-resolved {len(api_resolved):,}/{len(truly_missing):,} "
            f"residual proteins via STRING API in {time.time() - t0:.1f}s",
            file=sys.stderr,
        )
    elif isoform_base:
        print(
            f"  Resolved {len(isoform_base):,} isoforms locally "
            f"(API path skipped)",
            file=sys.stderr,
        )

    # Hoist the sort out of the hot loop.
    sorted_db_items = (
        sorted(database_pair_sets.items()) if database_pair_sets else []
    )

    bar = _make_progress_bar(len(results), desc="Enriching")
    for row in results:
        prot_a = row.get("protein_a", "")
        prot_b = row.get("protein_b", "")
        info_a = (
            lookup.get(prot_a)
            or lookup.get(isoform_base.get(prot_a))
            or ensp_to_info.get(api_resolved.get(prot_a, ""), {})
        )
        info_b = (
            lookup.get(prot_b)
            or lookup.get(isoform_base.get(prot_b))
            or ensp_to_info.get(api_resolved.get(prot_b, ""), {})
        )

        row["gene_symbol_a"] = info_a.get("gene_symbol", "")
        row["gene_symbol_b"] = info_b.get("gene_symbol", "")
        row["protein_name_a"] = info_a.get("protein_name", "")
        row["protein_name_b"] = info_b.get("protein_name", "")
        row["ensembl_id_a"] = info_a.get("ensembl_protein_id", "")
        row["ensembl_id_b"] = info_b.get("ensembl_protein_id", "")
        row["secondary_accessions_a"] = info_a.get("secondary_accessions", "")
        row["secondary_accessions_b"] = info_b.get("secondary_accessions", "")

        if sorted_db_items:
            pair = normalise_pair(prot_a, prot_b)
            sources = [name for name, ps in sorted_db_items if pair in ps]
            row["database_source"] = "|".join(sources)
            if database_evidence:
                evidence_set: set[str] = set()
                for db_name in sources:
                    ev = database_evidence.get(db_name)
                    if ev:
                        evidence_set.update(ev)
                row["evidence_types"] = "|".join(sorted(evidence_set))
            else:
                row["evidence_types"] = ""
        else:
            row["database_source"] = ""
            row["evidence_types"] = ""
        bar.update(1)
    bar.__exit__(None, None, None)

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

def append_checkpoint(results: list[dict], output_path: str, since_index: int) -> int:
    """Append only new results to the checkpoint file (avoids rewriting all).

    For large runs this is much faster than save_checkpoint() which rewrites
    the entire results list on every call.

    Args:
        results: Full results list accumulated so far.
        output_path: The main output CSV path (checkpoint path derived from it).
        since_index: Index of the first new result to append.

    Returns:
        Updated since_index (len(results)) for the next call.
    """
    ckpt = _checkpoint_path(output_path)
    with open(ckpt, 'a', encoding='utf-8') as fh:
        for row in results[since_index:]:
            fh.write(json.dumps(row, default=str) + '\n')
    n_appended = len(results) - since_index
    logger.info("Checkpoint appended: %d new rows -> %s", n_appended, ckpt)
    return len(results)

def remove_checkpoint(output_path: str) -> None:
    """Remove the checkpoint file after successful completion."""
    ckpt = _checkpoint_path(output_path)
    if ckpt.exists():
        ckpt.unlink()
        logger.info("Checkpoint removed: %s", ckpt)


# ── Incremental mode (--skip-existing) helpers ──────────────────────────────


def _resolve_audit_root() -> Path:
    """Resolve the audit-root directory, matching complex_resolver._resolve_audit_dir().

    The resolver and the toolkit MUST agree on this path so the toolkit can
    find the just-written ``runs/<id>/`` and the previous ``latest/`` mirror.
    Honours ``PROTEIN_TOOLKIT_PROJECT_ROOT`` (so tests can redirect via
    ``monkeypatch.setenv``); falls back to ``$CWD/data/complex_manifest_audit``.
    """
    env_root = os.environ.get("PROTEIN_TOOLKIT_PROJECT_ROOT")
    base = Path(env_root).expanduser().resolve() if env_root else Path.cwd()
    return base / "data" / "complex_manifest_audit"


def _stat_size_mtime(path: Path | None) -> tuple[int, int]:
    """Return ``(size_bytes, mtime_ns)`` for path, or ``(0, 0)`` if missing/unstattable."""
    if path is None:
        return 0, 0
    try:
        st = path.stat()
    except OSError:
        return 0, 0
    return st.st_size, st.st_mtime_ns


def _load_previous_fingerprints(
    audit_root: Path,
) -> dict[str, tuple[int, int, int, int]]:
    """Parse ``<audit_root>/latest/complex_manifest.tsv`` into a fingerprint dict.

    Returns ``{name: (pdb_size, pkl_size, pdb_mtime_ns, pkl_mtime_ns)}``.
    Returns ``{}`` when ``latest/`` does not yet exist (first incremental ever).

    MUST be called before the resolver runs in incremental mode — the resolver
    refreshes ``latest/`` as a side effect, after which this function would
    compare the current manifest to itself.
    """
    manifest_path = audit_root / "latest" / "complex_manifest.tsv"
    if not manifest_path.is_file():
        return {}
    fingerprints: dict[str, tuple[int, int, int, int]] = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            name = (row.get("name") or "").strip()
            if not name:
                continue
            try:
                fingerprints[name] = (
                    int(row.get("pdb_size_bytes", 0) or 0),
                    int(row.get("pkl_size_bytes", 0) or 0),
                    int(row.get("pdb_mtime_ns", 0) or 0),
                    int(row.get("pkl_mtime_ns", 0) or 0),
                )
            except ValueError:
                continue
    return fingerprints


def _load_previous_run_id(audit_root: Path) -> str:
    """Read the previous ``latest_run_id.txt`` value or return ``"none"``."""
    pointer = audit_root / "latest_run_id.txt"
    if not pointer.is_file():
        return "none"
    return pointer.read_text(encoding="utf-8").strip() or "none"


def _read_skip_existing_names(results_csv_path: Path) -> set[str]:
    """Read the ``complex_name`` column from a historical results CSV.

    Raises ``FileNotFoundError`` if the path doesn't exist, or ``ValueError``
    if the file is missing the ``complex_name`` column. Duplicates are
    deduplicated with a stderr warning.
    """
    if not results_csv_path.is_file():
        raise FileNotFoundError(
            f"--skip-existing CSV not found: {results_csv_path}"
        )
    names: set[str] = set()
    seen: set[str] = set()
    duplicates = 0
    with results_csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "complex_name" not in reader.fieldnames:
            raise ValueError(
                f"--skip-existing CSV {results_csv_path} is missing required "
                f"'complex_name' column. Note: results_strict_calibrated_*.csv "
                f"is an analysis subset and must not be used for skip-existing; "
                f"use the full results.csv from the toolkit pipeline."
            )
        for row in reader:
            name = (row.get("complex_name") or "").strip()
            if not name:
                continue
            if name in seen:
                duplicates += 1
            else:
                seen.add(name)
            names.add(name)
    if duplicates:
        print(
            f"  Warning: --skip-existing CSV {results_csv_path.name} had "
            f"{duplicates} duplicate complex_name row(s); using deduped set "
            f"({len(names):,} unique names)",
            file=sys.stderr,
        )
    return names


def _select_chunk(
    sorted_eligible: list[tuple[str, dict]],
    limit: int | None,
    completed_names: set[str],
) -> tuple[list[tuple[str, dict]], list[tuple[str, dict]]]:
    """Apply --limit then --resume filtering in the correct order.

    Args:
        sorted_eligible: Alphabetically-sorted (name, paths) pairs already filtered
            by --skip-existing (so only rows new vs the historical baseline).
        limit: Optional max chunk size. None = unlimited.
        completed_names: Names already present in the in-flight checkpoint
            (from --resume). Removed from the chunk *after* limit is applied,
            so chunk membership is stable across crash + resume - provided the
            historical baseline used by --skip-existing has not changed.

    Returns:
        (selected_chunk, to_process)
        selected_chunk: The full chunk membership (limit-applied), used for
            provenance logging - the deterministic "this is what this chunk owns".
        to_process: selected_chunk minus completed_names, the actual work list.
    """
    if limit is not None and len(sorted_eligible) > limit:
        selected_chunk = sorted_eligible[:limit]
    else:
        selected_chunk = sorted_eligible
    if completed_names:
        to_process = [item for item in selected_chunk if item[0] not in completed_names]
    else:
        to_process = selected_chunk
    return selected_chunk, to_process


def _compute_incremental_delta(
    complexes: dict[str, dict[str, Path]],
    previous_fingerprints: dict[str, tuple[int, int, int, int]],
    previous_run_id: str,
    results_csv_path: Path,
    run_dir: Path,
    *,
    output_csv: str,
    output_jsonl: str | None,
) -> dict:
    """Categorise discovered complexes against the historical results CSV.

    Writes ``manifest_delta.tsv``, ``already_processed.tsv``,
    ``changed_existing.tsv``, ``missing_since_previous_manifest.tsv``, and
    ``incremental_run_summary.txt`` into ``run_dir``. Refreshes the audit
    ``latest/`` mirror so it now includes the delta files.

    Reads only the historical results CSV and the previous fingerprints; never
    consults the in-flight checkpoint file. ``--resume`` filtering is applied
    separately by the caller (sequential filter invariant).

    Returns a dict with keys ``delta`` (set of names), ``already_processed``
    (list of ``(name, status)``), ``changed_existing`` (list of
    ``(name, status, prev, curr)``), ``missing`` (list of names), and
    ``previous_run_id``.
    """
    # Imported here to avoid a top-level dependency cycle and keep the
    # incremental path self-contained.
    from complex_resolver import _atomic_write_tsv, _refresh_latest

    audit_root = run_dir.parent.parent  # runs/<id>/.. = runs/, runs/.. = audit_root
    run_id = run_dir.name

    historical_names = _read_skip_existing_names(results_csv_path)
    had_previous_manifest = bool(previous_fingerprints)

    delta_names: list[str] = []
    already_rows: list[tuple[str, str]] = []
    changed_rows: list[
        tuple[str, str, tuple[int, int, int, int], tuple[int, int, int, int]]
    ] = []

    for name in sorted(complexes):
        if name not in historical_names:
            delta_names.append(name)
            continue
        # Already in historical CSV — skip and classify by fingerprint status.
        prev = previous_fingerprints.get(name)
        if prev is None:
            status = (
                "missing_previous_record"
                if had_previous_manifest
                else "no_previous_fingerprint"
            )
            already_rows.append((name, status))
            continue
        paths = complexes[name]
        pdb_size, pdb_mtime = _stat_size_mtime(paths.get("pdb"))
        pkl_size, pkl_mtime = _stat_size_mtime(paths.get("pkl"))
        if (pdb_size, pdb_mtime, pkl_size, pkl_mtime) == (0, 0, 0, 0):
            already_rows.append((name, "missing_current_stat"))
            continue
        curr = (pdb_size, pkl_size, pdb_mtime, pkl_mtime)
        if curr == prev:
            already_rows.append((name, "matched"))
        else:
            changed_rows.append((name, "changed", prev, curr))

    current_names = set(complexes.keys())
    missing_names = sorted(previous_fingerprints.keys() - current_names)

    # Write the four delta TSVs.
    _atomic_write_tsv(
        run_dir / "manifest_delta.tsv",
        ["name"],
        [[n] for n in delta_names],
    )
    _atomic_write_tsv(
        run_dir / "already_processed.tsv",
        ["name", "fingerprint_status"],
        [[n, s] for n, s in already_rows],
    )
    _atomic_write_tsv(
        run_dir / "changed_existing.tsv",
        [
            "name", "fingerprint_status",
            "prev_pdb_size_bytes", "curr_pdb_size_bytes",
            "prev_pkl_size_bytes", "curr_pkl_size_bytes",
            "prev_pdb_mtime_ns", "curr_pdb_mtime_ns",
            "prev_pkl_mtime_ns", "curr_pkl_mtime_ns",
        ],
        [
            [
                n, s,
                str(p[0]), str(c[0]),
                str(p[1]), str(c[1]),
                str(p[2]), str(c[2]),
                str(p[3]), str(c[3]),
            ]
            for n, s, p, c in changed_rows
        ],
    )
    _atomic_write_tsv(
        run_dir / "missing_since_previous_manifest.tsv",
        ["name"],
        [[n] for n in missing_names],
    )

    # Self-describing summary file (matches plan schema).
    summary_lines = [
        f"skip_existing_reference:         {results_csv_path}",
        f"historical_interfaces_reference: {output_jsonl if output_jsonl else 'none'}",
        f"output_csv:                      {output_csv}",
        f"output_interfaces_jsonl:         {output_jsonl if output_jsonl else 'none'}",
        f"previous_manifest_run_id:        {previous_run_id}",
        f"current_manifest_run_id:         {run_id}",
        (
            f"counts:                          "
            f"delta={len(delta_names)}, "
            f"already_processed={len(already_rows)}, "
            f"changed_existing={len(changed_rows)}, "
            f"missing={len(missing_names)}"
        ),
    ]
    (run_dir / "incremental_run_summary.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8",
    )

    # Refresh latest/ so the mirror includes the delta TSVs and summary too.
    _refresh_latest(audit_root, run_dir, run_id)

    return {
        "delta": set(delta_names),
        "already_processed": already_rows,
        "changed_existing": changed_rows,
        "missing": missing_names,
        "previous_run_id": previous_run_id,
    }

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

def _build_worker_exception_row(complex_name: str, file_paths: dict, error: Exception) -> dict:
    """Construct a sentinel row for a worker that raised before producing a row.

    Worker-exception rows are explicit sentinels: they carry
    ``partial_reason="worker_exception"`` so they are visible in
    ``_aggregate_summary_statistics``, but they do not introduce a fourth
    ``quality_tier_v2`` value (it is left ``None``). They count as both partial
    and incomplete; both filters exclude them from calibrated/recoverable
    analyses.
    """
    return {
        'complex_name': complex_name,
        'has_pdb': 'pdb' in file_paths,
        'has_pkl': 'pkl' in file_paths,
        'geometry_available': False,
        'composite_is_calibrated': False,
        'partial_reason': PARTIAL_REASON_WORKER_EXCEPTION,
        'quality_tier': 'Error',
        'quality_tier_v2': None,    # do NOT introduce a new v2 tier value
        'composite_screen_status': COMPOSITE_SCREEN_STATUS_UNAVAILABLE,
        'tier_scope': None,         # explicit sentinel: not classifiable
        'complex_type': None,       # explicit sentinel: not classifiable
        '_error': str(error),
    }


def _safe_process_single_complex(complex_name: str, file_paths: dict, **kwargs) -> dict:
    """Crash-safe wrapper around `process_single_complex`.

    Catches any exception escaping the per-complex pipeline and returns a
    sentinel row instead. Used by both sequential (`workers == 1`) and parallel
    paths so neither can kill the whole batch on a single complex's failure.
    """
    try:
        return process_single_complex(complex_name, file_paths, **kwargs)
    except Exception as error:
        print(
            f"\n  Error processing {complex_name}: {error}",
            file=sys.stderr,
        )
        return _build_worker_exception_row(complex_name, file_paths, error)


def _worker_process_complex(args_tuple: tuple) -> dict:
    """Top-level wrapper for process_single_complex that unpacks a tuple.
    ProcessPoolExecutor requires a picklable callable with a single argument. This unpacks the argument tuple and forwards to the real function.
    """
    complex_name, file_paths, kwargs = args_tuple
    return _safe_process_single_complex(complex_name, file_paths, **kwargs)

def run_batch_parallel(
    sorted_complexes: list[tuple[str, dict[str, Path]]],
    *,
    run_interface: bool,
    run_interface_pae: bool,
    export_interfaces: bool,
    stash_variant_data: bool = False,
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
        stash_variant_data: Whether to preserve chain_info/pdb_path for variant mapping.
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
        stash_variant_data=stash_variant_data,
        verbose=verbose and workers == 1,  # Verbose only meaningful in sequential mode
    )

    # Build work items
    work_items = [
        (name, paths, shared_kwargs)
        for name, paths in sorted_complexes
    ]

    results = list(resumed_results)  # Accumulate into a mutable list
    newly_processed = 0
    last_checkpoint_count = len(results)  # Track for append-only checkpoints
    start_time = time.monotonic()

    with _make_progress_bar(total, desc="Analysing complexes") as pbar:
        # Account for already-completed work
        if resumed_results:
            pbar.update(len(resumed_results))

        if workers == 1:
            #-------------------Sequential mode (preserves verbose output)---------------------
            for complex_name, file_paths, kwargs in work_items:
                # _safe_process_single_complex catches any exception escaping
                # the per-complex pipeline and returns a worker-exception
                # sentinel row instead — so a single bad complex cannot kill
                # the whole sequential batch.
                row = _safe_process_single_complex(complex_name, file_paths, **kwargs)
                results.append(row)
                newly_processed += 1
                tier = row.get('quality_tier', '?')
                pbar.set_postfix_str(f"{complex_name} -> {tier}")
                pbar.update(1)

                if enable_checkpoint and newly_processed % CHECKPOINT_INTERVAL == 0:
                    last_checkpoint_count = append_checkpoint(
                        results, output_path, last_checkpoint_count)

        else:
            #----------------------Parallel mode------------------------------------------------
            # verbose per-complex output is suppressed in parallel mode because interleaved prints from multiple workers are unreadable
            # Instead we show the most recent complex and its quality tier in the progress bar
            print(f"Starting {workers} worker processes...", flush=True)
            # Capture file_paths per future so the exception handler can build a
            # fully-populated sentinel row even though the worker may have died
            # before returning anything.
            future_to_item: dict = {}
            with ProcessPoolExecutor(max_workers=workers, initializer=_worker_initializer) as executor:
                for item in work_items:
                    fut = executor.submit(_worker_process_complex, item)
                    future_to_item[fut] = (item[0], item[1])
                for future in as_completed(future_to_item):
                    complex_name, fp = future_to_item[future]
                    try:
                        row = future.result(timeout=300)
                        results.append(row)
                        newly_processed += 1
                        tier = row.get('quality_tier', '?')
                        pbar.set_postfix_str(f"{complex_name} -> {tier}")
                    except Exception as error:
                        # The worker itself died (e.g. BrokenProcessPool, timeout).
                        # `_safe_process_single_complex` would normally catch
                        # in-worker exceptions, so reaching here means the
                        # subprocess never returned a row.
                        print(f"\n  Error processing {complex_name}: {error}",
                              file=sys.stderr)
                        results.append(_build_worker_exception_row(complex_name, fp, error))
                        newly_processed += 1
                    pbar.update(1)
                    if enable_checkpoint and newly_processed % CHECKPOINT_INTERVAL == 0:
                        last_checkpoint_count = append_checkpoint(
                            results, output_path, last_checkpoint_count)

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

    Tolerates incomplete rows (missing identity / quality_tier_v2 — the HPC
    crash mode). `partial_reason`-tagged rows that still carry identity remain
    in the existing tier counts; they are also surfaced separately as a
    diagnostic block.
    """
    total = len(results)

    # Single-pass partition. "Incomplete" means we can't classify the row at
    # all (missing complex_type or quality_tier_v2). "Partial" is orthogonal —
    # the row has identity but the worker stamped a diagnostic reason.
    valid: list[dict] = []
    incomplete: list[dict] = []
    for row in results:
        if row.get('complex_type') and row.get('quality_tier_v2'):
            valid.append(row)
        else:
            incomplete.append(row)

    partial_rows = [r for r in results if r.get('partial_reason')]
    calibrated_rows = [r for r in results if r.get('composite_is_calibrated')]

    stats: dict = {
        'total_complexes': total,
        'incomplete_count': len(incomplete),
        'partial_count': len(partial_rows),
        'partial_reason_counts': Counter(r.get('partial_reason') for r in partial_rows),
        'calibrated_count': len(calibrated_rows),
        'homodimer_count': sum(1 for row in valid if row.get('complex_type') == 'Homodimer'),
        'heterodimer_count': sum(1 for row in valid if row.get('complex_type') == 'Heterodimer'),
        'quality_high': sum(1 for row in valid if row.get('quality_tier') == 'High'),
        'quality_medium': sum(1 for row in valid if row.get('quality_tier') == 'Medium'),
        'quality_low': sum(1 for row in valid if row.get('quality_tier') == 'Low'),
        'iptm_values': [row['iptm'] for row in valid if row.get('iptm')],
        'pdockq_values': [row['pdockq'] for row in valid if row.get('pdockq')],
        'below50_values': [row['plddt_below50_fraction'] for row in valid if row.get('plddt_below50_fraction') is not None],
        'below70_values': [row['plddt_below70_fraction'] for row in valid if row.get('plddt_below70_fraction') is not None],
        'pkl_source_count': sum(1 for row in valid if row.get('plddt_source') == 'pkl'),
        'pdb_fallback_count': sum(1 for row in valid if row.get('plddt_source') == 'pdb'),
        'no_plddt_count': sum(1 for row in valid if row.get('plddt_source') is None),
    }

    if include_interface:
        stats['contact_counts'] = [row['n_interface_contacts'] for row in valid if row.get('n_interface_contacts') is not None]
        stats['if_plddt_values'] = [row['interface_plddt_combined'] for row in valid if row.get('interface_plddt_combined') is not None]
        stats['delta_values'] = [row['interface_vs_bulk_delta'] for row in valid if row.get('interface_vs_bulk_delta') is not None]
        all_flags: dict[str, int] = defaultdict(int)
        for row in valid:
            flags_str = row.get('interface_flags', '')
            if flags_str:
                for flag in flags_str.split(','):
                    all_flags[flag.strip()] += 1
        stats['all_flags'] = dict(all_flags)
        # Aggregate both fractions so the summary block can show the PAE-only distribution
        # (backward-compatible name for any log parsing) and the strict distribution that
        # now feeds the composite score.
        stats['confident_fractions'] = [row['pae_confident_contact_fraction'] for row in valid if row.get('pae_confident_contact_fraction') is not None]
        stats['strict_confident_fractions'] = [row['strict_confident_contact_fraction'] for row in valid if row.get('strict_confident_contact_fraction') is not None]
        stats['composite_scores'] = [row['interface_confidence_score'] for row in valid if row.get('interface_confidence_score') is not None]

        v2_tiers = [row.get('quality_tier_v2') for row in valid if row.get('quality_tier_v2') is not None]
        if v2_tiers:
            stats['v2_high'] = sum(1 for t in v2_tiers if t == 'High')
            stats['v2_medium'] = sum(1 for t in v2_tiers if t == 'Medium')
            stats['v2_low'] = sum(1 for t in v2_tiers if t == 'Low')
            stats['v2_upgrades'] = sum(1 for r in valid if r.get('quality_tier') != r.get('quality_tier_v2') and r.get('quality_tier_v2') == 'High')
            stats['v2_downgrades'] = sum(1 for r in valid if r.get('quality_tier') == 'High' and r.get('quality_tier_v2') != 'High')

    return stats


def print_summary(results: list[dict], include_interface: bool = False) -> None:
    """Print a human-readable summary of the batch analysis results.
    Args:
        results: List of per-complex result dictionaries.
        include_interface: Whether to include interface statistics.
    """
    stats = _aggregate_summary_statistics(results, include_interface)
    total = stats.get('total_complexes', 0)
    if total == 0:
        print("\nDataset Summary:\n  Total complexes: 0 (nothing to summarise)")
        return

    print(f"\nDataset Summary:")
    print(f"  Total complexes: {total}")
    print(f"  Homodimers:      {stats['homodimer_count']} ({100 * stats['homodimer_count'] / total:.1f}%)")
    print(f"  Heterodimers:    {stats['heterodimer_count']} ({100 * stats['heterodimer_count'] / total:.1f}%)")

    # HPC failure-mode diagnostics. These distinguish three orthogonal scopes
    # so a reader cannot confuse the headline tier counts (above) with the
    # strict-calibrated subset (below).
    incomplete_count = stats.get('incomplete_count', 0)
    partial_count = stats.get('partial_count', 0)
    if incomplete_count or partial_count:
        print(f"\nDiagnostics (HPC failure-mode tracking):")
        print(f"  Incomplete rows (no complex_type / quality_tier_v2):           {incomplete_count}")
        print(f"  Partial rows (partial_reason populated, kept in tier counts):  {partial_count}")
        for reason, count in sorted(
            stats.get('partial_reason_counts', {}).items(),
            key=lambda kv: -kv[1],
        ):
            label = reason or "(empty)"
            print(f"    {label}: {count}")
        print(f"  Strict-calibrated rows (composite_is_calibrated=True):         {stats.get('calibrated_count', 0)}")

    print(f"\nQuality Distribution (v1, two-metric baseline - for v1↔v2 comparison only):")
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

    # PAE-specific summary (reports both PAE-only and strict fractions)
    if stats['confident_fractions']:
        print(f"\nInterface PAE:")
        print(f"  Mean PAE-only confident fraction:   {statistics.mean(stats['confident_fractions']):.3f}")
        if stats.get('strict_confident_fractions'):
            print(f"  Mean strict confident fraction:     {statistics.mean(stats['strict_confident_fractions']):.3f}")
        high_conf = sum(1 for f in stats['confident_fractions'] if f > 0.5)
        print(f"  Complexes with >50% PAE-only confident contacts: "
              f"{high_conf} ({100 * high_conf / len(stats['confident_fractions']):.1f}%)")

    # Composite interface confidence score
    if stats['composite_scores']:
        scores = stats['composite_scores']
        print(f"\nComposite Interface Confidence (Phase 4):")
        print(f"  Mean: {statistics.mean(scores):.3f}, "
              f"Median: {statistics.median(scores):.3f}, "
              f"Min: {min(scores):.3f}, Max: {max(scores):.3f}")

    # Quality tier v2 reclassification summary - this is the canonical
    # distribution used by every figure (Figs 1-16) via the
    # `quality_tier_v2` column. The v1 block above is kept only for
    # transparency about what the composite-score reclassification adds.
    if 'v2_high' in stats:
        print(f"\nQuality Distribution (v2, composite-aware - used by all figures):")
        print(f"  High:   {stats['v2_high']} ({100 * stats['v2_high'] / total:.1f}%)")
        print(f"  Medium: {stats['v2_medium']} ({100 * stats['v2_medium'] / total:.1f}%)")
        print(f"  Low:    {stats['v2_low']} ({100 * stats['v2_low'] / total:.1f}%)")
        print(f"  Reclassified vs v1: {stats['v2_upgrades']} upgraded to High, "
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

    # With clustering (sequence cluster annotation and homologous pairs)
    python toolkit.py --dir "D:\\ProteinComplexes" --output results.csv --interface --pae --enrich "D:\\protein-complexes-toolkit\\data\\ppi\\9606.protein.aliases.v12.0.txt" --clustering

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
                             "Used for initial complex analysis. Values >1 enable "
                             "multiprocessing via ProcessPoolExecutor.")
    parser.add_argument("--checkpoint", action="store_true",
                        help="Enable periodic checkpointing (saves progress every "
                             f"{CHECKPOINT_INTERVAL} complexes to <output>{CHECKPOINT_SUFFIX})")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from a previous checkpoint. Implies --checkpoint. "
                             "Already-processed complexes are skipped.")
    parser.add_argument("--skip-existing", metavar="RESULTS_CSV", default=None,
                        help="Append-only incremental mode. Reads complex_name "
                             "from RESULTS_CSV (the historical results.csv — NOT "
                             "a strict-calibrated subset) and processes only "
                             "complexes not present there. Pair with "
                             "--export-interfaces to produce an incremental JSONL. "
                             "Writes a fingerprinted audit snapshot under "
                             "data/complex_manifest_audit/runs/<auto_run_id>/. "
                             "Compatible with --resume but the two flags serve "
                             "different purposes: --skip-existing is an "
                             "append-only cross-run filter against a completed "
                             "historical CSV; --resume is in-flight crash recovery "
                             "from the current run's checkpoint. If both are "
                             "supplied, the toolkit first builds the incremental "
                             "delta from --skip-existing, then removes complexes "
                             "already completed in the current checkpoint.")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="Process at most N complexes per run, taken from the "
                             "alphabetically-sorted post-(--skip-existing) delta. "
                             "Used to chunk large HPC runs into memory-bounded batches. "
                             "The chunk membership is fixed BEFORE --resume filtering, "
                             "so a crashed-and-resumed chunk processes exactly the same "
                             "complexes as a clean run - provided --skip-existing's "
                             "RESULTS_CSV is unchanged between attempts. Combine with "
                             "--skip-existing across successive runs to walk the dataset "
                             "in deterministic chunks. NOTE: --skip-existing must point "
                             "at the unfiltered historical results.csv (NOT a "
                             "strict-calibrated subset), or partial/zero-contact rows "
                             "will be reprocessed every chunk. Omit for unlimited (default).")
    parser.add_argument("--enrich", metavar="ALIASES_PATH",
                        help="Enrich output with gene symbols, protein names, and "
                             "cross-references using a STRING aliases file.")
    parser.add_argument("--databases", metavar="DATA_DIR",
                        help="Tag each complex with its database source(s) by checking "
                             "against STRING, BioGRID, HuRI, and HuMAP. Requires --enrich.")
    parser.add_argument("--string-min-score", type=int, default=700,
                        help="Minimum STRING confidence score for database source matching "
                             "(default: 700). Only used with --databases.")
    parser.add_argument("--no-api", action="store_true",
                        help="Disable STRING API validation fallback. Use only offline "
                             "data files for ID resolution and enrichment.")
    parser.add_argument("--clustering", choices=['string', 'foldseek', 'hybrid'],
                        nargs='?', const='string', default=None,
                        help="Enable sequence clustering analysis. Modes: string (default), "
                             "foldseek (deferred), hybrid (deferred). Reads STRING clusters "
                             "file to identify homologous pairs. Requires --enrich for ID "
                             "resolution.")
    parser.add_argument("--clusters-file", metavar="PATH",
                        help="Path to STRING clusters file. Default: "
                             "data/clusters/9606.clusters.proteins.v12.0.txt. "
                             "Only used with --clustering.")
    parser.add_argument("--variants", metavar="VARIANTS_DIR",
                        nargs='?', const='__default__', default=None,
                        help="Map genetic variants to interface residues using "
                             "UniProt/ClinVar/ExAC databases. Requires --interface "
                             "--pae --enrich. Default directory: data/variants/. "
                             "Optionally specify a custom variants directory path.")
    parser.add_argument("--no-clinvar", action="store_true",
                        help="Skip ClinVar enrichment when using --variants (faster). "
                             "Only UniProt variants and ExAC constraint scores are used.")
    parser.add_argument("--stability", metavar="STABILITY_DIR",
                        nargs='?', const='__default__', default=None,
                        help="Score variant stability using EVE evolutionary predictions. "
                             "Requires --variants. Default: data/stability/")
    parser.add_argument("--protvar", metavar="FOLDX_EXPORT",
                        nargs='?', const='__default__', default=None,
                        help="Score variants with offline AlphaMissense + monomeric FoldX data. "
                             "Requires --variants. "
                             "Default: data/stability/afdb_foldx_export_20250210.csv")
    parser.add_argument("--am-file", metavar="AM_PATH", default=None,
                        help="Path to AlphaMissense_aa_substitutions.tsv. "
                             "Default: data/stability/AlphaMissense_aa_substitutions.tsv")
    parser.add_argument("--disease", metavar="PATHWAYS_DIR",
                        nargs='?', const='__default__', default=None,
                        help="Annotate with disease associations, PTMs, GO terms, "
                             "drug targets from UniProt XML. Requires --enrich. "
                             "Default: data/pathways/")
    parser.add_argument("--pathways", action="store_true",
                        help="Map proteins to Reactome pathways (local files) and "
                             "optionally run STRING API enrichment. Requires --enrich. "
                             "Uses data/pathways/ for local Reactome files. "
                             "STRING API enrichment skipped with --no-api.")
    parser.add_argument("--pymol", action="store_true",
                        help="Generate PyMOL .pml scripts for High-tier complexes. "
                             "Requires --interface --pae. Scripts written to "
                             "{dir}/pymol_scripts/ by default.")
    parser.add_argument("--pymol-output", metavar="DIR", default=None,
                        help="Custom output directory for PyMOL scripts. "
                             "Default: ./pymol_scripts/")
    parser.add_argument("--pymol-render", action="store_true",
                        help="Include ray-tracing + PNG rendering commands in "
                             "generated .pml scripts (for pymol -c batch mode).")
    parser.add_argument("--pymol-min-tier", default="High",
                        choices=["High", "Medium", "Low"],
                        help="Minimum quality tier for PyMOL script generation. "
                             "Default: High.")
    parser.add_argument("--full-pipeline", action="store_true",
                        help="Activate all pipeline phases (A-F) using default "
                             "data paths. Only --dir and optionally --workers "
                             "are required. Validates all data files exist "
                             "before processing starts.")
    return parser


def main() -> None:
    """Run the batch processing pipeline."""
    parser = build_argument_parser()
    args = parser.parse_args()

    # ── Full-pipeline expansion ──────────────────────────────────────
    if args.full_pipeline:
        from data_registry import validate_data_dependencies, get_default_path

        # Set all phase flags to their default-path sentinels
        args.interface = True
        args.pae = True
        args.enrich = get_default_path("string_aliases")
        args.databases = str(Path(get_default_path("string_links")).parent)
        args.clustering = args.clustering or "string"
        if args.variants is None:
            args.variants = "__default__"
        if args.stability is None:
            args.stability = "__default__"
        if args.protvar is None:
            args.protvar = "__default__"
        if args.disease is None:
            args.disease = "__default__"
        args.pathways = True
        args.pymol = True
        args.checkpoint = True

        # Pre-flight: validate all data files exist
        print("Full pipeline mode: validating data dependencies...",
              file=sys.stderr)
        errors = validate_data_dependencies(verbose=True)
        if errors:
            print(f"\nError: {len(errors)} required data file(s) missing. "
                  "Cannot start --full-pipeline.", file=sys.stderr)
            sys.exit(1)
        print(file=sys.stderr)

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

    if args.clustering and not args.enrich:
        print("Error: --clustering requires --enrich for ID resolution", file=sys.stderr)
        sys.exit(1)

    if args.variants is not None:
        # --variants was used (either with or without explicit path)
        if not (args.interface and args.pae and args.enrich):
            print("Error: --variants requires --interface --pae --enrich", file=sys.stderr)
            sys.exit(1)
        # Resolve variants directory path (sentinel '__default__' means --variants without path)
        if args.variants == '__default__':
            from variant_mapper import DEFAULT_VARIANTS_DIR
            args.variants = str(DEFAULT_VARIANTS_DIR)
    # Distinguish "not used" from "used without path" for later checks
    args._variants_enabled = args.variants is not None

    if args.stability is not None:
        if not getattr(args, '_variants_enabled', False):
            print("Error: --stability requires --variants", file=sys.stderr)
            sys.exit(1)
        if args.stability == '__default__':
            from stability_scorer import DEFAULT_STABILITY_DIR
            args.stability = str(DEFAULT_STABILITY_DIR)
    args._stability_enabled = args.stability is not None

    if args.protvar is not None:
        if not getattr(args, '_variants_enabled', False):
            print("Error: --protvar requires --variants", file=sys.stderr)
            sys.exit(1)
        if args.protvar == '__default__':
            from protvar_client import DEFAULT_FOLDX_EXPORT
            args.protvar = str(DEFAULT_FOLDX_EXPORT)
        if not Path(args.protvar).exists():
            print(f"Error: AFDB FoldX export not found: {args.protvar}",
                  file=sys.stderr)
            sys.exit(1)
        # Resolve AlphaMissense file path
        if args.am_file is None:
            from protvar_client import DEFAULT_AM_FILE
            args.am_file = str(DEFAULT_AM_FILE)
        if not Path(args.am_file).exists():
            print(f"Error: AlphaMissense file not found: {args.am_file}",
                  file=sys.stderr)
            sys.exit(1)
    args._protvar_enabled = args.protvar is not None

    if args.disease is not None:
        if not args.enrich:
            print("Error: --disease requires --enrich", file=sys.stderr)
            sys.exit(1)
        if args.disease == '__default__':
            from disease_annotations import DEFAULT_DISEASE_DIR
            args.disease = str(DEFAULT_DISEASE_DIR)
    args._disease_enabled = args.disease is not None

    if args.pathways:
        if not args.enrich:
            print("Error: --pathways requires --enrich", file=sys.stderr)
            sys.exit(1)
    args._pathways_enabled = getattr(args, 'pathways', False)

    if args.pymol:
        if not (args.interface and args.pae):
            print("Error: --pymol requires --interface --pae", file=sys.stderr)
            sys.exit(1)
    args._pymol_enabled = getattr(args, 'pymol', False)

    if args.resume:
        args.checkpoint = True

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer (omit the flag for unlimited)")

    if args.workers < 1:
        print("Error: --workers must be >= 1", file=sys.stderr)
        sys.exit(1)

    if args.workers > 1 and args.verbose:
        print("Note: Verbose per-complex output is suppressed in parallel mode.",
              file=sys.stderr)

    # ── Discover data files (incremental-aware) ─────────────────────────
    #
    # Sequential filter invariant: --skip-existing reads ONLY the historical
    # results CSV; --resume reads ONLY the in-flight checkpoint. They are
    # independent filters applied in order: skip-existing first, resume second.
    #
    # When --skip-existing is active we MUST capture the previous fingerprints
    # from latest/ BEFORE the resolver runs, because the resolver refreshes
    # latest/ as a side effect — reading it after would compare the new
    # manifest to itself and produce an empty changed_existing.tsv.
    print(f"Scanning data directory: {args.dir}")

    skip_existing_active = args.skip_existing is not None
    incremental_run_dir: Path | None = None
    delta_filter: set[str] | None = None  # None = no skip-existing filter
    incremental_delta_info: dict | None = None

    if skip_existing_active:
        audit_root = _resolve_audit_root()
        previous_fingerprints = _load_previous_fingerprints(audit_root)
        previous_run_id = _load_previous_run_id(audit_root)
        discovery = find_paired_data_files(
            args.dir, purpose="incremental", return_audit=True,
        )
        complexes = discovery.complexes
        incremental_run_dir = discovery.run_dir
    else:
        complexes = find_paired_data_files(args.dir)

    print(f"Found {len(complexes)} unique complexes")

    if len(complexes) == 0:
        print("No PDB/PKL files found!")
        sys.exit(1)

    # Compute incremental delta (writes audit TSVs as a side effect).
    if skip_existing_active and incremental_run_dir is not None:
        incremental_delta_info = _compute_incremental_delta(
            complexes,
            previous_fingerprints,
            previous_run_id,
            Path(args.skip_existing),
            incremental_run_dir,
            output_csv=args.output,
            output_jsonl=args.export_interfaces,
        )
        delta_filter = incremental_delta_info["delta"]
        n_delta = len(delta_filter)
        n_already = len(incremental_delta_info["already_processed"])
        n_changed = len(incremental_delta_info["changed_existing"])
        n_missing = len(incremental_delta_info["missing"])
        print(
            f"Incremental mode: delta={n_delta:,} new, "
            f"{n_already:,} already processed, "
            f"{n_changed:,} changed-existing (audit-only), "
            f"{n_missing:,} missing-since-previous",
            file=sys.stderr,
        )
        if n_changed:
            sample_names = [
                row[0] for row in incremental_delta_info["changed_existing"][:5]
            ]
            extra = "..." if n_changed > 5 else ""
            print(
                f"  Warning: {n_changed} complex(es) have fingerprints "
                f"differing from the previous baseline manifest: "
                f"{', '.join(sample_names)}{extra}. They are recorded in "
                f"changed_existing.tsv but NOT processed. If those rows "
                f"matter for downstream figures, run a full re-pipeline "
                f"rather than an incremental update.",
                file=sys.stderr,
            )
    elif skip_existing_active and incremental_run_dir is None:
        # Loose / Test_Data layout: resolver did not produce an audit dir.
        # Honour the historical-CSV filter so the path still works for tests
        # and ad-hoc local runs; no fingerprinted audit is written.
        historical_names = _read_skip_existing_names(Path(args.skip_existing))
        delta_filter = {n for n in complexes if n not in historical_names}
        print(
            f"Incremental mode (loose layout, no audit dir): "
            f"delta={len(delta_filter):,} new",
            file=sys.stderr,
        )

    # Resume from checkpoint if requested (in-flight crash recovery — strictly
    # independent from skip-existing's historical-CSV filter).
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

    # Sequential filter:
    #   1. apply --skip-existing (delta_filter) and sort alphabetically
    #   2. apply --limit to pin chunk membership (deterministic across crashes)
    #   3. apply --resume (completed_names) to drop already-checkpointed rows
    #      from the selected chunk
    #
    # --limit MUST sit between skip-existing and resume; applying it after
    # resume causes chunk-boundary drift on a crashed-and-resumed run.
    eligible_complexes: list[tuple[str, dict]] = []
    for name, paths in sorted(complexes.items()):
        if delta_filter is not None and name not in delta_filter:
            continue
        eligible_complexes.append((name, paths))
    total_eligible = len(eligible_complexes)

    selected_chunk, sorted_complexes = _select_chunk(
        eligible_complexes, args.limit, completed_names,
    )
    skipped_completed = len(selected_chunk) - len(sorted_complexes)

    if args.limit is not None:
        if selected_chunk:
            print(
                f"Limit requested: selected {len(selected_chunk):,} of "
                f"{total_eligible:,} eligible complexes "
                f"(chunk first: {selected_chunk[0][0]}, "
                f"chunk last: {selected_chunk[-1][0]}, "
                f"checkpoint-completed within chunk: {skipped_completed:,}, "
                f"remaining to process: {len(sorted_complexes):,})",
                file=sys.stderr,
            )
        else:
            print(
                "Limit requested: 0 eligible complexes after --skip-existing.",
                file=sys.stderr,
            )

    if not sorted_complexes and resumed_results:
        print("All complexes already processed - writing final output.")
        results = list(resumed_results)
        results.sort(key=lambda r: r.get('complex_name', ''))
    elif not sorted_complexes and skip_existing_active:
        # Empty incremental delta — write header-only CSV + JSONL and exit.
        # Skip the heavy annotation/enrichment phases since there's no data.
        print(
            "Incremental delta is empty — all current complexes are already "
            f"present in {args.skip_existing}. Writing header-only output "
            "and exiting cleanly.",
            file=sys.stderr,
        )
        empty_fieldnames = get_csv_fieldnames(
            include_interface=args.interface,
            include_pae=args.pae,
            include_enrichment=bool(args.enrich),
            include_clustering=args.clustering is not None,
            include_variants=getattr(args, '_variants_enabled', False),
            include_stability=getattr(args, '_stability_enabled', False),
            include_protvar=getattr(args, '_protvar_enabled', False),
            include_disease=getattr(args, '_disease_enabled', False),
            include_pathways=getattr(args, '_pathways_enabled', False),
        )
        with open(args.output, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=empty_fieldnames, extrasaction='ignore')
            writer.writeheader()
        if args.export_interfaces:
            jsonl_path = Path(args.export_interfaces)
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            jsonl_path.write_text("", encoding="utf-8")
        print(f"Empty results written to: {args.output}")
        if args.export_interfaces:
            print(f"Empty interface JSONL written to: {args.export_interfaces}")
        sys.exit(0)
    else:
        # Process complexes (sequential or parallel)
        results = run_batch_parallel(
            sorted_complexes,
            run_interface=args.interface,
            run_interface_pae=args.pae,
            export_interfaces=bool(args.export_interfaces),
            stash_variant_data=getattr(args, '_variants_enabled', False),
            verbose=args.verbose,
            workers=args.workers,
            output_path=args.output,
            enable_checkpoint=args.checkpoint,
            resumed_results=resumed_results,
        )

    # Species annotation (reviewed_human / trembl_human / non_human per chain + complex)
    species_start = time.time()
    from id_mapper import SpeciesClassifier
    print("Annotating species status ...", file=sys.stderr)
    classifier = SpeciesClassifier(verbose=args.verbose)
    annotate_species(results, classifier)
    n_rev = sum(1 for r in results if r.get('species_status') == 'reviewed_human')
    n_tre = sum(1 for r in results if r.get('species_status') == 'trembl_human')
    n_non = sum(1 for r in results if r.get('species_status') == 'non_human')
    print(
        f"  Species split: reviewed_human={n_rev:,}  trembl_human={n_tre:,}  "
        f"non_human={n_non:,}  ({time.time() - species_start:.1f}s)",
        file=sys.stderr,
    )

    # Enrichment (gene symbols, database sources)
    include_enrichment = False
    if args.enrich:
        enrich_start = time.time()
        from id_mapper import IDMapper, build_uniprot_lookup
        print(f"Loading ID mapper from: {args.enrich}", file=sys.stderr)
        mapper = IDMapper(args.enrich, verbose=True, api_fallback=not args.no_api)
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
                api_validate=not args.no_api,
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

        enrich_results(results, lookup, db_pair_sets, db_evidence,
                       mapper=mapper if not args.no_api else None)
        include_enrichment = True
        enrich_elapsed = time.time() - enrich_start
        print(f"Enrichment complete: {len(results)} complexes annotated "
              f"in {enrich_elapsed:.1f}s", file=sys.stderr)

    # Clustering (sequence cluster annotation and homologous pair detection)
    include_clustering = False
    if args.clustering:
        from protein_clustering import (
            validate_clustering_mode, load_clusters, build_cluster_index,
            build_uniprot_cluster_index, build_cluster_to_uniprot,
            annotate_results_with_clustering, enrich_with_homology_scores,
        )
        validate_clustering_mode(args.clustering)

        cluster_start = time.time()
        clusters_df = load_clusters(filepath=args.clusters_file, verbose=True)
        cluster_to_proteins, protein_to_clusters = build_cluster_index(clusters_df)
        print(f"  {len(cluster_to_proteins):,} clusters, "
              f"{len(protein_to_clusters):,} proteins", file=sys.stderr)

        uniprot_index = build_uniprot_cluster_index(
            protein_to_clusters, mapper, verbose=True,
        )
        cluster_to_uniprot = build_cluster_to_uniprot(uniprot_index)

        # Use known interaction pairs for filtering homologous pairs (if databases loaded)
        known = None
        if db_pair_sets:
            known = set()
            for ps in db_pair_sets.values():
                known.update(ps)

        annotate_results_with_clustering(
            results, uniprot_index, cluster_to_uniprot,
            known_pairs=known, verbose=True,
        )

        # Optional: API homology scores
        if not args.no_api:
            enrich_with_homology_scores(
                results, uniprot_index, mapper, verbose=True,
            )

        include_clustering = True
        cluster_elapsed = time.time() - cluster_start
        print(f"Clustering complete: {len(results)} complexes annotated "
              f"in {cluster_elapsed:.1f}s", file=sys.stderr)

        # Free clustering data structures no longer needed
        del clusters_df, cluster_to_proteins, protein_to_clusters
        del uniprot_index, cluster_to_uniprot

    # Free enrichment data structures no longer needed after clustering
    if args.enrich:
        try:
            del dbs
        except NameError:
            pass
        del mapper, lookup
        db_pair_sets = None
        db_evidence = None

    # Variant mapping (structural context, enrichment, gene constraint)
    include_variants = False
    if getattr(args, '_variants_enabled', False):
        from variant_mapper import (
            load_uniprot_variants, load_clinvar_variants,
            load_exac_constraint, build_variant_index,
            enrich_with_clinvar, annotate_results_with_variants,
            UNIPROT_VARIANTS_FILENAME, CLINVAR_VARIANTS_FILENAME,
            EXAC_CONSTRAINT_FILENAME,
        )
        variant_start = time.time()
        variants_dir = Path(args.variants)

        # Collect unique accessions and gene symbols from results
        accessions = set()
        gene_symbols_set = set()
        for row in results:
            accessions.add(row.get('protein_a', ''))
            accessions.add(row.get('protein_b', ''))
            for gs_key in ('gene_symbol_a', 'gene_symbol_b'):
                gs = row.get(gs_key, '')
                if gs:
                    gene_symbols_set.add(gs)
        accessions.discard('')

        # Also try base accessions (strip isoform suffixes)
        from id_mapper import split_isoform
        base_accessions = set()
        for acc in accessions:
            base, _ = split_isoform(acc)
            base_accessions.add(base)
        all_accessions = frozenset(accessions | base_accessions)

        print(f"Loading variant databases from: {variants_dir}", file=sys.stderr)
        print(f"  Searching for variants in {len(all_accessions)} accessions, "
              f"{len(gene_symbols_set)} gene symbols", file=sys.stderr)

        # Load UniProt variants (chunked streaming)
        uniprot_path = variants_dir / UNIPROT_VARIANTS_FILENAME
        variants_df = load_uniprot_variants(
            uniprot_path, all_accessions, verbose=True,
        )
        variant_idx = build_variant_index(variants_df)
        print(f"  Variant index: {sum(len(v) for v in variant_idx.values()):,} variants "
              f"across {len(variant_idx)} proteins", file=sys.stderr)

        # ClinVar enrichment (optional)
        if not args.no_clinvar:
            clinvar_path = variants_dir / CLINVAR_VARIANTS_FILENAME
            if clinvar_path.exists():
                # Collect rsIDs from variant index
                rsids = frozenset(
                    str(v['rsid']) for variants in variant_idx.values()
                    for v in variants if v.get('rsid') and str(v['rsid']) != 'nan'
                )
                clinvar_df = load_clinvar_variants(
                    clinvar_path, rsids=rsids, verbose=True,
                )
                enrich_with_clinvar(variant_idx, clinvar_df, verbose=True)
            else:
                print(f"  ClinVar file not found: {clinvar_path}", file=sys.stderr)

        # ExAC gene constraint
        exac_path = variants_dir / EXAC_CONSTRAINT_FILENAME
        if exac_path.exists():
            exac_df = load_exac_constraint(exac_path, gene_symbols=frozenset(gene_symbols_set))
            print(f"  ExAC constraint: {len(exac_df)} genes loaded", file=sys.stderr)
        else:
            exac_df = pd.DataFrame()
            print(f"  ExAC file not found: {exac_path}", file=sys.stderr)

        # Build gene symbol lookup from results
        gene_lookup: dict[str, str] = {}
        for row in results:
            pa = row.get('protein_a', '')
            pb = row.get('protein_b', '')
            ga = row.get('gene_symbol_a', '')
            gb = row.get('gene_symbol_b', '')
            if pa and ga:
                gene_lookup[pa] = ga
            if pb and gb:
                gene_lookup[pb] = gb

        # Annotate results with variant data (SASA already computed in workers)
        annotate_results_with_variants(
            results, variant_idx, exac_df, gene_lookup,
            verbose=True,
        )

        include_variants = True
        variant_elapsed = time.time() - variant_start
        print(f"Variant mapping complete: {len(results)} complexes annotated "
              f"in {variant_elapsed:.1f}s", file=sys.stderr)

        # Free variant data structures no longer needed
        del variants_df, variant_idx, gene_lookup
        try:
            del clinvar_df
        except NameError:
            pass
        del exac_df

    # Stability scoring (EVE evolutionary predictions)
    include_stability = False
    if getattr(args, '_stability_enabled', False):
        from stability_scorer import (
            load_eve_entry_name_map, build_eve_index,
            annotate_results_with_stability, EVE_IDMAPPING_FILENAME,
        )
        stability_start = time.time()
        stability_dir = Path(args.stability)

        # Load accession → entry name mapping
        map_path = stability_dir / EVE_IDMAPPING_FILENAME
        print(f"Loading EVE ID mapping from: {map_path}", file=sys.stderr)
        acc_to_entry = load_eve_entry_name_map(map_path)
        print(f"  Mapped {len(acc_to_entry):,} accessions to entry names", file=sys.stderr)

        # Build EVE index (lazy - only loads CSVs for pipeline accessions)
        # all_accessions was built by the --variants block (which --stability requires)
        eve_dir = stability_dir / "EVE_all_data"
        eve_index = build_eve_index(
            eve_dir, all_accessions, acc_to_entry, verbose=True,
        )
        print(f"  EVE index: {len(eve_index)} proteins with scores loaded", file=sys.stderr)

        # Annotate results with EVE stability scores
        annotate_results_with_stability(
            results, eve_index, acc_to_entry, verbose=True,
        )

        include_stability = True
        stability_elapsed = time.time() - stability_start
        print(f"Stability scoring complete: {len(results)} complexes annotated "
              f"in {stability_elapsed:.1f}s", file=sys.stderr)

        # Free EVE data structures no longer needed
        del acc_to_entry, eve_index

    # Offline AlphaMissense + monomeric FoldX scoring
    include_protvar = False
    protvar_index = None
    if getattr(args, '_protvar_enabled', False):
        from protvar_client import (
            build_protvar_index, annotate_results_with_protvar,
            _parse_variant_details_for_protvar,
        )
        protvar_start = time.time()

        # Collect accessions and variant positions from results
        all_protvar_accessions: set[str] = set()
        all_variant_positions: dict[str, set[int]] = {}

        for row in results:
            for suffix in ('a', 'b'):
                acc = row.get(f'protein_{suffix}', '')
                details = row.get(f'variant_details_{suffix}', '')
                if acc and details:
                    base = acc.split('-')[0] if '-' in acc else acc
                    all_protvar_accessions.add(base)
                    for _ref, pos, _alt in _parse_variant_details_for_protvar(details):
                        all_variant_positions.setdefault(base, set()).add(pos)

        n_proteins = len(all_protvar_accessions)
        n_variant_pos = sum(len(ps) for ps in all_variant_positions.values())
        print(f"Loading offline scores for {n_proteins} proteins "
              f"({n_variant_pos} variant positions)...", file=sys.stderr)

        protvar_index = build_protvar_index(
            accessions=all_protvar_accessions,
            variant_positions=all_variant_positions,
            foldx_path=args.protvar,
            am_path=args.am_file,
            verbose=True,
        )

        annotate_results_with_protvar(results, protvar_index, verbose=True)

        include_protvar = True
        protvar_elapsed = time.time() - protvar_start
        print(f"Offline scoring complete: {len(results)} complexes "
              f"annotated in {protvar_elapsed:.1f}s", file=sys.stderr)

        # Free temporary ProtVar collection structures
        del all_protvar_accessions, all_variant_positions

    # Disease annotations (UniProt disease, PTM, GO, drug target)
    include_disease = False
    if getattr(args, '_disease_enabled', False):
        from disease_annotations import (
            load_uniprot_annotations, annotate_results_with_disease,
            UNIPROT_XML_FILENAME,
        )
        disease_start = time.time()
        disease_dir = Path(args.disease)
        xml_path = disease_dir / UNIPROT_XML_FILENAME

        # Collect unique accessions from results
        disease_accessions: set[str] = set()
        for row in results:
            disease_accessions.add(row.get('protein_a', ''))
            disease_accessions.add(row.get('protein_b', ''))
        disease_accessions.discard('')
        # Also add base accessions (strip isoform suffixes)
        base_disease_acc: set[str] = set()
        for acc in disease_accessions:
            base_disease_acc.add(acc.split('-')[0] if '-' in acc else acc)
        all_disease_accessions = frozenset(disease_accessions | base_disease_acc)

        print(f"Loading disease annotations from: {xml_path}", file=sys.stderr)
        annotation_index = load_uniprot_annotations(
            xml_path, all_disease_accessions, verbose=True,
        )

        annotate_results_with_disease(
            results, annotation_index,
            api_fallback=not args.no_api,
            verbose=True,
        )

        include_disease = True
        disease_elapsed = time.time() - disease_start
        print(f"Disease annotation complete: {len(results)} complexes "
              f"annotated in {disease_elapsed:.1f}s", file=sys.stderr)

        del annotation_index

    # Pathway mapping (Reactome local + optional STRING API enrichment)
    include_pathways = False
    if getattr(args, '_pathways_enabled', False):
        from pathway_network import (
            load_reactome_mappings, compute_pathway_quality_stats,
            annotate_results_with_pathways, run_ppi_enrichment,
            run_string_enrichment, build_interaction_network,
            compute_network_stats, invert_reactome_index,
            run_per_pathway_ppi_enrichment,
            REACTOME_MAPPINGS_FILENAME, DEFAULT_PATHWAYS_DIR,
            _HAS_NETWORKX,
        )
        pathway_start = time.time()

        # Determine pathways directory (shared with --disease or default)
        pathways_dir = Path(args.disease) if getattr(args, '_disease_enabled', False) \
            else DEFAULT_PATHWAYS_DIR

        # Collect unique accessions from annotatable (human) rows only - STRING
        # enrichment, per-pathway PPI, and network stats are species-specific
        # so non-human accessions would only add noise.
        pathway_accessions: set[str] = set()
        for row in results:
            if not is_annotatable(row):
                continue
            pathway_accessions.add(row.get('protein_a', ''))
            pathway_accessions.add(row.get('protein_b', ''))
        pathway_accessions.discard('')
        base_pathway_acc: set[str] = set()
        for acc in pathway_accessions:
            base_pathway_acc.add(acc.split('-')[0] if '-' in acc else acc)
        all_pathway_accessions = frozenset(pathway_accessions | base_pathway_acc)

        # Load Reactome local mappings
        reactome_path = pathways_dir / REACTOME_MAPPINGS_FILENAME
        if reactome_path.exists():
            reactome_index = load_reactome_mappings(
                reactome_path, all_pathway_accessions, verbose=True,
            )
        else:
            print(f"Warning: Reactome mappings not found: {reactome_path}",
                  file=sys.stderr)
            reactome_index = {}

        # Compute pathway quality statistics
        pathway_stats = compute_pathway_quality_stats(results, reactome_index)

        # Optional: STRING API enrichment (only if --no-api not set)
        # Follows offline-first + API validation pattern: local Reactome provides
        # the base, STRING enrichment validates with p-values / FDR
        pathway_ppi_stats = None
        enrichment_df = None
        if not args.no_api:
            gene_symbols = []
            for row in results:
                if not is_annotatable(row):
                    continue
                for gs_key in ('gene_symbol_a', 'gene_symbol_b'):
                    gs = row.get(gs_key, '')
                    if gs:
                        gene_symbols.append(gs)
            gene_symbols = list(set(gene_symbols))
            if gene_symbols:
                enrichment_df = run_string_enrichment(gene_symbols, verbose=True)

            # Per-pathway PPI enrichment (replaces global enrichment)
            if reactome_index:
                pathway_proteins = invert_reactome_index(reactome_index)
                # Collect all shared pathway IDs across human complexes only
                shared_pathway_ids: set[str] = set()
                for row in results:
                    if not is_annotatable(row):
                        continue
                    pa = row.get('protein_a', '')
                    pb = row.get('protein_b', '')
                    ba = pa.split('-')[0] if '-' in pa else pa
                    bb = pb.split('-')[0] if '-' in pb else pb
                    pids_a = {m['pathway_id'] for m in
                              (reactome_index.get(pa, []) or reactome_index.get(ba, []))}
                    pids_b = {m['pathway_id'] for m in
                              (reactome_index.get(pb, []) or reactome_index.get(bb, []))}
                    shared_pathway_ids.update(pids_a & pids_b)
                if shared_pathway_ids:
                    pathway_ppi_stats = run_per_pathway_ppi_enrichment(
                        pathway_proteins, shared_pathway_ids, verbose=True,
                    )

        # Optional: build network and compute node-level stats
        network_stats = None
        if _HAS_NETWORKX and len(results) > 0:
            G = build_interaction_network(results)
            if G.number_of_nodes() > 0:
                network_stats = compute_network_stats(G)
                print(f"  Network: {G.number_of_nodes()} nodes, "
                      f"{G.number_of_edges()} edges", file=sys.stderr)
            del G

        annotate_results_with_pathways(
            results, reactome_index,
            pathway_stats=pathway_stats,
            pathway_ppi_stats=pathway_ppi_stats,
            network_stats=network_stats,
            enrichment_df=enrichment_df,
            verbose=True,
        )

        include_pathways = True
        pathway_elapsed = time.time() - pathway_start
        print(f"Pathway annotation complete: {len(results)} complexes "
              f"annotated in {pathway_elapsed:.1f}s", file=sys.stderr)

        del reactome_index, pathway_stats

    # Write CSV output
    print(f"Writing CSV to {args.output}...", file=sys.stderr)
    write_results_csv(
        results, args.output,
        include_interface=args.interface,
        include_pae=args.pae,
        include_enrichment=include_enrichment,
        include_clustering=include_clustering,
        include_variants=include_variants,
        include_stability=include_stability,
        include_protvar=include_protvar,
        include_disease=include_disease,
        include_pathways=include_pathways,
    )

    print(f"\n{'=' * 60}")
    print(f"Results saved to: {args.output}")

    # Write JSONL interface export if requested
    if args.export_interfaces:
        exported_count = write_interface_exports(results, args.export_interfaces)
        print(f"Interface export: {args.export_interfaces} ({exported_count} complexes)")

    # Generate PyMOL scripts if requested
    if args._pymol_enabled:
        import time as _time
        from pymol_scripts import generate_pymol_scripts_for_results
        _pymol_start = _time.time()
        pymol_dir = args.pymol_output or os.path.join(os.getcwd(), "pymol_scripts")
        n_pymol = generate_pymol_scripts_for_results(
            results, pdb_dir=args.dir, output_dir=pymol_dir,
            min_tier=args.pymol_min_tier,
            include_variants=getattr(args, '_variants_enabled', False),
            render_png=args.pymol_render, verbose=True,
        )
        _pymol_elapsed = _time.time() - _pymol_start
        print(f"PyMOL scripts: {n_pymol} .pml files in {pymol_dir} "
              f"({_pymol_elapsed:.1f}s)")

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