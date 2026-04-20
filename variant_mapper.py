"""
Variant mapping to AlphaFold2-predicted protein-protein complex structures.
Maps genetic variants from UniProt/ClinVar/ExAC databases onto predicted complex structures. 
Classifies each variant by structural context (interface_core, interface_rim, surface_non_interface, buried_core) and computes enrichment of pathogenic variants at interaction interfaces.

Architecture:
    - Offline-first: reads flat variant database files as primary data
    - Chunked streaming for large files (UniProt 33M rows, ClinVar 8.9M rows)
    - Integrates into toolkit.py via --variants flag
    - Standalone CLI for variant analysis on pre-computed JSONL exports

Data sources:
    - homo_sapiens_variation.txt (UniProt protein-level variants, HGVS format)
    - variant_summary.txt (ClinVar clinical significance and review quality)
    - forweb_cleaned_exac_r03_march16_z_data_pLI_CNV-final.txt (ExAC gene-level constraint)

Usage (standalone):
    python variant_mapper.py summary --variants-dir data/variants
    python variant_mapper.py lookup --variants-dir data/variants --protein P24534
    python variant_mapper.py map --interfaces interfaces.jsonl --pdb-dir DIR --variants-dir data/variants --output variant_analysis.csv

Usage (via toolkit.py):
    python toolkit.py --dir DIR --output results.csv --interface --pae --enrich ALIASES --variants
    python toolkit.py --dir DIR --output results.csv --interface --pae --enrich ALIASES --variants data/variants --no-clinvar
"""

import argparse
import csv
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Optional, Union
import numpy as np
import pandas as pd

try:
    from Bio.PDB import PDBParser, ShrakeRupley
    _HAS_BIOPYTHON = True
except ImportError:
    _HAS_BIOPYTHON = False

try:
    import biotite.structure as _bt_struc
    import biotite.structure.io.pdb as _bt_pdb_io
    _HAS_BIOTITE = True
except ImportError:
    _HAS_BIOTITE = False

#--------------------------------Constants---------------------------------------------

# Default file paths
DEFAULT_VARIANTS_DIR = Path(__file__).parent / "data" / "variants"
UNIPROT_VARIANTS_FILENAME = "homo_sapiens_variation.txt"
CLINVAR_VARIANTS_FILENAME = "variant_summary.txt"
EXAC_CONSTRAINT_FILENAME = "forweb_cleaned_exac_r03_march16_z_data_pLI_CNV-final.txt"

# UniProt file format: header at line 162 (0-indexed), separator at 163 and data from 164
UNIPROT_SKIP_ROWS = 164  # skip header block + column header + separator
UNIPROT_COLUMN_NAMES = [
    'gene_name', 'accession', 'variant_aa_change', 'source_db_id',
    'consequence_type', 'clinical_significance', 'phenotype',
    'phenotype_source', 'cytogenetic_band', 'chromosome_coordinate',
    'ensembl_gene_id', 'ensembl_transcript_id', 'ensembl_translation_id',
    'evidence',
]
# Columns to actually read (skip phenotype_source, cytogenetic_band, coordinates, ensembl IDs)
UNIPROT_USE_COLUMNS = [0, 1, 2, 3, 4, 5, 6, 13]
UNIPROT_USE_COLUMN_NAMES = [
    'gene_name', 'accession', 'variant_aa_change', 'source_db_id',
    'consequence_type', 'clinical_significance', 'phenotype', 'evidence',
]

# Consequence types to retain (protein-altering)
RELEVANT_CONSEQUENCES = frozenset({
    'missense variant', 'stop gained', 'stop lost',
    'frameshift variant', 'inframe deletion', 'inframe insertion',
    'initiator codon variant',
})

# Structural context distance thresholds (Angstroms, CB-CB distance)
INTERFACE_CORE_DISTANCE = 4.0   # < 4A: variant at cross-chain contact
INTERFACE_RIM_DISTANCE = 8.0    # 4-8A: variant near interface periphery
# > 8A from any interface residue: surface_non_interface or buried_core

# Structural context labels
CONTEXT_INTERFACE_CORE = 'interface_core'
CONTEXT_INTERFACE_RIM = 'interface_rim'
CONTEXT_SURFACE = 'surface_non_interface'
CONTEXT_BURIED = 'buried_core'
CONTEXT_UNMAPPED = 'unmapped'

# SASA classification threshold (relative solvent accessibility)
SASA_BURIED_THRESHOLD = 0.25  # RSA < 25% = buried

# Standard Gly-X-Gly max ASA values (Tien et al (PMC3836772), 2013 - Table 1)
# Used for normalising absolute SASA to relative SASA
MAX_ASA = {
    'ALA': 129.0, 'ARG': 274.0, 'ASN': 195.0, 'ASP': 193.0, 'CYS': 167.0,
    'GLN': 225.0, 'GLU': 223.0, 'GLY': 104.0, 'HIS': 224.0, 'ILE': 197.0,
    'LEU': 201.0, 'LYS': 236.0, 'MET': 224.0, 'PHE': 240.0, 'PRO': 159.0,
    'SER': 155.0, 'THR': 172.0, 'TRP': 285.0, 'TYR': 263.0, 'VAL': 174.0,
}
MAX_ASA_DEFAULT = 200.0  # fallback for non-standard residues

# Chunked parsing size
CHUNK_SIZE = 500_000 

# HGVS amino acid change pattern: p.Lys2Glu, p.Arg123Ter, p.Met1Val
HGVS_PATTERN = re.compile(r'p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2,3})')

# Display limit for variant details in CSV cells
VARIANT_DETAILS_DISPLAY_LIMIT = 20

# CSV columns added when --variants is used
CSV_FIELDNAMES_VARIANTS = [
    'n_variants_a', 'n_variants_b',
    'n_interface_variants_a', 'n_interface_variants_b',
    'n_pathogenic_interface_variants',
    'interface_variant_enrichment',
    'variant_details_a', 'variant_details_b',
    'gene_constraint_pli_a', 'gene_constraint_pli_b',
    'gene_constraint_mis_z_a', 'gene_constraint_mis_z_b',
]

# Three-letter to one-letter amino acid codes (import from toolkit if available)
try:
    from toolkit import THREE_TO_ONE as _TOOLKIT_THREE_TO_ONE
    # Extend with Ter (stop codon) which toolkit doesn't include
    THREE_TO_ONE = dict(_TOOLKIT_THREE_TO_ONE)
    THREE_TO_ONE['TER'] = '*'
except ImportError:
    THREE_TO_ONE = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
        'TER': '*',
    }

#-------------------------------------Section 1: Parsing--------------------------------------------

def parse_hgvs_position(hgvs: str) -> Optional[tuple[str, int, str]]:
    """Extract amino acid change from HGVS protein notation.
    Parses HGVS format like 'p.Lys2Glu' into (ref_aa_1letter, position, alt_aa_1letter). Supports standard amino acids and Ter (stop codon).
    Args:
        hgvs: HGVS protein notation string (e.g. 'p.Lys2Glu', 'p.Arg123Ter').
    Returns:
        Tuple of (ref_aa, position, alt_aa) where amino acids are 1-letter codes and position is the integer residue number. Returns None if unparseable.
    """
    if not hgvs or not isinstance(hgvs, str):
        return None
    match = HGVS_PATTERN.match(hgvs.strip())
    if not match:
        return None

    ref_3 = match.group(1).upper()
    position = int(match.group(2))
    alt_3 = match.group(3)

    # Handle 'Ter' which is 3 chars but uppercase differs
    if alt_3.lower() == 'ter':
        alt_3 = 'TER'
    else:
        alt_3 = alt_3.upper()

    ref_1 = THREE_TO_ONE.get(ref_3)
    alt_1 = THREE_TO_ONE.get(alt_3)

    if ref_1 is None or alt_1 is None:
        return None

    return (ref_1, position, alt_1)

def load_uniprot_variants(filepath: Union[str, Path], accessions: frozenset[str], consequence_types: frozenset[str] = RELEVANT_CONSEQUENCES, chunk_size: int = CHUNK_SIZE, verbose: bool = False) -> pd.DataFrame:
    """Load and filter UniProt variant file using chunked streaming.
    Reads the large homo_sapiens_variation.txt file in chunks, retaining only rows for the specified UniProt accessions and consequence types.
    Parses HGVS notation to extract integer positions and amino acid changes.
    Args:
        filepath: Path to homo_sapiens_variation.txt.
        accessions: Set of UniProt accessions to retain (e.g. {'P24534', 'A0A0B4J2C3'}).
        consequence_types: Consequence types to retain. Defaults to RELEVANT_CONSEQUENCES.
        chunk_size: Number of rows per chunk for streaming.
        verbose: Print progress to stderr.
    Returns:
        DataFrame with columns: accession, position, ref_aa, alt_aa, rsid, consequence, clinical_significance, phenotype, evidence.
        Deduplicated by (accession, position, alt_aa).
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"UniProt variants file not found: {filepath}")
    if not accessions:
        return pd.DataFrame(columns=['accession', 'position', 'ref_aa', 'alt_aa', 'rsid', 'consequence', 'clinical_significance', 'phenotype', 'evidence'])

    # Determine how many lines to skip: find the separator line (starts with ___)
    skip_rows = _detect_uniprot_header_end(filepath)

    if verbose:
        print(f"  Loading UniProt variants from: {filepath.name}", file=sys.stderr)
        print(f"  Filtering for {len(accessions)} accessions, "
              f"skipping {skip_rows} header lines", file=sys.stderr)

    chunks_collected = []
    total_rows_scanned = 0
    total_rows_kept = 0

    reader = pd.read_csv(
        filepath, sep='\t', header=None, skiprows=skip_rows,
        names=UNIPROT_COLUMN_NAMES, usecols=UNIPROT_USE_COLUMNS,
        dtype=str, na_values=['-'], encoding='utf-8',
        encoding_errors='replace',
        chunksize=chunk_size, on_bad_lines='skip',
    )

    for chunk in reader:
        total_rows_scanned += len(chunk)

        # Filter by accession
        mask = chunk['accession'].isin(accessions)
        filtered = chunk[mask].copy()
        if filtered.empty:
            continue

        # Filter by consequence type
        if consequence_types:
            mask_cons = filtered['consequence_type'].isin(consequence_types)
            filtered = filtered[mask_cons]

        if filtered.empty:
            continue

        total_rows_kept += len(filtered)
        chunks_collected.append(filtered)

        if verbose and total_rows_scanned % (chunk_size * 10) == 0:
            print(f"    Scanned {total_rows_scanned:,} rows, "
                  f"kept {total_rows_kept:,}...", file=sys.stderr)

    if verbose:
        print(f"  Scanned {total_rows_scanned:,} rows total, "
              f"kept {total_rows_kept:,} matching variants", file=sys.stderr)

    if not chunks_collected:
        return pd.DataFrame(columns=['accession', 'position', 'ref_aa', 'alt_aa', 'rsid', 'consequence', 'clinical_significance', 'phenotype', 'evidence'])
    
    df = pd.concat(chunks_collected, ignore_index=True)

    # Parse HGVS to extract position and amino acid changes
    parsed = df['variant_aa_change'].apply(parse_hgvs_position)
    valid_mask = parsed.notna()
    df = df[valid_mask].copy()
    parsed = parsed[valid_mask]

    df['ref_aa'] = parsed.apply(lambda x: x[0])
    df['position'] = parsed.apply(lambda x: x[1]).astype(int)
    df['alt_aa'] = parsed.apply(lambda x: x[2])

    # Normalise column names
    df = df.rename(columns={'source_db_id': 'rsid','consequence_type': 'consequence'})

    # Select and order output columns
    out_cols = ['accession', 'position', 'ref_aa', 'alt_aa', 'rsid', 'consequence', 'clinical_significance', 'phenotype', 'evidence']
    df = df[[c for c in out_cols if c in df.columns]]

    # Deduplicate by (accession, position, alt_aa) - same variant from multiple Ensembl transcripts appears as duplicate rows
    df = df.drop_duplicates(subset=['accession', 'position', 'alt_aa'], keep='first')
    df = df.reset_index(drop=True)

    if verbose:
        print(f"  After dedup: {len(df):,} unique variants", file=sys.stderr)
    return df

def _detect_uniprot_header_end(filepath: Union[str, Path]) -> int:
    """Detect the number of header lines to skip in UniProt variants file.
    Scans the first 200 lines looking for the separator line (starts with '___'). Data starts on the line after the separator.
    Args:
        filepath: Path to UniProt variants file.
    Returns:
        Number of lines to skip (header + column header + separator).
    """
    filepath = Path(filepath)
    with open(filepath, encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            if i > 200:
                break
            if line.startswith('___'):
                return i + 1  # skip the separator line too
    # Fallback for test files with shorter headers
    return UNIPROT_SKIP_ROWS

def load_clinvar_variants(filepath: Union[str, Path], rsids: Optional[frozenset[str]] = None, gene_symbols: Optional[frozenset[str]] = None, chunk_size: int = CHUNK_SIZE, verbose: bool = False) -> pd.DataFrame:
    """Load and filter ClinVar variant summary file using chunked streaming.
    Reads variant_summary.txt, filters to GRCh38 assembly (avoiding doubled rows), and optionally filters by rsID or gene symbol.
    Args:
        filepath: Path to variant_summary.txt.
        rsids: Set of rsIDs to filter by (e.g. {'rs100000005'}). Optional.
        gene_symbols: Set of gene symbols to filter by. Optional.
        chunk_size: Number of rows per chunk for streaming.
        verbose: Print progress to stderr.
    Returns:
        DataFrame with columns: rsid, gene_symbol, clinvar_significance,
        review_status, n_submitters, phenotype_list, origin.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"ClinVar file not found: {filepath}")

    if verbose:
        print(f"  Loading ClinVar from: {filepath.name}", file=sys.stderr)

    use_cols = ['#AlleleID', 'GeneSymbol', 'ClinicalSignificance', 'RS# (dbSNP)', 'ReviewStatus', 'NumberSubmitters', 'PhenotypeList', 'Origin', 'Assembly']
    chunks_collected = []
    total_rows_scanned = 0

    reader = pd.read_csv(
        filepath, sep='\t', usecols=use_cols, dtype=str,
        na_values=['-'], encoding='utf-8', encoding_errors='replace',
        chunksize=chunk_size, on_bad_lines='skip',
    )

    for chunk in reader:
        total_rows_scanned += len(chunk)

        # Filter to GRCh38 only (ClinVar has rows for both assemblies)
        mask = chunk['Assembly'] == 'GRCh38'
        filtered = chunk[mask].copy()

        if filtered.empty:
            continue

        # Filter by rsID if provided
        if rsids is not None:
            # ClinVar RS# column has just the number; our rsids have 'rs' prefix
            # Handle both formats
            filtered['_rs_key'] = 'rs' + filtered['RS# (dbSNP)'].astype(str)
            mask_rs = filtered['_rs_key'].isin(rsids) | filtered['RS# (dbSNP)'].isin(rsids)
            filtered = filtered[mask_rs].copy()

        # Filter by gene symbol if provided
        if gene_symbols is not None and not filtered.empty:
            mask_gene = filtered['GeneSymbol'].isin(gene_symbols)
            filtered = filtered[mask_gene]

        if filtered.empty:
            continue

        chunks_collected.append(filtered)

    if not chunks_collected:
        return pd.DataFrame(columns=['rsid', 'gene_symbol', 'clinvar_significance', 'review_status', 'n_submitters', 'phenotype_list', 'origin'])

    df = pd.concat(chunks_collected, ignore_index=True)

    # Normalise column names
    df = df.rename(columns={
        'RS# (dbSNP)': 'rsid',
        'GeneSymbol': 'gene_symbol',
        'ClinicalSignificance': 'clinvar_significance',
        'ReviewStatus': 'review_status',
        'NumberSubmitters': 'n_submitters',
        'PhenotypeList': 'phenotype_list',
        'Origin': 'origin',
    })

    # Prefix rsid with 'rs' if not already
    mask_numeric = df['rsid'].str.match(r'^\d+$', na=False)
    df.loc[mask_numeric, 'rsid'] = 'rs' + df.loc[mask_numeric, 'rsid']

    # Select output columns
    out_cols = ['rsid', 'gene_symbol', 'clinvar_significance', 'review_status', 'n_submitters', 'phenotype_list', 'origin']
    df = df[[c for c in out_cols if c in df.columns]]

    # Drop internal columns
    if '_rs_key' in df.columns:
        df = df.drop(columns=['_rs_key'])

    if verbose:
        print(f"  ClinVar: {len(df):,} GRCh38 variants loaded "
              f"(scanned {total_rows_scanned:,} rows)", file=sys.stderr)
    return df

def load_exac_constraint(filepath: Union[str, Path], gene_symbols: Optional[frozenset[str]] = None) -> pd.DataFrame:
    """Load ExAC gene-level constraint scores.
    Reads the small ExAC pLI/mis_z file directly (not chunked).
    Args:
        filepath: Path to ExAC constraint file.
        gene_symbols: Optional set of gene symbols to filter by.
    Returns:
        DataFrame with columns: gene, pLI, mis_z, lof_z, syn_z.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"ExAC constraint file not found: {filepath}")
    df = pd.read_csv(filepath, sep='\t', dtype={'gene': str})
    if gene_symbols is not None:
        df = df[df['gene'].isin(gene_symbols)]

    out_cols = ['gene', 'pLI', 'mis_z', 'lof_z', 'syn_z']
    df = df[[c for c in out_cols if c in df.columns]]
    df = df.reset_index(drop=True)
    return df

def build_variant_index(variants_df: pd.DataFrame) -> dict[str, list[dict]]:
    """Build a fast lookup index from parsed variants DataFrame.
    Groups variants by UniProt accession for O(1) per-protein lookup.
    Args:
        variants_df: DataFrame from load_uniprot_variants().
    Returns:
        Dict mapping accession to list of variant dicts, each containing: position, ref_aa, alt_aa, rsid, consequence, clinical_significance, phenotype, evidence.
    """
    index: dict[str, list[dict]] = {}
    for _, row in variants_df.iterrows():
        acc = row.get('accession', '')
        if not acc:
            continue
        variant = {
            'position': int(row['position']),
            'ref_aa': str(row.get('ref_aa', '')),
            'alt_aa': str(row.get('alt_aa', '')),
            'rsid': str(row.get('rsid', '')),
            'consequence': str(row.get('consequence', '')),
            'clinical_significance': str(row.get('clinical_significance', '')),
            'phenotype': str(row.get('phenotype', '')),
            'evidence': str(row.get('evidence', '')),
        }
        if acc not in index:
            index[acc] = []
        index[acc].append(variant)
    return index

def enrich_with_clinvar(variant_index: dict[str, list[dict]], clinvar_df: pd.DataFrame, verbose: bool = False) -> None:
    """Enrich variant index with ClinVar review details (in-place).
    Merges ClinVar review_status and n_submitters onto matching variants by rsID. Also upgrades clinical_significance if ClinVar provides a more specific classification.
    Args:
        variant_index: Dict from build_variant_index(), modified in-place.
        clinvar_df: DataFrame from load_clinvar_variants().
        verbose: Print progress to stderr.
    """
    if clinvar_df.empty:
        return

    # Build rsID -> ClinVar record lookup
    clinvar_lookup: dict[str, dict] = {}
    for _, row in clinvar_df.iterrows():
        rsid = str(row.get('rsid', ''))
        if rsid and rsid != 'nan':
            clinvar_lookup[rsid] = {
                'clinvar_significance': str(row.get('clinvar_significance', '')),
                'clinvar_review_status': str(row.get('review_status', '')),
                'clinvar_n_submitters': str(row.get('n_submitters', '')),
            }

    enriched_count = 0
    for _, variants in variant_index.items():
        for var in variants:
            rsid = var.get('rsid', '')
            if rsid in clinvar_lookup:
                var.update(clinvar_lookup[rsid])
                enriched_count += 1
    if verbose:
        print(f"  ClinVar enrichment: {enriched_count:,} variants matched "
              f"from {len(clinvar_lookup):,} ClinVar records", file=sys.stderr)

#-------------------------------Section 2: SASA Computation----------------------------------------

def compute_residue_sasa(pdb_path: Union[str, Path], chain_id: str) -> dict[int, float]:
    """Compute per-residue relative solvent accessibility (RSA) for a chain.
    Uses biotite (Cython-accelerated) when available, falling back to BioPython's Shrake-Rupley algorithm. 
    RSA is normalised by Gly-X-Gly max ASA values (Tien et al. 2013).
    Args:
        pdb_path: Path to PDB file.
        chain_id: Chain identifier (e.g. 'A', 'B').
    Returns:
        Dict mapping PDB residue number to RSA (0.0 = fully buried, 1.0+ = fully exposed).
    """
    # Use biotite for ~10x speedup when available
    if _HAS_BIOTITE:
        # Compute both chains but return only the requested one
        # (biotite parses the whole model anyway)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message='.*elements were guessed from atom name.*',
                category=UserWarning,
            )
            pdb_file = _bt_pdb_io.PDBFile.read(str(pdb_path))
            structure = pdb_file.get_structure(model=1)
        structure = structure[_bt_struc.filter_amino_acids(structure)]
        atom_sasa = _bt_struc.sasa(structure, vdw_radii='ProtOr', point_number=30)

        chain_mask = structure.chain_id == chain_id
        chain_atoms = structure[chain_mask]
        chain_sasa_vals = atom_sasa[chain_mask]

        sasa_map: dict[int, float] = {}
        for res_num in np.unique(chain_atoms.res_id):
            res_mask = chain_atoms.res_id == res_num
            abs_sasa = float(chain_sasa_vals[res_mask].sum())
            res_name = str(chain_atoms.res_name[res_mask][0])
            max_sasa = MAX_ASA.get(res_name, MAX_ASA_DEFAULT)
            sasa_map[int(res_num)] = abs_sasa / max_sasa if max_sasa > 0 else 0.0
        return sasa_map

    if not _HAS_BIOPYTHON:
        raise ImportError(
            "SASA computation requires biotite (recommended) or biopython: "
            "pip install biotite  OR  pip install biopython"
        )

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('complex', str(pdb_path))

    sr = ShrakeRupley()
    sr.compute(structure[0], level='R')

    sasa_map: dict[int, float] = {}
    for residue in structure[0][chain_id]:
        # Skip hetero-atoms and water
        if residue.id[0] != ' ':
            continue
        res_num = residue.id[1]
        res_name = residue.resname.strip()
        abs_sasa = residue.sasa
        max_sasa = MAX_ASA.get(res_name, MAX_ASA_DEFAULT)
        sasa_map[res_num] = abs_sasa / max_sasa if max_sasa > 0 else 0.0
    return sasa_map

def _compute_sasa_biotite(pdb_path: Union[str, Path], chain_a: str, chain_b: str) -> tuple[dict[int, float], dict[int, float]]:
    """Compute per-residue RSA for both chains using biotite (Cython-accelerated).
    ~10x faster than BioPython's ShrakeRupley for typical protein complexes.
    Uses 30 sphere points (r=0.991 correlation with 100-point reference) which is more than sufficient for the binary buried/surface classification.
    Args:
        pdb_path: Path to PDB file.
        chain_a: Chain identifier for protein A.
        chain_b: Chain identifier for protein B.
    Returns:
        Tuple of (sasa_map_a, sasa_map_b), each mapping residue number to RSA.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message='.*elements were guessed from atom name.*',
            category=UserWarning,
        )
        pdb_file = _bt_pdb_io.PDBFile.read(str(pdb_path))
        structure = pdb_file.get_structure(model=1)
    structure = structure[_bt_struc.filter_amino_acids(structure)]

    atom_sasa = _bt_struc.sasa(structure, vdw_radii='ProtOr', point_number=30)

    sasa_maps: list[dict[int, float]] = []
    for chain_id in (chain_a, chain_b):
        chain_mask = structure.chain_id == chain_id
        chain_atoms = structure[chain_mask]
        chain_sasa_vals = atom_sasa[chain_mask]

        sasa_map: dict[int, float] = {}
        for res_num in np.unique(chain_atoms.res_id):
            res_mask = chain_atoms.res_id == res_num
            abs_sasa = float(chain_sasa_vals[res_mask].sum())
            res_name = str(chain_atoms.res_name[res_mask][0])
            max_sasa = MAX_ASA.get(res_name, MAX_ASA_DEFAULT)
            sasa_map[int(res_num)] = abs_sasa / max_sasa if max_sasa > 0 else 0.0
        sasa_maps.append(sasa_map)
    return sasa_maps[0], sasa_maps[1]

def compute_residue_sasa_both_chains(pdb_path: Union[str, Path], chain_a: str, chain_b: str) -> tuple[dict[int, float], dict[int, float]]:
    """Compute per-residue RSA for both chains in a single parse + compute.
    Uses biotite (Cython-accelerated, ~10x faster) when available, with BioPython ShrakeRupley as fallback.
    Args:
        pdb_path: Path to PDB file.
        chain_a: Chain identifier for protein A (e.g. 'A').
        chain_b: Chain identifier for protein B (e.g. 'B').
    Returns:
        Tuple of (sasa_map_a, sasa_map_b), each mapping residue number to RSA.
    """
    # Prefer biotite for ~10x speedup (Cython vs pure Python)
    if _HAS_BIOTITE:
        return _compute_sasa_biotite(pdb_path, chain_a, chain_b)

    if not _HAS_BIOPYTHON:
        raise ImportError(
            "SASA computation requires biotite (recommended) or biopython: "
            "pip install biotite  OR  pip install biopython"
        )

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('complex', str(pdb_path))

    sr = ShrakeRupley()
    sr.compute(structure[0], level='R')

    sasa_maps: list[dict[int, float]] = []
    for chain_id in (chain_a, chain_b):
        sasa_map: dict[int, float] = {}
        for residue in structure[0][chain_id]:
            if residue.id[0] != ' ':
                continue
            res_num = residue.id[1]
            res_name = residue.resname.strip()
            abs_sasa = residue.sasa
            max_sasa = MAX_ASA.get(res_name, MAX_ASA_DEFAULT)
            sasa_map[res_num] = abs_sasa / max_sasa if max_sasa > 0 else 0.0
        sasa_maps.append(sasa_map)
    return sasa_maps[0], sasa_maps[1]

def is_buried(rsa: float) -> bool:
    """Check if a residue is buried based on relative solvent accessibility.
    Args:
        rsa: Relative solvent accessibility (0.0-1.0+).
    Returns:
        True if RSA < SASA_BURIED_THRESHOLD (25%).
    """
    return rsa < SASA_BURIED_THRESHOLD

def _compute_sasa_pair(pdb_path: str, chain_a: str, chain_b: str) -> tuple[dict[int, float], dict[int, float]]:
    """Compute SASA for both chains of a complex (picklable worker function).
    Top-level function suitable for ProcessPoolExecutor. Takes only primitive/picklable arguments and returns two SASA maps.
    Args:
        pdb_path: Path to PDB file (as string for pickling).
        chain_a: Chain identifier for protein A.
        chain_b: Chain identifier for protein B.
    Returns:
        Tuple of (sasa_map_a, sasa_map_b). On failure returns ({}, {}).
    """
    try:
        return compute_residue_sasa_both_chains(pdb_path, chain_a, chain_b)
    except Exception:
        return {}, {}

def precompute_sasa_parallel(results: list[dict], workers: int = 4, verbose: bool = False) -> dict[int, tuple[dict[int, float], dict[int, float]]]:
    """Pre-compute SASA maps for all complexes in parallel.
    Uses ProcessPoolExecutor to parallelise the BioPython ShrakeRupley SASA calculation, which is the primary performance bottleneck when processing thousands of complexes (~0.1-0.5s per structure).
    Args:
        results: List of per-complex result dicts (must contain '_pdb_path', '_chain_info', 'best_chain_pair' keys).
        workers: Number of parallel worker processes.  Set to 1 for serial execution (useful for debugging or when multiprocessing overhead exceeds the benefit).
        verbose: Print progress to stderr.
    Returns:
        Dict mapping result index → (sasa_map_a, sasa_map_b). Complexes that were skipped (no PDB path or chain info) are omitted.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # Collect work items: (index, pdb_path, chain_a, chain_b)
    work_items: list[tuple[int, str, str, str]] = []
    for i, row in enumerate(results):
        pdb_path = row.get('_pdb_path')
        chain_info = row.get('_chain_info')
        if pdb_path is None or chain_info is None:
            continue

        best_pair = row.get('best_chain_pair', '')
        if best_pair and '_' in best_pair:
            chain_a, chain_b = best_pair.split('_', 1)
        else:
            chain_a, chain_b = 'A', 'B'

        work_items.append((i, str(pdb_path), chain_a, chain_b))

    if not work_items:
        return {}

    if verbose:
        print(f"  Pre-computing SASA for {len(work_items):,} complexes "
              f"using {workers} worker(s)...", file=sys.stderr)

    sasa_cache: dict[int, tuple[dict[int, float], dict[int, float]]] = {}

    # Serial mode: avoid ProcessPoolExecutor overhead for small batches
    if workers <= 1:
        for idx, pdb_path, ca, cb in work_items:
            sasa_cache[idx] = _compute_sasa_pair(pdb_path, ca, cb)
        if verbose:
            print(f"  SASA pre-computation complete (serial): "
                  f"{len(sasa_cache):,} structures", file=sys.stderr)
        return sasa_cache

    # Parallel mode
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {
            executor.submit(_compute_sasa_pair, pdb_path, ca, cb): idx
            for idx, pdb_path, ca, cb in work_items
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                sasa_cache[idx] = future.result()
            except Exception:
                sasa_cache[idx] = ({}, {})
            completed += 1
            if verbose and completed % 100 == 0:
                print(f"  SASA progress: {completed:,}/{len(work_items):,}",
                      file=sys.stderr)
    if verbose:
        print(f"  SASA pre-computation complete (parallel): "
              f"{len(sasa_cache):,} structures", file=sys.stderr)
    return sasa_cache


#-------------------------------Section 3: Structural Mapping----------------------------------------

def compute_distance_to_interface(variant_cb_coord: np.ndarray, interface_cb_coords: np.ndarray) -> float:
    """Compute minimum CB-CB distance from a variant to any interface residue.
    Uses vectorised numpy computation for efficiency.
    Args:
        variant_cb_coord: (3,) array of variant residue CB coordinates.
        interface_cb_coords: (N, 3) array of interface residue CB coordinates.
    Returns:
        Minimum Euclidean distance in Angstroms. Returns float('inf') if interface_cb_coords is empty.
    """
    if interface_cb_coords.size == 0:
        return float('inf')

    distances = np.linalg.norm(interface_cb_coords - variant_cb_coord, axis=1)
    return float(distances.min())

def classify_structural_context(
    position: int,
    interface_residues: set[int],
    chain_res_numbers: list[int],
    cb_coords: np.ndarray,
    residue_sasa: dict[int, float],
    interface_cb_coords: Optional[np.ndarray] = None,
    cross_chain_cb_coords: Optional[np.ndarray] = None,
    interface_residue_list: Optional[list[int]] = None,
    res_to_idx: Optional[dict[int, int]] = None,
) -> dict:
    """Classify a variant's structural context within a protein complex.
    Classification logic:
        1. If position is in interface_residues:
           - CB distance < 4A from cross-chain contact partner -> interface_core
           - CB distance 4-8A from cross-chain partner -> interface_rim
        2. If position is NOT in interface_residues:
           - RSA < 25% -> buried_core
           - RSA >= 25% -> surface_non_interface
        3. If position not found in chain -> unmapped
    Args:
        position: PDB residue number of the variant.
        interface_residues: Set of PDB residue numbers at the interface.
        chain_res_numbers: List of PDB residue numbers for all CB atoms in this chain.
        cb_coords: (N_cb, 3) array of CB coordinates for this chain.
        residue_sasa: Dict mapping PDB residue number to RSA from compute_residue_sasa().
        interface_cb_coords: (M, 3) array of same-chain interface residue CB coordinates. Used for distance_to_interface and nearest_interface_residue for non-interface variants. If None, distance is not computed.
        cross_chain_cb_coords: (K, 3) array of partner chain's interface residue CB coordinates. Used for core/rim classification of interface variants. If None, falls back to interface_rim for interface residues.
        interface_residue_list: Resolved list of same-chain interface residue numbers matching rows of interface_cb_coords. If None, sorted(interface_residues) is used (may misalign if some residues are missing from chain_res_numbers).
        res_to_idx: Pre-built {residue_number: array_index} lookup dict for O(1) position lookups. If None, falls back to list.index() (O(n) per call).
    Returns:
        Dict with keys: context (str), distance_to_interface (float), nearest_interface_residue (int or None).
    """
    result = {
        'context': CONTEXT_UNMAPPED,
        'distance_to_interface': float('inf'),
        'nearest_interface_residue': None,
    }

    # Find the CB array index for this position (O(1) with dict, O(n) without)
    if res_to_idx is not None:
        cb_idx = res_to_idx.get(position)
        if cb_idx is None:
            return result
    else:
        if position not in chain_res_numbers:
            return result
        try:
            cb_idx = chain_res_numbers.index(position)
        except ValueError:
            return result

    variant_coord = cb_coords[cb_idx]

    # Compute distance to nearest same-chain interface residue
    if interface_cb_coords is not None and interface_cb_coords.size > 0:
        min_dist = compute_distance_to_interface(variant_coord, interface_cb_coords)
        result['distance_to_interface'] = min_dist

        # Find nearest interface residue (same-chain)
        distances = np.linalg.norm(interface_cb_coords - variant_coord, axis=1)
        nearest_idx = int(distances.argmin())
        iface_list = interface_residue_list if interface_residue_list is not None else sorted(interface_residues)
        if nearest_idx < len(iface_list):
            result['nearest_interface_residue'] = iface_list[nearest_idx]

    # Classify context
    if position in interface_residues:
        # Use cross-chain distance for core/rim distinction
        if cross_chain_cb_coords is not None and cross_chain_cb_coords.size > 0:
            cross_dist = compute_distance_to_interface(variant_coord, cross_chain_cb_coords)
        else:
            cross_dist = float('inf')

        if cross_dist < INTERFACE_CORE_DISTANCE:
            result['context'] = CONTEXT_INTERFACE_CORE
        elif cross_dist < INTERFACE_RIM_DISTANCE:
            result['context'] = CONTEXT_INTERFACE_RIM
        else:
            # At interface but > 8A from cross-chain partner (rare edge case)
            result['context'] = CONTEXT_INTERFACE_RIM
    else:
        # Not at interface - check SASA for buried vs surface
        rsa = residue_sasa.get(position, 0.5)  # default to surface if unknown
        if is_buried(rsa):
            result['context'] = CONTEXT_BURIED
        else:
            result['context'] = CONTEXT_SURFACE
    return result

def _build_interface_cb_coords(interface_residues: set[int], chain_res_numbers: list[int], cb_coords: np.ndarray, res_to_idx: Optional[dict[int, int]] = None) -> tuple[np.ndarray, list[int]]:
    """Extract CB coordinates for interface residues found in chain.
    Args:
        interface_residues: Set of interface residue PDB numbers.
        chain_res_numbers: List of PDB residue numbers for CB atoms.
        cb_coords: (N, 3) CB coordinates array.
        res_to_idx: Pre-built {residue_number: array_index} for O(1) lookups. If None, one is built from chain_res_numbers.
    Returns:
        Tuple of (interface_cb_coords array, resolved residue number list).
        The resolved list contains only residues actually found in chain_res_numbers, in sorted order, and aligns row-by-row with the returned CB array.
    """
    if res_to_idx is None:
        res_to_idx = {r: i for i, r in enumerate(chain_res_numbers)}
    indices = []
    resolved = []
    for res_num in sorted(interface_residues):
        idx = res_to_idx.get(res_num)
        if idx is not None:
            indices.append(idx)
            resolved.append(res_num)
    if indices:
        return cb_coords[indices], resolved
    return np.empty((0, 3)), []

def map_variants_to_complex(
    protein_id: str,
    chain_id: str,
    variant_index: dict[str, list[dict]],
    interface_residues: set[int],
    chain_res_numbers: list[int],
    cb_coords: np.ndarray,
    residue_sasa: dict[int, float],
    cross_chain_cb_coords: Optional[np.ndarray] = None,
) -> list[dict]:
    """Map all known variants for a protein onto a specific chain in a complex.

    For each variant: validates position exists in the chain, classifies
    its structural context, and computes distance to the nearest interface residue.

    Args:
        protein_id: UniProt accession (e.g. 'P24534').
        chain_id: Chain identifier (e.g. 'A').
        variant_index: Dict from build_variant_index().
        interface_residues: Set of confident interface residue PDB numbers.
        chain_res_numbers: List of PDB residue numbers for CB atoms in this chain.
        cb_coords: (N_cb, 3) CB coordinates for this chain.
        residue_sasa: Dict mapping PDB residue number to RSA.
        cross_chain_cb_coords: (K, 3) CB coordinates of the partner chain's
            interface residues. Used for core/rim classification of interface
            variants. If None, interface variants default to interface_rim.

    Returns:
        List of enriched variant dicts, each with additional keys:
        context, distance_to_interface, nearest_interface_residue, chain_id.
    """
    # Also try base accession (strip isoform suffix)
    from id_mapper import split_isoform
    base_acc, _ = split_isoform(protein_id)
    variants = variant_index.get(protein_id, [])
    if not variants and base_acc != protein_id:
        variants = variant_index.get(base_acc, [])

    if not variants:
        return []

    # Build O(1) residue-to-index lookup (used by both _build_interface_cb_coords
    # and classify_structural_context, avoiding repeated O(n) list.index() calls)
    res_to_idx = {r: i for i, r in enumerate(chain_res_numbers)}

    # Build same-chain interface CB coordinates for distance computation
    interface_cb_coords, iface_res_resolved = _build_interface_cb_coords(
        interface_residues, chain_res_numbers, cb_coords,
        res_to_idx=res_to_idx,
    )

    mapped_variants = []
    for var in variants:
        position = var['position']

        context_info = classify_structural_context(
            position, interface_residues, chain_res_numbers,
            cb_coords, residue_sasa, interface_cb_coords,
            cross_chain_cb_coords=cross_chain_cb_coords,
            interface_residue_list=iface_res_resolved,
            res_to_idx=res_to_idx,
        )

        enriched = dict(var)
        enriched['chain_id'] = chain_id
        enriched['context'] = context_info['context']
        enriched['distance_to_interface'] = context_info['distance_to_interface']
        enriched['nearest_interface_residue'] = context_info['nearest_interface_residue']

        mapped_variants.append(enriched)

    return mapped_variants


def compute_interface_variant_enrichment(
    n_interface_variants: int,
    n_total_variants: int,
    n_interface_residues: int,
    n_total_residues: int,
) -> float:
    """Compute fold-enrichment of variants at interface positions.

    Enrichment = (n_interface_variants / n_total_variants) /
                 (n_interface_residues / n_total_residues)

    A value > 1.0 means variants are enriched at the interface.
    Burke et al. (2023) found 2.3-fold enrichment of disease mutations.

    Args:
        n_interface_variants: Number of variants at interface residues.
        n_total_variants: Total number of variants for this protein.
        n_interface_residues: Number of interface residues.
        n_total_residues: Total number of residues in the protein.

    Returns:
        Fold-enrichment value. Returns 0.0 if any denominator is zero.
    """
    if n_total_variants == 0 or n_total_residues == 0 or n_interface_residues == 0:
        return 0.0

    observed_fraction = n_interface_variants / n_total_variants
    expected_fraction = n_interface_residues / n_total_residues

    if expected_fraction == 0:
        return 0.0

    return observed_fraction / expected_fraction


# ── Section 4: Annotation (toolkit integration) ─────────────────────

def format_variant_details(
    mapped_variants: list[dict],
    limit: int = VARIANT_DETAILS_DISPLAY_LIMIT,
) -> str:
    """Format mapped variants into a pipe-separated summary string.

    Format: ref_aa{position}alt_aa:context:clinical_significance

    Args:
        mapped_variants: List of enriched variant dicts from map_variants_to_complex().
        limit: Maximum number of variants to include. Remainder shown as '...(+N more)'.

    Returns:
        Pipe-separated string, e.g. 'K81P:interface_core:pathogenic|E82K:interface_rim:VUS'.
        Empty string if no variants.
    """
    if not mapped_variants:
        return ''

    details = []
    for var in mapped_variants[:limit]:
        ref = var.get('ref_aa', '?')
        pos = var.get('position', '?')
        alt = var.get('alt_aa', '?')
        ctx = var.get('context', 'unknown')
        clin = var.get('clinical_significance', '')
        if not clin or clin == 'nan':
            clin = '-'
        details.append(f"{ref}{pos}{alt}:{ctx}:{clin}")

    result = '|'.join(details)

    remaining = len(mapped_variants) - limit
    if remaining > 0:
        result += f"|...(+{remaining} more)"

    return result


def annotate_results_with_variants(
    results: list[dict],
    variant_index: dict[str, list[dict]],
    exac_df: pd.DataFrame,
    gene_symbol_lookup: dict[str, str],
    verbose: bool = False,
    workers: int = 1,
) -> None:
    """Annotate result rows with variant mapping data (in-place).

    Main entry point from toolkit.py. For each complex:
    1. Looks up variants for protein_a and protein_b
    2. Uses pre-stashed SASA maps (toolkit path) or computes them (standalone CLI)
    3. Maps variants to interface/non-interface positions
    4. Counts interface variants and pathogenic interface variants
    5. Computes fold-enrichment
    6. Attaches ExAC gene constraint scores

    Two execution paths:
    - **Toolkit path**: Results contain ``_sasa_a``/``_sasa_b`` (pre-computed
      in worker processes by ``process_single_complex()``). No additional SASA
      computation needed - eliminates the pickling bottleneck.
    - **Standalone CLI path**: Results contain ``_chain_info``/``_pdb_path``.
      SASA is computed via :func:`precompute_sasa_parallel`.

    Args:
        results: List of per-complex result dicts. Modified in-place.
        variant_index: Dict from build_variant_index().
        exac_df: DataFrame from load_exac_constraint().
        gene_symbol_lookup: Dict mapping UniProt accession to gene symbol.
        verbose: Print progress to stderr.
        workers: Number of parallel processes for SASA computation
            (standalone CLI path only). Defaults to 1 (serial).
    """
    # Build ExAC lookup by gene symbol
    exac_lookup: dict[str, dict] = {}
    for _, erow in exac_df.iterrows():
        gene = str(erow.get('gene', ''))
        if gene:
            exac_lookup[gene] = {
                'pLI': float(erow['pLI']) if pd.notna(erow.get('pLI')) else None,
                'mis_z': float(erow['mis_z']) if pd.notna(erow.get('mis_z')) else None,
            }

    # Detect execution path: toolkit (pre-stashed SASA) vs standalone (needs computation)
    has_prestashed = any('_sasa_a' in r for r in results[:10])
    sasa_cache: dict[int, tuple[dict, dict]] = {}

    if not has_prestashed:
        # Standalone CLI path: compute SASA from _chain_info/_pdb_path
        sasa_cache = precompute_sasa_parallel(results, workers=workers, verbose=verbose)

    # Lazy import avoids a circular dependency at module load time
    # (toolkit.py imports variant_mapper in multiple places).
    from toolkit import is_annotatable

    annotated_count = 0
    sasa_computed_count = 0

    for i, row in enumerate(results):
        protein_a = row.get('protein_a', '')
        protein_b = row.get('protein_b', '')
        conf_res_a = row.get('_confident_residue_numbers_a', [])
        conf_res_b = row.get('_confident_residue_numbers_b', [])
        best_pair = row.get('best_chain_pair', '')

        # Parse chain IDs from best_chain_pair (format: "A_B")
        if best_pair and '_' in best_pair:
            chain_a, chain_b = best_pair.split('_', 1)
        else:
            chain_a, chain_b = 'A', 'B'

        # Initialise variant columns to defaults
        row['n_variants_a'] = 0
        row['n_variants_b'] = 0
        row['n_interface_variants_a'] = 0
        row['n_interface_variants_b'] = 0
        row['n_pathogenic_interface_variants'] = 0
        row['interface_variant_enrichment'] = ''
        row['variant_details_a'] = ''
        row['variant_details_b'] = ''
        row['gene_constraint_pli_a'] = ''
        row['gene_constraint_pli_b'] = ''
        row['gene_constraint_mis_z_a'] = ''
        row['gene_constraint_mis_z_b'] = ''

        # Non-human rows: defaults above are correct (empty); UniProt/ClinVar/ExAC
        # don't cover them, so skip the lookup work. TrEMBL-human rows still run
        # the lookup because UniProt carries variants for them.
        if not is_annotatable(row):
            continue

        # Attach ExAC gene constraint scores
        gene_a = gene_symbol_lookup.get(protein_a, '')
        gene_b = gene_symbol_lookup.get(protein_b, '')

        if gene_a and gene_a in exac_lookup:
            ec = exac_lookup[gene_a]
            if ec['pLI'] is not None:
                row['gene_constraint_pli_a'] = f"{ec['pLI']:.4f}"
            if ec['mis_z'] is not None:
                row['gene_constraint_mis_z_a'] = f"{ec['mis_z']:.2f}"

        if gene_b and gene_b in exac_lookup:
            ec = exac_lookup[gene_b]
            if ec['pLI'] is not None:
                row['gene_constraint_pli_b'] = f"{ec['pLI']:.4f}"
            if ec['mis_z'] is not None:
                row['gene_constraint_mis_z_b'] = f"{ec['mis_z']:.2f}"

        # Resolve SASA and structural data depending on execution path
        if has_prestashed:
            # Toolkit path: SASA and structural data pre-stashed by worker
            sasa_a = row.get('_sasa_a', {})
            sasa_b = row.get('_sasa_b', {})
            res_numbers_a = row.get('_chain_res_numbers_a', [])
            res_numbers_b = row.get('_chain_res_numbers_b', [])
            cb_coords_a = np.array(row.get('_cb_coords_a', []))
            cb_coords_b = np.array(row.get('_cb_coords_b', []))
            has_structural = bool(sasa_a or sasa_b or res_numbers_a or res_numbers_b)
        else:
            # Standalone CLI path: use _chain_info + sasa_cache
            chain_info = row.get('_chain_info')
            pdb_path = row.get('_pdb_path')
            if chain_info is None or pdb_path is None:
                continue
            if i in sasa_cache:
                sasa_a, sasa_b = sasa_cache[i]
            else:
                sasa_a, sasa_b = {}, {}
            res_numbers_a = chain_info.chain_res_numbers.get(chain_a, [])
            res_numbers_b = chain_info.chain_res_numbers.get(chain_b, [])
            cb_coords_a = chain_info.cb_coords.get(chain_a, np.empty((0, 3)))
            cb_coords_b = chain_info.cb_coords.get(chain_b, np.empty((0, 3)))
            has_structural = True

        if not has_structural:
            continue

        if sasa_a or sasa_b:
            sasa_computed_count += 1

        # Map variants for protein A
        interface_a = set(conf_res_a) if conf_res_a else set()
        interface_b = set(conf_res_b) if conf_res_b else set()

        # Build cross-chain interface CB coords for core/rim classification
        cross_cb_for_a, _ = _build_interface_cb_coords(interface_b, res_numbers_b, cb_coords_b)
        cross_cb_for_b, _ = _build_interface_cb_coords(interface_a, res_numbers_a, cb_coords_a)

        mapped_a = map_variants_to_complex(
            protein_a, chain_a, variant_index, interface_a,
            res_numbers_a, cb_coords_a, sasa_a,
            cross_chain_cb_coords=cross_cb_for_a,
        )

        # Map variants for protein B
        mapped_b = map_variants_to_complex(
            protein_b, chain_b, variant_index, interface_b,
            res_numbers_b, cb_coords_b, sasa_b,
            cross_chain_cb_coords=cross_cb_for_b,
        )

        # Count variants
        n_variants_a = len(mapped_a)
        n_variants_b = len(mapped_b)
        n_if_vars_a = sum(1 for v in mapped_a
                         if v['context'] in (CONTEXT_INTERFACE_CORE, CONTEXT_INTERFACE_RIM))
        n_if_vars_b = sum(1 for v in mapped_b
                         if v['context'] in (CONTEXT_INTERFACE_CORE, CONTEXT_INTERFACE_RIM))

        # Count pathogenic variants at interface (both chains combined)
        pathogenic_keywords = {'pathogenic', 'likely pathogenic', 'pathogenic/likely pathogenic'}
        n_path_if = 0
        for v in mapped_a + mapped_b:
            if v['context'] in (CONTEXT_INTERFACE_CORE, CONTEXT_INTERFACE_RIM):
                clin = str(v.get('clinical_significance', '')).lower().strip()
                if clin in pathogenic_keywords:
                    n_path_if += 1

        # Compute enrichment
        n_if_residues = len(interface_a) + len(interface_b)
        n_total_residues_a = len(res_numbers_a)
        n_total_residues_b = len(res_numbers_b)
        n_total_residues = n_total_residues_a + n_total_residues_b
        n_total_variants = n_variants_a + n_variants_b
        n_if_variants = n_if_vars_a + n_if_vars_b

        enrichment = compute_interface_variant_enrichment(
            n_if_variants, n_total_variants, n_if_residues, n_total_residues,
        )

        # Fill row
        row['n_variants_a'] = n_variants_a
        row['n_variants_b'] = n_variants_b
        row['n_interface_variants_a'] = n_if_vars_a
        row['n_interface_variants_b'] = n_if_vars_b
        row['n_pathogenic_interface_variants'] = n_path_if
        row['interface_variant_enrichment'] = f"{enrichment:.4f}" if enrichment > 0 else ''
        row['variant_details_a'] = format_variant_details(mapped_a)
        row['variant_details_b'] = format_variant_details(mapped_b)

        annotated_count += 1

        # Strip private keys (not for CSV output)
        for key in ('_chain_info', '_pdb_path', '_confident_residue_numbers_a',
                     '_confident_residue_numbers_b', '_sasa_a', '_sasa_b',
                     '_chain_res_numbers_a', '_chain_res_numbers_b',
                     '_cb_coords_a', '_cb_coords_b'):
            row.pop(key, None)

    if verbose:
        print(f"  Variant annotation complete: {annotated_count:,} complexes annotated, "
              f"{sasa_computed_count:,} SASA computations", file=sys.stderr)


# ── Section 5: Standalone CLI ────────────────────────────────────────

def build_argument_parser() -> argparse.ArgumentParser:
    """Build argument parser for standalone variant mapping CLI."""
    parser = argparse.ArgumentParser(
        description="Map genetic variants to AlphaFold2-predicted protein complex structures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Summary statistics from variant databases
    python variant_mapper.py summary --variants-dir data/variants

    # Look up variants for a specific protein
    python variant_mapper.py lookup --variants-dir data/variants --protein P24534

    # Map variants onto a JSONL interface export
    python variant_mapper.py map --interfaces interfaces.jsonl --pdb-dir DIR --variants-dir data/variants --output variant_analysis.csv
        """,
    )

    subparsers = parser.add_subparsers(dest='command', required=True)

    # Summary subcommand
    summary_parser = subparsers.add_parser('summary',
        help="Print summary statistics from variant databases.")
    summary_parser.add_argument("--variants-dir", default=str(DEFAULT_VARIANTS_DIR),
        help="Directory containing variant database files.")

    # Lookup subcommand
    lookup_parser = subparsers.add_parser('lookup',
        help="Look up variants for a specific protein.")
    lookup_parser.add_argument("--variants-dir", default=str(DEFAULT_VARIANTS_DIR),
        help="Directory containing variant database files.")
    lookup_parser.add_argument("--protein", required=True,
        help="UniProt accession to look up (e.g. P24534).")

    # Map subcommand
    map_parser = subparsers.add_parser('map',
        help="Map variants onto interface residues from a JSONL export.")
    map_parser.add_argument("--interfaces", required=True,
        help="Path to JSONL interface export from toolkit.py --export-interfaces.")
    map_parser.add_argument("--pdb-dir", required=True,
        help="Directory containing PDB files.")
    map_parser.add_argument("--variants-dir", default=str(DEFAULT_VARIANTS_DIR),
        help="Directory containing variant database files.")
    map_parser.add_argument("--output", default="variant_analysis.csv",
        help="Output CSV file path.")
    map_parser.add_argument("--no-clinvar", action="store_true",
        help="Skip ClinVar enrichment (faster).")

    return parser


def main() -> None:
    """Run standalone variant mapping CLI."""
    parser = build_argument_parser()
    args = parser.parse_args()

    variants_dir = Path(args.variants_dir)

    if args.command == 'summary':
        _cli_summary(variants_dir)
    elif args.command == 'lookup':
        _cli_lookup(variants_dir, args.protein)
    elif args.command == 'map':
        _cli_map(args)


def _cli_summary(variants_dir: Path) -> None:
    """Print summary statistics from variant databases."""
    print(f"Variant data directory: {variants_dir}")
    print()

    # UniProt
    uniprot_path = variants_dir / UNIPROT_VARIANTS_FILENAME
    if uniprot_path.exists():
        size_gb = uniprot_path.stat().st_size / (1024 ** 3)
        print(f"UniProt variants: {uniprot_path.name} ({size_gb:.1f} GB)")
    else:
        print(f"UniProt variants: NOT FOUND ({uniprot_path})")

    # ClinVar
    clinvar_path = variants_dir / CLINVAR_VARIANTS_FILENAME
    if clinvar_path.exists():
        size_gb = clinvar_path.stat().st_size / (1024 ** 3)
        print(f"ClinVar variants: {clinvar_path.name} ({size_gb:.1f} GB)")
    else:
        print(f"ClinVar variants: NOT FOUND ({clinvar_path})")

    # ExAC
    exac_path = variants_dir / EXAC_CONSTRAINT_FILENAME
    if exac_path.exists():
        size_mb = exac_path.stat().st_size / (1024 ** 2)
        exac_df = load_exac_constraint(exac_path)
        print(f"ExAC constraint: {exac_path.name} ({size_mb:.1f} MB, {len(exac_df):,} genes)")
        if len(exac_df) > 0:
            print(f"  pLI range: {exac_df['pLI'].min():.4f} - {exac_df['pLI'].max():.4f}")
    else:
        print(f"ExAC constraint: NOT FOUND ({exac_path})")


def _cli_lookup(variants_dir: Path, protein: str) -> None:
    """Look up variants for a specific protein."""
    uniprot_path = variants_dir / UNIPROT_VARIANTS_FILENAME
    if not uniprot_path.exists():
        print(f"Error: UniProt file not found: {uniprot_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Looking up variants for: {protein}")
    df = load_uniprot_variants(uniprot_path, frozenset({protein}), verbose=True)

    if df.empty:
        print(f"No variants found for {protein}")
        return

    print(f"\nFound {len(df)} variants:")
    print(f"{'Pos':>5}  {'Change':>8}  {'Consequence':<20}  {'Clinical':>20}  {'rsID':<15}")
    print('-' * 75)
    for _, row in df.head(50).iterrows():
        pos = row['position']
        change = f"{row['ref_aa']}{pos}{row['alt_aa']}"
        cons = str(row.get('consequence', ''))[:20]
        clin = str(row.get('clinical_significance', '-'))[:20]
        rsid = str(row.get('rsid', ''))[:15]
        print(f"{pos:>5}  {change:>8}  {cons:<20}  {clin:>20}  {rsid:<15}")

    if len(df) > 50:
        print(f"  ... and {len(df) - 50} more variants")


def _cli_map(args) -> None:
    """Map variants onto interface residues from a JSONL export."""
    from pdockq import read_pdb_with_chain_info_New as read_pdb_with_chain_info

    variants_dir = Path(args.variants_dir)
    pdb_dir = Path(args.pdb_dir)
    interfaces_path = Path(args.interfaces)

    # Load JSONL interface export
    print(f"Loading interfaces from: {interfaces_path}")
    interface_records = []
    with open(interfaces_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                interface_records.append(json.loads(line))
    print(f"  Loaded {len(interface_records)} complex interface records")

    # Collect unique accessions
    accessions = set()
    for rec in interface_records:
        accessions.add(rec.get('protein_a', ''))
        accessions.add(rec.get('protein_b', ''))
    accessions.discard('')

    # Load variant databases
    print("Loading variant databases...")
    uniprot_path = variants_dir / UNIPROT_VARIANTS_FILENAME
    variants_df = load_uniprot_variants(uniprot_path, frozenset(accessions), verbose=True)
    variant_idx = build_variant_index(variants_df)

    # ClinVar enrichment
    if not args.no_clinvar:
        clinvar_path = variants_dir / CLINVAR_VARIANTS_FILENAME
        if clinvar_path.exists():
            rsids = frozenset(
                str(v['rsid']) for variants in variant_idx.values()
                for v in variants if v.get('rsid')
            )
            clinvar_df = load_clinvar_variants(clinvar_path, rsids=rsids, verbose=True)
            enrich_with_clinvar(variant_idx, clinvar_df, verbose=True)

    # ExAC constraint
    exac_path = variants_dir / EXAC_CONSTRAINT_FILENAME
    exac_df = load_exac_constraint(exac_path) if exac_path.exists() else pd.DataFrame()

    # Process each complex
    print(f"Mapping variants to {len(interface_records)} complexes...")
    output_rows = []

    for rec in interface_records:
        complex_name = rec.get('complex_name', '')
        protein_a = rec.get('protein_a', '')
        protein_b = rec.get('protein_b', '')

        # Find PDB file
        pdb_candidates = list(pdb_dir.glob(f"{complex_name}*.pdb"))
        if not pdb_candidates:
            continue

        pdb_path = pdb_candidates[0]
        chain_info = read_pdb_with_chain_info(str(pdb_path))

        conf_res_a = set(rec.get('confident_interface_residues_a', []))
        conf_res_b = set(rec.get('confident_interface_residues_b', []))

        # Compute SASA
        chain_ids = chain_info.chain_ids
        chain_a = chain_ids[0] if chain_ids else 'A'
        chain_b = chain_ids[1] if len(chain_ids) > 1 else 'B'

        try:
            sasa_a = compute_residue_sasa(pdb_path, chain_a)
            sasa_b = compute_residue_sasa(pdb_path, chain_b)
        except Exception:
            sasa_a, sasa_b = {}, {}

        res_a = chain_info.chain_res_numbers.get(chain_a, [])
        res_b = chain_info.chain_res_numbers.get(chain_b, [])
        cb_a = chain_info.cb_coords.get(chain_a, np.empty((0, 3)))
        cb_b = chain_info.cb_coords.get(chain_b, np.empty((0, 3)))

        # Build cross-chain interface CB coords for core/rim classification
        cross_cb_for_a, _ = _build_interface_cb_coords(conf_res_b, res_b, cb_b)
        cross_cb_for_b, _ = _build_interface_cb_coords(conf_res_a, res_a, cb_a)

        mapped_a = map_variants_to_complex(
            protein_a, chain_a, variant_idx, conf_res_a,
            res_a, cb_a, sasa_a,
            cross_chain_cb_coords=cross_cb_for_a,
        )

        mapped_b = map_variants_to_complex(
            protein_b, chain_b, variant_idx, conf_res_b,
            res_b, cb_b, sasa_b,
            cross_chain_cb_coords=cross_cb_for_b,
        )

        output_rows.append({
            'complex_name': complex_name,
            'protein_a': protein_a,
            'protein_b': protein_b,
            'n_variants_a': len(mapped_a),
            'n_variants_b': len(mapped_b),
            'variant_details_a': format_variant_details(mapped_a),
            'variant_details_b': format_variant_details(mapped_b),
        })

    # Write output
    if output_rows:
        fieldnames = list(output_rows[0].keys())
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)
        print(f"Output written to: {args.output} ({len(output_rows)} complexes)")
    else:
        print("No complexes processed.")


if __name__ == "__main__":
    main()
