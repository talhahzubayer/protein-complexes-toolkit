#!/usr/bin/env python3
"""
PPI Database Loading Module.
Parses protein-protein interaction data from 4 databases (STRING, BioGRID, HuRI, HuMAP) into standardised DataFrames with columns: protein_a, protein_b, source, confidence_score, evidence_type

Features:
    - STRING: combined-score filtering, automatic species-prefix stripping (Ensembl protein IDs)
    - BioGRID: column-name-based loading, human-only filtering (taxonomy ID 9606)
    - HuRI: binary TSV loading with Ensembl gene IDs
    - HuMAP: probability-based filtering with optional UniProt ID validation
    - Unified loader: loads all four databases in one call with configurable thresholds

Usage (as importable module):
    from database_loaders import load_string, load_biogrid, load_huri, load_humap
    df = load_string("data/ppi/9606.protein.links.v12.0.txt", min_score=700)
    df = load_biogrid("data/ppi/BIOGRID-ALL-5.0.253.tab3.txt")
    df = load_huri("data/ppi/HuRI.tsv")
    df = load_humap("data/ppi/humap2_ppis_ACC_20200821.pairsWprob", min_probability=0.5)

Usage (standalone):
    python database_loaders.py --data-dir data/ppi --database all --output interactions.csv -v
    python database_loaders.py --data-dir data/ppi --database string --min-string-score 700 -v
"""

import sys
import re
import argparse
import csv
import warnings
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

#-------------------Constants----------------------------
# Default data directory
DEFAULT_DATA_DIR = Path(__file__).parent / "data" / "ppi"

# Default filenames
STRING_LINKS_FILE = "9606.protein.links.v12.0.txt"
STRING_ALIASES_FILE = "9606.protein.aliases.v12.0.txt"
BIOGRID_FILE = "BIOGRID-ALL-5.0.253.tab3.txt"
HURI_FILE = "HuRI.tsv"
HUMAP_FILE = "humap2_ppis_ACC_20200821.pairsWprob"

# STRING score normalisation
STRING_MAX_SCORE = 1000

# BioGRID organism filter
HUMAN_TAXONOMY_ID = "9606"

# Chunked reading for large files
CHUNK_SIZE = 500_000

# BioGRID columns to read by name - this avoids loading all the columns
BIOGRID_USECOLS = [
    'Organism ID Interactor A',
    'Organism ID Interactor B',
    'SWISS-PROT Accessions Interactor A',
    'TREMBL Accessions Interactor A',
    'SWISS-PROT Accessions Interactor B',
    'TREMBL Accessions Interactor B',
    'Experimental System',
    'Experimental System Type',
]

# Standardised output column names
OUTPUT_COLUMNS = ['protein_a', 'protein_b', 'source', 'confidence_score', 'evidence_type']

# UniProt accession pattern for HuMAP validation with optional isoform suffix
# Matches: P12345, Q9UKT4-2, A0A0B4J2C3, A0A0B4J2C3-1
_UNIPROT_RE = re.compile(
    r'^[OPQ][0-9][A-Z0-9]{3}[0-9](-\d+)?$'
    r'|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}(-\d+)?$'
)

#--------------------Helper Functions-------------------------------

def _is_valid_uniprot(identifier: str) -> bool:
    """Check if a string matches the UniProt accession pattern.
    Args:
        identifier: Protein identifier to validate.
    Returns:
        True if it matches the UniProt accession regex.
    """
    return bool(_UNIPROT_RE.match(str(identifier)))


def _strip_taxonomy_prefix(ensembl_id: str) -> str:
    """Strip '9606.' prefix from STRING Ensembl protein IDs.
    Args:
        ensembl_id: STRING-format ID like '9606.ENSP00000000233'.
    Returns:
        Bare Ensembl protein ID like 'ENSP00000000233'.
    """
    return ensembl_id.removeprefix('9606.')

def _normalise_string_score(score: int) -> float:
    """Normalise STRING combined score from 0-1000 to 0.0-1.0.
    Args:
        score: Integer score from STRING (0-1000).
    Returns:
        Float score normalised to 0.0-1.0.
    """
    return score / STRING_MAX_SCORE

def _extract_first_uniprot(accession_field: str) -> Optional[str]:
    """Extract the first valid UniProt accession from a BioGRID field.
    BioGRID SWISS-PROT columns contain a single accession or '-' if missing.
    TREMBL columns may contain pipe-delimited multiple accessions.
    Args:
        accession_field: Raw BioGRID accession string,
            e.g. 'P45985' or 'Q59H94|F6THM6' or '-'.
    Returns:
        First valid UniProt accession, or None if field is '-' or empty.
    """
    if not accession_field or accession_field == '-':
        return None
    # Take first accession if pipe-delimited
    first = accession_field.split('|')[0].strip()
    return first if first and first != '-' else None

#---------------------------------------STRING Parser-------------------------------------------------

def load_string(filepath: Optional[str] = None, min_score: int = 0, verbose: bool = False) -> pd.DataFrame:
    """Load STRING protein interaction network for human (taxon 9606).
    Read the SPACE-delimited STRING links file, strip the '9606.' taxonomy prefix from Ensembl protein IDs, and normalises combined scores from 0-1000 to 0.0-1.0.
    Args:
        filepath: Path to STRING links file. Defaults to data/ppi/ location.
        min_score: Minimum combined_score to retain (0-1000 scale, pre-normalisation). Filters at read time to reduce memory.
        verbose: Print progress information.
    Returns:
        DataFrame with standardised OUTPUT_COLUMNS.
        protein_a and protein_b contain bare ENSP IDs.
    """
    if filepath is None:
        filepath = str(DEFAULT_DATA_DIR / STRING_LINKS_FILE)

    if verbose:
        print(f"  Loading STRING from: {filepath}", file=sys.stderr)

    df = pd.read_csv(filepath, sep=' ', dtype={'combined_score': np.int16}, engine='c')

    if verbose:
        print(f"  STRING raw: {len(df):,} interactions", file=sys.stderr)

    # Filter by minimum score before processing
    if min_score > 0:
        df = df[df['combined_score'] >= min_score]
        if verbose:
            print(f"  STRING after score >= {min_score}: {len(df):,} interactions",
                  file=sys.stderr)

    # Strip taxonomy prefix from IDs
    df['protein1'] = df['protein1'].map(_strip_taxonomy_prefix)
    df['protein2'] = df['protein2'].map(_strip_taxonomy_prefix)

    # Build standardised output
    result = pd.DataFrame({'protein_a': df['protein1'], 'protein_b': df['protein2'], 'source': 'STRING', 'confidence_score': df['combined_score'].map(_normalise_string_score), 'evidence_type': 'combined'})
    return result.reset_index(drop=True)

#----------------------------------BioGRID Parser-------------------------------------------------

def load_biogrid(filepath: Optional[str] = None, physical_only: bool = True, verbose: bool = False) -> pd.DataFrame:
    """Load BioGRID interactions filtered to human (taxonomy 9606).
    Reads the TAB-delimited BioGRID tab3 file in chunks, filters to rows where both interactors are human (Organism ID == 9606), and extracts UniProt accessions from SWISS-PROT and TREMBL columns.
    UniProt accession is resolved by preferring SWISS-PROT (manually reviewed) over TREMBL (not manually reviewed). Interactions where neither interactor has a UniProt accession are dropped.
    Args:
        filepath: Path to BioGRID tab3 file. Defaults to data/ppi/ location.
        physical_only: If True, filter to 'physical' experimental systemtype only. Excludes genetic interactions. Default True.
        verbose: Print progress information.
    Returns:
        DataFrame with standardised OUTPUT_COLUMNS.
        protein_a and protein_b contain UniProt accessions.
        confidence_score is NaN (BioGRID has no scores).
        evidence_type contains the Experimental System string.
    """
    if filepath is None:
        filepath = str(DEFAULT_DATA_DIR / BIOGRID_FILE)

    if verbose:
        print(f"  Loading BioGRID from: {filepath}", file=sys.stderr)

    # Read in chunks to manage memory (1.5 GB file - all organisms)
    chunks = []
    for chunk in pd.read_csv(
        filepath,
        sep='\t',
        usecols=BIOGRID_USECOLS,
        dtype=str,
        encoding='utf-8',
        on_bad_lines='skip',
        chunksize=CHUNK_SIZE,
    ):
        # Filter to human-human interactions
        mask = (
            (chunk['Organism ID Interactor A'] == HUMAN_TAXONOMY_ID) &
            (chunk['Organism ID Interactor B'] == HUMAN_TAXONOMY_ID)
        )
        if physical_only:
            mask = mask & (chunk['Experimental System Type'] == 'physical')

        filtered = chunk[mask]
        if len(filtered) > 0:
            chunks.append(filtered)

    if not chunks:
        if verbose:
            print("  BioGRID: no human interactions found", file=sys.stderr)
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = pd.concat(chunks, ignore_index=True)

    if verbose:
        print(f"  BioGRID human physical: {len(df):,} interactions", file=sys.stderr)

    # Extract UniProt accessions: prefer SWISS-PROT over TREMBL
    df['_uniprot_a'] = df['SWISS-PROT Accessions Interactor A'].map(_extract_first_uniprot)
    df.loc[df['_uniprot_a'].isna(), '_uniprot_a'] = df.loc[df['_uniprot_a'].isna(), 'TREMBL Accessions Interactor A'].map(_extract_first_uniprot)

    df['_uniprot_b'] = df['SWISS-PROT Accessions Interactor B'].map(_extract_first_uniprot)
    df.loc[df['_uniprot_b'].isna(), '_uniprot_b'] = df.loc[df['_uniprot_b'].isna(), 'TREMBL Accessions Interactor B'].map(_extract_first_uniprot)

    # Drop rows missing UniProt for either interactor
    df = df.dropna(subset=['_uniprot_a', '_uniprot_b'])

    if verbose:
        print(f"  BioGRID with UniProt IDs: {len(df):,} interactions", file=sys.stderr)

    result = pd.DataFrame({
        'protein_a': df['_uniprot_a'].values,
        'protein_b': df['_uniprot_b'].values,
        'source': 'BioGRID',
        'confidence_score': np.nan,
        'evidence_type': df['Experimental System'].values,
    })
    return result.reset_index(drop=True)


#----------------------------------HuRI Parser------------------------------------------------------

def load_huri(filepath: Optional[str] = None, verbose: bool = False) -> pd.DataFrame:
    """Load HuRI binary interactome (Yeast 2-Hybrid {Y2H} screening).
    Reads the TAB-delimited HuRI file (NO header row). Both columns contain Ensembl GENE IDs (ENSG format), not protein IDs. 
    These must be mapped to UniProt accessions via the ID mapper in a separate step.
    Args:
        filepath: Path to HuRI.tsv file. Defaults to data/ppi/ location.
        verbose: Print progress information.
    Returns:
        DataFrame with standardised OUTPUT_COLUMNS.
        protein_a and protein_b contain Ensembl GENE IDs (ENSG format).
        confidence_score is NaN (binary Y2H has no scores).
        evidence_type is 'Y2H'.
    Note:
        protein_a and protein_b contain ENSG IDs, not UniProt accessions.
        Use id_mapper.map_dataframe_to_uniprot() to convert before cross-database comparison.
    """
    if filepath is None:
        filepath = str(DEFAULT_DATA_DIR / HURI_FILE)

    if verbose:
        print(f"  Loading HuRI from: {filepath}", file=sys.stderr)

    df = pd.read_csv(
        filepath,
        sep='\t',
        header=None,
        names=['protein_a', 'protein_b'],
    )

    if verbose:
        print(f"  HuRI: {len(df):,} binary interactions", file=sys.stderr)

    result = pd.DataFrame({
        'protein_a': df['protein_a'],
        'protein_b': df['protein_b'],
        'source': 'HuRI',
        'confidence_score': np.nan,
        'evidence_type': 'Y2H',
    })
    return result.reset_index(drop=True)


#----------------------------------------HuMAP Parser-----------------------------------------------------------

def load_humap(filepath: Optional[str] = None, min_probability: float = 0.0, validate_ids: bool = True, verbose: bool = False) -> pd.DataFrame:
    """Load hu.MAP 2.0 pairwise protein interactions.
    Reads the SPACE-delimited HuMAP pairwise file (NO header row). Columns are protein_a (UniProt), protein_b (UniProt) and probability (0.0-1.0).
    Args:
        filepath: Path to HuMAP pairsWprob file. Defaults to data/ppi/ location.
        min_probability: Minimum probability score to retain. Filters at read time to reduce memory for the 17.5M-row file.
        validate_ids: If True, verify both protein IDs match the UniProt accession format and skip rows with non-UniProt IDs (e.g. Ensembl gene IDs or gene symbols). Default True.
        verbose: Print progress information.
    Returns:
        DataFrame with standardised OUTPUT_COLUMNS.
        protein_a and protein_b contain UniProt accessions.
        confidence_score contains the probability (0.0-1.0).
        evidence_type is 'mass_spec_cofractionation'.
    """
    if filepath is None:
        filepath = str(DEFAULT_DATA_DIR / HUMAP_FILE)

    if verbose:
        print(f"  Loading HuMAP from: {filepath}", file=sys.stderr)

    df = pd.read_csv(
        filepath,
        sep=r'\s+',
        header=None,
        names=['protein_a', 'protein_b', 'probability'],
        dtype={'probability': np.float32},
        engine='c',
    )

    if verbose:
        print(f"  HuMAP raw: {len(df):,} pairwise interactions", file=sys.stderr)

    if min_probability > 0.0:
        df = df[df['probability'] >= min_probability]
        if verbose:
            print(f"  HuMAP after prob >= {min_probability}: {len(df):,} interactions",
                  file=sys.stderr)

    # Validate that both IDs are UniProt accessions
    if validate_ids:
        valid_mask = (
            df['protein_a'].map(_is_valid_uniprot) &
            df['protein_b'].map(_is_valid_uniprot)
        )
        n_invalid = (~valid_mask).sum()
        if n_invalid > 0:
            invalid_rows = df[~valid_mask]
            invalid_ids = set()
            for _, row in invalid_rows.iterrows():
                if not _UNIPROT_RE.match(str(row['protein_a'])):
                    invalid_ids.add(str(row['protein_a']))
                if not _UNIPROT_RE.match(str(row['protein_b'])):
                    invalid_ids.add(str(row['protein_b']))
            warnings.warn(
                f"HuMAP: skipped {n_invalid} row(s) with non-UniProt IDs: "
                f"{', '.join(sorted(invalid_ids))}",
                stacklevel=2,
            )
            df = df[valid_mask]

    result = pd.DataFrame({
        'protein_a': df['protein_a'],
        'protein_b': df['protein_b'],
        'source': 'HuMAP',
        'confidence_score': df['probability'],
        'evidence_type': 'mass_spec_cofractionation',
    })
    return result.reset_index(drop=True)


#------------------------------------------------Unified Loader--------------------------------------------------------

def load_all_databases(data_dir: Optional[str] = None, string_min_score: int = 0, humap_min_probability: float = 0.0, biogrid_physical_only: bool = True, verbose: bool = False) -> dict[str, pd.DataFrame]:
    """Load all 4 PPI databases and return as a dictionary of DataFrames.
    Args:
        data_dir: Directory containing database files. Defaults to data/ppi/.
        string_min_score: Minimum STRING combined_score (0-1000).
        humap_min_probability: Minimum HuMAP probability.
        biogrid_physical_only: Filter BioGRID to physical interactions only.
        verbose: Print progress information.
    Returns:
        Dict mapping database name to DataFrame: {'STRING': df, 'BioGRID': df, 'HuRI': df, 'HuMAP': df}.
        Each DataFrame has the standard OUTPUT_COLUMNS.
    """
    if data_dir is None:
        data_dir = str(DEFAULT_DATA_DIR)

    base = Path(data_dir)
    results = {}

    if verbose:
        print("Loading PPI databases...", file=sys.stderr)

    results['STRING'] = load_string(
        filepath=str(base / STRING_LINKS_FILE),
        min_score=string_min_score,
        verbose=verbose,
    )

    results['BioGRID'] = load_biogrid(
        filepath=str(base / BIOGRID_FILE),
        physical_only=biogrid_physical_only,
        verbose=verbose,
    )

    results['HuRI'] = load_huri(
        filepath=str(base / HURI_FILE),
        verbose=verbose,
    )

    results['HuMAP'] = load_humap(
        filepath=str(base / HUMAP_FILE),
        min_probability=humap_min_probability,
        verbose=verbose,
    )

    if verbose:
        print("\nDatabase summary:", file=sys.stderr)
        for name, df in results.items():
            print(f"  {name}: {len(df):,} interactions", file=sys.stderr)

    return results


#-------------------------------CLI Entry Point-----------------------------------------

def build_argument_parser() -> argparse.ArgumentParser:
    """Create and return the argument parser for database_loaders."""
    parser = argparse.ArgumentParser(
        description="Load and standardise PPI database files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage (as importable module):
    from database_loaders import load_string, load_biogrid, load_huri, load_humap
    df = load_string("data/ppi/9606.protein.links.v12.0.txt", min_score=700)
    df = load_biogrid("data/ppi/BIOGRID-ALL-5.0.253.tab3.txt")
    df = load_huri("data/ppi/HuRI.tsv")
    df = load_humap("data/ppi/humap2_ppis_ACC_20200821.pairsWprob", min_probability=0.5)

Usage (standalone):
    python database_loaders.py --data-dir data/ppi --database all --output interactions.csv -v
    python database_loaders.py --data-dir data/ppi --database string --min-string-score 700 -v
""",
)
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Directory containing database files (default: data/ppi/)",
    )
    parser.add_argument(
        "--database",
        choices=['string', 'biogrid', 'huri', 'humap', 'all'],
        default='all',
        help="Which database to load (default: all)",
    )
    parser.add_argument(
        "--output",
        metavar="CSV_PATH",
        help="Output CSV path. If not specified, prints summary only.",
    )
    parser.add_argument(
        "--min-string-score",
        type=int,
        default=0,
        help="Minimum STRING combined_score (0-1000). Default: 0. "
             "Confidence tiers: 150 (low), 400 (medium), 700 (high), 900 (highest)",
    )
    parser.add_argument(
        "--min-humap-prob",
        type=float,
        default=0.0,
        help="Minimum HuMAP probability (0.0-1.0). Default: 0.0",
    )
    parser.add_argument(
        "--biogrid-physical-only",
        action="store_true",
        default=True,
        help="Filter BioGRID to physical interactions only (default: True)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress information",
    )
    return parser


def main() -> None:
    """Run the database loader CLI."""
    parser = build_argument_parser()
    args = parser.parse_args()

    base = Path(args.data_dir)

    if args.database == 'all':
        results = load_all_databases(
            data_dir=args.data_dir,
            string_min_score=args.min_string_score,
            humap_min_probability=args.min_humap_prob,
            biogrid_physical_only=args.biogrid_physical_only,
            verbose=args.verbose,
        )
        if args.output:
            combined = pd.concat(results.values(), ignore_index=True)
            combined.to_csv(args.output, index=False)
            print(f"\nCombined output: {len(combined):,} interactions -> {args.output}")
        else:
            print("\nSummary:")
            for name, df in results.items():
                print(f"  {name}: {len(df):,} interactions")
    else:
        db_map = {
            'string': lambda: load_string(
                str(base / STRING_LINKS_FILE),
                min_score=args.min_string_score,
                verbose=args.verbose,
            ),
            'biogrid': lambda: load_biogrid(
                str(base / BIOGRID_FILE),
                physical_only=args.biogrid_physical_only,
                verbose=args.verbose,
            ),
            'huri': lambda: load_huri(
                str(base / HURI_FILE),
                verbose=args.verbose,
            ),
            'humap': lambda: load_humap(
                str(base / HUMAP_FILE),
                min_probability=args.min_humap_prob,
                verbose=args.verbose,
            ),
        }

        df = db_map[args.database]()
        print(f"\n{args.database.upper()}: {len(df):,} interactions")

        if args.output:
            df.to_csv(args.output, index=False)
            print(f"Output saved to: {args.output}")

if __name__ == "__main__":
    main()
