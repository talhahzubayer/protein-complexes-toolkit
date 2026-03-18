"""
Offline pathogenicity and stability scoring for protein complex variants.

Integrates two pre-computed data sources to score variant effects without
any API dependency:

1. AlphaMissense (DeepMind) — deep-learning pathogenicity predictions for
   all possible human amino acid substitutions (~216M variants).
2. AFDB FoldX export (EBI) — pre-computed FoldX ΔΔG values on monomeric
   AlphaFold structures with per-position pLDDT (~209M substitutions).

Architecture:
    - Offline-first: reads local TSV/CSV files, no API calls
    - Lazy loading: streams files and filters for pipeline proteins only
    - Integrates into toolkit.py via --protvar flag (requires --variants)
    - Standalone CLI for score lookup and coverage statistics

Data sources:
    - AlphaMissense_aa_substitutions.tsv (pathogenicity scores)
    - afdb_foldx_export_20250210.csv (monomeric FoldX ΔΔG + pLDDT)

Usage (standalone):
    python protvar_client.py summary
    python protvar_client.py lookup --protein P61981
    python protvar_client.py lookup --protein P61981 --position 4

Usage (via toolkit.py):
    python toolkit.py --dir DIR --output results.csv --interface --pae --enrich ALIASES --variants --protvar
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Optional, Union


# ── Constants ────────────────────────────────────────────────────────

# Default file paths (both in data/stability/)
DEFAULT_FOLDX_EXPORT = (
    Path(__file__).parent / "data" / "stability" / "afdb_foldx_export_20250210.csv"
)
DEFAULT_AM_FILE = (
    Path(__file__).parent / "data" / "stability" / "AlphaMissense_aa_substitutions.tsv"
)

# FoldX destabilisation threshold (kcal/mol, literature convention)
FOLDX_DESTABILISING_THRESHOLD = 1.6

# Display limit for variant detail strings
PROTVAR_DETAILS_DISPLAY_LIMIT = 20

# Progress reporting interval (lines)
CHUNK_LOG_INTERVAL = 5_000_000

# Variant detail parsing pattern (shared with stability_scorer.py)
_VARIANT_DETAIL_PATTERN = re.compile(r'^([A-Z*])(\d+)([A-Z*]):')

# AlphaMissense variant parsing: REF POS ALT (e.g. M1A, V2G, K81P)
_AM_VARIANT_PATTERN = re.compile(r'^([A-Z])(\d+)([A-Z])$')

# CSV columns added when --protvar is used (8 columns, per-chain a/b)
CSV_FIELDNAMES_PROTVAR = [
    'protvar_am_mean_a', 'protvar_am_mean_b',
    'protvar_foldx_mean_a', 'protvar_foldx_mean_b',
    'protvar_am_n_pathogenic_a', 'protvar_am_n_pathogenic_b',
    'protvar_details_a', 'protvar_details_b',
]


# ── Section 1: AlphaMissense Loading ─────────────────────────────────

def _parse_am_variant(protein_variant: str) -> Optional[tuple[str, int, str]]:
    """Parse AlphaMissense protein_variant string to (ref_aa, position, alt_aa).

    Args:
        protein_variant: Variant string, e.g. 'M1A' (ref + position + alt).

    Returns:
        Tuple of (ref_aa, position, alt_aa), or None if unparseable.
    """
    match = _AM_VARIANT_PATTERN.match(protein_variant)
    if match:
        return match.group(1), int(match.group(2)), match.group(3)
    return None


def load_alphamissense_scores(
    filepath: Path,
    accessions: frozenset[str],
    variant_positions: Optional[dict[str, set[int]]] = None,
    verbose: bool = False,
) -> dict[str, dict[tuple[int, str], dict]]:
    """Stream AlphaMissense TSV and load scores for specified proteins.

    The file has 3 '#'-prefixed comment lines, a header, then tab-separated
    data: uniprot_id, protein_variant, am_pathogenicity, am_class.

    Args:
        filepath: Path to AlphaMissense_aa_substitutions.tsv.
        accessions: Set of UniProt accessions to load (base accessions).
        variant_positions: Optional dict mapping accession to set of positions
            to filter for.  If None, all positions for matching accessions
            are loaded.
        verbose: Print progress to stderr.

    Returns:
        Dict keyed by accession, sub-keyed by (position, alt_aa):
        {'P61981': {(1, 'A'): {'am_score': 0.363, 'am_class': 'ambiguous'}}}
    """
    if not filepath.exists():
        if verbose:
            print(f"  AlphaMissense file not found: {filepath}", file=sys.stderr)
        return {}

    index: dict[str, dict[tuple[int, str], dict]] = {}
    scanned = 0
    kept = 0

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            # Skip comment lines (#-prefixed header)
            if line.startswith('#'):
                continue
            # Skip the column header line
            if line.startswith('uniprot_id'):
                continue

            scanned += 1
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 4:
                continue

            uniprot_id = parts[0]
            if uniprot_id not in accessions:
                if scanned % CHUNK_LOG_INTERVAL == 0 and verbose:
                    print(f"    AlphaMissense: scanned {scanned:,} rows, "
                          f"kept {kept:,}...", file=sys.stderr)
                continue

            parsed = _parse_am_variant(parts[1])
            if parsed is None:
                continue

            ref_aa, pos, alt_aa = parsed

            # Position filtering
            if variant_positions is not None:
                acc_positions = variant_positions.get(uniprot_id)
                if acc_positions is not None and pos not in acc_positions:
                    if scanned % CHUNK_LOG_INTERVAL == 0 and verbose:
                        print(f"    AlphaMissense: scanned {scanned:,} rows, "
                              f"kept {kept:,}...", file=sys.stderr)
                    continue

            try:
                score = float(parts[2])
            except (ValueError, IndexError):
                continue

            am_class = parts[3].strip() if len(parts) > 3 else ''

            if uniprot_id not in index:
                index[uniprot_id] = {}
            index[uniprot_id][(pos, alt_aa)] = {
                'am_score': score,
                'am_class': am_class,
            }
            kept += 1

            if scanned % CHUNK_LOG_INTERVAL == 0 and verbose:
                print(f"    AlphaMissense: scanned {scanned:,} rows, "
                      f"kept {kept:,}...", file=sys.stderr)

    if verbose:
        print(f"  AlphaMissense: scanned {scanned:,} rows, "
              f"loaded {kept:,} scores for {len(index):,} proteins",
              file=sys.stderr)
    return index


# ── Section 2: AFDB FoldX Export Loading ─────────────────────────────

def load_foldx_export(
    filepath: Path,
    accessions: frozenset[str],
    variant_positions: Optional[dict[str, set[int]]] = None,
    verbose: bool = False,
) -> dict[str, dict[tuple[int, str], dict]]:
    """Stream AFDB FoldX export CSV and load DDG/pLDDT for specified proteins.

    The CSV has columns: uniprot_accession, uniprot_position,
    alphafold_fragment_id, alphafold_fragment_position, wild_type,
    mutated_type, foldx_ddg, plddt.

    Args:
        filepath: Path to afdb_foldx_export_20250210.csv.
        accessions: Set of UniProt accessions to load (base accessions).
        variant_positions: Optional dict mapping accession to set of positions
            to filter for.
        verbose: Print progress to stderr.

    Returns:
        Dict keyed by accession, sub-keyed by (position, alt_aa):
        {'P61981': {(1, 'A'): {'foldx_ddg': 0.114505, 'plddt': 54.50}}}
    """
    if not filepath.exists():
        if verbose:
            print(f"  FoldX export not found: {filepath}", file=sys.stderr)
        return {}

    index: dict[str, dict[tuple[int, str], dict]] = {}
    scanned = 0
    kept = 0

    with open(filepath, encoding="utf-8", newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip header
        if header is None:
            return {}

        for row in reader:
            scanned += 1
            if len(row) < 8:
                continue

            acc = row[0]
            if acc not in accessions:
                if scanned % CHUNK_LOG_INTERVAL == 0 and verbose:
                    print(f"    FoldX export: scanned {scanned:,} rows, "
                          f"kept {kept:,}...", file=sys.stderr)
                continue

            try:
                pos = int(row[1])
            except ValueError:
                continue

            # Position filtering
            if variant_positions is not None:
                acc_positions = variant_positions.get(acc)
                if acc_positions is not None and pos not in acc_positions:
                    if scanned % CHUNK_LOG_INTERVAL == 0 and verbose:
                        print(f"    FoldX export: scanned {scanned:,} rows, "
                              f"kept {kept:,}...", file=sys.stderr)
                    continue

            alt_aa = row[5]  # mutated_type
            try:
                ddg = float(row[6])
                plddt = float(row[7])
            except (ValueError, IndexError):
                continue

            if acc not in index:
                index[acc] = {}
            index[acc][(pos, alt_aa)] = {
                'foldx_ddg': ddg,
                'plddt': plddt,
            }
            kept += 1

            if scanned % CHUNK_LOG_INTERVAL == 0 and verbose:
                print(f"    FoldX export: scanned {scanned:,} rows, "
                      f"kept {kept:,}...", file=sys.stderr)

    if verbose:
        print(f"  FoldX export: scanned {scanned:,} rows, "
              f"loaded {kept:,} scores for {len(index):,} proteins",
              file=sys.stderr)
    return index


# ── Section 3: Index Building ────────────────────────────────────────

def build_protvar_index(
    accessions: set[str],
    variant_positions: Optional[dict[str, set[int]]] = None,
    foldx_path: Optional[Union[str, Path]] = None,
    am_path: Optional[Union[str, Path]] = None,
    verbose: bool = False,
) -> dict[str, dict[tuple[int, str], dict]]:
    """Build combined index from AlphaMissense + FoldX offline data.

    Streams both files, filtering for the given accessions and optional
    variant positions.  The returned index merges scores from both sources
    into a single nested dict.

    Args:
        accessions: Set of UniProt accessions to load.
        variant_positions: Optional position filter per accession.
        foldx_path: Path to AFDB FoldX export CSV.  None → default path.
        am_path: Path to AlphaMissense TSV.  None → default path.
        verbose: Print progress to stderr.

    Returns:
        Merged index: {accession: {(pos, alt): {am_score, am_class,
        foldx_ddg, plddt}}}
    """
    foldx_file = Path(foldx_path) if foldx_path else DEFAULT_FOLDX_EXPORT
    am_file = Path(am_path) if am_path else DEFAULT_AM_FILE

    # Strip isoform suffixes for base accession lookup
    base_accessions: set[str] = set()
    for acc in accessions:
        base_accessions.add(acc.split('-')[0] if '-' in acc else acc)
    frozen_accs = frozenset(base_accessions)

    if verbose:
        print(f"  Building offline score index for {len(frozen_accs):,} proteins...",
              file=sys.stderr)

    # Load AlphaMissense scores
    am_index = load_alphamissense_scores(
        am_file, frozen_accs, variant_positions, verbose,
    )

    # Load FoldX export scores
    foldx_index = load_foldx_export(
        foldx_file, frozen_accs, variant_positions, verbose,
    )

    # Merge into combined index
    all_accs = set(am_index.keys()) | set(foldx_index.keys())
    index: dict[str, dict[tuple[int, str], dict]] = {}

    for acc in all_accs:
        am_data = am_index.get(acc, {})
        fx_data = foldx_index.get(acc, {})
        all_keys = set(am_data.keys()) | set(fx_data.keys())

        merged: dict[tuple[int, str], dict] = {}
        for key in all_keys:
            entry: dict = {}
            if key in am_data:
                entry.update(am_data[key])
            if key in fx_data:
                entry.update(fx_data[key])
            merged[key] = entry
        index[acc] = merged

    if verbose:
        n_am = sum(1 for acc in index.values()
                   for v in acc.values() if 'am_score' in v)
        n_fx = sum(1 for acc in index.values()
                   for v in acc.values() if 'foldx_ddg' in v)
        print(f"  Combined index: {len(index):,} proteins, "
              f"{n_am:,} AlphaMissense scores, {n_fx:,} FoldX DDG values",
              file=sys.stderr)

    return index


# ── Section 4: Score Lookup ──────────────────────────────────────────

def lookup_score(
    index: dict,
    accession: str,
    position: int,
    alt_aa: str,
) -> Optional[dict]:
    """Look up scores for a specific variant in the combined index.

    Args:
        index: Combined index from build_protvar_index().
        accession: UniProt accession (isoform suffixes stripped automatically).
        position: Residue position (1-based).
        alt_aa: Mutant amino acid (single letter).

    Returns:
        Dict with available scores ({am_score, am_class, foldx_ddg, plddt}),
        or None if not found.
    """
    base = accession.split('-')[0] if '-' in accession else accession
    acc_data = index.get(base)
    if acc_data is None:
        return None
    return acc_data.get((position, alt_aa))


# ── Section 5: Variant Detail Parsing ────────────────────────────────

def _parse_variant_details_for_protvar(
    details_str: str,
) -> list[tuple[str, int, str]]:
    """Parse variant_details string to (ref_aa, position, alt_aa) tuples.

    Parses the pipe-separated variant detail strings produced by
    variant_mapper.format_variant_details().  Format:
        K81P:interface_core:pathogenic|E82K:interface_rim:VUS

    Args:
        details_str: Pipe-separated variant detail string.

    Returns:
        List of (ref_aa, position, alt_aa) tuples.
    """
    if not details_str:
        return []

    variants = []
    for part in details_str.split('|'):
        part = part.strip()
        if part.startswith('...(+'):
            continue
        match = _VARIANT_DETAIL_PATTERN.match(part)
        if match:
            ref = match.group(1)
            pos = int(match.group(2))
            alt = match.group(3)
            variants.append((ref, pos, alt))
    return variants


# ── Section 6: Formatting ────────────────────────────────────────────

def format_protvar_details(
    scored_variants: list[dict],
    limit: int = PROTVAR_DETAILS_DISPLAY_LIMIT,
) -> str:
    """Format scored variants into a pipe-separated summary string.

    Format: REF{POS}ALT:am={score}:{class}:foldx={ddg}

    Args:
        scored_variants: List of dicts with keys 'ref_aa', 'position',
            'alt_aa', 'am_score', 'am_class', 'foldx_ddg'.
        limit: Maximum number of variants to display.

    Returns:
        Pipe-separated string, e.g. 'M1A:am=0.36:ambiguous:foldx=0.11'.
        Empty string if no variants.
    """
    if not scored_variants:
        return ''

    details = []
    for var in scored_variants[:limit]:
        ref = var.get('ref_aa', '?')
        pos = var.get('position', '?')
        alt = var.get('alt_aa', '?')

        am = var.get('am_score')
        am_str = f"{am:.2f}" if am is not None else '-'

        am_cls = var.get('am_class', '-')
        if not am_cls:
            am_cls = '-'

        ddg = var.get('foldx_ddg')
        ddg_str = f"{ddg:.2f}" if ddg is not None else '-'

        details.append(f"{ref}{pos}{alt}:am={am_str}:{am_cls}:foldx={ddg_str}")

    result = '|'.join(details)

    remaining = len(scored_variants) - limit
    if remaining > 0:
        result += f"|...(+{remaining} more)"

    return result


# ── Section 7: Chain Scoring ─────────────────────────────────────────

def _score_chain_variants_protvar(
    accession: str,
    details_str: str,
    protvar_index: dict,
) -> dict:
    """Score variants for one chain using the combined offline index.

    Args:
        accession: UniProt accession for this chain.
        details_str: Pipe-separated variant detail string from Phase C.
        protvar_index: Combined index from build_protvar_index().

    Returns:
        Dict with keys:
            'am_mean': float or '' (mean AlphaMissense score),
            'foldx_mean': float or '' (mean FoldX ΔΔG),
            'am_n_pathogenic': int (count of pathogenic AM variants),
            'details': str (formatted detail string),
    """
    base_acc = accession.split('-')[0] if '-' in accession else accession
    acc_data = protvar_index.get(base_acc, {})

    variants = _parse_variant_details_for_protvar(details_str)

    am_scores: list[float] = []
    foldx_ddgs: list[float] = []
    n_pathogenic = 0
    scored_list: list[dict] = []

    for ref_aa, pos, alt_aa in variants:
        entry = acc_data.get((pos, alt_aa), {})

        am = entry.get('am_score')
        am_cls = entry.get('am_class', '')
        ddg = entry.get('foldx_ddg')

        if am is not None:
            am_scores.append(am)
            if am_cls == 'pathogenic':
                n_pathogenic += 1
        if ddg is not None:
            foldx_ddgs.append(ddg)

        scored_list.append({
            'ref_aa': ref_aa,
            'position': pos,
            'alt_aa': alt_aa,
            'am_score': am,
            'am_class': am_cls,
            'foldx_ddg': ddg,
        })

    return {
        'am_mean': round(mean(am_scores), 4) if am_scores else '',
        'foldx_mean': round(mean(foldx_ddgs), 4) if foldx_ddgs else '',
        'am_n_pathogenic': n_pathogenic,
        'details': format_protvar_details(scored_list),
    }


# ── Section 8: Annotation ───────────────────────────────────────────

def annotate_results_with_protvar(
    results: list[dict],
    protvar_index: dict,
    verbose: bool = False,
) -> None:
    """Annotate result rows with offline AlphaMissense + FoldX data (in-place).

    Main entry point from toolkit.py.  For each complex:
    1. Parses variant_details_a/b to extract variants
    2. Looks up AlphaMissense scores and FoldX DDG from offline index
    3. Computes means, pathogenic counts
    4. Formats detail strings

    Args:
        results: List of per-complex result dicts.  Modified in-place.
        protvar_index: Combined index from build_protvar_index().
        verbose: Print progress to stderr.
    """
    annotated = 0

    for row in results:
        for suffix in ('a', 'b'):
            acc = row.get(f'protein_{suffix}', '')
            details_str = row.get(f'variant_details_{suffix}', '')

            if acc and details_str:
                chain_result = _score_chain_variants_protvar(
                    acc, details_str, protvar_index,
                )
                row[f'protvar_am_mean_{suffix}'] = chain_result['am_mean']
                row[f'protvar_foldx_mean_{suffix}'] = chain_result['foldx_mean']
                row[f'protvar_am_n_pathogenic_{suffix}'] = chain_result['am_n_pathogenic']
                row[f'protvar_details_{suffix}'] = chain_result['details']

                if chain_result['am_mean'] != '':
                    annotated += 1
            else:
                row[f'protvar_am_mean_{suffix}'] = ''
                row[f'protvar_foldx_mean_{suffix}'] = ''
                row[f'protvar_am_n_pathogenic_{suffix}'] = ''
                row[f'protvar_details_{suffix}'] = ''

    if verbose:
        print(f"  ProtVar offline: annotated {annotated} chains with scores "
              f"across {len(results)} complexes", file=sys.stderr)


# ── Section 9: Standalone CLI ────────────────────────────────────────

def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for standalone use."""
    parser = argparse.ArgumentParser(
        description="Offline AlphaMissense + monomeric FoldX scoring for "
                    "protein variants.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--foldx-export",
        default=str(DEFAULT_FOLDX_EXPORT),
        help=f"Path to AFDB FoldX export CSV (default: {DEFAULT_FOLDX_EXPORT}).",
    )
    parser.add_argument(
        "--am-file",
        default=str(DEFAULT_AM_FILE),
        help=f"Path to AlphaMissense TSV (default: {DEFAULT_AM_FILE}).",
    )

    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # summary sub-command
    subparsers.add_parser("summary", help="Print data file statistics.")

    # lookup sub-command
    lookup_parser = subparsers.add_parser(
        "lookup", help="Look up scores for a specific protein.",
    )
    lookup_parser.add_argument(
        "--protein", required=True, help="UniProt accession.",
    )
    lookup_parser.add_argument(
        "--position", type=int, default=None,
        help="Residue position (optional, show all if omitted).",
    )

    return parser


def _cli_summary(foldx_path: str, am_path: str) -> None:
    """Print summary statistics for both data files (streaming, no full load)."""
    foldx_file = Path(foldx_path)
    am_file = Path(am_path)

    print("ProtVar Offline Data Summary")
    print("=" * 50)

    # AlphaMissense
    if am_file.exists():
        n_lines = 0
        proteins: set[str] = set()
        with open(am_file, encoding="utf-8") as f:
            for line in f:
                if line.startswith('#') or line.startswith('uniprot_id'):
                    continue
                parts = line.split('\t', 1)
                if parts:
                    proteins.add(parts[0])
                n_lines += 1
        print(f"\nAlphaMissense: {am_file.name}")
        print(f"  Rows: {n_lines:,}")
        print(f"  Proteins: {len(proteins):,}")
    else:
        print(f"\nAlphaMissense: NOT FOUND ({am_file})")

    # FoldX export
    if foldx_file.exists():
        n_lines = 0
        proteins = set()
        with open(foldx_file, encoding="utf-8", newline='') as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if row:
                    proteins.add(row[0])
                n_lines += 1
        print(f"\nAFDB FoldX Export: {foldx_file.name}")
        print(f"  Rows: {n_lines:,}")
        print(f"  Proteins: {len(proteins):,}")
    else:
        print(f"\nAFDB FoldX Export: NOT FOUND ({foldx_file})")


def _cli_lookup(
    foldx_path: str,
    am_path: str,
    accession: str,
    position: Optional[int],
) -> None:
    """Look up scores for a specific protein (or position)."""
    acc_set = {accession}
    pos_filter = None
    if position is not None:
        pos_filter = {accession: {position}}

    index = build_protvar_index(
        accessions=acc_set,
        variant_positions=pos_filter,
        foldx_path=foldx_path,
        am_path=am_path,
        verbose=True,
    )

    acc_data = index.get(accession, {})
    if not acc_data:
        print(f"No data found for {accession}", file=sys.stderr)
        return

    print(f"\nScores for {accession}:")
    print(f"{'Pos':>5} {'Alt':>3} {'AM Score':>9} {'AM Class':>12} "
          f"{'FoldX DDG':>10} {'pLDDT':>6}")
    print("-" * 55)

    for (pos, alt), scores in sorted(acc_data.items()):
        am = scores.get('am_score')
        am_str = f"{am:.4f}" if am is not None else '-'
        am_cls = scores.get('am_class', '-')
        ddg = scores.get('foldx_ddg')
        ddg_str = f"{ddg:.4f}" if ddg is not None else '-'
        plddt = scores.get('plddt')
        plddt_str = f"{plddt:.1f}" if plddt is not None else '-'
        print(f"{pos:>5} {alt:>3} {am_str:>9} {am_cls:>12} "
              f"{ddg_str:>10} {plddt_str:>6}")


def main() -> None:
    """CLI entry point."""
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.command == "summary":
        _cli_summary(args.foldx_export, args.am_file)
    elif args.command == "lookup":
        _cli_lookup(args.foldx_export, args.am_file,
                    args.protein, args.position)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
