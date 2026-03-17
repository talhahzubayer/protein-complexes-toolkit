"""
Stability scoring for AlphaFold2-predicted protein-protein complex variants.

Integrates EVE (Evolutionary model of Variant Effect) pre-computed pathogenicity
predictions with the variant mapping from Phase C. EVE provides deep-learning
pathogenicity scores for ~3,211 human proteins based on evolutionary sequence
conservation.

Architecture:
    - Offline-first: reads local EVE CSV files and UniProt ID mapping file
    - Lazy loading: only parses EVE CSVs for proteins present in the pipeline run
    - Integrates into toolkit.py via --stability flag (requires --variants)
    - Standalone CLI for EVE score lookup and coverage statistics

Data sources:
    - EVE_all_data/*.csv (per-protein EVE pathogenicity scores, keyed by entry name)
    - HUMAN_9606_idmapping.dat (UniProt accession-to-entry-name mapping)

Usage (standalone):
    python stability_scorer.py summary --stability-dir data/stability
    python stability_scorer.py lookup --stability-dir data/stability --protein P61981
    python stability_scorer.py lookup --stability-dir data/stability --protein P61981 --position 45

Usage (via toolkit.py):
    python toolkit.py --dir DIR --output results.csv --interface --pae --enrich ALIASES --variants --stability
    python toolkit.py --dir DIR --output results.csv --interface --pae --enrich ALIASES --variants --stability data/stability
"""

import argparse
import csv
import re
import sys
import warnings
from pathlib import Path
from statistics import mean
from typing import Optional, Union


# ── Constants ────────────────────────────────────────────────────────

# Default file paths
DEFAULT_STABILITY_DIR = Path(__file__).parent / "data" / "stability"
DEFAULT_EVE_DIR = DEFAULT_STABILITY_DIR / "EVE_all_data"
EVE_IDMAPPING_FILENAME = "HUMAN_9606_idmapping.dat"

# EVE CSV column names (from EVE bulk download, verified March 2026)
EVE_SCORE_COLUMN = "EVE_scores_ASM"
EVE_CLASS_COLUMN = "EVE_classes_75_pct_retained_ASM"
EVE_UNCERTAINTY_COLUMN = "uncertainty_ASM"
EVE_EVO_INDEX_COLUMN = "evolutionary_index_ASM"

# UniProt ID mapping file filter
IDMAPPING_ENTRY_NAME_TYPE = "UniProtKB-ID"

# EVE CSV filename suffix (all human EVE files end with _HUMAN.csv)
EVE_FILENAME_SUFFIX = "_HUMAN.csv"

# Display limit for stability details in CSV cells
STABILITY_DETAILS_DISPLAY_LIMIT = 20

# EVE classification labels (as they appear in EVE CSVs)
EVE_CLASS_PATHOGENIC = "Pathogenic"
EVE_CLASS_BENIGN = "Benign"
EVE_CLASS_UNCERTAIN = "Uncertain"

# Variant detail parsing pattern: REF{POS}ALT:context:clinical_significance
# Matches e.g. "K81P:interface_core:pathogenic" from variant_mapper format
_VARIANT_DETAIL_PATTERN = re.compile(r'^([A-Z*])(\d+)([A-Z*]):')

# CSV columns added when --stability is used
CSV_FIELDNAMES_STABILITY = [
    'eve_score_mean_a', 'eve_score_mean_b',
    'eve_n_pathogenic_a', 'eve_n_pathogenic_b',
    'eve_coverage_a', 'eve_coverage_b',
    'stability_details_a', 'stability_details_b',
]


# ── Section 1: EVE Entry-Name Mapping ────────────────────────────────

def load_eve_entry_name_map(
    map_path: Path,
    verbose: bool = False,
) -> dict[str, str]:
    """Parse UniProt ID mapping file to build accession-to-entry-name lookup.

    Reads the HUMAN_9606_idmapping.dat file (tab-separated, 3 columns:
    accession, type, value) and filters to rows where type == 'UniProtKB-ID'.
    These rows provide the UniProt entry name (e.g. '1433G_HUMAN') for each
    accession (e.g. 'P61981').

    Args:
        map_path: Path to HUMAN_9606_idmapping.dat file.
        verbose: Print progress to stderr.

    Returns:
        Dict mapping UniProt accession to entry name (e.g. {'P61981': '1433G_HUMAN'}).
        Empty dict if file not found (with warning).
    """
    if not map_path.exists():
        warnings.warn(
            f"UniProt ID mapping file not found: {map_path}. "
            f"EVE score lookup will be unavailable.",
            stacklevel=2,
        )
        return {}

    acc_to_entry: dict[str, str] = {}
    total_lines = 0

    with open(map_path, encoding='utf-8', errors='replace') as f:
        for line in f:
            total_lines += 1
            parts = line.rstrip('\n').split('\t')
            if len(parts) >= 3 and parts[1] == IDMAPPING_ENTRY_NAME_TYPE:
                acc_to_entry[parts[0]] = parts[2]

    if verbose:
        print(f"  ID mapping: {len(acc_to_entry):,} entry names from "
              f"{total_lines:,} lines in {map_path.name}", file=sys.stderr)

    return acc_to_entry


# ── Section 2: EVE Score Loading ─────────────────────────────────────

def load_eve_scores_for_protein(
    csv_path: Path,
) -> dict[tuple[str, int, str], dict]:
    """Parse a single EVE CSV file into a variant-keyed score dict.

    Each row in the EVE CSV represents a single amino acid substitution at a
    specific position. Rows without EVE scores (empty EVE_scores_ASM) are
    included with None values to distinguish 'no score' from 'variant not in file'.

    Args:
        csv_path: Path to an EVE CSV file (e.g. 1433G_HUMAN.csv).

    Returns:
        Dict keyed by (wt_aa, position, mt_aa) tuples with score dicts:
        {('R', 4, 'A'): {'eve_score': 0.773, 'eve_class': 'Pathogenic',
                          'eve_uncertainty': 0.536, 'evo_index': 9.882}}
        Score values are None when the EVE CSV row has empty score fields.

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If required columns are missing from the CSV.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"EVE CSV not found: {csv_path}")

    scores: dict[tuple[str, int, str], dict] = {}
    required_columns = {'wt_aa', 'position', 'mt_aa'}

    with open(csv_path, encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)

        # Validate required columns
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV file: {csv_path}")
        missing = required_columns - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"EVE CSV missing required columns {missing}: {csv_path}"
            )

        for row in reader:
            wt = row.get('wt_aa', '').strip()
            mt = row.get('mt_aa', '').strip()
            pos_str = row.get('position', '').strip()

            if not wt or not mt or not pos_str:
                continue

            try:
                pos = int(pos_str)
            except ValueError:
                continue

            # Parse score fields (empty string → None)
            eve_score_raw = row.get(EVE_SCORE_COLUMN, '').strip()
            eve_class_raw = row.get(EVE_CLASS_COLUMN, '').strip()
            eve_unc_raw = row.get(EVE_UNCERTAINTY_COLUMN, '').strip()
            evo_idx_raw = row.get(EVE_EVO_INDEX_COLUMN, '').strip()

            eve_score = float(eve_score_raw) if eve_score_raw else None
            eve_class = eve_class_raw if eve_class_raw else None
            eve_uncertainty = float(eve_unc_raw) if eve_unc_raw else None
            evo_index = float(evo_idx_raw) if evo_idx_raw else None

            scores[(wt, pos, mt)] = {
                'eve_score': eve_score,
                'eve_class': eve_class,
                'eve_uncertainty': eve_uncertainty,
                'evo_index': evo_index,
            }

    return scores


def build_eve_index(
    eve_dir: Path,
    accessions: frozenset[str],
    acc_to_entry: dict[str, str],
    verbose: bool = False,
) -> dict[str, dict[tuple[str, int, str], dict]]:
    """Build EVE score index for a set of UniProt accessions.

    Only loads EVE CSVs for proteins actually present in the pipeline run.
    Handles isoform accessions by stripping the '-N' suffix (EVE uses
    canonical sequences).

    Args:
        eve_dir: Path to EVE_all_data/ directory containing per-protein CSVs.
        accessions: Set of UniProt accessions to load scores for.
        acc_to_entry: Mapping from accession to entry name (from load_eve_entry_name_map).
        verbose: Print progress to stderr.

    Returns:
        Dict keyed by accession with score dicts:
        {'P61981': {('R', 4, 'A'): {'eve_score': 0.773, ...}, ...}}
        Accessions without EVE data are omitted (not included with empty dicts).
    """
    if not eve_dir.exists():
        if verbose:
            print(f"  EVE directory not found: {eve_dir}", file=sys.stderr)
        return {}

    eve_index: dict[str, dict[tuple[str, int, str], dict]] = {}
    loaded = 0
    skipped = 0

    # Deduplicate: strip isoform suffixes and collect unique base accessions
    # Map each original accession to the base accession used for EVE lookup
    acc_to_base: dict[str, str] = {}
    for acc in accessions:
        base = acc.split('-')[0] if '-' in acc else acc
        acc_to_base[acc] = base

    # Unique base accessions to look up
    unique_bases = frozenset(acc_to_base.values())

    for base_acc in sorted(unique_bases):
        entry_name = acc_to_entry.get(base_acc)
        if not entry_name:
            skipped += 1
            continue

        # Strip _HUMAN suffix if present to form filename
        csv_filename = f"{entry_name}.csv"
        csv_path = eve_dir / csv_filename

        if not csv_path.exists():
            skipped += 1
            continue

        try:
            scores = load_eve_scores_for_protein(csv_path)
            if scores:
                eve_index[base_acc] = scores
                loaded += 1
        except (ValueError, OSError) as e:
            if verbose:
                print(f"  Warning: failed to load EVE for {base_acc} "
                      f"({entry_name}): {e}", file=sys.stderr)
            skipped += 1

    if verbose:
        print(f"  EVE index: {loaded} proteins loaded, {skipped} skipped "
              f"(of {len(unique_bases)} unique accessions)", file=sys.stderr)

    return eve_index


def lookup_eve_score(
    eve_index: dict[str, dict[tuple[str, int, str], dict]],
    accession: str,
    ref_aa: str,
    position: int,
    alt_aa: str,
) -> Optional[dict]:
    """Look up EVE score for a single variant.

    Handles isoform accessions by stripping the '-N' suffix before lookup.

    Args:
        eve_index: EVE index from build_eve_index().
        accession: UniProt accession (e.g. 'P61981' or 'Q9UKT4-2').
        ref_aa: Reference amino acid (single letter).
        position: Residue position (1-based).
        alt_aa: Alternate amino acid (single letter).

    Returns:
        Score dict {'eve_score': float, 'eve_class': str, ...} or None if not found.
    """
    base = accession.split('-')[0] if '-' in accession else accession
    protein_scores = eve_index.get(base)
    if protein_scores is None:
        return None
    return protein_scores.get((ref_aa, position, alt_aa))


# ── Section 3: Annotation (toolkit integration) ─────────────────────

def _parse_variant_details_for_eve(
    details_str: str,
) -> list[tuple[str, int, str]]:
    """Parse variant_details string back to (ref_aa, position, alt_aa) tuples.

    The variant_details_a/b columns from variant_mapper use the format:
    'K81P:interface_core:pathogenic|E82K:interface_rim:VUS|...(+N more)'

    Args:
        details_str: Pipe-separated variant detail string from variant_mapper.

    Returns:
        List of (ref_aa, position, alt_aa) tuples for EVE lookup.
    """
    if not details_str:
        return []

    variants = []
    for part in details_str.split('|'):
        part = part.strip()
        if part.startswith('...(+'):
            continue  # skip truncation indicator

        match = _VARIANT_DETAIL_PATTERN.match(part)
        if match:
            ref = match.group(1)
            pos = int(match.group(2))
            alt = match.group(3)
            variants.append((ref, pos, alt))

    return variants


def format_stability_details(
    scored_variants: list[dict],
    limit: int = STABILITY_DETAILS_DISPLAY_LIMIT,
) -> str:
    """Format EVE-scored variants into a pipe-separated summary string.

    Format: REF{POS}ALT:eve={score}:{class}

    Args:
        scored_variants: List of dicts with keys 'ref_aa', 'position', 'alt_aa',
            'eve_score', 'eve_class'.
        limit: Maximum number of variants to include.

    Returns:
        Pipe-separated string, e.g. 'R4A:eve=0.77:Pathogenic|K81P:eve=0.23:Benign'.
        Empty string if no variants.
    """
    if not scored_variants:
        return ''

    details = []
    for var in scored_variants[:limit]:
        ref = var.get('ref_aa', '?')
        pos = var.get('position', '?')
        alt = var.get('alt_aa', '?')
        score = var.get('eve_score')
        eve_class = var.get('eve_class', '')

        score_str = f"{score:.2f}" if score is not None else '-'
        class_str = eve_class if eve_class else '-'
        details.append(f"{ref}{pos}{alt}:eve={score_str}:{class_str}")

    result = '|'.join(details)

    remaining = len(scored_variants) - limit
    if remaining > 0:
        result += f"|...(+{remaining} more)"

    return result


def annotate_results_with_stability(
    results: list[dict],
    eve_index: dict[str, dict[tuple[str, int, str], dict]],
    acc_to_entry: dict[str, str],
    verbose: bool = False,
) -> None:
    """Annotate result rows with EVE stability scores (in-place).

    Main entry point from toolkit.py. For each complex, parses the existing
    variant_details_a/b strings to extract variants, looks up EVE scores,
    and writes summary statistics and detail strings to the result dict.

    Args:
        results: List of per-complex result dicts (modified in-place).
        eve_index: EVE score index from build_eve_index().
        acc_to_entry: Accession-to-entry-name mapping (for coverage stats).
        verbose: Print progress to stderr.
    """
    annotated = 0
    with_eve = 0

    for row in results:
        protein_a = row.get('protein_a', '')
        protein_b = row.get('protein_b', '')
        details_a = row.get('variant_details_a', '')
        details_b = row.get('variant_details_b', '')

        # Process chain A
        eve_stats_a = _score_chain_variants(
            protein_a, details_a, eve_index,
        )
        row['eve_score_mean_a'] = eve_stats_a['score_mean']
        row['eve_n_pathogenic_a'] = eve_stats_a['n_pathogenic']
        row['eve_coverage_a'] = eve_stats_a['coverage']
        row['stability_details_a'] = eve_stats_a['details']

        # Process chain B
        eve_stats_b = _score_chain_variants(
            protein_b, details_b, eve_index,
        )
        row['eve_score_mean_b'] = eve_stats_b['score_mean']
        row['eve_n_pathogenic_b'] = eve_stats_b['n_pathogenic']
        row['eve_coverage_b'] = eve_stats_b['coverage']
        row['stability_details_b'] = eve_stats_b['details']

        annotated += 1
        if eve_stats_a['n_scored'] > 0 or eve_stats_b['n_scored'] > 0:
            with_eve += 1

    if verbose:
        print(f"  Stability annotation: {annotated} complexes processed, "
              f"{with_eve} with EVE scores", file=sys.stderr)


def _score_chain_variants(
    accession: str,
    details_str: str,
    eve_index: dict[str, dict[tuple[str, int, str], dict]],
) -> dict:
    """Score variants for one chain using EVE index.

    Args:
        accession: UniProt accession for this chain.
        details_str: Variant details string from variant_mapper.
        eve_index: EVE score index.

    Returns:
        Dict with keys: score_mean, n_pathogenic, coverage, n_scored, details.
    """
    variants = _parse_variant_details_for_eve(details_str)
    n_total = len(variants)

    if n_total == 0:
        return {
            'score_mean': '',
            'n_pathogenic': 0,
            'coverage': '',
            'n_scored': 0,
            'details': '',
        }

    scored_variants = []
    scores = []
    n_pathogenic = 0

    for ref, pos, alt in variants:
        eve_result = lookup_eve_score(eve_index, accession, ref, pos, alt)
        if eve_result is not None and eve_result.get('eve_score') is not None:
            scored_variants.append({
                'ref_aa': ref,
                'position': pos,
                'alt_aa': alt,
                'eve_score': eve_result['eve_score'],
                'eve_class': eve_result.get('eve_class', ''),
                'eve_uncertainty': eve_result.get('eve_uncertainty'),
                'evo_index': eve_result.get('evo_index'),
            })
            scores.append(eve_result['eve_score'])
            if eve_result.get('eve_class') == EVE_CLASS_PATHOGENIC:
                n_pathogenic += 1

    n_scored = len(scored_variants)
    coverage = n_scored / n_total if n_total > 0 else 0.0

    return {
        'score_mean': round(mean(scores), 4) if scores else '',
        'n_pathogenic': n_pathogenic,
        'coverage': round(coverage, 4) if n_total > 0 else '',
        'n_scored': n_scored,
        'details': format_stability_details(scored_variants),
    }


# ── Section 4: Standalone CLI ────────────────────────────────────────

def build_argument_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog='stability_scorer',
        description='EVE stability scoring for protein complex variants.',
    )
    parser.add_argument(
        '--stability-dir', type=str,
        default=str(DEFAULT_STABILITY_DIR),
        help=f"Path to stability data directory (default: {DEFAULT_STABILITY_DIR})",
    )

    subparsers = parser.add_subparsers(dest='command', help='Subcommand')

    # summary subcommand
    sub_summary = subparsers.add_parser(
        'summary',
        help='Print EVE data coverage statistics.',
    )

    # lookup subcommand
    sub_lookup = subparsers.add_parser(
        'lookup',
        help='Look up EVE scores for a protein or specific variant.',
    )
    sub_lookup.add_argument(
        '--protein', required=True,
        help='UniProt accession to look up (e.g. P61981).',
    )
    sub_lookup.add_argument(
        '--position', type=int, default=None,
        help='Optional residue position to filter to.',
    )

    return parser


def _cli_summary(stability_dir: Path) -> None:
    """Print EVE coverage statistics."""
    eve_dir = stability_dir / "EVE_all_data"
    map_path = stability_dir / EVE_IDMAPPING_FILENAME

    # Count EVE files
    if eve_dir.exists():
        eve_files = list(eve_dir.glob("*_HUMAN.csv"))
        print(f"EVE data directory: {eve_dir}")
        print(f"EVE CSV files: {len(eve_files)}")
    else:
        print(f"EVE data directory not found: {eve_dir}")
        return

    # Load mapping
    acc_to_entry = load_eve_entry_name_map(map_path, verbose=False)
    print(f"ID mapping file: {map_path}")
    print(f"Accession-to-entry-name mappings: {len(acc_to_entry):,}")

    # Check coverage: how many EVE files have a matching accession
    entry_to_acc = {v: k for k, v in acc_to_entry.items()}
    matched = 0
    for f in eve_files:
        entry_name = f.stem  # e.g. '1433G_HUMAN'
        if entry_name in entry_to_acc:
            matched += 1

    print(f"EVE files with mapped accession: {matched}/{len(eve_files)} "
          f"({100 * matched / len(eve_files):.1f}%)" if eve_files else "")


def _cli_lookup(stability_dir: Path, accession: str, position: Optional[int]) -> None:
    """Look up EVE scores for a protein or specific variant."""
    map_path = stability_dir / EVE_IDMAPPING_FILENAME
    eve_dir = stability_dir / "EVE_all_data"

    acc_to_entry = load_eve_entry_name_map(map_path, verbose=False)
    base = accession.split('-')[0] if '-' in accession else accession
    entry_name = acc_to_entry.get(base)

    if not entry_name:
        print(f"No entry name found for accession {accession} "
              f"(base: {base})", file=sys.stderr)
        sys.exit(1)

    csv_path = eve_dir / f"{entry_name}.csv"
    if not csv_path.exists():
        print(f"EVE CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Accession: {accession} (base: {base})")
    print(f"Entry name: {entry_name}")
    print(f"EVE CSV: {csv_path}")

    scores = load_eve_scores_for_protein(csv_path)
    print(f"Total variants in EVE: {len(scores)}")

    scored = sum(1 for v in scores.values() if v.get('eve_score') is not None)
    print(f"With EVE scores: {scored} ({100 * scored / len(scores):.1f}%)"
          if scores else "No scores found")

    if position is not None:
        # Filter to specific position
        pos_variants = {k: v for k, v in scores.items() if k[1] == position}
        if not pos_variants:
            print(f"\nNo variants at position {position}")
        else:
            print(f"\nVariants at position {position}:")
            for (wt, pos, mt), score_dict in sorted(pos_variants.items()):
                s = score_dict.get('eve_score')
                c = score_dict.get('eve_class', '-')
                s_str = f"{s:.4f}" if s is not None else '-'
                print(f"  {wt}{pos}{mt}: score={s_str}, class={c}")


def main() -> None:
    """CLI entry point."""
    parser = build_argument_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    stability_dir = Path(args.stability_dir)

    if args.command == 'summary':
        _cli_summary(stability_dir)
    elif args.command == 'lookup':
        _cli_lookup(stability_dir, args.protein, args.position)


if __name__ == '__main__':
    main()
