#!/usr/bin/env python3
"""
Database Overlap Analysis and Venn Diagram Generation - computes pairwise and multi-way overlaps between PPI databases (STRING, BioGRID, HuRI, HuMAP) after mapping all identifiers to UniProt. Generates Venn diagrams and overlap summary statistics.

Usage as module:
    from overlap_analysis import extract_pair_set, compute_overlaps, plot_venn_diagram
    pair_sets = {name: extract_pair_set(df) for name, df in databases.items()}
    stats = compute_overlaps(pair_sets)
    plot_venn_diagram(pair_sets, "output/venn_overlap.png")

Usage as CLI:
    python overlap_analysis.py --data-dir data/ppi --aliases data/ppi/9606.protein.aliases.v12.0.txt --output Output/venn.png -v
"""

import sys
import argparse
from pathlib import Path
from typing import Optional
from itertools import combinations

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#------Constants----------------------------------------------------
STRING_CONFIDENCE_THRESHOLDS = [0, 400, 700, 900]

OUTPUT_DPI = 200

# Database colours for consistent plotting
DB_COLOURS = {
    'STRING': '#4C72B0',
    'BioGRID': '#DD8452',
    'HuRI': '#55A868',
    'HuMAP': '#C44E52',
}


#---------Pair Normalisation------------------------------------------------
def normalise_pair(id_a: str, id_b: str) -> tuple[str, str]:
    """Return a canonical (sorted) pair for symmetric comparison.
    Args:
        id_a: First protein identifier.
        id_b: Second protein identifier.
    Returns:
        Tuple (min_id, max_id) ensuring A_B == B_A.
    """
    return (min(id_a, id_b), max(id_a, id_b))


def normalise_pair_base(id_a: str, id_b: str) -> tuple[str, str]:
    """Return a canonical sorted pair at base-accession level.
    Strips isoform suffixes (e.g., Q9UKT4-2 -> Q9UKT4) before sorting,
    so that isoform-specific and base accessions match in cross-database
    comparisons.
    Args:
        id_a: First protein identifier (may include isoform suffix).
        id_b: Second protein identifier.
    Returns:
        Tuple (min_base, max_base) ensuring A_B == B_A at base level.
    """
    from id_mapper import split_isoform
    base_a = split_isoform(id_a)[0]
    base_b = split_isoform(id_b)[0]
    return (min(base_a, base_b), max(base_a, base_b))


def extract_pair_set(
    df: pd.DataFrame,
    col_a: str = 'protein_a',
    col_b: str = 'protein_b',
) -> set[tuple[str, str]]:
    """Extract normalised pair set from a DataFrame.
    Args:
        df: DataFrame with protein pair columns.
        col_a: Column name for protein A.
        col_b: Column name for protein B.
    Returns:
        Set of (id_a, id_b) tuples, canonically ordered.
    """
    pairs = set()
    for a, b in zip(df[col_a].astype(str), df[col_b].astype(str)):
        if a and b and a != 'nan' and b != 'nan':
            pairs.add(normalise_pair(a, b))
    return pairs


def extract_pair_set_base(
    df: pd.DataFrame,
    col_a: str = 'protein_a',
    col_b: str = 'protein_b',
) -> set[tuple[str, str]]:
    """Extract normalised pair set at base-accession level.
    Like extract_pair_set() but strips isoform suffixes before
    normalising, enabling cross-database comparison where some
    databases (STRING, BioGRID) lack isoform specificity.
    Args:
        df: DataFrame with protein pair columns.
        col_a: Column name for protein A.
        col_b: Column name for protein B.
    Returns:
        Set of (base_id_a, base_id_b) tuples, canonically ordered.
    """
    pairs = set()
    for a, b in zip(df[col_a].astype(str), df[col_b].astype(str)):
        if a and b and a != 'nan' and b != 'nan':
            pairs.add(normalise_pair_base(a, b))
    return pairs


#--------Overlap Computation-------------------------------------------------
def compute_overlaps(
    pair_sets: dict[str, set[tuple[str, str]]],
) -> dict[str, dict]:
    """Compute overlap statistics across multiple databases.
    Args:
        pair_sets: Dict mapping database name to set of normalised pairs.
    Returns:
        Dict with keys:
        - 'per_database': {name: count} - unique pair count per DB
        - 'pairwise': {(db1, db2): count} - intersection counts
        - 'triple': {(db1, db2, db3): count} - triple intersection counts
        - 'all': count - pairs in all databases
        - 'unique_to': {name: count} - pairs found only in that DB
        - 'union': count - total unique pairs across all databases
    """
    names = list(pair_sets.keys())

    stats: dict[str, dict] = {
        'per_database': {},
        'pairwise': {},
        'triple': {},
        'all': 0,
        'unique_to': {},
        'union': 0,
    }

    # Per-database counts
    for name in names:
        stats['per_database'][name] = len(pair_sets[name])

    # Pairwise intersections
    for a, b in combinations(names, 2):
        overlap = pair_sets[a] & pair_sets[b]
        stats['pairwise'][(a, b)] = len(overlap)

    # Triple intersections
    for a, b, c in combinations(names, 3):
        overlap = pair_sets[a] & pair_sets[b] & pair_sets[c]
        stats['triple'][(a, b, c)] = len(overlap)

    # All-database intersection
    if len(names) >= 2:
        all_overlap = pair_sets[names[0]]
        for name in names[1:]:
            all_overlap = all_overlap & pair_sets[name]
        stats['all'] = len(all_overlap)

    # Union
    all_union = set()
    for s in pair_sets.values():
        all_union |= s
    stats['union'] = len(all_union)

    # Unique to each database
    for name in names:
        others = set()
        for other_name in names:
            if other_name != name:
                others |= pair_sets[other_name]
        stats['unique_to'][name] = len(pair_sets[name] - others)

    return stats


def print_overlap_summary(
    stats: dict[str, dict],
    file=None,
) -> None:
    """Print a formatted summary of overlap statistics.
    Args:
        stats: Output from compute_overlaps().
        file: Output stream (default: stdout).
    """
    if file is None:
        file = sys.stdout

    print("\n=== PPI Database Overlap Summary ===", file=file)
    print(f"\nTotal unique pairs (union): {stats['union']:,}", file=file)

    print("\nPer-database pair counts:", file=file)
    for name, count in stats['per_database'].items():
        print(f"  {name}: {count:,}", file=file)

    print("\nPairwise overlaps:", file=file)
    for (a, b), count in stats['pairwise'].items():
        print(f"  {a} & {b}: {count:,}", file=file)

    if stats['triple']:
        print("\nTriple overlaps:", file=file)
        for (a, b, c), count in stats['triple'].items():
            print(f"  {a} & {b} & {c}: {count:,}", file=file)

    print(f"\nIn all databases: {stats['all']:,}", file=file)

    print("\nUnique to each database:", file=file)
    for name, count in stats['unique_to'].items():
        print(f"  {name} only: {count:,}", file=file)


#--------------Venn Diagram Generation-------------------------------------
def plot_venn_diagram(
    pair_sets: dict[str, set[tuple[str, str]]],
    output_path: str,
    title: str = "PPI Database Overlap",
    verbose: bool = False,
) -> None:
    """Generate an overlap diagram of databases.
    For 2-3 databases, uses matplotlib_venn if available.
    For 4+ databases, generates an UpSet-style bar chart showing
    intersection sizes, which is more readable than a 4-set Venn.
    Args:
        pair_sets: Dict mapping database name to set of normalised pairs.
        output_path: Path for the output figure file.
        title: Figure title.
        verbose: Print overlap statistics.
    """
    names = list(pair_sets.keys())
    stats = compute_overlaps(pair_sets)

    if verbose:
        print_overlap_summary(stats, file=sys.stderr)

    if len(names) <= 3:
        _plot_venn_2_3(pair_sets, output_path, title)
    else:
        _plot_upset_style(pair_sets, output_path, title)


def _plot_venn_2_3(
    pair_sets: dict[str, set[tuple[str, str]]],
    output_path: str,
    title: str,
) -> None:
    """Plot a 2-set or 3-set Venn diagram."""
    names = list(pair_sets.keys())

    try:
        if len(names) == 2:
            from matplotlib_venn import venn2
            fig, ax = plt.subplots(figsize=(8, 6))
            venn2(
                [pair_sets[names[0]], pair_sets[names[1]]],
                set_labels=names,
                ax=ax,
            )
        else:
            from matplotlib_venn import venn3
            fig, ax = plt.subplots(figsize=(8, 6))
            venn3(
                [pair_sets[names[0]], pair_sets[names[1]], pair_sets[names[2]]],
                set_labels=names,
                ax=ax,
            )
        ax.set_title(title, fontsize=14)
        fig.tight_layout()
        fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches='tight')
        plt.close(fig)
    except ImportError:
        print("  matplotlib-venn not installed, falling back to bar chart",
              file=sys.stderr)
        _plot_upset_style(pair_sets, output_path, title)


def _plot_upset_style(
    pair_sets: dict[str, set[tuple[str, str]]],
    output_path: str,
    title: str,
) -> None:
    """Plot an UpSet-style intersection bar chart for 4+ databases.
    This is more readable than a 4-set Venn diagram and is commonly
    used in bioinformatics publications.
    """
    names = list(pair_sets.keys())
    n = len(names)

    # Compute all intersection sizes using inclusion-exclusion
    intersections = []
    for r in range(1, n + 1):
        for combo in combinations(range(n), r):
            # Intersection of selected databases
            result = pair_sets[names[combo[0]]]
            for idx in combo[1:]:
                result = result & pair_sets[names[idx]]

            # Exclusive intersection: in these databases AND NOT in any others
            for idx in range(n):
                if idx not in combo:
                    result = result - pair_sets[names[idx]]

            if len(result) > 0:
                member_flags = [i in combo for i in range(n)]
                intersections.append((member_flags, len(result)))

    # Sort by size descending
    intersections.sort(key=lambda x: x[1], reverse=True)

    if not intersections:
        return

    # Plot
    fig, (ax_bar, ax_dots) = plt.subplots(
        2, 1,
        figsize=(max(10, len(intersections) * 0.6), 8),
        gridspec_kw={'height_ratios': [3, 1]},
        sharex=True,
    )

    x = range(len(intersections))
    sizes = [s for _, s in intersections]
    colours = []
    for flags, _ in intersections:
        n_active = sum(flags)
        if n_active == 1:
            idx = flags.index(True)
            db_name = names[idx]
            colours.append(DB_COLOURS.get(db_name, '#888888'))
        else:
            colours.append('#888888')

    ax_bar.bar(x, sizes, color=colours, edgecolor='white', linewidth=0.5)
    ax_bar.set_ylabel('Number of Unique Pairs', fontsize=11)
    ax_bar.set_title(title, fontsize=14, pad=15)

    # Format y-axis with comma separators
    ax_bar.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: format(int(x), ','))
    )

    # Add count labels on top of bars
    for i, size in enumerate(sizes):
        if size > 0:
            ax_bar.text(i, size, f'{size:,}', ha='center', va='bottom',
                       fontsize=7, rotation=45)

    # Dot matrix showing which databases are in each intersection
    for i, (flags, _) in enumerate(intersections):
        for j, active in enumerate(flags):
            if active:
                ax_dots.plot(i, j, 'o', color='black', markersize=8)
            else:
                ax_dots.plot(i, j, 'o', color='#DDDDDD', markersize=6)
        # Connect active dots with a line
        active_indices = [j for j, a in enumerate(flags) if a]
        if len(active_indices) > 1:
            ax_dots.plot(
                [i] * len(active_indices),
                active_indices,
                '-', color='black', linewidth=1.5,
            )

    ax_dots.set_yticks(range(n))
    ax_dots.set_yticklabels(names, fontsize=10)
    ax_dots.set_ylim(-0.5, n - 0.5)
    ax_dots.invert_yaxis()
    ax_dots.set_xlim(-0.5, len(intersections) - 0.5)
    ax_dots.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    # Add database totals on the right
    for j, name in enumerate(names):
        total = len(pair_sets[name])
        ax_dots.text(
            len(intersections) + 0.3, j,
            f'{total:,}',
            va='center', fontsize=9, color='#555555',
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_threshold_comparison(
    string_filepath: str,
    other_pair_sets: dict[str, set[tuple[str, str]]],
    mapper,
    output_path: str,
    thresholds: Optional[list[int]] = None,
    verbose: bool = False,
) -> None:
    """Generate overlap comparisons at different STRING confidence thresholds.
    Produces a multi-panel figure showing how STRING's overlap with other
    databases changes as the confidence threshold increases.
    Args:
        string_filepath: Path to the full STRING links file.
        other_pair_sets: Dict of {name: pair_set} for BioGRID, HuRI, HuMAP.
        mapper: IDMapper instance for ENSP-to-UniProt mapping.
        output_path: Path for the output figure file.
        thresholds: STRING score thresholds to compare.
            Defaults to [0, 400, 700, 900].
        verbose: Print progress.
    """
    from database_loaders import load_string
    from id_mapper import map_dataframe_to_uniprot

    if thresholds is None:
        thresholds = STRING_CONFIDENCE_THRESHOLDS

    fig, axes = plt.subplots(1, len(thresholds), figsize=(5 * len(thresholds), 5))
    if len(thresholds) == 1:
        axes = [axes]

    for ax, threshold in zip(axes, thresholds):
        if verbose:
            print(f"  Processing STRING threshold >= {threshold}...", file=sys.stderr)

        string_df = load_string(string_filepath, min_score=threshold)
        string_mapped = map_dataframe_to_uniprot(string_df, mapper)
        string_pairs = extract_pair_set(string_mapped, col_a='uniprot_a', col_b='uniprot_b')

        # Compute overlaps with each other database
        overlap_counts = {}
        for name, other_set in other_pair_sets.items():
            overlap_counts[name] = len(string_pairs & other_set)

        # Simple bar chart for this threshold
        db_names = list(overlap_counts.keys())
        counts = list(overlap_counts.values())
        bar_colours = [DB_COLOURS.get(n, '#888888') for n in db_names]

        ax.barh(db_names, counts, color=bar_colours)
        ax.set_title(f'STRING ≥ {threshold}\n({len(string_pairs):,} pairs)', fontsize=10)
        ax.set_xlabel('Shared Pairs')

        for i, count in enumerate(counts):
            ax.text(count + max(counts) * 0.02, i, f'{count:,}', va='center', fontsize=8)

    fig.suptitle('STRING Overlap at Different Confidence Thresholds', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches='tight')
    plt.close(fig)


#-------------------CLI-----------------------------------------
def build_argument_parser() -> argparse.ArgumentParser:
    """Create and return the argument parser for overlap_analysis."""
    parser = argparse.ArgumentParser(
        description="Compute PPI database overlaps and generate Venn diagrams.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python overlap_analysis.py --data-dir data/ppi --aliases data/ppi/9606.protein.aliases.v12.0.txt --output Output/venn.png -v
    python overlap_analysis.py --data-dir data/ppi --aliases data/ppi/9606.protein.aliases.v12.0.txt --string-min-score 700 --output Output/venn_700.png
        """,
    )

    parser.add_argument(
        "--data-dir",
        default="data/ppi",
        help="Directory containing database files",
    )
    parser.add_argument(
        "--aliases",
        required=True,
        help="Path to STRING aliases file for ID mapping",
    )
    parser.add_argument(
        "--output",
        default="Output/venn_overlap.png",
        help="Output figure path (default: Output/venn_overlap.png)",
    )
    parser.add_argument(
        "--string-min-score",
        type=int,
        default=700,
        help="STRING minimum score for main analysis (default: 700)",
    )
    parser.add_argument(
        "--threshold-comparison",
        metavar="OUTPUT_PATH",
        help="Generate threshold comparison figure at this path",
    )
    parser.add_argument(
        "--report",
        metavar="OUTPUT_PATH",
        help="Write full overlap statistics to a text file",
    )
    parser.add_argument(
        "--base-level",
        action="store_true",
        help="Also compute overlaps at base-accession level (stripping isoform suffixes)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress",
    )

    return parser


def main() -> None:
    """Run the overlap analysis CLI."""
    parser = build_argument_parser()
    args = parser.parse_args()

    from database_loaders import load_all_databases
    from id_mapper import IDMapper, map_dataframe_to_uniprot

    # Load ID mapper
    print("Loading ID mapper...", file=sys.stderr)
    mapper = IDMapper(args.aliases, verbose=args.verbose)

    # Load all databases
    dbs = load_all_databases(
        data_dir=args.data_dir,
        string_min_score=args.string_min_score,
        verbose=args.verbose,
    )

    # Map STRING and HuRI to UniProt
    if args.verbose:
        print("\nMapping STRING IDs to UniProt...", file=sys.stderr)
    dbs['STRING'] = map_dataframe_to_uniprot(dbs['STRING'], mapper, verbose=args.verbose)

    if args.verbose:
        print("Mapping HuRI IDs to UniProt...", file=sys.stderr)
    dbs['HuRI'] = map_dataframe_to_uniprot(dbs['HuRI'], mapper, verbose=args.verbose)

    # Extract pair sets (use uniprot_a/b for mapped DBs, protein_a/b for direct)
    pair_sets = {}
    for name, df in dbs.items():
        if 'uniprot_a' in df.columns:
            pair_sets[name] = extract_pair_set(df, col_a='uniprot_a', col_b='uniprot_b')
        else:
            pair_sets[name] = extract_pair_set(df)

    # Compute and display overlaps (isoform-specific level)
    stats = compute_overlaps(pair_sets)
    print_overlap_summary(stats)

    # Generate Venn diagram
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plot_venn_diagram(pair_sets, args.output,
                      title="PPI Database Overlap (Isoform-Specific)",
                      verbose=args.verbose)
    print(f"\nVenn diagram saved to: {args.output}")

    # Base-accession level overlap (strips isoform suffixes)
    if args.base_level:
        if args.verbose:
            print("\nComputing base-accession level overlap...", file=sys.stderr)

        base_pair_sets = {}
        for name, df in dbs.items():
            if 'uniprot_a' in df.columns:
                base_pair_sets[name] = extract_pair_set_base(
                    df, col_a='uniprot_a', col_b='uniprot_b')
            else:
                base_pair_sets[name] = extract_pair_set_base(df)

        base_stats = compute_overlaps(base_pair_sets)
        print("\n--- Base-Accession Level (isoform suffixes stripped) ---")
        print_overlap_summary(base_stats)

        # Generate base-level Venn diagram
        base_output = Path(args.output)
        base_output_path = str(
            base_output.parent / f"{base_output.stem}_base{base_output.suffix}")
        plot_venn_diagram(base_pair_sets, base_output_path,
                          title="PPI Database Overlap (Base Accession)",
                          verbose=args.verbose)
        print(f"Base-level Venn diagram saved to: {base_output_path}")

    # Write report file
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, 'w') as f:
            f.write("PPI Database Overlap Report\n")
            f.write("=" * 40 + "\n")
            f.write(f"\nSTRING minimum score: {args.string_min_score}\n")
            f.write("\n--- Isoform-Specific Level ---")
            print_overlap_summary(stats, file=f)
            if args.base_level:
                f.write("\n\n--- Base-Accession Level ---")
                print_overlap_summary(base_stats, file=f)
        print(f"Report written to: {args.report}")

    # Optional threshold comparison
    if args.threshold_comparison:
        non_string = {k: v for k, v in pair_sets.items() if k != 'STRING'}
        string_filepath = str(Path(args.data_dir) / "9606.protein.links.v12.0.txt")
        plot_threshold_comparison(
            string_filepath, non_string, mapper,
            args.threshold_comparison,
            verbose=args.verbose,
        )
        print(f"Threshold comparison saved to: {args.threshold_comparison}")


if __name__ == "__main__":
    main()
