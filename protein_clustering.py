"""
Protein clustering and homology analysis using STRING sequence clusters.

Parses STRING pre-computed cluster assignments, maps them to UniProt
identifiers via the ID mapper, and annotates protein complex pairs with
cluster membership and homologous pair information.

Architecture:
    - Offline-first: reads STRING clusters flat file as primary data source
    - Optional API supplement via string_api.query_homology() for continuous
      paralogy scores (disabled with --no-api)
    - Integrates into toolkit.py via --clustering flag

Usage (standalone):
    python protein_clustering.py --clusters-file data/clusters/9606.clusters.proteins.v12.0.txt \\
        --aliases data/ppi/9606.protein.aliases.v12.0.txt --summary
    python protein_clustering.py --clusters-file data/clusters/... --aliases data/ppi/... --protein P04637
    python protein_clustering.py --clusters-file data/clusters/... --aliases data/ppi/... --pair P04637 Q04206

Usage (via toolkit.py):
    python toolkit.py --dir DIR --output results.csv --enrich ALIASES --clustering
    python toolkit.py --dir DIR --output results.csv --enrich ALIASES --clustering --clusters-file PATH
    python toolkit.py --dir DIR --output results.csv --enrich ALIASES --clustering foldseek  # raises NotImplementedError
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional, Union

import pandas as pd


# ── Constants ────────────────────────────────────────────────────────

# Default file paths
DEFAULT_CLUSTERS_DIR = Path(__file__).parent / "data" / "clusters"
STRING_CLUSTERS_FILE = "9606.clusters.proteins.v12.0.txt"

# File format
CLUSTERS_SEPARATOR = '\t'

# Valid clustering modes
VALID_CLUSTERING_MODES = ('string', 'foldseek', 'hybrid')

# Display limit for pipe-separated homologous pairs in CSV cells
HOMOLOGOUS_PAIRS_DISPLAY_LIMIT = 20

# Maximum cluster size for homologous pair detection (UniProt space).
# STRING clusters are hierarchical — the broadest clusters contain nearly all
# proteins (e.g. 144,000+ members) and are meaningless for homology detection.
# Clusters larger than this threshold are skipped during pair generation to
# avoid O(n^2) explosion. They are still reported in cluster annotation columns.
MAX_CLUSTER_SIZE_FOR_PAIRS = 500

# CSV columns added when --clustering is used
CSV_FIELDNAMES_CLUSTERING = [
    'sequence_cluster_ids',
    'sequence_cluster_count',
    'shared_cluster_ids',
    'shared_cluster_count',
    'homologous_pairs',
    'n_homologous_pairs',
    'homology_bitscore',
]


# ── Helper Functions ─────────────────────────────────────────────────

def _strip_taxonomy_prefix(protein_id: str) -> str:
    """Remove the '9606.' taxonomy prefix from a STRING protein ID.

    Args:
        protein_id: STRING-format protein ID (e.g. '9606.ENSP00000269305').

    Returns:
        Bare Ensembl protein ID (e.g. 'ENSP00000269305').
    """
    if protein_id.startswith('9606.'):
        return protein_id[5:]
    return protein_id


# ── Core Functions ───────────────────────────────────────────────────

def load_clusters(filepath: Optional[str] = None,
                  verbose: bool = False) -> pd.DataFrame:
    """Load STRING clusters file into a DataFrame.

    Reads the tab-separated clusters file, strips the '9606.' taxonomy
    prefix from protein IDs, and drops the taxonomy column (always 9606
    for this project).

    Args:
        filepath: Path to clusters file. Defaults to
            data/clusters/9606.clusters.proteins.v12.0.txt.
        verbose: Print progress information to stderr.

    Returns:
        DataFrame with columns: ['cluster_id', 'protein_id'] where
        protein_id is a bare ENSP identifier.
    """
    if filepath is None:
        filepath = str(DEFAULT_CLUSTERS_DIR / STRING_CLUSTERS_FILE)

    if verbose:
        print(f"  Loading clusters from: {filepath}", file=sys.stderr)

    df = pd.read_csv(
        filepath,
        sep=CLUSTERS_SEPARATOR,
        comment='#',
        names=['string_taxon_id', 'cluster_id', 'protein_id'],
        dtype={'string_taxon_id': str, 'cluster_id': str, 'protein_id': str},
    )

    # Strip taxonomy prefix from protein IDs
    df['protein_id'] = df['protein_id'].apply(_strip_taxonomy_prefix)

    # Drop taxonomy column (always 9606)
    df = df[['cluster_id', 'protein_id']].reset_index(drop=True)

    if verbose:
        n_clusters = df['cluster_id'].nunique()
        n_proteins = df['protein_id'].nunique()
        print(f"  Loaded {len(df):,} assignments: {n_clusters:,} clusters, "
              f"{n_proteins:,} unique proteins", file=sys.stderr)

    return df


def build_cluster_index(df: pd.DataFrame) -> tuple[dict[str, set[str]],
                                                     dict[str, set[str]]]:
    """Build bidirectional lookup dicts from a clusters DataFrame.

    Args:
        df: DataFrame with columns ['cluster_id', 'protein_id'].

    Returns:
        Tuple of (cluster_to_proteins, protein_to_clusters) where:
        - cluster_to_proteins: cluster_id -> set of ENSP IDs
        - protein_to_clusters: ENSP_id -> set of cluster_ids
    """
    cluster_to_proteins: dict[str, set[str]] = {}
    protein_to_clusters: dict[str, set[str]] = {}

    for cluster_id, protein_id in zip(df['cluster_id'], df['protein_id']):
        cluster_to_proteins.setdefault(cluster_id, set()).add(protein_id)
        protein_to_clusters.setdefault(protein_id, set()).add(cluster_id)

    return cluster_to_proteins, protein_to_clusters


def get_cluster_sizes(cluster_to_proteins: dict[str, set[str]]) -> dict[str, int]:
    """Return dict mapping cluster_id to member count.

    Args:
        cluster_to_proteins: Cluster-to-proteins index from build_cluster_index().

    Returns:
        Dict of {cluster_id: number_of_members}.
    """
    return {cid: len(members) for cid, members in cluster_to_proteins.items()}


def build_uniprot_cluster_index(
    protein_to_clusters: dict[str, set[str]],
    mapper: 'IDMapper',
    verbose: bool = False,
) -> dict[str, set[str]]:
    """Resolve ENSP IDs to UniProt and build a UniProt-keyed cluster index.

    For each ENSP in protein_to_clusters, resolves to UniProt accession(s)
    via the IDMapper. All UniProt accessions for a given ENSP inherit the
    same cluster set. ENSP IDs with no UniProt mapping are skipped.

    Args:
        protein_to_clusters: ENSP-to-clusters index from build_cluster_index().
        mapper: IDMapper instance with loaded STRING aliases.
        verbose: Print progress and statistics to stderr.

    Returns:
        Dict mapping UniProt accession -> set of cluster IDs.
    """
    uniprot_index: dict[str, set[str]] = {}
    n_resolved = 0
    n_skipped = 0

    for ensp, clusters in protein_to_clusters.items():
        uniprot_ids = mapper.ensembl_to_uniprot(ensp)
        if not uniprot_ids:
            n_skipped += 1
            continue
        n_resolved += 1
        for uid in uniprot_ids:
            if uid in uniprot_index:
                uniprot_index[uid].update(clusters)
            else:
                uniprot_index[uid] = set(clusters)

    if verbose:
        print(f"  UniProt cluster index: {len(uniprot_index):,} accessions "
              f"({n_resolved:,} ENSPs resolved, {n_skipped:,} skipped)",
              file=sys.stderr)

    return uniprot_index


def build_cluster_to_uniprot(
    uniprot_index: dict[str, set[str]],
) -> dict[str, set[str]]:
    """Invert the UniProt cluster index to cluster_id -> set of UniProt accessions.

    Args:
        uniprot_index: UniProt-to-clusters index from build_uniprot_cluster_index().

    Returns:
        Dict mapping cluster_id -> set of UniProt accessions.
    """
    cluster_to_uniprot: dict[str, set[str]] = {}
    for uid, clusters in uniprot_index.items():
        for cid in clusters:
            cluster_to_uniprot.setdefault(cid, set()).add(uid)
    return cluster_to_uniprot


def find_shared_clusters(
    prot_a: str,
    prot_b: str,
    uniprot_index: dict[str, set[str]],
) -> set[str]:
    """Return cluster IDs shared by both proteins (intersection).

    Args:
        prot_a: UniProt accession of protein A.
        prot_b: UniProt accession of protein B.
        uniprot_index: UniProt-to-clusters index.

    Returns:
        Set of cluster IDs present in both proteins' cluster sets.
        Empty set if either protein is not in the index.
    """
    clusters_a = uniprot_index.get(prot_a, set())
    clusters_b = uniprot_index.get(prot_b, set())
    return clusters_a & clusters_b


def find_homologous_pairs(
    prot_a: str,
    prot_b: str,
    uniprot_index: dict[str, set[str]],
    cluster_to_uniprot: dict[str, set[str]],
    known_pairs: Optional[set[tuple[str, str]]] = None,
) -> list[tuple[str, str]]:
    """Find other protein pairs where both proteins share clusters with the query pair.

    For each cluster shared by prot_a and prot_b, finds all other pairs of
    proteins in those same clusters. Excludes the query pair itself and
    self-pairs. Optionally filters to only pairs present in known interaction
    databases.

    Args:
        prot_a: UniProt accession of protein A.
        prot_b: UniProt accession of protein B.
        uniprot_index: UniProt-to-clusters index.
        cluster_to_uniprot: Cluster-to-UniProt index.
        known_pairs: If provided, only return pairs present in this set.
            Pairs should be normalised as (min, max) tuples.

    Returns:
        Deduplicated list of normalised (UniProt_X, UniProt_Y) tuples.
    """
    shared = find_shared_clusters(prot_a, prot_b, uniprot_index)
    if not shared:
        return []

    # Collect candidate proteins from shared clusters, skipping oversized ones.
    # STRING clusters are hierarchical — the broadest levels contain nearly all
    # proteins and would cause O(n^2) explosion in pair generation.
    candidates: set[str] = set()
    for cid in shared:
        members = cluster_to_uniprot.get(cid, set())
        if len(members) > MAX_CLUSTER_SIZE_FOR_PAIRS:
            continue
        candidates.update(members)

    if len(candidates) < 2:
        return []

    # Normalise the query pair for exclusion
    query_norm = (min(prot_a, prot_b), max(prot_a, prot_b))

    # Generate all candidate pairs from the pooled candidate set
    result_set: set[tuple[str, str]] = set()
    candidates_list = sorted(candidates)
    for i, ca in enumerate(candidates_list):
        for cb in candidates_list[i + 1:]:
            pair = (ca, cb)
            if pair == query_norm:
                continue
            result_set.add(pair)

    # Filter to known interaction pairs if provided
    if known_pairs is not None:
        result_set = result_set & known_pairs

    return sorted(result_set)


def annotate_pair_clusters(
    row: dict,
    uniprot_index: dict[str, set[str]],
) -> None:
    """Add cluster annotation columns to a single result row (in-place).

    Sets sequence_cluster_ids (union), sequence_cluster_count,
    shared_cluster_ids (intersection), and shared_cluster_count.

    Args:
        row: Result dict with 'protein_a' and 'protein_b' keys.
        uniprot_index: UniProt-to-clusters index.
    """
    prot_a = row.get('protein_a', '')
    prot_b = row.get('protein_b', '')

    clusters_a = uniprot_index.get(prot_a, set())
    clusters_b = uniprot_index.get(prot_b, set())

    # Union: all clusters for the pair
    union = clusters_a | clusters_b
    row['sequence_cluster_ids'] = '|'.join(sorted(union)) if union else ''
    row['sequence_cluster_count'] = len(union)

    # Intersection: clusters shared by both proteins
    shared = clusters_a & clusters_b
    row['shared_cluster_ids'] = '|'.join(sorted(shared)) if shared else ''
    row['shared_cluster_count'] = len(shared)


def annotate_results_with_clustering(
    results: list[dict],
    uniprot_index: dict[str, set[str]],
    cluster_to_uniprot: dict[str, set[str]],
    known_pairs: Optional[set[tuple[str, str]]] = None,
    verbose: bool = False,
) -> None:
    """Annotate all result rows with clustering columns (in-place).

    Main entry point called from toolkit.py. Adds cluster membership,
    shared clusters, and homologous pair information to each result.

    Args:
        results: List of result dicts (modified in-place).
        uniprot_index: UniProt-to-clusters index.
        cluster_to_uniprot: Cluster-to-UniProt index.
        known_pairs: If provided, filter homologous pairs to known interactions.
        verbose: Print progress to stderr.
    """
    if verbose:
        # Report how many clusters exceed the size cap for pair detection
        n_oversized = sum(1 for members in cluster_to_uniprot.values()
                          if len(members) > MAX_CLUSTER_SIZE_FOR_PAIRS)
        print(f"  Annotating {len(results):,} complexes with clustering...",
              file=sys.stderr)
        if n_oversized:
            print(f"  Skipping {n_oversized:,} clusters with >{MAX_CLUSTER_SIZE_FOR_PAIRS} "
                  f"members for homologous pair detection (still reported in annotations)",
                  file=sys.stderr)

    n_with_clusters = 0
    n_with_shared = 0
    n_with_homologs = 0

    try:
        from tqdm import tqdm
        iterator = tqdm(results, desc="Clustering", file=sys.stderr,
                        disable=not verbose)
    except ImportError:
        iterator = results

    for row in iterator:
        # Cluster membership annotation
        annotate_pair_clusters(row, uniprot_index)

        if row['sequence_cluster_count'] > 0:
            n_with_clusters += 1

        if row['shared_cluster_count'] > 0:
            n_with_shared += 1

        # Homologous pair detection
        prot_a = row.get('protein_a', '')
        prot_b = row.get('protein_b', '')
        homologs = find_homologous_pairs(
            prot_a, prot_b, uniprot_index, cluster_to_uniprot,
            known_pairs=known_pairs,
        )

        if homologs:
            n_with_homologs += 1
            # Format as pipe-separated "X_Y" pairs, truncate if too many
            pair_strs = [f"{a}_{b}" for a, b in homologs]
            if len(pair_strs) > HOMOLOGOUS_PAIRS_DISPLAY_LIMIT:
                displayed = pair_strs[:HOMOLOGOUS_PAIRS_DISPLAY_LIMIT]
                remaining = len(pair_strs) - HOMOLOGOUS_PAIRS_DISPLAY_LIMIT
                row['homologous_pairs'] = '|'.join(displayed) + f"|+{remaining} more"
            else:
                row['homologous_pairs'] = '|'.join(pair_strs)
        else:
            row['homologous_pairs'] = ''

        row['n_homologous_pairs'] = len(homologs)
        row['homology_bitscore'] = ''  # populated later by enrich_with_homology_scores

    if verbose:
        print(f"  Clustering summary: {n_with_clusters:,} complexes with cluster "
              f"annotations, {n_with_shared:,} with shared clusters, "
              f"{n_with_homologs:,} with homologous pairs", file=sys.stderr)


def validate_clustering_mode(mode: str) -> str:
    """Validate and return clustering mode.

    Args:
        mode: Clustering mode string ('string', 'foldseek', or 'hybrid').

    Returns:
        The validated mode string.

    Raises:
        NotImplementedError: If mode is 'foldseek' or 'hybrid' (deferred).
        ValueError: If mode is not a valid clustering mode.
    """
    if mode not in VALID_CLUSTERING_MODES:
        raise ValueError(
            f"Invalid clustering mode: '{mode}'. "
            f"Valid modes: {', '.join(VALID_CLUSTERING_MODES)}"
        )

    if mode == 'foldseek':
        raise NotImplementedError(
            "Foldseek clustering deferred — requires ≥35 GB RAM for AFDB50. "
            "See Roadmap Decision 9."
        )

    if mode == 'hybrid':
        raise NotImplementedError(
            "Hybrid clustering deferred — requires Foldseek (Step B.2). "
            "See Roadmap Decision 9."
        )

    return mode


# ── Optional API Integration ─────────────────────────────────────────

HOMOLOGY_API_BATCH_SIZE = 100


def enrich_with_homology_scores(
    results: list[dict],
    uniprot_index: dict[str, set[str]],
    mapper: 'IDMapper',
    cache_dir: Optional[Union[str, bool]] = None,
    verbose: bool = False,
) -> None:
    """Enrich result rows with continuous homology bitscores from STRING API.

    Collects all unique proteins across pairs that share clusters,
    deduplicates them, then queries the STRING API homology endpoint in
    chunked batches of ``HOMOLOGY_API_BATCH_SIZE`` proteins per call.
    Results are accumulated into a score lookup and distributed back to
    all matching result rows.

    Falls back gracefully if the API is unavailable or a batch fails
    (subsequent batches are skipped on failure).

    Args:
        results: List of result dicts (modified in-place).
        uniprot_index: UniProt-to-clusters index.
        mapper: IDMapper instance for UniProt -> ENSP resolution.
        cache_dir: Cache directory for API responses. None = auto-cache,
            False = no cache.
        verbose: Print progress to stderr.
    """
    try:
        from string_api import query_homology, StringAPIError
    except ImportError:
        if verbose:
            print("  Skipping homology scores: string_api module not available",
                  file=sys.stderr)
        for row in results:
            row['homology_bitscore'] = ''
        return

    # Collect unique protein pairs that share clusters
    pair_indices: dict[tuple[str, str], list[int]] = {}
    for i, row in enumerate(results):
        if row.get('shared_cluster_count', 0) == 0:
            row['homology_bitscore'] = ''
            continue
        prot_a = row.get('protein_a', '')
        prot_b = row.get('protein_b', '')
        key = (min(prot_a, prot_b), max(prot_a, prot_b))
        pair_indices.setdefault(key, []).append(i)

    if not pair_indices:
        if verbose:
            print("  Homology scores: no pairs with shared clusters",
                  file=sys.stderr)
        return

    # Collect unique proteins across all qualifying pairs
    unique_proteins = set()
    for a, b in pair_indices:
        unique_proteins.add(a)
        unique_proteins.add(b)

    # Sort and chunk into batches
    proteins_sorted = sorted(unique_proteins)
    n_batches = (len(proteins_sorted) + HOMOLOGY_API_BATCH_SIZE - 1) // HOMOLOGY_API_BATCH_SIZE

    if verbose:
        print(f"  Querying STRING API for homology scores "
              f"({len(unique_proteins):,} proteins, "
              f"{len(pair_indices):,} unique pairs, "
              f"{n_batches} API call{'s' if n_batches != 1 else ''})...",
              file=sys.stderr)

    # Query in chunks, accumulate into score_lookup
    score_lookup: dict[tuple[str, str], float] = {}

    for batch_idx in range(n_batches):
        start = batch_idx * HOMOLOGY_API_BATCH_SIZE
        end = start + HOMOLOGY_API_BATCH_SIZE
        batch = proteins_sorted[start:end]

        if verbose and n_batches > 1:
            print(f"    Batch {batch_idx + 1}/{n_batches} "
                  f"({len(batch)} proteins)...", file=sys.stderr)

        try:
            df = query_homology(batch, cache_dir=cache_dir)
            if not df.empty and 'bitscore' in df.columns:
                for _, api_row in df.iterrows():
                    sid_a = str(api_row.get('stringId_A', ''))
                    sid_b = str(api_row.get('stringId_B', ''))
                    # Resolve STRING IDs back to UniProt
                    uniprots_a = mapper.ensembl_to_uniprot(sid_a) if sid_a else []
                    uniprots_b = mapper.ensembl_to_uniprot(sid_b) if sid_b else []
                    up_a = uniprots_a[0] if uniprots_a else ''
                    up_b = uniprots_b[0] if uniprots_b else ''
                    if up_a and up_b:
                        pair_key = (min(up_a, up_b), max(up_a, up_b))
                        bs = float(api_row['bitscore'])
                        # Keep max bitscore for each pair
                        if pair_key not in score_lookup or bs > score_lookup[pair_key]:
                            score_lookup[pair_key] = bs
        except Exception as e:
            if verbose:
                warnings.warn(
                    f"STRING API homology batch {batch_idx + 1}/{n_batches} "
                    f"failed: {e}. Skipping remaining batches.",
                    stacklevel=2,
                )
            break

    # Distribute scores to result rows
    n_scored = 0
    for pair_key, idxs in pair_indices.items():
        score = score_lookup.get(pair_key, '')
        for i in idxs:
            results[i]['homology_bitscore'] = score
        if score != '':
            n_scored += len(idxs)

    if verbose:
        print(f"  Homology scores: {n_scored:,} rows scored "
              f"({len(score_lookup):,} unique pairs from API)",
              file=sys.stderr)


# ── CLI ──────────────────────────────────────────────────────────────

def build_argument_parser() -> argparse.ArgumentParser:
    """Create and return the argument parser for standalone clustering analysis."""
    parser = argparse.ArgumentParser(
        description="Protein clustering and homology analysis using STRING sequence clusters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Print cluster summary statistics
    python protein_clustering.py --clusters-file data/clusters/9606.clusters.proteins.v12.0.txt \\
        --aliases data/ppi/9606.protein.aliases.v12.0.txt --summary

    # Look up clusters for a specific protein
    python protein_clustering.py --clusters-file data/clusters/9606.clusters.proteins.v12.0.txt \\
        --aliases data/ppi/9606.protein.aliases.v12.0.txt --protein P04637

    # Find shared clusters and homologous pairs for a protein pair
    python protein_clustering.py --clusters-file data/clusters/9606.clusters.proteins.v12.0.txt \\
        --aliases data/ppi/9606.protein.aliases.v12.0.txt --pair P04637 Q04206
        """,
    )

    parser.add_argument("--clusters-file", required=True, metavar="PATH",
                        help="Path to STRING clusters file "
                             "(e.g. 9606.clusters.proteins.v12.0.txt)")
    parser.add_argument("--aliases", required=True, metavar="PATH",
                        help="Path to STRING aliases file for ID resolution "
                             "(e.g. 9606.protein.aliases.v12.0.txt)")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--summary", action="store_true",
                       help="Print cluster summary statistics")
    group.add_argument("--protein", metavar="ID",
                       help="Look up clusters for a specific protein (UniProt or ENSP)")
    group.add_argument("--pair", nargs=2, metavar=("PROT_A", "PROT_B"),
                       help="Find shared clusters and homologous pairs for a protein pair")

    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    return parser


def main() -> None:
    """Run standalone clustering analysis."""
    parser = build_argument_parser()
    args = parser.parse_args()

    # Load clusters
    clusters_df = load_clusters(args.clusters_file, verbose=args.verbose)
    cluster_to_proteins, protein_to_clusters = build_cluster_index(clusters_df)

    # Load ID mapper
    from id_mapper import IDMapper
    mapper = IDMapper(args.aliases, verbose=args.verbose)
    uniprot_index = build_uniprot_cluster_index(
        protein_to_clusters, mapper, verbose=args.verbose,
    )
    cluster_to_uniprot_map = build_cluster_to_uniprot(uniprot_index)

    if args.summary:
        sizes = get_cluster_sizes(cluster_to_proteins)
        print(f"Total clusters: {len(sizes):,}")
        print(f"Total proteins (ENSP): {len(protein_to_clusters):,}")
        print(f"Total proteins (UniProt): {len(uniprot_index):,}")
        print(f"Cluster sizes: min={min(sizes.values())}, "
              f"max={max(sizes.values())}, "
              f"median={sorted(sizes.values())[len(sizes) // 2]}")

        # Distribution summary
        small = sum(1 for s in sizes.values() if s <= 10)
        medium = sum(1 for s in sizes.values() if 10 < s <= 100)
        large = sum(1 for s in sizes.values() if 100 < s <= 1000)
        xlarge = sum(1 for s in sizes.values() if s > 1000)
        print(f"Size distribution: <=10: {small}, 11-100: {medium}, "
              f"101-1000: {large}, >1000: {xlarge}")

    elif args.protein:
        protein_id = args.protein
        clusters = uniprot_index.get(protein_id, set())
        if not clusters:
            # Try resolving via mapper
            resolved = mapper.resolve_id(protein_id, target='uniprot')
            if resolved:
                clusters = uniprot_index.get(resolved, set())
                if clusters:
                    print(f"Resolved {protein_id} -> {resolved}")
                    protein_id = resolved

        if clusters:
            sizes = get_cluster_sizes(cluster_to_proteins)
            print(f"Protein {protein_id} belongs to {len(clusters)} clusters:")
            for cid in sorted(clusters):
                size = sizes.get(cid, 0)
                print(f"  {cid}: {size} members")
        else:
            print(f"Protein {protein_id} not found in any cluster")

    elif args.pair:
        prot_a, prot_b = args.pair
        shared = find_shared_clusters(prot_a, prot_b, uniprot_index)
        print(f"Proteins {prot_a} and {prot_b}:")
        print(f"  Clusters for {prot_a}: {len(uniprot_index.get(prot_a, set()))}")
        print(f"  Clusters for {prot_b}: {len(uniprot_index.get(prot_b, set()))}")
        print(f"  Shared clusters: {len(shared)}")

        if shared:
            sizes = get_cluster_sizes(cluster_to_proteins)
            print(f"  Shared cluster details:")
            for cid in sorted(shared):
                print(f"    {cid}: {sizes.get(cid, 0)} members")

            homologs = find_homologous_pairs(
                prot_a, prot_b, uniprot_index, cluster_to_uniprot_map,
            )
            print(f"  Homologous pairs: {len(homologs)}")
            for a, b in homologs[:20]:
                print(f"    {a} - {b}")
            if len(homologs) > 20:
                print(f"    ... and {len(homologs) - 20} more")


if __name__ == "__main__":
    main()
