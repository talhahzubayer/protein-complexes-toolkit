"""
Pathway mapping and network analysis module for the protein complexes toolkit.

Maps pipeline proteins to Reactome pathways using local flat files (offline-first),
optionally enriches with STRING API functional enrichment (GO/KEGG/Reactome p-values),
and builds interaction networks with NetworkX for downstream visualisation.

Data sources
------------
Local (offline-first, in ``data/pathways/``):
  - ``UniProt2Reactome_All_Levels.txt`` — UniProt → Reactome pathway mapping
  - ``ReactomePathwaysRelation.txt`` — Pathway hierarchy (parent → child)

STRING API (optional, via ``string_api.py``):
  - ``query_enrichment()`` — Functional enrichment with FDR
  - ``query_ppi_enrichment()`` — Network enrichment test
  - ``query_network()`` — Interaction edges

Usage
-----
Standalone CLI::

    python pathway_network.py summary --csv results.csv
    python pathway_network.py network --csv results.csv --output-dir Output/networks/

Toolkit integration::

    python toolkit.py Test_Data/ -o results.csv --interface --pae --enrich aliases.txt --pathways
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Union

try:
    import networkx as nx
    _HAS_NETWORKX = True
except ImportError:
    _HAS_NETWORKX = False

# ── Constants ─────────────────────────────────────────────────────

DEFAULT_PATHWAYS_DIR = Path(__file__).parent / "data" / "pathways"
REACTOME_MAPPINGS_FILENAME = "UniProt2Reactome_All_Levels.txt"
REACTOME_HIERARCHY_FILENAME = "ReactomePathwaysRelation.txt"

NETWORK_PDOCKQ_THRESHOLD = 0.23       # Medium+ tier for network edges
NETWORK_MAX_NODES_FOR_PLOT = 500      # Cap for readable spring layout
PATHWAY_DISPLAY_LIMIT = 20            # Max pathways before truncation
STRING_PPI_ENRICHMENT_LIMIT = 1000    # Safe limit for PPI enrichment (2000 times out)
PPI_ENRICHMENT_MIN_PROTEINS = 2       # Minimum proteins for meaningful enrichment test

CSV_FIELDNAMES_PATHWAYS = [
    'reactome_pathways_a', 'reactome_pathways_b',
    'n_reactome_pathways_a', 'n_reactome_pathways_b',
    'n_shared_pathways',
    'pathway_quality_context',
    'ppi_enrichment_pvalue',
    'ppi_enrichment_ratio',
    'network_degree_a', 'network_degree_b',
]


# ── Local Reactome Loading ───────────────────────────────────────

def load_reactome_mappings(
    filepath: Union[str, Path],
    accessions: Optional[frozenset[str]] = None,
    species: str = "Homo sapiens",
    verbose: bool = False,
) -> dict[str, list[dict]]:
    """Load UniProt-to-Reactome pathway mappings, filtered to species.

    Parameters
    ----------
    filepath : str or Path
        Path to ``UniProt2Reactome_All_Levels.txt``.
    accessions : frozenset[str] or None
        If provided, only retain mappings for these accessions.
    species : str
        Species name to filter (default: Homo sapiens).
    verbose : bool
        Print progress to stderr.

    Returns
    -------
    dict[str, list[dict]]
        ``{uniprot_accession: [{pathway_id, pathway_name, evidence_code}, ...]}``.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Reactome mappings not found: {filepath}")

    index: dict[str, list[dict]] = defaultdict(list)
    n_total = 0
    n_kept = 0

    if verbose:
        print(f"  Loading Reactome mappings from: {filepath.name}", file=sys.stderr)

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            acc, pathway_id, _url, pathway_name, evidence, line_species = parts[:6]
            if line_species.strip() != species:
                continue
            n_total += 1
            if accessions is not None and acc not in accessions:
                continue
            n_kept += 1
            index[acc].append({
                "pathway_id": pathway_id,
                "pathway_name": pathway_name.strip(),
                "evidence_code": evidence.strip(),
            })

    if verbose:
        print(f"  Reactome: {n_total:,} human entries, "
              f"{n_kept:,} kept for {len(index):,} proteins", file=sys.stderr)

    return dict(index)


def load_reactome_hierarchy(
    filepath: Union[str, Path],
    verbose: bool = False,
) -> dict[str, list[str]]:
    """Load Reactome pathway hierarchy (parent → child).

    Parameters
    ----------
    filepath : str or Path
        Path to ``ReactomePathwaysRelation.txt``.

    Returns
    -------
    dict[str, list[str]]
        ``{parent_pathway_id: [child_pathway_id, ...]}``.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Reactome hierarchy not found: {filepath}")

    hierarchy: dict[str, list[str]] = defaultdict(list)
    n = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                hierarchy[parts[0]].append(parts[1])
                n += 1

    if verbose:
        print(f"  Reactome hierarchy: {n:,} parent-child relations, "
              f"{len(hierarchy):,} parent pathways", file=sys.stderr)

    return dict(hierarchy)


# ── Pathway Quality Statistics ───────────────────────────────────

def compute_pathway_quality_stats(
    results: list[dict],
    reactome_index: dict[str, list[dict]],
) -> dict[str, dict]:
    """Compute quality statistics per pathway.

    For each Reactome pathway, aggregate across all complexes where at least
    one protein is in that pathway.

    Parameters
    ----------
    results : list[dict]
        Pipeline result rows with ``pdockq``, ``quality_tier_v2`` columns.
    reactome_index : dict[str, list[dict]]
        Reactome mapping from ``load_reactome_mappings``.

    Returns
    -------
    dict[str, dict]
        ``{pathway_id: {pathway_name, mean_pdockq, frac_high, n_complexes}}``.
    """
    # Build protein → set of pathway_ids
    protein_pathways: dict[str, set[str]] = defaultdict(set)
    pathway_names: dict[str, str] = {}
    for acc, mappings in reactome_index.items():
        for m in mappings:
            pid = m["pathway_id"]
            protein_pathways[acc].add(pid)
            pathway_names[pid] = m["pathway_name"]

    # Collect per-pathway quality data
    pathway_pdockqs: dict[str, list[float]] = defaultdict(list)
    pathway_tiers: dict[str, list[str]] = defaultdict(list)

    for row in results:
        pdockq = row.get("pdockq")
        tier = row.get("quality_tier_v2", row.get("quality_tier", ""))
        if pdockq is None:
            continue

        # Collect pathways for both proteins
        pids: set[str] = set()
        for suffix in ("a", "b"):
            acc = row.get(f"protein_{suffix}", "")
            base = acc.split("-")[0] if "-" in acc else acc
            pids.update(protein_pathways.get(acc, set()))
            pids.update(protein_pathways.get(base, set()))

        for pid in pids:
            pathway_pdockqs[pid].append(float(pdockq))
            if tier:
                pathway_tiers[pid].append(tier)

    stats: dict[str, dict] = {}
    for pid, pdockqs in pathway_pdockqs.items():
        tiers = pathway_tiers.get(pid, [])
        n_high = sum(1 for t in tiers if t == "High")
        stats[pid] = {
            "pathway_name": pathway_names.get(pid, ""),
            "mean_pdockq": sum(pdockqs) / len(pdockqs),
            "frac_high": n_high / len(tiers) if tiers else 0.0,
            "n_complexes": len(pdockqs),
        }

    return stats


# ── STRING API Enrichment (Optional) ─────────────────────────────

def run_string_enrichment(
    gene_symbols: list[str],
    cache_dir: Optional[Union[str, bool]] = None,
    verbose: bool = False,
) -> Optional[object]:
    """Run STRING API functional enrichment in batches.

    Returns
    -------
    pd.DataFrame or None
        Enrichment results, or None if API unavailable / fails.
    """
    try:
        from string_api import query_enrichment, STRING_API_MAX_ENRICHMENT_BATCH
    except ImportError:
        if verbose:
            print("  STRING API module not available, skipping enrichment",
                  file=sys.stderr)
        return None

    batch_size = STRING_API_MAX_ENRICHMENT_BATCH
    all_results = []

    if verbose:
        print(f"  Running STRING enrichment for {len(gene_symbols)} proteins "
              f"(batch size: {batch_size})...", file=sys.stderr)

    n_batches = (len(gene_symbols) + batch_size - 1) // batch_size
    n_failed = 0
    for i in range(0, len(gene_symbols), batch_size):
        batch = gene_symbols[i:i + batch_size]
        batch_num = i // batch_size + 1
        try:
            result = query_enrichment(batch, cache_dir=cache_dir)
            all_results.append(result)
            if verbose:
                print(f"    Batch {batch_num}/{n_batches} ({len(batch)} proteins)...",
                      file=sys.stderr)
        except Exception as e:
            n_failed += 1
            if verbose:
                print(f"    Batch {batch_num}/{n_batches} failed: {e}",
                      file=sys.stderr)

    if verbose and (all_results or n_failed):
        total_terms = sum(len(r) for r in all_results)
        print(f"  STRING enrichment complete: {len(all_results)} batches succeeded, "
              f"{n_failed} failed, {total_terms} raw terms", file=sys.stderr)

    if not all_results:
        return None

    import pandas as pd
    return pd.concat(all_results, ignore_index=True)


def run_ppi_enrichment(
    gene_symbols: list[str],
    cache_dir: Optional[Union[str, bool]] = None,
    verbose: bool = False,
) -> Optional[dict]:
    """Run STRING PPI enrichment test.

    When the input exceeds STRING_PPI_ENRICHMENT_LIMIT (2000), a
    deterministic random sample is drawn instead of sending all proteins
    (STRING API rejects >2000 nodes with HTTP 400).

    Returns
    -------
    dict or None
        PPI enrichment stats, or None if API unavailable / fails.
    """
    try:
        from string_api import query_ppi_enrichment
    except ImportError:
        return None

    try:
        proteins = list(gene_symbols)
        sampled = False
        if len(proteins) > STRING_PPI_ENRICHMENT_LIMIT:
            if verbose:
                print(f"  PPI enrichment: sampling {STRING_PPI_ENRICHMENT_LIMIT} "
                      f"of {len(proteins)} proteins (STRING API limit)",
                      file=sys.stderr)
            random.seed(42)  # reproducible across runs
            proteins = random.sample(proteins, STRING_PPI_ENRICHMENT_LIMIT)
            sampled = True

        result = query_ppi_enrichment(proteins, cache_dir=cache_dir)
        if sampled:
            result["sampled"] = True
            result["total_proteins"] = len(gene_symbols)
            result["sampled_proteins"] = STRING_PPI_ENRICHMENT_LIMIT
        if verbose:
            p_val = result.get("p_value", "N/A")
            obs = result.get("number_of_edges", 0)
            exp = result.get("expected_number_of_edges", 0)
            extra = " (sampled)" if sampled else ""
            print(f"  PPI enrichment{extra}: p={p_val}, {obs} observed vs "
                  f"{exp} expected edges", file=sys.stderr)
        return result
    except Exception as e:
        if verbose:
            print(f"  PPI enrichment failed: {e}", file=sys.stderr)
        return None


def invert_reactome_index(
    reactome_index: dict[str, list[dict]],
) -> dict[str, set[str]]:
    """Invert protein→pathways index to pathway→proteins.

    Parameters
    ----------
    reactome_index : dict[str, list[dict]]
        From ``load_reactome_mappings()``.

    Returns
    -------
    dict[str, set[str]]
        ``{pathway_id: {accession, ...}}``.
    """
    pathway_proteins: dict[str, set[str]] = defaultdict(set)
    for acc, mappings in reactome_index.items():
        for m in mappings:
            pathway_proteins[m["pathway_id"]].add(acc)
    return dict(pathway_proteins)


def run_per_pathway_ppi_enrichment(
    pathway_proteins: dict[str, set[str]],
    pathway_ids: set[str],
    cache_dir: Optional[Union[str, bool]] = None,
    verbose: bool = False,
) -> dict[str, dict]:
    """Query STRING PPI enrichment per Reactome pathway.

    For each pathway in *pathway_ids*, sends its member proteins to
    ``query_ppi_enrichment`` and collects per-pathway enrichment stats.

    Pathways with fewer than ``PPI_ENRICHMENT_MIN_PROTEINS`` members are
    skipped.  Pathways exceeding ``STRING_PPI_ENRICHMENT_LIMIT`` are
    down-sampled with a deterministic seed (pathway ID hash).

    Parameters
    ----------
    pathway_proteins : dict[str, set[str]]
        From ``invert_reactome_index()``.
    pathway_ids : set[str]
        Pathway IDs to query (typically shared pathways across all complexes).
    cache_dir : str, bool, or None
        Passed through to ``query_ppi_enrichment``.
    verbose : bool
        Print progress to stderr.

    Returns
    -------
    dict[str, dict]
        ``{pathway_id: {p_value, ratio, n_proteins, sampled}}``.
    """
    try:
        from string_api import query_ppi_enrichment
    except ImportError:
        return {}

    eligible = sorted(pid for pid in pathway_ids
                      if len(pathway_proteins.get(pid, set())) >= PPI_ENRICHMENT_MIN_PROTEINS)

    if verbose:
        print(f"  Per-pathway PPI enrichment: {len(eligible)} pathways "
              f"(of {len(pathway_ids)} shared)", file=sys.stderr)

    results: dict[str, dict] = {}
    for i, pid in enumerate(eligible):
        proteins = list(pathway_proteins[pid])
        sampled = False

        if len(proteins) > STRING_PPI_ENRICHMENT_LIMIT:
            random.seed(hash(pid))
            proteins = random.sample(proteins, STRING_PPI_ENRICHMENT_LIMIT)
            sampled = True

        try:
            raw = query_ppi_enrichment(proteins, cache_dir=cache_dir)
        except Exception as e:
            if verbose:
                print(f"    Pathway {pid} failed: {e}", file=sys.stderr)
            continue

        p_val = raw.get("p_value")
        obs = raw.get("number_of_edges", 0)
        exp = raw.get("expected_number_of_edges", 0)
        ratio = (obs / exp) if exp and exp > 0 else 0.0

        results[pid] = {
            "p_value": p_val,
            "ratio": ratio,
            "n_proteins": len(pathway_proteins[pid]),
            "sampled": sampled,
        }

        if verbose and (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(eligible)} pathways queried...",
                  file=sys.stderr)

    if verbose:
        print(f"  Per-pathway PPI enrichment complete: "
              f"{len(results)}/{len(eligible)} succeeded", file=sys.stderr)

    return results


# ── Network Construction ─────────────────────────────────────────

def build_interaction_network(
    results: list[dict],
    min_pdockq: float = NETWORK_PDOCKQ_THRESHOLD,
) -> object:
    """Build a NetworkX graph from pipeline results.

    Nodes are proteins (UniProt accessions), edges are predicted interactions
    with pDockQ above the threshold.

    Parameters
    ----------
    results : list[dict]
        Pipeline result rows.
    min_pdockq : float
        Minimum pDockQ score for inclusion.

    Returns
    -------
    nx.Graph
        Interaction network.

    Raises
    ------
    ImportError
        If NetworkX is not installed.
    """
    if not _HAS_NETWORKX:
        raise ImportError(
            "NetworkX is required for network construction. "
            "Install it with: pip install networkx"
        )

    G = nx.Graph()

    for row in results:
        pdockq = row.get("pdockq")
        if pdockq is None or float(pdockq) < min_pdockq:
            continue

        protein_a = row.get("protein_a", "")
        protein_b = row.get("protein_b", "")
        if not protein_a or not protein_b:
            continue

        # Add nodes with attributes
        for suffix, protein in [("a", protein_a), ("b", protein_b)]:
            if protein not in G:
                G.add_node(protein,
                           gene_symbol=row.get(f"gene_symbol_{suffix}", ""),
                           n_diseases=row.get(f"n_diseases_{suffix}", 0),
                           is_drug_target=row.get(f"is_drug_target_{suffix}", False))

        # Add edge with attributes
        tier = row.get("quality_tier_v2", row.get("quality_tier", ""))
        G.add_edge(protein_a, protein_b,
                   pdockq=float(pdockq),
                   quality_tier=tier,
                   composite_score=row.get("interface_confidence_score", 0),
                   complex_name=row.get("complex_name", ""))

    return G


def compute_network_stats(G: object) -> dict[str, dict]:
    """Compute per-node network statistics.

    Returns
    -------
    dict[str, dict]
        ``{protein: {degree, clustering_coeff}}``.
    """
    if not _HAS_NETWORKX:
        return {}

    stats: dict[str, dict] = {}
    clustering = nx.clustering(G)
    for node in G.nodes():
        stats[node] = {
            "degree": G.degree(node),
            "clustering_coeff": clustering.get(node, 0.0),
        }
    return stats


def extract_subnetwork(
    graph: object,
    proteins: set[str],
) -> object:
    """Extract induced subgraph for a set of proteins.

    Parameters
    ----------
    graph : nx.Graph
        Full interaction network.
    proteins : set[str]
        UniProt accessions to include.

    Returns
    -------
    nx.Graph
        Induced subgraph containing only the specified proteins.
    """
    if not _HAS_NETWORKX:
        raise ImportError("NetworkX required for subnetwork extraction.")
    present = set(graph.nodes()) & proteins
    return graph.subgraph(present).copy()


def add_regulatory_overlay(
    graph: object,
    network_df: object,
) -> object:
    """Convert undirected graph to DiGraph with STRING edge directionality.

    Takes STRING API ``query_network()`` output and overlays edge score
    components as directed attributes. If a protein pair exists in both
    the AF2 graph and the STRING network, the STRING score components
    are added as edge attributes.

    Parameters
    ----------
    graph : nx.Graph
        Undirected interaction network from ``build_interaction_network``.
    network_df : pd.DataFrame
        Edge DataFrame from ``string_api.query_network()``.

    Returns
    -------
    nx.DiGraph
        Directed graph with STRING edge attributes overlaid.
    """
    if not _HAS_NETWORKX:
        raise ImportError("NetworkX required for regulatory overlay.")

    DG = nx.DiGraph(graph)

    if network_df is None or len(network_df) == 0:
        return DG

    # Build a lookup from STRING preferred names to edge data
    for _, row in network_df.iterrows():
        name_a = row.get("preferredName_A", "")
        name_b = row.get("preferredName_B", "")
        score = row.get("score", 0)

        # Try to match STRING names to graph nodes (via gene_symbol attribute)
        node_a = None
        node_b = None
        for node, data in DG.nodes(data=True):
            gs = data.get("gene_symbol", "")
            if gs == name_a:
                node_a = node
            elif gs == name_b:
                node_b = node

        if node_a and node_b and DG.has_edge(node_a, node_b):
            DG[node_a][node_b]["string_score"] = score
            DG[node_a][node_b]["string_source"] = "functional"

    return DG


def run_string_network(
    gene_symbols: list[str],
    network_type: str = "functional",
    cache_dir: Optional[Union[str, bool]] = None,
    verbose: bool = False,
) -> Optional[object]:
    """Wrapper around ``string_api.query_network()`` with error handling.

    Returns
    -------
    pd.DataFrame or None
        Network edges, or None if API unavailable / fails.
    """
    try:
        from string_api import query_network
    except ImportError:
        if verbose:
            print("  STRING API module not available, skipping network query",
                  file=sys.stderr)
        return None

    try:
        result = query_network(
            gene_symbols, network_type=network_type, cache_dir=cache_dir,
        )
        if verbose:
            print(f"  STRING network ({network_type}): {len(result)} edges",
                  file=sys.stderr)
        return result
    except Exception as e:
        if verbose:
            print(f"  STRING network query failed: {e}", file=sys.stderr)
        return None


# ── Network Visualisation ────────────────────────────────────────

def plot_network_by_pdockq(
    graph: object,
    output_path: Union[str, Path],
    max_nodes: int = NETWORK_MAX_NODES_FOR_PLOT,
) -> None:
    """Spring layout network with edges coloured by pDockQ.

    This is the spec requirement: 'Network plot coloured by predicted pDockQ'.

    Parameters
    ----------
    graph : nx.Graph
        Interaction network from ``build_interaction_network``.
    output_path : str or Path
        Path to save the figure (PNG).
    max_nodes : int
        Maximum nodes to display (top by degree).
    """
    if not _HAS_NETWORKX:
        raise ImportError("NetworkX required for network plots.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    G = graph
    # Cap nodes for readability
    if G.number_of_nodes() > max_nodes:
        top_nodes = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:max_nodes]
        G = G.subgraph(top_nodes).copy()

    if G.number_of_nodes() == 0:
        return

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    pos = nx.spring_layout(G, k=1.5 / (G.number_of_nodes() ** 0.5 + 1), seed=42)

    # Node sizes proportional to degree
    degrees = dict(G.degree())
    node_sizes = [max(30, degrees[n] * 20) for n in G.nodes()]

    # Edge colours from pDockQ
    edge_pdockqs = [G[u][v].get("pdockq", 0.0) for u, v in G.edges()]
    if edge_pdockqs:
        edge_norm = plt.Normalize(vmin=min(edge_pdockqs), vmax=max(edge_pdockqs))
        edge_colors = [cm.RdYlGn(edge_norm(p)) for p in edge_pdockqs]
        edge_widths = [0.5 + 2.0 * edge_norm(p) for p in edge_pdockqs]
    else:
        edge_colors = ["grey"]
        edge_widths = [1.0]

    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors,
                           width=edge_widths, alpha=0.6)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                           node_color="#3498db", alpha=0.8, edgecolors="white",
                           linewidths=0.5)

    # Labels for high-degree nodes only
    degree_threshold = max(1, sorted(degrees.values(), reverse=True)[
        min(20, len(degrees) - 1)] if degrees else 1)
    labels = {n: G.nodes[n].get("gene_symbol", n)
              for n in G.nodes() if degrees[n] >= degree_threshold}
    nx.draw_networkx_labels(G, pos, labels, font_size=7, font_weight="bold", ax=ax)

    # Colourbar for pDockQ
    if edge_pdockqs:
        sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=edge_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label("pDockQ", fontsize=10)

    ax.set_title(f"Interaction Network (n={G.number_of_nodes()} proteins, "
                 f"e={G.number_of_edges()} edges)", fontsize=13, fontweight="bold")
    ax.axis("off")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_network_by_quality(
    graph: object,
    output_path: Union[str, Path],
    max_nodes: int = NETWORK_MAX_NODES_FOR_PLOT,
) -> None:
    """Spring layout network with nodes coloured by quality tier.

    Parameters
    ----------
    graph : nx.Graph
        Interaction network from ``build_interaction_network``.
    output_path : str or Path
        Path to save the figure (PNG).
    max_nodes : int
        Maximum nodes to display.
    """
    if not _HAS_NETWORKX:
        raise ImportError("NetworkX required for network plots.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    TIER_COLORS = {"High": "#2ecc71", "Medium": "#f39c12", "Low": "#e74c3c"}

    G = graph
    if G.number_of_nodes() > max_nodes:
        top_nodes = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:max_nodes]
        G = G.subgraph(top_nodes).copy()

    if G.number_of_nodes() == 0:
        return

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    pos = nx.spring_layout(G, k=1.5 / (G.number_of_nodes() ** 0.5 + 1), seed=42)

    # Edge colour by quality tier (use the edge's quality_tier attribute)
    for u, v, data in G.edges(data=True):
        tier = data.get("quality_tier", "Low")
        color = TIER_COLORS.get(tier, "#95a5a6")
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=color, alpha=0.4, linewidth=0.8)

    degrees = dict(G.degree())
    node_sizes = [max(30, degrees[n] * 20) for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                           node_color="#3498db", alpha=0.8, edgecolors="white",
                           linewidths=0.5)

    degree_threshold = max(1, sorted(degrees.values(), reverse=True)[
        min(20, len(degrees) - 1)] if degrees else 1)
    labels = {n: G.nodes[n].get("gene_symbol", n)
              for n in G.nodes() if degrees[n] >= degree_threshold}
    nx.draw_networkx_labels(G, pos, labels, font_size=7, font_weight="bold", ax=ax)

    # Legend
    import matplotlib.patches as mpatches
    legend_handles = [mpatches.Patch(color=c, label=t) for t, c in TIER_COLORS.items()]
    ax.legend(handles=legend_handles, fontsize=9, loc="upper left", title="Quality Tier")

    ax.set_title(f"Interaction Network by Quality Tier", fontsize=13, fontweight="bold")
    ax.axis("off")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_disease_network(
    graph: object,
    output_path: Union[str, Path],
    max_nodes: int = NETWORK_MAX_NODES_FOR_PLOT,
) -> None:
    """Network with disease-associated nodes highlighted.

    Disease-associated nodes get a red border ring. Drug target nodes
    are drawn as diamonds. Variant burden shown as node border width.

    Parameters
    ----------
    graph : nx.Graph
        Interaction network from ``build_interaction_network``.
    output_path : str or Path
        Path to save the figure (PNG).
    max_nodes : int
        Maximum nodes to display.
    """
    if not _HAS_NETWORKX:
        raise ImportError("NetworkX required for network plots.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    G = graph
    if G.number_of_nodes() > max_nodes:
        top_nodes = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:max_nodes]
        G = G.subgraph(top_nodes).copy()

    if G.number_of_nodes() == 0:
        return

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    pos = nx.spring_layout(G, k=1.5 / (G.number_of_nodes() ** 0.5 + 1), seed=42)

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#cccccc",
                           width=0.5, alpha=0.5)

    degrees = dict(G.degree())

    # Separate disease and non-disease nodes
    disease_nodes = [n for n in G.nodes()
                     if G.nodes[n].get("n_diseases", 0) > 0]
    drug_nodes = [n for n in G.nodes()
                  if G.nodes[n].get("is_drug_target", False)]
    normal_nodes = [n for n in G.nodes()
                    if n not in disease_nodes and n not in drug_nodes]

    for nodes, color, marker, label in [
        (normal_nodes, "#95a5a6", "o", "No disease"),
        (disease_nodes, "#e74c3c", "o", "Disease-associated"),
        (drug_nodes, "#9b59b6", "D", "Drug target"),
    ]:
        if not nodes:
            continue
        sizes = [max(30, degrees.get(n, 1) * 20) for n in nodes]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, ax=ax,
                               node_size=sizes, node_color=color,
                               alpha=0.8, node_shape=marker,
                               edgecolors="black", linewidths=0.5,
                               label=label)

    degree_threshold = max(1, sorted(degrees.values(), reverse=True)[
        min(15, len(degrees) - 1)] if degrees else 1)
    labels = {n: G.nodes[n].get("gene_symbol", n)
              for n in G.nodes() if degrees[n] >= degree_threshold}
    nx.draw_networkx_labels(G, pos, labels, font_size=7, font_weight="bold", ax=ax)

    ax.legend(fontsize=9, loc="upper left")
    ax.set_title("Disease & Drug Target Network", fontsize=13, fontweight="bold")
    ax.axis("off")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Formatting ───────────────────────────────────────────────────

def format_reactome_pathways(
    mappings: list[dict],
    limit: int = PATHWAY_DISPLAY_LIMIT,
) -> str:
    """Format Reactome pathway mappings as pipe-separated string.

    Format: ``R-HSA-109581:Apoptosis|R-HSA-72766:Translation|...``

    Deduplicates by pathway_id (same pathway may appear with different evidence).
    """
    if not mappings:
        return ""
    # Deduplicate by pathway_id
    seen: set[str] = set()
    unique: list[dict] = []
    for m in mappings:
        pid = m["pathway_id"]
        if pid not in seen:
            seen.add(pid)
            unique.append(m)

    parts: list[str] = []
    for m in unique[:limit]:
        parts.append(f"{m['pathway_id']}:{m['pathway_name']}")
    result = "|".join(parts)
    if len(unique) > limit:
        result += f"|...(+{len(unique) - limit} more)"
    return result


def format_pathway_quality_context(stats: dict) -> str:
    """Format pathway quality stats as semicolon-separated string.

    Format: ``mean_pdockq=0.45;frac_high=0.30;n_complexes=12``
    """
    if not stats:
        return ""
    return (f"mean_pdockq={stats.get('mean_pdockq', 0):.3f};"
            f"frac_high={stats.get('frac_high', 0):.3f};"
            f"n_complexes={stats.get('n_complexes', 0)}")


# ── Annotation ───────────────────────────────────────────────────

def annotate_results_with_pathways(
    results: list[dict],
    reactome_index: dict[str, list[dict]],
    pathway_stats: Optional[dict[str, dict]] = None,
    ppi_stats: Optional[dict] = None,
    pathway_ppi_stats: Optional[dict[str, dict]] = None,
    network_stats: Optional[dict[str, dict]] = None,
    enrichment_df: Optional[object] = None,
    verbose: bool = False,
) -> None:
    """Annotate result rows with pathway and network columns.

    Modifies *results* **in-place** — no return value.

    When *enrichment_df* is provided (from STRING API ``query_enrichment``),
    it augments the local Reactome pathway assignments with statistical
    significance (FDR values). This follows the offline-first + API validation
    pattern: local Reactome provides the base, STRING enrichment validates
    with p-values and FDR.

    Parameters
    ----------
    results : list[dict]
        Per-complex result dicts from the pipeline.
    reactome_index : dict[str, list[dict]]
        Reactome mappings from ``load_reactome_mappings``.
    pathway_stats : dict or None
        Pathway quality stats from ``compute_pathway_quality_stats``.
    ppi_stats : dict or None
        Legacy global PPI enrichment from ``run_ppi_enrichment``.
        Used only as fallback when *pathway_ppi_stats* is not provided.
    pathway_ppi_stats : dict or None
        Per-pathway PPI enrichment from ``run_per_pathway_ppi_enrichment``.
        When provided, each row gets the enrichment of its best (most
        significant) shared pathway.
    network_stats : dict or None
        Per-node network stats from ``compute_network_stats``.
    enrichment_df : pd.DataFrame or None
        STRING API functional enrichment results. When provided, used to
        augment pathway_quality_context with FDR for statistically significant
        pathways.
    verbose : bool
        Print progress to stderr.
    """
    if verbose:
        print(f"  Annotating {len(results)} complexes with pathway data...",
              file=sys.stderr)

    # Pre-compute enrichment FDR lookup (STRING API validation of local Reactome)
    enrichment_fdr: dict[str, float] = {}
    if enrichment_df is not None and len(enrichment_df) > 0:
        for _, erow in enrichment_df.iterrows():
            category = erow.get("category", "")
            term = erow.get("term", "")
            fdr = erow.get("fdr", 1.0)
            # Match Reactome terms (category == "RCTM" in STRING output)
            if category in ("RCTM", "KEGG", "Component", "Function", "Process"):
                enrichment_fdr[term] = float(fdr)
        if verbose:
            sig_count = sum(1 for f in enrichment_fdr.values() if f < 0.05)
            print(f"  STRING enrichment: {len(enrichment_fdr)} terms, "
                  f"{sig_count} significant (FDR < 0.05)", file=sys.stderr)

    # Pre-compute global PPI enrichment values (legacy fallback)
    global_ppi_pvalue = ""
    global_ppi_ratio = ""
    if ppi_stats and not pathway_ppi_stats:
        p_val = ppi_stats.get("p_value")
        if p_val is not None:
            global_ppi_pvalue = (f"{p_val:.2e}"
                                 if isinstance(p_val, (int, float)) else str(p_val))
        obs = ppi_stats.get("number_of_edges", 0)
        exp = ppi_stats.get("expected_number_of_edges", 0)
        if exp and exp > 0:
            global_ppi_ratio = f"{obs / exp:.2f}"

    for row in results:
        protein_a = row.get("protein_a", "")
        protein_b = row.get("protein_b", "")
        base_a = protein_a.split("-")[0] if "-" in protein_a else protein_a
        base_b = protein_b.split("-")[0] if "-" in protein_b else protein_b

        # Reactome pathways per chain
        mappings_a = reactome_index.get(protein_a, []) or reactome_index.get(base_a, [])
        mappings_b = reactome_index.get(protein_b, []) or reactome_index.get(base_b, [])

        row["reactome_pathways_a"] = format_reactome_pathways(mappings_a)
        row["reactome_pathways_b"] = format_reactome_pathways(mappings_b)

        # Deduplicated counts
        pids_a = {m["pathway_id"] for m in mappings_a}
        pids_b = {m["pathway_id"] for m in mappings_b}
        row["n_reactome_pathways_a"] = len(pids_a)
        row["n_reactome_pathways_b"] = len(pids_b)
        row["n_shared_pathways"] = len(pids_a & pids_b)

        # Pathway quality context (use the most specific shared pathway)
        shared = pids_a & pids_b
        if shared and pathway_stats:
            # Pick the pathway with the most complexes
            best_pid = max(shared,
                          key=lambda p: pathway_stats.get(p, {}).get("n_complexes", 0))
            context = format_pathway_quality_context(
                pathway_stats.get(best_pid, {}))
            # Augment with STRING enrichment FDR if available
            if enrichment_fdr and best_pid in enrichment_fdr:
                context += f";enrichment_fdr={enrichment_fdr[best_pid]:.2e}"
            row["pathway_quality_context"] = context
        else:
            row["pathway_quality_context"] = ""

        # PPI enrichment — per-pathway when available, global fallback
        if pathway_ppi_stats and shared:
            # Pick shared pathway with the lowest (most significant) p-value
            best_ppi = None
            best_p = float("inf")
            for pid in shared:
                pstats = pathway_ppi_stats.get(pid)
                if pstats and pstats.get("p_value") is not None:
                    pv = float(pstats["p_value"])
                    if pv < best_p:
                        best_p = pv
                        best_ppi = pstats
            if best_ppi is not None:
                pv = best_ppi["p_value"]
                row["ppi_enrichment_pvalue"] = (
                    f"{pv:.2e}" if isinstance(pv, (int, float)) else str(pv))
                row["ppi_enrichment_ratio"] = f"{best_ppi['ratio']:.2f}"
            else:
                row["ppi_enrichment_pvalue"] = ""
                row["ppi_enrichment_ratio"] = ""
        else:
            row["ppi_enrichment_pvalue"] = global_ppi_pvalue
            row["ppi_enrichment_ratio"] = global_ppi_ratio

        # Network stats per protein
        if network_stats:
            stats_a = network_stats.get(protein_a, network_stats.get(base_a, {}))
            stats_b = network_stats.get(protein_b, network_stats.get(base_b, {}))
            row["network_degree_a"] = stats_a.get("degree", "")
            row["network_degree_b"] = stats_b.get("degree", "")
        else:
            row["network_degree_a"] = ""
            row["network_degree_b"] = ""

    if verbose:
        n_with_pathways = sum(
            1 for r in results
            if r.get("n_reactome_pathways_a", 0) > 0 or
               r.get("n_reactome_pathways_b", 0) > 0
        )
        print(f"  Pathway annotation: {n_with_pathways} complexes with "
              f"Reactome pathways", file=sys.stderr)


# ── Standalone CLI ───────────────────────────────────────────────

def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="pathway_network",
        description="Pathway mapping and network analysis for protein complexes.",
    )
    parser.add_argument(
        "--pathways-dir", type=str,
        default=str(DEFAULT_PATHWAYS_DIR),
        help=f"Path to directory with Reactome files. Default: {DEFAULT_PATHWAYS_DIR}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    # summary
    sub_summary = subparsers.add_parser(
        "summary",
        help="Print pathway coverage statistics for a pipeline CSV.",
    )
    sub_summary.add_argument("--csv", required=True,
                             help="Input CSV from toolkit.py.")

    # network
    sub_network = subparsers.add_parser(
        "network",
        help="Build and visualise interaction network from pipeline CSV.",
    )
    sub_network.add_argument("--csv", required=True,
                             help="Input CSV from toolkit.py.")
    sub_network.add_argument("--output-dir", type=str,
                             default=str(DEFAULT_PATHWAYS_DIR.parent.parent / "Output" / "networks"),
                             help="Output directory for network plots.")
    sub_network.add_argument("--min-pdockq", type=float,
                             default=NETWORK_PDOCKQ_THRESHOLD,
                             help=f"Minimum pDockQ for network edges (default: {NETWORK_PDOCKQ_THRESHOLD})")

    # enrichment
    sub_enrich = subparsers.add_parser(
        "enrichment",
        help="Run STRING API enrichment on pipeline proteins.",
    )
    sub_enrich.add_argument("--csv", required=True,
                             help="Input CSV from toolkit.py.")

    return parser


def _cli_summary(csv_path: str, pathways_dir: str) -> None:
    """Print pathway coverage summary."""
    import pandas as pd
    df = pd.read_csv(csv_path)

    accessions: set[str] = set()
    for col in ("protein_a", "protein_b"):
        if col in df.columns:
            accessions.update(df[col].dropna().unique())

    reactome_path = Path(pathways_dir) / REACTOME_MAPPINGS_FILENAME
    if not reactome_path.exists():
        print(f"Error: {reactome_path} not found", file=sys.stderr)
        sys.exit(1)

    reactome_index = load_reactome_mappings(
        reactome_path, frozenset(accessions), verbose=True,
    )

    # Count coverage
    mapped = sum(1 for acc in accessions if acc in reactome_index)
    all_pathways: set[str] = set()
    for mappings in reactome_index.values():
        for m in mappings:
            all_pathways.add(m["pathway_id"])

    print(f"\nPathway coverage summary:")
    print(f"  Total proteins: {len(accessions):,}")
    print(f"  Mapped to Reactome: {mapped:,} ({100*mapped/len(accessions):.1f}%)")
    print(f"  Unique pathways: {len(all_pathways):,}")

    # Top pathways by protein count
    pathway_counts: Counter = Counter()
    for mappings in reactome_index.values():
        for m in mappings:
            pathway_counts[m["pathway_id"]] += 1
    print(f"\n  Top 10 pathways by protein count:")
    for pid, count in pathway_counts.most_common(10):
        name = ""
        for mappings in reactome_index.values():
            for m in mappings:
                if m["pathway_id"] == pid:
                    name = m["pathway_name"]
                    break
            if name:
                break
        print(f"    {pid}: {name} ({count} proteins)")


def _cli_network(csv_path: str, output_dir: str, min_pdockq: float) -> None:
    """Build and visualise interaction network."""
    if not _HAS_NETWORKX:
        print("Error: NetworkX is required. Install with: pip install networkx",
              file=sys.stderr)
        sys.exit(1)

    import pandas as pd
    df = pd.read_csv(csv_path)
    results = df.to_dict("records")

    G = build_interaction_network(results, min_pdockq=min_pdockq)
    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  Min pDockQ threshold: {min_pdockq}")

    # Network statistics
    if G.number_of_nodes() > 0:
        degrees = [d for _, d in G.degree()]
        print(f"  Mean degree: {sum(degrees)/len(degrees):.2f}")
        print(f"  Max degree: {max(degrees)}")

        # Generate network plots
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating network plots in {out}...")
        plot_network_by_pdockq(G, out / "network_pdockq.png")
        print(f"  Saved: network_pdockq.png")

        plot_network_by_quality(G, out / "network_quality_tier.png")
        print(f"  Saved: network_quality_tier.png")

        # Disease network if disease data available
        has_disease = any(G.nodes[n].get("n_diseases", 0) > 0 for n in G.nodes())
        if has_disease:
            plot_disease_network(G, out / "network_disease.png")
            print(f"  Saved: network_disease.png")


def _cli_enrichment(csv_path: str) -> None:
    """Run STRING API functional enrichment on pipeline proteins."""
    import pandas as pd
    df = pd.read_csv(csv_path)

    # Collect unique gene symbols
    gene_symbols: set[str] = set()
    for col in ("gene_symbol_a", "gene_symbol_b"):
        if col in df.columns:
            gene_symbols.update(df[col].dropna().unique())

    if not gene_symbols:
        # Fall back to protein accessions
        for col in ("protein_a", "protein_b"):
            if col in df.columns:
                gene_symbols.update(df[col].dropna().unique())

    if not gene_symbols:
        print("Error: No gene symbols or protein accessions found in CSV.",
              file=sys.stderr)
        sys.exit(1)

    print(f"Running STRING API enrichment for {len(gene_symbols)} proteins...")

    # Functional enrichment
    enrichment_df = run_string_enrichment(list(gene_symbols), verbose=True)
    if enrichment_df is not None and len(enrichment_df) > 0:
        sig = enrichment_df[enrichment_df["fdr"] < 0.05] if "fdr" in enrichment_df.columns else enrichment_df
        print(f"\nFunctional enrichment results:")
        print(f"  Total terms: {len(enrichment_df)}")
        print(f"  Significant (FDR < 0.05): {len(sig)}")

        if "category" in sig.columns:
            for cat in sig["category"].unique():
                cat_df = sig[sig["category"] == cat]
                print(f"\n  {cat} ({len(cat_df)} terms):")
                for _, row in cat_df.head(5).iterrows():
                    desc = row.get("description", row.get("term", ""))
                    fdr = row.get("fdr", "")
                    n_genes = row.get("number_of_genes", "")
                    print(f"    - {desc} (FDR={fdr:.2e}, {n_genes} genes)")
    else:
        print("  No enrichment results (API may be unavailable).")

    # PPI enrichment
    ppi = run_ppi_enrichment(list(gene_symbols), verbose=True)
    if ppi:
        print(f"\nPPI enrichment:")
        print(f"  Nodes: {ppi.get('number_of_nodes', 'N/A')}")
        print(f"  Edges: {ppi.get('number_of_edges', 'N/A')} "
              f"(expected: {ppi.get('expected_number_of_edges', 'N/A')})")
        print(f"  p-value: {ppi.get('p_value', 'N/A')}")


def main() -> None:
    """CLI entry point."""
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "summary":
        _cli_summary(args.csv, args.pathways_dir)
    elif args.command == "network":
        _cli_network(args.csv, args.output_dir, args.min_pdockq)
    elif args.command == "enrichment":
        _cli_enrichment(args.csv)


if __name__ == "__main__":
    main()
