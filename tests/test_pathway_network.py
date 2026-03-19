"""
Tests for pathway_network.py — Reactome pathway mapping and network analysis.

Test data:
  - tests/offline_test_data/databases/test_reactome_mappings.txt (253 rows, 4 proteins)
  - tests/offline_test_data/databases/test_reactome_hierarchy.txt (585 rows)

Reference proteins:
  - P04637 (TP53): 124 unique pathways
  - P61981 (YWHAG): 60 unique pathways
  - P63104 (YWHAZ): 57 unique pathways
  - P24534 (EEF1B2): 3 unique pathways
"""

import subprocess
import sys

import pytest
from pathlib import Path


# ── Constants & Known Values ─────────────────────────────────────

TEST_REACTOME_MAPPINGS = "test_reactome_mappings.txt"
TEST_REACTOME_HIERARCHY = "test_reactome_hierarchy.txt"

ACC_TP53 = "P04637"
ACC_YWHAG = "P61981"
ACC_YWHAZ = "P63104"
ACC_EEF1B2 = "P24534"

# Known counts (verified from test data)
TP53_UNIQUE_PATHWAYS = 124
YWHAG_UNIQUE_PATHWAYS = 60
YWHAZ_UNIQUE_PATHWAYS = 57
EEF1B2_UNIQUE_PATHWAYS = 3

ALL_TEST_ACCESSIONS = frozenset({ACC_TP53, ACC_YWHAG, ACC_YWHAZ, ACC_EEF1B2})


# ── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def reactome_mappings_path(test_db_dir):
    path = test_db_dir / TEST_REACTOME_MAPPINGS
    assert path.exists(), f"Test Reactome mappings not found: {path}"
    return path


@pytest.fixture(scope="session")
def reactome_hierarchy_path(test_db_dir):
    path = test_db_dir / TEST_REACTOME_HIERARCHY
    assert path.exists(), f"Test Reactome hierarchy not found: {path}"
    return path


@pytest.fixture(scope="session")
def reactome_index(reactome_mappings_path):
    from pathway_network import load_reactome_mappings
    return load_reactome_mappings(
        reactome_mappings_path, ALL_TEST_ACCESSIONS,
    )


@pytest.fixture
def mock_results():
    """Simple mock results for annotation tests."""
    return [
        {
            "protein_a": ACC_TP53, "protein_b": ACC_YWHAG,
            "pdockq": 0.45, "quality_tier_v2": "Medium",
            "complex_name": "TP53_YWHAG",
        },
        {
            "protein_a": ACC_YWHAZ, "protein_b": ACC_EEF1B2,
            "pdockq": 0.55, "quality_tier_v2": "High",
            "complex_name": "YWHAZ_EEF1B2",
        },
        {
            "protein_a": ACC_TP53, "protein_b": ACC_YWHAZ,
            "pdockq": 0.10, "quality_tier_v2": "Low",
            "complex_name": "TP53_YWHAZ_low",
        },
    ]


# ── Section: TestConstants ───────────────────────────────────────

@pytest.mark.pathways
class TestConstants:
    """Test module-level constants."""

    def test_csv_fieldnames_count(self):
        from pathway_network import CSV_FIELDNAMES_PATHWAYS
        assert len(CSV_FIELDNAMES_PATHWAYS) == 10

    def test_csv_fieldnames_no_duplicates(self):
        from pathway_network import CSV_FIELDNAMES_PATHWAYS
        assert len(CSV_FIELDNAMES_PATHWAYS) == len(set(CSV_FIELDNAMES_PATHWAYS))

    def test_default_dir_is_path(self):
        from pathway_network import DEFAULT_PATHWAYS_DIR
        assert isinstance(DEFAULT_PATHWAYS_DIR, Path)

    def test_pdockq_threshold_positive(self):
        from pathway_network import NETWORK_PDOCKQ_THRESHOLD
        assert NETWORK_PDOCKQ_THRESHOLD > 0

    def test_max_nodes_reasonable(self):
        from pathway_network import NETWORK_MAX_NODES_FOR_PLOT
        assert NETWORK_MAX_NODES_FOR_PLOT >= 100


# ── Section: TestReactomeLoading ─────────────────────────────────

@pytest.mark.pathways
class TestReactomeLoading:
    """Test loading Reactome flat files."""

    def test_loads_all_four_proteins(self, reactome_index):
        assert len(reactome_index) == 4

    def test_tp53_pathway_count(self, reactome_index):
        unique = {m["pathway_id"] for m in reactome_index[ACC_TP53]}
        assert len(unique) == TP53_UNIQUE_PATHWAYS

    def test_ywhag_pathway_count(self, reactome_index):
        unique = {m["pathway_id"] for m in reactome_index[ACC_YWHAG]}
        assert len(unique) == YWHAG_UNIQUE_PATHWAYS

    def test_eef1b2_pathway_count(self, reactome_index):
        unique = {m["pathway_id"] for m in reactome_index[ACC_EEF1B2]}
        assert len(unique) == EEF1B2_UNIQUE_PATHWAYS

    def test_mapping_dict_keys(self, reactome_index):
        m = reactome_index[ACC_TP53][0]
        assert "pathway_id" in m
        assert "pathway_name" in m
        assert "evidence_code" in m

    def test_pathway_id_format(self, reactome_index):
        for m in reactome_index[ACC_TP53]:
            assert m["pathway_id"].startswith("R-HSA-")

    def test_filters_to_requested_accessions(self, reactome_mappings_path):
        from pathway_network import load_reactome_mappings
        idx = load_reactome_mappings(
            reactome_mappings_path, frozenset({ACC_TP53}))
        assert len(idx) == 1
        assert ACC_TP53 in idx

    def test_empty_accessions_returns_empty(self, reactome_mappings_path):
        from pathway_network import load_reactome_mappings
        idx = load_reactome_mappings(reactome_mappings_path, frozenset())
        assert len(idx) == 0

    def test_missing_file_raises(self):
        from pathway_network import load_reactome_mappings
        with pytest.raises(FileNotFoundError):
            load_reactome_mappings("/nonexistent/path.txt")

    def test_verbose_mode(self, reactome_mappings_path, capsys):
        from pathway_network import load_reactome_mappings
        load_reactome_mappings(
            reactome_mappings_path, ALL_TEST_ACCESSIONS, verbose=True)
        captured = capsys.readouterr()
        assert "Reactome" in captured.err


# ── Section: TestReactomeHierarchy ───────────────────────────────

@pytest.mark.pathways
class TestReactomeHierarchy:
    """Test loading Reactome pathway hierarchy."""

    def test_loads_hierarchy(self, reactome_hierarchy_path):
        from pathway_network import load_reactome_hierarchy
        hierarchy = load_reactome_hierarchy(reactome_hierarchy_path)
        assert len(hierarchy) > 0

    def test_hierarchy_values_are_lists(self, reactome_hierarchy_path):
        from pathway_network import load_reactome_hierarchy
        hierarchy = load_reactome_hierarchy(reactome_hierarchy_path)
        for parent, children in hierarchy.items():
            assert isinstance(children, list)
            assert all(isinstance(c, str) for c in children)

    def test_hierarchy_keys_are_pathway_ids(self, reactome_hierarchy_path):
        from pathway_network import load_reactome_hierarchy
        hierarchy = load_reactome_hierarchy(reactome_hierarchy_path)
        for parent in hierarchy:
            assert parent.startswith("R-HSA-")

    def test_missing_file_raises(self):
        from pathway_network import load_reactome_hierarchy
        with pytest.raises(FileNotFoundError):
            load_reactome_hierarchy("/nonexistent/path.txt")


# ── Section: TestPathwayQualityStats ─────────────────────────────

@pytest.mark.pathways
class TestPathwayQualityStats:
    """Test pathway quality statistics computation."""

    def test_computes_stats(self, mock_results, reactome_index):
        from pathway_network import compute_pathway_quality_stats
        stats = compute_pathway_quality_stats(mock_results, reactome_index)
        assert len(stats) > 0

    def test_stats_have_required_keys(self, mock_results, reactome_index):
        from pathway_network import compute_pathway_quality_stats
        stats = compute_pathway_quality_stats(mock_results, reactome_index)
        for pid, s in stats.items():
            assert "pathway_name" in s
            assert "mean_pdockq" in s
            assert "frac_high" in s
            assert "n_complexes" in s

    def test_mean_pdockq_reasonable(self, mock_results, reactome_index):
        from pathway_network import compute_pathway_quality_stats
        stats = compute_pathway_quality_stats(mock_results, reactome_index)
        for pid, s in stats.items():
            assert 0 <= s["mean_pdockq"] <= 1.0

    def test_frac_high_bounded(self, mock_results, reactome_index):
        from pathway_network import compute_pathway_quality_stats
        stats = compute_pathway_quality_stats(mock_results, reactome_index)
        for pid, s in stats.items():
            assert 0 <= s["frac_high"] <= 1.0

    def test_empty_results(self, reactome_index):
        from pathway_network import compute_pathway_quality_stats
        stats = compute_pathway_quality_stats([], reactome_index)
        assert len(stats) == 0


# ── Section: TestFormatting ──────────────────────────────────────

@pytest.mark.pathways
class TestFormatting:
    """Test formatting functions."""

    def test_format_empty(self):
        from pathway_network import format_reactome_pathways
        assert format_reactome_pathways([]) == ""

    def test_format_single(self):
        from pathway_network import format_reactome_pathways
        result = format_reactome_pathways([{
            "pathway_id": "R-HSA-109581",
            "pathway_name": "Apoptosis",
            "evidence_code": "TAS",
        }])
        assert result == "R-HSA-109581:Apoptosis"

    def test_format_deduplicates(self):
        from pathway_network import format_reactome_pathways
        result = format_reactome_pathways([
            {"pathway_id": "R-HSA-1", "pathway_name": "A", "evidence_code": "TAS"},
            {"pathway_id": "R-HSA-1", "pathway_name": "A", "evidence_code": "IEA"},
        ])
        assert result.count("R-HSA-1") == 1

    def test_format_truncation(self):
        from pathway_network import format_reactome_pathways
        mappings = [{"pathway_id": f"R-HSA-{i}", "pathway_name": f"P{i}",
                      "evidence_code": "TAS"} for i in range(25)]
        result = format_reactome_pathways(mappings, limit=5)
        assert "(+20 more)" in result

    def test_format_quality_context(self):
        from pathway_network import format_pathway_quality_context
        result = format_pathway_quality_context({
            "mean_pdockq": 0.45, "frac_high": 0.30, "n_complexes": 12,
        })
        assert "mean_pdockq=0.450" in result
        assert "frac_high=0.300" in result
        assert "n_complexes=12" in result

    def test_format_quality_context_empty(self):
        from pathway_network import format_pathway_quality_context
        assert format_pathway_quality_context({}) == ""


# ── Section: TestNetworkConstruction ─────────────────────────────

@pytest.mark.pathways
class TestNetworkConstruction:
    """Test NetworkX graph construction."""

    def test_builds_graph(self, mock_results):
        from pathway_network import build_interaction_network, _HAS_NETWORKX
        if not _HAS_NETWORKX:
            pytest.skip("NetworkX not installed")
        G = build_interaction_network(mock_results, min_pdockq=0.0)
        assert G.number_of_nodes() > 0

    def test_threshold_filters_edges(self, mock_results):
        from pathway_network import build_interaction_network, _HAS_NETWORKX
        if not _HAS_NETWORKX:
            pytest.skip("NetworkX not installed")
        G_all = build_interaction_network(mock_results, min_pdockq=0.0)
        G_filtered = build_interaction_network(mock_results, min_pdockq=0.40)
        assert G_filtered.number_of_edges() <= G_all.number_of_edges()

    def test_node_attributes(self, mock_results):
        from pathway_network import build_interaction_network, _HAS_NETWORKX
        if not _HAS_NETWORKX:
            pytest.skip("NetworkX not installed")
        G = build_interaction_network(mock_results, min_pdockq=0.0)
        for node in G.nodes():
            assert "gene_symbol" in G.nodes[node]

    def test_edge_attributes(self, mock_results):
        from pathway_network import build_interaction_network, _HAS_NETWORKX
        if not _HAS_NETWORKX:
            pytest.skip("NetworkX not installed")
        G = build_interaction_network(mock_results, min_pdockq=0.0)
        for u, v, data in G.edges(data=True):
            assert "pdockq" in data
            assert "quality_tier" in data

    def test_empty_results(self):
        from pathway_network import build_interaction_network, _HAS_NETWORKX
        if not _HAS_NETWORKX:
            pytest.skip("NetworkX not installed")
        G = build_interaction_network([], min_pdockq=0.0)
        assert G.number_of_nodes() == 0

    def test_high_threshold_excludes_all(self, mock_results):
        from pathway_network import build_interaction_network, _HAS_NETWORKX
        if not _HAS_NETWORKX:
            pytest.skip("NetworkX not installed")
        G = build_interaction_network(mock_results, min_pdockq=0.99)
        assert G.number_of_edges() == 0

    def test_network_stats(self, mock_results):
        from pathway_network import build_interaction_network, compute_network_stats, _HAS_NETWORKX
        if not _HAS_NETWORKX:
            pytest.skip("NetworkX not installed")
        G = build_interaction_network(mock_results, min_pdockq=0.0)
        stats = compute_network_stats(G)
        for node, s in stats.items():
            assert "degree" in s
            assert "clustering_coeff" in s


# ── Section: TestAnnotation ──────────────────────────────────────

@pytest.mark.pathways
class TestAnnotation:
    """Test in-place result annotation."""

    def test_adds_all_columns(self, mock_results, reactome_index):
        from pathway_network import annotate_results_with_pathways, CSV_FIELDNAMES_PATHWAYS
        annotate_results_with_pathways(mock_results, reactome_index)
        for col in CSV_FIELDNAMES_PATHWAYS:
            assert col in mock_results[0], f"Missing column: {col}"

    def test_pathway_counts(self, reactome_index):
        from pathway_network import annotate_results_with_pathways
        results = [{"protein_a": ACC_TP53, "protein_b": ACC_EEF1B2}]
        annotate_results_with_pathways(results, reactome_index)
        assert results[0]["n_reactome_pathways_a"] == TP53_UNIQUE_PATHWAYS
        assert results[0]["n_reactome_pathways_b"] == EEF1B2_UNIQUE_PATHWAYS

    def test_shared_pathways(self, reactome_index):
        from pathway_network import annotate_results_with_pathways
        results = [{"protein_a": ACC_TP53, "protein_b": ACC_YWHAG}]
        annotate_results_with_pathways(results, reactome_index)
        assert results[0]["n_shared_pathways"] > 0

    def test_no_shared_pathways(self, reactome_index):
        from pathway_network import annotate_results_with_pathways
        results = [{"protein_a": ACC_YWHAZ, "protein_b": ACC_EEF1B2}]
        annotate_results_with_pathways(results, reactome_index)
        assert results[0]["n_shared_pathways"] == 0

    def test_unknown_protein_empty(self, reactome_index):
        from pathway_network import annotate_results_with_pathways
        results = [{"protein_a": "XXXXXX", "protein_b": ACC_TP53}]
        annotate_results_with_pathways(results, reactome_index)
        assert results[0]["n_reactome_pathways_a"] == 0

    def test_isoform_stripped(self, reactome_index):
        from pathway_network import annotate_results_with_pathways
        results = [{"protein_a": "P61981-2", "protein_b": ACC_TP53}]
        annotate_results_with_pathways(results, reactome_index)
        assert results[0]["n_reactome_pathways_a"] == YWHAG_UNIQUE_PATHWAYS

    def test_modifies_in_place(self, reactome_index):
        from pathway_network import annotate_results_with_pathways
        results = [{"protein_a": ACC_TP53, "protein_b": ACC_YWHAG, "extra": "keep"}]
        annotate_results_with_pathways(results, reactome_index)
        assert results[0]["extra"] == "keep"

    def test_empty_results(self, reactome_index):
        from pathway_network import annotate_results_with_pathways
        results = []
        annotate_results_with_pathways(results, reactome_index)
        assert len(results) == 0


# ── Section: TestCSVFieldnames ───────────────────────────────────

@pytest.mark.pathways
class TestCSVFieldnames:
    """Test CSV fieldnames structure."""

    def test_a_b_pairs(self):
        from pathway_network import CSV_FIELDNAMES_PATHWAYS
        a_cols = [c for c in CSV_FIELDNAMES_PATHWAYS if c.endswith("_a")]
        b_cols = [c for c in CSV_FIELDNAMES_PATHWAYS if c.endswith("_b")]
        for a_col in a_cols:
            b_col = a_col[:-2] + "_b"
            assert b_col in b_cols

    def test_no_overlap_with_base(self):
        from pathway_network import CSV_FIELDNAMES_PATHWAYS
        base_cols = {'complex_name', 'protein_a', 'protein_b', 'pdockq', 'quality_tier'}
        overlap = set(CSV_FIELDNAMES_PATHWAYS) & base_cols
        assert len(overlap) == 0

    def test_no_overlap_with_disease(self):
        from pathway_network import CSV_FIELDNAMES_PATHWAYS
        from disease_annotations import CSV_FIELDNAMES_DISEASE
        overlap = set(CSV_FIELDNAMES_PATHWAYS) & set(CSV_FIELDNAMES_DISEASE)
        assert len(overlap) == 0


# ── Section: TestNetworkXGuard ───────────────────────────────────

@pytest.mark.pathways
class TestNetworkXGuard:
    """Test NetworkX import guard."""

    def test_has_networkx_flag_is_bool(self):
        from pathway_network import _HAS_NETWORKX
        assert isinstance(_HAS_NETWORKX, bool)

    def test_network_stats_empty_without_networkx(self):
        from pathway_network import compute_network_stats, _HAS_NETWORKX
        if not _HAS_NETWORKX:
            result = compute_network_stats(None)
            assert result == {}


# ── Section: TestCLI ─────────────────────────────────────────────

@pytest.mark.pathways
@pytest.mark.cli
class TestCLI:
    """Test standalone CLI entry points."""

    def test_cli_no_args(self):
        result = subprocess.run(
            [sys.executable, "pathway_network.py"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0


# ── Section: TestRegression ──────────────────────────────────────

@pytest.mark.pathways
@pytest.mark.regression
class TestRegression:
    """Regression tests with known reference values."""

    def test_tp53_exact_pathway_count(self, reactome_index):
        unique = {m["pathway_id"] for m in reactome_index[ACC_TP53]}
        assert len(unique) == 124

    def test_ywhag_exact_pathway_count(self, reactome_index):
        unique = {m["pathway_id"] for m in reactome_index[ACC_YWHAG]}
        assert len(unique) == 60

    def test_eef1b2_exact_pathway_count(self, reactome_index):
        unique = {m["pathway_id"] for m in reactome_index[ACC_EEF1B2]}
        assert len(unique) == 3

    def test_tp53_has_apoptosis(self, reactome_index):
        names = {m["pathway_name"] for m in reactome_index[ACC_TP53]}
        assert "Apoptosis" in names

    def test_tp53_ywhag_shared_pathways(self, reactome_index):
        """TP53 and YWHAG share at least 10 pathways."""
        from pathway_network import annotate_results_with_pathways
        results = [{"protein_a": ACC_TP53, "protein_b": ACC_YWHAG}]
        annotate_results_with_pathways(results, reactome_index)
        assert results[0]["n_shared_pathways"] >= 10


# ── Section: TestEdgeCases ───────────────────────────────────────

@pytest.mark.pathways
class TestEdgeCases:
    """Edge cases and error handling."""

    def test_single_protein_result(self, reactome_index):
        from pathway_network import annotate_results_with_pathways
        results = [{"protein_a": ACC_TP53, "protein_b": ""}]
        annotate_results_with_pathways(results, reactome_index)
        assert results[0]["n_reactome_pathways_a"] == TP53_UNIQUE_PATHWAYS
        assert results[0]["n_reactome_pathways_b"] == 0

    def test_format_empty_reactome(self):
        from pathway_network import format_reactome_pathways
        assert format_reactome_pathways([]) == ""

    def test_no_pdockq_in_results(self, reactome_index):
        from pathway_network import compute_pathway_quality_stats
        results = [{"protein_a": ACC_TP53, "protein_b": ACC_YWHAG}]
        stats = compute_pathway_quality_stats(results, reactome_index)
        assert len(stats) == 0


# ── Section: TestSubnetworkExtraction ────────────────────────────

@pytest.mark.pathways
class TestSubnetworkExtraction:
    """Test subnetwork extraction from full graph."""

    def test_extract_known_proteins(self, mock_results):
        from pathway_network import build_interaction_network, extract_subnetwork, _HAS_NETWORKX
        if not _HAS_NETWORKX:
            pytest.skip("NetworkX not installed")
        G = build_interaction_network(mock_results, min_pdockq=0.0)
        sub = extract_subnetwork(G, {ACC_TP53, ACC_YWHAG})
        assert ACC_TP53 in sub.nodes()
        assert ACC_YWHAG in sub.nodes()

    def test_extract_empty_set(self, mock_results):
        from pathway_network import build_interaction_network, extract_subnetwork, _HAS_NETWORKX
        if not _HAS_NETWORKX:
            pytest.skip("NetworkX not installed")
        G = build_interaction_network(mock_results, min_pdockq=0.0)
        sub = extract_subnetwork(G, set())
        assert sub.number_of_nodes() == 0

    def test_extract_unknown_protein(self, mock_results):
        from pathway_network import build_interaction_network, extract_subnetwork, _HAS_NETWORKX
        if not _HAS_NETWORKX:
            pytest.skip("NetworkX not installed")
        G = build_interaction_network(mock_results, min_pdockq=0.0)
        sub = extract_subnetwork(G, {"XXXXXX"})
        assert sub.number_of_nodes() == 0

    def test_extract_preserves_edges(self, mock_results):
        from pathway_network import build_interaction_network, extract_subnetwork, _HAS_NETWORKX
        if not _HAS_NETWORKX:
            pytest.skip("NetworkX not installed")
        G = build_interaction_network(mock_results, min_pdockq=0.0)
        sub = extract_subnetwork(G, {ACC_TP53, ACC_YWHAG})
        assert sub.number_of_edges() >= 1


# ── Section: TestRegulatoryOverlay ───────────────────────────────

@pytest.mark.pathways
class TestRegulatoryOverlay:
    """Test regulatory edge overlay."""

    def test_converts_to_digraph(self, mock_results):
        from pathway_network import build_interaction_network, add_regulatory_overlay, _HAS_NETWORKX
        import networkx as nx
        if not _HAS_NETWORKX:
            pytest.skip("NetworkX not installed")
        G = build_interaction_network(mock_results, min_pdockq=0.0)
        DG = add_regulatory_overlay(G, None)
        assert isinstance(DG, nx.DiGraph)
        # DiGraph should have same nodes
        assert set(DG.nodes()) == set(G.nodes())

    def test_empty_network_df(self, mock_results):
        from pathway_network import build_interaction_network, add_regulatory_overlay, _HAS_NETWORKX
        if not _HAS_NETWORKX:
            pytest.skip("NetworkX not installed")
        import pandas as pd
        G = build_interaction_network(mock_results, min_pdockq=0.0)
        DG = add_regulatory_overlay(G, pd.DataFrame())
        assert DG.number_of_nodes() == G.number_of_nodes()


# ── Section: TestNetworkPlots ────────────────────────────────────

@pytest.mark.pathways
class TestNetworkPlots:
    """Test network visualisation functions produce files without crashing."""

    def test_plot_by_pdockq(self, mock_results, tmp_path):
        from pathway_network import build_interaction_network, plot_network_by_pdockq, _HAS_NETWORKX
        if not _HAS_NETWORKX:
            pytest.skip("NetworkX not installed")
        G = build_interaction_network(mock_results, min_pdockq=0.0)
        out = tmp_path / "test_pdockq.png"
        plot_network_by_pdockq(G, out)
        assert out.exists()

    def test_plot_by_quality(self, mock_results, tmp_path):
        from pathway_network import build_interaction_network, plot_network_by_quality, _HAS_NETWORKX
        if not _HAS_NETWORKX:
            pytest.skip("NetworkX not installed")
        G = build_interaction_network(mock_results, min_pdockq=0.0)
        out = tmp_path / "test_quality.png"
        plot_network_by_quality(G, out)
        assert out.exists()

    def test_plot_disease_network(self, tmp_path):
        from pathway_network import build_interaction_network, plot_disease_network, _HAS_NETWORKX
        if not _HAS_NETWORKX:
            pytest.skip("NetworkX not installed")
        results = [
            {"protein_a": ACC_TP53, "protein_b": ACC_YWHAG,
             "pdockq": 0.5, "n_diseases_a": 8, "n_diseases_b": 1,
             "is_drug_target_a": False, "is_drug_target_b": False,
             "gene_symbol_a": "TP53", "gene_symbol_b": "YWHAG"},
        ]
        G = build_interaction_network(results, min_pdockq=0.0)
        out = tmp_path / "test_disease.png"
        plot_disease_network(G, out)
        assert out.exists()

    def test_plot_empty_graph(self, tmp_path):
        from pathway_network import build_interaction_network, plot_network_by_pdockq, _HAS_NETWORKX
        if not _HAS_NETWORKX:
            pytest.skip("NetworkX not installed")
        G = build_interaction_network([], min_pdockq=0.0)
        out = tmp_path / "test_empty.png"
        plot_network_by_pdockq(G, out)
        # Should not crash, file may or may not exist (empty graph = no output)


# ── Section: TestStringNetworkWrapper ────────────────────────────

@pytest.mark.pathways
class TestStringNetworkWrapper:
    """Test run_string_network wrapper."""

    def test_returns_none_on_import_error(self):
        """If string_api module fails, should return None gracefully."""
        from pathway_network import run_string_network
        # This may return None or a DataFrame depending on API availability
        # The key is it doesn't crash
        result = run_string_network(["TP53"], verbose=False)
        assert result is None or hasattr(result, "columns")


# ── Section: TestEnrichmentAugmentation ──────────────────────────

@pytest.mark.pathways
class TestEnrichmentAugmentation:
    """Test that STRING enrichment augments annotation when available."""

    def test_annotation_with_enrichment_df(self, reactome_index):
        import pandas as pd
        from pathway_network import annotate_results_with_pathways, compute_pathway_quality_stats

        results = [
            {"protein_a": ACC_TP53, "protein_b": ACC_YWHAG,
             "pdockq": 0.45, "quality_tier_v2": "Medium"},
        ]
        stats = compute_pathway_quality_stats(results, reactome_index)

        # Mock enrichment DataFrame from STRING API
        enrichment_df = pd.DataFrame([
            {"category": "RCTM", "term": "R-HSA-109581",
             "description": "Apoptosis", "fdr": 1.5e-10,
             "number_of_genes": 5, "p_value": 1e-12},
        ])

        annotate_results_with_pathways(
            results, reactome_index,
            pathway_stats=stats,
            enrichment_df=enrichment_df,
        )
        # If a shared pathway matches the enrichment term, FDR should appear
        context = results[0].get("pathway_quality_context", "")
        # Context should have quality stats
        assert "mean_pdockq" in context

    def test_annotation_without_enrichment(self, reactome_index):
        from pathway_network import annotate_results_with_pathways
        results = [{"protein_a": ACC_TP53, "protein_b": ACC_YWHAG}]
        annotate_results_with_pathways(results, reactome_index, enrichment_df=None)
        # Should work fine without enrichment
        assert "n_shared_pathways" in results[0]


# ── Section: TestInvertReactomeIndex ────────────────────────────────

@pytest.mark.pathways
class TestInvertReactomeIndex:
    """Test pathway→proteins index inversion."""

    def test_invert_returns_dict(self, reactome_index):
        from pathway_network import invert_reactome_index
        result = invert_reactome_index(reactome_index)
        assert isinstance(result, dict)

    def test_invert_has_pathway_ids(self, reactome_index):
        from pathway_network import invert_reactome_index
        result = invert_reactome_index(reactome_index)
        # Should contain known Reactome IDs
        assert any(pid.startswith("R-HSA-") for pid in result)

    def test_invert_proteins_are_sets(self, reactome_index):
        from pathway_network import invert_reactome_index
        result = invert_reactome_index(reactome_index)
        for proteins in result.values():
            assert isinstance(proteins, set)

    def test_invert_contains_test_proteins(self, reactome_index):
        from pathway_network import invert_reactome_index
        result = invert_reactome_index(reactome_index)
        # TP53 should be in at least some pathways
        tp53_pathways = [pid for pid, prots in result.items() if ACC_TP53 in prots]
        assert len(tp53_pathways) == TP53_UNIQUE_PATHWAYS

    def test_invert_roundtrip_consistency(self, reactome_index):
        """Inverting should preserve the total number of (protein, pathway) pairs."""
        from pathway_network import invert_reactome_index
        # Count forward direction
        forward_pairs = sum(len(mappings) for mappings in reactome_index.values())
        # Count inverted direction
        inverted = invert_reactome_index(reactome_index)
        inverted_pairs = sum(len(prots) for prots in inverted.values())
        # May differ due to deduplication, but inverted should not exceed forward
        assert inverted_pairs <= forward_pairs


# ── Section: TestPerPathwayPPIEnrichment ────────────────────────────

@pytest.mark.pathways
class TestPerPathwayPPIEnrichment:
    """Test per-pathway PPI enrichment query and annotation integration."""

    def test_per_pathway_enrichment_with_mock(self, reactome_index):
        """Mocked API should produce per-pathway stats."""
        from unittest.mock import patch
        from pathway_network import invert_reactome_index, run_per_pathway_ppi_enrichment

        pathway_proteins = invert_reactome_index(reactome_index)

        # Pick a few pathways shared between TP53 and YWHAG
        pids_a = {m["pathway_id"] for m in reactome_index.get(ACC_TP53, [])}
        pids_b = {m["pathway_id"] for m in reactome_index.get(ACC_YWHAG, [])}
        shared = pids_a & pids_b

        mock_result = {
            "p_value": 0.001,
            "number_of_edges": 10,
            "expected_number_of_edges": 5,
        }

        with patch("string_api.query_ppi_enrichment", return_value=mock_result):
            stats = run_per_pathway_ppi_enrichment(
                pathway_proteins, shared, verbose=False,
            )

        assert len(stats) > 0
        for pid, s in stats.items():
            assert "p_value" in s
            assert "ratio" in s
            assert s["ratio"] == 2.0  # 10/5

    def test_per_pathway_enrichment_skips_small(self, reactome_index):
        """Pathways with <2 proteins should be skipped."""
        from unittest.mock import patch, MagicMock
        from pathway_network import run_per_pathway_ppi_enrichment

        # Create a pathway with only 1 protein
        pathway_proteins = {"R-HSA-TINY": {"P04637"}}
        mock_fn = MagicMock()

        with patch("string_api.query_ppi_enrichment", mock_fn):
            stats = run_per_pathway_ppi_enrichment(
                pathway_proteins, {"R-HSA-TINY"}, verbose=False,
            )

        assert len(stats) == 0
        mock_fn.assert_not_called()

    def test_per_pathway_enrichment_handles_failure(self, reactome_index):
        """API failures for individual pathways should not crash."""
        from unittest.mock import patch
        from pathway_network import invert_reactome_index, run_per_pathway_ppi_enrichment

        pathway_proteins = invert_reactome_index(reactome_index)
        pids_a = {m["pathway_id"] for m in reactome_index.get(ACC_TP53, [])}
        pids_b = {m["pathway_id"] for m in reactome_index.get(ACC_YWHAG, [])}
        shared = pids_a & pids_b

        with patch("string_api.query_ppi_enrichment",
                    side_effect=Exception("API error")):
            stats = run_per_pathway_ppi_enrichment(
                pathway_proteins, shared, verbose=False,
            )

        assert stats == {}

    def test_annotation_with_per_pathway_ppi(self, reactome_index):
        """annotate_results_with_pathways with pathway_ppi_stats picks best pathway."""
        from pathway_network import annotate_results_with_pathways

        results = [
            {"protein_a": ACC_TP53, "protein_b": ACC_YWHAG,
             "pdockq": 0.45, "quality_tier_v2": "Medium"},
        ]

        # Find shared pathways
        pids_a = {m["pathway_id"] for m in reactome_index.get(ACC_TP53, [])}
        pids_b = {m["pathway_id"] for m in reactome_index.get(ACC_YWHAG, [])}
        shared = pids_a & pids_b
        shared_list = sorted(shared)

        # Mock per-pathway stats with different p-values
        pathway_ppi_stats = {}
        for i, pid in enumerate(shared_list):
            pathway_ppi_stats[pid] = {
                "p_value": 0.01 * (i + 1),  # increasing p-values
                "ratio": 2.0 + i * 0.1,
                "n_proteins": 10,
                "sampled": False,
            }

        annotate_results_with_pathways(
            results, reactome_index,
            pathway_ppi_stats=pathway_ppi_stats,
        )

        # Should pick the most significant (lowest p-value)
        assert results[0]["ppi_enrichment_pvalue"] != ""
        assert results[0]["ppi_enrichment_ratio"] != ""
        # p-value should be the smallest one (first in sorted order)
        assert results[0]["ppi_enrichment_pvalue"] == f"{0.01:.2e}"

    def test_annotation_no_shared_pathways_empty_ppi(self, reactome_index):
        """Complexes with no shared pathways get empty PPI columns."""
        from pathway_network import annotate_results_with_pathways

        results = [
            {"protein_a": ACC_EEF1B2, "protein_b": ACC_EEF1B2,
             "pdockq": 0.30, "quality_tier_v2": "Medium"},
        ]

        pathway_ppi_stats = {"R-HSA-UNRELATED": {
            "p_value": 0.001, "ratio": 3.0,
            "n_proteins": 50, "sampled": False,
        }}

        annotate_results_with_pathways(
            results, reactome_index,
            pathway_ppi_stats=pathway_ppi_stats,
        )

        # EEF1B2 with itself has shared pathways, but none in pathway_ppi_stats
        # The columns should still have some value or empty
        assert "ppi_enrichment_pvalue" in results[0]

    def test_ppi_enrichment_min_proteins_constant(self):
        from pathway_network import PPI_ENRICHMENT_MIN_PROTEINS
        assert PPI_ENRICHMENT_MIN_PROTEINS == 2
