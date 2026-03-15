"""
Tests for protein_clustering.py — STRING sequence cluster parsing,
UniProt index building, homologous pair detection, and CLI.

Test data: tests/offline_test_data/databases/test_string_clusters.txt
Contains 38 rows across 5 clusters:
    CL:6999  (5 members)  — contains ENSP00000269305 (P04637) and ENSP00000258149 (Q00987)
    CL:6997  (10 members) — contains ENSP00000269305 and ENSP00000258149
    CL:13942 (6 members)  — contains ENSP00000000233 (P26437)
    CL:13940 (11 members) — contains ENSP00000000233
    CL:28385 (6 members)  — contains ENSP00000002165 (Q7Z6V1)
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure the project root is importable
PROJECT_ROOT = Path(r"C:\Users\Talhah Zubayer\Documents\protein-complexes-toolkit")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from protein_clustering import (
    CSV_FIELDNAMES_CLUSTERING,
    DEFAULT_CLUSTERS_DIR,
    HOMOLOGOUS_PAIRS_DISPLAY_LIMIT,
    MAX_CLUSTER_SIZE_FOR_PAIRS,
    STRING_CLUSTERS_FILE,
    VALID_CLUSTERING_MODES,
    annotate_pair_clusters,
    annotate_results_with_clustering,
    build_cluster_index,
    build_cluster_to_uniprot,
    build_uniprot_cluster_index,
    find_homologous_pairs,
    find_shared_clusters,
    get_cluster_sizes,
    load_clusters,
    validate_clustering_mode,
)

pytestmark = pytest.mark.clustering

# ── Expected test data constants ─────────────────────────────────────

EXPECTED_ROWS = 38
EXPECTED_CLUSTERS = 5
EXPECTED_UNIQUE_PROTEINS = 27  # unique ENSP IDs across all clusters

# Test ENSP IDs present in BOTH test_aliases.txt and test_string_clusters.txt
# ENSP00000269305 → P04637 (TP53), in CL:6999 and CL:6997
# ENSP00000258149 → Q00987 (MDM2), in CL:6999 and CL:6997
# ENSP00000000233 → P26437 (ARF5), in CL:13942 and CL:13940
# ENSP00000002165 → Q7Z6V1 (FUCA2), in CL:28385

TEST_DB_DIR = PROJECT_ROOT / "tests" / "offline_test_data" / "databases"


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def clusters_df():
    """Load the test clusters excerpt."""
    return load_clusters(str(TEST_DB_DIR / "test_string_clusters.txt"))


@pytest.fixture(scope="module")
def cluster_indices(clusters_df):
    """Build bidirectional cluster index from test data."""
    return build_cluster_index(clusters_df)


@pytest.fixture(scope="module")
def cluster_to_proteins(cluster_indices):
    return cluster_indices[0]


@pytest.fixture(scope="module")
def protein_to_clusters(cluster_indices):
    return cluster_indices[1]


@pytest.fixture(scope="module")
def id_mapper_for_clustering():
    """Load IDMapper from test aliases file."""
    from id_mapper import IDMapper
    return IDMapper(str(TEST_DB_DIR / "test_aliases.txt"), api_fallback=False)


@pytest.fixture(scope="module")
def uniprot_index(protein_to_clusters, id_mapper_for_clustering):
    """Build UniProt cluster index."""
    return build_uniprot_cluster_index(
        protein_to_clusters, id_mapper_for_clustering,
    )


@pytest.fixture(scope="module")
def cluster_to_uniprot_map(uniprot_index):
    """Build cluster-to-UniProt index."""
    return build_cluster_to_uniprot(uniprot_index)


# ── TestConstants ────────────────────────────────────────────────────

class TestConstants:
    """Verify module-level constants."""

    def test_default_clusters_dir_is_path(self):
        assert isinstance(DEFAULT_CLUSTERS_DIR, Path)

    def test_valid_clustering_modes(self):
        assert 'string' in VALID_CLUSTERING_MODES
        assert 'foldseek' in VALID_CLUSTERING_MODES
        assert 'hybrid' in VALID_CLUSTERING_MODES

    def test_csv_fieldnames_clustering(self):
        assert 'sequence_cluster_ids' in CSV_FIELDNAMES_CLUSTERING
        assert 'sequence_cluster_count' in CSV_FIELDNAMES_CLUSTERING
        assert 'shared_cluster_ids' in CSV_FIELDNAMES_CLUSTERING
        assert 'shared_cluster_count' in CSV_FIELDNAMES_CLUSTERING
        assert 'homologous_pairs' in CSV_FIELDNAMES_CLUSTERING
        assert 'n_homologous_pairs' in CSV_FIELDNAMES_CLUSTERING
        assert 'homology_bitscore' in CSV_FIELDNAMES_CLUSTERING

    def test_clusters_file_constant(self):
        assert STRING_CLUSTERS_FILE == "9606.clusters.proteins.v12.0.txt"

    def test_max_cluster_size_for_pairs(self):
        assert isinstance(MAX_CLUSTER_SIZE_FOR_PAIRS, int)
        assert MAX_CLUSTER_SIZE_FOR_PAIRS > 0


# ── TestLoadClusters ─────────────────────────────────────────────────

class TestLoadClusters:
    """Tests for load_clusters()."""

    def test_returns_dataframe(self, clusters_df):
        assert isinstance(clusters_df, pd.DataFrame)

    def test_strips_taxonomy_prefix(self, clusters_df):
        assert not clusters_df['protein_id'].str.startswith('9606.').any()

    def test_correct_columns(self, clusters_df):
        assert list(clusters_df.columns) == ['cluster_id', 'protein_id']

    def test_row_count(self, clusters_df):
        assert len(clusters_df) == EXPECTED_ROWS

    def test_cluster_count(self, clusters_df):
        assert clusters_df['cluster_id'].nunique() == EXPECTED_CLUSTERS

    def test_known_protein_present(self, clusters_df):
        assert 'ENSP00000269305' in clusters_df['protein_id'].values

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_clusters("/nonexistent/path/clusters.txt")


# ── TestBuildClusterIndex ────────────────────────────────────────────

class TestBuildClusterIndex:
    """Tests for build_cluster_index()."""

    def test_bidirectional_consistency(self, cluster_to_proteins, protein_to_clusters):
        """Every protein in c2p appears in p2c and vice versa."""
        for cid, proteins in cluster_to_proteins.items():
            for pid in proteins:
                assert cid in protein_to_clusters[pid]
        for pid, clusters in protein_to_clusters.items():
            for cid in clusters:
                assert pid in cluster_to_proteins[cid]

    def test_many_to_many(self, protein_to_clusters):
        """ENSP00000269305 is in multiple clusters (CL:6999 and CL:6997)."""
        clusters = protein_to_clusters.get('ENSP00000269305', set())
        assert len(clusters) >= 2
        assert 'CL:6999' in clusters
        assert 'CL:6997' in clusters

    def test_cluster_members_correct(self, cluster_to_proteins):
        """CL:6999 has exactly 5 members."""
        members = cluster_to_proteins.get('CL:6999', set())
        assert len(members) == 5
        assert 'ENSP00000269305' in members
        assert 'ENSP00000258149' in members

    def test_empty_dataframe(self):
        empty_df = pd.DataFrame(columns=['cluster_id', 'protein_id'])
        c2p, p2c = build_cluster_index(empty_df)
        assert len(c2p) == 0
        assert len(p2c) == 0


# ── TestGetClusterSizes ──────────────────────────────────────────────

class TestGetClusterSizes:
    """Tests for get_cluster_sizes()."""

    def test_sizes_match_members(self, cluster_to_proteins):
        sizes = get_cluster_sizes(cluster_to_proteins)
        for cid, size in sizes.items():
            assert size == len(cluster_to_proteins[cid])

    def test_known_cluster_size(self, cluster_to_proteins):
        sizes = get_cluster_sizes(cluster_to_proteins)
        assert sizes['CL:6999'] == 5
        assert sizes['CL:6997'] == 10
        assert sizes['CL:13942'] == 6
        assert sizes['CL:13940'] == 11
        assert sizes['CL:28385'] == 6


# ── TestBuildUniprotClusterIndex ─────────────────────────────────────

class TestBuildUniprotClusterIndex:
    """Tests for ENSP -> UniProt resolution of cluster index."""

    def test_known_ensp_resolves(self, uniprot_index):
        """P04637 (from ENSP00000269305) should be in the index."""
        assert 'P04637' in uniprot_index

    def test_unresolvable_ensp_skipped(self, uniprot_index):
        """ENSP IDs not in test_aliases should not appear as keys."""
        # ENSP00000324404 is in CL:6999 but not in test_aliases.txt
        assert 'ENSP00000324404' not in uniprot_index

    def test_inherits_all_clusters(self, uniprot_index, protein_to_clusters):
        """P04637 should have at least the same clusters as ENSP00000269305."""
        ensp_clusters = protein_to_clusters.get('ENSP00000269305', set())
        uniprot_clusters = uniprot_index.get('P04637', set())
        # UniProt may have MORE clusters (from other ENSP mappings),
        # but should have at least all from this ENSP
        assert ensp_clusters.issubset(uniprot_clusters)

    def test_multiple_uniprot_per_ensp(self, uniprot_index):
        """Multiple UniProt accessions for ENSP00000000233 should all be indexed."""
        # ENSP00000000233 maps to P26437, P84085, A4D0Z3, C9J1Z8
        assert 'P26437' in uniprot_index
        # Both should have the same cluster set
        p26437_clusters = uniprot_index.get('P26437', set())
        assert 'CL:13942' in p26437_clusters
        assert 'CL:13940' in p26437_clusters


# ── TestBuildClusterToUniprot ────────────────────────────────────────

class TestBuildClusterToUniprot:
    """Tests for build_cluster_to_uniprot()."""

    def test_known_cluster_contains_uniprot(self, cluster_to_uniprot_map):
        """CL:6999 should contain P04637 and Q00987."""
        members = cluster_to_uniprot_map.get('CL:6999', set())
        assert 'P04637' in members
        assert 'Q00987' in members

    def test_all_clusters_present(self, cluster_to_uniprot_map, uniprot_index):
        """Every cluster in uniprot_index should appear in cluster_to_uniprot."""
        all_clusters = set()
        for clusters in uniprot_index.values():
            all_clusters.update(clusters)
        for cid in all_clusters:
            assert cid in cluster_to_uniprot_map


# ── TestFindSharedClusters ───────────────────────────────────────────

class TestFindSharedClusters:
    """Tests for cluster intersection between two proteins."""

    def test_shared_cluster_found(self, uniprot_index):
        """P04637 and Q00987 share CL:6999 and CL:6997."""
        shared = find_shared_clusters('P04637', 'Q00987', uniprot_index)
        assert 'CL:6999' in shared
        assert 'CL:6997' in shared
        assert len(shared) >= 2

    def test_no_shared_clusters(self, uniprot_index):
        """P04637 (TP53) and P26437 (ARF5) should not share small test clusters."""
        # They are in completely different cluster families in our test excerpt
        shared = find_shared_clusters('P04637', 'P26437', uniprot_index)
        # In the test excerpt they have no overlap (different cluster families)
        assert len(shared) == 0

    def test_unknown_protein_returns_empty(self, uniprot_index):
        shared = find_shared_clusters('P04637', 'XXXXXX', uniprot_index)
        assert len(shared) == 0

    def test_both_unknown_returns_empty(self, uniprot_index):
        shared = find_shared_clusters('XXXXXX', 'YYYYYY', uniprot_index)
        assert len(shared) == 0


# ── TestFindHomologousPairs ──────────────────────────────────────────

class TestFindHomologousPairs:
    """Tests for homologous pair detection."""

    def test_finds_pairs_in_shared_cluster(self, uniprot_index, cluster_to_uniprot_map):
        """P04637 and Q00987 share clusters, so homologous pairs should be found."""
        homologs = find_homologous_pairs(
            'P04637', 'Q00987', uniprot_index, cluster_to_uniprot_map,
        )
        assert len(homologs) > 0

    def test_excludes_query_pair(self, uniprot_index, cluster_to_uniprot_map):
        """The query pair itself should not appear in results."""
        homologs = find_homologous_pairs(
            'P04637', 'Q00987', uniprot_index, cluster_to_uniprot_map,
        )
        query = (min('P04637', 'Q00987'), max('P04637', 'Q00987'))
        assert query not in homologs

    def test_excludes_self_pairs(self, uniprot_index, cluster_to_uniprot_map):
        """No (X, X) pairs in results."""
        homologs = find_homologous_pairs(
            'P04637', 'Q00987', uniprot_index, cluster_to_uniprot_map,
        )
        for a, b in homologs:
            assert a != b

    def test_normalised_output(self, uniprot_index, cluster_to_uniprot_map):
        """All pairs should be (min, max) ordered."""
        homologs = find_homologous_pairs(
            'P04637', 'Q00987', uniprot_index, cluster_to_uniprot_map,
        )
        for a, b in homologs:
            assert a <= b

    def test_known_pairs_filter(self, uniprot_index, cluster_to_uniprot_map):
        """With known_pairs filter, only return matching pairs."""
        # Get all homologs first
        all_homologs = find_homologous_pairs(
            'P04637', 'Q00987', uniprot_index, cluster_to_uniprot_map,
        )
        assert len(all_homologs) > 0

        # Filter to empty set — should return nothing
        filtered = find_homologous_pairs(
            'P04637', 'Q00987', uniprot_index, cluster_to_uniprot_map,
            known_pairs=set(),
        )
        assert len(filtered) == 0

        # Filter to set containing one of the homologs
        one_pair = set([all_homologs[0]])
        filtered = find_homologous_pairs(
            'P04637', 'Q00987', uniprot_index, cluster_to_uniprot_map,
            known_pairs=one_pair,
        )
        assert len(filtered) == 1
        assert filtered[0] == all_homologs[0]

    def test_no_shared_clusters_returns_empty(self, uniprot_index, cluster_to_uniprot_map):
        """Proteins with no shared clusters should return no homologs."""
        homologs = find_homologous_pairs(
            'P04637', 'P26437', uniprot_index, cluster_to_uniprot_map,
        )
        assert len(homologs) == 0

    def test_oversized_clusters_skipped(self, uniprot_index):
        """Clusters exceeding MAX_CLUSTER_SIZE_FOR_PAIRS should be skipped."""
        # Build a fake cluster_to_uniprot with one oversized cluster
        oversized_members = {f"FAKE{i:04d}" for i in range(MAX_CLUSTER_SIZE_FOR_PAIRS + 10)}
        oversized_members.add('P04637')
        oversized_members.add('Q00987')
        fake_c2u = {'CL:HUGE': oversized_members}
        # Even though P04637 and Q00987 share a cluster, the oversized one is skipped
        homologs = find_homologous_pairs(
            'P04637', 'Q00987', uniprot_index, fake_c2u,
        )
        assert len(homologs) == 0


# ── TestAnnotatePairClusters ─────────────────────────────────────────

class TestAnnotatePairClusters:
    """Tests for single-row annotation."""

    def test_both_proteins_have_clusters(self, uniprot_index):
        """Both P04637 and Q00987 have clusters — union should be populated."""
        row = {'protein_a': 'P04637', 'protein_b': 'Q00987'}
        annotate_pair_clusters(row, uniprot_index)
        assert row['sequence_cluster_count'] > 0
        assert row['sequence_cluster_ids'] != ''
        assert row['shared_cluster_count'] > 0
        assert row['shared_cluster_ids'] != ''

    def test_one_protein_missing(self, uniprot_index):
        """One known protein, one unknown — union should still have clusters."""
        row = {'protein_a': 'P04637', 'protein_b': 'XXXXXX'}
        annotate_pair_clusters(row, uniprot_index)
        assert row['sequence_cluster_count'] > 0
        assert row['shared_cluster_count'] == 0
        assert row['shared_cluster_ids'] == ''

    def test_both_missing(self, uniprot_index):
        """Both proteins unknown — empty annotation."""
        row = {'protein_a': 'XXXXXX', 'protein_b': 'YYYYYY'}
        annotate_pair_clusters(row, uniprot_index)
        assert row['sequence_cluster_count'] == 0
        assert row['sequence_cluster_ids'] == ''
        assert row['shared_cluster_count'] == 0
        assert row['shared_cluster_ids'] == ''

    def test_pipe_separation(self, uniprot_index):
        """Cluster IDs should be pipe-separated and count should match."""
        row = {'protein_a': 'P04637', 'protein_b': 'Q00987'}
        annotate_pair_clusters(row, uniprot_index)
        ids = row['sequence_cluster_ids'].split('|')
        assert len(ids) == row['sequence_cluster_count']

    def test_shared_is_subset_of_union(self, uniprot_index):
        """Shared clusters must be a subset of union clusters."""
        row = {'protein_a': 'P04637', 'protein_b': 'Q00987'}
        annotate_pair_clusters(row, uniprot_index)
        union_ids = set(row['sequence_cluster_ids'].split('|'))
        shared_ids = set(row['shared_cluster_ids'].split('|'))
        assert shared_ids.issubset(union_ids)


# ── TestAnnotateResultsWithClustering ────────────────────────────────

class TestAnnotateResultsWithClustering:
    """Integration test for full annotation pipeline."""

    def test_all_columns_present(self, uniprot_index, cluster_to_uniprot_map):
        results = [{'protein_a': 'P04637', 'protein_b': 'Q00987'}]
        annotate_results_with_clustering(
            results, uniprot_index, cluster_to_uniprot_map,
        )
        for col in CSV_FIELDNAMES_CLUSTERING:
            assert col in results[0], f"Missing column: {col}"

    def test_modifies_in_place(self, uniprot_index, cluster_to_uniprot_map):
        results = [{'protein_a': 'P04637', 'protein_b': 'Q00987'}]
        original_id = id(results)
        annotate_results_with_clustering(
            results, uniprot_index, cluster_to_uniprot_map,
        )
        assert id(results) == original_id
        assert 'sequence_cluster_ids' in results[0]

    def test_homologous_pairs_populated(self, uniprot_index, cluster_to_uniprot_map):
        results = [{'protein_a': 'P04637', 'protein_b': 'Q00987'}]
        annotate_results_with_clustering(
            results, uniprot_index, cluster_to_uniprot_map,
        )
        assert results[0]['n_homologous_pairs'] > 0
        assert results[0]['homologous_pairs'] != ''

    def test_no_homologs_for_disjoint_pair(self, uniprot_index, cluster_to_uniprot_map):
        results = [{'protein_a': 'P04637', 'protein_b': 'P26437'}]
        annotate_results_with_clustering(
            results, uniprot_index, cluster_to_uniprot_map,
        )
        assert results[0]['n_homologous_pairs'] == 0
        assert results[0]['homologous_pairs'] == ''


# ── TestValidateClusteringMode ───────────────────────────────────────

class TestValidateClusteringMode:
    """Tests for CLI mode validation."""

    def test_string_accepted(self):
        assert validate_clustering_mode('string') == 'string'

    def test_foldseek_raises(self):
        with pytest.raises(NotImplementedError, match="Foldseek"):
            validate_clustering_mode('foldseek')

    def test_hybrid_raises(self):
        with pytest.raises(NotImplementedError, match="(?i)hybrid"):
            validate_clustering_mode('hybrid')

    def test_error_message_mentions_ram(self):
        with pytest.raises(NotImplementedError, match="35 GB"):
            validate_clustering_mode('foldseek')

    def test_invalid_mode_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid clustering mode"):
            validate_clustering_mode('invalid_mode')


# ── TestClusteringCLI ────────────────────────────────────────────────

@pytest.mark.cli
class TestClusteringCLI:
    """CLI argument parsing and execution tests."""

    def test_summary_mode(self):
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "protein_clustering.py"),
             "--clusters-file", str(TEST_DB_DIR / "test_string_clusters.txt"),
             "--aliases", str(TEST_DB_DIR / "test_aliases.txt"),
             "--summary"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0
        assert "Total clusters:" in result.stdout

    def test_protein_lookup(self):
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "protein_clustering.py"),
             "--clusters-file", str(TEST_DB_DIR / "test_string_clusters.txt"),
             "--aliases", str(TEST_DB_DIR / "test_aliases.txt"),
             "--protein", "P04637"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0
        assert "P04637" in result.stdout
        assert "CL:6999" in result.stdout

    def test_pair_lookup(self):
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "protein_clustering.py"),
             "--clusters-file", str(TEST_DB_DIR / "test_string_clusters.txt"),
             "--aliases", str(TEST_DB_DIR / "test_aliases.txt"),
             "--pair", "P04637", "Q00987"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0
        assert "Shared clusters:" in result.stdout
        assert "Homologous pairs:" in result.stdout


# ── TestToolkitIntegration ───────────────────────────────────────────

class TestToolkitIntegration:
    """Tests for toolkit.py --clustering integration."""

    def test_clustering_requires_enrich(self):
        """--clustering without --enrich should fail."""
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "toolkit.py"),
             "--dir", str(PROJECT_ROOT / "Test_Data"),
             "--output", "dummy.csv",
             "--clustering"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0
        assert "--clustering requires --enrich" in result.stderr

    def test_get_csv_fieldnames_includes_clustering(self):
        """get_csv_fieldnames with include_clustering=True adds clustering columns."""
        from toolkit import get_csv_fieldnames
        fieldnames = get_csv_fieldnames(include_clustering=True)
        for col in CSV_FIELDNAMES_CLUSTERING:
            assert col in fieldnames

    def test_get_csv_fieldnames_excludes_clustering_by_default(self):
        """get_csv_fieldnames without clustering flag doesn't add clustering columns."""
        from toolkit import get_csv_fieldnames
        fieldnames = get_csv_fieldnames()
        for col in CSV_FIELDNAMES_CLUSTERING:
            assert col not in fieldnames
