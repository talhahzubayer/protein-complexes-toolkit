"""
Tests for the offline ProtVar scoring module (AlphaMissense + monomeric FoldX).

Tests cover:
- Constants and CSV column definitions
- AlphaMissense TSV loading and variant parsing
- AFDB FoldX export CSV loading
- Combined index building and merging
- Score lookup
- Chain variant scoring
- Detail string formatting
- Full annotation pipeline
- Standalone CLI
- Regression values for known proteins

Uses small test data files in tests/offline_test_data/databases/:
- test_afdb_foldx_export.csv (P61981 pos 1-2, P24534 pos 1)
- test_alphamissense.tsv (P61981 pos 1-2, P24534 pos 1)
"""

import argparse
from pathlib import Path
from statistics import mean

import pytest

from protvar_client import (
    DEFAULT_FOLDX_EXPORT,
    DEFAULT_AM_FILE,
    FOLDX_DESTABILISING_THRESHOLD,
    PROTVAR_DETAILS_DISPLAY_LIMIT,
    CHUNK_LOG_INTERVAL,
    CSV_FIELDNAMES_PROTVAR,
    _parse_am_variant,
    load_alphamissense_scores,
    load_foldx_export,
    build_protvar_index,
    lookup_score,
    _parse_variant_details_for_protvar,
    format_protvar_details,
    _score_chain_variants_protvar,
    annotate_results_with_protvar,
    build_argument_parser,
)


# ── Fixtures ─────────────────────────────────────────────────────────

TEST_DATA_DIR = Path(__file__).parent / "offline_test_data" / "databases"
TEST_FOLDX_CSV = TEST_DATA_DIR / "test_afdb_foldx_export.csv"
TEST_AM_TSV = TEST_DATA_DIR / "test_alphamissense.tsv"


@pytest.fixture
def small_index():
    """Build a small index from test data for P61981 + P24534."""
    return build_protvar_index(
        accessions={'P61981', 'P24534'},
        foldx_path=TEST_FOLDX_CSV,
        am_path=TEST_AM_TSV,
        verbose=False,
    )


# ── Test Class 1: Constants ──────────────────────────────────────────

@pytest.mark.protvar
class TestConstants:
    """Verify module constants."""

    def test_default_foldx_export_path(self):
        assert 'afdb_foldx_export' in str(DEFAULT_FOLDX_EXPORT)
        assert str(DEFAULT_FOLDX_EXPORT).endswith('.csv')

    def test_default_am_file_path(self):
        assert 'AlphaMissense' in str(DEFAULT_AM_FILE)
        assert str(DEFAULT_AM_FILE).endswith('.tsv')

    def test_foldx_threshold(self):
        assert FOLDX_DESTABILISING_THRESHOLD == 1.6

    def test_details_display_limit(self):
        assert PROTVAR_DETAILS_DISPLAY_LIMIT == 20

    def test_chunk_log_interval(self):
        assert CHUNK_LOG_INTERVAL == 5_000_000

    def test_csv_fieldnames_count(self):
        assert len(CSV_FIELDNAMES_PROTVAR) == 8


# ── Test Class 2: CSV Column Names ───────────────────────────────────

@pytest.mark.protvar
class TestCSVFieldnames:
    """Verify CSV column name definitions."""

    def test_column_names(self):
        expected = [
            'protvar_am_mean_a', 'protvar_am_mean_b',
            'protvar_foldx_mean_a', 'protvar_foldx_mean_b',
            'protvar_am_n_pathogenic_a', 'protvar_am_n_pathogenic_b',
            'protvar_details_a', 'protvar_details_b',
        ]
        assert CSV_FIELDNAMES_PROTVAR == expected

    def test_all_columns_have_chain_suffix(self):
        for col in CSV_FIELDNAMES_PROTVAR:
            assert col.endswith('_a') or col.endswith('_b'), col

    def test_paired_columns(self):
        a_cols = [c for c in CSV_FIELDNAMES_PROTVAR if c.endswith('_a')]
        b_cols = [c for c in CSV_FIELDNAMES_PROTVAR if c.endswith('_b')]
        assert len(a_cols) == len(b_cols) == 4


# ── Test Class 3: AM Variant Parsing ─────────────────────────────────

@pytest.mark.protvar
@pytest.mark.alphamissense
class TestAMVariantParsing:
    """Test _parse_am_variant() function."""

    def test_simple_variant(self):
        assert _parse_am_variant('M1A') == ('M', 1, 'A')

    def test_multi_digit_position(self):
        assert _parse_am_variant('K81P') == ('K', 81, 'P')

    def test_large_position(self):
        assert _parse_am_variant('S2699R') == ('S', 2699, 'R')

    def test_invalid_variant_returns_none(self):
        assert _parse_am_variant('') is None
        assert _parse_am_variant('123') is None
        assert _parse_am_variant('M1') is None

    def test_lowercase_returns_none(self):
        assert _parse_am_variant('m1a') is None


# ── Test Class 4: AlphaMissense Loading ──────────────────────────────

@pytest.mark.protvar
@pytest.mark.alphamissense
class TestAlphaMissenseLoading:
    """Test load_alphamissense_scores() function."""

    def test_load_p61981(self):
        index = load_alphamissense_scores(TEST_AM_TSV, frozenset({'P61981'}))
        assert 'P61981' in index
        assert (1, 'A') in index['P61981']
        assert abs(index['P61981'][(1, 'A')]['am_score'] - 0.363) < 0.001

    def test_load_p24534(self):
        index = load_alphamissense_scores(TEST_AM_TSV, frozenset({'P24534'}))
        assert 'P24534' in index
        assert abs(index['P24534'][(1, 'D')]['am_score'] - 0.9669) < 0.001

    def test_am_class_loaded(self):
        index = load_alphamissense_scores(TEST_AM_TSV, frozenset({'P61981'}))
        assert index['P61981'][(1, 'A')]['am_class'] == 'ambiguous'
        assert index['P61981'][(1, 'D')]['am_class'] == 'pathogenic'
        assert index['P61981'][(1, 'F')]['am_class'] == 'benign'

    def test_filter_by_accession(self):
        index = load_alphamissense_scores(TEST_AM_TSV, frozenset({'P61981'}))
        assert 'P61981' in index
        assert 'P24534' not in index

    def test_filter_by_position(self):
        index = load_alphamissense_scores(
            TEST_AM_TSV, frozenset({'P61981'}),
            variant_positions={'P61981': {1}},
        )
        assert (1, 'A') in index['P61981']
        assert (2, 'A') not in index['P61981']

    def test_empty_accessions(self):
        index = load_alphamissense_scores(TEST_AM_TSV, frozenset())
        assert index == {}

    def test_missing_file(self):
        index = load_alphamissense_scores(
            Path('/nonexistent/file.tsv'), frozenset({'P61981'}),
        )
        assert index == {}

    def test_header_lines_skipped(self):
        index = load_alphamissense_scores(TEST_AM_TSV, frozenset({'P61981'}))
        assert 'P61981' in index
        assert len(index['P61981']) > 0

    def test_multiple_proteins(self):
        index = load_alphamissense_scores(
            TEST_AM_TSV, frozenset({'P61981', 'P24534'}),
        )
        assert 'P61981' in index
        assert 'P24534' in index

    def test_19_substitutions_per_position(self):
        index = load_alphamissense_scores(
            TEST_AM_TSV, frozenset({'P61981'}),
            variant_positions={'P61981': {1}},
        )
        pos1_entries = {k for k in index['P61981'] if k[0] == 1}
        assert len(pos1_entries) == 19


# ── Test Class 5: FoldX Export Loading ───────────────────────────────

@pytest.mark.protvar
class TestFoldXExportLoading:
    """Test load_foldx_export() function."""

    def test_load_p61981(self):
        index = load_foldx_export(TEST_FOLDX_CSV, frozenset({'P61981'}))
        assert 'P61981' in index
        assert (1, 'A') in index['P61981']
        assert abs(index['P61981'][(1, 'A')]['foldx_ddg'] - 0.114505) < 0.0001

    def test_plddt_loaded(self):
        index = load_foldx_export(TEST_FOLDX_CSV, frozenset({'P61981'}))
        assert abs(index['P61981'][(1, 'A')]['plddt'] - 54.50) < 0.01
        assert abs(index['P61981'][(2, 'A')]['plddt'] - 71.09) < 0.01

    def test_filter_by_accession(self):
        index = load_foldx_export(TEST_FOLDX_CSV, frozenset({'P24534'}))
        assert 'P24534' in index
        assert 'P61981' not in index

    def test_filter_by_position(self):
        index = load_foldx_export(
            TEST_FOLDX_CSV, frozenset({'P61981'}),
            variant_positions={'P61981': {2}},
        )
        assert (2, 'A') in index['P61981']
        assert (1, 'A') not in index['P61981']

    def test_empty_accessions(self):
        index = load_foldx_export(TEST_FOLDX_CSV, frozenset())
        assert index == {}

    def test_missing_file(self):
        index = load_foldx_export(
            Path('/nonexistent/file.csv'), frozenset({'P61981'}),
        )
        assert index == {}

    def test_negative_ddg_values(self):
        index = load_foldx_export(TEST_FOLDX_CSV, frozenset({'P61981'}))
        assert index['P61981'][(1, 'Q')]['foldx_ddg'] < 0

    def test_p24534_destabilising(self):
        index = load_foldx_export(TEST_FOLDX_CSV, frozenset({'P24534'}))
        ddg = index['P24534'][(1, 'F')]['foldx_ddg']
        assert ddg > FOLDX_DESTABILISING_THRESHOLD

    def test_multiple_proteins(self):
        index = load_foldx_export(
            TEST_FOLDX_CSV, frozenset({'P61981', 'P24534'}),
        )
        assert 'P61981' in index
        assert 'P24534' in index

    def test_19_substitutions_per_position(self):
        index = load_foldx_export(
            TEST_FOLDX_CSV, frozenset({'P61981'}),
            variant_positions={'P61981': {1}},
        )
        pos1_entries = {k for k in index['P61981'] if k[0] == 1}
        assert len(pos1_entries) == 19


# ── Test Class 6: Index Building ─────────────────────────────────────

@pytest.mark.protvar
class TestIndexBuilding:
    """Test build_protvar_index() function."""

    def test_merges_am_and_foldx(self, small_index):
        entry = small_index['P61981'][(1, 'A')]
        assert 'am_score' in entry
        assert 'am_class' in entry
        assert 'foldx_ddg' in entry
        assert 'plddt' in entry

    def test_isoform_stripping(self):
        index = build_protvar_index(
            accessions={'P61981-2'},
            foldx_path=TEST_FOLDX_CSV, am_path=TEST_AM_TSV,
        )
        assert 'P61981' in index

    def test_missing_am_file(self):
        index = build_protvar_index(
            accessions={'P61981'},
            foldx_path=TEST_FOLDX_CSV, am_path=Path('/nonexistent.tsv'),
        )
        assert 'P61981' in index
        entry = index['P61981'][(1, 'A')]
        assert 'foldx_ddg' in entry
        assert 'am_score' not in entry

    def test_missing_foldx_file(self):
        index = build_protvar_index(
            accessions={'P61981'},
            foldx_path=Path('/nonexistent.csv'), am_path=TEST_AM_TSV,
        )
        assert 'P61981' in index
        entry = index['P61981'][(1, 'A')]
        assert 'am_score' in entry
        assert 'foldx_ddg' not in entry

    def test_both_missing_returns_empty(self):
        index = build_protvar_index(
            accessions={'P61981'},
            foldx_path=Path('/nonexistent.csv'),
            am_path=Path('/nonexistent.tsv'),
        )
        assert index == {}

    def test_empty_accessions(self):
        index = build_protvar_index(
            accessions=set(),
            foldx_path=TEST_FOLDX_CSV, am_path=TEST_AM_TSV,
        )
        assert index == {}

    def test_position_filtering(self):
        index = build_protvar_index(
            accessions={'P61981'},
            variant_positions={'P61981': {1}},
            foldx_path=TEST_FOLDX_CSV, am_path=TEST_AM_TSV,
        )
        assert (1, 'A') in index['P61981']
        assert (2, 'A') not in index['P61981']

    def test_default_paths_are_set(self):
        """Default paths are defined (not validated — real files are large)."""
        assert DEFAULT_FOLDX_EXPORT is not None
        assert DEFAULT_AM_FILE is not None


# ── Test Class 7: Score Lookup ───────────────────────────────────────

@pytest.mark.protvar
class TestLookup:
    """Test lookup_score() function."""

    def test_existing_variant(self, small_index):
        result = lookup_score(small_index, 'P61981', 1, 'A')
        assert result is not None
        assert abs(result['am_score'] - 0.363) < 0.001
        assert abs(result['foldx_ddg'] - 0.114505) < 0.0001

    def test_missing_protein(self, small_index):
        assert lookup_score(small_index, 'NONEXISTENT', 1, 'A') is None

    def test_missing_position(self, small_index):
        assert lookup_score(small_index, 'P61981', 999, 'A') is None

    def test_isoform_stripping(self, small_index):
        result = lookup_score(small_index, 'P61981-2', 1, 'A')
        assert result is not None

    def test_plddt_in_result(self, small_index):
        result = lookup_score(small_index, 'P61981', 1, 'A')
        assert abs(result['plddt'] - 54.50) < 0.01

    def test_am_class_in_result(self, small_index):
        result = lookup_score(small_index, 'P61981', 1, 'D')
        assert result['am_class'] == 'pathogenic'


# ── Test Class 8: Variant Detail Parsing ─────────────────────────────

@pytest.mark.protvar
class TestVariantDetailParsing:
    """Test _parse_variant_details_for_protvar() function."""

    def test_single_variant(self):
        result = _parse_variant_details_for_protvar('K81P:interface_core:pathogenic')
        assert result == [('K', 81, 'P')]

    def test_multiple_variants(self):
        result = _parse_variant_details_for_protvar(
            'K81P:interface_core:pathogenic|E82K:interface_rim:VUS'
        )
        assert len(result) == 2

    def test_truncation_marker_skipped(self):
        result = _parse_variant_details_for_protvar(
            'K81P:interface_core:pathogenic|...(+5 more)'
        )
        assert len(result) == 1

    def test_empty_string(self):
        assert _parse_variant_details_for_protvar('') == []

    def test_stop_codon(self):
        result = _parse_variant_details_for_protvar('K81*:interface_core:pathogenic')
        assert result == [('K', 81, '*')]


# ── Test Class 9: Formatting ─────────────────────────────────────────

@pytest.mark.protvar
class TestFormatting:
    """Test format_protvar_details() function."""

    def test_single_variant(self):
        result = format_protvar_details([{
            'ref_aa': 'M', 'position': 1, 'alt_aa': 'A',
            'am_score': 0.363, 'am_class': 'ambiguous', 'foldx_ddg': 0.114,
        }])
        assert result == 'M1A:am=0.36:ambiguous:foldx=0.11'

    def test_missing_am_score(self):
        result = format_protvar_details([{
            'ref_aa': 'M', 'position': 1, 'alt_aa': 'A',
            'am_score': None, 'am_class': '', 'foldx_ddg': 0.5,
        }])
        assert 'am=-' in result

    def test_missing_foldx_ddg(self):
        result = format_protvar_details([{
            'ref_aa': 'M', 'position': 1, 'alt_aa': 'A',
            'am_score': 0.5, 'am_class': 'ambiguous', 'foldx_ddg': None,
        }])
        assert 'foldx=-' in result

    def test_empty_list(self):
        assert format_protvar_details([]) == ''

    def test_truncation(self):
        variants = [
            {'ref_aa': 'M', 'position': i, 'alt_aa': 'A',
             'am_score': 0.5, 'am_class': 'ambiguous', 'foldx_ddg': 0.1}
            for i in range(25)
        ]
        result = format_protvar_details(variants)
        assert '...(+5 more)' in result

    def test_custom_limit(self):
        variants = [
            {'ref_aa': 'M', 'position': i, 'alt_aa': 'A',
             'am_score': 0.5, 'am_class': 'ambiguous', 'foldx_ddg': 0.1}
            for i in range(5)
        ]
        result = format_protvar_details(variants, limit=3)
        assert '...(+2 more)' in result


# ── Test Class 10: Chain Scoring ─────────────────────────────────────

@pytest.mark.protvar
class TestChainScoring:
    """Test _score_chain_variants_protvar() function."""

    def test_am_mean_computed(self, small_index):
        result = _score_chain_variants_protvar(
            'P61981', 'M1A:interface_core:pathogenic', small_index,
        )
        assert abs(result['am_mean'] - 0.363) < 0.001

    def test_foldx_mean_computed(self, small_index):
        result = _score_chain_variants_protvar(
            'P61981', 'M1A:interface_core:pathogenic', small_index,
        )
        assert abs(result['foldx_mean'] - 0.114505) < 0.001

    def test_pathogenic_count(self, small_index):
        result = _score_chain_variants_protvar(
            'P61981', 'M1D:interface_core:pathogenic', small_index,
        )
        assert result['am_n_pathogenic'] == 1

    def test_benign_not_counted(self, small_index):
        result = _score_chain_variants_protvar(
            'P61981', 'M1F:interface_core:pathogenic', small_index,
        )
        assert result['am_n_pathogenic'] == 0

    def test_multiple_variants_mean(self, small_index):
        details = 'M1A:interface_core:pathogenic|M1D:interface_core:pathogenic'
        result = _score_chain_variants_protvar('P61981', details, small_index)
        expected_am = round(mean([0.363, 0.809]), 4)
        assert abs(result['am_mean'] - expected_am) < 0.001

    def test_empty_details(self, small_index):
        result = _score_chain_variants_protvar('P61981', '', small_index)
        assert result['am_mean'] == ''
        assert result['foldx_mean'] == ''
        assert result['am_n_pathogenic'] == 0

    def test_unknown_protein(self, small_index):
        result = _score_chain_variants_protvar(
            'NONEXISTENT', 'M1A:interface_core:pathogenic', small_index,
        )
        assert result['am_mean'] == ''
        assert result['foldx_mean'] == ''

    def test_details_string_generated(self, small_index):
        result = _score_chain_variants_protvar(
            'P61981', 'M1A:interface_core:pathogenic', small_index,
        )
        assert 'M1A:am=' in result['details']


# ── Test Class 11: Annotation ────────────────────────────────────────

@pytest.mark.protvar
class TestAnnotation:
    """Test annotate_results_with_protvar() function."""

    def test_annotates_both_chains(self, small_index):
        results = [{
            'complex_name': 'test', 'protein_a': 'P61981', 'protein_b': 'P24534',
            'variant_details_a': 'M1A:interface_core:pathogenic',
            'variant_details_b': 'M1D:interface_core:pathogenic',
        }]
        annotate_results_with_protvar(results, small_index)
        assert results[0]['protvar_am_mean_a'] != ''
        assert results[0]['protvar_am_mean_b'] != ''

    def test_empty_details_produces_empty_columns(self, small_index):
        results = [{
            'complex_name': 'test', 'protein_a': 'P61981', 'protein_b': 'P24534',
            'variant_details_a': '', 'variant_details_b': '',
        }]
        annotate_results_with_protvar(results, small_index)
        assert results[0]['protvar_am_mean_a'] == ''
        assert results[0]['protvar_foldx_mean_a'] == ''
        assert results[0]['protvar_am_n_pathogenic_a'] == ''
        assert results[0]['protvar_details_a'] == ''

    def test_all_8_columns_set(self, small_index):
        results = [{
            'complex_name': 'test', 'protein_a': 'P61981', 'protein_b': 'P24534',
            'variant_details_a': 'M1A:interface_core:pathogenic',
            'variant_details_b': 'M1A:interface_core:pathogenic',
        }]
        annotate_results_with_protvar(results, small_index)
        for col in CSV_FIELDNAMES_PROTVAR:
            assert col in results[0], f"Missing column: {col}"

    def test_missing_protein(self, small_index):
        results = [{
            'complex_name': 'test', 'protein_a': '', 'protein_b': '',
            'variant_details_a': '', 'variant_details_b': '',
        }]
        annotate_results_with_protvar(results, small_index)
        assert results[0]['protvar_am_mean_a'] == ''

    def test_isoform_accession(self, small_index):
        results = [{
            'complex_name': 'test', 'protein_a': 'P61981-2', 'protein_b': 'P24534',
            'variant_details_a': 'M1A:interface_core:pathogenic',
            'variant_details_b': '',
        }]
        annotate_results_with_protvar(results, small_index)
        assert results[0]['protvar_am_mean_a'] != ''

    def test_pathogenic_count_column(self, small_index):
        results = [{
            'complex_name': 'test', 'protein_a': 'P61981', 'protein_b': '',
            'variant_details_a': 'M1D:interface_core:pathogenic|M1F:surface_non_interface:benign',
            'variant_details_b': '',
        }]
        annotate_results_with_protvar(results, small_index)
        assert results[0]['protvar_am_n_pathogenic_a'] == 1

    def test_foldx_mean_column(self, small_index):
        results = [{
            'complex_name': 'test', 'protein_a': 'P61981', 'protein_b': '',
            'variant_details_a': 'M1A:interface_core:pathogenic',
            'variant_details_b': '',
        }]
        annotate_results_with_protvar(results, small_index)
        assert isinstance(results[0]['protvar_foldx_mean_a'], float)

    def test_multiple_results(self, small_index):
        results = [
            {'complex_name': 'test1', 'protein_a': 'P61981', 'protein_b': 'P24534',
             'variant_details_a': 'M1A:interface_core:pathogenic',
             'variant_details_b': 'M1A:interface_core:pathogenic'},
            {'complex_name': 'test2', 'protein_a': 'P61981', 'protein_b': '',
             'variant_details_a': 'V2A:surface_non_interface:benign',
             'variant_details_b': ''},
        ]
        annotate_results_with_protvar(results, small_index)
        assert results[0]['protvar_am_mean_a'] != ''
        assert results[1]['protvar_am_mean_a'] != ''

    def test_details_string_set(self, small_index):
        results = [{
            'complex_name': 'test', 'protein_a': 'P61981', 'protein_b': '',
            'variant_details_a': 'M1A:interface_core:pathogenic',
            'variant_details_b': '',
        }]
        annotate_results_with_protvar(results, small_index)
        assert 'M1A:am=' in results[0]['protvar_details_a']

    def test_modifies_in_place(self, small_index):
        results = [{
            'complex_name': 'test', 'protein_a': 'P61981', 'protein_b': '',
            'variant_details_a': 'M1A:interface_core:pathogenic',
            'variant_details_b': '',
        }]
        ret = annotate_results_with_protvar(results, small_index)
        assert ret is None


# ── Test Class 12: CLI ───────────────────────────────────────────────

@pytest.mark.protvar
@pytest.mark.cli
class TestCLI:
    """Test standalone CLI argument parsing."""

    def test_parser_creation(self):
        assert build_argument_parser() is not None

    def test_summary_subcommand(self):
        args = build_argument_parser().parse_args(['summary'])
        assert args.command == 'summary'

    def test_lookup_subcommand(self):
        args = build_argument_parser().parse_args(['lookup', '--protein', 'P61981'])
        assert args.command == 'lookup'
        assert args.protein == 'P61981'

    def test_lookup_with_position(self):
        args = build_argument_parser().parse_args(
            ['lookup', '--protein', 'P61981', '--position', '4'])
        assert args.position == 4

    def test_custom_foldx_export(self):
        args = build_argument_parser().parse_args(
            ['--foldx-export', '/custom/path.csv', 'summary'])
        assert args.foldx_export == '/custom/path.csv'

    def test_custom_am_file(self):
        args = build_argument_parser().parse_args(
            ['--am-file', '/custom/am.tsv', 'summary'])
        assert args.am_file == '/custom/am.tsv'


# ── Test Class 13: Regression Values ─────────────────────────────────

@pytest.mark.protvar
@pytest.mark.regression
class TestRegressionValues:
    """Verify exact numerical values from test data for P61981."""

    def test_p61981_m1a_am_score(self, small_index):
        result = lookup_score(small_index, 'P61981', 1, 'A')
        assert result['am_score'] == 0.363

    def test_p61981_m1a_foldx_ddg(self, small_index):
        result = lookup_score(small_index, 'P61981', 1, 'A')
        assert abs(result['foldx_ddg'] - 0.114505) < 1e-6

    def test_p61981_pos1_plddt(self, small_index):
        result = lookup_score(small_index, 'P61981', 1, 'A')
        assert result['plddt'] == 54.50

    def test_p61981_pos2_plddt(self, small_index):
        result = lookup_score(small_index, 'P61981', 2, 'A')
        assert result['plddt'] == 71.09

    def test_p24534_m1w_destabilising(self, small_index):
        result = lookup_score(small_index, 'P24534', 1, 'W')
        assert abs(result['foldx_ddg'] - 4.79105) < 1e-4
        assert result['foldx_ddg'] > FOLDX_DESTABILISING_THRESHOLD
