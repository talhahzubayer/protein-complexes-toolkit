"""
Tests for stability_scorer.py — EVE score loading, annotation, and CLI.

Test data:
    - tests/offline_test_data/databases/test_idmapping.dat (8 lines, 4 accessions)
    - tests/offline_test_data/databases/test_eve_scores.csv (80 rows from 1433G_HUMAN)
    - tests/offline_test_data/databases/test_eve_data/1433G_HUMAN.csv (copy of above)

Reference values (from 1433G_HUMAN.csv, verified March 2026):
    - Position 4: R→A, EVE_scores_ASM=0.7727140175285468, class=Pathogenic
    - Position 1: M→A, no EVE score (empty)
"""

import subprocess
import sys
from pathlib import Path

import pytest


# ── Constants & Known Values ─────────────────────────────────────

# From test_idmapping.dat
TEST_ACCESSION_WITH_EVE = 'P61981'    # maps to 1433G_HUMAN
TEST_ACCESSION_NO_EVE = 'P24534'      # maps to EF1B_HUMAN (no EVE CSV in test dir)
TEST_ENTRY_NAME = '1433G_HUMAN'

# From test_eve_scores.csv (positions 1, 4, 5, 10 from 1433G_HUMAN)
EVE_SCORE_R4A = 0.7727140175285468
EVE_CLASS_R4A = 'Pathogenic'
EVE_UNCERTAINTY_R4A = 0.5359761101025369


# ── Section 1: Entry-Name Mapping Tests ──────────────────────────

@pytest.mark.stability
class TestEVEEntryNameMap:
    """Tests for load_eve_entry_name_map()."""

    def test_loads_mapping_from_dat_file(self, test_eve_map_path):
        """Parses HUMAN_9606_idmapping.dat and extracts UniProtKB-ID rows."""
        from stability_scorer import load_eve_entry_name_map
        acc_to_entry = load_eve_entry_name_map(test_eve_map_path)
        assert isinstance(acc_to_entry, dict)
        assert len(acc_to_entry) > 0

    def test_correct_mapping_p61981(self, test_eve_map_path):
        """P61981 maps to 1433G_HUMAN."""
        from stability_scorer import load_eve_entry_name_map
        acc_to_entry = load_eve_entry_name_map(test_eve_map_path)
        assert acc_to_entry['P61981'] == '1433G_HUMAN'

    def test_correct_mapping_p31946(self, test_eve_map_path):
        """P31946 maps to 1433B_HUMAN."""
        from stability_scorer import load_eve_entry_name_map
        acc_to_entry = load_eve_entry_name_map(test_eve_map_path)
        assert acc_to_entry['P31946'] == '1433B_HUMAN'

    def test_filters_non_entry_name_rows(self, test_eve_map_path):
        """Only UniProtKB-ID rows are included, not Gene_Name etc."""
        from stability_scorer import load_eve_entry_name_map
        acc_to_entry = load_eve_entry_name_map(test_eve_map_path)
        # test_idmapping.dat has 4 UniProtKB-ID rows and 4 Gene_Name rows
        assert len(acc_to_entry) == 4

    def test_missing_file_returns_empty_with_warning(self, tmp_path):
        """Missing mapping file returns empty dict and emits warning."""
        from stability_scorer import load_eve_entry_name_map
        with pytest.warns(UserWarning, match="ID mapping file not found"):
            result = load_eve_entry_name_map(tmp_path / "nonexistent.dat")
        assert result == {}

    def test_empty_file_returns_empty(self, tmp_path):
        """Empty mapping file returns empty dict."""
        from stability_scorer import load_eve_entry_name_map
        empty_file = tmp_path / "empty.dat"
        empty_file.write_text("")
        result = load_eve_entry_name_map(empty_file)
        assert result == {}

    def test_malformed_lines_skipped(self, tmp_path):
        """Lines with fewer than 3 tab-separated fields are skipped."""
        from stability_scorer import load_eve_entry_name_map
        bad_file = tmp_path / "bad.dat"
        bad_file.write_text("P61981\tUniProtKB-ID\t1433G_HUMAN\n"
                            "BADLINE\n"
                            "P31946\tUniProtKB-ID\n")  # only 2 fields
        result = load_eve_entry_name_map(bad_file)
        assert len(result) == 1
        assert result['P61981'] == '1433G_HUMAN'

    def test_verbose_prints_stats(self, test_eve_map_path, capsys):
        """Verbose mode prints mapping statistics to stderr."""
        from stability_scorer import load_eve_entry_name_map
        load_eve_entry_name_map(test_eve_map_path, verbose=True)
        captured = capsys.readouterr()
        assert "entry names" in captured.err
        assert "lines" in captured.err


# ── Section 2: EVE Score Loading Tests ───────────────────────────

@pytest.mark.stability
class TestEVEScoreLoading:
    """Tests for load_eve_scores_for_protein()."""

    def test_loads_scores_from_csv(self, test_eve_dir):
        """Parses EVE CSV and returns non-empty dict."""
        from stability_scorer import load_eve_scores_for_protein
        csv_path = test_eve_dir / "1433G_HUMAN.csv"
        scores = load_eve_scores_for_protein(csv_path)
        assert isinstance(scores, dict)
        assert len(scores) > 0

    def test_correct_score_r4a(self, test_eve_dir):
        """R4A variant has expected EVE score."""
        from stability_scorer import load_eve_scores_for_protein
        csv_path = test_eve_dir / "1433G_HUMAN.csv"
        scores = load_eve_scores_for_protein(csv_path)
        result = scores[('R', 4, 'A')]
        assert abs(result['eve_score'] - EVE_SCORE_R4A) < 1e-10

    def test_correct_class_r4a(self, test_eve_dir):
        """R4A variant classified as Pathogenic."""
        from stability_scorer import load_eve_scores_for_protein
        csv_path = test_eve_dir / "1433G_HUMAN.csv"
        scores = load_eve_scores_for_protein(csv_path)
        assert scores[('R', 4, 'A')]['eve_class'] == 'Pathogenic'

    def test_correct_uncertainty_r4a(self, test_eve_dir):
        """R4A variant has expected uncertainty value."""
        from stability_scorer import load_eve_scores_for_protein
        csv_path = test_eve_dir / "1433G_HUMAN.csv"
        scores = load_eve_scores_for_protein(csv_path)
        assert abs(scores[('R', 4, 'A')]['eve_uncertainty'] - EVE_UNCERTAINTY_R4A) < 1e-10

    def test_empty_score_is_none(self, test_eve_dir):
        """Position 1 variants have None scores (not in EVE model range)."""
        from stability_scorer import load_eve_scores_for_protein
        csv_path = test_eve_dir / "1433G_HUMAN.csv"
        scores = load_eve_scores_for_protein(csv_path)
        result = scores[('M', 1, 'A')]
        assert result['eve_score'] is None
        assert result['eve_class'] is None

    def test_tuple_key_format(self, test_eve_dir):
        """Keys are (str, int, str) tuples."""
        from stability_scorer import load_eve_scores_for_protein
        csv_path = test_eve_dir / "1433G_HUMAN.csv"
        scores = load_eve_scores_for_protein(csv_path)
        key = next(iter(scores))
        assert isinstance(key, tuple)
        assert len(key) == 3
        assert isinstance(key[0], str)
        assert isinstance(key[1], int)
        assert isinstance(key[2], str)

    def test_total_row_count(self, test_eve_dir):
        """Test CSV has 80 rows (positions 1,4,5,10 x 20 substitutions each)."""
        from stability_scorer import load_eve_scores_for_protein
        csv_path = test_eve_dir / "1433G_HUMAN.csv"
        scores = load_eve_scores_for_protein(csv_path)
        assert len(scores) == 80

    def test_file_not_found_raises(self, tmp_path):
        """Missing CSV raises FileNotFoundError."""
        from stability_scorer import load_eve_scores_for_protein
        with pytest.raises(FileNotFoundError):
            load_eve_scores_for_protein(tmp_path / "nonexistent.csv")

    def test_missing_columns_raises(self, tmp_path):
        """CSV without required columns raises ValueError."""
        from stability_scorer import load_eve_scores_for_protein
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("col_a,col_b\n1,2\n")
        with pytest.raises(ValueError, match="missing required columns"):
            load_eve_scores_for_protein(bad_csv)

    def test_empty_csv_raises(self, tmp_path):
        """Completely empty CSV raises ValueError."""
        from stability_scorer import load_eve_scores_for_protein
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("")
        with pytest.raises(ValueError, match="Empty CSV"):
            load_eve_scores_for_protein(empty_csv)


# ── Section 3: EVE Index Building Tests ──────────────────────────

@pytest.mark.stability
class TestEVEIndex:
    """Tests for build_eve_index()."""

    def test_builds_index_for_known_accession(self, test_eve_dir, test_eve_map_path):
        """Builds index containing P61981 (1433G_HUMAN)."""
        from stability_scorer import build_eve_index, load_eve_entry_name_map
        acc_to_entry = load_eve_entry_name_map(test_eve_map_path)
        index = build_eve_index(test_eve_dir, frozenset({'P61981'}), acc_to_entry)
        assert 'P61981' in index
        assert len(index['P61981']) == 80

    def test_missing_accession_omitted(self, test_eve_dir, test_eve_map_path):
        """Accession without EVE CSV is not in index."""
        from stability_scorer import build_eve_index, load_eve_entry_name_map
        acc_to_entry = load_eve_entry_name_map(test_eve_map_path)
        index = build_eve_index(test_eve_dir, frozenset({'P24534'}), acc_to_entry)
        assert 'P24534' not in index

    def test_unknown_accession_omitted(self, test_eve_dir, test_eve_map_path):
        """Accession not in mapping file is not in index."""
        from stability_scorer import build_eve_index, load_eve_entry_name_map
        acc_to_entry = load_eve_entry_name_map(test_eve_map_path)
        index = build_eve_index(test_eve_dir, frozenset({'XXXXXX'}), acc_to_entry)
        assert len(index) == 0

    def test_isoform_accession_stripped(self, test_eve_dir, test_eve_map_path):
        """Isoform accession P61981-2 maps to base P61981 for EVE lookup."""
        from stability_scorer import build_eve_index, load_eve_entry_name_map
        acc_to_entry = load_eve_entry_name_map(test_eve_map_path)
        index = build_eve_index(test_eve_dir, frozenset({'P61981-2'}), acc_to_entry)
        assert 'P61981' in index

    def test_empty_accessions_returns_empty(self, test_eve_dir, test_eve_map_path):
        """Empty accession set returns empty index."""
        from stability_scorer import build_eve_index, load_eve_entry_name_map
        acc_to_entry = load_eve_entry_name_map(test_eve_map_path)
        index = build_eve_index(test_eve_dir, frozenset(), acc_to_entry)
        assert len(index) == 0

    def test_missing_eve_dir_returns_empty(self, tmp_path, test_eve_map_path):
        """Non-existent EVE directory returns empty index."""
        from stability_scorer import build_eve_index, load_eve_entry_name_map
        acc_to_entry = load_eve_entry_name_map(test_eve_map_path)
        index = build_eve_index(tmp_path / "nope", frozenset({'P61981'}), acc_to_entry)
        assert len(index) == 0

    def test_verbose_prints_stats(self, test_eve_dir, test_eve_map_path, capsys):
        """Verbose mode prints loading statistics."""
        from stability_scorer import build_eve_index, load_eve_entry_name_map
        acc_to_entry = load_eve_entry_name_map(test_eve_map_path)
        build_eve_index(test_eve_dir, frozenset({'P61981'}), acc_to_entry, verbose=True)
        captured = capsys.readouterr()
        assert "loaded" in captured.err

    def test_multiple_accessions_mixed(self, test_eve_dir, test_eve_map_path):
        """Index built for mixed set: some with EVE data, some without."""
        from stability_scorer import build_eve_index, load_eve_entry_name_map
        acc_to_entry = load_eve_entry_name_map(test_eve_map_path)
        index = build_eve_index(
            test_eve_dir, frozenset({'P61981', 'P24534', 'XXXXXX'}), acc_to_entry,
        )
        assert 'P61981' in index
        assert 'P24534' not in index
        assert 'XXXXXX' not in index


# ── Section 4: EVE Score Lookup Tests ────────────────────────────

@pytest.mark.stability
class TestEVELookup:
    """Tests for lookup_eve_score()."""

    def test_exact_match(self, test_eve_dir, test_eve_map_path):
        """Looks up R4A variant for P61981."""
        from stability_scorer import build_eve_index, load_eve_entry_name_map, lookup_eve_score
        acc_to_entry = load_eve_entry_name_map(test_eve_map_path)
        index = build_eve_index(test_eve_dir, frozenset({'P61981'}), acc_to_entry)
        result = lookup_eve_score(index, 'P61981', 'R', 4, 'A')
        assert result is not None
        assert abs(result['eve_score'] - EVE_SCORE_R4A) < 1e-10

    def test_missing_variant_returns_none(self, test_eve_dir, test_eve_map_path):
        """Variant not in EVE CSV returns None."""
        from stability_scorer import build_eve_index, load_eve_entry_name_map, lookup_eve_score
        acc_to_entry = load_eve_entry_name_map(test_eve_map_path)
        index = build_eve_index(test_eve_dir, frozenset({'P61981'}), acc_to_entry)
        # Position 999 doesn't exist in test data
        result = lookup_eve_score(index, 'P61981', 'R', 999, 'A')
        assert result is None

    def test_missing_protein_returns_none(self, test_eve_dir, test_eve_map_path):
        """Protein not in index returns None."""
        from stability_scorer import build_eve_index, load_eve_entry_name_map, lookup_eve_score
        acc_to_entry = load_eve_entry_name_map(test_eve_map_path)
        index = build_eve_index(test_eve_dir, frozenset({'P61981'}), acc_to_entry)
        result = lookup_eve_score(index, 'XXXXXX', 'R', 4, 'A')
        assert result is None

    def test_isoform_lookup(self, test_eve_dir, test_eve_map_path):
        """Isoform accession P61981-2 resolves to base accession."""
        from stability_scorer import build_eve_index, load_eve_entry_name_map, lookup_eve_score
        acc_to_entry = load_eve_entry_name_map(test_eve_map_path)
        index = build_eve_index(test_eve_dir, frozenset({'P61981'}), acc_to_entry)
        result = lookup_eve_score(index, 'P61981-2', 'R', 4, 'A')
        assert result is not None

    def test_empty_index_returns_none(self):
        """Lookup on empty index returns None."""
        from stability_scorer import lookup_eve_score
        result = lookup_eve_score({}, 'P61981', 'R', 4, 'A')
        assert result is None


# ── Section 5: Variant Detail Parsing Tests ──────────────────────

@pytest.mark.stability
class TestParseVariantDetails:
    """Tests for _parse_variant_details_for_eve()."""

    def test_parses_single_variant(self):
        """Parses a single variant detail string."""
        from stability_scorer import _parse_variant_details_for_eve
        result = _parse_variant_details_for_eve("K81P:interface_core:pathogenic")
        assert result == [('K', 81, 'P')]

    def test_parses_multiple_variants(self):
        """Parses pipe-separated variants."""
        from stability_scorer import _parse_variant_details_for_eve
        result = _parse_variant_details_for_eve(
            "K81P:interface_core:pathogenic|E82K:interface_rim:VUS"
        )
        assert result == [('K', 81, 'P'), ('E', 82, 'K')]

    def test_skips_truncation_indicator(self):
        """Skips '...(+N more)' suffix."""
        from stability_scorer import _parse_variant_details_for_eve
        result = _parse_variant_details_for_eve(
            "K81P:interface_core:pathogenic|...(+5 more)"
        )
        assert result == [('K', 81, 'P')]

    def test_empty_string_returns_empty(self):
        """Empty string returns empty list."""
        from stability_scorer import _parse_variant_details_for_eve
        assert _parse_variant_details_for_eve("") == []

    def test_stop_codon_variant(self):
        """Handles stop codon (*) in variant."""
        from stability_scorer import _parse_variant_details_for_eve
        result = _parse_variant_details_for_eve("R100*:buried_core:-")
        assert result == [('R', 100, '*')]


# ── Section 6: Format Stability Details Tests ────────────────────

@pytest.mark.stability
class TestFormatStabilityDetails:
    """Tests for format_stability_details()."""

    def test_empty_list_returns_empty(self):
        """Empty list returns empty string."""
        from stability_scorer import format_stability_details
        assert format_stability_details([]) == ''

    def test_single_variant_format(self):
        """Single variant formatted correctly."""
        from stability_scorer import format_stability_details
        variants = [{'ref_aa': 'R', 'position': 4, 'alt_aa': 'A',
                      'eve_score': 0.77, 'eve_class': 'Pathogenic'}]
        result = format_stability_details(variants)
        assert result == 'R4A:eve=0.77:Pathogenic'

    def test_multiple_variants_pipe_separated(self):
        """Multiple variants are pipe-separated."""
        from stability_scorer import format_stability_details
        variants = [
            {'ref_aa': 'R', 'position': 4, 'alt_aa': 'A',
             'eve_score': 0.77, 'eve_class': 'Pathogenic'},
            {'ref_aa': 'K', 'position': 10, 'alt_aa': 'E',
             'eve_score': 0.23, 'eve_class': 'Benign'},
        ]
        result = format_stability_details(variants)
        assert 'R4A:eve=0.77:Pathogenic' in result
        assert 'K10E:eve=0.23:Benign' in result
        assert '|' in result

    def test_none_score_shows_dash(self):
        """None EVE score renders as dash."""
        from stability_scorer import format_stability_details
        variants = [{'ref_aa': 'M', 'position': 1, 'alt_aa': 'A',
                      'eve_score': None, 'eve_class': ''}]
        result = format_stability_details(variants)
        assert 'eve=-' in result

    def test_truncation_indicator(self):
        """Excess variants show truncation count."""
        from stability_scorer import format_stability_details
        variants = [{'ref_aa': 'A', 'position': i, 'alt_aa': 'G',
                      'eve_score': 0.5, 'eve_class': 'Uncertain'}
                     for i in range(25)]
        result = format_stability_details(variants, limit=20)
        assert '...(+5 more)' in result


# ── Section 7: Annotation Tests ──────────────────────────────────

@pytest.mark.stability
class TestAnnotation:
    """Tests for annotate_results_with_stability()."""

    def _make_result_row(self, protein_a='P61981', protein_b='P24534',
                         details_a='', details_b=''):
        """Helper to create a minimal result dict for testing."""
        return {
            'protein_a': protein_a,
            'protein_b': protein_b,
            'variant_details_a': details_a,
            'variant_details_b': details_b,
        }

    def test_adds_all_columns(self, test_eve_dir, test_eve_map_path):
        """All 8 stability columns are added to result dict."""
        from stability_scorer import (
            annotate_results_with_stability, build_eve_index,
            load_eve_entry_name_map, CSV_FIELDNAMES_STABILITY,
        )
        acc_to_entry = load_eve_entry_name_map(test_eve_map_path)
        index = build_eve_index(test_eve_dir, frozenset({'P61981'}), acc_to_entry)
        results = [self._make_result_row()]
        annotate_results_with_stability(results, index, acc_to_entry)
        for col in CSV_FIELDNAMES_STABILITY:
            assert col in results[0], f"Missing column: {col}"

    def test_no_variants_produces_empty_columns(self, test_eve_dir, test_eve_map_path):
        """Complex with no variants gets empty/zero stability columns."""
        from stability_scorer import (
            annotate_results_with_stability, build_eve_index,
            load_eve_entry_name_map,
        )
        acc_to_entry = load_eve_entry_name_map(test_eve_map_path)
        index = build_eve_index(test_eve_dir, frozenset({'P61981'}), acc_to_entry)
        results = [self._make_result_row(details_a='', details_b='')]
        annotate_results_with_stability(results, index, acc_to_entry)
        assert results[0]['eve_score_mean_a'] == ''
        assert results[0]['eve_n_pathogenic_a'] == 0
        assert results[0]['stability_details_a'] == ''

    def test_scored_variant_populates_columns(self, test_eve_dir, test_eve_map_path):
        """Variant with EVE data produces non-empty score columns."""
        from stability_scorer import (
            annotate_results_with_stability, build_eve_index,
            load_eve_entry_name_map,
        )
        acc_to_entry = load_eve_entry_name_map(test_eve_map_path)
        index = build_eve_index(test_eve_dir, frozenset({'P61981'}), acc_to_entry)
        results = [self._make_result_row(
            details_a='R4A:interface_core:pathogenic',
        )]
        annotate_results_with_stability(results, index, acc_to_entry)
        assert results[0]['eve_score_mean_a'] != ''
        assert results[0]['eve_n_pathogenic_a'] == 1
        assert results[0]['eve_coverage_a'] == 1.0
        assert 'R4A' in results[0]['stability_details_a']

    def test_unscored_variant_zero_coverage(self, test_eve_dir, test_eve_map_path):
        """Variant at position 1 has no EVE score → coverage 0."""
        from stability_scorer import (
            annotate_results_with_stability, build_eve_index,
            load_eve_entry_name_map,
        )
        acc_to_entry = load_eve_entry_name_map(test_eve_map_path)
        index = build_eve_index(test_eve_dir, frozenset({'P61981'}), acc_to_entry)
        results = [self._make_result_row(
            details_a='M1A:surface_non_interface:-',
        )]
        annotate_results_with_stability(results, index, acc_to_entry)
        # M1A exists in EVE CSV but has no score → coverage should be 0
        assert results[0]['eve_coverage_a'] == 0.0

    def test_mixed_scored_unscored(self, test_eve_dir, test_eve_map_path):
        """Mix of scored and unscored variants produces partial coverage."""
        from stability_scorer import (
            annotate_results_with_stability, build_eve_index,
            load_eve_entry_name_map,
        )
        acc_to_entry = load_eve_entry_name_map(test_eve_map_path)
        index = build_eve_index(test_eve_dir, frozenset({'P61981'}), acc_to_entry)
        results = [self._make_result_row(
            details_a='R4A:interface_core:pathogenic|M1A:surface_non_interface:-',
        )]
        annotate_results_with_stability(results, index, acc_to_entry)
        assert results[0]['eve_coverage_a'] == 0.5  # 1 of 2 scored

    def test_empty_eve_index(self):
        """Empty EVE index produces zero/empty columns without error."""
        from stability_scorer import annotate_results_with_stability
        results = [self._make_result_row(
            details_a='R4A:interface_core:pathogenic',
        )]
        annotate_results_with_stability(results, {}, {})
        assert results[0]['eve_n_pathogenic_a'] == 0
        assert results[0]['eve_coverage_a'] == 0.0

    def test_modifies_in_place(self, test_eve_dir, test_eve_map_path):
        """Results are modified in-place (no new list returned)."""
        from stability_scorer import (
            annotate_results_with_stability, build_eve_index,
            load_eve_entry_name_map,
        )
        acc_to_entry = load_eve_entry_name_map(test_eve_map_path)
        index = build_eve_index(test_eve_dir, frozenset({'P61981'}), acc_to_entry)
        results = [self._make_result_row()]
        ret = annotate_results_with_stability(results, index, acc_to_entry)
        assert ret is None  # in-place modification

    def test_verbose_prints_stats(self, test_eve_dir, test_eve_map_path, capsys):
        """Verbose mode prints annotation progress."""
        from stability_scorer import (
            annotate_results_with_stability, build_eve_index,
            load_eve_entry_name_map,
        )
        acc_to_entry = load_eve_entry_name_map(test_eve_map_path)
        index = build_eve_index(test_eve_dir, frozenset({'P61981'}), acc_to_entry)
        results = [self._make_result_row()]
        annotate_results_with_stability(results, index, acc_to_entry, verbose=True)
        captured = capsys.readouterr()
        assert "complexes processed" in captured.err


# ── Section 8: CSV Fieldnames Tests ──────────────────────────────

@pytest.mark.stability
class TestCSVFieldnames:
    """Tests for CSV column constant."""

    def test_column_count(self):
        """CSV_FIELDNAMES_STABILITY has 8 columns."""
        from stability_scorer import CSV_FIELDNAMES_STABILITY
        assert len(CSV_FIELDNAMES_STABILITY) == 8

    def test_all_columns_are_strings(self):
        """All column names are non-empty strings."""
        from stability_scorer import CSV_FIELDNAMES_STABILITY
        for col in CSV_FIELDNAMES_STABILITY:
            assert isinstance(col, str) and len(col) > 0

    def test_no_duplicate_columns(self):
        """No duplicate column names."""
        from stability_scorer import CSV_FIELDNAMES_STABILITY
        assert len(CSV_FIELDNAMES_STABILITY) == len(set(CSV_FIELDNAMES_STABILITY))


# ── Section 9: CLI Tests ─────────────────────────────────────────

@pytest.mark.stability
@pytest.mark.cli
class TestCLI:
    """Tests for standalone CLI."""

    def test_parser_construction(self):
        """Argument parser builds without error."""
        from stability_scorer import build_argument_parser
        parser = build_argument_parser()
        assert parser is not None

    def test_summary_subcommand(self):
        """Summary subcommand parsed correctly."""
        from stability_scorer import build_argument_parser
        parser = build_argument_parser()
        args = parser.parse_args(['summary'])
        assert args.command == 'summary'

    def test_lookup_subcommand(self):
        """Lookup subcommand with --protein parsed correctly."""
        from stability_scorer import build_argument_parser
        parser = build_argument_parser()
        args = parser.parse_args(['lookup', '--protein', 'P61981'])
        assert args.command == 'lookup'
        assert args.protein == 'P61981'
        assert args.position is None

    def test_lookup_with_position(self):
        """Lookup subcommand with --position parsed correctly."""
        from stability_scorer import build_argument_parser
        parser = build_argument_parser()
        args = parser.parse_args(['lookup', '--protein', 'P61981', '--position', '4'])
        assert args.position == 4

    def test_cli_summary_runs(self, test_eve_map_path, test_eve_dir):
        """CLI summary subcommand runs end-to-end."""
        stability_dir = test_eve_dir.parent  # offline_test_data/databases
        result = subprocess.run(
            [sys.executable, '-m', 'stability_scorer',
             '--stability-dir', str(stability_dir), 'summary'],
            capture_output=True, text=True, timeout=30,
        )
        # May not find all files in test dir, but should not crash
        assert result.returncode == 0 or "not found" in result.stdout


# ── Section 10: Regression Values ────────────────────────────────

@pytest.mark.stability
@pytest.mark.regression
class TestRegressionValues:
    """Regression tests for known EVE score values."""

    def test_eve_score_r4a_regression(self, test_eve_dir, test_eve_map_path):
        """EVE score for 1433G_HUMAN R4A matches known value."""
        from stability_scorer import build_eve_index, load_eve_entry_name_map, lookup_eve_score
        acc_to_entry = load_eve_entry_name_map(test_eve_map_path)
        index = build_eve_index(test_eve_dir, frozenset({'P61981'}), acc_to_entry)
        result = lookup_eve_score(index, 'P61981', 'R', 4, 'A')
        assert abs(result['eve_score'] - 0.7727140175285468) < 1e-12
        assert result['eve_class'] == 'Pathogenic'
        assert abs(result['eve_uncertainty'] - 0.5359761101025369) < 1e-12

    def test_position_1_no_score_regression(self, test_eve_dir, test_eve_map_path):
        """Position 1 variants have no EVE scores (initiator methionine)."""
        from stability_scorer import build_eve_index, load_eve_entry_name_map, lookup_eve_score
        acc_to_entry = load_eve_entry_name_map(test_eve_map_path)
        index = build_eve_index(test_eve_dir, frozenset({'P61981'}), acc_to_entry)
        result = lookup_eve_score(index, 'P61981', 'M', 1, 'A')
        assert result is not None  # row exists
        assert result['eve_score'] is None  # but no score
