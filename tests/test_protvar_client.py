#!/usr/bin/env python3
"""
Tests for protvar_client.py — ProtVar API cross-validation module.

All tests use offline cached/mocked data. No live API calls are made.

Test data in tests/offline_test_data/databases/protvar_responses/:
    score_P61981_4.json         — AlphaMissense/EVE/ESM/CONSERV scores
    interaction_P61981_4.json   — 3 interaction partners (position at interface)
    foldx_P61981_4.json         — FoldX ΔΔG for 9 substitutions
    score_P24534_81.json        — Scores for second test protein
    interaction_P24534_81.json  — Empty list (no interactions)
    foldx_P24534_81.json        — FoldX ΔΔG for 5 substitutions
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from protvar_client import (
    # Constants
    PROTVAR_API_BASE_URL,
    PROTVAR_API_RATE_LIMIT_PAUSE,
    PROTVAR_API_MAX_RETRIES,
    PROTVAR_API_BACKOFF_FACTOR,
    PROTVAR_API_TIMEOUT,
    PROTVAR_API_DEFAULT_CACHE_DIR,
    PROTVAR_DETAILS_DISPLAY_LIMIT,
    FOLDX_DESTABILISING_THRESHOLD,
    CSV_FIELDNAMES_PROTVAR,
    _VARIANT_DETAIL_PATTERN,
    # Exception
    ProtVarAPIError,
    # Internal helpers
    _resolve_cache_dir,
    _cache_key,
    _read_cache,
    _write_cache,
    # Public query functions
    get_scores,
    get_interactions,
    get_foldx,
    # Index building
    build_protvar_index,
    # Score extraction
    extract_am_score,
    extract_am_class,
    extract_foldx_ddg,
    check_protvar_interface,
    compute_interface_agreement,
    # Annotation
    _parse_variant_details_for_protvar,
    format_protvar_details,
    _score_chain_variants_protvar,
    annotate_results_with_protvar,
    # CLI
    build_argument_parser,
)

# ── Known Test Values ────────────────────────────────────────────────

# P61981 position 4 (R → A)
TEST_ACC_WITH_INTERACTIONS = 'P61981'
TEST_POS_WITH_INTERACTIONS = 4
TEST_AM_SCORE_R4A = 0.9137  # AlphaMissense score for R→A
TEST_AM_CLASS_R4A = 'PATHOGENIC'
TEST_FOLDX_DDG_R4A = 3.17722
TEST_N_INTERACTIONS_P61981_4 = 3  # 3 partners in test data

# P24534 position 81 (no interactions)
TEST_ACC_NO_INTERACTIONS = 'P24534'
TEST_POS_NO_INTERACTIONS = 81
TEST_AM_SCORE_P81A = 0.1611  # AlphaMissense score for P→A at pos 81
TEST_FOLDX_DDG_P81A = 2.39718


# ── Helper: Load Test Data ───────────────────────────────────────────

def _load_test_json(responses_dir: Path, filename: str) -> list:
    """Load a test JSON file and return its content."""
    path = responses_dir / filename
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _make_result_row(
    protein_a: str = 'P61981',
    protein_b: str = 'P24534',
    details_a: str = '',
    details_b: str = '',
) -> dict:
    """Create a minimal result row for annotation testing."""
    return {
        'protein_a': protein_a,
        'protein_b': protein_b,
        'variant_details_a': details_a,
        'variant_details_b': details_b,
    }


# ── Test Class 1: Constants ──────────────────────────────────────────

@pytest.mark.protvar
class TestProtVarConstants:
    """Verify module constants are set correctly."""

    def test_api_base_url_format(self):
        """Base URL starts with https and points to EBI ProtVar."""
        assert PROTVAR_API_BASE_URL.startswith("https://")
        assert "ProtVar" in PROTVAR_API_BASE_URL

    def test_csv_column_count(self):
        """CSV fieldnames has exactly 8 columns."""
        assert len(CSV_FIELDNAMES_PROTVAR) == 8

    def test_rate_limit_positive(self):
        """Rate limit pause is a positive number."""
        assert PROTVAR_API_RATE_LIMIT_PAUSE > 0

    def test_foldx_threshold(self):
        """FoldX destabilisation threshold is 1.6 kcal/mol."""
        assert FOLDX_DESTABILISING_THRESHOLD == 1.6


# ── Test Class 2: Cache Helpers ──────────────────────────────────────

@pytest.mark.protvar
class TestCacheHelpers:
    """Test internal caching functions."""

    def test_resolve_cache_dir_none_returns_default(self):
        """None → default cache directory."""
        result = _resolve_cache_dir(None)
        assert result == PROTVAR_API_DEFAULT_CACHE_DIR

    def test_resolve_cache_dir_false_returns_none(self):
        """False → None (disabled)."""
        result = _resolve_cache_dir(False)
        assert result is None

    def test_resolve_cache_dir_string_returns_path(self):
        """String → Path."""
        result = _resolve_cache_dir("/tmp/custom_cache")
        assert result == Path("/tmp/custom_cache")

    def test_cache_key_deterministic(self):
        """Same inputs produce same key."""
        key1 = _cache_key("score", "P61981", 4)
        key2 = _cache_key("score", "P61981", 4)
        assert key1 == key2

    def test_cache_key_different_for_different_inputs(self):
        """Different inputs produce different keys."""
        key1 = _cache_key("score", "P61981", 4)
        key2 = _cache_key("score", "P61981", 5)
        key3 = _cache_key("prediction/interaction", "P61981", 4)
        assert key1 != key2
        assert key1 != key3

    def test_cache_roundtrip(self, tmp_path):
        """Write + read returns original data."""
        test_data = [{"name": "AM", "mt": "A", "amPathogenicity": 0.91}]
        key = _cache_key("score", "P61981", 4)
        _write_cache(tmp_path, key, "score", test_data)
        result = _read_cache(tmp_path, key)
        assert result == test_data

    def test_cache_miss_returns_none(self, tmp_path):
        """Reading nonexistent cache returns None."""
        result = _read_cache(tmp_path, "nonexistent_key_abc123")
        assert result is None

    def test_write_cache_creates_directory(self, tmp_path):
        """Cache write creates subdirectory if needed."""
        nested = tmp_path / "sub" / "dir"
        _write_cache(nested, "key123", "score", [])
        assert (nested / "key123.json").exists()


# ── Test Class 3: Get Scores ─────────────────────────────────────────

@pytest.mark.protvar
class TestGetScores:
    """Test get_scores() with cached data."""

    def test_cached_scores_returns_data(self, test_protvar_responses_dir, tmp_path):
        """Returns score list from pre-seeded cache."""
        # Seed cache
        data = _load_test_json(test_protvar_responses_dir, "score_P61981_4.json")
        key = _cache_key("score", "P61981", 4)
        _write_cache(tmp_path, key, "score", data)

        result = get_scores("P61981", 4, cache_dir=str(tmp_path))
        assert isinstance(result, list)
        assert len(result) > 0

    def test_am_entries_present(self, test_protvar_responses_dir, tmp_path):
        """Score data contains AlphaMissense entries."""
        data = _load_test_json(test_protvar_responses_dir, "score_P61981_4.json")
        key = _cache_key("score", "P61981", 4)
        _write_cache(tmp_path, key, "score", data)

        result = get_scores("P61981", 4, cache_dir=str(tmp_path))
        am_entries = [s for s in result if s.get('name') == 'AM']
        assert len(am_entries) > 0

    def test_eve_entries_present(self, test_protvar_responses_dir, tmp_path):
        """Score data contains EVE entries."""
        data = _load_test_json(test_protvar_responses_dir, "score_P61981_4.json")
        key = _cache_key("score", "P61981", 4)
        _write_cache(tmp_path, key, "score", data)

        result = get_scores("P61981", 4, cache_dir=str(tmp_path))
        eve_entries = [s for s in result if s.get('name') == 'EVE']
        assert len(eve_entries) > 0

    def test_conserv_entry_present(self, test_protvar_responses_dir, tmp_path):
        """Score data contains CONSERV entry."""
        data = _load_test_json(test_protvar_responses_dir, "score_P61981_4.json")
        key = _cache_key("score", "P61981", 4)
        _write_cache(tmp_path, key, "score", data)

        result = get_scores("P61981", 4, cache_dir=str(tmp_path))
        conserv = [s for s in result if s.get('name') == 'CONSERV']
        assert len(conserv) == 1
        assert conserv[0]['score'] == 0.74


# ── Test Class 4: Get Interactions ───────────────────────────────────

@pytest.mark.protvar
class TestGetInteractions:
    """Test get_interactions() with cached data."""

    def test_cached_interactions_returns_data(self, test_protvar_responses_dir, tmp_path):
        """Returns interaction list from pre-seeded cache."""
        data = _load_test_json(test_protvar_responses_dir, "interaction_P61981_4.json")
        key = _cache_key("prediction/interaction", "P61981", 4)
        _write_cache(tmp_path, key, "prediction/interaction", data)

        result = get_interactions("P61981", 4, cache_dir=str(tmp_path))
        assert isinstance(result, list)
        assert len(result) == TEST_N_INTERACTIONS_P61981_4

    def test_interaction_has_pdockq(self, test_protvar_responses_dir, tmp_path):
        """Each interaction has a pDockQ score."""
        data = _load_test_json(test_protvar_responses_dir, "interaction_P61981_4.json")
        key = _cache_key("prediction/interaction", "P61981", 4)
        _write_cache(tmp_path, key, "prediction/interaction", data)

        result = get_interactions("P61981", 4, cache_dir=str(tmp_path))
        for entry in result:
            assert 'pdockq' in entry
            assert isinstance(entry['pdockq'], (int, float))

    def test_empty_interactions(self, test_protvar_responses_dir, tmp_path):
        """Returns empty list when no interactions exist."""
        data = _load_test_json(test_protvar_responses_dir, "interaction_P24534_81.json")
        key = _cache_key("prediction/interaction", "P24534", 81)
        _write_cache(tmp_path, key, "prediction/interaction", data)

        result = get_interactions("P24534", 81, cache_dir=str(tmp_path))
        assert result == []

    def test_interaction_has_residue_lists(self, test_protvar_responses_dir, tmp_path):
        """Interactions include aresidues and bresidues lists."""
        data = _load_test_json(test_protvar_responses_dir, "interaction_P61981_4.json")
        key = _cache_key("prediction/interaction", "P61981", 4)
        _write_cache(tmp_path, key, "prediction/interaction", data)

        result = get_interactions("P61981", 4, cache_dir=str(tmp_path))
        for entry in result:
            assert 'aresidues' in entry
            assert 'bresidues' in entry
            assert isinstance(entry['aresidues'], list)


# ── Test Class 5: Get FoldX ─────────────────────────────────────────

@pytest.mark.protvar
class TestGetFoldx:
    """Test get_foldx() with cached data."""

    def test_cached_foldx_returns_data(self, test_protvar_responses_dir, tmp_path):
        """Returns FoldX list from pre-seeded cache."""
        data = _load_test_json(test_protvar_responses_dir, "foldx_P61981_4.json")
        key = _cache_key("prediction/foldx", "P61981", 4)
        _write_cache(tmp_path, key, "prediction/foldx", data)

        result = get_foldx("P61981", 4, cache_dir=str(tmp_path))
        assert isinstance(result, list)
        assert len(result) > 0

    def test_foldx_has_ddg(self, test_protvar_responses_dir, tmp_path):
        """Each FoldX entry has a foldxDdg value."""
        data = _load_test_json(test_protvar_responses_dir, "foldx_P61981_4.json")
        key = _cache_key("prediction/foldx", "P61981", 4)
        _write_cache(tmp_path, key, "prediction/foldx", data)

        result = get_foldx("P61981", 4, cache_dir=str(tmp_path))
        for entry in result:
            assert 'foldxDdg' in entry
            assert isinstance(entry['foldxDdg'], (int, float))

    def test_foldx_has_wildtype(self, test_protvar_responses_dir, tmp_path):
        """Each FoldX entry has wildType and mutatedType."""
        data = _load_test_json(test_protvar_responses_dir, "foldx_P61981_4.json")
        key = _cache_key("prediction/foldx", "P61981", 4)
        _write_cache(tmp_path, key, "prediction/foldx", data)

        result = get_foldx("P61981", 4, cache_dir=str(tmp_path))
        for entry in result:
            assert 'wildType' in entry
            assert 'mutatedType' in entry

    def test_foldx_regression_r4a(self, test_protvar_responses_dir, tmp_path):
        """FoldX ΔΔG for R4A matches known value."""
        data = _load_test_json(test_protvar_responses_dir, "foldx_P61981_4.json")
        key = _cache_key("prediction/foldx", "P61981", 4)
        _write_cache(tmp_path, key, "prediction/foldx", data)

        result = get_foldx("P61981", 4, cache_dir=str(tmp_path))
        r4a = [f for f in result if f.get('mutatedType') == 'A']
        assert len(r4a) == 1
        assert abs(r4a[0]['foldxDdg'] - TEST_FOLDX_DDG_R4A) < 0.001


# ── Test Class 6: Score Extraction Helpers ───────────────────────────

@pytest.mark.protvar
class TestExtractHelpers:
    """Test score extraction helper functions."""

    def test_extract_am_score_found(self, test_protvar_responses_dir):
        """AM score extracted for known substitution."""
        scores = _load_test_json(test_protvar_responses_dir, "score_P61981_4.json")
        result = extract_am_score(scores, 'A')
        assert result == TEST_AM_SCORE_R4A

    def test_extract_am_score_not_found(self, test_protvar_responses_dir):
        """AM score returns None for non-existent substitution."""
        scores = _load_test_json(test_protvar_responses_dir, "score_P61981_4.json")
        result = extract_am_score(scores, 'R')  # R is the wildtype, not a mutation
        assert result is None

    def test_extract_am_class_found(self, test_protvar_responses_dir):
        """AM class extracted for known substitution."""
        scores = _load_test_json(test_protvar_responses_dir, "score_P61981_4.json")
        result = extract_am_class(scores, 'A')
        assert result == TEST_AM_CLASS_R4A

    def test_extract_am_score_empty_list(self):
        """AM score from empty list returns None."""
        assert extract_am_score([], 'A') is None

    def test_extract_foldx_ddg_found(self, test_protvar_responses_dir):
        """FoldX ΔΔG extracted for known substitution."""
        foldx = _load_test_json(test_protvar_responses_dir, "foldx_P61981_4.json")
        result = extract_foldx_ddg(foldx, 'A')
        assert abs(result - TEST_FOLDX_DDG_R4A) < 0.001

    def test_extract_foldx_ddg_not_found(self, test_protvar_responses_dir):
        """FoldX ΔΔG returns None for non-existent substitution."""
        foldx = _load_test_json(test_protvar_responses_dir, "foldx_P61981_4.json")
        result = extract_foldx_ddg(foldx, 'Z')
        assert result is None

    def test_extract_foldx_ddg_empty_list(self):
        """FoldX ΔΔG from empty list returns None."""
        assert extract_foldx_ddg([], 'A') is None


# ── Test Class 7: Interface Check ────────────────────────────────────

@pytest.mark.protvar
class TestInterfaceCheck:
    """Test check_protvar_interface() function."""

    def test_position_at_interface(self, test_protvar_responses_dir):
        """Position 4 of P61981 is at interface (in bresidues)."""
        interactions = _load_test_json(test_protvar_responses_dir,
                                       "interaction_P61981_4.json")
        assert check_protvar_interface(interactions, "P61981", 4) is True

    def test_position_not_at_interface(self, test_protvar_responses_dir):
        """Position 999 is not at any interface."""
        interactions = _load_test_json(test_protvar_responses_dir,
                                       "interaction_P61981_4.json")
        assert check_protvar_interface(interactions, "P61981", 999) is False

    def test_no_interactions_returns_false(self):
        """Empty interaction list → not at interface."""
        assert check_protvar_interface([], "P61981", 4) is False

    def test_isoform_accession_matching(self, test_protvar_responses_dir):
        """Isoform stripping allows matching (P61981-2 matches P61981)."""
        interactions = _load_test_json(test_protvar_responses_dir,
                                       "interaction_P61981_4.json")
        # P61981-2 should still match entries with P61981
        assert check_protvar_interface(interactions, "P61981-2", 4) is True


# ── Test Class 8: Interface Agreement ────────────────────────────────

@pytest.mark.protvar
class TestInterfaceAgreement:
    """Test compute_interface_agreement() function."""

    def test_full_agreement(self, test_protvar_responses_dir):
        """All toolkit interface positions confirmed by ProtVar → 1.0."""
        interactions = _load_test_json(test_protvar_responses_dir,
                                       "interaction_P61981_4.json")
        protvar_data = {4: {'interactions': interactions}}
        result = compute_interface_agreement({4}, protvar_data, "P61981")
        assert result == 1.0

    def test_no_agreement(self, test_protvar_responses_dir):
        """ProtVar has empty interactions → 0.0."""
        protvar_data = {4: {'interactions': []}}
        result = compute_interface_agreement({4}, protvar_data, "P61981")
        assert result == 0.0

    def test_empty_toolkit_positions(self):
        """No toolkit interface positions → empty string."""
        result = compute_interface_agreement(set(), {}, "P61981")
        assert result == ''

    def test_no_protvar_data(self):
        """Positions not in ProtVar index → empty string."""
        result = compute_interface_agreement({100, 200}, {}, "P61981")
        assert result == ''

    def test_partial_agreement(self, test_protvar_responses_dir):
        """Mix of confirmed and non-confirmed positions."""
        interactions = _load_test_json(test_protvar_responses_dir,
                                       "interaction_P61981_4.json")
        # Position 4 is at interface, position 999 is not
        protvar_data = {
            4: {'interactions': interactions},
            999: {'interactions': interactions},  # reuse but 999 not in residue lists
        }
        result = compute_interface_agreement({4, 999}, protvar_data, "P61981")
        assert result == 0.5


# ── Test Class 9: Parse Variant Details ──────────────────────────────

@pytest.mark.protvar
class TestParseVariantDetails:
    """Test _parse_variant_details_for_protvar() function."""

    def test_single_variant(self):
        """Parses single variant correctly."""
        result = _parse_variant_details_for_protvar("R4A:interface_core:pathogenic")
        assert result == [('R', 4, 'A')]

    def test_multiple_variants(self):
        """Parses multiple pipe-separated variants."""
        result = _parse_variant_details_for_protvar(
            "R4A:interface_core:pathogenic|K10E:surface_non_interface:-"
        )
        assert result == [('R', 4, 'A'), ('K', 10, 'E')]

    def test_truncation_indicator_skipped(self):
        """Truncation indicator ...(+N more) is ignored."""
        result = _parse_variant_details_for_protvar(
            "R4A:interface_core:pathogenic|...(+5 more)"
        )
        assert result == [('R', 4, 'A')]

    def test_empty_string(self):
        """Empty string returns empty list."""
        assert _parse_variant_details_for_protvar('') == []

    def test_stop_codon_variant(self):
        """Stop codon variant with * is parsed."""
        result = _parse_variant_details_for_protvar("R4*:interface_core:pathogenic")
        assert result == [('R', 4, '*')]


# ── Test Class 10: Format Details ────────────────────────────────────

@pytest.mark.protvar
class TestFormatDetails:
    """Test format_protvar_details() function."""

    def test_empty_list(self):
        """Empty input returns empty string."""
        assert format_protvar_details([]) == ''

    def test_single_variant_format(self):
        """Single variant formatted correctly."""
        variants = [{'ref_aa': 'R', 'position': 4, 'alt_aa': 'A',
                      'am_score': 0.9137, 'am_class': 'PATHOGENIC',
                      'foldx_ddg': 3.177}]
        result = format_protvar_details(variants)
        assert result == 'R4A:am=0.91:PATHOGENIC:foldx=3.18'

    def test_none_scores_show_dash(self):
        """None scores display as dash."""
        variants = [{'ref_aa': 'K', 'position': 10, 'alt_aa': 'E',
                      'am_score': None, 'am_class': None,
                      'foldx_ddg': None}]
        result = format_protvar_details(variants)
        assert result == 'K10E:am=-:-:foldx=-'

    def test_truncation_with_limit(self):
        """Excess variants show truncation indicator."""
        variants = [
            {'ref_aa': 'R', 'position': i, 'alt_aa': 'A',
             'am_score': 0.5, 'am_class': 'AMBIGUOUS', 'foldx_ddg': 1.0}
            for i in range(1, 6)
        ]
        result = format_protvar_details(variants, limit=3)
        assert '...(+2 more)' in result
        assert result.count('|') == 3  # 3 entries + truncation


# ── Test Class 11: Annotation ────────────────────────────────────────

@pytest.mark.protvar
class TestAnnotation:
    """Test annotate_results_with_protvar() and _score_chain_variants_protvar()."""

    def _build_test_index(self, responses_dir):
        """Build a protvar_index from test data."""
        return {
            'P61981': {
                4: {
                    'scores': _load_test_json(responses_dir, "score_P61981_4.json"),
                    'interactions': _load_test_json(responses_dir, "interaction_P61981_4.json"),
                    'foldx': _load_test_json(responses_dir, "foldx_P61981_4.json"),
                },
            },
            'P24534': {
                81: {
                    'scores': _load_test_json(responses_dir, "score_P24534_81.json"),
                    'interactions': _load_test_json(responses_dir, "interaction_P24534_81.json"),
                    'foldx': _load_test_json(responses_dir, "foldx_P24534_81.json"),
                },
            },
        }

    def test_adds_all_columns(self, test_protvar_responses_dir):
        """All 8 CSV columns are set after annotation."""
        idx = self._build_test_index(test_protvar_responses_dir)
        results = [_make_result_row(
            details_a='R4A:interface_core:pathogenic',
            details_b='P81A:surface_non_interface:-',
        )]
        annotate_results_with_protvar(results, idx)
        for col in CSV_FIELDNAMES_PROTVAR:
            assert col in results[0], f"Missing column: {col}"

    def test_am_mean_computed(self, test_protvar_responses_dir):
        """AlphaMissense mean is computed correctly for chain A."""
        idx = self._build_test_index(test_protvar_responses_dir)
        results = [_make_result_row(details_a='R4A:interface_core:pathogenic')]
        annotate_results_with_protvar(results, idx)
        # Single variant R4A, AM score = 0.9137
        assert results[0]['protvar_am_mean_a'] == TEST_AM_SCORE_R4A

    def test_foldx_mean_computed(self, test_protvar_responses_dir):
        """FoldX mean ΔΔG is computed correctly for chain A."""
        idx = self._build_test_index(test_protvar_responses_dir)
        results = [_make_result_row(details_a='R4A:interface_core:pathogenic')]
        annotate_results_with_protvar(results, idx)
        assert abs(results[0]['protvar_foldx_mean_a'] - TEST_FOLDX_DDG_R4A) < 0.001

    def test_no_variants_produces_empty(self, test_protvar_responses_dir):
        """No variant details → empty columns."""
        idx = self._build_test_index(test_protvar_responses_dir)
        results = [_make_result_row()]
        annotate_results_with_protvar(results, idx)
        assert results[0]['protvar_am_mean_a'] == ''
        assert results[0]['protvar_details_a'] == ''

    def test_empty_index_produces_empty_details(self):
        """Empty ProtVar index → no scores but columns still set."""
        results = [_make_result_row(details_a='R4A:interface_core:pathogenic')]
        annotate_results_with_protvar(results, {})
        assert results[0]['protvar_am_mean_a'] == ''
        assert results[0]['protvar_details_a'] != ''  # formatted but with dashes

    def test_both_chains_annotated(self, test_protvar_responses_dir):
        """Both chain A and chain B are annotated independently."""
        idx = self._build_test_index(test_protvar_responses_dir)
        results = [_make_result_row(
            details_a='R4A:interface_core:pathogenic',
            details_b='P81A:surface_non_interface:-',
        )]
        annotate_results_with_protvar(results, idx)
        # Chain A has interactions, chain B does not
        assert results[0]['protvar_am_mean_a'] != ''
        assert results[0]['protvar_am_mean_b'] != ''

    def test_modifies_in_place(self, test_protvar_responses_dir):
        """Results list is modified in-place, not copied."""
        idx = self._build_test_index(test_protvar_responses_dir)
        results = [_make_result_row(details_a='R4A:interface_core:pathogenic')]
        original_id = id(results[0])
        annotate_results_with_protvar(results, idx)
        assert id(results[0]) == original_id

    def test_verbose_prints_stats(self, test_protvar_responses_dir, capsys):
        """Verbose mode prints annotation statistics."""
        idx = self._build_test_index(test_protvar_responses_dir)
        results = [_make_result_row(details_a='R4A:interface_core:pathogenic')]
        annotate_results_with_protvar(results, idx, verbose=True)
        captured = capsys.readouterr()
        assert 'ProtVar' in captured.err
        assert 'annotated' in captured.err


# ── Test Class 12: CSV Fieldnames ────────────────────────────────────

@pytest.mark.protvar
class TestCSVFieldnames:
    """Test CSV column configuration."""

    def test_column_count(self):
        """Exactly 8 ProtVar columns."""
        assert len(CSV_FIELDNAMES_PROTVAR) == 8

    def test_all_columns_have_suffix(self):
        """All columns end with _a or _b."""
        for col in CSV_FIELDNAMES_PROTVAR:
            assert col.endswith('_a') or col.endswith('_b'), \
                f"Column {col} missing _a/_b suffix"

    def test_no_duplicate_columns(self):
        """No duplicate column names."""
        assert len(CSV_FIELDNAMES_PROTVAR) == len(set(CSV_FIELDNAMES_PROTVAR))


# ── Test Class 13: CLI ───────────────────────────────────────────────

@pytest.mark.protvar
@pytest.mark.cli
class TestCLI:
    """Test standalone CLI argument parsing."""

    def test_parser_construction(self):
        """Parser builds without error."""
        parser = build_argument_parser()
        assert parser is not None

    def test_summary_subcommand(self):
        """Summary subcommand is recognised."""
        parser = build_argument_parser()
        args = parser.parse_args(["summary"])
        assert args.command == "summary"

    def test_lookup_subcommand(self):
        """Lookup subcommand parses protein and position."""
        parser = build_argument_parser()
        args = parser.parse_args(["lookup", "--protein", "P61981", "--position", "4"])
        assert args.command == "lookup"
        assert args.protein == "P61981"
        assert args.position == 4

    def test_custom_cache_dir(self):
        """--cache-dir is parsed correctly."""
        parser = build_argument_parser()
        args = parser.parse_args(["--cache-dir", "/tmp/my_cache", "summary"])
        assert args.cache_dir == "/tmp/my_cache"


# ── Test Class 14: Regression Values ────────────────────────────────

@pytest.mark.protvar
@pytest.mark.regression
class TestRegressionValues:
    """Regression tests with known reference values."""

    def test_am_score_r4a_regression(self, test_protvar_responses_dir):
        """AlphaMissense score for P61981 R4A matches known value."""
        scores = _load_test_json(test_protvar_responses_dir, "score_P61981_4.json")
        am = extract_am_score(scores, 'A')
        assert am == TEST_AM_SCORE_R4A

    def test_foldx_ddg_r4a_regression(self, test_protvar_responses_dir):
        """FoldX ΔΔG for P61981 R4A matches known value."""
        foldx = _load_test_json(test_protvar_responses_dir, "foldx_P61981_4.json")
        ddg = extract_foldx_ddg(foldx, 'A')
        assert abs(ddg - TEST_FOLDX_DDG_R4A) < 0.001

    def test_interface_check_regression(self, test_protvar_responses_dir):
        """P61981 position 4 is confirmed at interface by ProtVar."""
        interactions = _load_test_json(test_protvar_responses_dir,
                                       "interaction_P61981_4.json")
        assert check_protvar_interface(interactions, "P61981", 4) is True

    def test_no_interaction_regression(self, test_protvar_responses_dir):
        """P24534 position 81 has no ProtVar interactions."""
        interactions = _load_test_json(test_protvar_responses_dir,
                                       "interaction_P24534_81.json")
        assert interactions == []
        assert check_protvar_interface(interactions, "P24534", 81) is False
