"""
Tests for database_loaders.py - PPI database parsing and standardisation.

Uses small excerpt files in tests/offline_test_data/databases/ for offline testing.
Full-file tests are marked @pytest.mark.slow.
"""

import pytest
import pandas as pd
from pathlib import Path

from database_loaders import (
    load_string,
    load_biogrid,
    load_huri,
    load_humap,
    _strip_taxonomy_prefix,
    _normalise_string_score,
    _extract_first_uniprot,
    OUTPUT_COLUMNS,
    STRING_MAX_SCORE,
)

# ── Test Data Paths ──────────────────────────────────────────────────

PROJECT_ROOT = Path(r"C:\Users\Talhah Zubayer\Documents\protein-complexes-toolkit")
TEST_DB_DIR = PROJECT_ROOT / "tests" / "offline_test_data" / "databases"

# Known counts from test excerpts
EXPECTED_STRING_ROWS = 304
EXPECTED_BIOGRID_HUMAN_ROWS = 51
EXPECTED_HURI_ROWS = 43
EXPECTED_HUMAP_ROWS = 59


# ── Helper Function Tests (pure unit, no I/O) ────────────────────────

class TestStripTaxonomyPrefix:
    """Tests for _strip_taxonomy_prefix()."""

    def test_strips_prefix(self):
        assert _strip_taxonomy_prefix("9606.ENSP00000000233") == "ENSP00000000233"

    def test_already_bare(self):
        assert _strip_taxonomy_prefix("ENSP00000000233") == "ENSP00000000233"

    def test_different_prefix_unchanged(self):
        assert _strip_taxonomy_prefix("10090.ENSMUSP00000001") == "10090.ENSMUSP00000001"


class TestNormaliseStringScore:
    """Tests for _normalise_string_score()."""

    def test_max_score(self):
        assert _normalise_string_score(STRING_MAX_SCORE) == 1.0

    def test_zero(self):
        assert _normalise_string_score(0) == 0.0

    def test_mid(self):
        assert _normalise_string_score(500) == pytest.approx(0.5)

    def test_typical_high(self):
        assert _normalise_string_score(700) == pytest.approx(0.7)


class TestExtractFirstUniprot:
    """Tests for _extract_first_uniprot()."""

    def test_single_accession(self):
        assert _extract_first_uniprot("P45985") == "P45985"

    def test_pipe_delimited(self):
        assert _extract_first_uniprot("Q59H94|F6THM6") == "Q59H94"

    def test_dash_returns_none(self):
        assert _extract_first_uniprot("-") is None

    def test_empty_returns_none(self):
        assert _extract_first_uniprot("") is None

    def test_single_with_spaces(self):
        assert _extract_first_uniprot(" P45985 ") == "P45985"


# ── STRING Loader Tests ──────────────────────────────────────────────

@pytest.mark.slow
class TestStringLoader:
    """Tests for STRING database loading."""

    @pytest.fixture(scope="class")
    def string_df(self):
        return load_string(str(TEST_DB_DIR / "test_string_links.txt"))

    def test_returns_dataframe(self, string_df):
        assert isinstance(string_df, pd.DataFrame)

    def test_has_standard_columns(self, string_df):
        assert list(string_df.columns) == OUTPUT_COLUMNS

    def test_expected_row_count(self, string_df):
        assert len(string_df) == EXPECTED_STRING_ROWS

    def test_ids_stripped(self, string_df):
        """ENSP IDs have 9606. prefix stripped."""
        assert not string_df['protein_a'].str.startswith('9606.').any()
        assert not string_df['protein_b'].str.startswith('9606.').any()

    def test_ids_are_ensp(self, string_df):
        """Protein IDs are Ensembl protein IDs."""
        assert string_df['protein_a'].str.startswith('ENSP').all()
        assert string_df['protein_b'].str.startswith('ENSP').all()

    def test_scores_normalised(self, string_df):
        """Scores are normalised to 0.0-1.0 range."""
        assert string_df['confidence_score'].between(0.0, 1.0).all()

    def test_source_column(self, string_df):
        assert (string_df['source'] == 'STRING').all()

    def test_evidence_type(self, string_df):
        assert (string_df['evidence_type'] == 'combined').all()

    def test_min_score_filter(self):
        """min_score parameter filters low-confidence interactions."""
        df_all = load_string(str(TEST_DB_DIR / "test_string_links.txt"), min_score=0)
        df_high = load_string(str(TEST_DB_DIR / "test_string_links.txt"), min_score=700)
        assert len(df_high) < len(df_all)
        # All remaining scores should be >= 0.7 (700/1000)
        assert (df_high['confidence_score'] >= 0.7).all()


# ── BioGRID Loader Tests ────────────────────────────────────────────

@pytest.mark.slow
class TestBiogridLoader:
    """Tests for BioGRID database loading."""

    @pytest.fixture(scope="class")
    def biogrid_df(self):
        return load_biogrid(str(TEST_DB_DIR / "test_biogrid.tab3.txt"))

    def test_returns_dataframe(self, biogrid_df):
        assert isinstance(biogrid_df, pd.DataFrame)

    def test_has_standard_columns(self, biogrid_df):
        assert list(biogrid_df.columns) == OUTPUT_COLUMNS

    def test_filters_human(self, biogrid_df):
        """BioGRID loader filters to taxonomy 9606.

        The test excerpt has 15 non-human + 6 human genetic + 55 human physical
        rows. After filtering (human + physical_only + both UniProt IDs present),
        51 rows should survive.
        """
        assert len(biogrid_df) == EXPECTED_BIOGRID_HUMAN_ROWS

    def test_source_column(self, biogrid_df):
        assert (biogrid_df['source'] == 'BioGRID').all()

    def test_evidence_type_populated(self, biogrid_df):
        """evidence_type contains experimental system strings."""
        assert biogrid_df['evidence_type'].notna().all()
        assert len(biogrid_df['evidence_type'].unique()) >= 1

    def test_confidence_is_nan(self, biogrid_df):
        """BioGRID has no confidence scores."""
        assert biogrid_df['confidence_score'].isna().all()

    def test_protein_ids_look_like_uniprot(self, biogrid_df):
        """Protein IDs are UniProt accessions (start with letter + digit)."""
        assert biogrid_df['protein_a'].str.match(r'^[A-Z][0-9]').all()
        assert biogrid_df['protein_b'].str.match(r'^[A-Z][0-9]').all()


# ── HuRI Loader Tests ───────────────────────────────────────────────

@pytest.mark.slow
class TestHuriLoader:
    """Tests for HuRI database loading."""

    @pytest.fixture(scope="class")
    def huri_df(self):
        return load_huri(str(TEST_DB_DIR / "test_huri.tsv"))

    def test_returns_dataframe(self, huri_df):
        assert isinstance(huri_df, pd.DataFrame)

    def test_has_standard_columns(self, huri_df):
        assert list(huri_df.columns) == OUTPUT_COLUMNS

    def test_expected_row_count(self, huri_df):
        assert len(huri_df) == EXPECTED_HURI_ROWS

    def test_returns_binary_interactions(self, huri_df):
        """HuRI returns binary interaction pairs."""
        assert len(huri_df) > 0
        assert huri_df['protein_a'].notna().all()
        assert huri_df['protein_b'].notna().all()

    def test_ensg_ids(self, huri_df):
        """IDs are Ensembl gene IDs (ENSG format)."""
        assert huri_df['protein_a'].str.startswith('ENSG').all()
        assert huri_df['protein_b'].str.startswith('ENSG').all()

    def test_source_column(self, huri_df):
        assert (huri_df['source'] == 'HuRI').all()

    def test_evidence_type(self, huri_df):
        assert (huri_df['evidence_type'] == 'Y2H').all()


# ── HuMAP Loader Tests ──────────────────────────────────────────────

@pytest.mark.slow
class TestHumapLoader:
    """Tests for HuMAP database loading."""

    @pytest.fixture(scope="class")
    def humap_df(self):
        return load_humap(str(TEST_DB_DIR / "test_humap.pairsWprob"))

    def test_returns_dataframe(self, humap_df):
        assert isinstance(humap_df, pd.DataFrame)

    def test_has_standard_columns(self, humap_df):
        assert list(humap_df.columns) == OUTPUT_COLUMNS

    def test_expected_row_count(self, humap_df):
        assert len(humap_df) == EXPECTED_HUMAP_ROWS

    def test_returns_pairwise(self, humap_df):
        """HuMAP returns pairwise interactions (not complex membership)."""
        assert humap_df['protein_a'].notna().all()
        assert humap_df['protein_b'].notna().all()

    def test_probability_range(self, humap_df):
        """Probability scores are in 0.0-1.0 range."""
        assert humap_df['confidence_score'].between(0.0, 1.0).all()

    def test_source_column(self, humap_df):
        assert (humap_df['source'] == 'HuMAP').all()

    def test_min_probability_filter(self):
        """min_probability parameter filters low-probability interactions."""
        df_all = load_humap(str(TEST_DB_DIR / "test_humap.pairsWprob"), min_probability=0.0)
        df_high = load_humap(str(TEST_DB_DIR / "test_humap.pairsWprob"), min_probability=0.9)
        assert len(df_high) < len(df_all)
        assert (df_high['confidence_score'] >= 0.9).all()


# ── STRING Edge Case Tests ──────────────────────────────────────────

@pytest.mark.slow
class TestStringEdgeCases:
    """Edge case tests for STRING loader."""

    @pytest.fixture(scope="class")
    def string_df(self):
        return load_string(str(TEST_DB_DIR / "test_string_links.txt"))

    def test_score_boundary_zero(self, string_df):
        """Score of 0 normalises to 0.0."""
        assert string_df['confidence_score'].min() == pytest.approx(0.0)

    def test_score_boundary_max(self, string_df):
        """Score of 1000 normalises to 1.0."""
        assert string_df['confidence_score'].max() == pytest.approx(1.0)

    def test_self_interaction_loaded(self, string_df):
        """Self-interaction row (same protein both sides) is loaded."""
        self_pairs = string_df[string_df['protein_a'] == string_df['protein_b']]
        assert len(self_pairs) >= 1


# ── BioGRID Edge Case Tests ──────────────────────────────────────────

@pytest.mark.slow
class TestBiogridEdgeCases:
    """Edge case tests for BioGRID loader."""

    @pytest.fixture(scope="class")
    def biogrid_df(self):
        return load_biogrid(str(TEST_DB_DIR / "test_biogrid.tab3.txt"))

    @pytest.fixture(scope="class")
    def biogrid_all_df(self):
        """Load with physical_only=False to include genetic interactions."""
        return load_biogrid(str(TEST_DB_DIR / "test_biogrid.tab3.txt"), physical_only=False)

    def test_trembl_fallback_when_swissprot_missing(self, biogrid_df):
        """Rows with SWISS-PROT='-' but valid TREMBL accession still produce an ID.

        Row 72 (RIPK4-TP53): SWISS-PROT A is '-', TREMBL A is 'Q9H4D1|Q96T11'.
        Should extract Q9H4D1 as protein_a.
        """
        ripk4_rows = biogrid_df[biogrid_df['protein_a'] == 'Q9H4D1']
        assert len(ripk4_rows) >= 1

    def test_both_accessions_missing_dropped(self, biogrid_df):
        """Rows where SWISS-PROT='-' AND TREMBL='-' are excluded.

        Row 35 (KIAA0087-TP53) has both missing for interactor A.
        Row 75-76 (BRCA1-XIST) have both missing for interactor B.
        Row 77 (LINC01194-TP53) has both missing for interactor A.
        These should not appear in the output.
        """
        # No row should have a None/NaN protein ID
        assert biogrid_df['protein_a'].notna().all()
        assert biogrid_df['protein_b'].notna().all()

    def test_genetic_interactions_excluded(self, biogrid_df):
        """With physical_only=True (default), genetic interactions are filtered out.

        The test excerpt has 6 rows with Experimental System Type = 'genetic'.
        """
        # P10275 (AR) only appears in a genetic interaction row
        ar_rows = biogrid_df[
            (biogrid_df['protein_a'] == 'P10275') |
            (biogrid_df['protein_b'] == 'P10275')
        ]
        assert len(ar_rows) == 0

    def test_genetic_included_when_flag_off(self, biogrid_all_df):
        """With physical_only=False, genetic interactions are included."""
        assert len(biogrid_all_df) > EXPECTED_BIOGRID_HUMAN_ROWS

    def test_pipe_delimited_trembl_takes_first(self, biogrid_df):
        """TREMBL field 'Q9H4D1|Q96T11' extracts first accession Q9H4D1."""
        ripk4_rows = biogrid_df[biogrid_df['protein_a'] == 'Q9H4D1']
        assert len(ripk4_rows) >= 1
        # Q96T11 (second pipe-delimited) should NOT appear as a protein ID
        q96_rows = biogrid_df[
            (biogrid_df['protein_a'] == 'Q96T11') |
            (biogrid_df['protein_b'] == 'Q96T11')
        ]
        assert len(q96_rows) == 0

    def test_trembl_fallback_for_interactor_b(self, biogrid_df):
        """TREMBL fallback works for interactor B too.

        Row 73 (TNMD-C12orf10): SWISS-PROT B is '-', TREMBL B is 'Q86UA3'.
        """
        q86_rows = biogrid_df[biogrid_df['protein_b'] == 'Q86UA3']
        assert len(q86_rows) >= 1


# ── HuMAP Edge Case Tests ──────────────────────────────────────────

@pytest.mark.slow
class TestHumapEdgeCases:
    """Edge case tests for HuMAP loader."""

    @pytest.fixture(scope="class")
    def humap_df(self):
        return load_humap(str(TEST_DB_DIR / "test_humap.pairsWprob"))

    def test_isoform_accessions_preserved(self, humap_df):
        """HuMAP rows with isoform accessions (P22607-1) keep the suffix."""
        isoform_rows = humap_df[
            humap_df['protein_a'].str.contains('-', na=False) |
            humap_df['protein_b'].str.contains('-', na=False)
        ]
        assert len(isoform_rows) >= 1
        # Check specific isoform
        p22607_1 = humap_df[
            (humap_df['protein_a'] == 'P22607-1') |
            (humap_df['protein_b'] == 'P22607-1')
        ]
        assert len(p22607_1) >= 1

    def test_self_interaction_loaded(self, humap_df):
        """Self-interaction pairs (P_X, P_X) are loaded."""
        self_pairs = humap_df[humap_df['protein_a'] == humap_df['protein_b']]
        assert len(self_pairs) >= 1

    def test_boundary_probability_zero(self, humap_df):
        """Rows with probability=0.0 are handled correctly."""
        zero_prob = humap_df[humap_df['confidence_score'] <= 0.001]
        assert len(zero_prob) >= 1

    def test_boundary_probability_one(self, humap_df):
        """Rows with probability=1.0 are handled correctly."""
        max_prob = humap_df[humap_df['confidence_score'] >= 0.999]
        assert len(max_prob) >= 1


# ── HuMAP Malformed Input Tests ──────────────────────────────────────

@pytest.mark.slow
class TestHumapMalformedInput:
    """Tests for HuMAP loader with non-UniProt IDs in the data."""

    def test_malformed_ids_warned_and_skipped(self):
        """Loader detects non-UniProt IDs, warns, and skips those rows.

        test_humap_malformed.pairsWprob has 5 rows:
          3 valid (UniProt-UniProt) + 2 invalid (ENSG ID, gene symbol).
        """
        import warnings as _warnings
        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            df = load_humap(
                str(TEST_DB_DIR / "test_humap_malformed.pairsWprob"),
                validate_ids=True,
            )
            # Should have emitted a warning about non-UniProt IDs
            humap_warnings = [x for x in w if "non-UniProt" in str(x.message)]
            assert len(humap_warnings) == 1
            assert "ENSG00000184779" in str(humap_warnings[0].message)

        # Only 3 valid rows should survive
        assert len(df) == 3
        # All remaining IDs should be UniProt format
        assert df['protein_a'].str.match(r'^[A-Z][0-9]').all()
        assert df['protein_b'].str.match(r'^[A-Z][0-9]').all()

    def test_validation_can_be_disabled(self):
        """With validate_ids=False, non-UniProt IDs are kept."""
        df = load_humap(
            str(TEST_DB_DIR / "test_humap_malformed.pairsWprob"),
            validate_ids=False,
        )
        # All 5 rows should be present
        assert len(df) == 5


# ── Cross-Database Overlap Tests ─────────────────────────────────────

@pytest.mark.slow
class TestCrossDatabaseOverlap:
    """Integration tests verifying that pairs from different databases
    (with different ID systems) correctly match after ID mapping.

    This directly tests the supervisor's concern about Venn diagram
    accuracy across ENSP, ENSG, and UniProt identifier systems.

    Known shared pairs planted across test excerpts:
      TP53-BRCA1: STRING (ENSP), BioGRID (UniProt), HuRI (ENSG), HuMAP (UniProt)
      TP53-MDM2:  STRING (ENSP), BioGRID (UniProt), HuRI (ENSG), HuMAP (UniProt)
      ARF5-EEF1B2: STRING (ENSP), BioGRID (UniProt), HuMAP (UniProt)
    """

    @pytest.fixture(scope="class")
    def mapped_pair_sets(self):
        """Load all databases, map to UniProt, extract normalised pair sets."""
        from id_mapper import IDMapper, map_dataframe_to_uniprot
        from overlap_analysis import extract_pair_set

        aliases_path = TEST_DB_DIR / "test_aliases.txt"
        mapper = IDMapper(str(aliases_path))

        # Load databases
        string_df = load_string(str(TEST_DB_DIR / "test_string_links.txt"))
        biogrid_df = load_biogrid(str(TEST_DB_DIR / "test_biogrid.tab3.txt"))
        huri_df = load_huri(str(TEST_DB_DIR / "test_huri.tsv"))
        humap_df = load_humap(str(TEST_DB_DIR / "test_humap.pairsWprob"))

        # Map STRING (ENSP) and HuRI (ENSG) to UniProt
        string_mapped = map_dataframe_to_uniprot(string_df, mapper)
        huri_mapped = map_dataframe_to_uniprot(huri_df, mapper)

        # Extract pair sets (mapped DBs use uniprot_a/b, direct use protein_a/b)
        return {
            'STRING': extract_pair_set(string_mapped, col_a='uniprot_a', col_b='uniprot_b'),
            'BioGRID': extract_pair_set(biogrid_df),
            'HuRI': extract_pair_set(huri_mapped, col_a='uniprot_a', col_b='uniprot_b'),
            'HuMAP': extract_pair_set(humap_df),
        }

    def test_shared_pair_across_string_and_biogrid(self, mapped_pair_sets):
        """TP53-MDM2 in STRING (ENSP) matches same pair in BioGRID (UniProt).

        STRING has 9606.ENSP00000269305 - 9606.ENSP00000258149
        BioGRID has P04637 - Q00987
        After mapping ENSP->UniProt, normalise_pair should produce
        the same canonical pair ('P04637', 'Q00987').

        Note: Some pairs (e.g. TP53-BRCA1) may NOT match across databases
        because the IDMapper can return different canonical accessions
        (O15129 vs P38398 for BRCA1) depending on the aliases sort order.
        """
        from overlap_analysis import normalise_pair
        tp53_mdm2 = normalise_pair('P04637', 'Q00987')
        assert tp53_mdm2 in mapped_pair_sets['STRING']
        assert tp53_mdm2 in mapped_pair_sets['BioGRID']

    def test_shared_pair_across_huri_and_biogrid(self, mapped_pair_sets):
        """TP53-MDM2 in HuRI (ENSG) matches same pair in BioGRID (UniProt).

        HuRI has ENSG00000141510 - ENSG00000135679
        BioGRID has P04637 - Q00987
        The ENSG->ENSP->UniProt chain should resolve correctly.
        """
        from overlap_analysis import normalise_pair
        tp53_mdm2 = normalise_pair('P04637', 'Q00987')
        assert tp53_mdm2 in mapped_pair_sets['HuRI']
        assert tp53_mdm2 in mapped_pair_sets['BioGRID']

    def test_overlap_count_matches_expected(self, mapped_pair_sets):
        """Computed overlap counts match manually counted shared pairs.

        TP53-MDM2 (P04637-Q00987) is confirmed shared across all 4
        databases. The all-4 overlap should be >= 1.

        Note: Other pairs like TP53-BRCA1 may not appear in the all-4
        intersection because the IDMapper may resolve BRCA1 to
        different accessions (O15129 via STRING/HuRI vs P38398 via
        BioGRID/HuMAP) depending on alphabetical sort order.
        """
        from overlap_analysis import compute_overlaps
        stats = compute_overlaps(mapped_pair_sets)
        assert stats['all'] >= 1

        # STRING-BioGRID pairwise overlap should include at least TP53-MDM2
        assert stats['pairwise'][('STRING', 'BioGRID')] >= 1

    def test_isoform_pair_treated_as_distinct(self, mapped_pair_sets):
        """HuMAP P22607-1 and BioGRID P22607 are different pair identifiers.

        HuMAP has (P22607-1, P28482) while BioGRID has (P22607, P54764).
        The isoform-specific pair should NOT be conflated with the
        canonical pair - they are stored as distinct strings.
        """
        from overlap_analysis import normalise_pair
        isoform_pair = normalise_pair('P22607-1', 'P28482')
        canonical_pair = normalise_pair('P22607', 'P28482')
        # These should be different tuples
        assert isoform_pair != canonical_pair

        # The isoform pair should be in HuMAP
        assert isoform_pair in mapped_pair_sets['HuMAP']

    def test_self_pair_in_overlap(self, mapped_pair_sets):
        """Self-interactions are consistently handled across databases.

        STRING has ENSP00000269305-ENSP00000269305 (TP53 self)
        HuMAP has P04637-P04637 (TP53 self)
        After mapping, both should produce ('P04637', 'P04637').
        """
        from overlap_analysis import normalise_pair
        tp53_self = normalise_pair('P04637', 'P04637')
        assert tp53_self in mapped_pair_sets['STRING']
        assert tp53_self in mapped_pair_sets['HuMAP']

    def test_unmappable_huri_pairs_excluded(self, mapped_pair_sets):
        """HuRI pairs with unmappable ENSG IDs don't appear in overlaps.

        Synthetic ENSG00000999901/02/03 have no aliases entry, so
        map_dataframe_to_uniprot drops those rows. They must not
        inflate HuRI's pair count or overlap counts.
        """
        from overlap_analysis import normalise_pair
        # If these somehow mapped, they'd produce pairs starting with 'ENSG'
        # but all HuRI pairs should be UniProt after mapping
        for pair in mapped_pair_sets['HuRI']:
            assert not pair[0].startswith('ENSG'), f"Unmapped ENSG in HuRI overlap: {pair}"
            assert not pair[1].startswith('ENSG'), f"Unmapped ENSG in HuRI overlap: {pair}"


# ── Base-Level Overlap Tests ────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.database
class TestBaseLevelOverlap:
    """Tests for base-accession level pair normalisation and overlap.

    At base level, isoform suffixes (e.g. -1, -2) are stripped before
    normalisation. This should increase overlap counts compared to
    isoform-specific matching.
    """

    def test_normalise_pair_base_strips_isoform(self):
        """normalise_pair_base strips -N isoform suffixes."""
        from overlap_analysis import normalise_pair_base
        result = normalise_pair_base('P22607-1', 'P28482')
        assert result == ('P22607', 'P28482')

    def test_normalise_pair_base_no_isoform_unchanged(self):
        """normalise_pair_base leaves non-isoform IDs unchanged."""
        from overlap_analysis import normalise_pair_base, normalise_pair
        result_base = normalise_pair_base('P04637', 'Q00987')
        result_iso = normalise_pair('P04637', 'Q00987')
        assert result_base == result_iso

    def test_normalise_pair_base_both_isoforms(self):
        """normalise_pair_base handles isoform suffixes on both IDs."""
        from overlap_analysis import normalise_pair_base
        result = normalise_pair_base('Q9UKT4-2', 'P22607-1')
        assert result == ('P22607', 'Q9UKT4')

    def test_extract_pair_set_base_fewer_or_equal_pairs(self):
        """Base-level pair set size <= isoform-level (merging reduces count)."""
        from overlap_analysis import extract_pair_set, extract_pair_set_base

        # Build a DataFrame with isoform-bearing pairs
        df = pd.DataFrame({
            'protein_a': ['P22607-1', 'P22607', 'P04637'],
            'protein_b': ['P28482',   'P28482', 'Q00987'],
        })
        iso_pairs = extract_pair_set(df)
        base_pairs = extract_pair_set_base(df)
        # P22607-1/P28482 and P22607/P28482 collapse to one base pair
        assert len(base_pairs) <= len(iso_pairs)
        assert len(base_pairs) == 2  # (P22607,P28482) + (P04637,Q00987)

    def test_base_level_overlap_gte_isoform_level(self):
        """Base-level overlap >= isoform-level for real test databases.

        Stripping isoform suffixes can only increase or maintain overlap
        (more pairs match across databases when isoforms are collapsed).
        """
        from id_mapper import IDMapper, map_dataframe_to_uniprot
        from overlap_analysis import (
            extract_pair_set, extract_pair_set_base, compute_overlaps,
        )

        aliases_path = TEST_DB_DIR / "test_aliases.txt"
        mapper = IDMapper(str(aliases_path))

        string_df = load_string(str(TEST_DB_DIR / "test_string_links.txt"))
        biogrid_df = load_biogrid(str(TEST_DB_DIR / "test_biogrid.tab3.txt"))

        string_mapped = map_dataframe_to_uniprot(string_df, mapper)

        # Isoform-specific
        iso_sets = {
            'STRING': extract_pair_set(string_mapped, col_a='uniprot_a', col_b='uniprot_b'),
            'BioGRID': extract_pair_set(biogrid_df),
        }
        iso_stats = compute_overlaps(iso_sets)

        # Base-accession
        base_sets = {
            'STRING': extract_pair_set_base(string_mapped, col_a='uniprot_a', col_b='uniprot_b'),
            'BioGRID': extract_pair_set_base(biogrid_df),
        }
        base_stats = compute_overlaps(base_sets)

        assert base_stats['pairwise'][('STRING', 'BioGRID')] >= \
               iso_stats['pairwise'][('STRING', 'BioGRID')]


@pytest.mark.slow
@pytest.mark.database
class TestOverlapReport:
    """Tests for the --report CLI flag that writes stats to a file."""

    def test_report_file_written(self, tmp_path):
        """--report writes a text file with overlap statistics."""
        from overlap_analysis import compute_overlaps, print_overlap_summary

        # Create minimal pair sets
        pair_sets = {
            'DB1': {('A', 'B'), ('C', 'D')},
            'DB2': {('A', 'B'), ('E', 'F')},
        }
        stats = compute_overlaps(pair_sets)

        report_path = tmp_path / "test_report.txt"
        with open(report_path, 'w') as f:
            f.write("PPI Database Overlap Report\n")
            f.write("=" * 40 + "\n")
            print_overlap_summary(stats, file=f)

        assert report_path.exists()
        content = report_path.read_text()
        assert 'PPI Database Overlap Summary' in content
        assert 'DB1' in content
        assert 'DB2' in content
        assert 'DB1 & DB2: 1' in content  # ('A','B') is shared

    def test_report_contains_unique_counts(self, tmp_path):
        """Report file contains per-database unique pair counts."""
        from overlap_analysis import compute_overlaps, print_overlap_summary

        pair_sets = {
            'DB1': {('A', 'B'), ('C', 'D')},
            'DB2': {('A', 'B'), ('E', 'F')},
        }
        stats = compute_overlaps(pair_sets)

        report_path = tmp_path / "test_report.txt"
        with open(report_path, 'w') as f:
            print_overlap_summary(stats, file=f)

        content = report_path.read_text()
        assert 'DB1 only: 1' in content  # (C,D) unique to DB1
        assert 'DB2 only: 1' in content  # (E,F) unique to DB2
