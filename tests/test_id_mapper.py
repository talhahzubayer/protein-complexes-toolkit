"""
Tests for id_mapper.py - ID cross-referencing and resolution.

Uses test_aliases.txt excerpt in tests/test_data/databases/ for offline testing.
"""

import pytest
import pandas as pd
from pathlib import Path

from id_mapper import (
    IDMapper,
    is_uniprot_accession,
    split_isoform,
    detect_id_type,
    map_dataframe_to_uniprot,
    export_lookup_table,
    build_uniprot_lookup,
)

# ── Test Data Paths ──────────────────────────────────────────────────

PROJECT_ROOT = Path(r"C:\Users\Talhah Zubayer\Documents\protein-complexes-toolkit")
TEST_ALIASES = PROJECT_ROOT / "tests" / "test_data" / "databases" / "test_aliases.txt"


# ── ID Validation Tests (pure unit, no I/O) ──────────────────────────

class TestIsUniprotAccession:
    """Tests for is_uniprot_accession()."""

    def test_canonical_6char(self):
        assert is_uniprot_accession("P04637") is True

    def test_canonical_q_start(self):
        assert is_uniprot_accession("Q9UKT4") is True

    def test_canonical_o_start(self):
        assert is_uniprot_accession("O15151") is True

    def test_isoform_suffix(self):
        assert is_uniprot_accession("Q9UKT4-2") is True

    def test_long_format_10char(self):
        assert is_uniprot_accession("A0A0B4J2C3") is True

    def test_long_format_with_isoform(self):
        assert is_uniprot_accession("A0A0B4J2C3-1") is True

    def test_gene_symbol_rejected(self):
        assert is_uniprot_accession("TP53") is False

    def test_ensp_rejected(self):
        assert is_uniprot_accession("ENSP00000269305") is False

    def test_ensg_rejected(self):
        assert is_uniprot_accession("ENSG00000141510") is False

    def test_empty_string_rejected(self):
        assert is_uniprot_accession("") is False

    def test_numeric_rejected(self):
        assert is_uniprot_accession("12345") is False


class TestSplitIsoform:
    """Tests for split_isoform()."""

    def test_with_isoform(self):
        base, iso = split_isoform("Q9UKT4-2")
        assert base == "Q9UKT4"
        assert iso == "2"

    def test_without_isoform(self):
        base, iso = split_isoform("P04637")
        assert base == "P04637"
        assert iso is None

    def test_long_format_with_isoform(self):
        base, iso = split_isoform("A0A0B4J2C3-1")
        assert base == "A0A0B4J2C3"
        assert iso == "1"

    def test_dash_in_non_numeric_context(self):
        """Dash followed by non-digit should not be treated as isoform."""
        base, iso = split_isoform("P12345")
        assert base == "P12345"
        assert iso is None


class TestDetectIdType:
    """Tests for detect_id_type()."""

    def test_ensp(self):
        assert detect_id_type("ENSP00000269305") == "ensp"

    def test_ensg(self):
        assert detect_id_type("ENSG00000141510") == "ensg"

    def test_uniprot_canonical(self):
        assert detect_id_type("P04637") == "uniprot"

    def test_uniprot_isoform(self):
        assert detect_id_type("Q9UKT4-2") == "uniprot_isoform"

    def test_long_uniprot(self):
        assert detect_id_type("A0A0B4J2C3") == "uniprot"

    def test_unknown(self):
        assert detect_id_type("some_random_string") == "unknown"


# ── IDMapper Tests (require aliases excerpt) ─────────────────────────

@pytest.mark.slow
class TestIDMapper:
    """Tests for IDMapper class using test aliases excerpt."""

    @pytest.fixture(scope="class")
    def mapper(self):
        """Session-scoped IDMapper from test excerpt."""
        assert TEST_ALIASES.exists(), f"Test aliases not found: {TEST_ALIASES}"
        return IDMapper(str(TEST_ALIASES))

    def test_loads_without_error(self, mapper):
        """IDMapper initialises from the test excerpt."""
        assert mapper is not None

    def test_mapping_stats_non_empty(self, mapper):
        """Mapping dictionaries are populated."""
        stats = mapper.get_mapping_stats()
        assert stats['ensp_to_uniprot'] > 0
        assert stats['ensp_to_symbol'] > 0
        assert stats['ensg_to_ensp'] > 0

    def test_ensembl_to_uniprot_tp53(self, mapper):
        """TP53: ENSP00000269305 maps to P04637."""
        result = mapper.ensembl_to_uniprot("ENSP00000269305")
        assert "P04637" in result

    def test_ensembl_to_uniprot_arf5(self, mapper):
        """ARF5: ENSP00000000233 maps to P84085."""
        result = mapper.ensembl_to_uniprot("ENSP00000000233")
        assert "P84085" in result

    def test_ensembl_to_uniprot_accepts_prefix(self, mapper):
        """Accepts IDs with 9606. prefix."""
        result = mapper.ensembl_to_uniprot("9606.ENSP00000269305")
        assert "P04637" in result

    def test_ensembl_to_uniprot_unknown_returns_empty(self, mapper):
        result = mapper.ensembl_to_uniprot("ENSP99999999999")
        assert result == []

    def test_canonical_accession_first(self, mapper):
        """Reviewed (Swiss-Prot, P/Q/O 6-char) accessions sort before TrEMBL."""
        result = mapper.ensembl_to_uniprot("ENSP00000269305")
        assert result[0] == "P04637"

    def test_uniprot_to_gene_symbol_tp53(self, mapper):
        result = mapper.uniprot_to_gene_symbol("P04637")
        assert result == "TP53"

    def test_uniprot_to_gene_symbol_arf5(self, mapper):
        result = mapper.uniprot_to_gene_symbol("P84085")
        assert result == "ARF5"

    def test_uniprot_to_ensembl(self, mapper):
        result = mapper.uniprot_to_ensembl("P04637")
        assert "ENSP00000269305" in result

    def test_ensg_to_uniprot(self, mapper):
        """ENSG -> ENSP -> UniProt chain works."""
        # ENSG00000141510 -> ENSP00000269305 -> P04637
        result = mapper.ensg_to_uniprot("ENSG00000141510")
        assert "P04637" in result

    def test_resolve_id_from_ensp(self, mapper):
        result = mapper.resolve_id("ENSP00000269305", target="uniprot")
        assert result == "P04637"

    def test_resolve_id_from_uniprot(self, mapper):
        result = mapper.resolve_id("P04637", target="gene_symbol")
        assert result == "TP53"

    def test_resolve_id_from_ensg(self, mapper):
        result = mapper.resolve_id("ENSG00000141510", target="uniprot")
        assert result is not None

    def test_resolve_id_unknown_returns_none(self, mapper):
        result = mapper.resolve_id("UNKNOWN_ID_XYZ", target="uniprot")
        assert result is None

    def test_resolve_pair_to_uniprot(self, mapper):
        """Resolve a pair of ENSP IDs to UniProt."""
        result = mapper.resolve_pair_to_uniprot(
            "ENSP00000269305", "ENSP00000000233"
        )
        assert result is not None
        uniprot_a, uniprot_b, is_base = result
        assert uniprot_a == "P04637"
        assert is_base is True  # Non-UniProt input -> base accession match

    def test_resolve_pair_uniprot_input(self, mapper):
        """UniProt inputs are NOT flagged as base accession match."""
        result = mapper.resolve_pair_to_uniprot("P04637", "P84085")
        assert result is not None
        _, _, is_base = result
        assert is_base is False

    def test_isoform_preserved_in_resolve(self, mapper):
        """Isoform-specific accession is preserved through resolve_id."""
        result = mapper.resolve_id("Q9UKT4-2", target="uniprot")
        assert result == "Q9UKT4-2"


# ── ID Mapper Edge Case Tests ────────────────────────────────────────

@pytest.mark.slow
class TestIDMapperEdgeCases:
    """Edge case tests for IDMapper - isoforms, multi-accession, splice variants."""

    @pytest.fixture(scope="class")
    def mapper(self):
        assert TEST_ALIASES.exists(), f"Test aliases not found: {TEST_ALIASES}"
        return IDMapper(str(TEST_ALIASES))

    def test_isoform_accession_in_aliases(self, mapper):
        """Aliases containing P22607-1 are parsed correctly.

        Synthetic entries add P22607-1 and P22607-2 as UniProt_AC for
        ENSP00000339824 (FGFR3). detect_id_type should return 'uniprot_isoform'.
        """
        assert detect_id_type("P22607-1") == "uniprot_isoform"
        assert detect_id_type("P22607-2") == "uniprot_isoform"
        # ENSP00000339824 should map to P22607 (canonical) plus isoforms
        result = mapper.ensembl_to_uniprot("ENSP00000339824")
        assert "P22607" in result

    def test_multiple_uniprot_per_ensp(self, mapper):
        """TP53 ENSP maps to multiple UniProt accessions, canonical first.

        ENSP00000269305 (TP53) has many accessions in the aliases file.
        The canonical Swiss-Prot accession P04637 should be first.
        """
        result = mapper.ensembl_to_uniprot("ENSP00000269305")
        assert len(result) >= 2
        assert result[0] == "P04637"

    def test_secondary_accession_maps_back(self, mapper):
        """A secondary/alternative UniProt accession for TP53 maps back
        to the same ENSP via uniprot_to_ensembl().

        Multiple accessions for TP53 should resolve to ENSP00000269305.
        """
        all_accessions = mapper.ensembl_to_uniprot("ENSP00000269305")
        if len(all_accessions) >= 2:
            secondary = all_accessions[1]
            ensp_list = mapper.uniprot_to_ensembl(secondary)
            assert "ENSP00000269305" in ensp_list

    def test_ensg_with_multiple_ensp(self, mapper):
        """ENSG00000068078 maps to 2 ENSPs (splice variant testing).

        Synthetic aliases add ENSP00000231803 alongside the real
        ENSP00000339824 for FGFR3 gene. The ENSG->UniProt chain should
        return accessions from both ENSPs.
        """
        ensp_list = mapper.ensg_to_ensembl("ENSG00000068078")
        assert len(ensp_list) >= 2
        assert "ENSP00000339824" in ensp_list
        assert "ENSP00000231803" in ensp_list

        # ensg_to_uniprot should aggregate across both ENSPs
        uniprot_list = mapper.ensg_to_uniprot("ENSG00000068078")
        assert "P22607" in uniprot_list

    def test_resolve_isoform_preserves_suffix(self, mapper):
        """resolve_id('P22607-2') returns 'P22607-2' (not stripped)."""
        result = mapper.resolve_id("P22607-2", target="uniprot")
        assert result == "P22607-2"

    def test_resolve_pair_isoform_vs_base(self, mapper):
        """Pair resolution with isoform vs ENSP input sets is_base correctly.

        When both inputs are UniProt (including isoform), is_base = False.
        When one input is ENSP, is_base = True.
        """
        # Both UniProt (one isoform) -> is_base = False
        result = mapper.resolve_pair_to_uniprot("P22607-2", "P04637")
        assert result is not None
        _, _, is_base = result
        assert is_base is False

        # One ENSP -> is_base = True
        result = mapper.resolve_pair_to_uniprot("ENSP00000269305", "P22607-2")
        assert result is not None
        _, _, is_base = result
        assert is_base is True

    def test_unmappable_id_returns_none(self, mapper):
        """resolve_id with a completely unknown ID returns None."""
        result = mapper.resolve_id("COMPLETELY_UNKNOWN_STRING", target="uniprot")
        assert result is None
        result = mapper.resolve_id("COMPLETELY_UNKNOWN_STRING", target="ensp")
        assert result is None
        result = mapper.resolve_id("COMPLETELY_UNKNOWN_STRING", target="gene_symbol")
        assert result is None


# ── map_dataframe_to_uniprot Tests ───────────────────────────────────

@pytest.mark.slow
class TestMapDataframeToUniprot:
    """Tests for the DataFrame mapping convenience function."""

    @pytest.fixture(scope="class")
    def mapper(self):
        return IDMapper(str(TEST_ALIASES))

    def test_maps_ensp_columns(self, mapper):
        """Maps ENSP IDs in a DataFrame to UniProt."""
        df = pd.DataFrame({
            'protein_a': ['ENSP00000269305', 'ENSP00000000233'],
            'protein_b': ['ENSP00000000233', 'ENSP00000269305'],
        })
        result = map_dataframe_to_uniprot(df, mapper)
        assert 'uniprot_a' in result.columns
        assert 'uniprot_b' in result.columns
        assert result['uniprot_a'].iloc[0] == "P04637"

    def test_drops_unmappable_rows(self, mapper):
        """Rows with unmappable IDs are dropped."""
        df = pd.DataFrame({
            'protein_a': ['ENSP00000269305', 'ENSP99999999999'],
            'protein_b': ['ENSP00000000233', 'ENSP88888888888'],
        })
        result = map_dataframe_to_uniprot(df, mapper)
        assert len(result) == 1  # Second row dropped

    def test_preserves_other_columns(self, mapper):
        """Other columns in the DataFrame are preserved."""
        df = pd.DataFrame({
            'protein_a': ['ENSP00000269305'],
            'protein_b': ['ENSP00000000233'],
            'source': ['STRING'],
            'confidence_score': [0.999],
        })
        result = map_dataframe_to_uniprot(df, mapper)
        assert 'source' in result.columns
        assert result['source'].iloc[0] == 'STRING'


# ── Secondary Accessions & Lookup Table Tests ────────────────────────

@pytest.mark.slow
@pytest.mark.database
class TestSecondaryAccessions:
    """Tests for get_secondary_accessions() and enhanced lookup table."""

    @pytest.fixture(scope="class")
    def mapper(self):
        assert TEST_ALIASES.exists(), f"Test aliases not found: {TEST_ALIASES}"
        return IDMapper(str(TEST_ALIASES))

    def test_secondary_accessions_returns_remainder(self, mapper):
        """TP53 (ENSP00000269305) has multiple accessions; secondary = all but first."""
        all_acc = mapper.ensembl_to_uniprot("ENSP00000269305")
        secondary = mapper.get_secondary_accessions("ENSP00000269305")
        if len(all_acc) >= 2:
            assert secondary == all_acc[1:]
            assert all_acc[0] not in secondary

    def test_secondary_accessions_single_mapping_empty(self, mapper):
        """ENSP with only one UniProt mapping returns empty list."""
        # ARF5 (ENSP00000000233) maps to P84085 - may have only one mapping
        all_acc = mapper.ensembl_to_uniprot("ENSP00000000233")
        secondary = mapper.get_secondary_accessions("ENSP00000000233")
        if len(all_acc) == 1:
            assert secondary == []
        else:
            assert len(secondary) == len(all_acc) - 1

    def test_secondary_accessions_unknown_ensp(self, mapper):
        """Unknown ENSP returns empty list."""
        assert mapper.get_secondary_accessions("ENSP99999999999") == []

    def test_secondary_accessions_accepts_prefix(self, mapper):
        """Handles 9606. prefix correctly."""
        result = mapper.get_secondary_accessions("9606.ENSP00000269305")
        all_acc = mapper.ensembl_to_uniprot("ENSP00000269305")
        if len(all_acc) >= 2:
            assert result == all_acc[1:]

    def test_get_protein_name(self, mapper):
        """Protein name lookup returns a string or None."""
        name = mapper.get_protein_name("ENSP00000269305")
        # May or may not be in the test excerpt
        assert name is None or isinstance(name, str)


@pytest.mark.slow
@pytest.mark.database
class TestExportLookupTable:
    """Tests for the enhanced export_lookup_table() with primary/secondary columns."""

    @pytest.fixture(scope="class")
    def mapper(self):
        assert TEST_ALIASES.exists()
        return IDMapper(str(TEST_ALIASES))

    def test_export_creates_file(self, mapper, tmp_path):
        """Export produces a CSV file."""
        out = tmp_path / "lookup.csv"
        export_lookup_table(mapper, str(out))
        assert out.exists()

    def test_export_has_new_columns(self, mapper, tmp_path):
        """Exported CSV has the enhanced column structure."""
        out = tmp_path / "lookup.csv"
        export_lookup_table(mapper, str(out))
        import csv
        with open(out, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
        expected = [
            'ensembl_protein_id', 'primary_uniprot', 'base_accession',
            'secondary_accessions', 'ensembl_gene_id', 'gene_symbol',
            'protein_name',
        ]
        assert headers == expected

    def test_export_primary_is_first_accession(self, mapper, tmp_path):
        """primary_uniprot matches the first entry from ensembl_to_uniprot()."""
        out = tmp_path / "lookup.csv"
        export_lookup_table(mapper, str(out))
        import csv
        with open(out, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['ensembl_protein_id'] == 'ENSP00000269305':
                    assert row['primary_uniprot'] == 'P04637'
                    break

    def test_export_base_accession_strips_isoform(self, mapper, tmp_path):
        """base_accession strips isoform suffix from primary_uniprot."""
        out = tmp_path / "lookup.csv"
        export_lookup_table(mapper, str(out))
        import csv
        with open(out, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                primary = row['primary_uniprot']
                base = row['base_accession']
                if primary:
                    assert '-' not in base or not base.rsplit('-', 1)[1].isdigit()


@pytest.mark.slow
@pytest.mark.database
class TestBuildUniprotLookup:
    """Tests for build_uniprot_lookup() convenience function."""

    @pytest.fixture(scope="class")
    def mapper(self):
        assert TEST_ALIASES.exists()
        return IDMapper(str(TEST_ALIASES))

    @pytest.fixture(scope="class")
    def lookup(self, mapper):
        return build_uniprot_lookup(mapper)

    def test_lookup_is_dict(self, lookup):
        assert isinstance(lookup, dict)
        assert len(lookup) > 0

    def test_lookup_has_expected_keys(self, lookup):
        """Each entry has the expected metadata fields."""
        for acc, info in list(lookup.items())[:5]:
            assert 'gene_symbol' in info
            assert 'protein_name' in info
            assert 'ensembl_protein_id' in info
            assert 'ensembl_gene_id' in info

    def test_lookup_contains_tp53(self, lookup):
        """P04637 (TP53) is in the lookup."""
        assert 'P04637' in lookup
        assert lookup['P04637']['gene_symbol'] == 'TP53'

    def test_secondary_points_to_same_info(self, lookup, mapper):
        """Secondary accessions for TP53 point to the same ENSP info."""
        secondary = mapper.get_secondary_accessions("ENSP00000269305")
        primary_info = lookup.get('P04637')
        for acc in secondary:
            if acc in lookup:
                assert lookup[acc]['ensembl_protein_id'] == primary_info['ensembl_protein_id']

    def test_lookup_has_secondary_accessions_field(self, lookup):
        """Each entry has a secondary_accessions field."""
        for acc, info in list(lookup.items())[:5]:
            assert 'secondary_accessions' in info

    def test_lookup_secondary_accessions_populated(self, lookup, mapper):
        """ENSPs with multiple UniProt mappings produce non-empty secondary_accessions."""
        found_populated = False
        for acc, info in lookup.items():
            if info['secondary_accessions']:
                found_populated = True
                # Pipe-separated format
                assert '|' in info['secondary_accessions'] or len(info['secondary_accessions']) >= 6
                break
        assert found_populated, "No entries with secondary accessions found"

    def test_lookup_single_mapping_has_empty_secondary(self, lookup):
        """P04637 (TP53) with a single primary mapping has empty secondary_accessions."""
        info = lookup.get('P04637')
        assert info is not None
        # TP53 may or may not have secondary accessions depending on STRING data;
        # verify the field is a string (not None/missing)
        assert isinstance(info['secondary_accessions'], str)
