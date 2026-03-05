"""
Tests for toolkit.py - batch orchestration, file parsing, quality classification.

Regression values captured on 2026-03-04 from reference complex 1
(A0A0B4J2C3_P24534) on Windows 11.
"""

import csv
import subprocess
import sys
from pathlib import Path

import pytest

from toolkit import (
    parse_complex_name,
    find_paired_data_files,
    extract_plddt_from_pdb,
    classify_prediction_quality,
    classify_prediction_quality_v2,
    process_single_complex,
    get_csv_fieldnames,
    write_results_csv,
    enrich_results,
    CSV_FIELDNAMES_BASE,
    CSV_FIELDNAMES_INTERFACE,
    CSV_FIELDNAMES_INTERFACE_PAE,
    CSV_FIELDNAMES_ENRICHMENT,
    CSV_FIELDNAMES_FLAGS,
    THREE_TO_ONE,
    IPTM_HIGH_THRESHOLD,
    IPTM_MEDIUM_THRESHOLD,
    PDOCKQ_HIGH_THRESHOLD,
    PDOCKQ_MEDIUM_THRESHOLD,
    UPGRADE_LOW_THRESHOLD,
    UPGRADE_MEDIUM_THRESHOLD,
    DOWNGRADE_HIGH_THRESHOLD,
)

PROJECT_ROOT = Path(r"C:\Users\Talhah Zubayer\Documents\protein-complexes-toolkit")

# ── Regression expected values (complex 1: A0A0B4J2C3_P24534) ────
EXPECTED_IPTM_1 = 0.611156165599823
EXPECTED_PDOCKQ_1 = 0.3301
EXPECTED_QUALITY_TIER_1 = 'Medium'
EXPECTED_QUALITY_TIER_V2_1 = 'Medium'
EXPECTED_COMPLEX_TYPE_1 = 'Heterodimer'
EXPECTED_N_CONTACTS_1 = 92
EXPECTED_COMPOSITE_SCORE_1 = 0.6726
EXPECTED_PAIRED_COUNT = 61


# ── parse_complex_name Tests (pure unit) ─────────────────────────

class TestParseComplexName:
    """Tests for parse_complex_name across all naming conventions."""

    def test_old_convention_pdb(self):
        name, prot_a, prot_b, ctype = parse_complex_name("A0A0B4J2C3_P24534.pdb")
        assert name == "A0A0B4J2C3_P24534"
        assert prot_a == "A0A0B4J2C3"
        assert prot_b == "P24534"
        assert ctype == "Heterodimer"

    def test_old_convention_results_pkl(self):
        name, prot_a, prot_b, ctype = parse_complex_name("A0A0B4J2C3_P24534.results.pkl")
        assert name == "A0A0B4J2C3_P24534"
        assert prot_a == "A0A0B4J2C3"
        assert prot_b == "P24534"

    def test_new_convention_pdb(self):
        name, prot_a, prot_b, ctype = parse_complex_name(
            "A0A0A0MQZ0_P40933_relaxed_model_1_multimer_v3_pred_0.pdb"
        )
        assert name == "A0A0A0MQZ0_P40933"
        assert prot_a == "A0A0A0MQZ0"
        assert prot_b == "P40933"
        assert ctype == "Heterodimer"

    def test_new_convention_pkl(self):
        name, prot_a, prot_b, ctype = parse_complex_name(
            "A0A0A0MQZ0_P40933_result_model_1_multimer_v3_pred_0.pkl"
        )
        assert name == "A0A0A0MQZ0_P40933"
        assert prot_a == "A0A0A0MQZ0"
        assert prot_b == "P40933"

    def test_homodimer(self):
        name, prot_a, prot_b, ctype = parse_complex_name("A0A0H3C8Q1_A0A0H3C8Q1.pdb")
        assert name == "A0A0H3C8Q1_A0A0H3C8Q1"
        assert prot_a == "A0A0H3C8Q1"
        assert prot_b == "A0A0H3C8Q1"
        assert ctype == "Homodimer"

    def test_doubled_name(self):
        name, prot_a, prot_b, ctype = parse_complex_name(
            "P0C0L2_P0C0L2_P0C0L2_P0C0L2.results.pkl"
        )
        assert name == "P0C0L2_P0C0L2"
        assert prot_a == "P0C0L2"
        assert prot_b == "P0C0L2"
        assert ctype == "Homodimer"

    def test_isoform_dash(self):
        name, prot_a, prot_b, ctype = parse_complex_name("P63208-1_Q6PJ61.pdb")
        assert prot_a == "P63208-1"
        assert prot_b == "Q6PJ61"
        assert ctype == "Heterodimer"

    def test_pdb_and_pkl_agree(self):
        """Same complex produces same name from both PDB and PKL filenames."""
        name_pdb, _, _, _ = parse_complex_name("A0A0B4J2C3_P24534.pdb")
        name_pkl, _, _, _ = parse_complex_name("A0A0B4J2C3_P24534.results.pkl")
        assert name_pdb == name_pkl

    def test_new_convention_pdb_pkl_agree(self):
        name_pdb, _, _, _ = parse_complex_name(
            "A0A0A0MQZ0_P40933_relaxed_model_1_multimer_v3_pred_0.pdb"
        )
        name_pkl, _, _, _ = parse_complex_name(
            "A0A0A0MQZ0_P40933_result_model_1_multimer_v3_pred_0.pkl"
        )
        assert name_pdb == name_pkl


# ── Quality Classification Tests (pure unit) ─────────────────────

class TestClassifyPredictionQuality:
    """Tests for v1 quality classification."""

    def test_high(self):
        assert classify_prediction_quality(0.8, 0.6) == 'High'

    def test_medium(self):
        assert classify_prediction_quality(0.55, 0.25) == 'Medium'

    def test_low(self):
        assert classify_prediction_quality(0.4, 0.1) == 'Low'

    def test_none_iptm(self):
        assert classify_prediction_quality(None, 0.6) == 'Low'

    def test_none_both(self):
        assert classify_prediction_quality(None, None) == 'Low'

    def test_boundary_high(self):
        """Exactly at high threshold."""
        result = classify_prediction_quality(
            IPTM_HIGH_THRESHOLD, PDOCKQ_HIGH_THRESHOLD
        )
        assert result == 'High'


class TestClassifyPredictionQualityV2:
    """Tests for v2 quality classification with composite score."""

    def test_no_confidence_falls_back_to_v1(self):
        assert classify_prediction_quality_v2(0.55, 0.25, None) == 'Medium'

    def test_upgrade_low_to_high(self):
        result = classify_prediction_quality_v2(0.4, 0.1, UPGRADE_LOW_THRESHOLD + 0.01)
        assert result == 'High'

    def test_upgrade_medium_to_high(self):
        result = classify_prediction_quality_v2(0.55, 0.25, UPGRADE_MEDIUM_THRESHOLD + 0.01)
        assert result == 'High'

    def test_downgrade_high_to_medium(self):
        result = classify_prediction_quality_v2(0.8, 0.6, DOWNGRADE_HIGH_THRESHOLD - 0.01)
        assert result == 'Medium'

    def test_no_change_high_adequate_confidence(self):
        result = classify_prediction_quality_v2(0.8, 0.6, 0.70)
        assert result == 'High'


# ── CSV Fieldnames Tests ─────────────────────────────────────────

class TestCSVFieldnames:
    """Tests for get_csv_fieldnames."""

    def test_base_only(self):
        fields = get_csv_fieldnames(include_interface=False, include_pae=False)
        assert fields == CSV_FIELDNAMES_BASE

    def test_with_interface(self):
        fields = get_csv_fieldnames(include_interface=True, include_pae=False)
        for col in CSV_FIELDNAMES_INTERFACE:
            assert col in fields

    def test_with_interface_and_pae(self):
        fields = get_csv_fieldnames(include_interface=True, include_pae=True)
        for col in CSV_FIELDNAMES_INTERFACE:
            assert col in fields
        for col in CSV_FIELDNAMES_INTERFACE_PAE:
            assert col in fields
        for col in CSV_FIELDNAMES_FLAGS:
            assert col in fields


# ── File Discovery Tests ─────────────────────────────────────────

@pytest.mark.slow
class TestFindPairedDataFiles:
    """Tests for find_paired_data_files."""

    def test_returns_dict(self, test_data_dir):
        pairs = find_paired_data_files(str(test_data_dir))
        assert isinstance(pairs, dict)
        assert len(pairs) > 0

    def test_expected_count(self, test_data_dir):
        pairs = find_paired_data_files(str(test_data_dir))
        assert len(pairs) == EXPECTED_PAIRED_COUNT

    def test_finds_pdb_and_pkl(self, test_data_dir):
        pairs = find_paired_data_files(str(test_data_dir))
        both_found = sum(1 for p in pairs.values() if 'pdb' in p and 'pkl' in p)
        assert both_found > 0

    def test_paths_exist(self, test_data_dir):
        pairs = find_paired_data_files(str(test_data_dir))
        for complex_name, files in pairs.items():
            for ftype, fpath in files.items():
                assert fpath.exists(), \
                    f"{complex_name} {ftype} path does not exist: {fpath}"


# ── PDB pLDDT Extraction Tests ──────────────────────────────────

@pytest.mark.slow
class TestExtractPlddtFromPdb:
    """Tests for extract_plddt_from_pdb."""

    def test_returns_dict(self, ref_pdb_1):
        result = extract_plddt_from_pdb(ref_pdb_1)
        assert isinstance(result, dict)
        assert 'plddt_mean' in result
        assert 'num_residues' in result

    def test_fraction_range(self, ref_pdb_1):
        result = extract_plddt_from_pdb(ref_pdb_1)
        assert 0 <= result['plddt_below50_fraction'] <= 1
        assert 0 <= result['plddt_below70_fraction'] <= 1

    def test_nonexistent_returns_none(self):
        result = extract_plddt_from_pdb(Path("/nonexistent/fake.pdb"))
        assert result is None


# ── process_single_complex Tests ─────────────────────────────────

@pytest.mark.slow
class TestProcessSingleComplex:
    """Tests for process_single_complex with real data."""

    @pytest.fixture
    def file_paths_1(self, ref_pdb_1, ref_pkl_1):
        return {'pdb': ref_pdb_1, 'pkl': ref_pkl_1}

    def test_basic_mode(self, file_paths_1):
        row = process_single_complex('A0A0B4J2C3_P24534', file_paths_1)
        assert row['complex_name'] == 'A0A0B4J2C3_P24534'
        assert 'iptm' in row
        assert 'pdockq' in row
        assert 'quality_tier' in row

    def test_with_interface(self, file_paths_1):
        row = process_single_complex(
            'A0A0B4J2C3_P24534', file_paths_1, run_interface=True
        )
        assert 'n_interface_contacts' in row
        assert 'interface_plddt_combined' in row

    def test_with_pae(self, file_paths_1):
        row = process_single_complex(
            'A0A0B4J2C3_P24534', file_paths_1,
            run_interface=True, run_interface_pae=True
        )
        assert 'interface_pae_mean' in row
        assert 'confident_contact_fraction' in row
        assert 'interface_confidence_score' in row
        assert 'quality_tier_v2' in row

    def test_deterministic(self, file_paths_1):
        row1 = process_single_complex(
            'A0A0B4J2C3_P24534', file_paths_1,
            run_interface=True, run_interface_pae=True
        )
        row2 = process_single_complex(
            'A0A0B4J2C3_P24534', file_paths_1,
            run_interface=True, run_interface_pae=True
        )
        for key in row1:
            assert row1[key] == row2[key], f"Key '{key}' differs: {row1[key]} vs {row2[key]}"

    @pytest.mark.regression
    def test_regression(self, file_paths_1):
        row = process_single_complex(
            'A0A0B4J2C3_P24534', file_paths_1,
            run_interface=True, run_interface_pae=True
        )
        assert row['iptm'] == pytest.approx(EXPECTED_IPTM_1, abs=1e-6)
        assert row['pdockq'] == pytest.approx(EXPECTED_PDOCKQ_1, abs=1e-3)
        assert row['quality_tier'] == EXPECTED_QUALITY_TIER_1
        assert row['quality_tier_v2'] == EXPECTED_QUALITY_TIER_V2_1
        assert row['complex_type'] == EXPECTED_COMPLEX_TYPE_1
        assert row['n_interface_contacts'] == EXPECTED_N_CONTACTS_1
        assert row['interface_confidence_score'] == pytest.approx(EXPECTED_COMPOSITE_SCORE_1, abs=1e-3)


# ── CSV Output Tests ─────────────────────────────────────────────

@pytest.mark.slow
class TestWriteResultsCsv:
    """Tests for write_results_csv round-trip."""

    def test_creates_and_reads_back(self, test_output_dir, ref_pdb_1, ref_pkl_1):
        file_paths = {'pdb': ref_pdb_1, 'pkl': ref_pkl_1}
        row = process_single_complex('A0A0B4J2C3_P24534', file_paths)
        output = test_output_dir / "toolkit_roundtrip.csv"
        write_results_csv([row], str(output))
        assert output.exists()

        with open(output, 'r', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]['complex_name'] == 'A0A0B4J2C3_P24534'


# ── CLI Tests ────────────────────────────────────────────────────

@pytest.mark.cli
class TestToolkitCLI:
    """Tests for toolkit.py CLI."""

    def test_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "toolkit.py"), "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0


# ── Enrichment Tests ─────────────────────────────────────────────

class TestThreeToOne:
    """Tests for THREE_TO_ONE amino acid code mapping."""

    def test_all_20_amino_acids(self):
        """All 20 standard amino acids are present."""
        assert len(THREE_TO_ONE) == 20

    def test_ala_is_a(self):
        assert THREE_TO_ONE['ALA'] == 'A'

    def test_trp_is_w(self):
        assert THREE_TO_ONE['TRP'] == 'W'


class TestCSVFieldnamesEnrichment:
    """Tests for enrichment column constants."""

    def test_enrichment_columns_exist(self):
        assert 'gene_symbol_a' in CSV_FIELDNAMES_ENRICHMENT
        assert 'database_source' in CSV_FIELDNAMES_ENRICHMENT
        assert 'sequence_a' in CSV_FIELDNAMES_ENRICHMENT

    def test_with_enrichment_flag(self):
        fields = get_csv_fieldnames(include_enrichment=True)
        assert 'gene_symbol_a' in fields
        assert 'database_source' in fields

    def test_enrichment_after_base(self):
        """Enrichment columns come after base columns."""
        fields = get_csv_fieldnames(include_enrichment=True)
        base_end = fields.index(CSV_FIELDNAMES_BASE[-1])
        enrich_start = fields.index(CSV_FIELDNAMES_ENRICHMENT[0])
        assert enrich_start > base_end


@pytest.mark.slow
class TestConstantColumns:
    """Tests for species and structure_source constant columns."""

    @pytest.fixture(scope="class")
    def file_paths_1(self, ref_pdb_1, ref_pkl_1):
        return {'pdb': ref_pdb_1, 'pkl': ref_pkl_1}

    def test_species_populated(self, file_paths_1):
        row = process_single_complex('A0A0B4J2C3_P24534', file_paths_1)
        assert row['species'] == 'Homo sapiens (9606)'

    def test_structure_source_populated(self, file_paths_1):
        row = process_single_complex('A0A0B4J2C3_P24534', file_paths_1)
        assert row['structure_source'] == 'AlphaFold2_prediction'

    def test_species_in_base_fieldnames(self):
        assert 'species' in CSV_FIELDNAMES_BASE
        assert 'structure_source' in CSV_FIELDNAMES_BASE


@pytest.mark.slow
class TestSequenceExtraction:
    """Tests for amino acid sequence extraction from PDB."""

    @pytest.fixture(scope="class")
    def row_with_sequences(self, ref_pdb_1, ref_pkl_1):
        file_paths = {'pdb': ref_pdb_1, 'pkl': ref_pkl_1}
        return process_single_complex('A0A0B4J2C3_P24534', file_paths)

    def test_sequence_a_is_string(self, row_with_sequences):
        assert isinstance(row_with_sequences.get('sequence_a'), str)
        assert len(row_with_sequences['sequence_a']) > 0

    def test_sequence_b_is_string(self, row_with_sequences):
        assert isinstance(row_with_sequences.get('sequence_b'), str)
        assert len(row_with_sequences['sequence_b']) > 0

    def test_sequence_only_valid_chars(self, row_with_sequences):
        """Sequences contain only valid amino acid codes or X."""
        valid = set('ARNDCQEGHILKMFPSTWYV' + 'X')
        for seq_key in ('sequence_a', 'sequence_b'):
            seq = row_with_sequences[seq_key]
            invalid = set(seq) - valid
            assert not invalid, f"Invalid characters in {seq_key}: {invalid}"


@pytest.mark.slow
@pytest.mark.database
class TestEnrichResults:
    """Tests for enrich_results() function."""

    def test_enrich_with_mock_lookup(self):
        """Enrich adds gene symbols from lookup dict."""
        results = [{'protein_a': 'P04637', 'protein_b': 'Q00987'}]
        lookup = {
            'P04637': {
                'gene_symbol': 'TP53',
                'protein_name': 'Cellular tumor antigen p53',
                'ensembl_protein_id': 'ENSP00000269305',
                'ensembl_gene_id': 'ENSG00000141510',
                'secondary_accessions': 'O15129|Q53GA5',
            },
        }
        enrich_results(results, lookup)
        assert results[0]['gene_symbol_a'] == 'TP53'
        assert results[0]['gene_symbol_b'] == ''  # Q00987 not in lookup

    def test_enrich_secondary_accessions(self):
        """Enrich populates secondary_accessions from lookup dict."""
        results = [{'protein_a': 'P04637', 'protein_b': 'Q00987'}]
        lookup = {
            'P04637': {
                'gene_symbol': 'TP53',
                'protein_name': '',
                'ensembl_protein_id': 'ENSP00000269305',
                'ensembl_gene_id': '',
                'secondary_accessions': 'O15129|Q53GA5',
            },
            'Q00987': {
                'gene_symbol': 'MDM2',
                'protein_name': '',
                'ensembl_protein_id': 'ENSP00000258149',
                'ensembl_gene_id': '',
                'secondary_accessions': '',
            },
        }
        enrich_results(results, lookup)
        assert results[0]['secondary_accessions_a'] == 'O15129|Q53GA5'
        assert results[0]['secondary_accessions_b'] == ''

    def test_enrich_database_source(self):
        """Enrich adds database_source when pair sets are provided."""
        results = [{'protein_a': 'P04637', 'protein_b': 'Q00987'}]
        lookup = {}
        pair_sets = {
            'STRING': {('P04637', 'Q00987')},
            'BioGRID': {('P04637', 'Q00987')},
            'HuRI': set(),
        }
        enrich_results(results, lookup, database_pair_sets=pair_sets)
        assert 'STRING' in results[0]['database_source']
        assert 'BioGRID' in results[0]['database_source']
        assert 'HuRI' not in results[0]['database_source']

    def test_enrich_database_evidence_types(self):
        """Enrich populates evidence_types from pre-computed evidence sets."""
        results = [{'protein_a': 'P04637', 'protein_b': 'Q00987'}]
        lookup = {}
        pair_sets = {
            'STRING': {('P04637', 'Q00987')},
            'BioGRID': {('P04637', 'Q00987')},
        }
        evidence = {
            'STRING': {'combined_score'},
            'BioGRID': {'physical', 'co-fractionation'},
        }
        enrich_results(results, lookup, pair_sets, evidence)
        ev = results[0]['evidence_types']
        assert 'combined_score' in ev
        assert 'physical' in ev
        assert 'co-fractionation' in ev

    def test_enrich_without_databases(self):
        """Enrich without database_pair_sets leaves source empty."""
        results = [{'protein_a': 'P04637', 'protein_b': 'Q00987'}]
        enrich_results(results, {})
        assert results[0]['database_source'] == ''
        assert results[0]['evidence_types'] == ''
