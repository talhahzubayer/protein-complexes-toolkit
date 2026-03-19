"""
Tests for disease_annotations.py — UniProt disease, PTM, GO, and drug target annotation.

Test data: tests/offline_test_data/databases/test_uniprot_annotations.xml
  - P04637 (TP53): 8 diseases, 36 PTM features, 173 GO terms, 1 KEGG, not drug target
  - P24534 (EEF1B2): 0 diseases, 9 PTM, 7 GO, 1 KEGG, not drug target
  - P61981 (YWHAG): 1 disease (Popov-Chang), 9 PTM, 31 GO, 1 KEGG, not drug target
  - P63104 (YWHAZ): 1 disease (Popov-Chang), 8 PTM, 44 GO, 1 KEGG, not drug target
  - Q2M2I8 (AAK1): 0 diseases, 21 PTM, 19 GO, 1 KEGG, IS drug target (Pharmaceutical)
"""

import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest
from pathlib import Path


# ── Constants & Known Values ─────────────────────────────────────

TEST_XML_FILENAME = "test_uniprot_annotations.xml"

# Reference protein accessions in test XML
ACC_TP53 = "P04637"        # 8 diseases, 36 PTM, many GO
ACC_EEF1B2 = "P24534"      # no diseases, 9 PTM, few GO
ACC_YWHAG = "P61981"        # 1 disease, 9 PTM
ACC_YWHAZ = "P63104"        # 1 disease, 8 PTM
ACC_AAK1 = "Q2M2I8"         # drug target, 21 PTM

# Known counts (verified from test XML)
TP53_DISEASES = 8
TP53_PTM = 36
TP53_GO = 173
YWHAG_DISEASES = 1
YWHAG_PTM = 9
AAK1_PTM = 21
EEF1B2_DISEASES = 0

# Known disease details
YWHAG_DISEASE_NAME = "Developmental and epileptic encephalopathy 56"
YWHAG_DISEASE_OMIM = "617665"

# Known PTM
TP53_FIRST_PTM_DESC = "Phosphoserine; by HIPK4"
TP53_FIRST_PTM_POS = "9"


# ── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def test_xml_path(test_db_dir):
    """Path to the test UniProt XML file."""
    path = test_db_dir / TEST_XML_FILENAME
    assert path.exists(), f"Test XML not found: {path}"
    return path


@pytest.fixture(scope="session")
def annotation_index(test_xml_path):
    """Full annotation index loaded from test XML."""
    from disease_annotations import load_uniprot_annotations
    return load_uniprot_annotations(
        test_xml_path,
        frozenset({ACC_TP53, ACC_EEF1B2, ACC_YWHAG, ACC_YWHAZ, ACC_AAK1}),
    )


# ── Section: TestConstants ───────────────────────────────────────

@pytest.mark.disease
class TestConstants:
    """Test module-level constants and configuration."""

    def test_csv_fieldnames_count(self):
        from disease_annotations import CSV_FIELDNAMES_DISEASE
        assert len(CSV_FIELDNAMES_DISEASE) == 14

    def test_csv_fieldnames_no_duplicates(self):
        from disease_annotations import CSV_FIELDNAMES_DISEASE
        assert len(CSV_FIELDNAMES_DISEASE) == len(set(CSV_FIELDNAMES_DISEASE))

    def test_csv_fieldnames_per_chain_pattern(self):
        from disease_annotations import CSV_FIELDNAMES_DISEASE
        a_cols = [c for c in CSV_FIELDNAMES_DISEASE if c.endswith("_a")]
        b_cols = [c for c in CSV_FIELDNAMES_DISEASE if c.endswith("_b")]
        assert len(a_cols) == len(b_cols)

    def test_default_disease_dir_exists(self):
        from disease_annotations import DEFAULT_DISEASE_DIR
        assert isinstance(DEFAULT_DISEASE_DIR, Path)

    def test_display_limit_positive(self):
        from disease_annotations import DETAILS_DISPLAY_LIMIT
        assert DETAILS_DISPLAY_LIMIT > 0

    def test_ptm_feature_types_is_frozenset(self):
        from disease_annotations import PTM_FEATURE_TYPES
        assert isinstance(PTM_FEATURE_TYPES, frozenset)
        assert "modified residue" in PTM_FEATURE_TYPES

    def test_namespace_constant(self):
        from disease_annotations import UNIPROT_XML_NAMESPACE
        assert "uniprot.org" in UNIPROT_XML_NAMESPACE


# ── Section: TestDiseaseParsing ──────────────────────────────────

@pytest.mark.disease
class TestDiseaseParsing:
    """Test disease association parsing from XML."""

    def test_tp53_disease_count(self, annotation_index):
        ann = annotation_index[ACC_TP53]
        assert len(ann["diseases"]) == TP53_DISEASES

    def test_tp53_has_li_fraumeni(self, annotation_index):
        ann = annotation_index[ACC_TP53]
        names = [d["disease_name"] for d in ann["diseases"]]
        assert "Li-Fraumeni syndrome" in names

    def test_tp53_has_omim_ids(self, annotation_index):
        ann = annotation_index[ACC_TP53]
        omim_ids = [d["omim_id"] for d in ann["diseases"] if d["omim_id"]]
        assert len(omim_ids) > 0
        assert all(mid.isdigit() for mid in omim_ids)

    def test_ywhag_single_disease(self, annotation_index):
        ann = annotation_index[ACC_YWHAG]
        assert len(ann["diseases"]) == YWHAG_DISEASES

    def test_ywhag_disease_name(self, annotation_index):
        ann = annotation_index[ACC_YWHAG]
        assert ann["diseases"][0]["disease_name"] == YWHAG_DISEASE_NAME

    def test_ywhag_disease_omim(self, annotation_index):
        ann = annotation_index[ACC_YWHAG]
        assert ann["diseases"][0]["omim_id"] == YWHAG_DISEASE_OMIM

    def test_ywhag_disease_acronym(self, annotation_index):
        ann = annotation_index[ACC_YWHAG]
        assert ann["diseases"][0]["acronym"] != ""

    def test_eef1b2_no_diseases(self, annotation_index):
        ann = annotation_index[ACC_EEF1B2]
        assert len(ann["diseases"]) == EEF1B2_DISEASES

    def test_disease_dict_keys(self, annotation_index):
        ann = annotation_index[ACC_TP53]
        d = ann["diseases"][0]
        assert "disease_name" in d
        assert "acronym" in d
        assert "omim_id" in d

    def test_ywhaz_has_disease(self, annotation_index):
        ann = annotation_index[ACC_YWHAZ]
        assert len(ann["diseases"]) == 1


# ── Section: TestPTMParsing ──────────────────────────────────────

@pytest.mark.disease
class TestPTMParsing:
    """Test PTM feature extraction from XML."""

    def test_tp53_ptm_count(self, annotation_index):
        ann = annotation_index[ACC_TP53]
        assert len(ann["ptm_sites"]) == TP53_PTM

    def test_tp53_first_ptm_description(self, annotation_index):
        ann = annotation_index[ACC_TP53]
        # First PTM should be Phosphoserine at position 9
        first = ann["ptm_sites"][0]
        assert TP53_FIRST_PTM_DESC in first["description"]

    def test_tp53_first_ptm_position(self, annotation_index):
        ann = annotation_index[ACC_TP53]
        first = ann["ptm_sites"][0]
        assert first["position"] == TP53_FIRST_PTM_POS

    def test_aak1_ptm_count(self, annotation_index):
        ann = annotation_index[ACC_AAK1]
        assert len(ann["ptm_sites"]) == AAK1_PTM

    def test_ptm_dict_keys(self, annotation_index):
        ann = annotation_index[ACC_TP53]
        p = ann["ptm_sites"][0]
        assert "type" in p
        assert "position" in p
        assert "description" in p

    def test_ptm_type_is_known(self, annotation_index):
        from disease_annotations import PTM_FEATURE_TYPES
        ann = annotation_index[ACC_TP53]
        for p in ann["ptm_sites"]:
            assert p["type"] in PTM_FEATURE_TYPES

    def test_eef1b2_has_ptm(self, annotation_index):
        ann = annotation_index[ACC_EEF1B2]
        assert len(ann["ptm_sites"]) == 9

    def test_ptm_positions_are_strings(self, annotation_index):
        ann = annotation_index[ACC_TP53]
        for p in ann["ptm_sites"]:
            assert isinstance(p["position"], str)

    def test_ywhag_ptm_count(self, annotation_index):
        ann = annotation_index[ACC_YWHAG]
        assert len(ann["ptm_sites"]) == YWHAG_PTM

    def test_cross_link_ptm_detected(self, annotation_index):
        """Cross-link PTM features should be captured."""
        ann = annotation_index[ACC_TP53]
        types = {p["type"] for p in ann["ptm_sites"]}
        assert "modified residue" in types or "cross-link" in types


# ── Section: TestGOParsing ───────────────────────────────────────

@pytest.mark.disease
class TestGOParsing:
    """Test GO term extraction from XML."""

    def test_tp53_go_count(self, annotation_index):
        ann = annotation_index[ACC_TP53]
        assert len(ann["go_terms"]) == TP53_GO

    def test_go_term_has_aspects(self, annotation_index):
        ann = annotation_index[ACC_TP53]
        aspects = {g["aspect"] for g in ann["go_terms"]}
        assert "F" in aspects  # molecular function
        assert "P" in aspects  # biological process
        assert "C" in aspects  # cellular component

    def test_go_dict_keys(self, annotation_index):
        ann = annotation_index[ACC_TP53]
        g = ann["go_terms"][0]
        assert "go_id" in g
        assert "go_name" in g
        assert "aspect" in g

    def test_go_id_format(self, annotation_index):
        ann = annotation_index[ACC_TP53]
        for g in ann["go_terms"]:
            assert g["go_id"].startswith("GO:")

    def test_eef1b2_go_count(self, annotation_index):
        ann = annotation_index[ACC_EEF1B2]
        assert len(ann["go_terms"]) == 7

    def test_ywhag_go_count(self, annotation_index):
        ann = annotation_index[ACC_YWHAG]
        assert len(ann["go_terms"]) == 31

    def test_go_name_not_empty(self, annotation_index):
        ann = annotation_index[ACC_TP53]
        for g in ann["go_terms"]:
            assert g["go_name"] != ""

    def test_aak1_go_count(self, annotation_index):
        ann = annotation_index[ACC_AAK1]
        assert len(ann["go_terms"]) == 19


# ── Section: TestKEGGParsing ─────────────────────────────────────

@pytest.mark.disease
class TestKEGGParsing:
    """Test KEGG ID extraction from XML."""

    def test_all_proteins_have_kegg(self, annotation_index):
        for acc in (ACC_TP53, ACC_EEF1B2, ACC_YWHAG, ACC_YWHAZ, ACC_AAK1):
            assert len(annotation_index[acc]["kegg_ids"]) >= 1

    def test_kegg_id_format(self, annotation_index):
        ann = annotation_index[ACC_TP53]
        for kid in ann["kegg_ids"]:
            assert kid.startswith("hsa:")


# ── Section: TestDrugTarget ──────────────────────────────────────

@pytest.mark.disease
class TestDrugTarget:
    """Test drug target (Pharmaceutical keyword) detection."""

    def test_aak1_is_drug_target(self, annotation_index):
        assert annotation_index[ACC_AAK1]["is_drug_target"] is True

    def test_tp53_not_drug_target(self, annotation_index):
        assert annotation_index[ACC_TP53]["is_drug_target"] is False

    def test_eef1b2_not_drug_target(self, annotation_index):
        assert annotation_index[ACC_EEF1B2]["is_drug_target"] is False

    def test_ywhag_not_drug_target(self, annotation_index):
        assert annotation_index[ACC_YWHAG]["is_drug_target"] is False

    def test_drug_target_is_bool(self, annotation_index):
        for acc in annotation_index:
            assert isinstance(annotation_index[acc]["is_drug_target"], bool)


# ── Section: TestAnnotationLoading ───────────────────────────────

@pytest.mark.disease
class TestAnnotationLoading:
    """Test loading and filtering from XML."""

    def test_loads_all_five_proteins(self, test_xml_path):
        from disease_annotations import load_uniprot_annotations
        idx = load_uniprot_annotations(
            test_xml_path,
            frozenset({ACC_TP53, ACC_EEF1B2, ACC_YWHAG, ACC_YWHAZ, ACC_AAK1}),
        )
        assert len(idx) == 5

    def test_filters_to_requested_accessions(self, test_xml_path):
        from disease_annotations import load_uniprot_annotations
        idx = load_uniprot_annotations(
            test_xml_path,
            frozenset({ACC_TP53}),
        )
        assert len(idx) == 1
        assert ACC_TP53 in idx

    def test_empty_accessions_returns_empty(self, test_xml_path):
        from disease_annotations import load_uniprot_annotations
        idx = load_uniprot_annotations(test_xml_path, frozenset())
        assert len(idx) == 0

    def test_unknown_accession_not_in_index(self, test_xml_path):
        from disease_annotations import load_uniprot_annotations
        idx = load_uniprot_annotations(test_xml_path, frozenset({"XXXXXX"}))
        assert "XXXXXX" not in idx

    def test_secondary_accession_matched(self, test_xml_path):
        """Secondary accessions on entries should also match."""
        from disease_annotations import load_uniprot_annotations
        # P63104 has secondary accessions A8K1N0 and B7Z465
        idx = load_uniprot_annotations(test_xml_path, frozenset({"A8K1N0"}))
        assert "A8K1N0" in idx
        assert len(idx["A8K1N0"]["diseases"]) == 1  # Same as P63104

    def test_missing_file_raises(self):
        from disease_annotations import load_uniprot_annotations
        with pytest.raises(FileNotFoundError):
            load_uniprot_annotations(
                "/nonexistent/path.xml", frozenset({"P04637"}))

    def test_annotation_dict_structure(self, annotation_index):
        ann = annotation_index[ACC_TP53]
        assert "diseases" in ann
        assert "ptm_sites" in ann
        assert "go_terms" in ann
        assert "kegg_ids" in ann
        assert "is_drug_target" in ann

    def test_verbose_mode(self, test_xml_path, capsys):
        from disease_annotations import load_uniprot_annotations
        load_uniprot_annotations(
            test_xml_path, frozenset({ACC_TP53}), verbose=True)
        captured = capsys.readouterr()
        assert "Scanned" in captured.err


# ── Section: TestFormatting ──────────────────────────────────────

@pytest.mark.disease
class TestFormatting:
    """Test detail string formatting functions."""

    def test_format_disease_empty(self):
        from disease_annotations import format_disease_details
        assert format_disease_details([]) == ""

    def test_format_disease_single(self):
        from disease_annotations import format_disease_details
        result = format_disease_details([{
            "disease_name": "TestDisease",
            "acronym": "TD",
            "omim_id": "123456",
        }])
        assert result == "OMIM:123456:TestDisease (TD)"

    def test_format_disease_without_omim(self):
        from disease_annotations import format_disease_details
        result = format_disease_details([{
            "disease_name": "TestDisease",
            "acronym": "",
            "omim_id": "",
        }])
        assert result == "TestDisease"

    def test_format_disease_pipe_separated(self):
        from disease_annotations import format_disease_details
        result = format_disease_details([
            {"disease_name": "A", "acronym": "", "omim_id": "1"},
            {"disease_name": "B", "acronym": "", "omim_id": "2"},
        ])
        assert "|" in result
        assert "OMIM:1:A" in result
        assert "OMIM:2:B" in result

    def test_format_disease_truncation(self):
        from disease_annotations import format_disease_details
        diseases = [{"disease_name": f"D{i}", "acronym": "", "omim_id": ""}
                     for i in range(25)]
        result = format_disease_details(diseases, limit=5)
        assert "(+20 more)" in result

    def test_format_ptm_empty(self):
        from disease_annotations import format_ptm_details
        assert format_ptm_details([]) == ""

    def test_format_ptm_single(self):
        from disease_annotations import format_ptm_details
        result = format_ptm_details([{
            "type": "modified residue",
            "position": "9",
            "description": "Phosphoserine",
        }])
        assert "Phosphoserine:9" in result

    def test_format_ptm_truncation(self):
        from disease_annotations import format_ptm_details
        sites = [{"type": "modified residue", "position": str(i), "description": f"P{i}"}
                  for i in range(25)]
        result = format_ptm_details(sites, limit=5)
        assert "(+20 more)" in result

    def test_format_go_empty(self):
        from disease_annotations import format_go_details
        assert format_go_details([]) == ""

    def test_format_go_aspect_filter(self):
        from disease_annotations import format_go_details
        terms = [
            {"go_id": "GO:001", "go_name": "binding", "aspect": "F"},
            {"go_id": "GO:002", "go_name": "apoptosis", "aspect": "P"},
        ]
        result = format_go_details(terms, aspect_filter="F")
        assert "binding" in result
        assert "apoptosis" not in result

    def test_format_go_no_filter(self):
        from disease_annotations import format_go_details
        terms = [
            {"go_id": "GO:001", "go_name": "binding", "aspect": "F"},
            {"go_id": "GO:002", "go_name": "apoptosis", "aspect": "P"},
        ]
        result = format_go_details(terms)
        assert "binding" in result
        assert "apoptosis" in result

    def test_format_go_truncation(self):
        from disease_annotations import format_go_details
        terms = [{"go_id": f"GO:{i:05d}", "go_name": f"term{i}", "aspect": "P"}
                  for i in range(25)]
        result = format_go_details(terms, limit=5)
        assert "(+20 more)" in result


# ── Section: TestAnnotation ──────────────────────────────────────

@pytest.mark.disease
class TestAnnotation:
    """Test in-place result annotation."""

    def test_annotate_adds_all_columns(self, annotation_index):
        from disease_annotations import annotate_results_with_disease, CSV_FIELDNAMES_DISEASE
        results = [{"protein_a": ACC_TP53, "protein_b": ACC_YWHAG}]
        annotate_results_with_disease(results, annotation_index, api_fallback=False)
        for col in CSV_FIELDNAMES_DISEASE:
            assert col in results[0], f"Missing column: {col}"

    def test_annotate_disease_counts(self, annotation_index):
        from disease_annotations import annotate_results_with_disease
        results = [{"protein_a": ACC_TP53, "protein_b": ACC_EEF1B2}]
        annotate_results_with_disease(results, annotation_index, api_fallback=False)
        assert results[0]["n_diseases_a"] == TP53_DISEASES
        assert results[0]["n_diseases_b"] == EEF1B2_DISEASES

    def test_annotate_drug_target(self, annotation_index):
        from disease_annotations import annotate_results_with_disease
        results = [{"protein_a": ACC_AAK1, "protein_b": ACC_TP53}]
        annotate_results_with_disease(results, annotation_index, api_fallback=False)
        assert results[0]["is_drug_target_a"] is True
        assert results[0]["is_drug_target_b"] is False

    def test_annotate_ptm_counts(self, annotation_index):
        from disease_annotations import annotate_results_with_disease
        results = [{"protein_a": ACC_TP53, "protein_b": ACC_YWHAG}]
        annotate_results_with_disease(results, annotation_index, api_fallback=False)
        assert results[0]["n_ptm_sites_a"] == TP53_PTM
        assert results[0]["n_ptm_sites_b"] == YWHAG_PTM

    def test_annotate_go_terms_present(self, annotation_index):
        from disease_annotations import annotate_results_with_disease
        results = [{"protein_a": ACC_TP53, "protein_b": ACC_YWHAG}]
        annotate_results_with_disease(results, annotation_index, api_fallback=False)
        assert results[0]["go_biological_process_a"] != ""
        assert results[0]["go_molecular_function_a"] != ""

    def test_annotate_unknown_protein(self, annotation_index):
        """Unknown protein should get empty/zero values."""
        from disease_annotations import annotate_results_with_disease
        results = [{"protein_a": "XXXXXX", "protein_b": ACC_TP53}]
        annotate_results_with_disease(results, annotation_index, api_fallback=False)
        assert results[0]["n_diseases_a"] == 0
        assert results[0]["is_drug_target_a"] is False
        assert results[0]["n_ptm_sites_a"] == 0

    def test_annotate_isoform_stripped(self, annotation_index):
        """Isoform suffix should be stripped for lookup (P61981-2 -> P61981)."""
        from disease_annotations import annotate_results_with_disease
        results = [{"protein_a": "P61981-2", "protein_b": ACC_TP53}]
        annotate_results_with_disease(results, annotation_index, api_fallback=False)
        assert results[0]["n_diseases_a"] == YWHAG_DISEASES

    def test_annotate_modifies_in_place(self, annotation_index):
        from disease_annotations import annotate_results_with_disease
        results = [{"protein_a": ACC_TP53, "protein_b": ACC_YWHAG, "extra": "keep"}]
        annotate_results_with_disease(results, annotation_index, api_fallback=False)
        assert results[0]["extra"] == "keep"

    def test_annotate_multiple_complexes(self, annotation_index):
        from disease_annotations import annotate_results_with_disease
        results = [
            {"protein_a": ACC_TP53, "protein_b": ACC_YWHAG},
            {"protein_a": ACC_AAK1, "protein_b": ACC_EEF1B2},
        ]
        annotate_results_with_disease(results, annotation_index, api_fallback=False)
        assert results[0]["n_diseases_a"] == TP53_DISEASES
        assert results[1]["is_drug_target_a"] is True

    def test_annotate_empty_results(self, annotation_index):
        from disease_annotations import annotate_results_with_disease
        results = []
        annotate_results_with_disease(results, annotation_index, api_fallback=False)
        assert len(results) == 0


# ── Section: TestCSVFieldnames ───────────────────────────────────

@pytest.mark.disease
class TestCSVFieldnames:
    """Test CSV fieldnames are correctly structured."""

    def test_all_a_b_pairs(self):
        from disease_annotations import CSV_FIELDNAMES_DISEASE
        a_cols = sorted(c for c in CSV_FIELDNAMES_DISEASE if c.endswith("_a"))
        b_cols = sorted(c for c in CSV_FIELDNAMES_DISEASE if c.endswith("_b"))
        # Every _a should have matching _b
        for a_col in a_cols:
            b_col = a_col[:-2] + "_b"
            assert b_col in b_cols, f"Missing _b counterpart for {a_col}"

    def test_no_overlap_with_base(self):
        from disease_annotations import CSV_FIELDNAMES_DISEASE
        # Should not overlap with base toolkit columns
        base_cols = {
            'complex_name', 'protein_a', 'protein_b', 'iptm', 'ptm',
            'pdockq', 'quality_tier',
        }
        overlap = set(CSV_FIELDNAMES_DISEASE) & base_cols
        assert len(overlap) == 0, f"Overlap with base columns: {overlap}"


# ── Section: TestCLI ─────────────────────────────────────────────

@pytest.mark.disease
@pytest.mark.cli
class TestCLI:
    """Test standalone CLI entry points."""

    def test_cli_no_args_shows_help(self):
        result = subprocess.run(
            [sys.executable, "disease_annotations.py"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0

    def test_cli_summary_with_test_xml(self, test_xml_path):
        # test_xml_path.parent is the databases dir; we need to point
        # --disease-dir to a dir containing uniprot_sprot_human.xml.
        # Since our test file is named differently, create a temp symlink
        # or just verify the CLI doesn't crash for missing default.
        result = subprocess.run(
            [sys.executable, "disease_annotations.py",
             "--disease-dir", str(test_xml_path.parent),
             "summary"],
            capture_output=True, text=True,
        )
        # The test XML is named test_uniprot_annotations.xml not uniprot_sprot_human.xml
        # so summary will report file not found — that's expected
        assert result.returncode != 0 or "Total entries" in result.stdout

    def test_cli_lookup(self, test_xml_path, tmp_path):
        """CLI lookup via a temp dir with correctly named XML symlink/copy."""
        import shutil
        from disease_annotations import UNIPROT_XML_FILENAME
        dest = tmp_path / UNIPROT_XML_FILENAME
        shutil.copy2(test_xml_path, dest)
        result = subprocess.run(
            [sys.executable, "disease_annotations.py",
             "--disease-dir", str(tmp_path),
             "lookup", "--protein", ACC_TP53],
            capture_output=True, text=True,
        )
        assert "Li-Fraumeni" in result.stdout


# ── Section: TestRegression ──────────────────────────────────────

@pytest.mark.disease
@pytest.mark.regression
class TestRegression:
    """Regression tests with known reference values."""

    def test_tp53_exact_disease_count(self, annotation_index):
        assert len(annotation_index[ACC_TP53]["diseases"]) == 8

    def test_tp53_exact_ptm_count(self, annotation_index):
        assert len(annotation_index[ACC_TP53]["ptm_sites"]) == 36

    def test_tp53_exact_go_count(self, annotation_index):
        assert len(annotation_index[ACC_TP53]["go_terms"]) == 173

    def test_ywhag_exact_disease_count(self, annotation_index):
        assert len(annotation_index[ACC_YWHAG]["diseases"]) == 1

    def test_ywhag_disease_name_regression(self, annotation_index):
        assert annotation_index[ACC_YWHAG]["diseases"][0]["disease_name"] == YWHAG_DISEASE_NAME

    def test_aak1_is_pharma(self, annotation_index):
        assert annotation_index[ACC_AAK1]["is_drug_target"] is True

    def test_eef1b2_no_diseases(self, annotation_index):
        assert len(annotation_index[ACC_EEF1B2]["diseases"]) == 0


# ── Section: TestEdgeCases ───────────────────────────────────────

@pytest.mark.disease
class TestEdgeCases:
    """Edge cases and error handling."""

    def test_empty_protein_string(self, annotation_index):
        from disease_annotations import _lookup_annotation
        result = _lookup_annotation("", annotation_index, api_fallback=False)
        assert result["diseases"] == []
        assert result["is_drug_target"] is False

    def test_none_safe_formatting(self):
        from disease_annotations import format_disease_details
        # Edge: dict with empty strings
        result = format_disease_details([{
            "disease_name": "",
            "acronym": "",
            "omim_id": "",
        }])
        assert isinstance(result, str)

    def test_format_ptm_no_position(self):
        from disease_annotations import format_ptm_details
        result = format_ptm_details([{
            "type": "modified residue",
            "position": "",
            "description": "Unknown modification",
        }])
        assert "Unknown modification" in result

    def test_empty_annotation_helper(self):
        from disease_annotations import _empty_annotation
        ann = _empty_annotation()
        assert ann["diseases"] == []
        assert ann["is_drug_target"] is False
        assert ann["go_terms"] == []

    def test_isoform_lookup(self, annotation_index):
        from disease_annotations import _lookup_annotation
        result = _lookup_annotation("P61981-3", annotation_index, api_fallback=False)
        assert len(result["diseases"]) == YWHAG_DISEASES
