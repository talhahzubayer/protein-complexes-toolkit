#!/usr/bin/env python3
"""
Tests for foldx_scorer.py — FoldX local stability prediction module.

All tests use offline data and mocked subprocess calls. No real FoldX binary
is executed.

Test data in tests/offline_test_data/databases/foldx_outputs/:
    Dif_test_Repair.fxout              — BuildModel difference output (3 runs)
    Interaction_test_Repair_AC.fxout   — AnalyseComplex interaction output
"""

import json
import pytest
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from foldx_scorer import (
    # Constants
    DEFAULT_FOLDX_BINARY,
    DEFAULT_ROTABASE,
    DEFAULT_FOLDX_CACHE_DIR,
    FOLDX_NUMBER_OF_RUNS,
    FOLDX_TIMEOUT_SECONDS,
    FOLDX_DESTABILISING_THRESHOLD,
    FOLDX_ELIGIBLE_CONTEXTS,
    FOLDX_ELIGIBLE_CLINICAL,
    FOLDX_MIN_PLDDT,
    FOLDX_QUALITY_TIERS,
    FOLDX_DETAILS_DISPLAY_LIMIT,
    _VARIANT_DETAIL_PATTERN,
    CSV_FIELDNAMES_FOLDX,
    # Exception
    FoldXError,
    # Binary validation
    validate_foldx_binary,
    # Caching
    _foldx_cache_key,
    _read_foldx_cache,
    _write_foldx_cache,
    # Variant filtering
    _parse_variant_detail_parts,
    _filter_variants_for_foldx,
    # PDB preparation
    _strip_remarks,
    _run_repair_pdb,
    # FoldX execution
    _format_foldx_mutation,
    _write_individual_list,
    _run_buildmodel,
    _parse_buildmodel_output,
    _parse_analysecomplex_output,
    # DDG computation
    compute_ddg_for_variant,
    compute_ddg_batch,
    # Annotation
    format_foldx_details,
    _score_chain_variants_foldx,
    annotate_results_with_foldx,
    # CLI
    build_argument_parser,
    _cli_summary,
)

# ── Known Test Values ────────────────────────────────────────────────

# DDG values from Dif_test_Repair.fxout (3 runs)
TEST_DDG_RUN_1 = 2.34
TEST_DDG_RUN_2 = 2.51
TEST_DDG_RUN_3 = 2.18
TEST_DDG_MEAN = round((TEST_DDG_RUN_1 + TEST_DDG_RUN_2 + TEST_DDG_RUN_3) / 3, 4)

# Interaction energy from Interaction_test_Repair_AC.fxout
TEST_INTERACTION_ENERGY = -8.45

# Sample variant details string (from variant_mapper format)
SAMPLE_DETAILS = "K81P:interface_core:pathogenic|E82K:interface_rim:VUS|R50A:surface_non_interface:benign|G10S:buried_core:likely_pathogenic"
SAMPLE_DETAILS_TRUNCATED = "K81P:interface_core:pathogenic|E82K:interface_rim:VUS|...(+5 more)"


# ═════════════════════════════════════════════════════════════════════
#  Test Classes
# ═════════════════════════════════════════════════════════════════════

@pytest.mark.foldx
class TestConstants:
    """Test module-level constants and CSV fieldnames."""

    def test_csv_fieldnames_count(self):
        """CSV_FIELDNAMES_FOLDX has exactly 8 columns."""
        assert len(CSV_FIELDNAMES_FOLDX) == 8

    def test_csv_fieldnames_all_have_suffix(self):
        """All FoldX CSV fieldnames end with _a or _b."""
        for name in CSV_FIELDNAMES_FOLDX:
            assert name.endswith('_a') or name.endswith('_b'), \
                f"{name} does not end with _a or _b"

    def test_csv_fieldnames_paired(self):
        """FoldX CSV fieldnames come in a/b pairs."""
        stems = set()
        for name in CSV_FIELDNAMES_FOLDX:
            stem = name[:-2]  # Remove _a or _b
            stems.add(stem)
        assert len(stems) == 4  # 4 unique stems: ddg_mean, n_destabilising, coverage, details

    def test_destabilising_threshold(self):
        """Destabilising threshold is 1.6 kcal/mol (convention)."""
        assert FOLDX_DESTABILISING_THRESHOLD == 1.6

    def test_number_of_runs(self):
        """Default number of FoldX BuildModel runs is 3."""
        assert FOLDX_NUMBER_OF_RUNS == 3

    def test_eligible_contexts(self):
        """Only interface_core and interface_rim are eligible."""
        assert FOLDX_ELIGIBLE_CONTEXTS == frozenset({'interface_core', 'interface_rim'})

    def test_quality_tiers(self):
        """Only High and Medium tiers are eligible."""
        assert FOLDX_QUALITY_TIERS == frozenset({'High', 'Medium'})

    def test_variant_detail_pattern(self):
        """Regex pattern matches REF{POS}ALT: format."""
        match = _VARIANT_DETAIL_PATTERN.match("K81P:interface_core:pathogenic")
        assert match is not None
        assert match.group(1) == 'K'
        assert match.group(2) == '81'
        assert match.group(3) == 'P'

    def test_default_foldx_binary_path(self):
        """Default binary path points to foldx5_Windows directory."""
        assert 'foldx5_Windows' in str(DEFAULT_FOLDX_BINARY)
        assert str(DEFAULT_FOLDX_BINARY).endswith('.exe')

    def test_default_cache_dir(self):
        """Default cache dir is data/foldx_cache."""
        assert str(DEFAULT_FOLDX_CACHE_DIR).endswith('foldx_cache')


@pytest.mark.foldx
class TestBinaryValidation:
    """Tests for validate_foldx_binary()."""

    def test_valid_binary_no_error(self):
        """No error when both binary and rotabase exist."""
        # The real binary is in the repo
        if DEFAULT_FOLDX_BINARY.exists() and DEFAULT_ROTABASE.exists():
            validate_foldx_binary(DEFAULT_FOLDX_BINARY, DEFAULT_ROTABASE)
        else:
            pytest.skip("FoldX binary not present")

    def test_missing_binary_raises(self, tmp_path):
        """FoldXError raised when binary doesn't exist."""
        fake_binary = tmp_path / "nonexistent.exe"
        with pytest.raises(FoldXError, match="FoldX binary not found"):
            validate_foldx_binary(fake_binary, DEFAULT_ROTABASE)

    def test_missing_rotabase_raises(self, tmp_path):
        """FoldXError raised when rotabase.txt doesn't exist."""
        # Create a fake binary
        fake_binary = tmp_path / "foldx.exe"
        fake_binary.write_text("fake")
        fake_rotabase = tmp_path / "missing_rotabase.txt"
        with pytest.raises(FoldXError, match="rotabase.txt not found"):
            validate_foldx_binary(fake_binary, fake_rotabase)


@pytest.mark.foldx
class TestCaching:
    """Tests for _foldx_cache_key, _read_foldx_cache, _write_foldx_cache."""

    def test_cache_key_deterministic(self, tmp_path):
        """Same inputs produce same cache key."""
        pdb = tmp_path / "test.pdb"
        pdb.write_text("ATOM test")
        key1 = _foldx_cache_key(pdb, 'A', 81, 'K', 'P')
        key2 = _foldx_cache_key(pdb, 'A', 81, 'K', 'P')
        assert key1 == key2

    def test_cache_key_changes_with_position(self, tmp_path):
        """Different positions produce different cache keys."""
        pdb = tmp_path / "test.pdb"
        pdb.write_text("ATOM test")
        key1 = _foldx_cache_key(pdb, 'A', 81, 'K', 'P')
        key2 = _foldx_cache_key(pdb, 'A', 82, 'K', 'P')
        assert key1 != key2

    def test_cache_key_changes_with_mutant(self, tmp_path):
        """Different mutants produce different cache keys."""
        pdb = tmp_path / "test.pdb"
        pdb.write_text("ATOM test")
        key1 = _foldx_cache_key(pdb, 'A', 81, 'K', 'P')
        key2 = _foldx_cache_key(pdb, 'A', 81, 'K', 'A')
        assert key1 != key2

    def test_cache_key_changes_with_chain(self, tmp_path):
        """Different chains produce different cache keys."""
        pdb = tmp_path / "test.pdb"
        pdb.write_text("ATOM test")
        key1 = _foldx_cache_key(pdb, 'A', 81, 'K', 'P')
        key2 = _foldx_cache_key(pdb, 'B', 81, 'K', 'P')
        assert key1 != key2

    def test_write_then_read(self, tmp_path):
        """Written DDG value can be read back."""
        _write_foldx_cache(tmp_path, "testkey123", 2.34, "KA81P")
        result = _read_foldx_cache(tmp_path, "testkey123")
        assert result == pytest.approx(2.34)

    def test_read_missing_returns_none(self, tmp_path):
        """Cache miss returns None."""
        result = _read_foldx_cache(tmp_path, "nonexistent_key")
        assert result is None

    def test_cache_creates_directory(self, tmp_path):
        """Writing to a non-existent directory creates it."""
        cache_dir = tmp_path / "new_cache"
        _write_foldx_cache(cache_dir, "key1", 1.5, "KA81P")
        assert cache_dir.exists()
        assert (cache_dir / "key1.json").exists()

    def test_cache_file_contains_metadata(self, tmp_path):
        """Cache file contains mutation metadata."""
        _write_foldx_cache(tmp_path, "meta_key", 3.14, "RA4G")
        with open(tmp_path / "meta_key.json") as f:
            data = json.load(f)
        assert data['ddg'] == pytest.approx(3.14)
        assert data['_mutation'] == 'RA4G'
        assert '_timestamp' in data


@pytest.mark.foldx
class TestVariantFiltering:
    """Tests for _parse_variant_detail_parts() and _filter_variants_for_foldx()."""

    def test_parse_standard_details(self):
        """Parse a standard variant details string."""
        parts = _parse_variant_detail_parts(
            "K81P:interface_core:pathogenic|E82K:interface_rim:VUS"
        )
        assert len(parts) == 2
        assert parts[0]['ref_aa'] == 'K'
        assert parts[0]['position'] == 81
        assert parts[0]['alt_aa'] == 'P'
        assert parts[0]['context'] == 'interface_core'
        assert parts[0]['clinical'] == 'pathogenic'

    def test_parse_empty_string(self):
        """Empty string returns empty list."""
        assert _parse_variant_detail_parts('') == []

    def test_parse_skips_truncation(self):
        """Truncation indicators are skipped."""
        parts = _parse_variant_detail_parts(
            "K81P:interface_core:pathogenic|...(+5 more)"
        )
        assert len(parts) == 1

    def test_filter_skips_low_quality(self):
        """Low quality tier returns empty list."""
        result = _filter_variants_for_foldx(SAMPLE_DETAILS, 'Low', 80.0)
        assert result == []

    def test_filter_accepts_medium_quality(self):
        """Medium quality tier passes filter."""
        result = _filter_variants_for_foldx(SAMPLE_DETAILS, 'Medium', 80.0)
        assert len(result) > 0

    def test_filter_accepts_high_quality(self):
        """High quality tier passes filter."""
        result = _filter_variants_for_foldx(SAMPLE_DETAILS, 'High', 80.0)
        assert len(result) > 0

    def test_filter_filters_non_interface_context(self):
        """Non-interface variants are filtered out."""
        result = _filter_variants_for_foldx(SAMPLE_DETAILS, 'High', 80.0)
        contexts = {v['context'] for v in result}
        assert 'surface_non_interface' not in contexts
        assert 'buried_core' not in contexts

    def test_filter_accepts_interface_core(self):
        """interface_core context passes filter."""
        result = _filter_variants_for_foldx(
            "K81P:interface_core:pathogenic", 'High', 80.0
        )
        assert len(result) == 1
        assert result[0]['context'] == 'interface_core'

    def test_filter_accepts_interface_rim(self):
        """interface_rim context passes filter."""
        result = _filter_variants_for_foldx(
            "E82K:interface_rim:VUS", 'High', 80.0
        )
        assert len(result) == 1

    def test_filter_filters_benign_clinical(self):
        """Benign variants are filtered out."""
        result = _filter_variants_for_foldx(
            "R50A:interface_core:benign", 'High', 80.0
        )
        assert len(result) == 0

    def test_filter_accepts_pathogenic(self):
        """Pathogenic variants pass filter."""
        result = _filter_variants_for_foldx(
            "K81P:interface_core:pathogenic", 'High', 80.0
        )
        assert len(result) == 1

    def test_filter_accepts_vus(self):
        """VUS variants pass filter."""
        result = _filter_variants_for_foldx(
            "E82K:interface_rim:VUS", 'High', 80.0
        )
        assert len(result) == 1

    def test_filter_accepts_unknown_dash(self):
        """Unknown clinical significance ('-') passes filter."""
        result = _filter_variants_for_foldx(
            "K81P:interface_core:-", 'High', 80.0
        )
        assert len(result) == 1

    def test_filter_respects_plddt_threshold(self):
        """Low pLDDT returns empty list."""
        result = _filter_variants_for_foldx(SAMPLE_DETAILS, 'High', 50.0)
        assert result == []

    def test_filter_excludes_stop_codons(self):
        """Stop codon variants (alt_aa='*') are excluded."""
        result = _filter_variants_for_foldx(
            "K81*:interface_core:pathogenic", 'High', 80.0
        )
        assert len(result) == 0

    def test_filter_empty_details(self):
        """Empty variant details returns empty list regardless of tier."""
        result = _filter_variants_for_foldx('', 'High', 80.0)
        assert result == []

    def test_filter_combined(self):
        """Full filtering: only eligible variants from sample string."""
        result = _filter_variants_for_foldx(SAMPLE_DETAILS, 'High', 80.0)
        # K81P (interface_core, pathogenic) and E82K (interface_rim, VUS) pass
        # R50A (surface_non_interface) and G10S (buried_core) are filtered
        assert len(result) == 2
        positions = {v['position'] for v in result}
        assert positions == {81, 82}


@pytest.mark.foldx
class TestMutationFormatting:
    """Tests for _format_foldx_mutation()."""

    def test_standard_mutation(self):
        """Standard mutation formats correctly."""
        assert _format_foldx_mutation('K', 'A', 81, 'P') == 'KA81P'

    def test_chain_b(self):
        """Chain B mutation formats correctly."""
        assert _format_foldx_mutation('E', 'B', 82, 'K') == 'EB82K'

    def test_single_digit_position(self):
        """Single digit position formats correctly."""
        assert _format_foldx_mutation('R', 'A', 4, 'A') == 'RA4A'


@pytest.mark.foldx
class TestPDBPreparation:
    """Tests for _strip_remarks() and RepairPDB caching."""

    def test_strip_remarks(self, tmp_path):
        """REMARK lines are stripped from PDB file."""
        input_pdb = tmp_path / "input.pdb"
        output_pdb = tmp_path / "output.pdb"
        input_pdb.write_text(
            "REMARK 1 This is a remark\n"
            "REMARK 2 Another remark\n"
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n"
            "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00\n"
            "END\n"
        )
        _strip_remarks(input_pdb, output_pdb)
        content = output_pdb.read_text()
        assert 'REMARK' not in content
        assert 'ATOM' in content
        assert 'END' in content

    def test_strip_remarks_no_remarks(self, tmp_path):
        """PDB without REMARK lines is unchanged."""
        input_pdb = tmp_path / "input.pdb"
        output_pdb = tmp_path / "output.pdb"
        original = "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\nEND\n"
        input_pdb.write_text(original)
        _strip_remarks(input_pdb, output_pdb)
        assert output_pdb.read_text() == original

    def test_repair_pdb_cached(self, tmp_path):
        """RepairPDB returns cached result on second call."""
        pdb_file = tmp_path / "test.pdb"
        pdb_file.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\nEND\n")
        cache_dir = tmp_path / "cache"
        repair_dir = cache_dir / "repaired"
        repair_dir.mkdir(parents=True)

        # Pre-populate the cache
        import hashlib
        pdb_identity = f"{pdb_file.name}:{pdb_file.stat().st_size}"
        repair_key = hashlib.sha256(pdb_identity.encode()).hexdigest()[:16]
        cached_file = repair_dir / f"{repair_key}_Repair.pdb"
        cached_file.write_text("REPAIRED PDB CONTENT")

        # Should return cached file without running FoldX
        result = _run_repair_pdb(pdb_file, DEFAULT_FOLDX_BINARY, DEFAULT_ROTABASE, cache_dir)
        assert result == cached_file
        assert result.read_text() == "REPAIRED PDB CONTENT"


@pytest.mark.foldx
class TestIndividualList:
    """Tests for _write_individual_list()."""

    def test_single_mutation(self, tmp_path):
        """Single mutation written correctly."""
        output = tmp_path / "individual_list.txt"
        _write_individual_list(["KA81P"], output)
        content = output.read_text()
        assert content == "KA81P;\n"

    def test_multiple_mutations(self, tmp_path):
        """Multiple mutations written on separate lines."""
        output = tmp_path / "individual_list.txt"
        _write_individual_list(["KA81P", "EB82K"], output)
        content = output.read_text()
        assert content == "KA81P;\nEB82K;\n"


@pytest.mark.foldx
class TestOutputParsing:
    """Tests for _parse_buildmodel_output() and _parse_analysecomplex_output()."""

    def test_parse_buildmodel_3_runs(self, test_foldx_outputs_dir):
        """Parse BuildModel output with 3 runs."""
        fxout = test_foldx_outputs_dir / "Dif_test_Repair.fxout"
        ddg_values = _parse_buildmodel_output(fxout)
        assert len(ddg_values) == 3
        assert ddg_values[0] == pytest.approx(TEST_DDG_RUN_1)
        assert ddg_values[1] == pytest.approx(TEST_DDG_RUN_2)
        assert ddg_values[2] == pytest.approx(TEST_DDG_RUN_3)

    def test_parse_buildmodel_mean(self, test_foldx_outputs_dir):
        """Mean DDG across 3 runs matches expected value."""
        fxout = test_foldx_outputs_dir / "Dif_test_Repair.fxout"
        ddg_values = _parse_buildmodel_output(fxout)
        from statistics import mean
        assert mean(ddg_values) == pytest.approx(TEST_DDG_MEAN, abs=0.001)

    def test_parse_buildmodel_missing_file(self, tmp_path):
        """Missing .fxout file raises FoldXError."""
        with pytest.raises(FoldXError, match="not found"):
            _parse_buildmodel_output(tmp_path / "nonexistent.fxout")

    def test_parse_buildmodel_empty_file(self, tmp_path):
        """Empty .fxout file returns empty list."""
        empty = tmp_path / "empty.fxout"
        empty.write_text("")
        ddg_values = _parse_buildmodel_output(empty)
        assert ddg_values == []

    def test_parse_analysecomplex(self, test_foldx_outputs_dir):
        """Parse AnalyseComplex output for interaction energy."""
        fxout = test_foldx_outputs_dir / "Interaction_test_Repair_AC.fxout"
        result = _parse_analysecomplex_output(fxout)
        assert 'interaction_energy' in result
        assert result['interaction_energy'] == pytest.approx(TEST_INTERACTION_ENERGY)

    def test_parse_analysecomplex_missing_file(self, tmp_path):
        """Missing AnalyseComplex file returns empty dict."""
        result = _parse_analysecomplex_output(tmp_path / "nonexistent.fxout")
        assert result == {}


@pytest.mark.foldx
class TestFoldXExecution:
    """Tests for subprocess execution — all mocked, no real binary."""

    @patch('subprocess.run')
    def test_run_buildmodel_success(self, mock_run, tmp_path):
        """BuildModel subprocess runs successfully."""
        # Set up mock
        mock_run.return_value = MagicMock(returncode=0, stdout='', stderr='')

        # Create required files
        pdb = tmp_path / "test.pdb"
        pdb.write_text("ATOM test\n")
        rotabase = tmp_path / "rotabase.txt"
        rotabase.write_text("rotamer data")
        work_dir = tmp_path / "work"
        work_dir.mkdir()

        # Create expected output file (FoldX would create this)
        dif_file = work_dir / "Dif_test.fxout"
        dif_file.write_text("test.pdb\t0.1\t2.34\t-0.1\n")

        result = _run_buildmodel(
            pdb, "KA81P", tmp_path / "foldx.exe", rotabase, work_dir,
        )
        assert result.exists()
        mock_run.assert_called_once()

    @patch('subprocess.run')
    def test_run_buildmodel_timeout(self, mock_run, tmp_path):
        """BuildModel timeout raises FoldXError."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd='foldx', timeout=120)

        pdb = tmp_path / "test.pdb"
        pdb.write_text("ATOM test\n")
        rotabase = tmp_path / "rotabase.txt"
        rotabase.write_text("rotamer data")
        work_dir = tmp_path / "work"
        work_dir.mkdir()

        with pytest.raises(FoldXError, match="timed out"):
            _run_buildmodel(
                pdb, "KA81P", tmp_path / "foldx.exe", rotabase, work_dir,
            )

    @patch('subprocess.run')
    def test_run_buildmodel_nonzero_exit(self, mock_run, tmp_path):
        """BuildModel non-zero exit code raises FoldXError."""
        mock_run.return_value = MagicMock(returncode=1, stdout='', stderr='Error message')

        pdb = tmp_path / "test.pdb"
        pdb.write_text("ATOM test\n")
        rotabase = tmp_path / "rotabase.txt"
        rotabase.write_text("rotamer data")
        work_dir = tmp_path / "work"
        work_dir.mkdir()

        with pytest.raises(FoldXError, match="failed"):
            _run_buildmodel(
                pdb, "KA81P", tmp_path / "foldx.exe", rotabase, work_dir,
            )


@pytest.mark.foldx
class TestDDGComputation:
    """Tests for compute_ddg_for_variant() and compute_ddg_batch()."""

    def test_compute_ddg_uses_cache(self, tmp_path):
        """compute_ddg_for_variant returns cached value without running FoldX."""
        pdb = tmp_path / "test.pdb"
        pdb.write_text("ATOM test\n")

        # Pre-populate cache
        key = _foldx_cache_key(pdb, 'A', 81, 'K', 'P')
        _write_foldx_cache(tmp_path / "cache", key, 2.34, "KA81P")

        result = compute_ddg_for_variant(
            pdb, 'A', 'K', 81, 'P',
            cache_dir=tmp_path / "cache",
        )
        assert result == pytest.approx(2.34)

    @patch('foldx_scorer._run_buildmodel')
    def test_compute_ddg_returns_mean(self, mock_build, tmp_path):
        """compute_ddg_for_variant returns mean DDG across runs."""
        pdb = tmp_path / "test.pdb"
        pdb.write_text("ATOM test\n")

        # Create a fake .fxout file
        fxout = tmp_path / "Dif_test.fxout"
        fxout.write_text(
            "test_1_0.pdb\t0.1\t2.34\t-0.1\n"
            "test_1_1.pdb\t0.1\t2.51\t-0.1\n"
            "test_1_2.pdb\t0.1\t2.18\t-0.1\n"
        )
        mock_build.return_value = fxout

        result = compute_ddg_for_variant(
            pdb, 'A', 'K', 81, 'P',
            cache_dir=tmp_path / "cache",
        )
        assert result == pytest.approx(TEST_DDG_MEAN, abs=0.001)

    @patch('foldx_scorer._run_buildmodel')
    def test_compute_ddg_returns_none_on_failure(self, mock_build, tmp_path):
        """compute_ddg_for_variant returns None when FoldX fails."""
        pdb = tmp_path / "test.pdb"
        pdb.write_text("ATOM test\n")

        mock_build.side_effect = FoldXError("test error")

        result = compute_ddg_for_variant(
            pdb, 'A', 'K', 81, 'P',
            cache_dir=tmp_path / "cache",
        )
        assert result is None

    def test_batch_skips_cached(self, tmp_path):
        """compute_ddg_batch skips variants that are already cached."""
        pdb = tmp_path / "test.pdb"
        pdb.write_text("ATOM test\n")
        cache = tmp_path / "cache"

        # Pre-populate cache for variant 1
        key = _foldx_cache_key(pdb, 'A', 81, 'K', 'P')
        _write_foldx_cache(cache, key, 2.34, "KA81P")

        variants = [
            {'ref_aa': 'K', 'position': 81, 'alt_aa': 'P'},
        ]

        # This should return the cached value without running FoldX
        result = compute_ddg_batch(
            pdb, 'A', variants, cache_dir=cache,
        )
        assert ('K', 81, 'P') in result
        assert result[('K', 81, 'P')] == pytest.approx(2.34)


@pytest.mark.foldx
class TestDetailFormatting:
    """Tests for format_foldx_details()."""

    def test_empty_list(self):
        """Empty list returns empty string."""
        assert format_foldx_details([]) == ''

    def test_single_destabilising(self):
        """Single destabilising variant formatted correctly."""
        result = format_foldx_details([
            {'ref_aa': 'K', 'position': 81, 'alt_aa': 'P', 'ddg': 2.34}
        ])
        assert result == 'K81P:ddg=2.34:destabilising'

    def test_single_stable(self):
        """Single stable variant formatted correctly."""
        result = format_foldx_details([
            {'ref_aa': 'E', 'position': 82, 'alt_aa': 'K', 'ddg': 0.45}
        ])
        assert result == 'E82K:ddg=0.45:stable'

    def test_none_ddg(self):
        """Variant with None DDG shows no_data."""
        result = format_foldx_details([
            {'ref_aa': 'R', 'position': 4, 'alt_aa': 'A', 'ddg': None}
        ])
        assert result == 'R4A:ddg=-:no_data'

    def test_multiple_variants(self):
        """Multiple variants are pipe-separated."""
        result = format_foldx_details([
            {'ref_aa': 'K', 'position': 81, 'alt_aa': 'P', 'ddg': 2.34},
            {'ref_aa': 'E', 'position': 82, 'alt_aa': 'K', 'ddg': 0.45},
        ])
        assert '|' in result
        parts = result.split('|')
        assert len(parts) == 2

    def test_truncation(self):
        """Details beyond limit show truncation indicator."""
        variants = [
            {'ref_aa': 'K', 'position': i, 'alt_aa': 'P', 'ddg': 1.0}
            for i in range(25)
        ]
        result = format_foldx_details(variants, limit=20)
        assert '...(+5 more)' in result

    def test_exact_threshold(self):
        """DDG exactly at threshold is NOT destabilising (strictly >)."""
        result = format_foldx_details([
            {'ref_aa': 'K', 'position': 81, 'alt_aa': 'P', 'ddg': 1.6}
        ])
        assert 'stable' in result


@pytest.mark.foldx
class TestAnnotation:
    """Tests for _score_chain_variants_foldx() and annotate_results_with_foldx()."""

    def test_skips_low_tier_complexes(self, tmp_path):
        """Low tier complexes are skipped entirely."""
        pdb_dir = tmp_path / "pdbs"
        pdb_dir.mkdir()

        results = [{
            'complex_name': 'test_complex',
            'quality_tier_v2': 'Low',
            'variant_details_a': 'K81P:interface_core:pathogenic',
            'variant_details_b': '',
            'best_chain_pair': 'A_B',
            'interface_plddt_mean': 80.0,
            'protein_a': 'P12345',
            'protein_b': 'Q67890',
        }]

        annotate_results_with_foldx(results, pdb_dir, cache_dir=tmp_path / "cache")

        # All columns should have default values
        assert results[0]['foldx_ddg_mean_a'] == ''
        assert results[0]['foldx_n_destabilising_a'] == 0
        assert results[0]['foldx_details_a'] == ''

    def test_sets_all_8_columns(self, tmp_path):
        """All 8 FoldX columns are set even when skipped."""
        pdb_dir = tmp_path / "pdbs"
        pdb_dir.mkdir()

        results = [{
            'complex_name': 'test_complex',
            'quality_tier_v2': 'Low',
            'variant_details_a': '',
            'variant_details_b': '',
            'best_chain_pair': 'A_B',
            'interface_plddt_mean': 80.0,
        }]

        annotate_results_with_foldx(results, pdb_dir, cache_dir=tmp_path / "cache")

        for col in CSV_FIELDNAMES_FOLDX:
            assert col in results[0], f"Missing column: {col}"

    def test_missing_pdb_graceful(self, tmp_path):
        """Missing PDB file doesn't crash — complex is skipped."""
        pdb_dir = tmp_path / "empty_pdbs"
        pdb_dir.mkdir()

        results = [{
            'complex_name': 'nonexistent_complex',
            'quality_tier_v2': 'High',
            'variant_details_a': 'K81P:interface_core:pathogenic',
            'variant_details_b': '',
            'best_chain_pair': 'A_B',
            'interface_plddt_mean': 80.0,
        }]

        # Should not raise
        annotate_results_with_foldx(
            results, pdb_dir, cache_dir=tmp_path / "cache", verbose=True,
        )
        assert results[0]['foldx_ddg_mean_a'] == ''

    def test_empty_variant_details_no_error(self, tmp_path):
        """Complex with empty variant details doesn't crash."""
        pdb_dir = tmp_path / "pdbs"
        pdb_dir.mkdir()

        results = [{
            'complex_name': 'test_complex',
            'quality_tier_v2': 'High',
            'variant_details_a': '',
            'variant_details_b': '',
            'best_chain_pair': 'A_B',
            'interface_plddt_mean': 80.0,
        }]

        annotate_results_with_foldx(results, pdb_dir, cache_dir=tmp_path / "cache")
        assert results[0]['foldx_details_a'] == ''


@pytest.mark.foldx
class TestScoreChainVariants:
    """Tests for _score_chain_variants_foldx() with mocked FoldX."""

    @patch('foldx_scorer.compute_ddg_batch')
    def test_returns_dict_with_expected_keys(self, mock_batch, tmp_path):
        """Return dict has expected keys."""
        pdb = tmp_path / "test.pdb"
        pdb.write_text("ATOM test\n")
        mock_batch.return_value = {('K', 81, 'P'): 2.34}

        result = _score_chain_variants_foldx(
            "K81P:interface_core:pathogenic",
            'High', 80.0,
            pdb, 'A',
            DEFAULT_FOLDX_BINARY, DEFAULT_ROTABASE,
            tmp_path / "cache",
        )
        assert 'ddg_mean' in result
        assert 'n_destabilising' in result
        assert 'coverage' in result
        assert 'details' in result

    @patch('foldx_scorer.compute_ddg_batch')
    def test_destabilising_count(self, mock_batch, tmp_path):
        """Count of destabilising variants is correct."""
        pdb = tmp_path / "test.pdb"
        pdb.write_text("ATOM test\n")
        mock_batch.return_value = {
            ('K', 81, 'P'): 2.34,  # destabilising (>1.6)
            ('E', 82, 'K'): 0.45,  # stable
        }

        result = _score_chain_variants_foldx(
            "K81P:interface_core:pathogenic|E82K:interface_rim:VUS",
            'High', 80.0,
            pdb, 'A',
            DEFAULT_FOLDX_BINARY, DEFAULT_ROTABASE,
            tmp_path / "cache",
        )
        assert result['n_destabilising'] == 1

    def test_low_tier_returns_empty(self, tmp_path):
        """Low quality tier returns empty results."""
        pdb = tmp_path / "test.pdb"
        pdb.write_text("ATOM test\n")

        result = _score_chain_variants_foldx(
            "K81P:interface_core:pathogenic",
            'Low', 80.0,
            pdb, 'A',
            DEFAULT_FOLDX_BINARY, DEFAULT_ROTABASE,
            tmp_path / "cache",
        )
        assert result['ddg_mean'] == ''
        assert result['n_destabilising'] == 0
        assert result['n_eligible'] == 0


@pytest.mark.foldx
class TestCSVFieldnames:
    """Tests for CSV_FIELDNAMES_FOLDX integration."""

    def test_fieldnames_length(self):
        """Exactly 8 FoldX CSV columns."""
        assert len(CSV_FIELDNAMES_FOLDX) == 8

    def test_fieldnames_in_toolkit(self):
        """FoldX fieldnames are included when include_foldx=True."""
        from toolkit import get_csv_fieldnames
        with_foldx = get_csv_fieldnames(include_foldx=True)
        without_foldx = get_csv_fieldnames(include_foldx=False)
        assert len(with_foldx) == len(without_foldx) + 8
        for col in CSV_FIELDNAMES_FOLDX:
            assert col in with_foldx

    def test_fieldnames_not_in_base(self):
        """FoldX fieldnames are NOT in base CSV output."""
        from toolkit import get_csv_fieldnames
        base = get_csv_fieldnames()
        for col in CSV_FIELDNAMES_FOLDX:
            assert col not in base


@pytest.mark.foldx
@pytest.mark.cli
class TestCLI:
    """Tests for standalone CLI."""

    def test_parser_has_subcommands(self):
        """Parser has summary and lookup subcommands."""
        parser = build_argument_parser()
        # Parse summary
        args = parser.parse_args(['summary'])
        assert args.command == 'summary'

    def test_parser_lookup(self):
        """Parser parses lookup args correctly."""
        parser = build_argument_parser()
        args = parser.parse_args([
            'lookup',
            '--pdb', 'test.pdb',
            '--chain', 'A',
            '--position', '81',
            '--wildtype', 'K',
            '--mutant', 'P',
        ])
        assert args.command == 'lookup'
        assert args.pdb == 'test.pdb'
        assert args.chain == 'A'
        assert args.position == 81

    def test_no_command_exits(self):
        """No subcommand prints help (exit code 0)."""
        parser = build_argument_parser()
        args = parser.parse_args([])
        assert args.command is None

    def test_summary_nonexistent_cache(self, tmp_path, capsys):
        """Summary on nonexistent cache prints informative message."""
        _cli_summary(tmp_path / "nonexistent")
        captured = capsys.readouterr()
        assert "does not exist" in captured.out

    def test_summary_empty_cache(self, tmp_path, capsys):
        """Summary on empty cache shows zero results."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        _cli_summary(cache_dir)
        captured = capsys.readouterr()
        assert "0" in captured.out


@pytest.mark.foldx
@pytest.mark.regression
class TestRegression:
    """Regression tests with known test data."""

    def test_destabilising_threshold_applied_correctly(self):
        """DDG values above/below threshold are classified correctly."""
        # Above threshold
        result = format_foldx_details([
            {'ref_aa': 'K', 'position': 81, 'alt_aa': 'P', 'ddg': 2.0}
        ])
        assert 'destabilising' in result

        # Below threshold
        result = format_foldx_details([
            {'ref_aa': 'E', 'position': 82, 'alt_aa': 'K', 'ddg': 1.0}
        ])
        assert 'stable' in result

    def test_buildmodel_output_parsing_values(self, test_foldx_outputs_dir):
        """Known DDG values from test .fxout file match expected."""
        fxout = test_foldx_outputs_dir / "Dif_test_Repair.fxout"
        ddg_values = _parse_buildmodel_output(fxout)
        assert len(ddg_values) == 3
        # Known values from test data
        assert ddg_values[0] == pytest.approx(2.34, abs=0.01)
        assert ddg_values[1] == pytest.approx(2.51, abs=0.01)
        assert ddg_values[2] == pytest.approx(2.18, abs=0.01)

    def test_filter_sample_details_known_output(self):
        """Known filtering of sample variant details string."""
        result = _filter_variants_for_foldx(SAMPLE_DETAILS, 'High', 80.0)
        # K81P: interface_core + pathogenic -> eligible
        # E82K: interface_rim + VUS -> eligible
        # R50A: surface_non_interface -> filtered
        # G10S: buried_core -> filtered (even though clinical is likely_pathogenic)
        assert len(result) == 2
        assert result[0]['position'] == 81
        assert result[1]['position'] == 82
