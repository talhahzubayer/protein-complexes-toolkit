"""
Tests for read_af2_nojax.py — JAX-free PKL extraction.

Regression values captured on 2026-03-04 from reference complex 1
(A0A0B4J2C3_P24534) on Windows 11.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from read_af2_nojax import (
    load_pkl_without_jax,
    extract_scalar,
    extract_metrics,
    list_keys,
    MAX_RECURSION_DEPTH,
    PLDDT_VERY_HIGH_THRESHOLD,
    PLDDT_HIGH_THRESHOLD,
    PLDDT_LOW_THRESHOLD,
)

PROJECT_ROOT = Path(r"C:\Users\Talhah Zubayer\Documents\protein-complexes-toolkit")

# ── Regression expected values (complex 1: A0A0B4J2C3_P24534) ────
EXPECTED_IPTM_1 = 0.611156165599823
EXPECTED_PTM_1 = 0.5183634993925526
EXPECTED_PLDDT_MEAN_1 = 77.7675944316459
EXPECTED_PLDDT_MEDIAN_1 = 86.37260904287213
EXPECTED_PAE_MEAN_1 = 19.177913665771484
EXPECTED_NUM_RESIDUES_1 = 422
EXPECTED_PLDDT_VERY_HIGH_1 = 160
EXPECTED_PLDDT_HIGH_1 = 165
EXPECTED_PLDDT_LOW_1 = 30
EXPECTED_PLDDT_VERY_LOW_1 = 67


# ── Pure Unit Tests (no file I/O) ────────────────────────────────

class TestExtractScalar:
    """Tests for the extract_scalar utility function."""

    def test_python_int(self):
        assert extract_scalar(42) == 42.0
        assert isinstance(extract_scalar(42), float)

    def test_python_float(self):
        assert extract_scalar(3.14) == 3.14

    def test_numpy_scalar(self):
        result = extract_scalar(np.float32(0.75))
        assert result == pytest.approx(0.75, abs=1e-6)
        assert isinstance(result, float)

    def test_numpy_0d_array(self):
        result = extract_scalar(np.array(0.5))
        assert result == pytest.approx(0.5)

    def test_numpy_1d_size_1(self):
        result = extract_scalar(np.array([0.9]))
        assert result == pytest.approx(0.9)

    def test_numpy_multi_element_returns_none(self):
        result = extract_scalar(np.array([1, 2, 3]))
        assert result is None

    def test_none_returns_none(self):
        assert extract_scalar(None) is None

    def test_string_returns_none(self):
        assert extract_scalar("hello") is None


class TestConstants:
    """Verify module-level constants."""

    def test_max_recursion_depth(self):
        assert MAX_RECURSION_DEPTH == 100

    def test_plddt_thresholds(self):
        assert PLDDT_VERY_HIGH_THRESHOLD == 90
        assert PLDDT_HIGH_THRESHOLD == 70
        assert PLDDT_LOW_THRESHOLD == 50

    def test_threshold_ordering(self):
        assert PLDDT_LOW_THRESHOLD < PLDDT_HIGH_THRESHOLD < PLDDT_VERY_HIGH_THRESHOLD


class TestLoadPklErrors:
    """Edge case tests for load_pkl_without_jax."""

    def test_nonexistent_file_raises(self):
        with pytest.raises((FileNotFoundError, SystemExit)):
            load_pkl_without_jax("/nonexistent/path/fake.pkl")


class TestListKeys:
    """Tests for list_keys utility."""

    def test_returns_dict_with_string_values(self):
        fake = {'iptm': 0.5, 'plddt': np.array([1, 2, 3])}
        result = list_keys(fake)
        assert isinstance(result, dict)
        for v in result.values():
            assert isinstance(v, str)

    def test_numpy_array_describes_shape(self):
        fake = {'arr': np.zeros((10, 10))}
        result = list_keys(fake)
        assert 'ndarray' in result['arr'] or '(10, 10)' in result['arr']


# ── Real Data Tests ───────────────────────────────────────────────

@pytest.mark.slow
class TestLoadPklRealData:
    """Tests using real PKL files from Test_Data."""

    def test_returns_dict_with_expected_keys(self, loaded_pkl_1):
        assert isinstance(loaded_pkl_1, dict)
        assert len(loaded_pkl_1) > 0
        assert 'iptm' in loaded_pkl_1 or 'iptm' in str(loaded_pkl_1.keys())
        assert 'plddt' in loaded_pkl_1
        assert 'predicted_aligned_error' in loaded_pkl_1

    def test_arrays_are_numpy(self, loaded_pkl_1):
        plddt = loaded_pkl_1['plddt']
        pae = loaded_pkl_1['predicted_aligned_error']
        assert isinstance(plddt, np.ndarray), f"plddt is {type(plddt)}, not ndarray"
        assert isinstance(pae, np.ndarray), f"pae is {type(pae)}, not ndarray"

    def test_pae_is_square_matching_plddt(self, loaded_pkl_1):
        pae = np.asarray(loaded_pkl_1['predicted_aligned_error'])
        plddt = np.asarray(loaded_pkl_1['plddt'])
        assert pae.ndim == 2
        assert pae.shape[0] == pae.shape[1], "PAE matrix is not square"
        assert pae.shape[0] == len(plddt), "PAE dimension doesn't match pLDDT length"

    def test_new_naming_convention(self, ref_pkl_2):
        result = load_pkl_without_jax(ref_pkl_2)
        assert isinstance(result, dict)
        assert 'plddt' in result


@pytest.mark.slow
class TestExtractMetricsRealData:
    """Tests for extract_metrics using real data."""

    def test_returns_expected_keys(self, extracted_metrics_1):
        expected_keys = {
            'iptm', 'ptm', 'ranking_confidence',
            'plddt_mean', 'plddt_median', 'plddt_min', 'plddt_max', 'plddt_std',
            'num_residues', 'pae_mean',
            'plddt_very_high', 'plddt_high', 'plddt_low', 'plddt_very_low',
        }
        for key in expected_keys:
            assert key in extracted_metrics_1, f"Missing key: {key}"

    def test_values_are_python_types(self, extracted_metrics_1):
        """All values must be JSON-serialisable (python float/int, not numpy)."""
        import json
        # Should not raise
        json.dumps(extracted_metrics_1)

        for key, val in extracted_metrics_1.items():
            if isinstance(val, (float, int, str, list, type(None))):
                continue
            pytest.fail(f"Key '{key}' has non-JSON type: {type(val)}")

    def test_iptm_range(self, extracted_metrics_1):
        iptm = extracted_metrics_1['iptm']
        assert 0 <= iptm <= 1, f"ipTM {iptm} out of range [0, 1]"

    def test_ptm_range(self, extracted_metrics_1):
        ptm = extracted_metrics_1['ptm']
        assert 0 <= ptm <= 1, f"pTM {ptm} out of range [0, 1]"

    def test_plddt_range(self, extracted_metrics_1):
        assert 0 <= extracted_metrics_1['plddt_min'] <= extracted_metrics_1['plddt_max'] <= 100

    def test_plddt_bands_sum_to_num_residues(self, extracted_metrics_1):
        total = (
            extracted_metrics_1['plddt_very_high']
            + extracted_metrics_1['plddt_high']
            + extracted_metrics_1['plddt_low']
            + extracted_metrics_1['plddt_very_low']
        )
        assert total == extracted_metrics_1['num_residues'], \
            f"Band sum {total} != num_residues {extracted_metrics_1['num_residues']}"

    @pytest.mark.regression
    def test_regression_values(self, extracted_metrics_1):
        """Exact values for reference complex 1."""
        m = extracted_metrics_1
        assert m['iptm'] == pytest.approx(EXPECTED_IPTM_1, abs=1e-6)
        assert m['ptm'] == pytest.approx(EXPECTED_PTM_1, abs=1e-6)
        assert m['plddt_mean'] == pytest.approx(EXPECTED_PLDDT_MEAN_1, abs=0.01)
        assert m['plddt_median'] == pytest.approx(EXPECTED_PLDDT_MEDIAN_1, abs=0.01)
        assert m['pae_mean'] == pytest.approx(EXPECTED_PAE_MEAN_1, abs=0.01)
        assert m['num_residues'] == EXPECTED_NUM_RESIDUES_1
        assert m['plddt_very_high'] == EXPECTED_PLDDT_VERY_HIGH_1
        assert m['plddt_high'] == EXPECTED_PLDDT_HIGH_1
        assert m['plddt_low'] == EXPECTED_PLDDT_LOW_1
        assert m['plddt_very_low'] == EXPECTED_PLDDT_VERY_LOW_1


# ── CLI Tests ─────────────────────────────────────────────────────

@pytest.mark.cli
class TestCLI:
    """Tests for CLI entry point."""

    def test_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "read_af2_nojax.py"), "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
