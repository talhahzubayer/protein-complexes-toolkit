"""
Tests for visualise_results.py - data loading, column detection, and figure generation.

Tests use a CSV generated from the pipeline (via the pipeline_csv fixture)
rather than the pre-existing results.csv. This ensures the visualisation
pipeline is tested end-to-end against real pipeline output.

All generated figures are saved to tests/test_output/figures/.
"""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import visualise_results
from visualise_results import (
    load_data,
    detect_columns,
    plot_fig1_quality_scatter,
    plot_fig2_pae_health_check,
    plot_fig3_interface_pae_by_tier,
    plot_fig4_composite_validation,
    plot_fig5_interface_vs_bulk,
    plot_fig6_paradox_spotlight,
    plot_fig7_homo_vs_hetero,
    plot_fig8_metric_disagreement,
    plot_fig9_correlation_flags,
    plot_fig10_chain_count_profile,
    _get_paradox_mask,
    TIER_COLORS,
    TIER_ORDER,
)

PROJECT_ROOT = Path(r"C:\Users\Talhah Zubayer\Documents\protein-complexes-toolkit")


# ── Module Import Test ────────────────────────────────────────────

class TestModuleImport:
    """Verify the module can be imported without errors."""

    def test_imports_successfully(self):
        import visualise_results  # noqa: F401

    def test_tier_colors_has_all_tiers(self):
        assert set(TIER_COLORS.keys()) == {'High', 'Medium', 'Low'}

    def test_tier_order(self):
        assert TIER_ORDER == ['High', 'Medium', 'Low']


# ── Data Loading Tests (using pipeline-generated CSV) ─────────────

@pytest.mark.slow
class TestLoadData:
    """Tests for load_data using the pipeline-generated CSV."""

    def test_loads_non_empty(self, pipeline_csv):
        df = load_data(str(pipeline_csv))
        assert len(df) > 0

    def test_has_base_columns(self, pipeline_csv):
        df = load_data(str(pipeline_csv))
        for col in ['complex_name', 'iptm', 'pdockq', 'quality_tier']:
            assert col in df.columns, f"Missing column: {col}"

    def test_has_interface_columns(self, pipeline_csv):
        df = load_data(str(pipeline_csv))
        for col in ['n_interface_contacts', 'interface_plddt_combined',
                     'interface_pae_mean', 'quality_tier_v2']:
            assert col in df.columns, f"Missing interface column: {col}"

    def test_iptm_positive(self, pipeline_csv):
        df = load_data(str(pipeline_csv))
        valid = df[df['iptm'].notna()]
        assert (valid['iptm'] > 0).all(), "Found non-positive ipTM values"

    def test_pdockq_no_nan(self, pipeline_csv):
        """pDockQ NaN should be filled with 0 by load_data."""
        df = load_data(str(pipeline_csv))
        assert df['pdockq'].isna().sum() == 0

    def test_complex_type_lowercase(self, pipeline_csv):
        df = load_data(str(pipeline_csv))
        if 'complex_type' in df.columns:
            valid = df[df['complex_type'].notna()]
            assert valid['complex_type'].isin(['homodimer', 'heterodimer']).all()


# ── Column Detection Tests ────────────────────────────────────────

@pytest.mark.slow
class TestDetectColumns:
    """Tests for detect_columns on pipeline-generated CSV."""

    def test_full_csv_flags(self, pipeline_csv):
        """Pipeline CSV with --interface --pae should have all column groups."""
        df = load_data(str(pipeline_csv))
        flags = detect_columns(df)
        assert flags['has_v2_data'], "Missing quality_tier_v2"
        assert flags['has_interface_data'], "Missing interface data"
        assert flags['has_pae_interface'], "Missing interface PAE"
        assert flags['has_composite'], "Missing composite score"
        assert flags['has_chain_info'], "Missing chain info"

    def test_base_only_csv(self):
        """Synthetic base-only DataFrame should have no interface flags."""
        df = pd.DataFrame({
            'complex_name': ['X_Y'],
            'iptm': [0.5],
            'pdockq': [0.3],
        })
        flags = detect_columns(df)
        assert not flags['has_v2_data']
        assert not flags['has_interface_data']
        assert not flags['has_pae_interface']


# ── Figure Generation Tests ───────────────────────────────────────

@pytest.mark.slow
class TestFigureGeneration:
    """Test that figure-generating functions run without error.

    Each test verifies:
    1. The function doesn't raise an exception
    2. The expected output file is created in test_output/figures/

    Uses the pipeline-generated CSV (not the pre-existing results.csv)
    to ensure figures work with real pipeline output.
    """

    @pytest.fixture(scope="class")
    def figures_dir(self, test_output_dir):
        """Create and set the figures output directory."""
        fig_dir = test_output_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        # Point the module's OUTPUT_DIR to our test directory
        visualise_results.OUTPUT_DIR = str(fig_dir)
        return fig_dir

    @pytest.fixture(scope="class")
    def loaded_df(self, pipeline_csv):
        """Load the pipeline-generated CSV once for all figure tests."""
        return load_data(str(pipeline_csv))

    @pytest.fixture(scope="class")
    def col_flags(self, loaded_df):
        return detect_columns(loaded_df)

    def test_fig1_quality_scatter(self, loaded_df, col_flags, figures_dir):
        plot_fig1_quality_scatter(loaded_df, col_flags, density_mode=False)
        assert any(f.startswith("1_") for f in os.listdir(figures_dir)), \
            "Fig 1 output file not found"

    def test_fig2_pae_health_check(self, loaded_df, figures_dir):
        plot_fig2_pae_health_check(loaded_df)
        assert any(f.startswith("2_") for f in os.listdir(figures_dir)), \
            "Fig 2 output file not found"

    def test_fig3_interface_pae_by_tier(self, loaded_df, col_flags, figures_dir):
        if not (col_flags['has_v2_data'] and col_flags['has_interface_data']):
            pytest.skip("Requires V2 + interface data")
        plot_fig3_interface_pae_by_tier(loaded_df)
        assert any(f.startswith("3_") for f in os.listdir(figures_dir)), \
            "Fig 3 output file not found"

    def test_fig4_composite_validation(self, loaded_df, col_flags, figures_dir):
        if not (col_flags['has_v2_data'] and col_flags['has_interface_data']):
            pytest.skip("Requires V2 + interface data")
        plot_fig4_composite_validation(loaded_df, density_mode=False)
        assert any(f.startswith("4_") for f in os.listdir(figures_dir)), \
            "Fig 4 output file not found"

    def test_fig5_interface_vs_bulk(self, loaded_df, col_flags, figures_dir):
        if not (col_flags['has_v2_data'] and col_flags['has_interface_data']):
            pytest.skip("Requires V2 + interface data")
        plot_fig5_interface_vs_bulk(loaded_df, density_mode=False)
        assert any(f.startswith("5_") for f in os.listdir(figures_dir)), \
            "Fig 5 output file not found"

    def test_fig6_paradox_spotlight(self, loaded_df, col_flags, figures_dir):
        if not (col_flags['has_v2_data'] and col_flags['has_interface_data']):
            pytest.skip("Requires V2 + interface data")
        if _get_paradox_mask(loaded_df).sum() == 0:
            pytest.skip("No paradox complexes in test data")
        plot_fig6_paradox_spotlight(loaded_df)
        assert any(f.startswith("6_") for f in os.listdir(figures_dir)), \
            "Fig 6 output file not found"

    def test_fig7_homo_vs_hetero(self, loaded_df, col_flags, figures_dir):
        if not (col_flags['has_v2_data'] and col_flags['has_interface_data']):
            pytest.skip("Requires V2 + interface data")
        plot_fig7_homo_vs_hetero(loaded_df)
        assert any(f.startswith("7_") for f in os.listdir(figures_dir)), \
            "Fig 7 output file not found"

    def test_fig8_metric_disagreement(self, loaded_df, col_flags, figures_dir):
        if not (col_flags['has_v2_data'] and col_flags['has_interface_data']):
            pytest.skip("Requires V2 + interface data")
        plot_fig8_metric_disagreement(loaded_df, density_mode=False)
        assert any(f.startswith("8_") for f in os.listdir(figures_dir)), \
            "Fig 8 output file not found"

    def test_fig9_correlation_flags(self, loaded_df, col_flags, figures_dir):
        if not (col_flags['has_v2_data'] and col_flags['has_interface_data']):
            pytest.skip("Requires V2 + interface data")
        plot_fig9_correlation_flags(loaded_df)
        assert any(f.startswith("9_") for f in os.listdir(figures_dir)), \
            "Fig 9 output file not found"

    def test_fig10_chain_count_profile(self, loaded_df, col_flags, figures_dir):
        if not col_flags['has_chain_info']:
            pytest.skip("Requires n_chains column")
        n_groups = loaded_df['n_chains'].nunique()
        if n_groups < 2:
            pytest.skip("Test data has only 1 chain-count group (all dimers)")
        plot_fig10_chain_count_profile(loaded_df, density_mode=False)
        assert any(f.startswith("10_") for f in os.listdir(figures_dir)), \
            "Fig 10 output file not found"


# ── CLI Tests ─────────────────────────────────────────────────────

@pytest.mark.cli
class TestCLI:
    """Tests for visualise_results.py CLI."""

    def test_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "visualise_results.py"), "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0

    @pytest.mark.slow
    def test_cli_generates_figures(self, pipeline_csv, test_output_dir):
        """Run the CLI with the pipeline-generated CSV and verify it succeeds."""
        cli_fig_dir = test_output_dir / "cli_figures"
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "visualise_results.py"),
                str(pipeline_csv),
                "--output-dir", str(cli_fig_dir),
            ],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, f"CLI failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        assert cli_fig_dir.exists(), "Output directory not created"
        png_files = list(cli_fig_dir.glob("*.png"))
        assert len(png_files) >= 2, f"Expected at least 2 figures, got {len(png_files)}"
