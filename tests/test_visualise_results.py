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

import numpy as np

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
    plot_fig11_variant_consequence_flow,
    plot_fig12_variant_density_heatmap,
    plot_fig13_variant_burden,
    plot_fig14_pathway_coherence,
    plot_fig15_disease_enrichment,
    plot_fig16_pathway_network,
    plot_fig17_stability_crossvalidation,
    plot_fig18_clustering_validation,
    _get_paradox_mask,
    _parse_variant_details,
    _aggregate_all_variants,
    _normalise_significance,
    _parse_disease_name,
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


# ── Variant Detail Parsing Tests ─────────────────────────────────

class TestVariantDetailParsing:
    """Tests for _parse_variant_details, _aggregate_all_variants, _normalise_significance."""

    def test_parse_basic(self):
        result = _parse_variant_details('K81P:interface_core:Pathogenic|E82K:interface_rim:VUS')
        assert len(result) == 2
        assert result[0] == {'mutation': 'K81P', 'context': 'interface_core', 'significance': 'Pathogenic'}
        assert result[1] == {'mutation': 'E82K', 'context': 'interface_rim', 'significance': 'VUS'}

    def test_parse_empty(self):
        assert _parse_variant_details('') == []
        assert _parse_variant_details(None) == []
        assert _parse_variant_details(float('nan')) == []

    def test_parse_overflow_skipped(self):
        s = 'K81P:interface_core:Pathogenic|...(+5 more)'
        result = _parse_variant_details(s)
        assert len(result) == 1
        assert result[0]['mutation'] == 'K81P'

    def test_normalise_significance_pathogenic(self):
        assert _normalise_significance('Pathogenic') == 'Pathogenic'
        assert _normalise_significance('Pathogenic/Likely pathogenic') == 'Likely pathogenic'

    def test_normalise_significance_benign(self):
        assert _normalise_significance('Benign') == 'Benign'
        assert _normalise_significance('Benign/Likely benign') == 'Benign'
        assert _normalise_significance('Likely benign') == 'Benign'

    def test_normalise_significance_vus(self):
        assert _normalise_significance('Uncertain significance') == 'VUS'

    def test_normalise_significance_unknown(self):
        assert _normalise_significance('-') == 'Unknown'
        assert _normalise_significance('') == 'Unknown'

    def test_aggregate_all_variants(self):
        df = pd.DataFrame({
            'complex_name': ['X_Y'],
            'variant_details_a': ['K81P:interface_core:Pathogenic|E82K:interface_rim:VUS'],
            'variant_details_b': ['R45W:buried_core:-'],
        })
        result = _aggregate_all_variants(df)
        assert len(result) == 3
        assert set(result.columns) == {'complex_name', 'chain', 'mutation', 'context', 'significance'}
        assert (result[result['chain'] == 'a'].shape[0]) == 2
        assert (result[result['chain'] == 'b'].shape[0]) == 1


# ── Column Detection: Variant Flag Test ──────────────────────────

class TestDetectColumnsVariant:
    """Tests for the has_variant_data flag in detect_columns."""

    def test_variant_columns_detected(self):
        df = pd.DataFrame({
            'complex_name': ['X_Y'],
            'n_variants_a': [3],
            'variant_details_a': ['K81P:interface_core:Pathogenic'],
        })
        flags = detect_columns(df)
        assert flags['has_variant_data'] is True

    def test_no_variant_columns(self):
        df = pd.DataFrame({'complex_name': ['X_Y'], 'iptm': [0.5]})
        flags = detect_columns(df)
        assert flags['has_variant_data'] is False


# ── Variant Figure Generation Tests (Figs 11-13) ────────────────

@pytest.mark.variants
class TestVariantFigureGeneration:
    """Tests for Figs 11-13 using a synthetic variant DataFrame."""

    @pytest.fixture(scope="class")
    def variant_figures_dir(self, test_output_dir):
        fig_dir = test_output_dir / "variant_figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        visualise_results.OUTPUT_DIR = str(fig_dir)
        return fig_dir

    @pytest.fixture(scope="class")
    def variant_df(self):
        """30-row synthetic DataFrame with realistic variant columns."""
        import numpy as np
        rng = np.random.RandomState(42)
        contexts = ['interface_core', 'interface_rim', 'surface_non_interface', 'buried_core']
        significances = ['Pathogenic', 'Likely pathogenic', 'Uncertain significance', 'Benign', '-']
        rows = []
        for i in range(30):
            # Build variant_details strings
            n_a = (i % 5) + 1
            n_b = (i % 4)
            details_a_parts = []
            for _ in range(n_a):
                mut = f'{chr(65 + rng.randint(0, 26))}{rng.randint(1, 300)}{chr(65 + rng.randint(0, 26))}'
                ctx = contexts[rng.randint(0, 4)]
                sig = significances[rng.randint(0, 5)]
                details_a_parts.append(f'{mut}:{ctx}:{sig}')
            details_b_parts = []
            for _ in range(n_b):
                mut = f'{chr(65 + rng.randint(0, 26))}{rng.randint(1, 300)}{chr(65 + rng.randint(0, 26))}'
                ctx = contexts[rng.randint(0, 4)]
                sig = significances[rng.randint(0, 5)]
                details_b_parts.append(f'{mut}:{ctx}:{sig}')

            rows.append({
                'complex_name': f'PROT{i:02d}_PROT{i + 30:02d}',
                'iptm': 0.3 + 0.5 * (i / 30),
                'pdockq': 0.2 + 0.4 * (i / 30),
                'quality_tier': ['High', 'Medium', 'Low'][i % 3],
                'quality_tier_v2': ['High', 'Medium', 'Low'][i % 3],
                'interface_confidence_score': 0.3 + 0.5 * (i / 30),
                'n_interface_contacts': 20 + i * 3,
                'n_interface_residues_a': 10 + i,
                'n_interface_residues_b': 12 + i,
                'n_variants_a': n_a,
                'n_variants_b': n_b,
                'n_interface_variants_a': i % 3,
                'n_interface_variants_b': i % 2,
                'n_pathogenic_interface_variants': 1 if i % 7 == 0 else 0,
                'interface_variant_enrichment': 2.5 if i % 5 == 0 else (0.8 if i % 3 != 0 else 0),
                'variant_details_a': '|'.join(details_a_parts),
                'variant_details_b': '|'.join(details_b_parts) if details_b_parts else '',
                'gene_constraint_pli_a': 0.95,
                'gene_constraint_pli_b': 0.1,
                'gene_constraint_mis_z_a': 3.2,
                'gene_constraint_mis_z_b': -0.5,
            })
        return pd.DataFrame(rows)

    def test_fig11_variant_consequence_flow(self, variant_df, variant_figures_dir):
        plot_fig11_variant_consequence_flow(variant_df)
        assert any(f.startswith("11_") for f in os.listdir(variant_figures_dir)), \
            "Fig 11 output file not found"

    def test_fig12_variant_density_heatmap(self, variant_df, variant_figures_dir):
        plot_fig12_variant_density_heatmap(variant_df)
        assert any(f.startswith("12_") for f in os.listdir(variant_figures_dir)), \
            "Fig 12 output file not found"

    def test_fig13_variant_burden(self, variant_df, variant_figures_dir):
        plot_fig13_variant_burden(variant_df, density_mode=False)
        assert any(f.startswith("13_") for f in os.listdir(variant_figures_dir)), \
            "Fig 13 output file not found"


# ── Disease Name Parsing Tests ────────────────────────────────────

class TestDiseaseNameParsing:
    """Tests for _parse_disease_name helper used by Fig 15 Panel B."""

    def test_omim_with_acronym(self):
        assert _parse_disease_name('OMIM:618428:Popov-Chang syndrome (POPCHAS)') == 'Popov-Chang syndrome (POPCHAS)'

    def test_omim_no_acronym(self):
        assert _parse_disease_name('OMIM:154700:Marfan syndrome') == 'Marfan syndrome'

    def test_plain_name(self):
        assert _parse_disease_name('Cancer') == 'Cancer'

    def test_name_with_acronym(self):
        assert _parse_disease_name('Cardiovascular disease (CVD)') == 'Cardiovascular disease (CVD)'

    def test_empty(self):
        assert _parse_disease_name('') == ''

    def test_none(self):
        assert _parse_disease_name(None) == ''


# ── Phase E Column Detection Tests ───────────────────────────────

class TestDetectColumnsPhaseE:
    """Tests for Phase E column detection flags."""

    def test_disease_flag_present(self):
        df = pd.DataFrame({'n_diseases_a': [1]})
        assert detect_columns(df)['has_disease_data'] is True

    def test_disease_flag_absent(self):
        df = pd.DataFrame({'iptm': [0.5]})
        assert detect_columns(df)['has_disease_data'] is False

    def test_pathway_flag_present(self):
        df = pd.DataFrame({'reactome_pathways_a': ['R-HSA-1234:Test']})
        assert detect_columns(df)['has_pathway_data'] is True

    def test_pathway_flag_absent(self):
        df = pd.DataFrame({'iptm': [0.5]})
        assert detect_columns(df)['has_pathway_data'] is False


# ── Phase E Figure Generation Tests (Figs 14-15, 17) ────────────

@pytest.mark.phase_e
class TestPhaseEFigureGeneration:
    """Tests for Figs 14-15, 17 using a synthetic Phase E DataFrame."""

    @pytest.fixture(scope="class")
    def phase_e_figures_dir(self, test_output_dir):
        fig_dir = test_output_dir / "phase_e_figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        visualise_results.OUTPUT_DIR = str(fig_dir)
        return fig_dir

    @pytest.fixture(scope="class")
    def phase_e_df(self):
        """30-row synthetic DataFrame with pathway and disease columns."""
        rng = np.random.RandomState(42)

        pathway_pool = [
            'R-HSA-109581:Apoptosis',
            'R-HSA-72766:Translation',
            'R-HSA-168256:Immune System',
            'R-HSA-1640170:Cell Cycle',
            'R-HSA-162582:Signal Transduction',
            'R-HSA-392499:Metabolism of proteins',
            'R-HSA-1430728:Metabolism',
            'R-HSA-556833:Metabolism of lipids',
        ]

        # Disease name pool for Panel B testing
        disease_pool = [
            'OMIM:618428:Night blindness (CSNB1H)',
            'OMIM:130000:Ehlers-Danlos syndrome (EDS)',
            'OMIM:154700:Marfan syndrome',
            'Cancer',
            'OMIM:107970:Cardiovascular disease (CVD)',
            'OMIM:176000:Prostate cancer',
            'Retinitis pigmentosa',
            'OMIM:601419:Lodder-Merla syndrome',
        ]

        rows = []
        for i in range(30):
            tier = ['High', 'Medium', 'Low'][i % 3]

            # Pathway columns
            n_pw_a = rng.randint(1, 5)
            n_pw_b = rng.randint(1, 5)
            pw_a = rng.choice(pathway_pool, size=min(n_pw_a, len(pathway_pool)), replace=False)
            pw_b = rng.choice(pathway_pool, size=min(n_pw_b, len(pathway_pool)), replace=False)
            # Ensure all 4 bins have data
            if i < 8:
                shared = rng.randint(0, 4)
            elif i < 15:
                shared = rng.randint(4, 11)
            elif i < 22:
                shared = rng.randint(11, 31)
            else:
                shared = rng.randint(31, 55)

            # Disease details with parseable entries
            n_dis_a = rng.randint(0, 4)
            n_dis_b = rng.randint(0, 3)
            dis_a = '|'.join(rng.choice(disease_pool, size=n_dis_a, replace=False)) if n_dis_a > 0 else ''
            dis_b = '|'.join(rng.choice(disease_pool, size=n_dis_b, replace=False)) if n_dis_b > 0 else ''

            rows.append({
                'complex_name': f'PROT{i:02d}_PROT{i + 30:02d}',
                'iptm': 0.3 + 0.5 * (i / 30),
                'pdockq': 0.2 + 0.4 * (i / 30),
                'quality_tier': tier,
                'quality_tier_v2': tier,
                'interface_confidence_score': 0.3 + 0.5 * (i / 30),
                'n_interface_contacts': 20 + i * 3,
                'n_interface_residues_a': 10 + i,
                'n_interface_residues_b': 12 + i,
                # Pathway columns
                'reactome_pathways_a': '|'.join(pw_a),
                'reactome_pathways_b': '|'.join(pw_b),
                'n_reactome_pathways_a': len(pw_a),
                'n_reactome_pathways_b': len(pw_b),
                'n_shared_pathways': shared,
                'pathway_quality_context': f'mean_pdockq={0.3 + 0.01 * i:.3f};frac_high=0.300;n_complexes=10',
                'ppi_enrichment_pvalue': f'{rng.uniform(0, 0.05):.2e}',
                'ppi_enrichment_ratio': f'{rng.lognormal(0.5, 0.8):.2f}',
                'network_degree_a': rng.randint(1, 20),
                'network_degree_b': rng.randint(1, 20),
                # Disease columns
                'n_diseases_a': n_dis_a,
                'n_diseases_b': n_dis_b,
                'disease_details_a': dis_a,
                'disease_details_b': dis_b,
                'is_drug_target_a': bool(rng.random() > 0.9),
                'is_drug_target_b': False,
            })
        return pd.DataFrame(rows)

    def test_fig14_pathway_coherence(self, phase_e_df, phase_e_figures_dir):
        plot_fig14_pathway_coherence(phase_e_df)
        assert any(f.startswith("14_") for f in os.listdir(phase_e_figures_dir)), \
            "Fig 14 output file not found"

    def test_fig15_disease_enrichment(self, phase_e_df, phase_e_figures_dir):
        plot_fig15_disease_enrichment(phase_e_df)
        assert any(f.startswith("15_") for f in os.listdir(phase_e_figures_dir)), \
            "Fig 15 output file not found"

    def test_fig16_removed(self):
        """Verify Fig 16 function no longer exists."""
        assert not hasattr(visualise_results, 'plot_fig16_drug_target_quality'), \
            "plot_fig16_drug_target_quality should have been removed"

    def test_fig16_pathway_network(self, phase_e_df, phase_e_figures_dir):
        # Use top-5 pathways with low edge threshold for synthetic 30-row data
        plot_fig16_pathway_network(phase_e_df,
                                   max_pathways=5,
                                   min_shared_complexes=1)
        assert any(f.startswith("16_") for f in os.listdir(phase_e_figures_dir)), \
            "Fig 16 output file not found"

    def test_fig16_no_16b_disease_network(self, phase_e_figures_dir):
        """Verify 16b disease network is no longer generated."""
        assert not any(f.startswith("16b_") for f in os.listdir(phase_e_figures_dir)), \
            "16b_Disease_Network.png should not exist"

    def test_fig18_ptm_removed(self):
        """Verify old PTM Fig 18 function no longer exists."""
        assert not hasattr(visualise_results, 'plot_fig18_ptm_interface_landscape'), \
            "plot_fig18_ptm_interface_landscape should have been removed"

    def test_fig14_skips_missing_columns(self, phase_e_figures_dir):
        """Fig 14 should gracefully skip when pathway columns are missing."""
        df = pd.DataFrame({'iptm': [0.5], 'pdockq': [0.3]})
        plot_fig14_pathway_coherence(df)

    def test_fig15_skips_missing_columns(self, phase_e_figures_dir):
        """Fig 15 should gracefully skip when disease columns are missing."""
        df = pd.DataFrame({'iptm': [0.5], 'pdockq': [0.3]})
        plot_fig15_disease_enrichment(df)


class TestStabilityFigure:
    """Tests for Fig 17: Stability Predictor Cross-Validation."""

    @pytest.fixture(scope="class")
    def stability_figures_dir(self, test_output_dir):
        fig_dir = test_output_dir / "stability_figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        visualise_results.OUTPUT_DIR = str(fig_dir)
        return fig_dir

    @pytest.fixture(scope="class")
    def stability_df(self):
        """30-row synthetic DataFrame with stability + ProtVar columns."""
        rng = np.random.RandomState(42)
        tiers = ['High'] * 10 + ['Medium'] * 10 + ['Low'] * 10
        rows = []
        for i, tier in enumerate(tiers):
            eve_a = rng.uniform(0.1, 0.9) if rng.random() > 0.7 else np.nan
            eve_b = rng.uniform(0.1, 0.9) if rng.random() > 0.7 else np.nan
            am_a = rng.uniform(0.1, 0.9) if rng.random() > 0.2 else np.nan
            am_b = rng.uniform(0.1, 0.9) if rng.random() > 0.2 else np.nan
            fx_a = rng.uniform(0.1, 5.0) if rng.random() > 0.4 else np.nan
            fx_b = rng.uniform(0.1, 5.0) if rng.random() > 0.4 else np.nan
            rows.append({
                'quality_tier_v2': tier,
                'pdockq': rng.uniform(0.1, 0.8),
                'eve_score_mean_a': eve_a,
                'eve_score_mean_b': eve_b,
                'eve_coverage_a': 0.5 if pd.notna(eve_a) else 0.0,
                'eve_coverage_b': 0.5 if pd.notna(eve_b) else 0.0,
                'protvar_am_mean_a': am_a,
                'protvar_am_mean_b': am_b,
                'protvar_foldx_mean_a': fx_a,
                'protvar_foldx_mean_b': fx_b,
            })
        return pd.DataFrame(rows)

    def test_fig17_generates(self, stability_df, stability_figures_dir):
        plot_fig17_stability_crossvalidation(stability_df)
        assert any(f.startswith("17_") for f in os.listdir(stability_figures_dir)), \
            "Fig 17 output file not found"

    def test_fig17_skips_missing_columns(self, stability_figures_dir):
        """Fig 17 should gracefully skip when stability columns are missing."""
        df = pd.DataFrame({'iptm': [0.5], 'pdockq': [0.3]})
        plot_fig17_stability_crossvalidation(df)

    def test_fig17_handles_all_nan_eve(self, stability_figures_dir):
        """Fig 17 Panel A skips gracefully when all EVE values are NaN."""
        rng = np.random.RandomState(99)
        rows = []
        for tier in ['High'] * 10 + ['Medium'] * 10 + ['Low'] * 10:
            rows.append({
                'quality_tier_v2': tier,
                'pdockq': rng.uniform(0.1, 0.8),
                'eve_score_mean_a': np.nan,
                'eve_score_mean_b': np.nan,
                'eve_coverage_a': 0.0,
                'eve_coverage_b': 0.0,
                'protvar_am_mean_a': rng.uniform(0.1, 0.9),
                'protvar_am_mean_b': rng.uniform(0.1, 0.9),
                'protvar_foldx_mean_a': rng.uniform(0.1, 5.0),
                'protvar_foldx_mean_b': rng.uniform(0.1, 5.0),
            })
        df = pd.DataFrame(rows)
        # Should not crash — Panel A shows "Insufficient overlap"
        plot_fig17_stability_crossvalidation(df)


class TestClusteringFigure:
    """Tests for Fig 18: Sequence Clustering Validation."""

    @pytest.fixture(scope="class")
    def clustering_figures_dir(self, test_output_dir):
        fig_dir = test_output_dir / "clustering_figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        visualise_results.OUTPUT_DIR = str(fig_dir)
        return fig_dir

    @pytest.fixture(scope="class")
    def clustering_df(self):
        """30-row synthetic DataFrame with clustering columns."""
        rng = np.random.RandomState(42)
        tiers = ['High'] * 10 + ['Medium'] * 10 + ['Low'] * 10
        rows = []
        for i, tier in enumerate(tiers):
            is_homo = i < 5  # 5 homodimers
            seq_count = rng.randint(5, 100)
            if is_homo:
                shared_count = seq_count  # Perfect y=x for homodimers
            else:
                shared_count = rng.randint(0, seq_count + 1)
            rows.append({
                'quality_tier_v2': tier,
                'complex_type': 'homodimer' if is_homo else 'heterodimer',
                'sequence_cluster_count': seq_count,
                'shared_cluster_count': shared_count,
                'pdockq': rng.uniform(0.1, 0.8),
            })
        return pd.DataFrame(rows)

    def test_fig18_generates(self, clustering_df, clustering_figures_dir):
        plot_fig18_clustering_validation(clustering_df)
        assert any(f.startswith("18_") for f in os.listdir(clustering_figures_dir)), \
            "Fig 18 output file not found"

    def test_fig18_skips_missing_columns(self, clustering_figures_dir):
        """Fig 18 should gracefully skip when clustering columns are missing."""
        df = pd.DataFrame({'iptm': [0.5], 'pdockq': [0.3]})
        plot_fig18_clustering_validation(df)

    def test_fig18_homodimer_ground_truth(self, clustering_df):
        """All homodimers in fixture should have shared == sequence."""
        homo = clustering_df[clustering_df['complex_type'] == 'homodimer']
        assert (homo['shared_cluster_count'] == homo['sequence_cluster_count']).all()
