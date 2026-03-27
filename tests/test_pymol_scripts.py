"""
Tests for pymol_scripts.py — PyMOL .pml script generation and py3Dmol fallback.

Test data:
    - Reference complex 1: A0A0B4J2C3_P24534 (heterodimer, Test_Data/)
    - Expected: 82 interface residues (39 A + 43 B), chain pair A-B

Tests work WITHOUT PyMOL installed — they verify .pml text content.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest


# ── Constants & Known Values ─────────────────────────────────────

REF_COMPLEX_1 = 'A0A0B4J2C3_P24534'
REF_INTERFACE_RESIDUES_A = 39
REF_INTERFACE_RESIDUES_B = 43
REF_INTERFACE_TOTAL = 82
REF_CHAIN_A = 'A'
REF_CHAIN_B = 'B'


# ── Section 1: Constants ─────────────────────────────────────────

@pytest.mark.pymol
class TestConstants:
    """Verify module constants are properly defined."""

    def test_chain_colours_has_entries(self):
        """CHAIN_COLOURS maps at least 6 chain IDs."""
        from pymol_scripts import CHAIN_COLOURS
        assert len(CHAIN_COLOURS) >= 6

    def test_chain_colours_has_common_chains(self):
        """CHAIN_COLOURS includes A and B."""
        from pymol_scripts import CHAIN_COLOURS
        assert 'A' in CHAIN_COLOURS
        assert 'B' in CHAIN_COLOURS

    def test_plddt_colour_bands_covers_4_ranges(self):
        """PLDDT_COLOUR_BANDS defines exactly 4 confidence bands."""
        from pymol_scripts import PLDDT_COLOUR_BANDS
        assert len(PLDDT_COLOUR_BANDS) == 4

    def test_variant_context_colours_has_all_4_contexts(self):
        """VARIANT_CONTEXT_COLOURS maps all 4 structural contexts."""
        from pymol_scripts import VARIANT_CONTEXT_COLOURS
        assert 'interface_core' in VARIANT_CONTEXT_COLOURS
        assert 'interface_rim' in VARIANT_CONTEXT_COLOURS
        assert 'surface_non_interface' in VARIANT_CONTEXT_COLOURS
        assert 'buried_core' in VARIANT_CONTEXT_COLOURS

    def test_interface_colour_is_string(self):
        """INTERFACE_COLOUR is a non-empty string."""
        from pymol_scripts import INTERFACE_COLOUR
        assert isinstance(INTERFACE_COLOUR, str)
        assert len(INTERFACE_COLOUR) > 0


# ── Section 2: PML Header ───────────────────────────────────────

@pytest.mark.pymol
class TestPMLHeader:
    """Tests for generate_pml_header()."""

    def test_contains_load_command(self):
        """Header includes a 'load' command with the PDB path."""
        from pymol_scripts import generate_pml_header
        header = generate_pml_header('/path/to/complex.pdb', 'test_complex')
        assert 'load /path/to/complex.pdb' in header

    def test_contains_complex_name(self):
        """Header uses the complex name as PyMOL object name."""
        from pymol_scripts import generate_pml_header
        header = generate_pml_header('/path/to/complex.pdb', 'my_complex')
        assert 'my_complex' in header

    def test_contains_show_cartoon(self):
        """Header includes basic display commands."""
        from pymol_scripts import generate_pml_header
        header = generate_pml_header('/path/to/complex.pdb', 'test')
        assert 'show cartoon' in header
        assert 'hide everything' in header
        assert 'bg_color white' in header


# ── Section 3: Chain Colouring ───────────────────────────────────

@pytest.mark.pymol
class TestChainColouring:
    """Tests for generate_chain_colouring()."""

    def test_both_chains_coloured(self):
        """Output contains colour commands for both chains."""
        from pymol_scripts import generate_chain_colouring
        result = generate_chain_colouring('A', 'B')
        assert 'chain A' in result
        assert 'chain B' in result
        assert result.count('color ') == 2

    def test_different_colours_for_chains(self):
        """Chains A and B get different colours."""
        from pymol_scripts import generate_chain_colouring, CHAIN_COLOURS
        result = generate_chain_colouring('A', 'B')
        colour_a = CHAIN_COLOURS['A']
        colour_b = CHAIN_COLOURS['B']
        assert colour_a != colour_b
        assert colour_a in result
        assert colour_b in result

    def test_unknown_chain_gets_fallback_colour(self):
        """Chains not in CHAIN_COLOURS get fallback colours."""
        from pymol_scripts import generate_chain_colouring
        result = generate_chain_colouring('Z', 'Y')
        # Should still produce valid commands, not crash
        assert 'chain Z' in result
        assert 'chain Y' in result

    def test_contains_section_comment(self):
        """Output has a section header comment."""
        from pymol_scripts import generate_chain_colouring
        result = generate_chain_colouring('A', 'B')
        assert '# -- Chain colouring --' in result


# ── Section 4: pLDDT Colouring ───────────────────────────────────

@pytest.mark.pymol
class TestPLDDTColouring:
    """Tests for generate_plddt_colouring()."""

    def test_has_four_bands(self):
        """Output defines 4 pLDDT selection bands."""
        from pymol_scripts import generate_plddt_colouring
        result = generate_plddt_colouring()
        assert 'plddt_vhigh' in result
        assert 'plddt_high' in result
        assert 'plddt_low' in result
        assert 'plddt_vlow' in result

    def test_uses_b_factor_selections(self):
        """Selections use B-factor conditions."""
        from pymol_scripts import generate_plddt_colouring
        result = generate_plddt_colouring()
        assert 'b > 90' in result
        assert 'b <= 50' in result

    def test_uses_canonical_af2_colours(self):
        """Colours match the AlphaFold2 confidence scheme."""
        from pymol_scripts import generate_plddt_colouring
        result = generate_plddt_colouring()
        # Blue for very high confidence
        assert '0x0053D6' in result
        # Orange for very low confidence
        assert '0xFF7D45' in result

    def test_ends_with_deselect(self):
        """Output ends with 'deselect' to clear selection."""
        from pymol_scripts import generate_plddt_colouring
        result = generate_plddt_colouring()
        assert 'deselect' in result

    def test_contains_section_comment(self):
        """Output has a section header comment."""
        from pymol_scripts import generate_plddt_colouring
        result = generate_plddt_colouring()
        assert '# -- pLDDT colouring' in result


# ── Section 5: Interface Highlighting ────────────────────────────

@pytest.mark.pymol
class TestInterfaceHighlighting:
    """Tests for generate_interface_highlighting()."""

    def test_creates_selections_for_both_chains(self):
        """Output defines selections for interface residues on both chains."""
        from pymol_scripts import generate_interface_highlighting
        result = generate_interface_highlighting('A', 'B', [10, 20], [30, 40])
        assert 'select interface_a' in result
        assert 'select interface_b' in result

    def test_residue_numbers_in_resi_string(self):
        """PDB residue numbers appear in the resi selection."""
        from pymol_scripts import generate_interface_highlighting
        result = generate_interface_highlighting('A', 'B', [10, 20, 30], [])
        assert '10+20+30' in result

    def test_shows_sticks(self):
        """Interface residues are shown as sticks."""
        from pymol_scripts import generate_interface_highlighting
        result = generate_interface_highlighting('A', 'B', [10], [20])
        assert 'show sticks' in result

    def test_colours_with_interface_colour(self):
        """Interface residues are coloured with INTERFACE_COLOUR."""
        from pymol_scripts import generate_interface_highlighting, INTERFACE_COLOUR
        result = generate_interface_highlighting('A', 'B', [10], [20])
        assert INTERFACE_COLOUR in result

    def test_empty_lists_return_empty_string(self):
        """Empty interface lists produce no output."""
        from pymol_scripts import generate_interface_highlighting
        result = generate_interface_highlighting('A', 'B', [], [])
        assert result == ''

    def test_single_chain_only(self):
        """Works when only one chain has interface residues."""
        from pymol_scripts import generate_interface_highlighting
        result = generate_interface_highlighting('A', 'B', [10, 20], [])
        assert 'interface_a' in result
        assert 'interface_b' not in result


# ── Section 6: Variant Highlighting ──────────────────────────────

@pytest.mark.pymol
class TestVariantHighlighting:
    """Tests for generate_variant_highlighting()."""

    def test_creates_context_grouped_selections(self):
        """Variants grouped by structural context into named selections."""
        from pymol_scripts import generate_variant_highlighting
        variants = [
            {'position': 10, 'context': 'interface_core'},
            {'position': 20, 'context': 'interface_rim'},
        ]
        result = generate_variant_highlighting('A', 'B', variants, None)
        assert 'var_interface_core_A' in result
        assert 'var_interface_rim_A' in result

    def test_uses_spheres_representation(self):
        """Variant positions shown as spheres."""
        from pymol_scripts import generate_variant_highlighting
        variants = [{'position': 10, 'context': 'interface_core'}]
        result = generate_variant_highlighting('A', 'B', variants, None)
        assert 'show spheres' in result

    def test_context_colours_applied(self):
        """Each structural context gets its designated colour."""
        from pymol_scripts import generate_variant_highlighting, VARIANT_CONTEXT_COLOURS
        variants = [{'position': 10, 'context': 'interface_core'}]
        result = generate_variant_highlighting('A', 'B', variants, None)
        expected_colour = VARIANT_CONTEXT_COLOURS['interface_core']
        assert expected_colour in result

    def test_none_variants_return_empty_string(self):
        """None for both variant lists produces no output."""
        from pymol_scripts import generate_variant_highlighting
        result = generate_variant_highlighting('A', 'B', None, None)
        assert result == ''

    def test_empty_lists_return_empty_string(self):
        """Empty lists produce no output."""
        from pymol_scripts import generate_variant_highlighting
        result = generate_variant_highlighting('A', 'B', [], [])
        assert result == ''

    def test_both_chains_with_variants(self):
        """Variants on both chains are included."""
        from pymol_scripts import generate_variant_highlighting
        var_a = [{'position': 10, 'context': 'interface_core'}]
        var_b = [{'position': 30, 'context': 'surface_non_interface'}]
        result = generate_variant_highlighting('A', 'B', var_a, var_b)
        assert 'chain A' in result
        assert 'chain B' in result

    def test_unknown_context_gets_default_colour(self):
        """Variants with unknown context get the fallback colour."""
        from pymol_scripts import generate_variant_highlighting, VARIANT_DEFAULT_COLOUR
        variants = [{'position': 10, 'context': 'some_new_context'}]
        result = generate_variant_highlighting('A', 'B', variants, None)
        assert VARIANT_DEFAULT_COLOUR in result

    def test_duplicate_positions_deduplicated(self):
        """Multiple variants at the same position use sorted unique resi."""
        from pymol_scripts import generate_variant_highlighting
        variants = [
            {'position': 10, 'context': 'interface_core'},
            {'position': 10, 'context': 'interface_core'},
        ]
        result = generate_variant_highlighting('A', 'B', variants, None)
        # Should have '10' not '10+10'
        assert '10+10' not in result


# ── Section 7: Build Complete Script ─────────────────────────────

@pytest.mark.pymol
class TestBuildScript:
    """Tests for build_pymol_script()."""

    def test_sections_in_order(self):
        """Complete script has header, chains, pLDDT, interface, rendering."""
        from pymol_scripts import build_pymol_script
        script = build_pymol_script(
            '/path/complex.pdb', 'test', 'A', 'B', [10, 20], [30, 40])
        # Check ordering by finding positions
        pos_load = script.index('load')
        pos_chain = script.index('Chain colouring')
        pos_plddt = script.index('pLDDT colouring')
        pos_iface = script.index('Interface residues')
        pos_camera = script.index('Camera and rendering')
        assert pos_load < pos_chain < pos_plddt < pos_iface < pos_camera

    def test_render_png_false_comments_ray(self):
        """render_png=False comments out ray and png commands."""
        from pymol_scripts import build_pymol_script
        script = build_pymol_script(
            '/path/complex.pdb', 'test', 'A', 'B', [10], [20],
            render_png=False)
        assert '# ray' in script
        assert '# png' in script

    def test_render_png_true_uncomments_ray(self):
        """render_png=True produces uncommented ray and png commands."""
        from pymol_scripts import build_pymol_script
        script = build_pymol_script(
            '/path/complex.pdb', 'test', 'A', 'B', [10], [20],
            render_png=True, output_png_path='/out/test.png')
        # Should have 'ray' without leading '#'
        lines = script.split('\n')
        ray_lines = [l for l in lines if 'ray ' in l and 'opaque' not in l]
        assert any(not l.strip().startswith('#') for l in ray_lines)

    def test_orient_and_zoom_present(self):
        """Script includes orient and zoom commands."""
        from pymol_scripts import build_pymol_script
        script = build_pymol_script(
            '/path/complex.pdb', 'test', 'A', 'B', [10], [20])
        assert 'orient' in script
        assert 'zoom' in script

    def test_variant_section_included_when_provided(self):
        """Variant highlighting section appears when variants are given."""
        from pymol_scripts import build_pymol_script
        variants = [{'position': 10, 'context': 'interface_core'}]
        script = build_pymol_script(
            '/path/complex.pdb', 'test', 'A', 'B', [10], [20],
            variant_records_a=variants)
        assert 'Variant positions' in script

    def test_no_variant_section_without_variants(self):
        """Script has no variant section when no variants provided."""
        from pymol_scripts import build_pymol_script
        script = build_pymol_script(
            '/path/complex.pdb', 'test', 'A', 'B', [10], [20])
        assert 'Variant positions' not in script


# ── Section 8: Variant Detail Parsing ────────────────────────────

@pytest.mark.pymol
class TestVariantDetailParsing:
    """Tests for parse_variant_details_for_pymol()."""

    def test_parses_pipe_separated_details(self):
        """Parses standard variant detail format."""
        from pymol_scripts import parse_variant_details_for_pymol
        records = parse_variant_details_for_pymol(
            'K81P:interface_core:pathogenic|E82K:interface_rim:VUS')
        assert len(records) == 2
        assert records[0]['position'] == 81
        assert records[0]['context'] == 'interface_core'
        assert records[1]['position'] == 82

    def test_empty_string_returns_empty_list(self):
        """Empty string produces no records."""
        from pymol_scripts import parse_variant_details_for_pymol
        assert parse_variant_details_for_pymol('') == []

    def test_dash_returns_empty_list(self):
        """Dash placeholder produces no records."""
        from pymol_scripts import parse_variant_details_for_pymol
        assert parse_variant_details_for_pymol('-') == []

    def test_skips_truncation_marker(self):
        """Handles '...(+N more)' truncation markers gracefully."""
        from pymol_scripts import parse_variant_details_for_pymol
        records = parse_variant_details_for_pymol(
            'K81P:interface_core:pathogenic|...(+5 more)')
        assert len(records) == 1

    def test_extracts_ref_alt_clinical(self):
        """Extracts ref, alt, and clinical significance fields."""
        from pymol_scripts import parse_variant_details_for_pymol
        records = parse_variant_details_for_pymol('R45G:buried_core:benign')
        assert records[0]['ref'] == 'R'
        assert records[0]['alt'] == 'G'
        assert records[0]['clinical'] == 'benign'


# ── Section 9: Script Generation (I/O, slow) ────────────────────

@pytest.mark.pymol
@pytest.mark.slow
class TestScriptGeneration:
    """Tests for generate_pymol_scripts_for_results() and extract_interface_data()."""

    def test_extract_interface_data(self, ref_pdb_1):
        """extract_interface_data returns chain pair and interface residues."""
        from pymol_scripts import extract_interface_data
        iface = extract_interface_data(ref_pdb_1)
        assert iface['chain_a'] == REF_CHAIN_A
        assert iface['chain_b'] == REF_CHAIN_B
        assert isinstance(iface['interface_resi_a'], list)
        assert isinstance(iface['interface_resi_b'], list)

    def test_extract_interface_residue_count(self, ref_pdb_1):
        """Reference complex 1 has 82 interface residues (39 + 43)."""
        from pymol_scripts import extract_interface_data
        iface = extract_interface_data(ref_pdb_1)
        assert len(iface['interface_resi_a']) == REF_INTERFACE_RESIDUES_A
        assert len(iface['interface_resi_b']) == REF_INTERFACE_RESIDUES_B

    def test_extract_interface_missing_pdb(self, tmp_path):
        """extract_interface_data raises FileNotFoundError for missing PDB."""
        from pymol_scripts import extract_interface_data
        with pytest.raises(FileNotFoundError):
            extract_interface_data(tmp_path / 'nonexistent.pdb')

    def test_generates_pml_file(self, ref_pdb_1, tmp_path):
        """Generates a .pml file for a High-tier complex."""
        from pymol_scripts import generate_pymol_scripts_for_results
        results = [{
            'complex_name': REF_COMPLEX_1,
            'quality_tier_v2': 'High',
        }]
        n = generate_pymol_scripts_for_results(
            results, pdb_dir=str(ref_pdb_1.parent), output_dir=str(tmp_path))
        assert n == 1
        pml_path = tmp_path / f"{REF_COMPLEX_1}.pml"
        assert pml_path.exists()
        content = pml_path.read_text(encoding='utf-8')
        assert 'load' in content
        assert REF_COMPLEX_1 in content

    def test_skips_low_tier(self, ref_pdb_1, tmp_path):
        """Low-tier complexes are skipped with default min_tier='High'."""
        from pymol_scripts import generate_pymol_scripts_for_results
        results = [{
            'complex_name': REF_COMPLEX_1,
            'quality_tier_v2': 'Low',
        }]
        n = generate_pymol_scripts_for_results(
            results, pdb_dir=str(ref_pdb_1.parent), output_dir=str(tmp_path))
        assert n == 0

    def test_creates_output_directory(self, ref_pdb_1, tmp_path):
        """Output directory is created if it doesn't exist."""
        from pymol_scripts import generate_pymol_scripts_for_results
        out = tmp_path / 'new_subdir' / 'pymol'
        results = [{
            'complex_name': REF_COMPLEX_1,
            'quality_tier_v2': 'High',
        }]
        generate_pymol_scripts_for_results(
            results, pdb_dir=str(ref_pdb_1.parent), output_dir=str(out))
        assert out.is_dir()

    def test_skips_missing_pdb(self, tmp_path):
        """Complexes with missing PDB files are skipped."""
        from pymol_scripts import generate_pymol_scripts_for_results
        results = [{
            'complex_name': 'NONEXISTENT_COMPLEX',
            'quality_tier_v2': 'High',
        }]
        n = generate_pymol_scripts_for_results(
            results, pdb_dir=str(tmp_path), output_dir=str(tmp_path / 'out'))
        assert n == 0

    def test_includes_variant_details(self, ref_pdb_1, tmp_path):
        """When variant details are in result, they appear in the .pml."""
        from pymol_scripts import generate_pymol_scripts_for_results
        results = [{
            'complex_name': REF_COMPLEX_1,
            'quality_tier_v2': 'High',
            'variant_details_a': 'K10P:interface_core:pathogenic',
            'variant_details_b': '',
        }]
        generate_pymol_scripts_for_results(
            results, pdb_dir=str(ref_pdb_1.parent), output_dir=str(tmp_path),
            include_variants=True)
        content = (tmp_path / f"{REF_COMPLEX_1}.pml").read_text()
        assert 'Variant positions' in content
        assert 'interface_core' in content


# ── Section 10: py3Dmol Fallback ─────────────────────────────────

@pytest.mark.pymol
class TestPy3DmolFallback:
    """Tests for py3Dmol fallback functionality."""

    def test_has_py3dmol_flag_is_bool(self):
        """_HAS_PY3DMOL is a boolean flag."""
        from pymol_scripts import _HAS_PY3DMOL
        assert isinstance(_HAS_PY3DMOL, bool)

    def test_generate_py3dmol_view_returns_none_when_unavailable(self, ref_pdb_1):
        """Returns None when py3Dmol is not installed."""
        from pymol_scripts import generate_py3dmol_view, _HAS_PY3DMOL
        if _HAS_PY3DMOL:
            pytest.skip("py3Dmol is installed; cannot test fallback")
        result = generate_py3dmol_view(
            ref_pdb_1, 'A', 'B', [10, 20], [30, 40])
        assert result is None

    def test_generate_py3dmol_view_function_exists(self):
        """generate_py3dmol_view is importable."""
        from pymol_scripts import generate_py3dmol_view
        assert callable(generate_py3dmol_view)


# ── Section 11: CLI ──────────────────────────────────────────────

@pytest.mark.pymol
@pytest.mark.cli
class TestCLI:
    """Tests for standalone CLI invocation."""

    def test_help_exits_zero(self):
        """--help exits with code 0."""
        result = subprocess.run(
            [sys.executable, '-m', 'pymol_scripts', '--help'],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert 'generate' in result.stdout
        assert 'batch' in result.stdout

    def test_generate_missing_pdb_exits_error(self, tmp_path):
        """'generate' with nonexistent PDB exits with error."""
        result = subprocess.run(
            [sys.executable, '-m', 'pymol_scripts', 'generate',
             '--pdb', str(tmp_path / 'missing.pdb')],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0

    @pytest.mark.slow
    def test_generate_produces_pml(self, ref_pdb_1, tmp_path):
        """'generate' subcommand creates a .pml file."""
        result = subprocess.run(
            [sys.executable, '-m', 'pymol_scripts', 'generate',
             '--pdb', str(ref_pdb_1), '--output', str(tmp_path)],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0
        pml_files = list(tmp_path.glob('*.pml'))
        assert len(pml_files) == 1

    def test_batch_missing_csv_exits_error(self, tmp_path):
        """'batch' with nonexistent CSV exits with error."""
        result = subprocess.run(
            [sys.executable, '-m', 'pymol_scripts', 'batch',
             '--csv', str(tmp_path / 'missing.csv'),
             '--pdb-dir', str(tmp_path)],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0


# ── Section 12: Regression Tests ─────────────────────────────────

@pytest.mark.pymol
@pytest.mark.regression
@pytest.mark.slow
class TestRegressionComplex1:
    """Regression tests for reference complex 1."""

    def test_interface_residue_count_in_pml(self, ref_pdb_1, tmp_path):
        """Generated .pml references the correct number of interface residues."""
        from pymol_scripts import generate_pymol_scripts_for_results
        results = [{
            'complex_name': REF_COMPLEX_1,
            'quality_tier_v2': 'High',
        }]
        generate_pymol_scripts_for_results(
            results, pdb_dir=str(ref_pdb_1.parent), output_dir=str(tmp_path))
        content = (tmp_path / f"{REF_COMPLEX_1}.pml").read_text()
        # Count residue numbers in interface_a selection
        import re
        match_a = re.search(r'select interface_a, chain A and resi (.+)', content)
        assert match_a, "interface_a selection not found in .pml"
        resi_a = match_a.group(1).split('+')
        assert len(resi_a) == REF_INTERFACE_RESIDUES_A

        match_b = re.search(r'select interface_b, chain B and resi (.+)', content)
        assert match_b, "interface_b selection not found in .pml"
        resi_b = match_b.group(1).split('+')
        assert len(resi_b) == REF_INTERFACE_RESIDUES_B

    def test_chain_pair_is_a_b(self, ref_pdb_1):
        """Reference complex 1 best chain pair is A-B."""
        from pymol_scripts import extract_interface_data
        iface = extract_interface_data(ref_pdb_1)
        assert iface['chain_a'] == 'A'
        assert iface['chain_b'] == 'B'

    def test_pml_has_plddt_sections(self, ref_pdb_1, tmp_path):
        """Generated .pml contains pLDDT colouring section."""
        from pymol_scripts import generate_pymol_scripts_for_results
        results = [{
            'complex_name': REF_COMPLEX_1,
            'quality_tier_v2': 'High',
        }]
        generate_pymol_scripts_for_results(
            results, pdb_dir=str(ref_pdb_1.parent), output_dir=str(tmp_path))
        content = (tmp_path / f"{REF_COMPLEX_1}.pml").read_text()
        assert 'plddt_vhigh' in content
        assert 'plddt_vlow' in content
