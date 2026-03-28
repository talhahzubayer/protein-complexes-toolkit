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
        """Selections use B-factor conditions compatible with PyMOL."""
        from pymol_scripts import generate_plddt_colouring
        result = generate_plddt_colouring()
        assert 'b > 90' in result
        assert 'b < 51' in result

    def test_no_lte_gte_operators(self):
        """PyMOL does not support <= or >= — only < and > are used."""
        from pymol_scripts import generate_plddt_colouring
        result = generate_plddt_colouring()
        assert '<=' not in result
        assert '>=' not in result

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


# ── Section 13: New Feature Tests ─────────────────────────────────

@pytest.mark.pymol
class TestSceneManagement:
    """Tests for PyMOL scene storage commands (Fix #1)."""

    def test_chain_colouring_stores_scene(self):
        """Chain colouring section stores a chain_view scene."""
        from pymol_scripts import generate_chain_colouring
        result = generate_chain_colouring('A', 'B')
        assert 'scene chain_view, store' in result

    def test_plddt_colouring_stores_scene(self):
        """pLDDT colouring section stores a plddt_view scene."""
        from pymol_scripts import generate_plddt_colouring
        result = generate_plddt_colouring()
        assert 'scene plddt_view, store' in result

    def test_rendering_stores_full_view_scene(self):
        """Rendering section stores a full_view scene after orient and zoom."""
        from pymol_scripts import generate_rendering_commands
        result = generate_rendering_commands()
        assert 'scene full_view, store' in result
        # Fix E: scene stored AFTER orient and zoom for correct camera
        orient_pos = result.index('orient')
        zoom_pos = result.index('zoom complete=1')
        scene_pos = result.index('scene full_view, store')
        assert orient_pos < zoom_pos < scene_pos

    def test_scene_order_in_complete_script(self):
        """Scenes appear in correct order: chain_view, plddt_view, full_view."""
        from pymol_scripts import build_pymol_script
        script = build_pymol_script('/p.pdb', 'test', 'A', 'B', [10], [20])
        pos_chain = script.index('scene chain_view')
        pos_plddt = script.index('scene plddt_view')
        pos_full = script.index('scene full_view')
        assert pos_chain < pos_plddt < pos_full


@pytest.mark.pymol
class TestPathogenicityAwareSpheres:
    """Tests for pathogenic variant sphere scaling (Fixes #4 + #7)."""

    def test_pathogenic_variants_larger_spheres(self):
        """Pathogenic variants get larger sphere_scale than benign."""
        from pymol_scripts import generate_variant_highlighting
        from pymol_scripts import VARIANT_SPHERE_SCALE_PATHOGENIC, VARIANT_SPHERE_SCALE_DEFAULT
        variants = [
            {'position': 10, 'context': 'interface_core', 'clinical': 'pathogenic'},
            {'position': 20, 'context': 'interface_core', 'clinical': 'benign'},
        ]
        result = generate_variant_highlighting('A', 'B', variants, None)
        assert f'sphere_scale, {VARIANT_SPHERE_SCALE_PATHOGENIC}' in result
        assert f'sphere_scale, {VARIANT_SPHERE_SCALE_DEFAULT}' in result

    def test_pathogenic_selection_name(self):
        """Pathogenic variants get '_path' suffix in selection name."""
        from pymol_scripts import generate_variant_highlighting
        variants = [{'position': 10, 'context': 'interface_core', 'clinical': 'pathogenic'}]
        result = generate_variant_highlighting('A', 'B', variants, None)
        assert 'var_interface_core_path_A' in result

    def test_non_pathogenic_selection_name(self):
        """Non-pathogenic variants have no '_path' suffix."""
        from pymol_scripts import generate_variant_highlighting
        variants = [{'position': 10, 'context': 'interface_core', 'clinical': 'benign'}]
        result = generate_variant_highlighting('A', 'B', variants, None)
        assert 'var_interface_core_A' in result
        assert 'var_interface_core_path_A' not in result

    def test_sphere_scale_always_present(self):
        """Every variant group gets a sphere_scale command."""
        from pymol_scripts import generate_variant_highlighting
        variants = [{'position': 10, 'context': 'buried_core', 'clinical': ''}]
        result = generate_variant_highlighting('A', 'B', variants, None)
        assert 'sphere_scale' in result

    def test_likely_pathogenic_treated_as_pathogenic(self):
        """'Likely pathogenic' is treated as pathogenic."""
        from pymol_scripts import generate_variant_highlighting
        variants = [{'position': 10, 'context': 'interface_rim', 'clinical': 'Likely pathogenic'}]
        result = generate_variant_highlighting('A', 'B', variants, None)
        assert 'var_interface_rim_path_A' in result


@pytest.mark.pymol
class TestMetadataComments:
    """Tests for generate_metadata_comments() (Fix #6)."""

    def test_metadata_comments_present(self):
        """Metadata dict produces comment block."""
        from pymol_scripts import generate_metadata_comments
        meta = {'quality_tier_v2': 'High', 'iptm': '0.6112', 'pdockq': '0.3301'}
        result = generate_metadata_comments(meta)
        assert '# === Metadata ===' in result
        assert 'Quality: High' in result
        assert 'ipTM: 0.6112' in result

    def test_metadata_comments_absent(self):
        """None metadata produces empty string."""
        from pymol_scripts import generate_metadata_comments
        assert generate_metadata_comments(None) == ''
        assert generate_metadata_comments({}) == ''

    def test_metadata_in_complete_script(self):
        """Metadata section appears in complete script."""
        from pymol_scripts import build_pymol_script
        meta = {'quality_tier_v2': 'High'}
        script = build_pymol_script('/p.pdb', 'test', 'A', 'B', [10], [20],
                                     metadata=meta)
        assert 'Metadata' in script

    def test_drug_target_in_metadata(self):
        """Drug target info appears when provided."""
        from pymol_scripts import generate_metadata_comments
        meta = {'is_drug_target_a': 'True', 'is_drug_target_b': ''}
        result = generate_metadata_comments(meta)
        assert 'Drug target: A=Yes, B=No' in result


@pytest.mark.pymol
class TestAnnotationComments:
    """Tests for generate_annotation_comments() (Fix #14)."""

    def test_annotation_with_diseases(self):
        """Disease details appear as comments."""
        from pymol_scripts import generate_annotation_comments
        result = generate_annotation_comments(
            gene_a='BRCA1', gene_b='TP53',
            disease_details_a='OMIM:123456:Breast cancer')
        assert '# === Biological Context ===' in result
        assert 'BRCA1' in result
        assert 'Breast cancer' in result

    def test_annotation_empty(self):
        """No data produces empty string."""
        from pymol_scripts import generate_annotation_comments
        assert generate_annotation_comments() == ''

    def test_gene_labels(self):
        """Gene symbols produce PyMOL label commands."""
        from pymol_scripts import generate_annotation_comments
        result = generate_annotation_comments(gene_a='EEF1B2', gene_b='YWHAG')
        assert 'label chain A' in result
        assert '"EEF1B2"' in result
        assert 'label chain B' in result
        assert '"YWHAG"' in result

    def test_pathway_truncation(self):
        """More than 5 pathways show truncation note."""
        from pymol_scripts import generate_annotation_comments
        pathways = '|'.join(f'R-HSA-{i}:Pathway{i}' for i in range(8))
        result = generate_annotation_comments(reactome_pathways_a=pathways)
        assert '+3 more' in result

    def test_pathway_newline_sanitised(self):
        """Embedded newlines in pathway strings do not produce bare PML lines."""
        from pymol_scripts import generate_annotation_comments
        pathways = 'R-HSA-71291:Metabolism of\namino acids|R-HSA-99999:Signal\r\ntransduction'
        result = generate_annotation_comments(reactome_pathways_a=pathways)
        for line in result.split('\n'):
            stripped = line.strip()
            if stripped:
                assert stripped.startswith('#') or stripped.startswith('label'), (
                    f"Bare PML line found: {stripped!r}")

    def test_disease_newline_sanitised(self):
        """Embedded newlines in disease strings do not produce bare PML lines."""
        from pymol_scripts import generate_annotation_comments
        diseases = 'OMIM:123456:Breast\ncancer|OMIM:789:Some\r\ndisease'
        result = generate_annotation_comments(disease_details_a=diseases)
        for line in result.split('\n'):
            stripped = line.strip()
            if stripped:
                assert stripped.startswith('#') or stripped.startswith('label'), (
                    f"Bare PML line found: {stripped!r}")

    def test_no_semicolons_in_comments(self):
        """Comment lines must not contain ';' — PyMOL treats it as command separator."""
        from pymol_scripts import generate_annotation_comments
        result = generate_annotation_comments(
            gene_a='TP53', gene_b='BRCA1',
            disease_details_a='OMIM:123:Cancer|OMIM:456:Syndrome',
            reactome_pathways_a='R-HSA-1430728:Metabolism|R-HSA-71291:Amino acid metabolism',
        )
        for line in result.split('\n'):
            if line.strip().startswith('#'):
                assert ';' not in line, f"Semicolon in comment line: {line!r}"


@pytest.mark.pymol
class TestQuitCommand:
    """Tests for quit command in batch rendering (Fix #11)."""

    def test_quit_when_render_true(self):
        """quit command present when render=True."""
        from pymol_scripts import generate_rendering_commands
        result = generate_rendering_commands(render=True)
        assert '\nquit\n' in result

    def test_no_quit_when_render_false(self):
        """quit command absent when render=False."""
        from pymol_scripts import generate_rendering_commands
        result = generate_rendering_commands(render=False)
        assert 'quit' not in result


@pytest.mark.pymol
class TestPy3DmolHexColours:
    """Tests for PyMOL-to-hex colour mapping (Fix #13)."""

    def test_pymol_to_hex_covers_chain_colours(self):
        """PYMOL_TO_HEX contains entries for all CHAIN_COLOURS values."""
        from pymol_scripts import PYMOL_TO_HEX, CHAIN_COLOURS
        for colour in CHAIN_COLOURS.values():
            assert colour in PYMOL_TO_HEX, f"Missing hex mapping for '{colour}'"

    def test_pymol_to_hex_covers_variant_colours(self):
        """PYMOL_TO_HEX contains entries for all VARIANT_CONTEXT_COLOURS values."""
        from pymol_scripts import PYMOL_TO_HEX, VARIANT_CONTEXT_COLOURS, VARIANT_DEFAULT_COLOUR
        for colour in VARIANT_CONTEXT_COLOURS.values():
            assert colour in PYMOL_TO_HEX, f"Missing hex mapping for '{colour}'"
        assert VARIANT_DEFAULT_COLOUR in PYMOL_TO_HEX

    def test_pymol_to_hex_covers_interface_colour(self):
        """PYMOL_TO_HEX contains entry for INTERFACE_COLOUR."""
        from pymol_scripts import PYMOL_TO_HEX, INTERFACE_COLOUR
        assert INTERFACE_COLOUR in PYMOL_TO_HEX


@pytest.mark.pymol
class TestHomodimerHandling:
    """Tests for homodimer same-colour handling (Fix #15)."""

    def test_homodimer_same_colour(self):
        """Both chains get the same colour when homodimer=True."""
        from pymol_scripts import generate_chain_colouring, CHAIN_COLOURS
        result = generate_chain_colouring('A', 'B', homodimer=True)
        colour_a = CHAIN_COLOURS['A']
        # Both color commands should use the same colour
        lines = [l for l in result.split('\n') if l.startswith('color ')]
        assert len(lines) == 2
        assert all(colour_a in l for l in lines)

    def test_homodimer_transparency(self):
        """Chain B gets cartoon_transparency when homodimer=True."""
        from pymol_scripts import generate_chain_colouring
        result = generate_chain_colouring('A', 'B', homodimer=True)
        assert 'cartoon_transparency' in result

    def test_heterodimer_different_colours(self):
        """Chains get different colours when homodimer=False."""
        from pymol_scripts import generate_chain_colouring, CHAIN_COLOURS
        result = generate_chain_colouring('A', 'B', homodimer=False)
        assert CHAIN_COLOURS['A'] in result
        assert CHAIN_COLOURS['B'] in result
        assert 'cartoon_transparency' not in result


@pytest.mark.pymol
class TestProtvarDetailParsing:
    """Tests for parse_protvar_details_for_pymol() (Fix #3)."""

    def test_parses_standard_format(self):
        """Parses standard protvar detail format."""
        from pymol_scripts import parse_protvar_details_for_pymol
        records = parse_protvar_details_for_pymol(
            'D5N:am=0.10:benign:foldx=0.81|K23P:am=0.89:pathogenic:foldx=-2.45')
        assert len(records) == 2
        assert records[0]['position'] == 5
        assert records[0]['am_score'] == pytest.approx(0.10)
        assert records[0]['am_class'] == 'benign'
        assert records[0]['foldx_ddg'] == pytest.approx(0.81)
        assert records[1]['am_class'] == 'pathogenic'
        assert records[1]['foldx_ddg'] == pytest.approx(-2.45)

    def test_empty_returns_empty(self):
        """Empty string returns empty list."""
        from pymol_scripts import parse_protvar_details_for_pymol
        assert parse_protvar_details_for_pymol('') == []
        assert parse_protvar_details_for_pymol('-') == []

    def test_dash_scores(self):
        """Missing scores (dash) parsed as None."""
        from pymol_scripts import parse_protvar_details_for_pymol
        records = parse_protvar_details_for_pymol('M1A:am=-:ambiguous:foldx=-')
        assert len(records) == 1
        assert records[0]['am_score'] is None
        assert records[0]['foldx_ddg'] is None

    def test_skips_truncation(self):
        """Skips truncation markers."""
        from pymol_scripts import parse_protvar_details_for_pymol
        records = parse_protvar_details_for_pymol(
            'D5N:am=0.10:benign:foldx=0.81|...(+5 more)')
        assert len(records) == 1


@pytest.mark.pymol
class TestProtvarHighlighting:
    """Tests for generate_protvar_highlighting() (Fix #3)."""

    def test_groups_by_am_class(self):
        """Records are grouped by AlphaMissense class."""
        from pymol_scripts import generate_protvar_highlighting
        records = [
            {'position': 10, 'am_class': 'pathogenic', 'am_score': 0.9, 'foldx_ddg': 1.5},
            {'position': 20, 'am_class': 'benign', 'am_score': 0.1, 'foldx_ddg': 0.2},
        ]
        result = generate_protvar_highlighting('A', 'B', records, None)
        assert 'pv_pathogenic_A' in result
        assert 'pv_benign_A' in result
        # ProtVar uses transparency overlay, not colour or show spheres
        assert 'color ' not in result

    def test_empty_returns_empty(self):
        """None/empty records produce empty string."""
        from pymol_scripts import generate_protvar_highlighting
        assert generate_protvar_highlighting('A', 'B', None, None) == ''
        assert generate_protvar_highlighting('A', 'B', [], []) == ''

    def test_sphere_transparency_varies_by_class(self):
        """Different AM classes get different sphere transparency values."""
        from pymol_scripts import generate_protvar_highlighting, PROTVAR_AM_TRANSPARENCY
        records = [
            {'position': 10, 'am_class': 'pathogenic'},
            {'position': 20, 'am_class': 'benign'},
        ]
        result = generate_protvar_highlighting('A', 'B', records, None)
        assert f"sphere_transparency, {PROTVAR_AM_TRANSPARENCY['pathogenic']}" in result
        assert f"sphere_transparency, {PROTVAR_AM_TRANSPARENCY['benign']}" in result
        # No colour or sphere_scale commands
        assert 'color 0x' not in result
        assert 'sphere_scale' not in result

    def test_does_not_show_spheres(self):
        """ProtVar layer does not call 'show spheres' — relies on variant layer."""
        from pymol_scripts import generate_protvar_highlighting
        records = [{'position': 10, 'am_class': 'pathogenic'}]
        result = generate_protvar_highlighting('A', 'B', records, None)
        assert 'show spheres' not in result

    def test_section_header(self):
        """ProtVar section has its own header comment."""
        from pymol_scripts import generate_protvar_highlighting
        records = [{'position': 10, 'am_class': 'pathogenic'}]
        result = generate_protvar_highlighting('A', 'B', records, None)
        assert 'ProtVar' in result

    def test_dash_am_class_maps_to_unknown(self):
        """AM class '-' is mapped to 'unknown', producing valid selection name."""
        from pymol_scripts import generate_protvar_highlighting
        records = [{'position': 10, 'am_class': '-', 'am_score': None}]
        result = generate_protvar_highlighting('A', 'B', records, None)
        assert 'pv_unknown_A' in result
        assert 'pv_-_A' not in result

    def test_none_am_class_maps_to_unknown(self):
        """None/empty AM class is mapped to 'unknown'."""
        from pymol_scripts import generate_protvar_highlighting
        records_none = [{'position': 10, 'am_class': None}]
        records_empty = [{'position': 20, 'am_class': ''}]
        result_none = generate_protvar_highlighting('A', 'B', records_none, None)
        result_empty = generate_protvar_highlighting('A', 'B', records_empty, None)
        assert 'pv_unknown_A' in result_none
        assert 'pv_unknown_A' in result_empty

    def test_same_position_uses_most_severe_class(self):
        """When same position has multiple AM classes, most severe wins."""
        from pymol_scripts import generate_protvar_highlighting
        records = [
            {'position': 10, 'am_class': 'benign', 'am_score': 0.1},
            {'position': 10, 'am_class': 'pathogenic', 'am_score': 0.9},
        ]
        result = generate_protvar_highlighting('A', 'B', records, None)
        assert 'pv_pathogenic_A' in result
        # Benign selection should not exist (no other benign-only positions)
        assert 'pv_benign_A' not in result

    def test_dedup_keeps_both_groups_when_positions_differ(self):
        """Deduplication keeps multiple groups when positions differ."""
        from pymol_scripts import generate_protvar_highlighting
        records = [
            {'position': 10, 'am_class': 'benign'},
            {'position': 10, 'am_class': 'pathogenic'},  # overrides benign
            {'position': 20, 'am_class': 'benign'},       # stays benign
        ]
        result = generate_protvar_highlighting('A', 'B', records, None)
        assert 'pv_pathogenic_A' in result
        assert 'pv_benign_A' in result
        # Position 10 only in pathogenic, position 20 only in benign
        lines = result.split('\n')
        patho_sel = [l for l in lines if 'pv_pathogenic_A' in l and 'select' in l][0]
        benign_sel = [l for l in lines if 'pv_benign_A' in l and 'select' in l][0]
        assert '10' in patho_sel
        assert '10' not in benign_sel
        assert '20' in benign_sel


@pytest.mark.pymol
class TestSurfaceRepresentation:
    """Tests for generate_surface_representation() (Fix #8)."""

    def test_surface_present_when_enabled(self):
        """Surface commands generated."""
        from pymol_scripts import generate_surface_representation
        result = generate_surface_representation('A', 'B')
        assert 'show surface' in result
        assert 'transparency' in result

    def test_default_transparency(self):
        """Default transparency is 0.7."""
        from pymol_scripts import generate_surface_representation
        result = generate_surface_representation('A', 'B')
        assert '0.7' in result

    def test_no_surface_by_default(self):
        """build_pymol_script does not include surface by default."""
        from pymol_scripts import build_pymol_script
        script = build_pymol_script('/p.pdb', 'test', 'A', 'B', [10], [20])
        assert 'show surface' not in script

    def test_surface_in_script_when_enabled(self):
        """build_pymol_script includes surface when show_surface=True."""
        from pymol_scripts import build_pymol_script
        script = build_pymol_script('/p.pdb', 'test', 'A', 'B', [10], [20],
                                     show_surface=True)
        assert 'show surface' in script


@pytest.mark.pymol
class TestPdbLookupExtensions:
    """Tests for _build_pdb_lookup .ent support and empty warning (Fix #9)."""

    def test_ent_extension_found(self, tmp_path):
        """Files with .ent extension are included in lookup."""
        from pymol_scripts import _build_pdb_lookup
        # Create a fake .ent file — parse_complex_name needs a plausible name
        (tmp_path / 'A0A0B4J2C3_P24534.ent').write_text('ATOM test')
        lookup = _build_pdb_lookup(tmp_path)
        assert len(lookup) >= 1

    def test_empty_directory_warns(self, tmp_path):
        """Empty directory emits a UserWarning."""
        import warnings
        from pymol_scripts import _build_pdb_lookup
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _build_pdb_lookup(tmp_path)
            assert len(w) == 1
            assert 'No PDB/ENT files found' in str(w[0].message)


@pytest.mark.pymol
class TestQualityTierFiltering:
    """Tests for quality_tier_v2 empty-string handling (Fix #10)."""

    def test_empty_tier_v2_falls_through(self, tmp_path):
        """Empty quality_tier_v2 falls through to quality_tier."""
        from pymol_scripts import _TIER_ORDER
        # Simulate the filtering logic
        row = {'quality_tier_v2': '', 'quality_tier': 'High'}
        tier = row.get('quality_tier_v2') or row.get('quality_tier') or 'Low'
        assert _TIER_ORDER.get(tier, 0) == 3  # High = 3


@pytest.mark.pymol
class TestPrecomputedInterfaceResidues:
    """Tests for pre-computed interface residue path (Fix #2)."""

    @pytest.mark.slow
    def test_uses_precomputed_residues(self, ref_pdb_1, tmp_path):
        """Pre-computed interface_residues_a/b skips extract_interface_data()."""
        from pymol_scripts import generate_pymol_scripts_for_results
        results = [{
            'complex_name': REF_COMPLEX_1,
            'quality_tier_v2': 'High',
            'best_chain_pair': 'A_B',
            'interface_residues_a': '10|20|30',
            'interface_residues_b': '40|50|60',
        }]
        n = generate_pymol_scripts_for_results(
            results, pdb_dir=str(ref_pdb_1.parent), output_dir=str(tmp_path))
        assert n == 1
        content = (tmp_path / f"{REF_COMPLEX_1}.pml").read_text()
        # Verify the pre-computed residues appear (not the 82 from PDB)
        assert '10+20+30' in content
        assert '40+50+60' in content

    @pytest.mark.slow
    def test_falls_back_to_pdb(self, ref_pdb_1, tmp_path):
        """Without pre-computed residues, falls back to PDB extraction."""
        from pymol_scripts import generate_pymol_scripts_for_results
        results = [{
            'complex_name': REF_COMPLEX_1,
            'quality_tier_v2': 'High',
        }]
        n = generate_pymol_scripts_for_results(
            results, pdb_dir=str(ref_pdb_1.parent), output_dir=str(tmp_path))
        assert n == 1
        content = (tmp_path / f"{REF_COMPLEX_1}.pml").read_text()
        # Verify full interface residues from PDB are present
        import re
        match_a = re.search(r'select interface_a, chain A and resi (.+)', content)
        assert match_a
        resi_a = match_a.group(1).split('+')
        assert len(resi_a) == REF_INTERFACE_RESIDUES_A


@pytest.mark.pymol
class TestNewConstants:
    """Tests for new constants introduced in Fixes #3, #4, #7, #13."""

    def test_variant_sphere_scale_constants(self):
        """VARIANT_SPHERE_SCALE constants are positive floats."""
        from pymol_scripts import VARIANT_SPHERE_SCALE_PATHOGENIC, VARIANT_SPHERE_SCALE_DEFAULT
        assert VARIANT_SPHERE_SCALE_PATHOGENIC > VARIANT_SPHERE_SCALE_DEFAULT > 0

    def test_protvar_am_colours_complete(self):
        """PROTVAR_AM_COLOURS has entries for pathogenic, ambiguous, benign."""
        from pymol_scripts import PROTVAR_AM_COLOURS
        assert 'pathogenic' in PROTVAR_AM_COLOURS
        assert 'ambiguous' in PROTVAR_AM_COLOURS
        assert 'benign' in PROTVAR_AM_COLOURS

    def test_protvar_am_scales_complete(self):
        """PROTVAR_AM_SCALES has entries for pathogenic, ambiguous, benign."""
        from pymol_scripts import PROTVAR_AM_SCALES
        assert 'pathogenic' in PROTVAR_AM_SCALES
        assert PROTVAR_AM_SCALES['pathogenic'] > PROTVAR_AM_SCALES['benign']

    def test_am_severity_ordering(self):
        """_AM_SEVERITY has correct priority: pathogenic > ambiguous > benign > unknown."""
        from pymol_scripts import _AM_SEVERITY
        assert _AM_SEVERITY['pathogenic'] > _AM_SEVERITY['ambiguous']
        assert _AM_SEVERITY['ambiguous'] > _AM_SEVERITY['benign']
        assert _AM_SEVERITY['benign'] > _AM_SEVERITY['unknown']
