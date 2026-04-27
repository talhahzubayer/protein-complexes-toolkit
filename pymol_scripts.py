"""
PyMOL script generation for AlphaFold2-predicted protein-protein complexes.

Generates .pml command files that load a PDB structure and apply layered
visualisation: chain colouring, pLDDT confidence colouring, interface residue
highlighting, and (optionally) variant position colouring by structural context.

Architecture:
    - Pure string generators (no I/O) produce each visualisation layer
    - build_pymol_script() assembles layers into a complete .pml file
    - extract_interface_data() re-reads a PDB to get interface residue numbers
    - generate_pymol_scripts_for_results() is the toolkit.py entry point
    - Optional py3Dmol fallback for in-notebook rendering

Data sources:
    - PDB files from AlphaFold2 predictions (pLDDT in B-factor column)
    - Interface data via pdockq.read_pdb_with_chain_info_New / find_best_chain_pair_New
    - Variant details from variant_mapper.format_variant_details() output

Usage (standalone):
    python pymol_scripts.py generate --pdb complex.pdb
    python pymol_scripts.py generate --pdb complex.pdb --output pymol_output/ --render
    python pymol_scripts.py batch --csv results.csv --pdb-dir D:\\ProteinComplexes

Usage (via toolkit.py):
    python toolkit.py --dir DIR --output results.csv --interface --pae --pymol
    python toolkit.py --dir DIR --output results.csv --interface --pae --enrich ALIASES --variants --pymol --pymol-render
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Optional, Union

from file_io import open_text_maybe_compressed

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ── Constants ────────────────────────────────────────────────────────

DEFAULT_PYMOL_OUTPUT_DIR = "pymol_scripts"

# PyMOL colour names per chain (supports up to 10 chains)
CHAIN_COLOURS = {
    'A': 'marine',
    'B': 'salmon',
    'C': 'forest',
    'D': 'wheat',
    'E': 'slate',
    'F': 'lightorange',
    'G': 'palegreen',
    'H': 'lightpink',
    'I': 'palecyan',
    'J': 'lightteal',
}

# Canonical AlphaFold2 pLDDT colour scheme (Jumper et al. 2021)
# Each entry: (label, b_factor_condition, hex_colour)
PLDDT_COLOUR_BANDS = [
    ('plddt_vhigh', 'b > 90',             '0x0053D6'),   # Very high (blue)
    ('plddt_high',  'b > 70 and b < 91',  '0x65CBF3'),   # Confident (cyan)
    ('plddt_low',   'b > 50 and b < 71',  '0xFFDB13'),   # Low (yellow)
    ('plddt_vlow',  'b < 51',             '0xFF7D45'),   # Very low (orange)
]

# Interface residue colour
INTERFACE_COLOUR = 'hotpink'

# Variant structural context colours (matches variant_mapper.py classification)
VARIANT_CONTEXT_COLOURS = {
    'interface_core':        'red',
    'interface_rim':         'orange',
    'surface_non_interface': 'yellow',
    'buried_core':           'gray70',
}

# Default fallback colour for unknown variant context
VARIANT_DEFAULT_COLOUR = 'white'

# Sphere scale for pathogenic vs non-pathogenic variants
VARIANT_SPHERE_SCALE_PATHOGENIC = 0.6
VARIANT_SPHERE_SCALE_DEFAULT = 0.3

# Rendering defaults
DEFAULT_RAY_WIDTH = 2400
DEFAULT_RAY_HEIGHT = 1800
DEFAULT_DPI = 300

# Regex for parsing variant detail strings from variant_mapper.format_variant_details()
# Format: REF{POS}ALT:context:clinical_significance
# Example: K81P:interface_core:pathogenic
_VARIANT_DETAIL_RE = re.compile(
    r'^([A-Z*?])(\d+)([A-Z*?]):([^:]+):(.*)$'
)

# py3Dmol optional import
try:
    import py3Dmol
    _HAS_PY3DMOL = True
except ImportError:
    _HAS_PY3DMOL = False


# Mapping from PyMOL colour names to CSS hex for py3Dmol compatibility
PYMOL_TO_HEX = {
    'marine': '#0000CD',
    'salmon': '#FA8072',
    'forest': '#228B22',
    'wheat': '#F5DEB3',
    'slate': '#708090',
    'lightorange': '#FFB347',
    'palegreen': '#98FB98',
    'lightpink': '#FFB6C1',
    'palecyan': '#AFEEEE',
    'lightteal': '#20B2AA',
    'hotpink': '#FF69B4',
    'red': '#FF0000',
    'orange': '#FFA500',
    'yellow': '#FFFF00',
    'gray70': '#B3B3B3',
    'white': '#FFFFFF',
}

# Regex for parsing protvar detail strings
# Format: REF{POS}ALT:am={SCORE}:{CLASS}:foldx={DDG}
_PROTVAR_DETAIL_RE = re.compile(
    r'^([A-Z*?])(\d+)([A-Z*?]):am=([^:]*):([^:]*):foldx=(.+)$'
)

# AlphaMissense classification colours (hex for PyMOL CGO compatibility)
PROTVAR_AM_COLOURS = {
    'pathogenic': '0xFF0000',   # red
    'ambiguous':  '0xFFA500',   # orange
    'benign':     '0x0000FF',   # blue
}
PROTVAR_AM_SCALES = {
    'pathogenic': 0.8,
    'ambiguous':  0.5,
    'benign':     0.2,
}
# Transparency encoding for ProtVar AM class overlay (0.0 = fully opaque, 1.0 = invisible)
# Applied on top of variant structural context spheres to avoid overwriting colours
PROTVAR_AM_TRANSPARENCY = {
    'pathogenic': 0.0,
    'ambiguous':  0.4,
    'benign':     0.7,
}

# Severity priority for deduplicating multiple AM classes at the same position
_AM_SEVERITY = {
    'pathogenic': 3,
    'ambiguous':  2,
    'benign':     1,
    'unknown':    0,
}


# ── PML Command Generators ──────────────────────────────────────────

def generate_pml_header(pdb_path: str, complex_name: str) -> str:
    """Generate the header section of a .pml script.

    Loads the PDB, sets basic display and background. PyMOL's native ``load``
    command does not transparently decompress ``.bz2``; for ``.pdb.bz2``
    inputs the header emits an inline Python block that decompresses with
    ``bz2.open`` and feeds the structure via ``cmd.read_pdbstr``.

    Args:
        pdb_path: Absolute or relative path to the PDB file.
        complex_name: Identifier used as the PyMOL object name.

    Returns:
        Multi-line string with load, display, and background commands.
    """
    # Normalise path separators to forward slashes (PyMOL convention)
    pdb_path_clean = str(pdb_path).replace('\\', '/')

    if pdb_path_clean.endswith('.bz2'):
        # repr() defends against pathological characters (literal quotes,
        # backslashes) — overkill for AF2 paths but trivial cost.
        load_block = (
            f"python\n"
            f"import bz2\n"
            f"from pymol import cmd\n"
            f"with bz2.open({repr(pdb_path_clean)}, 'rt') as _h:\n"
            f"    cmd.read_pdbstr(_h.read(), {repr(complex_name)})\n"
            f"python end\n"
        )
    else:
        load_block = f"load {pdb_path_clean}, {complex_name}\n"

    return (
        f"# === PyMOL Script for {complex_name} ===\n"
        f"# Generated by protein-complexes-toolkit\n"
        f"\n"
        f"# -- Load structure --\n"
        f"{load_block}"
        f"hide everything\n"
        f"show cartoon\n"
        f"bg_color white\n"
    )


def generate_metadata_comments(metadata: dict) -> str:
    """Generate a metadata comment block for the script header.

    Args:
        metadata: Dict with optional keys: quality_tier_v2, composite_score,
            iptm, pdockq, n_pathogenic_interface_variants, is_drug_target_a,
            is_drug_target_b, gene_symbol_a, gene_symbol_b.

    Returns:
        Multi-line comment string, or empty string if metadata is empty/None.
    """
    if not metadata:
        return ''

    lines = ["\n# === Metadata ==="]

    # Quality line
    parts = []
    if metadata.get('quality_tier_v2'):
        parts.append(f"Quality: {metadata['quality_tier_v2']}")
    if metadata.get('composite_score') not in (None, '', '-'):
        parts.append(f"Composite: {metadata['composite_score']}")
    if metadata.get('iptm') not in (None, '', '-'):
        parts.append(f"ipTM: {metadata['iptm']}")
    if metadata.get('pdockq') not in (None, '', '-'):
        parts.append(f"pDockQ: {metadata['pdockq']}")
    if parts:
        lines.append(f"# {' | '.join(parts)}")

    # Drug target
    dt_a = metadata.get('is_drug_target_a', '')
    dt_b = metadata.get('is_drug_target_b', '')
    if dt_a or dt_b:
        lines.append(f"# Drug target: A={'Yes' if dt_a else 'No'}, "
                      f"B={'Yes' if dt_b else 'No'}")

    # Pathogenic interface variants
    n_path = metadata.get('n_pathogenic_interface_variants', '')
    if n_path not in (None, '', '-', '0', 0):
        lines.append(f"# Pathogenic interface variants: {n_path}")

    return '\n'.join(lines) + '\n'


def generate_annotation_comments(
    gene_a: str = '',
    gene_b: str = '',
    disease_details_a: str = '',
    disease_details_b: str = '',
    is_drug_target_a: str = '',
    is_drug_target_b: str = '',
    reactome_pathways_a: str = '',
    reactome_pathways_b: str = '',
) -> str:
    """Generate biological context annotations as comments and labels.

    Args:
        gene_a: Gene symbol for chain A.
        gene_b: Gene symbol for chain B.
        disease_details_a: Pipe-separated disease strings for chain A.
        disease_details_b: Pipe-separated disease strings for chain B.
        is_drug_target_a: Drug target flag for chain A.
        is_drug_target_b: Drug target flag for chain B.
        reactome_pathways_a: Pipe-separated pathway strings for chain A.
        reactome_pathways_b: Pipe-separated pathway strings for chain B.

    Returns:
        Multi-line string with comments and PyMOL label commands.
        Empty string if no annotation data is available.
    """
    has_data = any([gene_a, gene_b, disease_details_a, disease_details_b,
                    is_drug_target_a, is_drug_target_b,
                    reactome_pathways_a, reactome_pathways_b])
    if not has_data:
        return ''

    lines = ["\n# === Biological Context ==="]

    if gene_a or gene_b:
        lines.append(f"# Gene A: {gene_a or '-'} | Gene B: {gene_b or '-'}")

    # Diseases (first 5 per chain)
    for label, details in [('A', disease_details_a), ('B', disease_details_b)]:
        if details and details != '-':
            details = details.replace('\n', ' ').replace('\r', ' ')
            diseases = [d.strip() for d in details.split('|') if d.strip() and not d.startswith('...')]
            if diseases:
                shown = diseases[:5]
                suffix = f' (+{len(diseases) - 5} more)' if len(diseases) > 5 else ''
                lines.append(f"# Diseases ({label}): {' | '.join(shown)}{suffix}")

    # Drug target
    if is_drug_target_a or is_drug_target_b:
        lines.append(f"# Drug target: A={'Yes' if is_drug_target_a else 'No'}, "
                      f"B={'Yes' if is_drug_target_b else 'No'}")

    # Pathways (first 5 per chain)
    for label, pathways in [('A', reactome_pathways_a), ('B', reactome_pathways_b)]:
        if pathways and pathways != '-':
            pathways = pathways.replace('\n', ' ').replace('\r', ' ')
            pw_list = [p.strip() for p in pathways.split('|') if p.strip() and not p.startswith('...')]
            if pw_list:
                shown = pw_list[:5]
                suffix = f' (+{len(pw_list) - 5} more)' if len(pw_list) > 5 else ''
                lines.append(f"# Pathways ({label}): {' | '.join(shown)}{suffix}")

    # Gene symbol labels on the structure
    if gene_a:
        lines.append(f'label chain A and name CA and resi 1, "{gene_a}"')
    if gene_b:
        lines.append(f'label chain B and name CA and resi 1, "{gene_b}"')

    return '\n'.join(lines) + '\n'


def generate_chain_colouring(chain_a: str, chain_b: str, *, homodimer: bool = False) -> str:
    """Generate chain colouring commands.

    Args:
        chain_a: First chain identifier.
        chain_b: Second chain identifier.
        homodimer: If True, both chains use the same colour with chain B
            at 30% transparency to distinguish subunits.

    Returns:
        Multi-line string with chain colour commands.
    """
    colour_a = CHAIN_COLOURS.get(chain_a, 'marine')
    colour_b = colour_a if homodimer else CHAIN_COLOURS.get(chain_b, 'salmon')
    lines = [
        f"\n# -- Chain colouring --",
        f"color {colour_a}, chain {chain_a}",
        f"color {colour_b}, chain {chain_b}",
    ]
    if homodimer:
        lines.append(f"set cartoon_transparency, 0.3, chain {chain_b}")
        lines.append(f"# Homodimer: chain {chain_b} shown at 30% transparency")
    lines.append("scene chain_view, store")
    return '\n'.join(lines) + '\n'


def generate_plddt_colouring() -> str:
    """Generate pLDDT confidence colouring commands.

    Uses the canonical AlphaFold2 4-band colour scheme based on B-factor
    values (which store pLDDT in AlphaFold PDB files).

    Returns:
        Multi-line string with selection and colour commands.
    """
    lines = ["\n# -- pLDDT colouring (B-factor) --"]
    for label, condition, hex_colour in PLDDT_COLOUR_BANDS:
        lines.append(f"select {label}, {condition}")
        lines.append(f"color {hex_colour}, {label}")
    lines.append("deselect")
    lines.append("scene plddt_view, store")
    return '\n'.join(lines) + '\n'


def generate_interface_highlighting(
    chain_a: str,
    chain_b: str,
    interface_resi_a: list[int],
    interface_resi_b: list[int],
) -> str:
    """Generate interface residue highlighting commands.

    Shows interface residues as sticks coloured in hotpink.

    Args:
        chain_a: First chain identifier.
        chain_b: Second chain identifier.
        interface_resi_a: PDB residue numbers at the interface on chain A.
        interface_resi_b: PDB residue numbers at the interface on chain B.

    Returns:
        Multi-line string with selection, representation, and colour commands.
        Empty string if both lists are empty.
    """
    if not interface_resi_a and not interface_resi_b:
        return ''

    lines = ["\n# -- Interface residues --"]

    if interface_resi_a:
        resi_str = '+'.join(str(r) for r in sorted(interface_resi_a))
        lines.append(f"select interface_a, chain {chain_a} and resi {resi_str}")

    if interface_resi_b:
        resi_str = '+'.join(str(r) for r in sorted(interface_resi_b))
        lines.append(f"select interface_b, chain {chain_b} and resi {resi_str}")

    # Show and colour
    sel_parts = []
    if interface_resi_a:
        sel_parts.append('interface_a')
    if interface_resi_b:
        sel_parts.append('interface_b')
    sel_combined = ' or '.join(sel_parts)

    lines.append(f"show sticks, {sel_combined}")
    lines.append(f"color {INTERFACE_COLOUR}, {sel_combined}")
    lines.append("deselect")

    return '\n'.join(lines) + '\n'


def _is_pathogenic(clinical: str) -> bool:
    """Check whether a clinical significance string indicates pathogenicity."""
    if not clinical:
        return False
    cl = clinical.lower()
    return 'pathogenic' in cl and 'benign' not in cl


def generate_variant_highlighting(
    chain_a: str,
    chain_b: str,
    variant_records_a: Optional[list[dict]] = None,
    variant_records_b: Optional[list[dict]] = None,
) -> str:
    """Generate variant position highlighting commands.

    Shows variant positions as spheres coloured by structural context, with
    pathogenic variants rendered at a larger sphere scale than benign/VUS.

    Selection naming convention: ``var_{context}[_path]_{chain}`` — kept
    under 40 characters (PyMOL practical limit ~63).

    Args:
        chain_a: First chain identifier.
        chain_b: Second chain identifier.
        variant_records_a: Parsed variant dicts for chain A
            (keys: position, context, clinical). None to skip.
        variant_records_b: Parsed variant dicts for chain B. None to skip.

    Returns:
        Multi-line string with selection, representation, and colour commands.
        Empty string if no variants provided.
    """
    all_variants = []
    if variant_records_a:
        for v in variant_records_a:
            all_variants.append((chain_a, v))
    if variant_records_b:
        for v in variant_records_b:
            all_variants.append((chain_b, v))

    if not all_variants:
        return ''

    # Group variants by (chain, context, is_pathogenic) for efficient selection
    groups: dict[tuple[str, str, bool], list[int]] = {}
    for chain, var in all_variants:
        ctx = var.get('context', 'unknown')
        pathogenic = _is_pathogenic(var.get('clinical', ''))
        key = (chain, ctx, pathogenic)
        if key not in groups:
            groups[key] = []
        pos = var.get('position')
        if pos is not None:
            groups[key].append(int(pos))

    if not groups:
        return ''

    lines = ["\n# -- Variant positions --"]
    sel_names = []

    for (chain, ctx, pathogenic), positions in sorted(groups.items()):
        if not positions:
            continue
        safe_ctx = ctx.replace(' ', '_')
        path_tag = '_path' if pathogenic else ''
        sel_name = f"var_{safe_ctx}{path_tag}_{chain}"
        resi_str = '+'.join(str(r) for r in sorted(set(positions)))
        colour = VARIANT_CONTEXT_COLOURS.get(ctx, VARIANT_DEFAULT_COLOUR)
        scale = VARIANT_SPHERE_SCALE_PATHOGENIC if pathogenic else VARIANT_SPHERE_SCALE_DEFAULT

        lines.append(f"select {sel_name}, chain {chain} and resi {resi_str}")
        lines.append(f"show spheres, {sel_name}")
        lines.append(f"color {colour}, {sel_name}")
        lines.append(f"set sphere_scale, {scale}, {sel_name}")
        sel_names.append(sel_name)

    if sel_names:
        lines.append("deselect")

    return '\n'.join(lines) + '\n'


def generate_protvar_highlighting(
    chain_a: str,
    chain_b: str,
    protvar_records_a: Optional[list[dict]] = None,
    protvar_records_b: Optional[list[dict]] = None,
) -> str:
    """Generate ProtVar/AlphaMissense highlighting commands.

    Encodes AM pathogenicity class as sphere transparency on residues that
    already have spheres from the variant highlighting layer.  This avoids
    overwriting the structural-context colours set by
    ``generate_variant_highlighting``.

    Transparency mapping (PROTVAR_AM_TRANSPARENCY):
      pathogenic → 0.0 (fully opaque), ambiguous → 0.4, benign → 0.7.

    Args:
        chain_a: First chain identifier.
        chain_b: Second chain identifier.
        protvar_records_a: Parsed protvar dicts for chain A
            (keys: position, am_class, am_score, foldx_ddg). None to skip.
        protvar_records_b: Parsed protvar dicts for chain B. None to skip.

    Returns:
        Multi-line string with selection and transparency commands.
        Empty string if no records provided.
    """
    all_records = []
    if protvar_records_a:
        for v in protvar_records_a:
            all_records.append((chain_a, v))
    if protvar_records_b:
        for v in protvar_records_b:
            all_records.append((chain_b, v))

    if not all_records:
        return ''

    # First pass: find most severe AM class per (chain, position)
    # so each residue appears in exactly one AM selection (fixes C/D).
    pos_best: dict[tuple[str, int], str] = {}
    for chain, rec in all_records:
        am_class = rec.get('am_class') or 'unknown'
        if am_class == '-':
            am_class = 'unknown'
        pos = rec.get('position')
        if pos is None:
            continue
        pos = int(pos)
        key = (chain, pos)
        if key not in pos_best or _AM_SEVERITY.get(am_class, 0) > _AM_SEVERITY.get(pos_best[key], 0):
            pos_best[key] = am_class

    # Second pass: group by (chain, am_class)
    groups: dict[tuple[str, str], list[int]] = {}
    for (chain, pos), am_class in pos_best.items():
        groups.setdefault((chain, am_class), []).append(pos)

    if not groups:
        return ''

    lines = ["\n# -- ProtVar / AlphaMissense predictions (transparency overlay) --"]
    sel_names = []

    for (chain, am_class), positions in sorted(groups.items()):
        if not positions:
            continue
        safe_class = am_class.replace(' ', '_')
        sel_name = f"pv_{safe_class}_{chain}"
        resi_str = '+'.join(str(r) for r in sorted(set(positions)))
        transparency = PROTVAR_AM_TRANSPARENCY.get(am_class, 0.4)

        lines.append(f"select {sel_name}, chain {chain} and resi {resi_str}")
        lines.append(f"set sphere_transparency, {transparency}, {sel_name}")
        sel_names.append(sel_name)

    if sel_names:
        lines.append("deselect")

    return '\n'.join(lines) + '\n'


def generate_surface_representation(
    chain_a: str,
    chain_b: str,
    transparency: float = 0.7,
) -> str:
    """Generate a semi-transparent surface overlay.

    Args:
        chain_a: First chain identifier.
        chain_b: Second chain identifier.
        transparency: Surface transparency (0.0 opaque, 1.0 invisible).

    Returns:
        Multi-line string with surface and transparency commands.
    """
    return (
        f"\n# -- Surface representation --\n"
        f"show surface, chain {chain_a} or chain {chain_b}\n"
        f"set transparency, {transparency}\n"
    )


def generate_rendering_commands(
    output_png_path: Optional[str] = None,
    render: bool = False,
) -> str:
    """Generate camera orientation and optional rendering commands.

    Args:
        output_png_path: Path for PNG output file. Used only when render=True.
        render: If True, include ray-tracing and PNG save commands (uncommented).
            If False, include them as comments for manual use.

    Returns:
        Multi-line string with camera and rendering commands.
    """
    lines = [
        "\n# -- Camera and rendering --",
        "orient",
        "zoom complete=1",
        "scene full_view, store",
        "set ray_opaque_background, 1",
    ]

    prefix = '' if render else '# '
    lines.append(f"{prefix}ray {DEFAULT_RAY_WIDTH}, {DEFAULT_RAY_HEIGHT}")

    if output_png_path:
        png_path = str(output_png_path).replace('\\', '/')
        lines.append(f"{prefix}png {png_path}, dpi={DEFAULT_DPI}")
    else:
        lines.append(f"{prefix}png output.png, dpi={DEFAULT_DPI}")

    if render:
        lines.append("quit")

    return '\n'.join(lines) + '\n'


# ── Script Assembly ──────────────────────────────────────────────────

def build_pymol_script(
    pdb_path: str,
    complex_name: str,
    chain_a: str,
    chain_b: str,
    interface_resi_a: list[int],
    interface_resi_b: list[int],
    variant_records_a: Optional[list[dict]] = None,
    variant_records_b: Optional[list[dict]] = None,
    protvar_records_a: Optional[list[dict]] = None,
    protvar_records_b: Optional[list[dict]] = None,
    render_png: bool = False,
    output_png_path: Optional[str] = None,
    metadata: Optional[dict] = None,
    annotation: Optional[dict] = None,
    homodimer: bool = False,
    show_surface: bool = False,
) -> str:
    """Assemble a complete PyMOL .pml script from individual layers.

    Section order: header, metadata, annotation, chain colouring (scene),
    pLDDT colouring (scene), interface highlighting, [surface], variant
    highlighting, protvar highlighting, rendering (scene full_view).

    Args:
        pdb_path: Path to the PDB file.
        complex_name: Complex identifier (used as PyMOL object name).
        chain_a: First chain identifier.
        chain_b: Second chain identifier.
        interface_resi_a: PDB residue numbers at interface on chain A.
        interface_resi_b: PDB residue numbers at interface on chain B.
        variant_records_a: Parsed variant dicts for chain A. None to skip.
        variant_records_b: Parsed variant dicts for chain B. None to skip.
        protvar_records_a: Parsed protvar dicts for chain A. None to skip.
        protvar_records_b: Parsed protvar dicts for chain B. None to skip.
        render_png: If True, uncomment ray/png commands for batch rendering.
        output_png_path: Custom PNG output path. None uses default.
        metadata: Dict with quality/score info for comment header.
        annotation: Dict with disease/pathway/gene info for comments.
        homodimer: If True, use same colour for both chains.
        show_surface: If True, add semi-transparent surface overlay.

    Returns:
        Complete .pml script as a string.
    """
    sections = [
        generate_pml_header(pdb_path, complex_name),
        generate_metadata_comments(metadata),
        generate_annotation_comments(**annotation) if annotation else '',
        generate_chain_colouring(chain_a, chain_b, homodimer=homodimer),
        generate_plddt_colouring(),
        generate_interface_highlighting(chain_a, chain_b,
                                        interface_resi_a, interface_resi_b),
    ]
    if show_surface:
        sections.append(generate_surface_representation(chain_a, chain_b))
    sections.extend([
        generate_variant_highlighting(chain_a, chain_b,
                                      variant_records_a, variant_records_b),
        generate_protvar_highlighting(chain_a, chain_b,
                                      protvar_records_a, protvar_records_b),
        generate_rendering_commands(output_png_path, render=render_png),
    ])
    return ''.join(sections)


# ── Variant Detail Parsing ───────────────────────────────────────────

def parse_variant_details_for_pymol(details_str: str) -> list[dict]:
    """Parse a pipe-separated variant details string into dicts for PyMOL.

    Extracts position and structural context from the format produced by
    variant_mapper.format_variant_details():
        REF{POS}ALT:context:clinical_significance

    Args:
        details_str: Pipe-separated variant string, e.g.
            'K81P:interface_core:pathogenic|E82K:interface_rim:VUS'.

    Returns:
        List of dicts with keys: ref, position (int), alt, context, clinical.
        Malformed entries are silently skipped.
    """
    if not details_str or details_str == '-':
        return []

    records = []
    for part in details_str.split('|'):
        part = part.strip()
        if not part or part.startswith('...'):
            continue

        match = _VARIANT_DETAIL_RE.match(part)
        if match:
            records.append({
                'ref': match.group(1),
                'position': int(match.group(2)),
                'alt': match.group(3),
                'context': match.group(4),
                'clinical': match.group(5),
            })

    return records


def parse_protvar_details_for_pymol(details_str: str) -> list[dict]:
    """Parse a pipe-separated ProtVar details string into dicts for PyMOL.

    Extracts position and AlphaMissense/FoldX data from the format produced by
    protvar_client.format_protvar_details():
        REF{POS}ALT:am={SCORE}:{CLASS}:foldx={DDG}

    Args:
        details_str: Pipe-separated protvar string, e.g.
            'D5N:am=0.10:benign:foldx=0.81|K23P:am=0.89:pathogenic:foldx=-2.45'.

    Returns:
        List of dicts with keys: ref, position (int), alt, am_score (float|None),
        am_class (str), foldx_ddg (float|None).
        Malformed entries are silently skipped.
    """
    if not details_str or details_str == '-':
        return []

    records = []
    for part in details_str.split('|'):
        part = part.strip()
        if not part or part.startswith('...'):
            continue

        match = _PROTVAR_DETAIL_RE.match(part)
        if match:
            # Parse scores, treating '-' as None
            am_raw = match.group(4)
            foldx_raw = match.group(6)
            try:
                am_score = float(am_raw) if am_raw and am_raw != '-' else None
            except ValueError:
                am_score = None
            try:
                foldx_ddg = float(foldx_raw) if foldx_raw and foldx_raw != '-' else None
            except ValueError:
                foldx_ddg = None

            records.append({
                'ref': match.group(1),
                'position': int(match.group(2)),
                'alt': match.group(3),
                'am_score': am_score,
                'am_class': match.group(5),
                'foldx_ddg': foldx_ddg,
            })

    return records


# ── Interface Residue Extraction ─────────────────────────────────────

def extract_interface_data(pdb_path: Union[str, Path]) -> dict:
    """Re-read a PDB file and extract interface residue PDB numbers.

    Uses pdockq imports to load chain info and find the best interacting
    chain pair, then converts interface residue indices to PDB residue numbers.

    Args:
        pdb_path: Path to an AlphaFold2 PDB file.

    Returns:
        Dict with keys: chain_a, chain_b, interface_resi_a (list[int]),
        interface_resi_b (list[int]).

    Raises:
        FileNotFoundError: If PDB file does not exist.
        RuntimeError: If pdockq import or chain pair detection fails.
    """
    from pdockq import (
        read_pdb_with_chain_info_New as read_pdb_with_chain_info,
        find_best_chain_pair_New as find_best_chain_pair,
    )

    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    chain_info = read_pdb_with_chain_info(str(pdb_path))
    ch_a, ch_b, contact = find_best_chain_pair(chain_info, t=8)

    interface_resi_a = [
        chain_info.chain_res_numbers[ch_a][i]
        for i in sorted(contact.interface_residues_a)
    ]
    interface_resi_b = [
        chain_info.chain_res_numbers[ch_b][i]
        for i in sorted(contact.interface_residues_b)
    ]

    return {
        'chain_a': ch_a,
        'chain_b': ch_b,
        'interface_resi_a': interface_resi_a,
        'interface_resi_b': interface_resi_b,
    }


# ── PDB File Discovery ────────────────────────────────────────────

def _build_pdb_lookup(pdb_dir: Path) -> dict[str, Path]:
    """Map cleaned complex names → PDB paths across all supported layouts.

    Delegates to ``toolkit.find_paired_data_files()`` which already
    dispatches between the three layouts the toolkit supports — loose
    (``Test_Data/X_Y.pdb``), flat-dir (``X_Y/X_Y.pdb``), and sharded
    (``A0/X_Y/X_Y.pdb.bz2``) — and handles ``.pdb.bz2`` transparently.
    A small ``.ent`` backstop preserves legacy-flat support since
    ``complex_resolver.py`` only recognises ``.pdb``/``.pdb.bz2``.

    Args:
        pdb_dir: Directory containing PDB files (any supported layout).

    Returns:
        Dict mapping cleaned complex names to their PDB file paths.
    """
    from toolkit import find_paired_data_files, parse_complex_name

    lookup: dict[str, Path] = {}
    pairs = find_paired_data_files(str(pdb_dir))
    for name, paths in pairs.items():
        if 'pdb' in paths:
            lookup[name] = paths['pdb']

    # .ent backstop for the legacy flat-loose case (resolver only sees .pdb).
    try:
        for f in pdb_dir.iterdir():
            if f.is_file() and f.suffix.lower() == '.ent':
                clean_name, _, _, _ = parse_complex_name(f.name)
                lookup.setdefault(clean_name, f)
    except (OSError, FileNotFoundError):
        pass

    if not lookup:
        import warnings
        warnings.warn(f"No PDB/ENT files found in {pdb_dir}")
    return lookup


# ── Main Generation Function ────────────────────────────────────────

# Quality tier ordering for min_tier filtering
_TIER_ORDER = {'High': 3, 'Medium': 2, 'Low': 1}


def generate_pymol_scripts_for_results(
    results: list[dict],
    pdb_dir: str,
    output_dir: Optional[str] = None,
    min_tier: str = 'High',
    include_variants: bool = True,
    render_png: bool = False,
    verbose: bool = False,
) -> int:
    """Generate PyMOL .pml scripts for qualifying complexes.

    Main entry point from toolkit.py. Iterates results, filters by quality
    tier, re-reads PDBs for interface data, and writes one .pml file per
    qualifying complex.

    Args:
        results: List of per-complex result dicts from the pipeline.
        pdb_dir: Directory containing PDB files.
        output_dir: Output directory for .pml files. None uses
            ./pymol_scripts/ in the current working directory.
        min_tier: Minimum quality_tier_v2 for script generation.
        include_variants: If True, parse variant_details and add variant
            highlighting to scripts.
        render_png: If True, uncomment ray/png commands in generated scripts.
        verbose: Print progress to stderr.

    Returns:
        Number of .pml scripts generated.
    """
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), DEFAULT_PYMOL_OUTPUT_DIR)

    os.makedirs(output_dir, exist_ok=True)

    # Build PDB lookup using toolkit's name-cleaning logic (scan once)
    pdb_lookup = _build_pdb_lookup(Path(pdb_dir))
    if verbose:
        print(f"  [PyMOL] Found {len(pdb_lookup)} PDB files in {pdb_dir}",
              file=sys.stderr)

    min_tier_val = _TIER_ORDER.get(min_tier, 3)

    # Pre-count qualifying complexes for accurate progress bar total
    qualifying = [
        row for row in results
        if _TIER_ORDER.get(
            row.get('quality_tier_v2') or row.get('quality_tier') or 'Low', 0
        ) >= min_tier_val
    ]

    n_generated = 0
    n_skipped = 0

    # Set up progress bar (tqdm if available, print fallback otherwise)
    use_bar = tqdm is not None and verbose
    if use_bar:
        pbar = tqdm(
            total=len(qualifying), desc="PyMOL scripts", unit="script",
            file=sys.stderr, ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                       "[{elapsed}<{remaining}, {rate_fmt}]",
        )
    elif verbose:
        print(f"  [PyMOL] Generating scripts for {len(qualifying)} "
              f"qualifying complexes...", file=sys.stderr)

    for row in qualifying:
        complex_name = row.get('complex_name', '')

        # Locate PDB file via pre-built lookup
        pdb_path = pdb_lookup.get(complex_name)
        if pdb_path is None:
            if verbose and not use_bar:
                print(f"  [PyMOL] PDB not found for {complex_name}, skipping",
                      file=sys.stderr)
            n_skipped += 1
            if use_bar:
                pbar.update(1)
            continue

        # Use pre-computed interface residues if available (from toolkit pipeline or CSV)
        iface_str_a = row.get('interface_residues_a', '')
        iface_str_b = row.get('interface_residues_b', '')
        best_pair = row.get('best_chain_pair', '')

        if (iface_str_a or iface_str_b) and best_pair:
            if '_' in best_pair:
                _ch_a, _ch_b = best_pair.split('_', 1)
            else:
                _ch_a, _ch_b = 'A', 'B'
            iface = {
                'chain_a': _ch_a,
                'chain_b': _ch_b,
                'interface_resi_a': [int(x) for x in iface_str_a.split('|') if x],
                'interface_resi_b': [int(x) for x in iface_str_b.split('|') if x],
            }
        else:
            # Fallback: re-read PDB (standalone CLI or legacy CSV without interface_residues)
            try:
                iface = extract_interface_data(pdb_path)
            except Exception as exc:
                if verbose and not use_bar:
                    print(f"  [PyMOL] Error extracting interface for "
                          f"{complex_name}: {exc}", file=sys.stderr)
                n_skipped += 1
                if use_bar:
                    pbar.update(1)
                continue

        # Parse variant details if available
        var_a = None
        var_b = None
        if include_variants:
            details_a = row.get('variant_details_a', '')
            details_b = row.get('variant_details_b', '')
            if details_a:
                var_a = parse_variant_details_for_pymol(details_a)
            if details_b:
                var_b = parse_variant_details_for_pymol(details_b)

        # Parse protvar details if available
        pv_a = None
        pv_b = None
        if include_variants:
            pv_details_a = row.get('protvar_details_a', '')
            pv_details_b = row.get('protvar_details_b', '')
            if pv_details_a:
                pv_a = parse_protvar_details_for_pymol(pv_details_a)
            if pv_details_b:
                pv_b = parse_protvar_details_for_pymol(pv_details_b)

        # Build metadata dict from row
        metadata = {
            'quality_tier_v2': row.get('quality_tier_v2', ''),
            'composite_score': row.get('interface_confidence_score', ''),
            'iptm': row.get('iptm', ''),
            'pdockq': row.get('pdockq', ''),
            'n_pathogenic_interface_variants': row.get('n_pathogenic_interface_variants', ''),
            'is_drug_target_a': row.get('is_drug_target_a', ''),
            'is_drug_target_b': row.get('is_drug_target_b', ''),
        }

        # Build annotation dict from row
        annotation = {
            'gene_a': row.get('gene_symbol_a', ''),
            'gene_b': row.get('gene_symbol_b', ''),
            'disease_details_a': row.get('disease_details_a', ''),
            'disease_details_b': row.get('disease_details_b', ''),
            'is_drug_target_a': row.get('is_drug_target_a', ''),
            'is_drug_target_b': row.get('is_drug_target_b', ''),
            'reactome_pathways_a': row.get('reactome_pathways_a', ''),
            'reactome_pathways_b': row.get('reactome_pathways_b', ''),
        }

        # Detect homodimer
        is_homodimer = (row.get('protein_a', '') == row.get('protein_b', '')
                        and row.get('protein_a', '') != '')

        # Build PNG output path if rendering
        png_path = None
        if render_png:
            png_path = str(Path(output_dir) / f"{complex_name}.png")

        # Generate script
        script = build_pymol_script(
            pdb_path=str(pdb_path.resolve()),
            complex_name=complex_name,
            chain_a=iface['chain_a'],
            chain_b=iface['chain_b'],
            interface_resi_a=iface['interface_resi_a'],
            interface_resi_b=iface['interface_resi_b'],
            variant_records_a=var_a,
            variant_records_b=var_b,
            protvar_records_a=pv_a,
            protvar_records_b=pv_b,
            render_png=render_png,
            output_png_path=png_path,
            metadata=metadata,
            annotation=annotation,
            homodimer=is_homodimer,
        )

        # Write .pml file
        pml_path = Path(output_dir) / f"{complex_name}.pml"
        with open(pml_path, 'w', encoding='utf-8') as f:
            f.write(script)

        n_generated += 1

        if use_bar:
            pbar.update(1)
        elif verbose and n_generated % 100 == 0:
            print(f"  [PyMOL] {n_generated} scripts generated...",
                  file=sys.stderr)

    if use_bar:
        pbar.close()

    if verbose:
        print(f"  [PyMOL] Complete: {n_generated} scripts generated, "
              f"{n_skipped} skipped", file=sys.stderr)

    return n_generated


# ── py3Dmol Fallback ─────────────────────────────────────────────────

def generate_py3dmol_view(
    pdb_path: Union[str, Path],
    chain_a: str,
    chain_b: str,
    interface_resi_a: list[int],
    interface_resi_b: list[int],
    variant_records_a: Optional[list[dict]] = None,
    variant_records_b: Optional[list[dict]] = None,
    width: int = 800,
    height: int = 600,
):
    """Generate a py3Dmol interactive view for Jupyter notebook rendering.

    Mirrors the PyMOL script layers (chain colouring, interface highlighting,
    variant colouring) in an interactive 3D viewer.

    Args:
        pdb_path: Path to the PDB file.
        chain_a: First chain identifier.
        chain_b: Second chain identifier.
        interface_resi_a: PDB residue numbers at interface on chain A.
        interface_resi_b: PDB residue numbers at interface on chain B.
        variant_records_a: Parsed variant dicts for chain A. None to skip.
        variant_records_b: Parsed variant dicts for chain B. None to skip.
        width: Viewer width in pixels.
        height: Viewer height in pixels.

    Returns:
        py3Dmol.view object (call .show() to render), or None if py3Dmol
        is not installed.
    """
    if not _HAS_PY3DMOL:
        return None

    pdb_path = Path(pdb_path)
    with open_text_maybe_compressed(pdb_path) as f:
        pdb_string = f.read()

    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_string, 'pdb')

    # Chain colouring (cartoon) — use hex for py3Dmol compatibility
    colour_a = PYMOL_TO_HEX.get(CHAIN_COLOURS.get(chain_a, 'marine'), '#0000CD')
    colour_b = PYMOL_TO_HEX.get(CHAIN_COLOURS.get(chain_b, 'salmon'), '#FA8072')
    view.setStyle({'chain': chain_a}, {'cartoon': {'color': colour_a}})
    view.setStyle({'chain': chain_b}, {'cartoon': {'color': colour_b}})

    # Interface highlighting (sticks)
    iface_hex = PYMOL_TO_HEX.get(INTERFACE_COLOUR, '#FF69B4')
    if interface_resi_a:
        view.addStyle(
            {'chain': chain_a, 'resi': interface_resi_a},
            {'stick': {'color': iface_hex}},
        )
    if interface_resi_b:
        view.addStyle(
            {'chain': chain_b, 'resi': interface_resi_b},
            {'stick': {'color': iface_hex}},
        )

    # Variant highlighting (spheres by context)
    for chain, records in [(chain_a, variant_records_a),
                           (chain_b, variant_records_b)]:
        if not records:
            continue
        for var in records:
            pos = var.get('position')
            ctx = var.get('context', 'unknown')
            pymol_colour = VARIANT_CONTEXT_COLOURS.get(ctx, VARIANT_DEFAULT_COLOUR)
            hex_colour = PYMOL_TO_HEX.get(pymol_colour, pymol_colour)
            if pos is not None:
                view.addStyle(
                    {'chain': chain, 'resi': int(pos)},
                    {'sphere': {'color': hex_colour, 'radius': 0.8}},
                )

    view.zoomTo()
    return view


# ── Standalone CLI ───────────────────────────────────────────────────

def _cli_generate(args) -> None:
    """Handle the 'generate' subcommand: single PDB -> .pml."""
    pdb_path = Path(args.pdb)
    if not pdb_path.exists():
        print(f"Error: PDB file not found: {pdb_path}", file=sys.stderr)
        sys.exit(1)

    complex_name = args.name or pdb_path.stem
    output_dir = args.output or '.'
    os.makedirs(output_dir, exist_ok=True)

    # Extract interface data
    print(f"Reading PDB: {pdb_path}", file=sys.stderr)
    iface = extract_interface_data(pdb_path)
    print(f"  Chain pair: {iface['chain_a']}-{iface['chain_b']}", file=sys.stderr)
    print(f"  Interface residues: {len(iface['interface_resi_a'])} (A) + "
          f"{len(iface['interface_resi_b'])} (B)", file=sys.stderr)

    png_path = None
    if args.render:
        png_path = str(Path(output_dir) / f"{complex_name}.png")

    script = build_pymol_script(
        pdb_path=str(pdb_path.resolve()),
        complex_name=complex_name,
        chain_a=iface['chain_a'],
        chain_b=iface['chain_b'],
        interface_resi_a=iface['interface_resi_a'],
        interface_resi_b=iface['interface_resi_b'],
        render_png=args.render,
        output_png_path=png_path,
    )

    pml_path = Path(output_dir) / f"{complex_name}.pml"
    with open(pml_path, 'w', encoding='utf-8') as f:
        f.write(script)

    print(f"Script written: {pml_path}", file=sys.stderr)
    if args.render:
        print(f"  Render with: pymol -c {pml_path}", file=sys.stderr)


def _cli_batch(args) -> None:
    """Handle the 'batch' subcommand: CSV + PDB dir -> .pml scripts."""
    csv_path = Path(args.csv)
    pdb_dir = Path(args.pdb_dir)

    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)
    if not pdb_dir.is_dir():
        print(f"Error: PDB directory not found: {pdb_dir}", file=sys.stderr)
        sys.exit(1)

    # Read results from CSV
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        results = list(reader)

    print(f"Loaded {len(results)} rows from {csv_path}", file=sys.stderr)

    output_dir = args.output or os.path.join(os.getcwd(), DEFAULT_PYMOL_OUTPUT_DIR)

    n_generated = generate_pymol_scripts_for_results(
        results,
        pdb_dir=str(pdb_dir),
        output_dir=output_dir,
        min_tier=args.min_tier,
        include_variants=not args.no_variants,
        render_png=args.render,
        verbose=True,
    )

    print(f"\n{n_generated} PyMOL scripts written to {output_dir}",
          file=sys.stderr)


def main() -> None:
    """CLI entry point for pymol_scripts.py."""
    parser = argparse.ArgumentParser(
        description="Generate PyMOL .pml scripts for AlphaFold2 complexes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # -- generate subcommand --
    p_gen = subparsers.add_parser(
        'generate',
        help="Generate a .pml script for a single PDB file.",
    )
    p_gen.add_argument('--pdb', required=True,
                       help="Path to the PDB file.")
    p_gen.add_argument('--output', default=None,
                       help="Output directory (default: current directory).")
    p_gen.add_argument('--render', action='store_true',
                       help="Include ray-tracing and PNG commands (uncommented).")
    p_gen.add_argument('--name', default=None,
                       help="Override the complex name (default: PDB filename stem).")
    p_gen.set_defaults(func=_cli_generate)

    # -- batch subcommand --
    p_batch = subparsers.add_parser(
        'batch',
        help="Generate .pml scripts for all qualifying complexes from a CSV.",
    )
    p_batch.add_argument('--csv', required=True,
                         help="Path to toolkit results CSV file.")
    p_batch.add_argument('--pdb-dir', required=True,
                         help="Directory containing PDB files.")
    p_batch.add_argument('--output', default=None,
                         help="Output directory for .pml files "
                              "(default: ./pymol_scripts/).")
    p_batch.add_argument('--min-tier', default='High',
                         choices=['High', 'Medium', 'Low'],
                         help="Minimum quality tier for script generation "
                              "(default: High).")
    p_batch.add_argument('--render', action='store_true',
                         help="Include ray-tracing and PNG commands (uncommented).")
    p_batch.add_argument('--no-variants', action='store_true',
                         help="Skip variant highlighting.")
    p_batch.set_defaults(func=_cli_batch)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
