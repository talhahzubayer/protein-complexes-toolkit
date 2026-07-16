#!/usr/bin/env python
"""render_complex_summary.py — one polished summary PNG from one AlphaFold2-Multimer dimer.

A deliberately narrow public CLI. It takes a single directory containing one PDB and one
PKL, computes the existing production metrics (via ``toolkit.process_single_complex`` — no
scientific formulas or tier thresholds are reimplemented here), renders two PyMOL views in
the same orientation (chain/interface and pLDDT confidence), composes the approved generic
figure (two legends, the two views, one metric box, an explanatory footer), writes exactly
one PNG, and removes every intermediate file automatically.

Scope: dimers only (the interface-informed assessment is calibrated on dimers). No network
access is required; gene/protein names fall back to accessions when the local alias table
is unavailable. PyMOL is an external runtime invoked via ``--pymol-executable``.

    python render_complex_summary.py \\
        --input-dir "/path/to/P61088_Q9H4P4" \\
        --output    "/path/to/P61088_Q9H4P4_summary.png"

This output is an inspection/communication figure. It is not evidence that the predicted
interaction is biologically real.
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

# The CLI lives at the repository root alongside the other production scripts; make the
# repo importable regardless of the caller's working directory.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---- Reused production code (authoritative; nothing scientific is reimplemented) ----------
import toolkit                                   # process_single_complex, _strip_filename_suffixes
import complex_resolver                          # PDB/PKL suffix conventions + canonical resolver
from file_io import open_text_maybe_compressed   # transparent .gz/.bz2 decompression
from pymol_scripts import PLDDT_COLOUR_BANDS      # canonical AlphaFold pLDDT colours (name, expr, hex)


# --------------------------------------------------------------------------------------------
# Palette (approved dissertation-figure look). Chain/interface colours are cosmetic; the
# pLDDT band colours are taken verbatim from pymol_scripts.PLDDT_COLOUR_BANDS so the figure
# matches the toolkit's canonical AlphaFold scheme. Tier colours match
# visualise_results.TIER_COLORS.
CHAIN_A_RGB = (0.298, 0.447, 0.690)   # muted blue  #4C72B0
CHAIN_B_RGB = (0.173, 0.627, 0.600)   # teal        #2CA099
IFACE_RGB   = (1.000, 0.122, 0.573)   # magenta     #FF1F92
TIER_COLOURS = {"High": "#2ecc71", "Medium": "#f39c12", "Low": "#e74c3c"}  # visualise_results.TIER_COLORS
INK = "#1a1a1a"
GREY = "0.45"

# pLDDT bands keyed by name. _PLDDT for matplotlib ("#RRGGBB"); _PLDDT_PML for the PyMOL
# .pml ("0xRRGGBB", the canonical form used by pymol_scripts.generate_plddt_colouring).
_PLDDT = {name: "#" + hexc[2:] for name, _expr, hexc in PLDDT_COLOUR_BANDS}
_PLDDT_PML = {name: hexc for name, _expr, hexc in PLDDT_COLOUR_BANDS}
PLDDT_LEGEND = [
    ("Very high, pLDDT > 90",       _PLDDT["plddt_vhigh"]),
    ("Confident, 70 < pLDDT ≤ 90",  _PLDDT["plddt_high"]),
    ("Low, 50 < pLDDT ≤ 70",        _PLDDT["plddt_low"]),
    ("Very low, pLDDT ≤ 50",        _PLDDT["plddt_vlow"]),
]

# Accepted input file forms (mirrors complex_resolver's conventions, plus .gz per the task).
PDB_SUFFIXES = (".pdb", ".pdb.gz", ".pdb.bz2")
PKL_SUFFIXES = (".pkl", ".pkl.gz", ".pkl.bz2",
                ".results.pkl", ".results.pkl.gz", ".results.pkl.bz2")

_DPI = 200
DEFAULT_WIDTH = 2000    # divisible by _DPI -> exact pixel size for the default case
DEFAULT_HEIGHT = 1600


class RenderError(Exception):
    """Ordinary, user-facing failure (no traceback is printed for these)."""


# ============================== 1. file resolution =========================================

def _matches_any(name: str, suffixes) -> bool:
    low = name.lower()
    return any(low.endswith(s) for s in suffixes)


def find_prediction_files(input_dir: Path) -> tuple[Path, Path]:
    """Resolve exactly one PDB and one PKL from ``input_dir``.

    Reuses complex_resolver's canonical name-based resolution first (which also flags
    ambiguity), then falls back to a suffix glob so arbitrarily-named or .gz inputs work.
    Fails clearly on a missing or ambiguous PDB/PKL.
    """
    if not input_dir.exists():
        raise RenderError(f"input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise RenderError(f"input path is not a directory: {input_dir}")

    # Canonical resolution (files named after the directory, .pdb/.pdb.bz2 etc.).
    resolved = complex_resolver._resolve_files(input_dir, input_dir.name)
    if resolved.pdb_ambiguous:
        raise RenderError(
            f"more than one plausible PDB matched in {input_dir}; "
            f"leave exactly one PDB in the directory")
    if resolved.pkl_ambiguous:
        raise RenderError(
            f"more than one plausible PKL matched in {input_dir}; "
            f"leave exactly one PKL in the directory")
    pdb, pkl = resolved.pdb, resolved.pkl

    # Suffix-glob fallback for anything not named after the directory (and for .gz).
    if pdb is None or pkl is None:
        entries = [p for p in sorted(input_dir.iterdir()) if p.is_file()]
        pdbs = [p for p in entries if _matches_any(p.name, PDB_SUFFIXES)]
        pkls = [p for p in entries if _matches_any(p.name, PKL_SUFFIXES)]
        if pdb is None:
            if not pdbs:
                raise RenderError(f"no PDB file found in {input_dir} "
                                  f"(expected one of {', '.join(PDB_SUFFIXES)})")
            if len(pdbs) > 1:
                raise RenderError("more than one plausible PDB found in "
                                  f"{input_dir}: {', '.join(p.name for p in pdbs)}")
            pdb = pdbs[0]
        if pkl is None:
            if not pkls:
                raise RenderError(f"no PKL file found in {input_dir} "
                                  f"(expected one of {', '.join(PKL_SUFFIXES)})")
            if len(pkls) > 1:
                raise RenderError("more than one plausible PKL found in "
                                  f"{input_dir}: {', '.join(p.name for p in pkls)}")
            pkl = pkls[0]
    return pdb, pkl


def _complex_name_from_pdb(pdb_path: Path) -> str:
    """Derive the ``{A}_{B}`` complex name using the production filename parser."""
    name = pdb_path.name
    for comp in (".gz", ".bz2"):
        if name.lower().endswith(comp):
            name = name[: -len(comp)]
    return toolkit._strip_filename_suffixes(name)


# ============================== 2. metrics (production code) ================================

_REQUIRED_FINITE = (
    "iptm", "pdockq", "interface_plddt_combined", "strict_confident_contact_fraction",
    "interface_pae_mean", "interface_symmetry", "contacts_per_interface_residue",
    "interface_confidence_score", "n_interface_contacts", "n_strict_confident_contacts",
)


def _is_finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _truthy(value) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def calculate_complex_metrics(complex_name: str, pdb_path: Path, pkl_path: Path) -> dict:
    """Run the production single-complex path and validate a fully-scored dimer row."""
    try:
        row = toolkit.process_single_complex(
            complex_name, {"pdb": pdb_path, "pkl": pkl_path},
            run_interface=True, run_interface_pae=True, verbose=False)
    except Exception as exc:  # noqa: BLE001 - surface as a clean user error
        raise RenderError(f"failed to process the complex ({pkl_path.name}): {exc}") from exc

    if row.get("quality_tier") == "Error" or row.get("_error"):
        raise RenderError(f"the complex could not be processed: {row.get('_error', 'worker error')}")

    # Dimer-only scope.
    try:
        n_chains = int(float(row.get("n_chains")))
    except (TypeError, ValueError):
        n_chains = None
    if n_chains != 2:
        raise RenderError(
            f"this renderer is dimer-only, but the structure has "
            f"{n_chains if n_chains is not None else 'an unknown number of'} chains")

    reason = str(row.get("partial_reason") or "").strip()
    if reason:
        raise RenderError(f"the complex is not fully recoverable (partial_reason={reason})")
    if not _truthy(row.get("composite_is_calibrated")):
        raise RenderError("the interface-informed composite is not calibrated for this dimer; "
                          "cannot produce a meaningful summary")
    if int(float(row.get("n_interface_contacts") or 0)) <= 0:
        raise RenderError("no cross-chain interface contacts were detected")
    for key in _REQUIRED_FINITE:
        if not _is_finite(row.get(key)):
            raise RenderError(f"required metric '{key}' is missing or non-finite")
    for tier_key in ("quality_tier", "quality_tier_v2"):
        if str(row.get(tier_key) or "").strip() not in ("Low", "Medium", "High"):
            raise RenderError(f"quality tier '{tier_key}' is unavailable")
    for res_key in ("interface_residues_a", "interface_residues_b"):
        if not str(row.get(res_key) or "").strip():
            raise RenderError(f"interface residues ('{res_key}') could not be recovered")
    return row


# ============================== 3. display names (offline) =================================

def resolve_display_names(row: dict) -> tuple[str, str]:
    """Return (name_a, name_b): production gene names, else local ID map, else accessions.

    Never requires network access and never fails if the local alias table is absent.
    """
    acc_a = str(row.get("protein_a") or "").strip() or "chain A"
    acc_b = str(row.get("protein_b") or "").strip() or "chain B"

    def _clean(v):
        s = str(v or "").strip()
        return s if s and s.lower() not in ("nan", "none") else None

    name_a = _clean(row.get("gene_symbol_a"))
    name_b = _clean(row.get("gene_symbol_b"))
    if name_a and name_b:
        return name_a, name_b

    try:
        from id_mapper import IDMapper, DEFAULT_ALIASES_PATH
        if Path(DEFAULT_ALIASES_PATH).is_file():
            mapper = IDMapper(api_fallback=False)  # local only, no network
            name_a = name_a or mapper.uniprot_to_gene_symbol(acc_a)
            name_b = name_b or mapper.uniprot_to_gene_symbol(acc_b)
    except Exception:  # noqa: BLE001 - names are best-effort; accessions are a valid fallback
        pass
    return (name_a or acc_a), (name_b or acc_b)


# ============================== 4. PyMOL rendering =========================================

def _resi_selection(pipe_residues: str) -> str:
    """'42|50|78' -> '42+50+78' for a PyMOL resi selection."""
    return "+".join(tok for tok in str(pipe_residues).split("|") if tok.strip())


def build_pml(structure_pdb: Path, row: dict, chain_png: Path, plddt_png: Path,
              render_w: int, render_h: int) -> str:
    """Build the two-view .pml: chain/interface, then pLDDT recolour in the SAME camera."""
    resi_a = _resi_selection(row["interface_residues_a"])
    resi_b = _resi_selection(row["interface_residues_b"])
    vhigh, conf = _PLDDT_PML["plddt_vhigh"], _PLDDT_PML["plddt_high"]
    low, vlow = _PLDDT_PML["plddt_low"], _PLDDT_PML["plddt_vlow"]

    def px(p): return str(p).replace("\\", "/")

    return f"""reinitialize
load {px(structure_pdb)}, cplx
hide everything
bg_color white
set ray_opaque_background, 0
set orthoscopic, 1
set ray_shadows, 1
set ray_shadow_decay_factor, 0.20
set antialias, 2
set ray_trace_mode, 0
set specular, 0.25
set cartoon_fancy_helices, 1
set cartoon_side_chain_helper, 1
set stick_radius, 0.25
set_color chainA_col, [{CHAIN_A_RGB[0]:.3f}, {CHAIN_A_RGB[1]:.3f}, {CHAIN_A_RGB[2]:.3f}]
set_color chainB_col, [{CHAIN_B_RGB[0]:.3f}, {CHAIN_B_RGB[1]:.3f}, {CHAIN_B_RGB[2]:.3f}]
set_color iface_col,  [{IFACE_RGB[0]:.3f}, {IFACE_RGB[1]:.3f}, {IFACE_RGB[2]:.3f}]
show cartoon, cplx
select iface, (cplx and chain A and resi {resi_a}) or (cplx and chain B and resi {resi_b})
deselect
# --- View 1: chain / interface (orient + zoom ONCE; camera reused by view 2) ---
color chainA_col, cplx and chain A
color chainB_col, cplx and chain B
show sticks, iface
color iface_col, iface
orient cplx
zoom cplx, 5
png {px(chain_png)}, width={render_w}, height={render_h}, dpi=300, ray=1
# --- View 2: pLDDT bands (same camera, no re-orient). Strict > operators only. ---
hide sticks, iface
color {vlow}, cplx
color {low},  cplx and b > 50
color {conf}, cplx and b > 70
color {vhigh}, cplx and b > 90
png {px(plddt_png)}, width={render_w}, height={render_h}, dpi=300, ray=1
"""


def _pymol_subprocess_env(pymol_executable: str) -> dict:
    """Environment for the PyMOL subprocess only (parent Python is untouched).

    When PyMOL is a conda/micromamba env executable (``<prefix>/Scripts/pymol[.exe]`` or
    ``<prefix>/bin/pymol``), prepend that env's runtime dirs so the executable finds its
    own libraries without the user having to activate the environment. Harmless when the
    executable is a plain system binary (the extra dirs simply do not exist).
    """
    env = os.environ.copy()
    exe = Path(pymol_executable)
    if exe.parent.name.lower() in ("scripts", "bin") and exe.parent.parent.is_dir():
        prefix = exe.parent.parent
        extra = [str(prefix / sub) if sub else str(prefix)
                 for sub in ("", "Library/bin", "Library/mingw-w64/bin", "Library/usr/bin", "bin", "lib")]
        extra = [d for d in extra if Path(d).is_dir()]
        if extra:
            env["PATH"] = os.pathsep.join(extra) + os.pathsep + env.get("PATH", "")
    return env


def render_views(pdb_path: Path, row: dict, temp_dir: Path, pymol_executable: str,
                 render_w: int, render_h: int) -> tuple[Path, Path]:
    """Decompress the PDB, run PyMOL headless once, return (chain_png, plddt_png)."""
    structure_pdb = temp_dir / "decompressed_structure.pdb"
    with open_text_maybe_compressed(pdb_path) as src, \
            structure_pdb.open("w", encoding="utf-8") as dst:
        shutil.copyfileobj(src, dst)

    chain_png = temp_dir / "chain_interface.png"
    plddt_png = temp_dir / "plddt_confidence.png"
    pml_path = temp_dir / "render.pml"
    pml_path.write_text(build_pml(structure_pdb, row, chain_png, plddt_png, render_w, render_h),
                        encoding="utf-8")

    try:
        subprocess.run([pymol_executable, "-cq", str(pml_path)],
                       check=True, capture_output=True, text=True,
                       env=_pymol_subprocess_env(pymol_executable))
    except FileNotFoundError as exc:
        raise RenderError(f"PyMOL executable not found: {pymol_executable!r}. "
                          f"Install headless PyMOL and pass --pymol-executable.") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()[-1500:]
        raise RenderError(f"PyMOL rendering failed (exit {exc.returncode}):\n{detail}") from exc

    for png in (chain_png, plddt_png):
        if not png.is_file() or png.stat().st_size == 0:
            raise RenderError(f"PyMOL did not produce {png.name}; check the PyMOL install")
    return chain_png, plddt_png


# ============================== 5. value formatting ========================================

def _num(row, key):
    v = row.get(key)
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _fmt(value, spec, suffix=""):
    return f"{value:{spec}}{suffix}" if value is not None else "n/a"


# ============================== 6. figure composition ======================================

def compose_figure(chain_png: Path, plddt_png: Path, row: dict, names: tuple[str, str],
                   output_path: Path, width: int, height: int) -> None:
    """Compose the generic figure with matplotlib and write it atomically as one PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, FancyBboxPatch
    from matplotlib.offsetbox import TextArea, HPacker, AnnotationBbox, DrawingArea
    import numpy as np

    s = width / DEFAULT_WIDTH  # controlled scaling of fonts/markers with custom dimensions

    def autocrop(path):
        im = plt.imread(path)
        alpha = im[..., 3] > 0.02 if im.shape[2] == 4 else ~np.all(im[..., :3] > 0.97, axis=-1)
        ys, xs = np.where(alpha)
        if len(ys) == 0:
            return im
        return im[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    name_a, name_b = names
    v1, v2 = row["quality_tier"], row["quality_tier_v2"]
    iptm, pdockq = _num(row, "iptm"), _num(row, "pdockq")
    iplddt = _num(row, "interface_plddt_combined")
    strict = _num(row, "strict_confident_contact_fraction")
    ipae = _num(row, "interface_pae_mean")
    sym = _num(row, "interface_symmetry")
    density = _num(row, "contacts_per_interface_residue")
    comp = _num(row, "interface_confidence_score")
    n_contacts = _num(row, "n_interface_contacts")
    n_strict = _num(row, "n_strict_confident_contacts")
    pct = (strict * 100) if strict is not None else None

    fig = plt.figure(figsize=(width / _DPI, height / _DPI), dpi=_DPI)
    gs = fig.add_gridspec(4, 2, height_ratios=[0.82, 3.05, 1.5, 0.42],
                          width_ratios=[1.5, 1.0], hspace=0.05, wspace=0.03,
                          top=0.985, bottom=0.02, left=0.045, right=0.985)

    # ---- shared legends (offsetbox -> guaranteed spacing) ----
    axl = fig.add_subplot(gs[0, :]); axl.axis("off")

    def swatch_da(color, sz):
        da = DrawingArea(sz, sz, 0, 0)
        da.add_artist(Rectangle((0, 0), sz, sz, facecolor=color, edgecolor="none"))
        return da

    def legend_row(y, header, items, hfs, ifs, sep_pair, sep_item):
        kids = [TextArea(header, textprops=dict(fontsize=hfs, fontweight="bold"))]
        for lab, col in items:
            kids.append(HPacker(children=[swatch_da(col, ifs * 1.5),
                                          TextArea(lab, textprops=dict(fontsize=ifs))],
                                align="center", sep=sep_pair, pad=0))
        pack = HPacker(children=kids, align="center", sep=sep_item, pad=0)
        axl.add_artist(AnnotationBbox(pack, (0.0, y), xycoords="axes fraction",
                                      box_alignment=(0, 0.5), frameon=False, pad=0))

    legend_row(0.74, "Structure colouring:",
               [("Chain A", CHAIN_A_RGB), ("Chain B", CHAIN_B_RGB), ("Predicted interface", IFACE_RGB)],
               10.5 * s, 10.5 * s, 6 * s, 24 * s)
    legend_row(0.24, "pLDDT confidence colouring:", PLDDT_LEGEND, 9.5 * s, 8 * s, 4 * s, 11 * s)

    # ---- image row ----
    axc = fig.add_subplot(gs[1, 0]); axc.axis("off"); axc.imshow(autocrop(chain_png))
    axc.set_title("Chain and interface view", fontsize=9.5 * s, color=GREY, pad=2)
    axp = fig.add_subplot(gs[1, 1]); axp.axis("off"); axp.imshow(autocrop(plddt_png))
    axp.set_title("pLDDT confidence view", fontsize=9.5 * s, color=GREY, pad=2)

    # ---- metrics box ----
    axm = fig.add_subplot(gs[2, :]); axm.axis("off")
    axm.add_patch(FancyBboxPatch((0.002, 0.04), 0.996, 0.92, transform=axm.transAxes,
                  boxstyle="round,pad=0.004", facecolor=(0.957, 0.966, 0.983),
                  edgecolor=(0.78, 0.82, 0.88), lw=1.0))
    axm.text(0.014, 0.855, f"{name_a} and {name_b}", transform=axm.transAxes,
             fontsize=12 * s, fontweight="bold", va="center")

    def rich(x, y, segs, fontsize, ha="left"):
        areas = [TextArea(t, textprops=dict(color=c, fontweight=w, fontsize=fontsize)) for t, c, w in segs]
        pack = HPacker(children=areas, align="baseline", pad=0, sep=0)
        axm.add_artist(AnnotationBbox(pack, (x, y), xycoords="axes fraction",
                                      box_alignment=((0 if ha == "left" else 1), 0.5),
                                      frameon=False, pad=0))

    rich(0.986, 0.855,
         [("Baseline quality tier: ", INK, "normal"), (v1, TIER_COLOURS[v1], "bold"),
          ("   →   Interface-informed quality tier: ", INK, "normal"), (v2, TIER_COLOURS[v2], "bold")],
         10.5 * s, ha="right")
    fs = 8.2 * s
    rich(0.014, 0.60, [("Baseline metrics:  ", INK, "bold"),
         (f"ipTM = {_fmt(iptm, '.3f')}; pDockQ = {_fmt(pdockq, '.3f')}.", INK, "normal")], fs)
    frac = (f"{_fmt(strict, '.3f')} ({int(n_strict) if n_strict is not None else 'n/a'} of "
            f"{int(n_contacts) if n_contacts is not None else 'n/a'} contacts, {_fmt(pct, '.1f', '%')})")
    rich(0.014, 0.375, [("Interface confidence:  ", INK, "bold"),
         (f"mean interface pLDDT = {_fmt(iplddt, '.1f')}; strict confident-contact fraction = {frac}; "
          f"mean interface PAE = {_fmt(ipae, '.2f', ' Å')}.", INK, "normal")], fs)
    rich(0.014, 0.15, [("Geometry and composite:  ", INK, "bold"),
         (f"interface symmetry = {_fmt(sym, '.3f')}; contact density = {_fmt(density, '.3f')} "
          f"contacts per interface residue; interface confidence score = {_fmt(comp, '.3f')}.", INK, "normal")], fs)

    # ---- footer ----
    axf = fig.add_subplot(gs[3, :]); axf.axis("off")
    axf.text(0.0, 0.62, "Strict confident-contact fraction is the proportion of predicted interface "
             "contacts with bidirectional PAE < 5 Å and pLDDT ≥ 70 at both contacting residues.",
             fontsize=8.0 * s, style="italic", color=GREY, va="center")
    axf.text(0.0, 0.20, "This is an illustrative prediction and is not evidence that the interaction "
             "is biologically real.", fontsize=8.0 * s, style="italic", color=GREY, va="center")

    fig.patch.set_facecolor("white")

    # ---- atomic write: compose to a temp file in the destination dir, validate, replace ----
    tmp = output_path.with_name(output_path.name + ".tmp")
    try:
        try:
            fig.savefig(tmp, dpi=_DPI, facecolor="white", format="png")
        finally:
            plt.close(fig)
        actual = _validate_png(tmp, width, height)
        os.replace(tmp, output_path)   # atomic on the destination filesystem
        return actual
    except BaseException:
        tmp.unlink(missing_ok=True)    # never leave a truncated/partial file behind
        raise


def _png_dimensions(path: Path) -> tuple[int, int]:
    """Read (width, height) from a PNG's IHDR chunk using only the standard library."""
    with path.open("rb") as fh:
        head = fh.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        raise RenderError("composed image is not a valid PNG")
    return struct.unpack(">II", head[16:24])


def _validate_png(path: Path, width: int, height: int) -> tuple[int, int]:
    """Confirm ``path`` is a readable PNG close to the expected size (±2 px for rounding)."""
    try:
        w, h = _png_dimensions(path)
    except RenderError:
        path.unlink(missing_ok=True)
        raise
    if abs(w - width) > 2 or abs(h - height) > 2:
        path.unlink(missing_ok=True)
        raise RenderError(f"composed PNG has unexpected dimensions {w}x{h} (expected ~{width}x{height})")
    return w, h


# ============================== 7. CLI =====================================================

def _positive_int(value: str) -> int:
    try:
        iv = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer")
    if iv <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {iv}")
    return iv


def build_argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="render_complex_summary.py",
        description="Render one polished summary PNG from one AlphaFold2-Multimer dimer "
                    "directory (one PDB + one PKL). Dimer-only; reuses the toolkit's "
                    "production metrics; requires a headless PyMOL executable.",
        epilog="The output is an inspection/communication figure, not evidence that the "
               "interaction is biologically real.")
    p.add_argument("--input-dir", required=True, type=Path,
                   help="Directory containing exactly one PDB and one PKL "
                        "(.pdb/.pdb.gz/.pdb.bz2 and .pkl/.pkl.gz/.pkl.bz2).")
    p.add_argument("--output", required=True, type=Path,
                   help="Path of the single PNG to write (must end in .png).")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite the output file if it already exists (default: fail).")
    p.add_argument("--width", type=_positive_int, default=DEFAULT_WIDTH,
                   help=f"Output width in pixels (default: {DEFAULT_WIDTH}).")
    p.add_argument("--height", type=_positive_int, default=DEFAULT_HEIGHT,
                   help=f"Output height in pixels (default: {DEFAULT_HEIGHT}).")
    p.add_argument("--pymol-executable", default=None,
                   help="Path to a headless PyMOL executable. If omitted, uses "
                        "$PYMOL_EXECUTABLE, then 'pymol' on PATH.")
    return p


def _resolve_pymol(explicit: str | None) -> str:
    """Resolve the PyMOL executable: --pymol-executable, else $PYMOL_EXECUTABLE, else PATH.

    The $PYMOL_EXECUTABLE tier lets a user point at a PyMOL living in a separate
    environment once (e.g. export it in their venv activate script) and then never
    pass the flag again.
    """
    if explicit:
        return explicit
    env_exe = os.environ.get("PYMOL_EXECUTABLE")
    if env_exe:
        return env_exe
    found = shutil.which("pymol")
    if found:
        return found
    raise RenderError("no PyMOL executable found (checked --pymol-executable, "
                      "$PYMOL_EXECUTABLE, and 'pymol' on PATH); install headless PyMOL "
                      "(e.g. conda-forge pymol-open-source) and set one of them")


def run(args: argparse.Namespace) -> None:
    output_path = args.output
    if output_path.suffix.lower() != ".png":
        raise RenderError(f"--output must end in .png, got {output_path.name!r}")
    if output_path.exists() and not args.overwrite:
        raise RenderError(f"output already exists: {output_path} (use --overwrite to replace)")
    parent = output_path.parent if str(output_path.parent) else Path(".")
    if not parent.exists():
        raise RenderError(f"output directory does not exist: {parent}")

    # Validate inputs first, then confirm PyMOL is available before the (slower) metric run.
    pdb_path, pkl_path = find_prediction_files(args.input_dir)
    pymol_executable = _resolve_pymol(args.pymol_executable)
    complex_name = _complex_name_from_pdb(pdb_path)
    row = calculate_complex_metrics(complex_name, pdb_path, pkl_path)
    names = resolve_display_names(row)

    with tempfile.TemporaryDirectory(prefix="complex_render_") as temp_dir:
        chain_png, plddt_png = render_views(
            pdb_path, row, Path(temp_dir), pymol_executable,
            render_w=max(1000, args.width), render_h=max(900, int(args.height * 0.95)))
        actual_w, actual_h = compose_figure(
            chain_png, plddt_png, row, names, output_path, args.width, args.height)

    print(f"Wrote {output_path}  ({actual_w}x{actual_h})")


def main(argv=None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        run(args)
    except RenderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
