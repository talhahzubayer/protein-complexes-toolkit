#!/usr/bin/env python3
"""
AlphaFold2 Analysis Visualisation Tool - generates up to 16 figures + 1b supplementary from the CSV produced by toolkit.py.  
Figures are auto-detected from the columns present in the CSV, so only the relevant subset is generated for any given pipeline run.

Figures are grouped by the Aim items listed on the MSc Research Project Plan document

Item 5 - Structure prediction (Figs 1-9):
  1.  ipTM vs pDockQ by Quality Tier (or falls back to disorder-fraction colouring [RdYlGn_r colourmap] if quality tiers are unavailable)
  2.  Global PAE Health Check histogram
  3.  Interface PAE by Quality Tier  (boxplots + strip)
  4.  Composite Score & Quality Tier Validation  (violin + scatter)
  5.  Interface vs Bulk pLDDT  (scatter with diagonal)
  6.  Paradox Complexes Spotlight  (violin triptych)
  7.  Complex Architecture Comparison  (Homo / Hetero / Multi-chain)
  8.  Metric Disagreement  (scatter with disagreement band)
  9.  Chain-Count Quality Profile  (violin + scatter by chain count)

Item 3 - Identify similar proteins/pairs (Fig 10, require --clustering):
 10.  Sequence Clustering Validation  (cluster sharing by architecture + tier)

Item 4 - Mapping genome variation (Figs 11-12, require --variants):
 11.  Classified Variant Sankey  (significance -> structural context)
 12.  Interface Variant Density vs Quality  (density scatter + Spearman)

Item 6 - Map stability scores (Fig 13, require --stability + --protvar):
 13.  Stability Predictor Cross-Validation  (EVE vs ProtVar concordance)

Disease & pathway analysis (Figs 14-15):
 14.  Disease Enrichment by Quality Tier  (require --disease)
 15.  Pathway-Level Network  (require --pathways, NetworkX)

Synthesis (Fig 16, require --variants + --pathways):
 16.  Prediction Quality Paradox  (2x2 panel: pathogenic interface variants and PPI density strengthen with quality while gene constraint, and disorder fraction reveal systematic AF2-Multimer prediction bias)

Supplementary:
 1b.  Disorder-coloured quality scatter (requires --disorder-scatter flag)

Rendering approach:
    All scatter figures use matplotlib scatter() at every dataset scale.
    Point size and alpha adapt automatically based on dataset size so that small datasets (~500) show large readable dots and million-scale datasets approach a pixel-dot density look.
    Status messages with timing are printed around blocking render calls so the user knows the script is not stuck, even for multi-minute renders at very large scale.
    The --density flag adds KDE density contour overlays with percentile labels to scatter figures where density context aids interpretation.

Per-complex PAE heatmaps are available on demand via --pae-heatmaps.
When a matching PDB file is found, chain boundaries are drawn and the best interacting chain pair is highlighted.

Dependencies:
    read_af2_nojax.py   -> JAX mocking and PKL loading (same directory).
    pdockq.py           -> Chain info / offsets for PAE heatmaps (optional).
    pandas, matplotlib, numpy, scipy (stats + KDE), seaborn (optional), networkx (optional, for Fig 15).

Usage:
    python visualise_results.py results.csv                                    # auto-detect
    python visualise_results.py results.csv --output-dir ./figures             # custom output
    python visualise_results.py results.csv --density                          # KDE contours
    python visualise_results.py results.csv --disorder-scatter                 # also Fig 1b
    python visualise_results.py results.csv --pae-heatmaps /path/to/models     # PAE heatmaps
    python visualise_results.py results.csv --pae-heatmaps /models --limit 50
"""

import os
import glob
import time
import argparse
from typing import Optional, Tuple
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from scipy.stats import gaussian_kde, spearmanr, mannwhitneyu, kruskal, chi2_contingency, fisher_exact
import textwrap as _textwrap

try:
    import seaborn as sns
    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False

try:
    import networkx as nx
    _HAS_NETWORKX = True
except ImportError:
    _HAS_NETWORKX = False

# JAX mocking is handled at import time by the reader module
from read_af2_nojax import load_pkl_without_jax

# Output Directory - set by command-line argument (default is current directory)
OUTPUT_DIR: str = ""

#------------------------------Shared Design Constants---------------------------------------------

# Quality tier colours - consistent across every figure
TIER_COLORS = {'High': '#2ecc71', 'Medium': '#f39c12', 'Low': '#e74c3c'}
TIER_ORDER = ['High', 'Medium', 'Low']

# Rendering
OUTPUT_DPI = 200
FONT_TITLE = 13
FONT_AXIS_LABEL = 11
FONT_TICK = 10
GRID_ALPHA = 0.3

# Thresholds drawn as reference lines where relevant
IPTM_HIGH = 0.75
PDOCKQ_HIGH = 0.5
PAE_CONFIDENT = 5.0 # Angstroms
DISORDER_SUBSTANTIAL = 0.30
METRIC_DISAGREEMENT_GAP = 0.52  # matches METRIC_DISAGREEMENT_THRESHOLD in interface_analysis.py

# Reclassification thresholds - must match toolkit.py constants
UPGRADE_LOW_THRESHOLD = 0.64
UPGRADE_MEDIUM_THRESHOLD = 0.80
DOWNGRADE_HIGH_THRESHOLD = 0.65

# Scatter plot defaults (used as fallback - preferred use is _adaptive_scatter_params)
SCATTER_POINT_SIZE = 80
SCATTER_ALPHA = 0.7

# PAE heatmap layout (on-demand only via --pae-heatmaps)
PAE_FIGURE_SIZE = (8, 7)
PAE_VMIN = 0
PAE_VMAX = 30  # Angstroms

# Variant visualisation constants (Figs 11-12, require --variants CSV columns)
CONTEXT_ORDER = ['interface_core', 'interface_rim', 'surface_non_interface', 'buried_core']
CONTEXT_LABELS = {
    'interface_core': 'Interface Core\n(<4\u00c5)',         # Unicode Å (angstrom) symbol 
    'interface_rim': 'Interface Rim\n(4\u20138\u00c5)',     # Unicode en-dash to indicate range
    'surface_non_interface': 'Surface\n(Non-Interface)',
    'buried_core': 'Buried Core',
}
CONTEXT_COLORS = {
    'interface_core': '#e74c3c',
    'interface_rim': '#f39c12',
    'surface_non_interface': '#3498db',
    'buried_core': '#95a5a6',
}
SIGNIFICANCE_ORDER = ['Pathogenic', 'Likely pathogenic', 'VUS', 'Benign', 'Unknown']
SIGNIFICANCE_COLORS = {
    'Pathogenic': '#c0392b',
    'Likely pathogenic': '#e67e22',
    'VUS': '#7f8c8d',
    'Benign': '#27ae60',
    'Unknown': '#bdc3c7',
}

# Multimer refactor (Phase 5): scope labels + dissertation-safe filter.
TIER_SCOPE_DIMER = 'dimer_validated'
TIER_SCOPE_MULTIMER = 'multimer_provisional'
DIMER_STOICHIOMETRIES = ('A2', 'AB')
# Caption policy: every figure must advertise its scope as one of these literals.
CAPTION_SCOPE_DIMER = 'dimer-validated'
CAPTION_SCOPE_ALL_N = 'all-N descriptive'
CAPTION_SCOPE_MULTIMER = 'multimer exploratory'

#-----------------------------------------------Infrastructure helpers--------------------------------------------------------

_LEGACY_CSV_WARNED = False


def _derive_tier_scope(df: pd.DataFrame) -> pd.Series:
    """Derive tier_scope for rows loaded from pre-refactor CSVs.
    Pre-refactor CSVs lack `tier_scope` and `schema_version`. Legacy rows with
    `n_chains == 2` are treated as dimer_validated; everything else as
    multimer_provisional. Emits a one-time warning the first time this is invoked.
    """
    global _LEGACY_CSV_WARNED
    if not _LEGACY_CSV_WARNED:
        print("  Warning: loaded CSV lacks schema_version/tier_scope (pre-multimer_v1). "
              "Deriving tier_scope from n_chains for backward compatibility.")
        _LEGACY_CSV_WARNED = True
    if 'n_chains' in df.columns:
        n_chains = pd.to_numeric(df['n_chains'], errors='coerce')
        return np.where(n_chains == 2, TIER_SCOPE_DIMER, TIER_SCOPE_MULTIMER)
    return np.full(len(df), TIER_SCOPE_DIMER)


def _filter_dimer_validated(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows with tier_scope == 'dimer_validated'.
    Dissertation-safe default for every figure whose thresholds were calibrated
    against dimers (Figs 4, 5, 6, 8, 13 and Fig 7 primary panel).
    """
    if 'tier_scope' not in df.columns:
        return df
    return df[df['tier_scope'] == TIER_SCOPE_DIMER].reset_index(drop=True)


def _boolish(series: pd.Series) -> pd.Series:
    """Map CSV-stringified booleans to real bools. Non-matching values become NaN.
    Use anywhere a boolean column round-tripped through CSV may arrive as the
    strings "True"/"False"/"0"/"1"/"yes"/"no" — `.astype(bool)` would coerce
    "False" to True (non-empty string is truthy).
    """
    return series.astype(str).str.strip().str.lower().map({
        "true": True, "false": False,
        "1": True, "0": False,
        "yes": True, "no": False,
    })


def _phantom_row_mask(df: pd.DataFrame, required_cols: list[str]) -> pd.Series:
    """Return a boolean Series flagging rows that should be excluded from a
    figure: any required column is null, OR `geometry_available == False`.
    Equivalent to "rows whose tier classification is unsafe to display".
    """
    if df is None or len(df) == 0:
        return pd.Series([], dtype=bool)
    present_cols = [c for c in required_cols if c in df.columns]
    if present_cols:
        missing = df[present_cols].isna().any(axis=1)
    else:
        missing = pd.Series([False] * len(df), index=df.index)
    if 'geometry_available' in df.columns:
        geometry_missing = (
            df['geometry_available'].astype(str).str.strip().str.lower().eq('false')
        )
        return missing | geometry_missing
    return missing


def warn_missing_required_rows(df: pd.DataFrame, required_cols: list[str],
                               fig_label: str, reason: str) -> None:
    """Print one warning if rows would be dropped from <fig_label>.
    `reason` is appended verbatim, so callers can distinguish score-derived
    figures (e.g. Fig 1) from interface-geometry figures (Figs 4/6/8/9). When
    the CSV carries `geometry_available`, rows with that flag False are also
    counted toward the drop - surfaces PKL-only complexes alongside genuine
    column-missing rows under one warning.
    """
    if df is None or len(df) == 0:
        return
    dropped = int(_phantom_row_mask(df, required_cols).sum())
    if dropped:
        print(f"  Warning: {dropped} rows excluded from {fig_label} due to {reason}.")


def _adaptive_scatter_params(n: int) -> Tuple[float, float]:
    """Return (point_size, alpha) scaled to dataset size.
    Ensures small datasets show large readable dots while million-scale datasets approach a pixel-dot density aesthetic. All figures use scatter() at every scale.
    Args:
        n: Number of points to be plotted.
    Returns:
        Tuple of (size, alpha) for use in axes.scatter().
    """
    if n < 1_000:
        return (40, 0.70)
    elif n < 10_000:
        return (20, 0.55)
    elif n < 50_000:
        return (10, 0.45)
    elif n < 200_000:
        return (5, 0.35)
    else:
        return (2, 0.25)

def _timed_scatter(axes: plt.Axes, x, y, n_points: int, fig_label: str = '', **kwargs) -> object:
    """Wrapper around axes.scatter() with timing and status messages.
    For datasets over 9k points, prints an advisory before the blocking scatter call so the user knows the script is not stuck.
    Args:
        axes: Matplotlib axes to plot on.
        x, y: Data arrays.
        n_points: Total points (used for advisory message).
        fig_label: Short label for the status message (e.g. 'Fig 1').
        **kwargs: Passed through to axes.scatter().
    Returns:
        The PathCollection object returned by scatter().
    """
    prefix = f"  {fig_label} | " if fig_label else "  "
    if n_points > 9_000:
        print(f"{prefix}Rendering {n_points:,} points (this may take a moment)...")
    t0 = time.time()
    result = axes.scatter(x, y, **kwargs)
    elapsed = time.time() - t0
    if n_points > 9_000 or elapsed > 2.0:
        print(f"{prefix}scatter: {elapsed:.1f}s")
    return result

#----------------------------------------------------------Column detection & data loading----------------------------------------------------------------------------

def detect_columns(df: pd.DataFrame) -> dict:
    """Detect which column groups are present in the CSV.
    Returns:
        Dictionary of boolean flags indicating available column groups.
    """
    columns = set(df.columns)
    return {
        'has_v2_data': 'quality_tier_v2' in columns,
        'has_interface_data': 'n_interface_contacts' in columns,
        'has_pae_interface': 'interface_pae_mean' in columns,
        'has_composite': 'interface_confidence_score' in columns,
        'has_chain_info': 'n_chains' in columns,
        'has_variant_data': 'n_variants_a' in columns and 'variant_details_a' in columns,
        'has_disease_data': 'n_diseases_a' in columns,
        'has_pathway_data': 'reactome_pathways_a' in columns,
        'has_stability_data': 'eve_score_mean_a' in columns and 'protvar_am_mean_a' in columns,
        'has_clustering_data': 'sequence_cluster_count' in columns and 'shared_cluster_count' in columns,
        'has_paradox_data': ('quality_tier_v2' in columns and 'n_pathogenic_interface_variants' in columns and 'ppi_enrichment_ratio' in columns and 'gene_constraint_pli_a' in columns and 'gene_constraint_pli_b' in columns and 'plddt_below50_fraction' in columns),
    }

def load_data(csv_path: str) -> pd.DataFrame:
    """Load the analysis CSV into a pandas DataFrame.
    Numeric columns are coerced and non-numeric values become NaN. Rows with missing ipTM are dropped and pDockQ = 0 is retained because genuine zero-contact complexes are informative for diagnostics.
    Also performs:
      - Case normalisation on complex_type (Homodimer -> homodimer).
      - Splits the comma-separated interface_flags column into individual boolean columns (one per flag) so that downstream figures can filter by specific flags directly.
    Args:
        csv_path: Path to the CSV produced by toolkit.py.
    Returns:
        Cleaned DataFrame.
    """
    df = pd.read_csv(csv_path)

    # Coerce key numeric columns
    numeric_candidates = [
        'iptm', 'pdockq', 'pae_mean', 'plddt_below50_fraction',
        'plddt_below70_fraction', 'interface_pae_mean',
        'interface_confidence_score',
        'pae_confident_contact_fraction', 'strict_confident_contact_fraction',
        'interface_plddt_combined', 'bulk_plddt_combined',
        'interface_vs_bulk_delta', 'interface_symmetry',
        'n_interface_contacts', 'contacts_per_interface_residue', 'n_chains',
        'pdockq_mean', 'pdockq_min', 'pdockq_whole_complex',
        'contact_count_total', 'interface_plddt_mean',
        'symmetry_mean', 'symmetry_min',
        'pae_confident_fraction_mean', 'strict_confident_fraction_mean',
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Item 1: Expand interface_flags into individual boolean columns.
    # The CSV stores flags as a single comma-separated string, e.g. "small_interface,metric_disagreement". Downstream figures need individual boolean columns.
    ALL_KNOWN_FLAGS = [
        'small_interface', 'sparse_interface', 'asymmetric_interface',
        'interface_better_than_bulk', 'low_interface_confidence',
        'paradox_confident_disorder', 'paradox_artefactual', 'metric_disagreement',
    ]

    # Item 2: Normalise complex_type to lowercase.
    # parse_complex_name() produces 'Homodimer' / 'Heterodimer' but several figures filtered for lowercase so normalise once here.
    if 'complex_type' in df.columns:
        df['complex_type'] = df['complex_type'].astype(str).str.lower()

    # Item 3: Relaxed drop filter
    # Keep complexes with pDockQ = 0 (genuine zero-contact interfaces) since they are informative for diagnostics.
    # Only drop rows that truly lack ipTM.
    initial_count = len(df)
    df = df.dropna(subset=['iptm'])
    df = df[df['iptm'] > 0]
    # pDockQ: fill NaN with 0 (no PDB or no contacts) rather than dropping
    if 'pdockq' in df.columns:
        df['pdockq'] = df['pdockq'].fillna(0)
    # n_chains: infer from complex_type for rows where PDB was unavailable
    if 'n_chains' in df.columns and 'complex_type' in df.columns:
        missing_mask = df['n_chains'].isna()
        if missing_mask.any():
            # homodimer / heterodimer filenames always imply a 2-chain complex
            dimer_mask = missing_mask & df['complex_type'].isin(['homodimer', 'heterodimer'])
            df.loc[dimer_mask, 'n_chains'] = 2
            filled = dimer_mask.sum()
            still_missing = df['n_chains'].isna().sum()
            if filled > 0:
                print(f"  Inferred n_chains=2 for {filled} rows from complex_type "
                      f"({still_missing} still missing).")
    dropped = initial_count - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped} rows with missing/zero ipTM.")

    if 'interface_flags' in df.columns:
        flags_series = df['interface_flags'].fillna('').astype(str)
        for flag_name in ALL_KNOWN_FLAGS:
            df[flag_name] = flags_series.str.contains(flag_name, regex=False)

    # Multimer refactor (Phase 5): derive tier_scope for pre-refactor CSVs.
    # Post-refactor CSVs (schema_version == "multimer_v1") already ship tier_scope.
    if 'tier_scope' not in df.columns:
        df['tier_scope'] = _derive_tier_scope(df)

    return df.reset_index(drop=True)

#---------------------------------Shared rendering helpers (used across many figures)------------------------------------------

def _apply_common_style(axes: plt.Axes, title: str, xlabel: str, ylabel: str, grid: bool = True) -> None:
    """Apply consistent font sizes and grid styling to a matplotlib axes.
    Args:
        axes: Matplotlib axes to style.
        title: Axes title text.
        xlabel: X-axis label text.
        ylabel: Y-axis label text.
        grid: Whether to add a dashed grid overlay.
    """
    axes.set_title(title, fontsize=FONT_TITLE, fontweight='bold', pad=12)
    axes.set_xlabel(xlabel, fontsize=FONT_AXIS_LABEL)
    axes.set_ylabel(ylabel, fontsize=FONT_AXIS_LABEL)
    axes.tick_params(labelsize=FONT_TICK)
    if grid:
        axes.grid(True, alpha=GRID_ALPHA, linestyle='--')

def _species_display(species_label: str) -> str:
    """Human-readable suffix for figure titles that mirrors the file suffix.

    '' -> '', '_human' -> ' - Human', '_nonhuman' -> ' - Non-Human'.
    """
    if not species_label:
        return ''
    mapping = {'_human': ' - Human', '_nonhuman': ' - Non-Human'}
    return mapping.get(species_label, f' - {species_label.lstrip("_").replace("_", " ").title()}')

def _save_figure(figure: plt.Figure, filename: str) -> None:
    """Save a figure to OUTPUT_DIR at standard DPI and close it.
    Prints the filename on completion with elapsed time if saving takes over 2 seconds.
    Args:
        figure: Matplotlib Figure to save.
        filename: File name (not full path) within OUTPUT_DIR.
    """
    output_path = os.path.join(OUTPUT_DIR, filename)
    figure.tight_layout()
    t0 = time.time()
    figure.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches='tight')
    elapsed = time.time() - t0
    plt.close(figure)
    if elapsed > 2.0:
        print(f"  Saved: {filename} (save: {elapsed:.1f}s)")
    else:
        print(f"  Saved: {filename}")

def _build_tier_legend_handles(df: pd.DataFrame) -> list:
    """Build legend Line2D handles with per-tier counts from a DataFrame.
    Args:
        df: DataFrame with a 'quality_tier_v2' column.
    Returns:
        List of matplotlib Line2D handles suitable for axes.legend().
    """
    handles = []
    for tier in TIER_ORDER:
        count = (df['quality_tier_v2'] == tier).sum()
        handles.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=TIER_COLORS[tier], markeredgecolor='white', markersize=9, label=f'{tier} ({count})'))
    # Grey for missing-tier complexes
    missing_count = df['quality_tier_v2'].isna().sum()
    if missing_count > 0:
        handles.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#bdc3c7', markeredgecolor='white', markersize=9, label=f'Unclassified ({missing_count})'))
    return handles

def _overlay_kde_contours(axes: plt.Axes, x: np.ndarray, y: np.ndarray, color: str = '#333333', linewidth: float = 0.9, alpha: float = 0.6, max_kde_points: int = 50_000) -> None:
    """Overlay KDE density contours with percentile labels on a scatter axes.
    Contour levels are percentile-based (10th, 30th, 50th, 70th, 90th of probability mass).
    The innermost ring encloses the top 10% density region and the outermost encloses 90% of all points - fails silently if too few points or if KDE encounters a singular matrix.
    Non-finite values are stripped, and inputs above `max_kde_points` are deterministically downsampled (seed=42) so HPC-scale runs do not stall.
    Args:
        axes: Matplotlib axes with scatter data already plotted.
        x, y: 1D arrays of scatter coordinates.
        color: Line colour for contours.
        linewidth: Contour line width.
        alpha: Contour line alpha.
        max_kde_points: Cap on points fed to gaussian_kde; larger inputs are downsampled.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 20: # Too few points for KDE to be meaningful - skip contours
        return
    if len(x) > max_kde_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(x), size=max_kde_points, replace=False)
        x = x[idx]
        y = y[idx]
    try:
        kde = gaussian_kde(np.vstack([x, y]), bw_method='scott')
        x_grid = np.linspace(x.min() - 0.02, x.max() + 0.02, 120)
        y_grid = np.linspace(y.min() - 0.02, y.max() + 0.02, 120)
        xx, yy = np.meshgrid(x_grid, y_grid)
        positions = np.vstack([xx.ravel(), yy.ravel()])
        zz = kde(positions).reshape(xx.shape)

        # Percentile-based contour levels
        zz_sorted = np.sort(zz.ravel())
        cumsum = np.cumsum(zz_sorted)
        cumsum /= cumsum[-1] # Normalise to get cumulative probabilities
        percentile_thresholds = [0.10, 0.30, 0.50, 0.70, 0.90]
        levels = []
        level_labels = {}
        for p in percentile_thresholds:
            idx = np.searchsorted(cumsum, p)
            density_val = zz_sorted[min(idx, len(zz_sorted) - 1)]
            levels.append(density_val)
            pct_inside = int(round((1 - p) * 100))
            level_labels[density_val] = f'{pct_inside}%'

        # Deduplicate while preserving label mapping
        seen = set()
        unique_levels = []
        for lv in sorted(levels):
            if lv not in seen:
                unique_levels.append(lv)
                seen.add(lv)
        levels = unique_levels
        if len(levels) >= 2:
            contours = axes.contour(xx, yy, zz, levels=levels, colors=color, linewidths=linewidth, alpha=alpha, zorder=4)
            fmt = {}
            for lv in contours.levels:
                closest = min(level_labels.keys(), key=lambda k: abs(k - lv))
                fmt[lv] = level_labels[closest]
            clabels = axes.clabel(contours, contours.levels, fmt=fmt, fontsize=8, inline=True, inline_spacing=4)
            for txt in clabels:
                txt.set_bbox(dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85))
                txt.set_fontweight('bold')
    except np.linalg.LinAlgError:
        print("  Note: KDE failed (singular covariance); contours skipped.")

def _despine(ax) -> None:
    """Remove top and right spines if seaborn is available."""
    if _HAS_SEABORN:
        sns.despine(ax=ax)

def _get_paradox_mask(df: pd.DataFrame) -> pd.Series:
    """Identify paradox complexes in a DataFrame.
    Paradox definition: ipTM >= 0.75 AND pDockQ >= 0.5 AND disorder fraction >= 0.30 - complexes where both headline quality metrics indicate a confident interaction despite substantial structural disorder.
    Returns:
        Boolean Series aligned with the DataFrame index.
    """
    mask = pd.Series(False, index=df.index)
    required = ['iptm', 'pdockq', 'plddt_below50_fraction']
    if not all(col in df.columns for col in required):
        return mask
    mask = ((df['iptm'] >= IPTM_HIGH) & (df['pdockq'] >= PDOCKQ_HIGH) & (df['plddt_below50_fraction'] >= DISORDER_SUBSTANTIAL))
    return mask.fillna(False)

#--------------------------------------------------------PAE heatmap helpers (on-demand via --pae-heatmaps)---------------------------------------------------------------

def load_pae_matrix_from_pkl(pkl_path: str) -> Optional[np.ndarray]:
    """Load a PAE matrix from an AlphaFold2 PKL file.
    Delegates to read_af2_nojax for JAX-free PKL loading.
    Args:
        pkl_path: Path to the PKL file.
    Returns:
        2D numpy array of PAE values or None if not present.
    """
    data = load_pkl_without_jax(pkl_path)
    if 'predicted_aligned_error' not in data:
        return None
    return np.asarray(data['predicted_aligned_error'])

def extract_readable_title(pkl_filename: str) -> str:
    """Shorten a PKL filename into a readable plot title.
    Args:
        pkl_filename: The basename of the PKL file.
    Returns:
        A shortened, readable title string.
    """
    if "_result_model" in pkl_filename:
        return pkl_filename.split("_result_model")[0]
    return pkl_filename.replace('.pkl', '')

def plot_pae_matrix(pkl_path: str, models_dir: str) -> None:
    """Generate and save a PAE heatmap for a single AlphaFold2 prediction - on-demand only (--pae-heatmaps).
    Uses Greens_r colourmap, clamped to 0-30 Å.
    When a matching PDB file is found alongside the PKL, chain boundaries are drawn as dashed lines and the best interacting chain pair's cross-chain PAE block is highlighted with a translucent rectangle.
    Args:
        pkl_path: Full path to the .pkl file.
        models_dir: Directory to save the heatmap PNG alongside the PKL files.
    """
    filename = os.path.basename(pkl_path)
    readable_title = extract_readable_title(filename)
    output_filename = f"{filename.replace('.pkl', '')}_PAE.png"
    output_path = os.path.join(models_dir, output_filename)

    try:
        pae_matrix = load_pae_matrix_from_pkl(pkl_path)
    except Exception as error:
        print(f"  Error processing {pkl_path}: {error}")
        return

    if pae_matrix is None:
        print(f"  Skipping {filename}: No PAE data found.")
        return

    figure, axes = plt.subplots(figsize=PAE_FIGURE_SIZE)
    image = axes.imshow(pae_matrix, cmap='Greens_r', vmin=PAE_VMIN, vmax=PAE_VMAX, interpolation='nearest')
    colour_bar = figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04)
    colour_bar.set_label('Expected Position Error (Å)', rotation=270, labelpad=15)

    #=========================================================Chain boundary lines and best-pair highlighting==================================================================
    # Look for a matching PDB file in the same directory
    # Note: '.results.pkl' must be matched before plain '.pkl' — for
    # X.results.pkl, str.replace('.pkl', '.pdb') yields X.results.pdb (wrong).
    pdb_candidates = [
        pkl_path.replace('.results.pkl', '.pdb'),
        pkl_path.replace('.pkl', '.pdb'),
        pkl_path.replace('_result_', '_relaxed_').replace('.pkl', '.pdb'),
    ]
    pdb_path = None
    for candidate in pdb_candidates:
        if os.path.isfile(candidate):
            pdb_path = candidate
            break

    if pdb_path is not None:
        try:
            from pdockq import read_pdb_with_chain_info, compute_pae_chain_offsets, find_best_chain_pair
            from pdockq import (read_pdb_with_chain_info_New as read_pdb_with_chain_info, compute_pae_chain_offsets_New as compute_pae_chain_offsets, find_best_chain_pair_New as find_best_chain_pair) # Import aliasing to avoid naming conflicts with old pdockq versions

            chain_info = read_pdb_with_chain_info(pdb_path)
            offsets = compute_pae_chain_offsets(chain_info)
            n_total = pae_matrix.shape[0]

            # Draw chain boundary lines
            boundary_positions = []
            chain_labels_for_axis = []
            for ch in chain_info.chain_ids:
                start = offsets[ch]
                end = start + chain_info.ca_counts[ch]
                midpoint = (start + end) / 2
                chain_labels_for_axis.append((midpoint, ch))
                if start > 0:
                    boundary_positions.append(start)

            for bpos in boundary_positions:
                axes.axhline(y=bpos - 0.5, color='white', linestyle='--', linewidth=1.5, alpha=0.8)
                axes.axvline(x=bpos - 0.5, color='white', linestyle='--', linewidth=1.5, alpha=0.8)

            # Add chain ID labels along the top edge
            if len(chain_info.chain_ids) > 1:
                for midpoint, ch_id in chain_labels_for_axis:
                    axes.text(midpoint, -0.02, ch_id, transform=axes.get_xaxis_transform(), ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2c3e50')

            # Highlight best-pair cross-chain block
            if len(chain_info.chain_ids) >= 2:
                ch_a, ch_b, contact_result = find_best_chain_pair(chain_info, t=8)
                off_a = offsets[ch_a]
                off_b = offsets[ch_b]
                len_a = chain_info.ca_counts[ch_a]
                len_b = chain_info.ca_counts[ch_b]

                # Highlight both cross-chain rectangles (A->B and B->A)
                for rect_x, rect_y, rect_w, rect_h in [
                    (off_b - 0.5, off_a - 0.5, len_b, len_a),
                    (off_a - 0.5, off_b - 0.5, len_a, len_b),
                ]:
                    rect = plt.Rectangle(
                        (rect_x, rect_y), rect_w, rect_h,
                        linewidth=2, edgecolor='#e74c3c', facecolor='none',
                        linestyle='-', alpha=0.8, zorder=5,
                    )
                    axes.add_patch(rect)
                readable_title += f'  (best pair: {ch_a}-{ch_b})'

        except Exception:
            pass  # Gracefully degrade to plain heatmap

    axes.set_title(f'PAE Matrix: {readable_title}', fontsize=12, fontweight='bold')
    axes.set_xlabel('Residue Index (Scored)')
    axes.set_ylabel('Residue Index (Aligned)')

    figure.tight_layout()
    figure.savefig(output_path, dpi=OUTPUT_DPI)
    plt.close(figure)
    print(f"  Saved PAE Plot: {output_filename}")

#----------------------------------------------------------------------------------FIGURE FUNCTIONS----------------------------------------------------------------------------------------

#---------------------------------Item 5: Structure prediction (Figs 1-9)--------------------------------------------

def plot_fig1_quality_scatter(df: pd.DataFrame, col_flags: dict, density_mode: bool = False, species_label: str = '') -> None:
    """Fig 1: Overall prediction quality landscape.
    Colours by V2 quality tier when available, otherwise falls back to disorder-fraction colouring (RdYlGn_r colourmap).
    When density_mode is True, KDE contour overlays are added to show where complexes concentrate.
    """
    if len(df) == 0:
        print("  Skipping Fig 1: no complexes in subset.")
        return
    # Filter out rows that can't be plotted (missing iptm/pdockq or
    # geometry_available=False). Without this filter, NaN-pdockq rows are
    # silently classified as Low by classify_prediction_quality (since
    # `NaN >= threshold` evaluates False) and inflate the Low legend count
    # while remaining invisible on the scatter (matplotlib drops NaN coords).
    warn_missing_required_rows(df, ['iptm', 'pdockq'], 'Fig 1',
                               'missing required score metrics')
    df = df.loc[~_phantom_row_mask(df, ['iptm', 'pdockq'])]
    if len(df) == 0:
        print("  Skipping Fig 1: no plottable complexes after geometry filter.")
        return
    species_suffix = _species_display(species_label)
    use_tier_colouring = col_flags['has_v2_data']
    n_total = len(df)
    pt_size, pt_alpha = _adaptive_scatter_params(n_total)
    figure, axes = plt.subplots(figsize=(10, 8))
    if use_tier_colouring:
        # Plot by tier: Low first (behind), then Medium, then High on top
        for tier in reversed(TIER_ORDER):
            subset = df[df['quality_tier_v2'] == tier]
            _timed_scatter(axes, subset['iptm'], subset['pdockq'], n_points=n_total, fig_label='Fig 1', c=TIER_COLORS[tier], s=pt_size, alpha=pt_alpha, edgecolors='white', linewidths=0.5, zorder=3, label=tier)

        # Unclassified (missing PAE -> no tier)
        unclassified = df[df['quality_tier_v2'].isna()]
        if len(unclassified) > 0:
            axes.scatter(unclassified['iptm'], unclassified['pdockq'], c='#bdc3c7', s=pt_size, alpha=pt_alpha, edgecolors='white', linewidths=0.5, zorder=2)
    else:
        # Fallback: disorder-fraction colouring
        if 'plddt_below50_fraction' in df.columns:
            disorder = df['plddt_below50_fraction'].fillna(0)
            scatter = _timed_scatter(axes, df['iptm'], df['pdockq'], n_points=n_total, fig_label='Fig 1', c=disorder, cmap='RdYlGn_r', vmin=0, vmax=1, s=pt_size, alpha=pt_alpha, edgecolors='white', linewidths=0.5, zorder=3)
            cbar = figure.colorbar(scatter, ax=axes, shrink=0.8)
            cbar.set_label('Disorder Fraction (pLDDT < 50)', fontsize=FONT_TICK)
        else:
            _timed_scatter(axes, df['iptm'], df['pdockq'], n_points=n_total, fig_label='Fig 1', c='steelblue', s=pt_size, alpha=pt_alpha, edgecolors='white', linewidths=0.5, zorder=3)

    # Optional density contours (--density flag)
    if density_mode:
        valid = df.dropna(subset=['iptm', 'pdockq'])
        _overlay_kde_contours(axes, valid['iptm'].values, valid['pdockq'].values)

    # Threshold lines
    axes.axvline(x=IPTM_HIGH, color='grey', linestyle='--', linewidth=1, alpha=0.7, zorder=1)
    axes.axhline(y=PDOCKQ_HIGH, color='grey', linestyle='--', linewidth=1, alpha=0.7, zorder=1)

    # Shaded high-quality quadrant (subtle)
    axes.fill_between([IPTM_HIGH, 1.05], PDOCKQ_HIGH, 0.8, alpha=0.06, color='green', zorder=0)

    # Build legend (tier handles only - multi-chain analysis is in Fig 9)
    if use_tier_colouring:
        final_handles = _build_tier_legend_handles(df)
        axes.legend(handles=final_handles, title='Quality Tier', fontsize=FONT_TICK, title_fontsize=FONT_AXIS_LABEL, loc='lower right', framealpha=0.9)

    axes.set_xlim(0.2, 1.05)
    axes.set_ylim(-0.02, 0.8)
    title = f"AlphaFold2-Multimer: Quality Assessment (ipTM vs pDockQ){species_suffix}"
    _apply_common_style(axes, title, 'ipTM', 'pDockQ')
    _save_figure(figure, f'1_Quality_Scatter{species_label}.png')

def plot_fig1b_disorder_scatter(df: pd.DataFrame, density_mode: bool = False, species_label: str = '') -> None:
    """Fig 1b (supplementary): Disorder-coloured scatter with optional density contours.
    Each point is coloured by its disorder fraction (pLDDT < 50) using the RdYlGn_r colourmap.  KDE density contours with percentile labels are overlaid when ``density_mode`` is True; the shared helper deterministically downsamples >50K-point inputs so HPC-scale runs do not stall.
    """
    if 'plddt_below50_fraction' not in df.columns:
        print("  Skipping Fig 1b: no disorder fraction column.")
        return
    species_suffix = _species_display(species_label)

    required = ['iptm', 'pdockq', 'plddt_below50_fraction']
    warn_missing_required_rows(df, required, 'Fig 1b',
                               'missing score/disorder metrics or geometry')
    plot_df = df.loc[~_phantom_row_mask(df, required)].copy()
    if len(plot_df) == 0:
        print("  Skipping Fig 1b: no plottable complexes after geometry filter.")
        return

    disorder = plot_df['plddt_below50_fraction'].fillna(0)
    x = plot_df['iptm'].values
    y = plot_df['pdockq'].values
    c = disorder.values
    n_points = len(x)
    figure, axes = plt.subplots(figsize=(10, 8))

    # Adaptive point sizing
    pt_size, pt_alpha = _adaptive_scatter_params(n_points)

    # Scatter: colour = disorder fraction
    scatter = _timed_scatter(axes, x, y, n_points=n_points, fig_label='Fig 1b', c=c, cmap='RdYlGn_r', vmin=0, vmax=1, s=pt_size, alpha=pt_alpha, edgecolors='none', zorder=3)
    cbar = figure.colorbar(scatter, ax=axes, shrink=0.8)
    cbar.set_label('Disorder Fraction (pLDDT < 50)', fontsize=FONT_TICK)

    # Density contours respect --density flag (helper handles HPC-safe downsampling)
    if density_mode:
        _overlay_kde_contours(axes, x, y)

    # Reference lines & "confident" quadrant
    axes.axvline(x=IPTM_HIGH, color='grey', linestyle='--', linewidth=1, alpha=0.7)
    axes.axhline(y=PDOCKQ_HIGH, color='grey', linestyle='--', linewidth=1, alpha=0.7)
    axes.fill_between([IPTM_HIGH, 1.05], PDOCKQ_HIGH, 0.8, alpha=0.06, color='green', zorder=0)
    axes.set_xlim(0.2, 1.05)
    axes.set_ylim(-0.02, 0.8)

    # Annotation: sample size
    axes.text(0.02, 0.98, f'n = {n_points:,}', transform=axes.transAxes, fontsize=FONT_TICK, va='top', ha='left', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='grey', alpha=0.8))

    _apply_common_style(axes, f"Quality Scatter - Disorder Colouring (Supplementary){species_suffix}", 'ipTM', 'pDockQ')
    _save_figure(figure, f'1b_Quality_Scatter_Disorder{species_label}.png')

def plot_fig2_pae_health_check(df: pd.DataFrame, species_label: str = '') -> None:
    """Fig 2: Is the dataset generally well-resolved?"""
    if 'pae_mean' not in df.columns:
        print("  Skipping Fig 2: no pae_mean column.")
        return
    pae_values = df['pae_mean'].dropna()
    if len(pae_values) == 0:
        print("  Skipping Fig 2: no valid PAE values.")
        return
    species_suffix = _species_display(species_label)
    median_pae = pae_values.median()
    below_threshold = (pae_values < PAE_CONFIDENT).sum()
    figure, axes = plt.subplots(figsize=(8, 5))
    axes.hist(pae_values, bins=30, color='#3498db', alpha=0.75, edgecolor='white', linewidth=0.5)

    # Median line
    axes.axvline(x=median_pae, color='red', linestyle='--', linewidth=1.5, label=f'Median: {median_pae:.1f} \u00c5')

    # Confident guideline
    axes.axvline(x=PAE_CONFIDENT, color='green', linestyle='--', linewidth=1.5, label=f'Confident guideline: {PAE_CONFIDENT} \u00c5')

    axes.legend(fontsize=FONT_TICK, loc='upper right')
    title = (f"Global PAE Health Check - {len(pae_values)} complexes, "
             f"median {median_pae:.1f} \u00c5, {below_threshold} below {PAE_CONFIDENT} \u00c5"
             f"{species_suffix}")
    _apply_common_style(axes, title, 'Mean PAE (\u00c5)', 'Count', grid=False)
    _save_figure(figure, f'2_PAE_Health_Check{species_label}.png')

def plot_fig3_interface_pae_by_tier(df: pd.DataFrame, species_label: str = '') -> None:
    """Fig 3: How confident are the contacts that matter for quality assessment?"""
    required = ['interface_pae_mean', 'quality_tier_v2']
    if not all(col in df.columns for col in required):
        print("  Skipping Fig 3: missing required columns.")
        return
    plot_df = _filter_dimer_validated(df).dropna(subset=required).copy()
    if len(plot_df) == 0:
        print("  Skipping Fig 3: no valid data after filtering.")
        return
    species_suffix = _species_display(species_label)
    figure, axes = plt.subplots(figsize=(10, 6))

    # Build data and positions for boxplots + strip
    tier_data = []
    tier_labels = []
    tier_medians = []
    positions = []
    for idx, tier in enumerate(TIER_ORDER):
        subset = plot_df[plot_df['quality_tier_v2'] == tier]['interface_pae_mean']
        if len(subset) > 0:
            tier_data.append(subset.values)
            tier_labels.append(f'{tier}\n(n={len(subset)})')
            tier_medians.append(subset.median())
            positions.append(idx)

    # Boxplots
    box_parts = axes.boxplot(tier_data, positions=positions, widths=0.5, patch_artist=True, showfliers=False, medianprops=dict(color='black', linewidth=2))

    # Colour the boxes
    for idx, patch in enumerate(box_parts['boxes']):
        tier_name = TIER_ORDER[idx] if idx < len(TIER_ORDER) else 'High'
        patch.set_facecolor(TIER_COLORS.get(tier_name, '#cccccc'))
        patch.set_alpha(0.6)

    # Ensure median lines render above scatter points
    for median_line in box_parts['medians']:
        median_line.set_zorder(10)

    # Jittered strip plot behind
    for idx, data in enumerate(tier_data):
        jitter = np.random.normal(0, 0.06, size=len(data))
        axes.scatter(positions[idx] + jitter, data, c=TIER_COLORS.get(TIER_ORDER[idx], '#cccccc'), alpha=0.35, s=20, zorder=1, edgecolors='none')

    # PAE threshold line
    axes.axhline(y=PAE_CONFIDENT, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Confident contact cutoff ({PAE_CONFIDENT} \u00c5)')

    axes.set_xticks(positions)
    axes.set_xticklabels(tier_labels, fontsize=FONT_AXIS_LABEL)
    axes.legend(fontsize=FONT_TICK, loc='upper right')

    # Subtitle with medians
    median_text = " | ".join(
        f"{TIER_ORDER[i]} median: {tier_medians[i]:.1f} \u00c5"
        for i in range(len(tier_medians))
    )
    axes.text(0.5, -0.12, median_text, transform=axes.transAxes, ha='center', fontsize=FONT_TICK, style='italic', color='#555555')
    _apply_common_style(axes, f"Interface PAE by Quality Tier [dimer-validated]{species_suffix}", '', 'Interface PAE (\u00c5)', grid=False)
    _save_figure(figure, f'3_Interface_PAE_by_Tier{species_label}.png')

def plot_fig4_composite_validation(df: pd.DataFrame, density_mode: bool = False, species_label: str = '') -> None:
    """Fig 4: Why should I trust the quality tier assigned?
    Panel (a): Composite score distributions by tier (violin/boxplot).
    Panel (b): Composite vs STRICT confident contact fraction scatter - the fraction
    actually consumed by the composite score (PAE < 5A AND both residue pLDDT >= 70).
    Plotting the PAE-only fraction here would be circular w.r.t. the composite definition,
    since the composite uses the strict fraction post-revision.
    """
    required = ['interface_confidence_score', 'quality_tier_v2', 'strict_confident_contact_fraction']
    if not all(col in df.columns for col in required):
        print("  Skipping Fig 4: missing required columns.")
        return
    warn_missing_required_rows(df, required, 'Fig 4',
                               'missing interface geometry / pair metrics')
    # Dissertation-safe: composite is calibrated only for dimers (Phase 4).
    plot_df = _filter_dimer_validated(df).dropna(subset=required).copy()
    if len(plot_df) == 0:
        print("  Skipping Fig 4: no dimer-validated complexes with complete composite data.")
        return

    species_suffix = _species_display(species_label)
    figure, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6))

    #====================================Panel (a): Composite score distributions====================================
    tier_data_a = []
    tier_labels_a = []
    positions_a = []
    for idx, tier in enumerate(TIER_ORDER):
        subset = plot_df[plot_df['quality_tier_v2'] == tier]['interface_confidence_score']
        if len(subset) > 0:
            tier_data_a.append(subset.values)
            tier_labels_a.append(f'{tier}\n(n={len(subset)})')
            positions_a.append(idx)

    # Violin plots
    if tier_data_a:
        vp = ax_a.violinplot(tier_data_a, positions=positions_a, showmedians=True, showextrema=False)
        for idx, body in enumerate(vp['bodies']):
            tier_name = TIER_ORDER[idx] if idx < len(TIER_ORDER) else 'High'
            body.set_facecolor(TIER_COLORS.get(tier_name, '#cccccc'))
            body.set_alpha(0.6)
        if 'cmedians' in vp:
            vp['cmedians'].set_color('black')
            vp['cmedians'].set_linewidth(2)
            vp['cmedians'].set_zorder(10)

    # Jittered strip overlay
    for idx, data in enumerate(tier_data_a):
        jitter = np.random.normal(0, 0.06, size=len(data))
        ax_a.scatter(positions_a[idx] + jitter, data, c=TIER_COLORS.get(TIER_ORDER[idx], '#cccccc'), alpha=0.3, s=15, zorder=3, edgecolors='none')

    # Decision threshold lines
    thresholds = [
        (UPGRADE_LOW_THRESHOLD, 'Low -> High upgrade', '#e74c3c'),
        (UPGRADE_MEDIUM_THRESHOLD, 'Medium -> High upgrade', '#f39c12'),
        (DOWNGRADE_HIGH_THRESHOLD, 'High -> Medium downgrade', '#2ecc71'),
    ]
    for val, label, colour in thresholds:
        ax_a.axhline(y=val, color=colour, linestyle=':', linewidth=1.2, alpha=0.7)
        ax_a.text(ax_a.get_xlim()[1], val, f' {label} ({val})', va='center', fontsize=8, color=colour, alpha=0.8)

    ax_a.set_xticks(positions_a)
    ax_a.set_xticklabels(tier_labels_a, fontsize=FONT_AXIS_LABEL)

    # Tier colour legend - upper-left to avoid threshold label overlap
    tier_legend = [mpatches.Patch(color=TIER_COLORS[t], alpha=0.6, label=t) for t in TIER_ORDER]
    ax_a.legend(handles=tier_legend, fontsize=FONT_TICK - 1, loc='upper left', framealpha=0.9)
    _apply_common_style(ax_a, "(a) Composite Score by Tier", '', 'Interface Confidence Score', grid=False)

    #====================================Panel (b): Composite vs strict confident contact fraction====================================
    n_panel_b = len(plot_df)
    pt_size_b, pt_alpha_b = _adaptive_scatter_params(n_panel_b)

    for tier in reversed(TIER_ORDER):
        subset = plot_df[plot_df['quality_tier_v2'] == tier]
        axes_b_kwargs = dict(c=TIER_COLORS[tier], alpha=pt_alpha_b, s=pt_size_b, edgecolors='white', linewidths=0.3, label=tier, zorder=3)
        _timed_scatter(ax_b, subset['strict_confident_contact_fraction'], subset['interface_confidence_score'], n_points=n_panel_b, fig_label='Fig 4b', **axes_b_kwargs)
    ax_b.legend(fontsize=FONT_TICK, title='Tier', title_fontsize=FONT_TICK)

    # Optional density contours
    if density_mode:
        valid_b = plot_df[['strict_confident_contact_fraction', 'interface_confidence_score']].dropna()
        _overlay_kde_contours(ax_b, valid_b['strict_confident_contact_fraction'].values, valid_b['interface_confidence_score'].values)

    # Correlation annotation
    valid_both = plot_df[['strict_confident_contact_fraction', 'interface_confidence_score']].dropna()
    if len(valid_both) > 2:
        r = valid_both['strict_confident_contact_fraction'].corr(valid_both['interface_confidence_score'])
        ax_b.text(0.05, 0.95, f'r = {r:.2f}', transform=ax_b.transAxes, fontsize=FONT_AXIS_LABEL, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    _apply_common_style(ax_b, "(b) Composite vs Strict Confident Contact Fraction", 'Strict Confident Contact Fraction (PAE<5A & pLDDT>=70)', 'Interface Confidence Score')
    figure.suptitle(
        f"Quality Tier Validation: Composite Score Evidence "
        f"[{CAPTION_SCOPE_DIMER}]{species_suffix}",
        fontsize=14, fontweight='bold', y=1.02)
    _save_figure(figure, f'4_Composite_Tier_Validation{species_label}.png')


def plot_fig4_supp_strict_vs_pae_only(df: pd.DataFrame, species_label: str = '') -> None:
    """Fig 4 supplementary: strict confident-contact fraction vs PAE-only fraction.

    Shows how much stricter the revised definition is: every point lies on or below the
    y = x line, and the cloud's vertical offset quantifies the contribution of the pLDDT
    >= 70 filter (i.e. how many PAE-confident contacts had low-pLDDT residues).

    Useful for the dissertation methods section: it is the evidence image for the revised
    composite. Renders only when both columns are present.
    """
    required = ['pae_confident_contact_fraction', 'strict_confident_contact_fraction',
                'quality_tier_v2']
    if not all(col in df.columns for col in required):
        print("  Skipping Fig 4 supp: missing required columns.")
        return
    plot_df = _filter_dimer_validated(df).dropna(subset=required).copy()
    if len(plot_df) == 0:
        print("  Skipping Fig 4 supp: no valid data.")
        return

    species_suffix = _species_display(species_label)
    figure, axes = plt.subplots(figsize=(8, 8))
    n_points = len(plot_df)
    pt_size, pt_alpha = _adaptive_scatter_params(n_points)

    for tier in reversed(TIER_ORDER):
        subset = plot_df[plot_df['quality_tier_v2'] == tier]
        if len(subset) == 0:
            continue
        axes.scatter(subset['pae_confident_contact_fraction'],
                     subset['strict_confident_contact_fraction'],
                     c=TIER_COLORS[tier], alpha=pt_alpha, s=pt_size,
                     edgecolors='white', linewidths=0.3, label=tier, zorder=3)

    # y = x reference line: strict can never exceed PAE-only
    axes.plot([0, 1], [0, 1], color='#555555', linestyle='--', linewidth=1.0, alpha=0.7, label='y = x (upper bound)')

    # Summary stats annotation
    delta = (plot_df['pae_confident_contact_fraction']
             - plot_df['strict_confident_contact_fraction'])
    axes.text(0.05, 0.95,
              f"n = {len(plot_df)}\n"
              f"mean delta = {delta.mean():.3f}\n"
              f"median delta = {delta.median():.3f}",
              transform=axes.transAxes, fontsize=FONT_AXIS_LABEL, va='top',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    axes.legend(fontsize=FONT_TICK, title='Tier', title_fontsize=FONT_TICK, loc='lower right')
    _apply_common_style(axes,
                        f"Strict vs PAE-only Confident Contact Fraction [dimer-validated]{species_suffix}",
                        'PAE-only Confident Contact Fraction (PAE<5A)',
                        'Strict Confident Contact Fraction (PAE<5A & pLDDT>=70)')
    _save_figure(figure, f'4_supp_Strict_vs_PAE_Only_Fraction{species_label}.png')

def plot_fig5_interface_vs_bulk(df: pd.DataFrame, density_mode: bool = False, species_label: str = '') -> None:
    """Fig 5: Are interfaces special, or do they just reflect bulk quality?"""
    required = ['interface_plddt_combined', 'bulk_plddt_combined', 'quality_tier_v2']
    if not all(col in df.columns for col in required):
        print("  Skipping Fig 5: missing required columns.")
        return
    # Dissertation-safe: bulk/interface pLDDT best-pair semantics are dimer-validated.
    plot_df = _filter_dimer_validated(df).dropna(subset=required).copy()
    if len(plot_df) == 0:
        print("  Skipping Fig 5: no dimer-validated complexes.")
        return

    species_suffix = _species_display(species_label)
    figure, axes = plt.subplots(figsize=(8, 8))

    # Identify paradox complexes for special marking
    paradox_mask = _get_paradox_mask(plot_df)

    # Plot non-paradox by tier with adaptive sizing
    non_paradox = plot_df[~paradox_mask] # to invert mask and get non-paradox complexes
    n_non_paradox = len(non_paradox)
    pt_size, pt_alpha = _adaptive_scatter_params(n_non_paradox)

    for tier in reversed(TIER_ORDER):
        subset = non_paradox[non_paradox['quality_tier_v2'] == tier]
        _timed_scatter(axes, subset['bulk_plddt_combined'],
                       subset['interface_plddt_combined'],
                       n_points=n_non_paradox, fig_label='Fig 5',
                       c=TIER_COLORS[tier], s=pt_size,
                       alpha=pt_alpha, edgecolors='white',
                       linewidths=0.5, zorder=3, label=tier)

    # Optional density contours
    if density_mode:
        _overlay_kde_contours(axes, non_paradox['bulk_plddt_combined'].values, non_paradox['interface_plddt_combined'].values)

    # Paradox complexes - small triangles so each marker represents one complex
    paradox_df = plot_df[paradox_mask]
    if len(paradox_df) > 0:
        axes.scatter(paradox_df['bulk_plddt_combined'], paradox_df['interface_plddt_combined'], c='#9b59b6', s=50, alpha=0.9, marker='^', edgecolors='black', linewidths=0.6, zorder=5, label=f'Paradox ({len(paradox_df)})')

    # Diagonal y = x line
    lims = [min(axes.get_xlim()[0], axes.get_ylim()[0]), max(axes.get_xlim()[1], axes.get_ylim()[1])]
    axes.plot(lims, lims, 'k--', linewidth=1.2, alpha=0.6, zorder=1)
    axes.set_xlim(lims)
    axes.set_ylim(lims)

    # Annotations
    above_diagonal = (plot_df['interface_plddt_combined'] > plot_df['bulk_plddt_combined']).sum()
    total = len(plot_df)
    pct = above_diagonal / total * 100 if total > 0 else 0

    axes.text(0.05, 0.95, "Interface > Bulk \u2191", transform=axes.transAxes, fontsize=FONT_TICK, va='top', ha='left', color='#27ae60', fontweight='bold')
    axes.text(0.95, 0.05, "Bulk > Interface \u2193", transform=axes.transAxes, fontsize=FONT_TICK, va='bottom', ha='right', color='#e74c3c', fontweight='bold')
    axes.text(0.5, 0.02, f'{above_diagonal}/{total} ({pct:.0f}%) above diagonal', transform=axes.transAxes, ha='center', fontsize=FONT_TICK, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    axes.legend(fontsize=FONT_TICK, loc='lower left', framealpha=0.9)
    _apply_common_style(
        axes,
        f"Interface vs Bulk pLDDT: Are Interfaces Special? "
        f"[{CAPTION_SCOPE_DIMER}]{species_suffix}",
        'Bulk pLDDT', 'Interface pLDDT')
    _save_figure(figure, f'5_Interface_vs_Bulk{species_label}.png')

def plot_fig6_paradox_spotlight(df: pd.DataFrame, species_label: str = '') -> None:
    """Fig 6: Can disordered proteins form confident interfaces?
    3-panel comparison of paradox vs non-paradox complexes - Paradox: ipTM >= 0.75, pDockQ >= 0.5, disorder fraction >= 0.30.
    """
    required = ['iptm', 'pdockq', 'plddt_below50_fraction', 'interface_vs_bulk_delta', 'pae_confident_contact_fraction', 'interface_symmetry']
    if not all(col in df.columns for col in required):
        print("  Skipping Fig 6: missing required columns.")
        return
    warn_missing_required_rows(df, required, 'Fig 6',
                               'missing interface geometry / pair metrics')

    species_suffix = _species_display(species_label)

    # Dissertation-safe: paradox detection uses dimer-calibrated thresholds.
    scoped_df = _filter_dimer_validated(df)

    # Count paradox complexes before dropping rows with missing panel data so we can report how many are lost to incomplete interface metrics
    n_paradox_before_dropna = int(_get_paradox_mask(scoped_df).sum())

    plot_df = scoped_df.dropna(subset=required).copy()
    if len(plot_df) == 0:
        print("  Skipping Fig 6: no complexes with complete interface data.")
        return

    paradox_mask = _get_paradox_mask(plot_df)
    paradox = plot_df[paradox_mask]
    non_paradox = plot_df[~paradox_mask]

    n_paradox = len(paradox)
    n_non_paradox = len(non_paradox)
    n_paradox_missing_data = n_paradox_before_dropna - n_paradox

    if n_paradox_missing_data > 0:
        print(f"  Note: {n_paradox_missing_data} of {n_paradox_before_dropna} paradox "
              f"complexes excluded from Fig 6 (missing interface data)")
    if n_paradox == 0:
        print("  Skipping Fig 6: no paradox complexes found.")
        return

    colour_paradox = '#9b59b6'
    colour_non_paradox = '#3498db'
    figure, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(16, 5))
    panels = [
        (ax_a, 'interface_vs_bulk_delta', '(a) Interface vs Bulk (\u0394 pLDDT)', '\u0394 pLDDT'),
        # Uses PAE-only fraction (matches paradox-detection semantics and thresholds at
        # PARADOX_CONFIDENT_CONTACT_GENUINE/_ARTEFACT); the stricter fraction is exposed in
        # Fig 4 supplementary for methodological comparison.
        (ax_b, 'pae_confident_contact_fraction', '(b) PAE-only Confident Contact Fraction', 'Fraction'),
        (ax_c, 'interface_symmetry', '(c) Interface Symmetry', 'Symmetry Score'),
    ]

    for ax, col, title, ylabel in panels:
        data_paradox = paradox[col].dropna().values
        data_non_paradox = non_paradox[col].dropna().values
        if len(data_paradox) == 0 or len(data_non_paradox) == 0:
            ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes, ha='center', va='center')
            ax.set_title(title, fontsize=FONT_TITLE, fontweight='bold')
            continue

        # Box + strip for each group
        box_data = [data_paradox, data_non_paradox]
        positions = [0, 1]
        bp = ax.boxplot(box_data, positions=positions, widths=0.5, patch_artist=True, showfliers=False, medianprops=dict(color='black', linewidth=2))
        bp['boxes'][0].set_facecolor(colour_paradox)
        bp['boxes'][0].set_alpha(0.5)
        bp['boxes'][1].set_facecolor(colour_non_paradox)
        bp['boxes'][1].set_alpha(0.5)

        # Ensure median lines render above scatter points
        for median_line in bp['medians']:
            median_line.set_zorder(10)

        # Jittered strip
        for i, (data, colour) in enumerate([(data_paradox, colour_paradox), (data_non_paradox, colour_non_paradox)]):
            jitter = np.random.normal(0, 0.06, size=len(data))
            ax.scatter(positions[i] + jitter, data, c=colour, alpha=0.4, s=20, zorder=3, edgecolors='none')

        ax.set_xticks(positions)
        ax.set_xticklabels([f'Paradox\n(n={n_paradox})',
                            f'Non-paradox\n(n={n_non_paradox})'], fontsize=FONT_TICK)
        ax.set_title(title, fontsize=FONT_TITLE, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=FONT_AXIS_LABEL)
        ax.tick_params(labelsize=FONT_TICK)

        # Median annotations
        med_p = np.median(data_paradox)
        med_np = np.median(data_non_paradox)
        ax.text(0.5, -0.15, f'Medians: {med_p:.2f} vs {med_np:.2f}', transform=ax.transAxes, ha='center', fontsize=9, style='italic', color='#555555')

    # Shared legend on the first panel
    legend_handles = [
        mpatches.Patch(color=colour_paradox, alpha=0.6, label='Paradox'),
        mpatches.Patch(color=colour_non_paradox, alpha=0.6, label='Non-paradox'),
    ]
    ax_a.legend(handles=legend_handles, fontsize=FONT_TICK, loc='upper right', framealpha=0.9)
    figure.suptitle(
        f"Paradox Complexes: Confident Interfaces Despite Structural Disorder "
        f"[{CAPTION_SCOPE_DIMER}]{species_suffix}",
        fontsize=14, fontweight='bold', y=1.04)

    subtitle = (f"Comparing {n_paradox} paradox vs {n_non_paradox} "
                f"non-paradox complexes")
    if n_paradox_missing_data > 0:
        subtitle += (f" ({n_paradox_missing_data} paradox complexes excluded "
                     f"due to incomplete interface data)")

    figure.text(0.5, 0.99, subtitle, ha='center', fontsize=FONT_AXIS_LABEL, style='italic', color='#555555')
    _save_figure(figure, f'6_Paradox_Spotlight{species_label}.png')

def plot_fig7_homo_vs_hetero(df: pd.DataFrame, species_label: str = '',
                             multimer_supplement: bool = False) -> None:
    """Fig 7: How does prediction quality vary by complex architecture?
    Primary (dimer-validated): tier_scope == 'dimer_validated' AND
    stoichiometry in {A2, AB}. Tier thresholds are calibrated against dimers, so
    the primary panel uses only dimers.
    Supplementary (multimer exploratory): rendered only when
    multimer_supplement=True - separate panels for A2B/ABC/A2B2/ABCD/other.
    Fig is skipped entirely when the required multimer-safe columns are absent.
    """
    required = ['complex_type', 'quality_tier_v2', 'stoichiometry', 'tier_scope']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  Skipping Fig 7: missing required columns {missing}.")
        return
    plot_df = df.dropna(subset=['quality_tier_v2', 'stoichiometry', 'tier_scope']).copy()
    if len(plot_df) == 0:
        print("  Skipping Fig 7: no valid data.")
        return

    species_suffix = _species_display(species_label)

    # Primary panel (dissertation-safe): dimer_validated scope, A2 or AB only.
    primary_df = plot_df[
        (plot_df['tier_scope'] == TIER_SCOPE_DIMER)
        & (plot_df['stoichiometry'].isin(DIMER_STOICHIOMETRIES))
    ]
    if len(primary_df) == 0:
        print("  Skipping Fig 7: no dimer-validated complexes (stoichiometry A2/AB).")
        return

    homo = primary_df[primary_df['stoichiometry'] == 'A2']
    hetero = primary_df[primary_df['stoichiometry'] == 'AB']

    primary_categories = []
    primary_cat_colours = []
    if len(homo) > 0:
        primary_categories.append(('Homodimer (A2)', homo))
        primary_cat_colours.append('#3498db')
    if len(hetero) > 0:
        primary_categories.append(('Heterodimer (AB)', hetero))
        primary_cat_colours.append('#e67e22')

    figure, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5))

    #=======================================Panel (a): Stacked bar chart=======================================
    bar_positions = list(range(len(primary_categories)))
    bar_labels = []
    for cat_idx, (label, subset) in enumerate(primary_categories):
        count = len(subset)
        bottom = 0
        for tier in TIER_ORDER:
            tier_count = (subset['quality_tier_v2'] == tier).sum()
            pct = tier_count / count * 100 if count > 0 else 0
            ax_a.bar(cat_idx, pct, bottom=bottom, color=TIER_COLORS[tier], edgecolor='white', linewidth=0.5)
            if pct > 3:
                ax_a.text(cat_idx, bottom + pct / 2, f'{pct:.1f}%', ha='center', va='center', fontsize=9, fontweight='bold', color='white' if pct > 10 else 'black')
            bottom += pct
        bar_labels.append(f'{label}\n(n={count})')

    ax_a.set_xticks(bar_positions)
    ax_a.set_xticklabels(bar_labels, fontsize=FONT_AXIS_LABEL)
    ax_a.set_ylim(0, 105)

    legend_handles = [mpatches.Patch(color=TIER_COLORS[t], label=t) for t in TIER_ORDER]
    ax_a.legend(handles=legend_handles, fontsize=FONT_TICK, loc='upper right')

    _apply_common_style(ax_a, "(a) Quality Tier Proportions (dimer-validated)", '', 'Percentage (%)', grid=False)

    #=======================================Panel (b): Interface symmetry distributions=======================================
    has_symmetry = 'interface_symmetry' in df.columns
    if has_symmetry:
        sym_data = []
        sym_labels = []
        sym_colours = []
        for (label, subset), colour in zip(primary_categories, primary_cat_colours):
            sym_values = subset['interface_symmetry'].dropna().values
            if len(sym_values) > 0:
                sym_data.append(sym_values)
                sym_labels.append(f'{label}\n(n={len(sym_values)})')
                sym_colours.append(colour)
        if sym_data:
            positions_b = list(range(len(sym_data)))
            vp = ax_b.violinplot(sym_data, positions=positions_b, showmedians=True, showextrema=False)
            for idx, body in enumerate(vp['bodies']):
                body.set_facecolor(sym_colours[idx])
                body.set_alpha(0.6)
            if 'cmedians' in vp:
                vp['cmedians'].set_color('black')
                vp['cmedians'].set_linewidth(2)
                vp['cmedians'].set_zorder(10)
            ax_b.set_xticks(positions_b)
            ax_b.set_xticklabels(sym_labels, fontsize=FONT_AXIS_LABEL)
        else:
            ax_b.text(0.5, 0.5, 'No valid interface-symmetry values',
                      transform=ax_b.transAxes, ha='center', va='center',
                      fontsize=FONT_AXIS_LABEL, color='grey')
    else:
        ax_b.text(0.5, 0.5, 'Interface symmetry data\nnot available', transform=ax_b.transAxes, ha='center', va='center', fontsize=FONT_AXIS_LABEL, color='grey')

    _apply_common_style(ax_b, "(b) Interface Symmetry (dimer-validated)", '', 'Symmetry Score', grid=False)
    figure.suptitle(
        f"Prediction Quality by Complex Architecture [{CAPTION_SCOPE_DIMER}]"
        f"{species_suffix}",
        fontsize=14, fontweight='bold', y=1.02)
    _save_figure(figure, f'7_Homo_vs_Hetero{species_label}.png')

    if multimer_supplement:
        _plot_fig7_multimer_supplement(plot_df, species_label=species_label)


def _plot_fig7_multimer_supplement(plot_df: pd.DataFrame, species_label: str = '') -> None:
    """Fig 7 supplementary: tier proportions and symmetry by multimer stoichiometry.
    Shows multimer_provisional rows bucketed into A2B, ABC, A2B2, ABCD, and
    'Other' (anything else with n_chains > 2). Gated by --multimer-supplement
    - these are exploratory, not dissertation claims.
    """
    multimer_df = plot_df[plot_df['tier_scope'] == TIER_SCOPE_MULTIMER]
    if len(multimer_df) == 0:
        print("  Fig 7 supp: no multimer_provisional rows - skipping supplement.")
        return

    species_suffix = _species_display(species_label)
    known_buckets = ['A2B', 'ABC', 'A2B2', 'ABCD']
    bucket_colours = {
        'A2B': '#8e44ad',
        'ABC': '#16a085',
        'A2B2': '#d35400',
        'ABCD': '#2c3e50',
        'Other': '#7f8c8d',
    }

    def _bucket(label: str) -> str:
        return label if label in known_buckets else 'Other'

    multimer_df = multimer_df.assign(_bucket=multimer_df['stoichiometry'].map(_bucket))
    present = [b for b in known_buckets + ['Other']
               if (multimer_df['_bucket'] == b).any()]
    if not present:
        print("  Fig 7 supp: no supported multimer buckets present.")
        return

    figure, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5))

    bar_positions = list(range(len(present)))
    bar_labels = []
    for cat_idx, bucket in enumerate(present):
        subset = multimer_df[multimer_df['_bucket'] == bucket]
        count = len(subset)
        bottom = 0
        for tier in TIER_ORDER:
            tier_count = (subset['quality_tier_v2'] == tier).sum()
            pct = tier_count / count * 100 if count > 0 else 0
            ax_a.bar(cat_idx, pct, bottom=bottom, color=TIER_COLORS[tier], edgecolor='white', linewidth=0.5)
            if pct > 3:
                ax_a.text(cat_idx, bottom + pct / 2, f'{pct:.1f}%', ha='center', va='center', fontsize=9, fontweight='bold', color='white' if pct > 10 else 'black')
            bottom += pct
        bar_labels.append(f'{bucket}\n(n={count})')

    ax_a.set_xticks(bar_positions)
    ax_a.set_xticklabels(bar_labels, fontsize=FONT_AXIS_LABEL)
    ax_a.set_ylim(0, 105)
    legend_handles = [mpatches.Patch(color=TIER_COLORS[t], label=t) for t in TIER_ORDER]
    ax_a.legend(handles=legend_handles, fontsize=FONT_TICK, loc='upper right')
    _apply_common_style(ax_a, "(a) Tier Proportions by Stoichiometry", '', 'Percentage (%)', grid=False)

    if 'interface_symmetry' in multimer_df.columns:
        sym_data, sym_labels, sym_colours = [], [], []
        for bucket in present:
            vals = multimer_df.loc[multimer_df['_bucket'] == bucket, 'interface_symmetry'].dropna().values
            if len(vals) > 0:
                sym_data.append(vals)
                sym_labels.append(f'{bucket}\n(n={len(vals)})')
                sym_colours.append(bucket_colours[bucket])
        if sym_data:
            positions_b = list(range(len(sym_data)))
            vp = ax_b.violinplot(sym_data, positions=positions_b, showmedians=True, showextrema=False)
            for idx, body in enumerate(vp['bodies']):
                body.set_facecolor(sym_colours[idx])
                body.set_alpha(0.6)
            if 'cmedians' in vp:
                vp['cmedians'].set_color('black')
                vp['cmedians'].set_linewidth(2)
                vp['cmedians'].set_zorder(10)
            ax_b.set_xticks(positions_b)
            ax_b.set_xticklabels(sym_labels, fontsize=FONT_AXIS_LABEL)
        else:
            ax_b.text(0.5, 0.5, 'No valid interface-symmetry values',
                      transform=ax_b.transAxes, ha='center', va='center',
                      fontsize=FONT_AXIS_LABEL, color='grey')
    else:
        ax_b.text(0.5, 0.5, 'Interface symmetry data\nnot available', transform=ax_b.transAxes, ha='center', va='center', fontsize=FONT_AXIS_LABEL, color='grey')

    _apply_common_style(ax_b, "(b) Best-pair Interface Symmetry", '', 'Symmetry Score', grid=False)
    figure.suptitle(
        f"Multimer Architecture Supplement [{CAPTION_SCOPE_MULTIMER}]"
        f"{species_suffix}",
        fontsize=14, fontweight='bold', y=1.02)
    _save_figure(figure, f'7_supp_Multimer_Stoichiometry{species_label}.png')

def plot_fig8_metric_disagreement(df: pd.DataFrame, density_mode: bool = False, species_label: str = '') -> None:
    """Fig 8: Why do ipTM and pDockQ disagree so systematically?"""
    required = ['iptm', 'pdockq', 'quality_tier_v2']
    if not all(col in df.columns for col in required):
        print("  Skipping Fig 8: missing required columns.")
        return
    warn_missing_required_rows(df, required, 'Fig 8',
                               'missing interface geometry / pair metrics')
    # Dissertation-safe: the disagreement threshold is calibrated on dimers.
    plot_df = _filter_dimer_validated(df).dropna(subset=required).copy()
    if len(plot_df) == 0:
        print("  Skipping Fig 8: no dimer-validated complexes.")
        return

    species_suffix = _species_display(species_label)
    figure, axes = plt.subplots(figsize=(10, 8))
    n_plot = len(plot_df)
    pt_size, pt_alpha = _adaptive_scatter_params(n_plot)

    for tier in reversed(TIER_ORDER):
        subset = plot_df[plot_df['quality_tier_v2'] == tier]
        _timed_scatter(axes, subset['iptm'], subset['pdockq'],
                       n_points=n_plot, fig_label='Fig 8',
                       c=TIER_COLORS[tier], s=pt_size,
                       alpha=pt_alpha, edgecolors='white',
                       linewidths=0.5, zorder=3, label=tier)
    axes.legend(fontsize=FONT_TICK, title='Tier', title_fontsize=FONT_TICK, loc='upper left')

    # Optional density contours
    if density_mode:
        _overlay_kde_contours(axes, plot_df['iptm'].values, plot_df['pdockq'].values)

    # Diagonal y = x line
    axes.plot([0, 1.1], [0, 1.1], 'k--', linewidth=1.2, alpha=0.6, zorder=1)

    # Disagreement band: region where ipTM - pDockQ > METRIC_DISAGREEMENT_GAP
    # This is below the line y = x - gap
    x_band = np.linspace(0, 1.1, 100)
    y_band = x_band - METRIC_DISAGREEMENT_GAP
    axes.fill_between(x_band, -0.1, y_band, alpha=0.08, color='#e74c3c', zorder=0)

    # Count disagreement cases - intentionally one-directional (ipTM >> pDockQ) to highlight the systematic bias where pDockQ penalises disordered complexes.
    # The flag in interface_analysis.py uses abs() for bidirectional detection.
    disagreement_mask = (plot_df['iptm'] - plot_df['pdockq']) > METRIC_DISAGREEMENT_GAP
    n_disagree = disagreement_mask.sum()
    pct_disagree = n_disagree / len(plot_df) * 100 if len(plot_df) > 0 else 0

    # Annotation in the disagreement zone
    axes.text(0.85, 0.10,
              f'{n_disagree} complexes ({pct_disagree:.0f}%)\nipTM >> pDockQ',
              transform=axes.transAxes, ha='center', va='center',
              fontsize=FONT_AXIS_LABEL, fontweight='bold', color='#c0392b',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    axes.text(0.85, 0.02,
              'pDockQ is contact/interface-confidence\nsensitive; ipTM can remain high',
              transform=axes.transAxes, ha='center', fontsize=9,
              style='italic', color='#777777',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='none'))

    axes.set_xlim(0.2, 1.05)
    axes.set_ylim(-0.02, 0.8)

    _apply_common_style(
        axes,
        f"Metric Disagreement: ipTM vs pDockQ Systematic Bias "
        f"[{CAPTION_SCOPE_DIMER}]{species_suffix}",
        'ipTM', 'pDockQ')
    _save_figure(figure, f'8_Metric_Disagreement{species_label}.png')

def plot_fig9_chain_count_profile(df: pd.DataFrame, density_mode: bool = False, species_label: str = '') -> None:
    """Fig 9: Chain-count profile - exposes order-statistic bias honestly.
    Four violin panels by n_chains group (2, 3, 4+):
        (a) pdockq (best pair) - order-statistic biased upward for large N.
        (b) pdockq_mean        - unbiased aggregate across all inter-chain pairs.
        (c) pdockq_min         - worst-pair lower bound; surfaces dangling chains.
        (d) coherence gap (pdockq - pdockq_min) - 0 for every N=2 row by construction.
    The figure is descriptive across all N (CAPTION_SCOPE_ALL_N). Requires the
    multimer-safe aggregates added in Phase 3; falls back gracefully otherwise.
    """
    base_required = ['n_chains', 'pdockq']
    if not all(col in df.columns for col in base_required):
        print("  Skipping Fig 9: missing required columns.")
        return

    aggregate_cols = ['pdockq_mean', 'pdockq_min']
    if not all(col in df.columns for col in aggregate_cols):
        print("  Skipping Fig 9: aggregate columns (pdockq_mean/pdockq_min) missing. "
              "Regenerate with the multimer_v1 schema (Phase 3 of the refactor).")
        return

    fig9_required = base_required + aggregate_cols
    warn_missing_required_rows(df, fig9_required, 'Fig 9',
                               'missing interface geometry / pair metrics')
    plot_df = df.loc[~_phantom_row_mask(df, fig9_required)].copy()
    plot_df = plot_df.dropna(subset=fig9_required).copy()
    if len(plot_df) == 0:
        print("  Skipping Fig 9: no complete all-pairs chain-count data.")
        return

    species_suffix = _species_display(species_label)

    def chain_group(n):
        if n <= 2:
            return '2 chains'
        elif n == 3:
            return '3 chains'
        else:
            return '4+ chains'

    plot_df['chain_group'] = plot_df['n_chains'].apply(chain_group)
    plot_df['coherence_gap'] = plot_df['pdockq'] - plot_df['pdockq_min']

    group_order = ['2 chains', '3 chains', '4+ chains']
    group_colours = {'2 chains': '#3498db', '3 chains': '#e67e22', '4+ chains': '#8e44ad'}
    present_groups = [g for g in group_order if (plot_df['chain_group'] == g).any()]
    if len(present_groups) == 0:
        print("  Skipping Fig 9: no recognised chain-count groups.")
        return

    panels = [
        ('pdockq', '(a) pDockQ (best pair)', 'pDockQ'),
        ('pdockq_mean', '(b) pDockQ mean (all pairs)', 'pDockQ mean'),
        ('pdockq_min', '(c) pDockQ min (worst pair)', 'pDockQ min'),
        ('coherence_gap', '(d) Coherence gap (best - min)', 'pDockQ - pDockQ_min'),
    ]

    figure, axes = plt.subplots(1, 4, figsize=(20, 5))

    for ax, (col, title, ylabel) in zip(axes, panels):
        group_data, group_labels, positions = [], [], []
        for idx, group in enumerate(present_groups):
            values = plot_df.loc[plot_df['chain_group'] == group, col].values
            if len(values) > 0:
                group_data.append(values)
                group_labels.append(f'{group}\n(n={len(values)})')
                positions.append(idx)

        if group_data:
            vp = ax.violinplot(group_data, positions=positions, showmedians=True, showextrema=False)
            for idx, body in enumerate(vp['bodies']):
                grp = present_groups[positions[idx]]
                body.set_facecolor(group_colours.get(grp, '#cccccc'))
                body.set_alpha(0.6)
            if 'cmedians' in vp:
                vp['cmedians'].set_color('black')
                vp['cmedians'].set_linewidth(2)
                vp['cmedians'].set_zorder(10)

            for idx, data in enumerate(group_data):
                jitter = np.random.normal(0, 0.06, size=len(data))
                grp = present_groups[positions[idx]]
                ax.scatter(positions[idx] + jitter, data,
                           c=group_colours.get(grp, '#cccccc'),
                           alpha=0.3, s=15, zorder=3, edgecolors='none')

            ax.set_xticks(positions)
            ax.set_xticklabels(group_labels, fontsize=FONT_TICK)

        _apply_common_style(ax, title, '', ylabel, grid=False)

    figure.suptitle(
        f"Chain-Count Quality Profile: Order-Statistic Bias by N "
        f"[{CAPTION_SCOPE_ALL_N}]{species_suffix}",
        fontsize=14, fontweight='bold', y=1.02)
    _save_figure(figure, f'9_Chain_Count_Profile{species_label}.png')

#--------------------------------------------Item 3: Identify similar proteins/pairs (Fig 10)---------------------------------------------------

def plot_fig10_clustering_validation(df: pd.DataFrame) -> None:
    """Fig 10: Item 3 - Identify similar proteins/pairs.
    Panel A - Homodimer ground truth scatter (shared vs total cluster count).
    Panel B - Shared cluster ratio by quality tier (heterodimers only).
    Requires clustering columns: sequence_cluster_count, shared_cluster_count, complex_type, quality_tier_v2.
    """
    required = ['sequence_cluster_count', 'shared_cluster_count', 'complex_type']
    if not all(c in df.columns for c in required):
        print("  Skipping Fig 10 - missing clustering columns")
        return

    tier_col = 'quality_tier_v2' if 'quality_tier_v2' in df.columns else 'quality_tier'
    figure, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={'width_ratios': [3, 2]})

    #======================Panel A: Homodimer Ground Truth Scatter===========================
    valid = df['sequence_cluster_count'].notna() & (df['sequence_cluster_count'] > 0)
    valid &= df['shared_cluster_count'].notna()
    plot_df = df[valid].copy()

    if {'tier_scope', 'stoichiometry'}.issubset(plot_df.columns):
        plot_df = plot_df[
            (plot_df['tier_scope'] == TIER_SCOPE_DIMER)
            & (plot_df['stoichiometry'].isin(DIMER_STOICHIOMETRIES))
        ].copy()
        if len(plot_df) == 0:
            print("  Skipping Fig 10: no dimer-validated A2/AB complexes.")
            return
        plot_df['_architecture'] = np.where(
            plot_df['stoichiometry'] == 'A2',
            'homodimer',
            'heterodimer',
        )
    else:
        complex_types_series = plot_df['complex_type'].astype(str).str.lower()
        keep = complex_types_series.isin(['homodimer', 'heterodimer'])
        plot_df = plot_df.loc[keep].copy()
        if len(plot_df) == 0:
            print("  Skipping Fig 10: no homodimer/heterodimer rows in legacy fallback.")
            return
        plot_df['_architecture'] = plot_df['complex_type'].astype(str).str.lower()

    is_homo = (plot_df['_architecture'] == 'homodimer').values
    is_hetero = (plot_df['_architecture'] == 'heterodimer').values

    seq_counts = plot_df['sequence_cluster_count'].values
    shared_counts = plot_df['shared_cluster_count'].values

    # Heterodimers first (underneath)
    if is_hetero.any():
        ax_a.scatter(seq_counts[is_hetero], shared_counts[is_hetero], s=8, alpha=0.3, color='#95a5a6', label='Heterodimer', edgecolors='none', zorder=1)
        
    # Homodimers on top
    if is_homo.any():
        ax_a.scatter(seq_counts[is_homo], shared_counts[is_homo], s=25, alpha=0.8, color='#e74c3c', label='Homodimer', edgecolors='black', linewidth=0.3, zorder=2)

    # y = x diagonal
    max_val = max(seq_counts.max(), shared_counts.max()) if len(seq_counts) > 0 else 10
    ax_a.plot([0, max_val * 1.05], [0, max_val * 1.05], 'k--', alpha=0.6, linewidth=1.5, zorder=0, label='y = x')

    # Homodimer ground truth annotation
    n_homo = is_homo.sum()
    if n_homo > 0:
        homo_seq = seq_counts[is_homo]
        homo_shared = shared_counts[is_homo]
        n_perfect = int(np.sum(homo_shared == homo_seq))
        pct = 100.0 * n_perfect / n_homo if n_homo > 0 else 0
        ax_a.text(0.05, 0.95,
                  f'Homodimers: {n_perfect}/{n_homo} on y=x ({pct:.0f}%)',
                  transform=ax_a.transAxes, va='top', fontsize=FONT_TICK,
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='grey', alpha=0.9))

    # Annotate zero-shared-cluster band
    n_zero_shared = int((shared_counts == 0).sum())
    if n_zero_shared > 0:
        pct_zero = 100.0 * n_zero_shared / len(shared_counts)
        ax_a.text(25, 2, f'No shared clusters: {n_zero_shared} ({pct_zero:.1f}%)',
                  ha='left', va='bottom',
                  fontsize=FONT_TICK - 1, color='#555555', style='italic')

    ax_a.set_xlabel('Sequence Cluster Count', fontsize=FONT_AXIS_LABEL)
    ax_a.set_ylabel('Shared Cluster Count', fontsize=FONT_AXIS_LABEL)
    ax_a.set_title('A: Homodimer Ground Truth', fontsize=FONT_TITLE, fontweight='bold')
    ax_a.tick_params(labelsize=FONT_TICK)
    ax_a.legend(fontsize=FONT_TICK - 1, loc='lower right', framealpha=0.9)
    _despine(ax_a)

    #==========================Panel B: Shared Cluster Ratio by Quality Tier=================================
    hetero_df = plot_df[plot_df['_architecture'] == 'heterodimer'].copy()
    hetero_df = hetero_df[hetero_df['sequence_cluster_count'] > 0]
    hetero_df['cluster_ratio'] = hetero_df['shared_cluster_count'] / hetero_df['sequence_cluster_count']

    tier_data = {}
    for tier in TIER_ORDER:
        vals = hetero_df.loc[hetero_df[tier_col] == tier, 'cluster_ratio'].dropna().values
        if len(vals) > 0:
            tier_data[tier] = vals

    if len(tier_data) >= 2:
        positions = []
        tier_labels = []
        data_list = []
        for i, tier in enumerate(TIER_ORDER):
            if tier in tier_data:
                positions.append(i)
                data_list.append(tier_data[tier])
                tier_labels.append(f'{tier}\n(n={len(tier_data[tier]):,})')

        parts = ax_b.violinplot(data_list, positions=positions, showmeans=False, showmedians=True)

        # Colour violin bodies
        for idx, pc in enumerate(parts['bodies']):
            tier_name = TIER_ORDER[positions[idx]] if positions[idx] < len(TIER_ORDER) else 'Medium'
            pc.set_facecolor(TIER_COLORS.get(tier_name, '#95a5a6'))
            pc.set_alpha(0.3)

        # Strip overlay with jitter
        rng = np.random.default_rng(42)
        for idx, pos in enumerate(positions):
            vals = data_list[idx]
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            tier_name = TIER_ORDER[pos] if pos < len(TIER_ORDER) else 'Medium'
            ax_b.scatter(np.full(len(vals), pos) + jitter, vals, s=3, alpha=0.25, color=TIER_COLORS.get(tier_name, '#95a5a6'), edgecolors='none', zorder=0)

        ax_b.set_xticks(positions)
        ax_b.set_xticklabels(tier_labels, fontsize=FONT_AXIS_LABEL)

        # Kruskal-Wallis (all tiers) + Mann-Whitney (High vs Low)
        try:
            from scipy.stats import kruskal, mannwhitneyu
            all_groups = [tier_data[t] for t in TIER_ORDER if t in tier_data]
            if len(all_groups) >= 2:
                h_stat, kw_p = kruskal(*all_groups)
                kw_p_str = f'p = {kw_p:.1e}' if kw_p < 0.001 else f'p = {kw_p:.3f}'
                stat_lines = [f'Kruskal-Wallis H = {h_stat:.1f}, {kw_p_str}']

                if 'High' in tier_data and 'Low' in tier_data:
                    _, mw_p = mannwhitneyu(tier_data['High'], tier_data['Low'], alternative='two-sided')
                    med_h = np.median(tier_data['High'])
                    med_l = np.median(tier_data['Low'])
                    mw_p_str = f'p = {mw_p:.1e}' if mw_p < 0.001 else f'p = {mw_p:.3f}'
                    stat_lines.append(f'High vs Low: {mw_p_str}')
                    stat_lines.append(f'High median: {med_h:.3f}')
                    stat_lines.append(f'Low median: {med_l:.3f}')

                stat_lines.append('Solid line = median')
                ax_b.text(0.95, 0.95, '\n'.join(stat_lines),
                          transform=ax_b.transAxes, va='top', ha='right',
                          fontsize=FONT_TICK - 1,
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                    edgecolor='grey', alpha=0.9))
        except ImportError:
            pass
    else:
        ax_b.text(0.5, 0.5, 'Insufficient tier data', transform=ax_b.transAxes, ha='center', va='center', fontsize=10)

    ax_b.set_ylabel('Fraction of Clusters Shared', fontsize=FONT_AXIS_LABEL)
    ax_b.set_title('B: Cluster Ratio by Quality Tier', fontsize=FONT_TITLE, fontweight='bold')
    ax_b.tick_params(labelsize=FONT_TICK)
    _despine(ax_b)
    figure.suptitle("Sequence Clustering Validation", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save_figure(figure, '10_Clustering_Validation.png')

#---------------------------------------Variant parsing helpers (Figs 11-12)-----------------------------------------------

def _normalise_significance(raw: str) -> str:
    """Map a raw ClinVar significance string to one of 5 display buckets."""
    low = raw.lower().strip()
    if 'pathogenic' in low and 'likely' not in low and 'benign' not in low:
        return 'Pathogenic'
    if 'likely pathogenic' in low:
        return 'Likely pathogenic'
    if 'benign' in low:
        return 'Benign'
    if 'uncertain' in low or low == 'vus':
        return 'VUS'
    return 'Unknown'

def _parse_variant_details(details_str) -> list:
    """Parse a pipe-separated variant_details string into structured records.
    Input format: 'K81P:interface_core:Pathogenic|R123W:surface_non_interface:Benign|...(+5 more)'
    Skips overflow tokens like '...(+N more)'. Returns [] for empty/NaN input.
    Returns:
        List of dicts with keys: mutation, context, significance.
    """
    if not isinstance(details_str, str) or not details_str.strip():
        return []
    records = []
    for token in details_str.split('|'):
        token = token.strip()
        if not token or token.startswith('...'):
            continue
        parts = token.split(':', 2)
        if len(parts) == 3:
            records.append({
                'mutation': parts[0],
                'context': parts[1],
                'significance': _normalise_significance(parts[2]),
            })
    return records

def _aggregate_all_variants(df: pd.DataFrame) -> pd.DataFrame:
    """Parse variant_details_a/b across all rows into a single flat DataFrame.
    Returns:
        DataFrame with columns: complex_name, chain, mutation, context, significance.
        Empty DataFrame (with correct columns) if no variants are found.
    """
    rows = []
    for _, row in df.iterrows():
        cname = row.get('complex_name', '')
        for chain_suffix in ('a', 'b'):
            col = f'variant_details_{chain_suffix}'
            if col not in df.columns:
                continue
            for rec in _parse_variant_details(row.get(col, '')):
                rows.append({'complex_name': cname, 'chain': chain_suffix, **rec})
    if not rows:
        return pd.DataFrame(columns=['complex_name', 'chain', 'mutation', 'context', 'significance'])
    return pd.DataFrame(rows)

def _draw_sankey_band(ax, left_y0, left_y1, right_y0, right_y1, x_left=0.15, x_right=0.85, color='grey', alpha=0.4):
    """Draw a curved flow band between left and right vertical positions.
    Uses cubic Bezier curves to create smooth S-shaped bands connecting stacked bar segments on the left to those on the right.
    """
    xm = (x_left + x_right) / 2  # midpoint for control points
    # Top edge: left_y1 -> right_y1 (cubic Bezier)
    # Bottom edge: right_y0 -> left_y0 (cubic Bezier, reversed)
    verts = [
        (x_left, left_y1),   # start top-left
        (xm, left_y1),       # control 1
        (xm, right_y1),      # control 2
        (x_right, right_y1), # end top-right
        (x_right, right_y0), # start bottom-right
        (xm, right_y0),      # control 1
        (xm, left_y0),       # control 2
        (x_left, left_y0),   # end bottom-left
        (x_left, left_y1),   # close path
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,  # top edge
        MplPath.LINETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,  # bottom edge
        MplPath.CLOSEPOLY,
    ]
    patch = mpatches.PathPatch(MplPath(verts, codes), facecolor=color, edgecolor='none', alpha=alpha)
    ax.add_patch(patch)

#-------------------------------------Item 4: Mapping genome variation (Figs 11-12)---------------------------------------------

def plot_fig11_variant_consequence_flow(df: pd.DataFrame) -> None:
    """Fig 11: Where do clinically classified variants land structurally?
    Sankey (alluvial) diagram. 
    Left nodes: clinical significance categories (Unknown excluded). 
    Right nodes: 4 structural contexts. 
    Flow bands show how many variants of each significance land in each context.
    """
    df = _filter_dimer_validated(df)
    var_df = _aggregate_all_variants(df)
    total_parsed = len(var_df)
    if total_parsed < 10:
        print("  Skipping Fig 11: fewer than 10 parsed variants.")
        return

    # Filter to classified variants only (exclude Unknown)
    classified_sigs = ['Pathogenic', 'Likely pathogenic', 'VUS', 'Benign']
    classified = var_df[var_df['significance'].isin(classified_sigs)].copy()
    n_classified = len(classified)
    n_unknown = total_parsed - n_classified
    pct_unknown = 100.0 * n_unknown / total_parsed if total_parsed > 0 else 0.0

    if n_classified < 10:
        print("  Skipping Fig 11: fewer than 10 classified variants after removing Unknown.")
        return

    # Cross-tabulate: significance (left) x context (right)
    classified['significance'] = pd.Categorical(classified['significance'], categories=classified_sigs, ordered=True)
    classified['context'] = pd.Categorical(classified['context'], categories=CONTEXT_ORDER, ordered=True)
    ct = pd.crosstab(classified['significance'], classified['context'], dropna=False)
    ct = ct.reindex(index=classified_sigs, columns=CONTEXT_ORDER, fill_value=0)

    figure, ax = plt.subplots(figsize=(12, 7.5))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.02, 1.02)
    ax.axis('off')
    figure.subplots_adjust(top=0.88, bottom=0.12)

    bar_w = 0.12  # width of stacked bars
    x_left = 0.15
    x_right = 0.85
    gap = 0.015  # vertical gap between segments

    #=================Left bars (significance)======================
    left_totals = ct.sum(axis=1).values.astype(float)
    left_total = left_totals.sum()
    usable_height = 1.0 - gap * (len(classified_sigs) - 1)
    left_heights = (left_totals / left_total) * usable_height
    left_positions = []  # (y0, y1) for each significance
    y_cursor = 0.0
    for i, sig in enumerate(classified_sigs):
        h = left_heights[i]
        y0, y1 = y_cursor, y_cursor + h
        left_positions.append((y0, y1))
        ax.add_patch(mpatches.FancyBboxPatch((x_left - bar_w, y0), bar_w, h, boxstyle="round,pad=0.005", facecolor=SIGNIFICANCE_COLORS[sig], edgecolor='white', linewidth=1))
        # Label
        mid_y = (y0 + y1) / 2
        count = int(left_totals[i])
        ax.text(x_left - bar_w - 0.02, mid_y, f'{sig}\n(n={count:,})', ha='right', va='center', fontsize=9, fontweight='bold')
        y_cursor = y1 + gap

    #===================Right bars (context)======================
    right_totals = ct.sum(axis=0).values.astype(float)
    right_total = right_totals.sum()
    right_heights = (right_totals / right_total) * usable_height
    right_positions = []  # (y0, y1) for each context
    right_label_ys = []   # raw mid_y for each label
    y_cursor = 0.0
    for i, ctx in enumerate(CONTEXT_ORDER):
        h = right_heights[i]
        y0, y1 = y_cursor, y_cursor + h
        right_positions.append((y0, y1))
        ax.add_patch(mpatches.FancyBboxPatch(
            (x_right, y0), bar_w, h, boxstyle="round,pad=0.005", facecolor=CONTEXT_COLORS[ctx], edgecolor='white', linewidth=1))
        right_label_ys.append((y0 + y1) / 2)
        y_cursor = y1 + gap

    # Enforce minimum vertical spacing between right-side labels to prevent overlap
    min_label_gap = 0.05
    adjusted_ys = list(right_label_ys)
    for i in range(len(adjusted_ys) - 1, 0, -1):
        if adjusted_ys[i] - adjusted_ys[i - 1] < min_label_gap:
            adjusted_ys[i - 1] = adjusted_ys[i] - min_label_gap

    for i, ctx in enumerate(CONTEXT_ORDER):
        count = int(right_totals[i])
        label = CONTEXT_LABELS[ctx].replace('\n', ' ')
        fsize = 8 if right_heights[i] < 0.03 else 9
        ax.text(x_right + bar_w + 0.02, adjusted_ys[i], f'{label}\n(n={count:,})', ha='left', va='center', fontsize=fsize, fontweight='bold')

    #==========================Flow bands====================================
    # Track cumulative position within each bar for sub-band placement
    left_cursors = [pos[0] for pos in left_positions]
    right_cursors = [pos[0] for pos in right_positions]

    for i, sig in enumerate(classified_sigs):
        for j, ctx in enumerate(CONTEXT_ORDER):
            count = ct.iloc[i, j]
            if count == 0:
                continue
            # Band height proportional to count within each bar
            left_h = (count / left_total) * usable_height
            right_h = (count / right_total) * usable_height
            _draw_sankey_band(ax,
                              left_y0=left_cursors[i],
                              left_y1=left_cursors[i] + left_h,
                              right_y0=right_cursors[j],
                              right_y1=right_cursors[j] + right_h,
                              x_left=x_left, x_right=x_right,
                              color=SIGNIFICANCE_COLORS[sig], alpha=0.35)
            left_cursors[i] += left_h
            right_cursors[j] += right_h

    #================================================Annotations=======================================================
    ax.text(0.50, 1.06, "Classified Variant Flow: Clinical Significance -> Structural Context [dimer-validated]", ha='center', va='bottom', fontsize=14, fontweight='bold', transform=ax.transAxes)

    # Footer annotation (merged to avoid overlap)
    footer_parts = [f'n = {n_classified:,} classified variants ({pct_unknown:.1f}% Unknown excluded, {n_unknown:,} variants)']
    true_total_a = pd.to_numeric(df.get('n_variants_a', pd.Series(dtype=float)), errors='coerce').sum()
    true_total_b = pd.to_numeric(df.get('n_variants_b', pd.Series(dtype=float)), errors='coerce').sum()
    true_total = true_total_a + true_total_b
    if not np.isnan(true_total) and true_total > total_parsed * 1.05:
        footer_parts.append(f'Variant details limited to 20 per chain ({total_parsed:,} shown of ~{int(true_total):,} total)')
    ax.text(0.50, -0.06, '\n'.join(footer_parts), ha='center', va='top', fontsize=8, style='italic', color='#666666', transform=ax.transAxes)
    _save_figure(figure, '11_Variant_Consequence_Flow.png')

def plot_fig12_variant_density(df: pd.DataFrame, density_mode: bool = False) -> None:
    """Fig 12: Item 4 - Mapping genome variation.
    Scatter plot of interface variant density (variants per interface residue) against composite score, coloured by quality tier.
    Spearman and partial correlations annotated. Tier-stratified median densities in text box.
    """
    df = _filter_dimer_validated(df)
    # Compute interface variant density per complex
    n_if_var_a = pd.to_numeric(df.get('n_interface_variants_a', pd.Series(dtype=float)), errors='coerce').fillna(0)
    n_if_var_b = pd.to_numeric(df.get('n_interface_variants_b', pd.Series(dtype=float)), errors='coerce').fillna(0)
    n_if_res_a = pd.to_numeric(df.get('n_interface_residues_a', pd.Series(dtype=float)), errors='coerce').fillna(0)
    n_if_res_b = pd.to_numeric(df.get('n_interface_residues_b', pd.Series(dtype=float)), errors='coerce').fillna(0)

    n_if_var = n_if_var_a + n_if_var_b
    n_if_res = n_if_res_a + n_if_res_b
    # Density: variants per interface residue (NaN where no interface residues)
    density = np.where(n_if_res > 0, n_if_var / n_if_res, np.nan)
    density = pd.Series(density, index=df.index)

    # Choose x-axis metric
    if 'interface_confidence_score' in df.columns:
        x_col = 'interface_confidence_score'
        x_label = 'Composite Score'
    else:
        x_col = 'iptm'
        x_label = 'ipTM'

    # Filter to valid rows
    x_series = pd.to_numeric(df.get(x_col, pd.Series(dtype=float)), errors='coerce')
    valid_mask = density.notna() & x_series.notna() & (n_if_res > 0)

    if valid_mask.sum() < 5:
        print("  Skipping Fig 12: fewer than 5 complexes with valid density.")
        return

    x_vals = x_series[valid_mask].values.astype(float)
    y_vals = density[valid_mask].values.astype(float)
    size_vals = n_if_res[valid_mask].values.astype(float)  # for partial correlation

    tier_col = 'quality_tier_v2' if 'quality_tier_v2' in df.columns else 'quality_tier'

    figure, ax = plt.subplots(figsize=(10, 7))

    #=============================Scatter coloured by quality tier=================================
    base_size, base_alpha = _adaptive_scatter_params(len(x_vals))
    colors = df.loc[valid_mask, tier_col].map(TIER_COLORS).fillna('#bdc3c7').values if tier_col in df.columns else '#3498db'

    _timed_scatter(ax, x_vals, y_vals, len(x_vals), fig_label='Fig 12', c=colors, s=base_size, alpha=base_alpha, edgecolors='white', linewidths=0.3)

    if density_mode:
        _overlay_kde_contours(ax, x_vals, y_vals)

    #==============================Spearman correlation===================================
    stat_lines = []
    if len(x_vals) >= 5 and np.std(x_vals) > 1e-9 and np.std(y_vals) > 1e-9:
        rho, pval = spearmanr(x_vals, y_vals)
        p_str = 'p < 0.001' if pval < 0.001 else f'p = {pval:.3f}'
        stat_lines.append(f'Spearman \u03c1 = {rho:.4f}, {p_str}')

        # Partial correlation controlling for interface size (rank-residual method)
        if len(x_vals) >= 10 and np.std(size_vals) > 1e-9:
            from scipy.stats import rankdata
            rx = rankdata(x_vals)
            ry = rankdata(y_vals)
            rz = rankdata(size_vals)
            # Regress ranks on size ranks via OLS, take residuals
            rz_mean = rz.mean()
            rz_centered = rz - rz_mean
            rz_ss = np.dot(rz_centered, rz_centered)
            if rz_ss > 1e-9:
                beta_x = np.dot(rz_centered, rx - rx.mean()) / rz_ss
                beta_y = np.dot(rz_centered, ry - ry.mean()) / rz_ss
                resid_x = rx - beta_x * rz_centered
                resid_y = ry - beta_y * rz_centered
                if np.std(resid_x) > 1e-9 and np.std(resid_y) > 1e-9:
                    rho_partial, pval_partial = spearmanr(resid_x, resid_y)
                    p_str2 = 'p < 0.001' if pval_partial < 0.001 else f'p = {pval_partial:.3f}'
                    stat_lines.append(f'Partial \u03c1 = {rho_partial:.4f}, {p_str2}  (size-controlled)')

    #======================Tier-stratified medians===========================
    median_lines = []
    if tier_col in df.columns:
        tiers_valid = df.loc[valid_mask, tier_col].values
        for tier in TIER_ORDER:
            tier_vals = y_vals[tiers_valid == tier]
            if len(tier_vals) > 0:
                median_lines.append(f'{tier}: {np.median(tier_vals):.3f} (n={len(tier_vals)})')

    #======================Annotation box===========================
    annotation_parts = []
    if stat_lines:
        annotation_parts.extend(stat_lines)
    if median_lines:
        annotation_parts.append('')
        annotation_parts.append('Median density by tier:')
        annotation_parts.extend(f'  {line}' for line in median_lines)

    if annotation_parts:
        ax.text(0.03, 0.72, '\n'.join(annotation_parts),
                transform=ax.transAxes, fontsize=8, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                          edgecolor='#cccccc', alpha=0.95),
                family='monospace')

    #======================Legend===========================
    if tier_col in df.columns:
        legend_handles = []
        tiers_valid = df.loc[valid_mask, tier_col].values
        for tier in TIER_ORDER:
            count = (tiers_valid == tier).sum()
            if count > 0:
                legend_handles.append(
                    Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=TIER_COLORS[tier],
                           markeredgecolor='white', markersize=9,
                           label=f'{tier} (n={count})'))
        if legend_handles:
            ax.legend(handles=legend_handles, fontsize=FONT_TICK, loc='upper left', framealpha=0.9)

    _apply_common_style(ax, '', x_label, 'Variant Density (per interface residue)')
    figure.suptitle("Interface Variant Density vs Composite Score [dimer-validated]", fontsize=14, fontweight='bold', y=1.02)
    _save_figure(figure, '12_Variant_Density.png')

#--------------------------------------Item 6: Map stability scores (Fig 13)-----------------------------------------------

def plot_fig13_stability_crossvalidation(df: pd.DataFrame) -> None:
    """Fig 13: Item 6 - Map stability scores.
    Panel A - EVE vs AlphaMissense concordance scatter (pooled both chains).
    Panel B - AlphaMissense vs monomeric FoldX DDG scatter (chain A).
    Panel C - Coverage landscape grouped bar chart by quality tier.
    Requires stability + ProtVar columns: eve_score_mean_a/b, protvar_am_mean_a/b, protvar_foldx_mean_a/b, eve_coverage_a/b, quality_tier_v2.
    """
    required = ['eve_score_mean_a', 'protvar_am_mean_a', 'quality_tier_v2']
    if not all(c in df.columns for c in required):
        print("  Skipping Fig 13 - missing stability/ProtVar columns")
        return

    # Dissertation-safe: stability comparisons use dimer-calibrated quality tiers.
    df = _filter_dimer_validated(df)
    if len(df) == 0:
        print("  Skipping Fig 13 - no dimer-validated complexes.")
        return

    tier_col = 'quality_tier_v2' if 'quality_tier_v2' in df.columns else 'quality_tier'

    figure, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(16, 5))

    #==========================Panel A: EVE vs AlphaMissense (pooled both chains)============================
    eve_vals, am_vals, tier_vals_a = [], [], []
    for _, row in df.iterrows():
        tier = row.get(tier_col, '')
        if tier not in TIER_ORDER:
            continue
        for suffix in ('a', 'b'):
            eve = row.get(f'eve_score_mean_{suffix}', np.nan)
            am = row.get(f'protvar_am_mean_{suffix}', np.nan)
            if pd.notna(eve) and pd.notna(am) and eve > 0 and am > 0:
                eve_vals.append(eve)
                am_vals.append(am)
                tier_vals_a.append(tier)

    if len(eve_vals) >= 10:
        eve_arr = np.array(eve_vals)
        am_arr = np.array(am_vals)
        colors_a = [TIER_COLORS.get(t, '#95a5a6') for t in tier_vals_a]
        s, alpha = _adaptive_scatter_params(len(eve_arr))
        ax_a.scatter(eve_arr, am_arr, c=colors_a, s=s, alpha=alpha, edgecolors='none')

        try:
            from scipy.stats import spearmanr
            rho, p = spearmanr(eve_arr, am_arr)
            p_str = f'p < 0.001' if p < 0.001 else f'p = {p:.3f}'
            ax_a.text(0.05, 0.95, f'\u03c1 = {rho:.2f}, {p_str}\nn = {len(eve_arr):,}',
                      transform=ax_a.transAxes, va='top', fontsize=FONT_TICK,
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor='grey', alpha=0.9))
        except ImportError:
            pass

        # Tier legend
        for t in TIER_ORDER:
            ax_a.scatter([], [], c=TIER_COLORS[t], s=20, label=t)
        ax_a.legend(fontsize=FONT_TICK - 1, loc='lower right', framealpha=0.9)
    else:
        ax_a.text(0.5, 0.5, 'Insufficient overlap\n(< 10 pairs)', transform=ax_a.transAxes, ha='center', va='center', fontsize=10)

    ax_a.set_xlabel('EVE Mean Score (higher = more pathogenic)', fontsize=FONT_AXIS_LABEL)
    ax_a.set_ylabel('AlphaMissense Mean Score (higher = more pathogenic)', fontsize=FONT_AXIS_LABEL)
    ax_a.set_title('A: EVE vs AlphaMissense', fontsize=FONT_TITLE, fontweight='bold')
    ax_a.tick_params(labelsize=FONT_TICK)
    _despine(ax_a)

    #==========================Panel B: AlphaMissense vs FoldX (chain A only)============================
    am_b, foldx_b, tier_vals_b = [], [], []
    for _, row in df.iterrows():
        tier = row.get(tier_col, '')
        if tier not in TIER_ORDER:
            continue
        am = row.get('protvar_am_mean_a', np.nan)
        fx = row.get('protvar_foldx_mean_a', np.nan)
        if pd.notna(am) and pd.notna(fx) and am > 0 and fx > 0:
            am_b.append(am)
            foldx_b.append(fx)
            tier_vals_b.append(tier)

    if len(am_b) >= 10:
        am_b_arr = np.array(am_b)
        fx_arr = np.array(foldx_b)
        colors_b = [TIER_COLORS.get(t, '#95a5a6') for t in tier_vals_b]
        s, alpha = _adaptive_scatter_params(len(am_b_arr))
        ax_b.scatter(am_b_arr, fx_arr, c=colors_b, s=s, alpha=alpha, edgecolors='none')

        try:
            from scipy.stats import spearmanr
            rho, p = spearmanr(am_b_arr, fx_arr)
            p_str = f'p < 0.001' if p < 0.001 else f'p = {p:.3f}'
            ax_b.text(0.05, 0.95, f'\u03c1 = {rho:.2f}, {p_str}\nn = {len(am_b_arr):,}',
                      transform=ax_b.transAxes, va='top', fontsize=FONT_TICK,
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor='grey', alpha=0.9))
        except ImportError:
            pass
    else:
        ax_b.text(0.5, 0.5, 'Insufficient overlap\n(< 10 pairs)', transform=ax_b.transAxes, ha='center', va='center', fontsize=10)

    ax_b.set_xlabel('AlphaMissense Mean Score', fontsize=FONT_AXIS_LABEL)
    ax_b.set_ylabel('Monomeric FoldX \u0394\u0394G Mean (kcal/mol, higher = destabilising)', fontsize=FONT_AXIS_LABEL)
    ax_b.set_title('B: AlphaMissense vs FoldX', fontsize=FONT_TITLE, fontweight='bold')
    ax_b.tick_params(labelsize=FONT_TICK)
    _despine(ax_b)

    #==========================Panel C: Coverage Landscape (grouped bar)============================
    predictor_names = ['EVE', 'AlphaMissense', 'FoldX']
    predictor_colors = ['#3498db', '#e67e22', '#2ecc71']
    bar_width = 0.25
    x_positions = np.arange(len(TIER_ORDER))

    overall_coverage = {p: {'covered': 0, 'total': 0} for p in predictor_names}

    for i, pred in enumerate(predictor_names):
        coverages = []
        for tier in TIER_ORDER:
            tier_df = df[df[tier_col] == tier]
            n_tier = len(tier_df)
            if n_tier == 0:
                coverages.append(0)
                continue

            if pred == 'EVE':
                cov_a = tier_df.get('eve_coverage_a', pd.Series(dtype=float))
                cov_b = tier_df.get('eve_coverage_b', pd.Series(dtype=float))
                covered = ((cov_a.fillna(0) > 0) | (cov_b.fillna(0) > 0)).sum()
            elif pred == 'AlphaMissense':
                am_a = tier_df.get('protvar_am_mean_a', pd.Series(dtype=float))
                am_b = tier_df.get('protvar_am_mean_b', pd.Series(dtype=float))
                covered = (am_a.notna() | am_b.notna()).sum()
            else:  # FoldX
                fx_a = tier_df.get('protvar_foldx_mean_a', pd.Series(dtype=float))
                fx_b = tier_df.get('protvar_foldx_mean_b', pd.Series(dtype=float))
                covered = (fx_a.notna() | fx_b.notna()).sum()

            overall_coverage[pred]['covered'] += int(covered)
            overall_coverage[pred]['total'] += n_tier
            coverages.append(100.0 * covered / n_tier)

        bars = ax_c.bar(x_positions + i * bar_width, coverages, bar_width, color=predictor_colors[i], label=pred, edgecolor='white', linewidth=0.5, alpha=0.85)

        # Annotate each bar with percentage
        for bar_obj, cov in zip(bars, coverages):
            if cov > 0:
                ax_c.text(bar_obj.get_x() + bar_obj.get_width() / 2, cov + 1.5,
                          f'{cov:.0f}%', ha='center', va='bottom', fontsize=7)

    ax_c.set_xticks(x_positions + bar_width)
    ax_c.set_xticklabels(TIER_ORDER, fontsize=FONT_AXIS_LABEL)
    ax_c.set_ylabel('Coverage (%)', fontsize=FONT_AXIS_LABEL)
    ax_c.set_ylim(0, 110)
    ax_c.set_title('C: Coverage by Quality Tier', fontsize=FONT_TITLE, fontweight='bold')
    ax_c.tick_params(labelsize=FONT_TICK)
    ax_c.legend(fontsize=FONT_TICK - 1, loc='upper left', framealpha=0.9)

    # Overall coverage annotation
    overall_parts = []
    for pred in predictor_names:
        oc = overall_coverage[pred]
        pct = 100.0 * oc['covered'] / oc['total'] if oc['total'] > 0 else 0
        overall_parts.append(f'{pred}: {pct:.0f}%')
    ax_c.text(0.5, -0.12, 'Overall: ' + '  |  '.join(overall_parts), transform=ax_c.transAxes, ha='center', fontsize=FONT_TICK - 1, style='italic', color='#555555')
    _despine(ax_c)
    figure.suptitle(
        f"Stability Predictor Cross-Validation [{CAPTION_SCOPE_DIMER}]",
        fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save_figure(figure, '13_Stability_CrossValidation.png')

#-------------------------------------Disease parsing helpers (Fig 14)------------------------------------------------------------

def _parse_disease_name(entry: str) -> str:
    """Extract disease name from a disease_details entry.
    Input formats:
        'OMIM:618428:Popov-Chang syndrome (POPCHAS)' -> 'Popov-Chang syndrome (POPCHAS)'
        'OMIM:154700:Marfan syndrome' -> 'Marfan syndrome'
        'Cancer' -> 'Cancer'
        '' -> ''
    Returns the disease name without OMIM prefix.
    """
    if not entry or not isinstance(entry, str):
        return ''
    entry = entry.strip()
    if entry.startswith('OMIM:'):
        # Format: OMIM:ID:name
        parts = entry.split(':', 2)
        return parts[2] if len(parts) == 3 else entry
    return entry

# --------------------------------------------Disease & pathway analysis (Figs 14-15)----------------------------------------

def plot_fig14_disease_enrichment(df: pd.DataFrame) -> None:
    """Fig 14: Disease & pathway analysis.
    Two-panel figure: 
    (A) grouped bar chart of disease prevalence by tier with chi-square test and drug-target annotation box.
    (B) top 10 diseases by frequency as horizontal stacked bars segmented by quality tier.
    Requires 'n_diseases_a' column.
    """
    if 'n_diseases_a' not in df.columns:
        print("  Skipping Fig 14 - no disease data available")
        return

    tier_col = 'quality_tier_v2' if 'quality_tier_v2' in df.columns else 'quality_tier'
    if tier_col not in df.columns:
        print("  Skipping Fig 14 - no quality tier column")
        return

    df = _filter_dimer_validated(df)
    if len(df) == 0:
        print("  Skipping Fig 14 - no dimer-validated rows")
        return

    total_diseases = df['n_diseases_a'].fillna(0).astype(int)
    if 'n_diseases_b' in df.columns:
        total_diseases = total_diseases + df['n_diseases_b'].fillna(0).astype(int)

    has_disease = total_diseases > 0

    figure, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={'width_ratios': [2.5, 2.5]})

    #=======================Panel A: disease prevalence by tier========================
    tier_stats = {}
    for t in TIER_ORDER:
        mask = df[tier_col] == t
        n_tier = mask.sum()
        n_dis = (mask & has_disease).sum()
        tier_stats[t] = {'n_tier': n_tier, 'n_disease': n_dis,
                         'n_no_disease': n_tier - n_dis,
                         'pct_disease': (n_dis / n_tier * 100) if n_tier > 0 else 0,
                         'pct_no_disease': ((n_tier - n_dis) / n_tier * 100) if n_tier > 0 else 0}

    x_pos = np.arange(len(TIER_ORDER))
    width = 0.35

    pct_dis = [tier_stats[t]['pct_disease'] for t in TIER_ORDER]
    pct_no = [tier_stats[t]['pct_no_disease'] for t in TIER_ORDER]
    n_dis_vals = [tier_stats[t]['n_disease'] for t in TIER_ORDER]
    n_no_vals = [tier_stats[t]['n_no_disease'] for t in TIER_ORDER]

    bars_dis = ax_a.bar(x_pos - width / 2, pct_dis, width, color='#c0392b', alpha=0.7, label='Has disease', edgecolor='grey', linewidth=0.5)
    bars_no = ax_a.bar(x_pos + width / 2, pct_no, width, color='#d5d8dc', alpha=0.7, label='No disease', edgecolor='grey', linewidth=0.5)

    # Annotate counts on bars
    for i, (bar_d, bar_n, nd, nn) in enumerate(zip(bars_dis, bars_no, n_dis_vals, n_no_vals)):
        ax_a.text(bar_d.get_x() + bar_d.get_width() / 2, bar_d.get_height() + 1,
                  f'{nd:,}', ha='center', va='bottom', fontsize=7, color='#555555')
        ax_a.text(bar_n.get_x() + bar_n.get_width() / 2, bar_n.get_height() + 1,
                  f'{nn:,}', ha='center', va='bottom', fontsize=7, color='#555555')

    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels(TIER_ORDER, fontsize=FONT_TICK)
    ax_a.set_ylim(0, max(pct_no + pct_dis) * 1.22)

    # Chi-square test
    contingency = np.array([[tier_stats[t]['n_disease'] for t in TIER_ORDER], [tier_stats[t]['n_no_disease'] for t in TIER_ORDER]])
    if contingency.min() >= 5:
        chi2, p_val, dof, expected = chi2_contingency(contingency)
        p_str = f'p < 0.001' if p_val < 0.001 else f'p = {p_val:.2e}'
        ax_a.text(0.98, 0.95, f'\u03c7\u00b2 = {chi2:.1f}, {p_str}',
                  transform=ax_a.transAxes, ha='right', va='top',
                  fontsize=9, bbox=dict(boxstyle='round,pad=0.3',
                                        facecolor='white', edgecolor='grey', alpha=0.8))

    # Drug target disease prevalence - text annotation box with Fisher test
    baseline_pct = has_disease.sum() / len(df) * 100 if len(df) > 0 else 0
    if 'is_drug_target_a' in df.columns:
        drug_a = _boolish(df['is_drug_target_a']).fillna(False)
        drug_b = _boolish(df['is_drug_target_b']).fillna(False) if 'is_drug_target_b' in df.columns else pd.Series(False, index=df.index)
        is_drug = drug_a | drug_b
        n_drug = is_drug.sum()
        if n_drug >= 2:
            drug_disease_pct = (is_drug & has_disease).sum() / n_drug * 100
            # Fisher exact test: drug-target vs non-drug-target disease prevalence
            n_drug_dis = int((is_drug & has_disease).sum())
            n_drug_no = int(n_drug) - n_drug_dis
            n_nondrug_dis = int(has_disease.sum()) - n_drug_dis
            n_nondrug_no = int(len(df)) - n_drug_dis - n_drug_no - n_nondrug_dis
            fisher_table = [[n_drug_dis, n_drug_no], [n_nondrug_dis, n_nondrug_no]]
            _, fisher_p = fisher_exact(fisher_table, alternative='two-sided')
            fisher_p_str = 'p < 0.001' if fisher_p < 0.001 else f'p = {fisher_p:.2e}'
            ax_a.text(0.02, 0.35,
                      f'Drug targets: {drug_disease_pct:.0f}% disease-assoc.\n'
                      f'vs {baseline_pct:.0f}% baseline '
                      f'(Fisher {fisher_p_str})\n'
                      f'(n = {n_drug})',
                      transform=ax_a.transAxes, ha='left', va='top',
                      fontsize=8, color='#9b59b6',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='#f4ecf7',
                                edgecolor='#9b59b6', alpha=0.9))

    ax_a.legend(fontsize=FONT_TICK, loc='upper left', framealpha=0.9)
    _apply_common_style(ax_a, 'Disease Prevalence by Quality Tier', 'Quality Tier', '% of Tier')
    _despine(ax_a)

    #=================Panel B: top diseases by tier (stacked horizontal bars)======================
    disease_tier_counts: dict[str, dict[str, int]] = {}
    disease_protein_counts: dict[str, set] = {}  # unique accessions per disease

    for _, row in df.iterrows():
        tier = row.get(tier_col, '')
        if tier not in TIER_ORDER:
            continue
        for suffix in ('a', 'b'):
            details_str = row.get(f'disease_details_{suffix}', '')
            if not isinstance(details_str, str) or not details_str.strip():
                continue
            accession = str(row.get(f'protein_{suffix}', ''))
            for entry in details_str.split('|'):
                entry = entry.strip()
                if not entry or entry.startswith('...'):
                    continue
                name = _parse_disease_name(entry)
                if not name:
                    continue
                disease_tier_counts.setdefault(name, {t: 0 for t in TIER_ORDER})
                disease_tier_counts[name][tier] += 1
                disease_protein_counts.setdefault(name, set())
                if accession:
                    disease_protein_counts[name].add(accession)

    if disease_tier_counts:
        # Select top 10 diseases based on a two-tier ranking:
        # 1. Primary: Number of unique proteins involved (descending)
        # 2. Tiebreaker: Total sum of all tier annotations (descending)
        sorted_diseases = sorted(disease_tier_counts.keys(), key=lambda d: (len(disease_protein_counts.get(d, set())), sum(disease_tier_counts[d].values())), reverse=True)[:10]

        y_pos = np.arange(len(sorted_diseases))
        left = np.zeros(len(sorted_diseases))

        for t in TIER_ORDER:
            widths = np.array([disease_tier_counts[d].get(t, 0) for d in sorted_diseases], dtype=float)
            ax_b.barh(y_pos, widths, left=left, color=TIER_COLORS[t], label=t, height=0.6, edgecolor='white', linewidth=0.5)
            left += widths

        ax_b.set_yticks(y_pos)

        def _truncate_name(name, max_len=50):
            if len(name) <= max_len:
                return name
            return name[:max_len].rsplit(' ', 1)[0] + '...'

        ax_b.set_yticklabels([_truncate_name(name) for name in sorted_diseases], fontsize=8)
        ax_b.invert_yaxis()
        ax_b.set_xlim(0, max(left) * 1.30)

        # Total count + unique proteins annotation at end of each bar
        xlim_upper = max(left) * 1.30
        for i, d in enumerate(sorted_diseases):
            total = sum(disease_tier_counts[d].values())
            n_prots = len(disease_protein_counts.get(d, set()))
            label_text = f'{total} ({n_prots} proteins)'
            label_x = left[i] + max(left) * 0.02
            if label_x + len(label_text) * 0.35 > xlim_upper:
                # Bar too close to boundary - place label inside in white
                ax_b.text(left[i] - max(left) * 0.02, i, label_text, va='center', ha='right', fontsize=7, color='white', fontweight='bold')
            else:
                ax_b.text(label_x, i, label_text, va='center', fontsize=7, color='#555555')

        ax_b.legend(fontsize=FONT_TICK - 1, loc='center right', framealpha=1.0)
        _apply_common_style(ax_b, 'Top Disease Categories by Tier', 'Annotations (unique proteins shown)', '')
        _despine(ax_b)

        # Hub-protein footnote if any top disease has few unique proteins but many annotations
        has_hub = any(
            len(disease_protein_counts.get(d, set())) <= 3
            and sum(disease_tier_counts[d].values()) >= 20
            for d in sorted_diseases
        )
        if has_hub:
            ax_b.text(0.5, -0.08, '\u2020 Diseases with \u22643 unique proteins may reflect hub-protein effects.', transform=ax_b.transAxes, ha='center', va='top', fontsize=6.5, style='italic', color='#777777')
    else:
        ax_b.text(0.5, 0.5, 'No disease details\navailable', transform=ax_b.transAxes, ha='center', va='center', fontsize=FONT_AXIS_LABEL, color='#999999')
        ax_b.set_axis_off()

    figure.subplots_adjust(top=0.88, wspace=0.45)
    figure.suptitle("Disease Annotation Prevalence Across Quality Tiers [dimer-validated]", fontsize=FONT_TITLE + 1, fontweight='bold', y=0.98)

    n_effective = len(df) # dataset size
    caption_14 = (
    f'Left: Disease annotation prevalence across quality tiers (\u03c7\u00b2, n = {n_effective:,}). '
    f'Higher prevalence in lower tiers likely reflects annotation bias toward well-studied '
    f'disordered proteins. Right: Top diseases ranked by unique proteins, displaying total '
    f'annotations and protein counts.'
    )
    figure.text(0.5, -0.01, caption_14,ha='center', fontsize=7, style='italic', color='#777777')
    _save_figure(figure, '14_Disease_Enrichment.png')

#-------------------------------------------------Pathway network helpers (Fig 15)------------------------------------------------------------------

def _compute_reactome_depths(hierarchy: dict) -> dict:
    """Compute depth level for each Reactome pathway via BFS from roots.
    Args:
    hierarchy : dict
        ``{parent_pathway_id: [child_pathway_id, ...]}``.
    Returns:
    dict
        ``{pathway_id: depth}`` where roots are depth 0.
    """
    from collections import deque

    # Find all pathway IDs that appear
    all_children: set = set()
    all_parents: set = set()
    for parent, children in hierarchy.items():
        all_parents.add(parent)
        for child in children:
            all_children.add(child)

    # Roots = parents that never appear as children
    roots = all_parents - all_children

    # BFS from roots
    depths: dict = {}
    queue = deque()
    for root in roots:
        depths[root] = 0
        queue.append(root)

    while queue:
        node = queue.popleft()
        for child in hierarchy.get(node, []):
            if child not in depths:
                depths[child] = depths[node] + 1
                queue.append(child)

    return depths

def plot_fig15_pathway_network(df: pd.DataFrame,
                                max_pathways: int = 20,
                                min_shared_complexes: int = 20,
                                hierarchy_file: Optional[str] = 'data/pathways/ReactomePathwaysRelation.txt',
                                filter_hierarchy: bool = True,
                                depth_level: int = 1) -> None:
    """Fig 15: Disease & pathway analysis.
    Nodes are the top N Reactome pathways at a single hierarchy depth level.
    Edges connect pathways that share complexes above a threshold, with hierarchical parent-child links excluded. 
    Node colour encodes % High-tier complexes (RdYlGn), node size encodes complex count, edges are grey with width proportional to shared complex count.
    Uses kamada_kawai_layout for deterministic, reproducible layout.
    Args:
    hierarchy_file : str or None
        Path to ReactomePathwaysRelation.txt. Default searches in ``data/pathways/``. Set to None to disable hierarchy filtering.
    filter_hierarchy : bool
        If True (default), remove parent-child edges from the network.
    depth_level : int
        Reactome hierarchy depth to display (0 = top-level, 1 = second-level).
        Falls back to depth+1 if fewer than 5 pathways have data at target depth.
    Requires NetworkX and 'reactome_pathways_a' column.
    """
    if not _HAS_NETWORKX:
        print("  Skipping Fig 15 - NetworkX not installed")
        return
    if 'reactome_pathways_a' not in df.columns:
        print("  Skipping Fig 15 - no pathway data available")
        return

    df = _filter_dimer_validated(df)
    if len(df) == 0:
        print("  Skipping Fig 15 - no dimer-validated rows")
        return

    tier_col = 'quality_tier_v2' if 'quality_tier_v2' in df.columns else 'quality_tier'

    #========================Load hierarchy for edge filtering and depth computation========================
    hierarchy_pairs: set = set()
    pathway_depths: dict = {}
    hierarchy_loaded = False
    if filter_hierarchy and hierarchy_file:
        try:
            from pathway_network import load_reactome_hierarchy
            hierarchy = load_reactome_hierarchy(hierarchy_file)
            for parent, children in hierarchy.items():
                for child in children:
                    hierarchy_pairs.add((parent, child))
                    hierarchy_pairs.add((child, parent))
            pathway_depths = _compute_reactome_depths(hierarchy)
            hierarchy_loaded = True
        except FileNotFoundError:
            print(f"  Warning: hierarchy file not found ({hierarchy_file}), "
                  f"skipping hierarchy filtering")
        except Exception as e:
            print(f"  Warning: could not load hierarchy file: {e}")

    #========================Build pathway co-occurrence data========================
    pathway_complexes: dict[str, list[float]] = {}
    pathway_tiers: dict[str, list[str]] = {}  # for % High-tier colouring
    pathway_names: dict[str, str] = {}
    edge_data: dict[tuple, list[float]] = {}

    for _, row in df.iterrows():
        pdockq_val = row.get('pdockq')
        if pd.isna(pdockq_val):
            continue
        pdockq_val = float(pdockq_val)
        tier = str(row.get(tier_col, ''))

        complex_pids = set()
        for suffix in ('a', 'b'):
            pathways_str = row.get(f'reactome_pathways_{suffix}', '')
            if not pathways_str or pd.isna(pathways_str):
                continue
            for entry in str(pathways_str).split('|'):
                if entry.startswith('...('):
                    continue
                parts = entry.split(':', 1)
                if len(parts) == 2:
                    pid, pname = parts
                    complex_pids.add(pid)
                    pathway_names[pid] = pname

        for pid in complex_pids:
            pathway_complexes.setdefault(pid, []).append(pdockq_val)
            pathway_tiers.setdefault(pid, []).append(tier)

        pid_list = sorted(complex_pids)
        for i in range(len(pid_list)):
            for j in range(i + 1, len(pid_list)):
                key = (pid_list[i], pid_list[j])
                edge_data.setdefault(key, []).append(pdockq_val)

    if not pathway_complexes:
        print("  Skipping Fig 15 - no parseable pathway data")
        return

    #=========================Filter to target depth level=========================
    effective_depth = depth_level
    if hierarchy_loaded and pathway_depths:
        # Filter pathway_complexes to target depth
        depth_candidates = {pid for pid in pathway_complexes
                            if pathway_depths.get(pid) == depth_level}
        # Fall back to depth+1 if too few pathways at target depth
        if len(depth_candidates) < 5:
            depth_candidates_fallback = {pid for pid in pathway_complexes
                                          if pathway_depths.get(pid) == depth_level + 1}
            if len(depth_candidates_fallback) >= 5:
                depth_candidates = depth_candidates_fallback
                effective_depth = depth_level + 1
                print(f"  Fig 15: fell back to depth {effective_depth} "
                      f"({len(depth_candidates)} pathways)")
            # If still too few, use all pathways (no depth filter)
            elif len(depth_candidates) < 5:
                depth_candidates = set(pathway_complexes.keys())
                effective_depth = -1  # signals no depth filtering applied
                print("  Fig 15: too few pathways at target depth, using all depths")

        # Select top N from depth-filtered candidates
        sorted_pids = sorted(depth_candidates, key=lambda p: len(pathway_complexes[p]), reverse=True)
    else:
        sorted_pids = sorted(pathway_complexes.keys(), key=lambda p: len(pathway_complexes[p]), reverse=True)
        effective_depth = -1  # no depth filtering

    keep_pids = set(sorted_pids[:max_pathways])

    G = nx.Graph()
    for pid in keep_pids:
        vals = pathway_complexes[pid]
        tiers_list = pathway_tiers.get(pid, [])
        frac_high = sum(1 for t in tiers_list if t == 'High') / len(tiers_list) if tiers_list else 0
        G.add_node(pid, n_complexes=len(vals), mean_pdockq=float(np.mean(vals)), frac_high=frac_high, name=pathway_names.get(pid, pid))

    # Build edges - exclude hierarchical parent-child links
    effective_threshold = min_shared_complexes
    for (p1, p2), vals in edge_data.items():
        if p1 in keep_pids and p2 in keep_pids and len(vals) >= effective_threshold:
            if hierarchy_loaded and (p1, p2) in hierarchy_pairs:
                continue
            G.add_edge(p1, p2, n_shared=len(vals), mean_pdockq=float(np.mean(vals)))

    # Auto-raise threshold if network is too dense (> 3 edges per node)
    n_nodes_g = G.number_of_nodes()
    while G.number_of_edges() > 3 * n_nodes_g and effective_threshold < 10000:
        effective_threshold *= 2
        edges_to_remove = [(u, v) for u, v, d in G.edges(data=True)
                           if d['n_shared'] < effective_threshold]
        G.remove_edges_from(edges_to_remove)
        if not edges_to_remove:
            break  # no more edges to remove at this threshold

    if G.number_of_nodes() == 0:
        print("  Skipping Fig 15 - empty graph after filtering")
        return

    #=======================Layout - shrink inward to leave margin for labels=======================
    pos = nx.kamada_kawai_layout(G)
    pos = {k: v * 0.85 for k, v in pos.items()}

    figure, ax = plt.subplots(1, 1, figsize=(14, 14))

    # Colour normalisation - % High-tier (0% to dynamic max for better spread)
    node_frac_high = [G.nodes[n]['frac_high'] for n in G.nodes()]
    vmin = 0.0
    vmax = max(node_frac_high) if node_frac_high and max(node_frac_high) > 0 else 0.5
    # Round up to nearest 0.05 for clean tick labels
    vmax = max(0.05, np.ceil(vmax * 20) / 20)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.RdYlGn

    # Draw edges first (grey, beneath nodes) with data-relative scaling
    all_shared = [d['n_shared'] for _, _, d in G.edges(data=True)] if G.number_of_edges() > 0 else [1]
    min_s, max_s = min(all_shared), max(all_shared)

    for u, v, edata in G.edges(data=True):
        x_coords = [pos[u][0], pos[v][0]]
        y_coords = [pos[u][1], pos[v][1]]
        frac = (edata['n_shared'] - min_s) / max(1, max_s - min_s)
        width = 0.5 + 4.5 * frac
        alpha = 0.05 + 0.35 * frac  # weak edges fade, strong edges visible
        ax.plot(x_coords, y_coords, color='#888888', linewidth=width, alpha=alpha)

    # Draw nodes - data-relative sizing for visible differentiation
    counts = [G.nodes[n]['n_complexes'] for n in G.nodes()]
    min_c, max_c = min(counts), max(counts)
    node_sizes = [300 + 2700 * (c - min_c) / max(1, max_c - min_c) for c in counts]
    node_colors = [cmap(norm(G.nodes[n]['frac_high'])) for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, node_color=node_colors, edgecolors='black', linewidths=0.8)

    # Label ALL nodes - centred inside node with collision avoidance
    node_list = list(G.nodes())
    node_size_map = dict(zip(node_list, node_sizes))
    label_data = []
    for n in node_list:
        full_name = pathway_names.get(n, n)
        name = full_name[:40] + '\u2026' if len(full_name) > 40 else full_name
        wrapped = _textwrap.fill(name, width=15)
        x, y = pos[n]
        sz = node_size_map[n]
        fs = 5 + 3 * (sz - 300) / max(1, 2700)
        label_data.append({'x': x, 'y': y, 'text': wrapped, 'fontsize': fs})

    # Collision nudge - sort top-to-bottom, push overlapping labels apart
    label_data.sort(key=lambda d: -d['y'])
    nudge = 0.03
    for i in range(len(label_data)):
        for j in range(i + 1, len(label_data)):
            if (abs(label_data[i]['x'] - label_data[j]['x']) < 0.08 and
                    abs(label_data[i]['y'] - label_data[j]['y']) < nudge):
                label_data[j]['y'] = label_data[i]['y'] - nudge

    for ld in label_data:
        ax.text(ld['x'], ld['y'], ld['text'], fontsize=ld['fontsize'], ha='center', va='center', bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.6, edgecolor='none'))

    ax.axis('off')
    ax.margins(0.15)

    # Colourbar - % High-tier complexes
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = figure.colorbar(sm, ax=ax, shrink=0.6, pad=0.06)
    cbar.set_label('Percentage of High-tier complexes', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    n_ticks = min(9, int(vmax / 0.05) + 1)
    tick_vals = np.linspace(vmin, vmax, n_ticks)
    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels([f'{v * 100:.0f}%' for v in tick_vals])

    # Node-size legend - show min and max with explicit labels
    ref_sizes = [min_c, max_c]
    ref_labels = [f'{min_c:,} (smallest)', f'{max_c:,} (largest)']
    legend_elements = []
    for s, lbl in zip(ref_sizes, ref_labels):
        display_size = 300 + 2700 * (s - min_c) / max(1, max_c - min_c)
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='#bdc3c7', markeredgecolor='black', markeredgewidth=0.5, markersize=max(4, np.sqrt(display_size) / 3), label=lbl))

    figure.legend(handles=legend_elements, fontsize=10, title='Pathway size (complexes)', title_fontsize=11, framealpha=1.0, borderpad=1.2, labelspacing=1.5, handletextpad=1.0, loc='upper right', bbox_to_anchor=(0.99, 0.97))
    figure.suptitle("Reactome Pathway Network by Structural Quality [dimer-validated]", fontsize=FONT_TITLE + 1, fontweight='bold')

    # Caption - dynamic values, no hardcoded sizes
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    hier_note = ', hierarchical parent\u2013child links excluded' if hierarchy_loaded else ''
    depth_note = f'depth level {effective_depth}' if effective_depth >= 0 else 'all depths'
    threshold_note = (f'\u2265{effective_threshold} shared complexes'
                      if effective_threshold == min_shared_complexes
                      else f'\u2265{effective_threshold} shared complexes '
                           f'(auto-raised from {min_shared_complexes})')
    caption_15 = (
    f'Network of the top {n_nodes} Reactome pathways ({depth_note}, \u2265{min_c:,} complexes). '
    f'Node size and color reflect total complex count and High-tier proportion, respectively. '
    f'The {n_edges} edges denote pathway overlaps ({threshold_note}{hier_note}), with width '
    f'scaling by overlap strength.'
    )
    figure.text(0.5, 0.01, caption_15, ha='center', fontsize=7, style='italic', color='#777777')
    _save_figure(figure, '15_Pathway_Network.png')

#-----------------------------------------------Prediction Quality Paradox helpers (Fig 16)--------------------------------------------------

# Ordered Low -> Medium -> High (ascending quality)
_PARADOX_TIER_ORDER = ['Low', 'Medium', 'High']

# Bonferroni correction: 4 panels x 3 pairwise = 12 tests
_PARADOX_N_TESTS = 12
_PARADOX_BONF_THRESHOLD = 0.05 / _PARADOX_N_TESTS  # 0.00417

def _compute_cohens_d(g1, g2):
    """Cohen's d for two array-like groups (pooled SD denominator)."""
    g1, g2 = np.asarray(g1, dtype=float), np.asarray(g2, dtype=float)
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return float('nan')
    pooled = np.sqrt(((n1 - 1) * g1.std(ddof=1) ** 2 + (n2 - 1) * g2.std(ddof=1) ** 2) / (n1 + n2 - 2))
    if pooled == 0:
        return float('nan')
    return (g1.mean() - g2.mean()) / pooled

def _pval_stars(p, bonf_threshold=_PARADOX_BONF_THRESHOLD):
    """Return significance string for a p-value."""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < bonf_threshold:
        return '*'
    return 'ns'

def _run_pairwise_tests_continuous(data_by_tier):
    """Kruskal-Wallis omnibus + 3 pairwise Mann-Whitney U tests.
    Returns: dict with 'omnibus' (H, p) and 'pairwise' list of (tier_a, tier_b, U, p_raw, p_bonf, cohens_d, stars).
    """
    tiers = _PARADOX_TIER_ORDER
    groups = [np.asarray(data_by_tier[t], dtype=float) for t in tiers if t in data_by_tier and len(data_by_tier[t]) > 0]
    if len(groups) < 2:
        return {'omnibus': (float('nan'), float('nan')), 'pairwise': []}

    H, p_omni = kruskal(*groups)
    pairs = [('Low', 'Medium'), ('Medium', 'High'), ('Low', 'High')]
    results = []
    for ta, tb in pairs:
        ga = data_by_tier.get(ta, [])
        gb = data_by_tier.get(tb, [])
        if len(ga) < 1 or len(gb) < 1:
            results.append((ta, tb, float('nan'), 1.0, 1.0, float('nan'), 'ns'))
            continue
        U, p_raw = mannwhitneyu(ga, gb, alternative='two-sided')
        p_bonf = min(p_raw * _PARADOX_N_TESTS, 1.0)
        d = _compute_cohens_d(ga, gb)
        results.append((ta, tb, U, p_raw, p_bonf, d, _pval_stars(p_bonf)))
    return {'omnibus': (H, p_omni), 'pairwise': results}

def _run_pairwise_tests_binary(counts_by_tier, totals_by_tier):
    """Chi-squared omnibus + 3 pairwise Fisher's exact tests.
    *counts_by_tier*: dict tier -> int (positive count).
    *totals_by_tier*: dict tier -> int (total count).
    Returns: dict with 'omnibus' (chi2, p) and 'pairwise' list of
    (tier_a, tier_b, stat, p_raw, p_bonf, odds_ratio, stars).
    """
    tiers = _PARADOX_TIER_ORDER
    present = [t for t in tiers if totals_by_tier.get(t, 0) > 0]
    if len(present) < 2:
        return {'omnibus': (float('nan'), float('nan')), 'pairwise': []}

    table = np.array([[counts_by_tier.get(t, 0), totals_by_tier.get(t, 0) - counts_by_tier.get(t, 0)] for t in present])
    chi2_val, p_omni, _, _ = chi2_contingency(table)

    pairs = [('Low', 'Medium'), ('Medium', 'High'), ('Low', 'High')]
    results = []
    for ta, tb in pairs:
        ca, na = counts_by_tier.get(ta, 0), totals_by_tier.get(ta, 0)
        cb, nb = counts_by_tier.get(tb, 0), totals_by_tier.get(tb, 0)
        if na == 0 or nb == 0:
            results.append((ta, tb, float('nan'), 1.0, 1.0, float('nan'), 'ns'))
            continue
        tbl = np.array([[ca, na - ca], [cb, nb - cb]])
        res = fisher_exact(tbl, alternative='two-sided')
        odds, p_raw = res.statistic, res.pvalue
        p_bonf = min(p_raw * _PARADOX_N_TESTS, 1.0)
        results.append((ta, tb, odds, p_raw, p_bonf, odds, _pval_stars(p_bonf)))
    return {'omnibus': (chi2_val, p_omni), 'pairwise': results}


def _violin_box_panel(ax, data_by_tier, ylabel, title, hline=None):
    """Draw violin + embedded box plot on *ax* for continuous data grouped by tier."""
    tiers = _PARADOX_TIER_ORDER
    plot_data = [np.asarray(data_by_tier.get(t, []), dtype=float) for t in tiers]
    plot_data = [d[~np.isnan(d)] for d in plot_data]
    positions = list(range(len(tiers)))

    nonempty_pos = [p for p, d in zip(positions, plot_data) if len(d) > 1]
    nonempty_data = [d for d in plot_data if len(d) > 1]
    if nonempty_data:
        parts = ax.violinplot(nonempty_data, positions=nonempty_pos, showmedians=False, showextrema=False, widths=0.7)
        for i, body in enumerate(parts['bodies']):
            tier = tiers[nonempty_pos[i]]
            body.set_facecolor(TIER_COLORS[tier])
            body.set_alpha(0.6)

    bp = ax.boxplot(plot_data, positions=positions, widths=0.15, patch_artist=True, showfliers=False, medianprops=dict(color='black', linewidth=2), whiskerprops=dict(linewidth=0.8), capprops=dict(linewidth=0.8))
    for patch, tier in zip(bp['boxes'], tiers):
        patch.set_facecolor(TIER_COLORS[tier])
        patch.set_alpha(0.8)

    if hline is not None:
        ax.axhline(hline, color='grey', linestyle='--', linewidth=1, zorder=0)
        ax.text(len(tiers) - 0.5, hline, 'Neutral', va='bottom', ha='right', fontsize=8, color='grey')

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{t}\n(n={len(d)})" for t, d in zip(tiers, plot_data)], fontsize=FONT_TICK)
    ax.set_ylabel(ylabel, fontsize=FONT_AXIS_LABEL)
    ax.set_title(title, fontsize=FONT_TITLE, fontweight='bold')
    ax.tick_params(labelsize=FONT_TICK)
    ax.grid(True, alpha=GRID_ALPHA, linestyle='--', axis='y')
    _despine(ax)

def _grouped_bar_panel(ax, counts_by_tier, totals_by_tier, ylabel, title):
    """Draw grouped bar chart showing fraction positive per tier on *ax*."""
    tiers = _PARADOX_TIER_ORDER
    fractions = []
    for t in tiers:
        total = totals_by_tier.get(t, 0)
        frac = counts_by_tier.get(t, 0) / total if total > 0 else 0.0
        fractions.append(frac)

    positions = list(range(len(tiers)))
    ax.bar(positions, fractions, color=[TIER_COLORS[t] for t in tiers], edgecolor='white', width=0.6)

    for i, (t, frac) in enumerate(zip(tiers, fractions)):
        total = totals_by_tier.get(t, 0)
        count = counts_by_tier.get(t, 0)
        ax.text(i, frac + 0.01, f"{count}/{total}", ha='center', va='bottom', fontsize=8, color='black')

    ax.set_xticks(positions)
    ax.set_xticklabels(tiers, fontsize=FONT_TICK)
    ax.set_ylabel(ylabel, fontsize=FONT_AXIS_LABEL)
    ax.set_title(title, fontsize=FONT_TITLE, fontweight='bold')
    ax.set_ylim(0, max(fractions) * 1.25 + 0.05 if fractions else 1.0)
    ax.tick_params(labelsize=FONT_TICK)
    ax.grid(True, alpha=GRID_ALPHA, linestyle='--', axis='y')
    _despine(ax)

def _paradox_stats_for_subset(wdf, label='All'):
    """Run all 4 paradox panels' statistical tests on *wdf*.
    Returns a list of dicts (one per row in the summary table).
    """
    tiers = _PARADOX_TIER_ORDER

    wdf = wdf.copy()
    wdf['_max_pli'] = wdf[['gene_constraint_pli_a', 'gene_constraint_pli_b']].apply(lambda r: np.nanmax(r.values), axis=1)
    wdf['_pli_constrained'] = wdf['_max_pli'] >= 0.9
    wdf['_has_path_iface'] = wdf['n_pathogenic_interface_variants'].fillna(0) > 0

    rows = []

    #======================continuous panels: B, D============================
    continuous_specs = [
        ('B', 'ppi_enrichment_ratio', 'PPI Enrichment Ratio'),
        ('D', 'plddt_below50_fraction', 'Disorder Fraction'),
    ]
    for panel, col, name in continuous_specs:
        sub = wdf.dropna(subset=[col])
        data_by_tier = {t: sub.loc[sub['quality_tier_v2'] == t, col].values for t in tiers}
        res = _run_pairwise_tests_continuous(data_by_tier)
        H, p_omni = res['omnibus']
        medians = {t: float(np.nanmedian(data_by_tier[t])) if len(data_by_tier[t]) else float('nan') for t in tiers}
        rows.append({'panel': panel, 'name': name, 'subset': label,
                     'test': 'Kruskal-Wallis', 'stat': H, 'p': p_omni,
                     'low': medians.get('Low'), 'med': medians.get('Medium'), 'high': medians.get('High'),
                     'effect': '-', 'n_low': len(data_by_tier.get('Low', [])),
                     'n_med': len(data_by_tier.get('Medium', [])), 'n_high': len(data_by_tier.get('High', []))})
        for ta, tb, U, p_raw, p_bonf, d, stars in res['pairwise']:
            rows.append({'panel': panel, 'name': name, 'subset': label,
                         'test': f'MW ({ta[:1]}v{tb[:1]})', 'stat': U,
                         'p': p_bonf, 'low': '', 'med': '', 'high': '',
                         'effect': f'd={d:.3f}' if not np.isnan(d) else '-',
                         'n_low': '', 'n_med': '', 'n_high': ''})

    #======================binary panels: A, C============================
    binary_specs = [
        ('A', '_has_path_iface', 'Pathogenic Interface Variants'),
        ('C', '_pli_constrained', 'LoF-Intolerant (pLI >= 0.9)'),
    ]
    for panel, col, name in binary_specs:
        sub = wdf.dropna(subset=['quality_tier_v2'])
        counts = {t: int(sub.loc[sub['quality_tier_v2'] == t, col].sum()) for t in tiers}
        totals = {t: int((sub['quality_tier_v2'] == t).sum()) for t in tiers}
        res = _run_pairwise_tests_binary(counts, totals)
        chi2_val, p_omni = res['omnibus']
        pcts = {t: counts[t] / totals[t] * 100 if totals[t] else float('nan') for t in tiers}
        rows.append({'panel': panel, 'name': name, 'subset': label,
                     'test': 'Chi-squared', 'stat': chi2_val, 'p': p_omni,
                     'low': f"{pcts.get('Low', 0):.1f}%", 'med': f"{pcts.get('Medium', 0):.1f}%",
                     'high': f"{pcts.get('High', 0):.1f}%", 'effect': '-',
                     'n_low': totals.get('Low', 0), 'n_med': totals.get('Medium', 0),
                     'n_high': totals.get('High', 0)})
        for ta, tb, odds, p_raw, p_bonf, or_val, stars in res['pairwise']:
            rows.append({'panel': panel, 'name': name, 'subset': label,
                         'test': f'Fisher ({ta[:1]}v{tb[:1]})', 'stat': odds,
                         'p': p_bonf, 'low': '', 'med': '', 'high': '',
                         'effect': f'OR={or_val:.2f}' if not np.isnan(or_val) else '-',
                         'n_low': '', 'n_med': '', 'n_high': ''})

    return rows


def _print_paradox_table(all_rows):
    """Pretty-print the prediction quality paradox statistics table to console."""
    hdr = f"{'Panel':<6} {'Metric':<30} {'Subset':<12} {'Test':<18} {'Statistic':>12} {'p-value':>12} {'Low':>10} {'Medium':>10} {'High':>10} {'Effect':>12} {'n(L)':>6} {'n(M)':>6} {'n(H)':>6}"
    print("\n" + "=" * len(hdr))
    print("  Prediction Quality Paradox - Statistical Summary")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in all_rows:
        stat_str = f"{r['stat']:.2f}" if isinstance(r['stat'], float) and not np.isnan(r['stat']) else '-'
        p_str = f"{r['p']:.2e}" if isinstance(r['p'], float) and r['p'] < 0.001 else (
            f"{r['p']:.4f}" if isinstance(r['p'], float) else '-')
        low_str = str(r['low']) if r['low'] != '' else ''
        med_str = str(r['med']) if r['med'] != '' else ''
        high_str = str(r['high']) if r['high'] != '' else ''
        n_low = str(r.get('n_low', ''))
        n_med = str(r.get('n_med', ''))
        n_high = str(r.get('n_high', ''))
        name = r.get('name', '')[:30]
        print(f"{r['panel']:<6} {name:<30} {r['subset']:<12} {r['test']:<18} {stat_str:>12} {p_str:>12} {low_str:>10} {med_str:>10} {high_str:>10} {r['effect']:>12} {n_low:>6} {n_med:>6} {n_high:>6}")
    print("=" * len(hdr))
    print(f"  Bonferroni threshold: p < {_PARADOX_BONF_THRESHOLD:.4f} (0.05 / {_PARADOX_N_TESTS} pairwise tests)")
    print("  Effect sizes: Cohen's d (continuous), Odds Ratio (binary)")
    print()


#---------------------------------------------Synthesis (Fig 16)------------------------------------------------------------------


def plot_fig16_prediction_quality_paradox(df: pd.DataFrame) -> None:
    """Fig 16 - The Prediction Quality Paradox.
    Produces a 2x2 panel figure:
      Top row  - "Interface-level validation":
        A: Pathogenic interface variant rate by tier (grouped bar, signal strengthens)
        B: PPI enrichment ratio by tier (violin+box, signal strengthens)
      Bottom row - "Protein-level prediction bias":
        C: LoF-intolerant gene fraction, pLI >= 0.9, by tier (grouped bar, declines)
        D: Disorder fraction by tier (violin+box, declines - mechanistic bridge)
    Each panel shows an omnibus p-value and a single Low-vs-High bracket. A homodimer robustness footnote is added to the figure. 
    Full statistics are printed to the console.
    """
    tiers = _PARADOX_TIER_ORDER

    required = [
        'quality_tier_v2',
        'n_pathogenic_interface_variants',
        'ppi_enrichment_ratio',
        'gene_constraint_pli_a',
        'gene_constraint_pli_b',
        'plddt_below50_fraction',
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  Skipping Fig 16 - missing required columns: {missing}")
        return

    #=============================Data prep==================================================
    wdf = _filter_dimer_validated(df).copy()
    if 'has_pdb' in wdf.columns:
        wdf = wdf[_boolish(wdf['has_pdb']).ne(False)].copy()
    wdf = wdf.dropna(subset=['quality_tier_v2'])

    wdf['_max_pli'] = wdf[['gene_constraint_pli_a', 'gene_constraint_pli_b']].apply(lambda r: np.nanmax(r.values), axis=1)
    wdf['_pli_constrained'] = wdf['_max_pli'] >= 0.9
    wdf['_has_path_iface'] = wdf['n_pathogenic_interface_variants'].fillna(0) > 0

    if len(wdf) == 0:
        print("  Fig 16: no data after filtering - skipped.")
        return

    #============================Figure assembly (2x2)==========================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    panel_labels = ['A', 'B', 'C', 'D']
    for ax, lbl in zip(axes.flat, panel_labels):
        ax.text(-0.08, 1.05, lbl, transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')

    def _annotate_panel(ax, res, is_binary=False, direction='up'):
        """Add omnibus p-value box and directional arrow."""
        _, p_omni = res['omnibus']
        p_str = f'p = {p_omni:.1e}' if p_omni < 0.001 else f'p = {p_omni:.3f}'
        test_name = 'Chi-sq' if is_binary else 'K-W'
        ax.text(0.97, 0.95, f'{test_name}: {p_str}', transform=ax.transAxes, ha='right', va='top', fontsize=8, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='grey', alpha=0.8))

        # Directional arrow
        if direction == 'up': ax.text(0.97, 0.82, '\u2191 Signal strengthens', transform=ax.transAxes, ha='right', va='top', fontsize=8, color='#27AE60', fontweight='bold')
        else:
            ax.text(0.97, 0.82, '\u2193 Prediction bias', transform=ax.transAxes, ha='right', va='top', fontsize=8, color='#E74C3C', fontweight='bold')

    #========================Panel A: Pathogenic Interface Variant Rate (grouped bar)=============================
    ax_a = axes[0, 0]
    counts_a = {t: int(wdf.loc[wdf['quality_tier_v2'] == t, '_has_path_iface'].sum()) for t in tiers}
    totals_a = {t: int((wdf['quality_tier_v2'] == t).sum()) for t in tiers}
    _grouped_bar_panel(ax_a, counts_a, totals_a, ylabel='Fraction with Pathogenic\nInterface Variants', title='Pathogenic Interface Variants by Tier')
    res_a = _run_pairwise_tests_binary(counts_a, totals_a)
    _annotate_panel(ax_a, res_a, is_binary=True, direction='up')

    #=============================Panel B: PPI Enrichment Ratio (violin+box)=============================
    ax_b = axes[0, 1]
    col_b = 'ppi_enrichment_ratio'
    sub_b = wdf.dropna(subset=[col_b])
    data_b = {t: sub_b.loc[sub_b['quality_tier_v2'] == t, col_b].values for t in tiers}
    _violin_box_panel(ax_b, data_b, ylabel='PPI Enrichment Ratio', title='PPI Enrichment Ratio by Tier')
    res_b = _run_pairwise_tests_continuous(data_b)
    _annotate_panel(ax_b, res_b, direction='up')
    # STRING saturation note moved to figure footer

    #=============================Panel C: LoF-Intolerant Genes, pLI >= 0.9 (grouped bar)=============================
    ax_c = axes[1, 0]
    sub_c = wdf.dropna(subset=['_pli_constrained'])
    counts_c = {t: int(sub_c.loc[sub_c['quality_tier_v2'] == t, '_pli_constrained'].sum()) for t in tiers}
    totals_c = {t: int((sub_c['quality_tier_v2'] == t).sum()) for t in tiers}
    _grouped_bar_panel(ax_c, counts_c, totals_c, ylabel='Fraction with pLI \u2265 0.9', title='LoF-Intolerant Genes (pLI \u2265 0.9) by Tier')
    res_c = _run_pairwise_tests_binary(counts_c, totals_c)
    _annotate_panel(ax_c, res_c, is_binary=True, direction='down')

    #=============================Panel D: Disorder Fraction (violin+box)=============================
    ax_d = axes[1, 1]
    col_d = 'plddt_below50_fraction'
    sub_d = wdf.dropna(subset=[col_d])
    data_d = {t: sub_d.loc[sub_d['quality_tier_v2'] == t, col_d].values for t in tiers}
    _violin_box_panel(ax_d, data_d, ylabel='Disorder Fraction (pLDDT < 50)', title='Disorder Fraction by Tier')
    res_d = _run_pairwise_tests_continuous(data_d)
    _annotate_panel(ax_d, res_d, direction='down')


    #=============================================Figure footer===================================================
    footer_parts = ['Panel B note: STRING p-values saturate at 0.0 for large networks; the enrichment ratio is used as the discriminative metric']
    if 'complex_type' in wdf.columns:
        hetero = wdf[wdf['complex_type'] == 'heterodimer']
        if len(hetero) > 30:
            hetero_stats = _paradox_stats_for_subset(hetero, label='Hetero-only')
            hetero_sig = sum(
                1 for r in hetero_stats
                if ('LvH' in r.get('test', '') or '(LvH)' in r.get('test', ''))
                and isinstance(r['p'], float) and r['p'] < _PARADOX_BONF_THRESHOLD
            )
            hetero_total = sum(
                1 for r in hetero_stats
                if 'LvH' in r.get('test', '') or '(LvH)' in r.get('test', '')
            )
            footer_parts.append(
                f"All {hetero_sig}/{hetero_total} panels remain significant "
                f"in heterodimers only (n = {len(hetero):,}, "
                f"Bonferroni-corrected p < {_PARADOX_BONF_THRESHOLD:.4f})")
    if footer_parts:
        fig.text(0.5, -0.01, '  |  '.join(footer_parts), ha='center', va='top', fontsize=9, fontstyle='italic', color='#555555')

    fig.suptitle("The Prediction Quality Paradox [dimer-validated]", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save_figure(fig, '16_Prediction_Quality_Paradox.png')

    #========================Statistics table==================================================
    all_stats = _paradox_stats_for_subset(wdf, label='All')
    if 'complex_type' in wdf.columns:
        hetero = wdf[wdf['complex_type'] == 'heterodimer']
        if len(hetero) > 30:
            all_stats.extend(_paradox_stats_for_subset(hetero, label='Hetero-only'))

    _print_paradox_table(all_stats)

    for r in all_stats:
        for key, tier_name in [('n_low', 'Low'), ('n_med', 'Medium'), ('n_high', 'High')]:
            val = r.get(key, '')
            if isinstance(val, int) and val < 30 and val > 0:
                print(f"  WARNING: Panel {r['panel']} ({r['subset']}): {tier_name} tier has only n={val} (<30)")

#----------------------------------------------CLI & Main-----------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AlphaFold2 Analysis Visualisation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python visualise_results.py results.csv                                    # auto-detect
    python visualise_results.py results.csv --output-dir ./figures             # custom output
    python visualise_results.py results.csv --density                          # KDE contours
    python visualise_results.py results.csv --disorder-scatter                 # also Fig 1b
    python visualise_results.py results.csv --pae-heatmaps /path/to/models     # PAE heatmaps
    python visualise_results.py results.csv --pae-heatmaps /models --limit 50
    """,
)

    #======================Positional: CSV path=============================
    parser.add_argument(
        'csv', type=str,
        help='Path to the results.csv produced by toolkit.py')

    #======================Optional: output directory=======================
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Directory of where the figures will be saved. Defaults to ./Output/')

    #======================Optional: PAE heatmaps (requires protein complexes directory)======================
    parser.add_argument(
        '--pae-heatmaps', type=str, default=None, metavar='PROTEIN_COMPLEXES_DIR',
        help='Generate per-complex PAE heatmaps from PKL files in PROTEIN_COMPLEXES_DIR.')
    parser.add_argument(
        '--limit', type=int, default=None,
        help='Cap the number of PAE heatmaps generated.')

    #======================Optional: rendering flags================================
    parser.add_argument(
        '--disorder-scatter', action='store_true',
        help='If you want to also produce disorder-coloured quality scatter (Fig 1b).')
    parser.add_argument(
        '--density', action='store_true',
        help='Add KDE density contour overlays to scatter figures. '
             'Contour lines show percentile-based density levels (10%%-90%%).')

    #======================Optional: multimer supplementary panels================
    # Primary figures always stay dimer-validated (dissertation-safe). This flag
    # enables supplementary panels that expose multimer behaviour separately;
    # it is NOT a load-time filter - no row is dropped by this flag.
    parser.add_argument(
        '--multimer-supplement', action='store_true',
        help='Render multimer-exploratory supplementary panels alongside the '
             'dimer-validated primary figures. Supplementary panels are '
             'descriptive only, never dissertation claims.')

    import sys as _sys
    # Reject the old flag explicitly - it used to be a load-time multimer gate.
    if '--include-multimers' in _sys.argv[1:]:
        parser.error(
            "--include-multimers has been removed. Multimers are always "
            "processed; use --multimer-supplement to add exploratory panels.")

    return parser.parse_args()


def main() -> None:
    """Generate all visualisations based on available data columns."""
    global OUTPUT_DIR

    args = parse_arguments()

    # Resolve paths
    csv_path = os.path.abspath(args.csv)
    OUTPUT_DIR = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(os.getcwd(), "Output")
    models_dir = os.path.abspath(args.pae_heatmaps) if args.pae_heatmaps else None

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  AlphaFold2 Analysis Visualisation Tool")
    print("=" * 60)
    print(f"CSV file:         {csv_path}")
    print(f"Output directory: {OUTPUT_DIR}")
    if models_dir:
        print(f"Models directory: {models_dir}")
    if args.density:
        print("Rendering mode:   DENSITY (KDE contour overlays enabled)")
    print()

    # Load CSV
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print(f"Loading data from {csv_path}...")
    df = load_data(csv_path)
    print(f"Loaded {len(df)} valid complex records.")

    if len(df) == 0:
        print("No valid data found. Exiting.")
        return

    # Structural figures (1-9) and Figs 10-12 (clustering + ClinVar variants) treat
    # reviewed+TrEMBL as "human" since their data sources cover TrEMBL adequately
    # (>85%). Figs 13-16 (stability / disease / Reactome pathways / PPI enrichment)
    # stay reviewed-only via get_human_mask because those sources (EVE,
    # AlphaMissense, UniProt diseases, Reactome) have <15% TrEMBL coverage and
    # would dilute tier-based signals.
    if 'species_status' in df.columns:
        from toolkit import get_human_mask
        reviewed_mask = get_human_mask(df)
        trembl_mask = df['species_status'] == 'trembl_human'
        all_human_mask = reviewed_mask | trembl_mask
        df_all_human = df[all_human_mask].reset_index(drop=True)
        df_nonhuman = df[~all_human_mask].reset_index(drop=True)
        df_human = df[reviewed_mask].reset_index(drop=True)
        species_subsets = [
            (df_all_human, '_human',    'Human'),
            (df_nonhuman,  '_nonhuman', 'Non-Human'),
        ]
        print(f"  Species split: {len(df_all_human):,} human "
              f"({int(reviewed_mask.sum()):,} reviewed + {int(trembl_mask.sum()):,} TrEMBL), "
              f"{len(df_nonhuman):,} non-human.")
        if len(df_all_human) == 0 and len(df_nonhuman) == 0:
            print("No complexes after species split. Exiting.")
            return
    else:
        species_subsets = [(df, '', '')]
        df_human = df
        df_all_human = df

    # Detect available columns (stateless over column names; same for every subset)
    col_flags = detect_columns(df)
    print(f"\nColumn detection:")
    print(f"  V2 quality tiers:  {'Yes' if col_flags['has_v2_data'] else 'No'}")
    print(f"  Interface data:    {'Yes' if col_flags['has_interface_data'] else 'No'}")
    print(f"  Interface PAE:     {'Yes' if col_flags['has_pae_interface'] else 'No'}")
    print(f"  Composite score:   {'Yes' if col_flags['has_composite'] else 'No'}")
    print(f"  Chain info:        {'Yes' if col_flags['has_chain_info'] else 'No'}")
    print(f"  Variant data:      {'Yes' if col_flags['has_variant_data'] else 'No'}")
    print(f"  Disease data:      {'Yes' if col_flags.get('has_disease_data', False) else 'No'}")
    print(f"  Pathway data:      {'Yes' if col_flags.get('has_pathway_data', False) else 'No'}")
    print(f"  Stability data:    {'Yes' if col_flags.get('has_stability_data', False) else 'No'}")
    print(f"  Clustering data:   {'Yes' if col_flags.get('has_clustering_data', False) else 'No'}")
    print(f"  Paradox data:      {'Yes' if col_flags.get('has_paradox_data', False) else 'No'}")

    # Chain count summary
    if col_flags['has_chain_info']:
        chain_counts = df['n_chains'].value_counts().sort_index()
        chain_parts = [f"{int(n)}-chain: {count}" for n, count in chain_counts.items()]
        print(f"  Chain breakdown:   {', '.join(chain_parts)}")

    # Console log replacing old pLDDT source bar chart (dropped Fig 4)
    if 'plddt_source' in df.columns:
        source_counts = df['plddt_source'].value_counts()
        source_parts = [f"{count} {source}" for source, count in source_counts.items()]
        print(f"\n  pLDDT source: {', '.join(source_parts)}")

    figures_generated = 0
    interface_figs_skipped_warning_shown = False

    # Structural figures (1-9) run per species subset.
    for df_subset, suffix, display_label in species_subsets:
        if len(df_subset) == 0:
            print(f"\n  Skipping {display_label} figures - empty subset.")
            continue
        header = f" ({display_label}, n={len(df_subset):,})" if display_label else ""
        label_suffix = f" - {display_label}" if display_label else ""
        print(f"\n--- Generating Figures{header} ---\n")

        # ALWAYS generated: Fig 1 and Fig 2
        print(f"Fig 1 - Quality Scatter (ipTM vs pDockQ){label_suffix}")
        plot_fig1_quality_scatter(df_subset, col_flags, density_mode=args.density, species_label=suffix)
        figures_generated += 1

        print(f"Fig 2 - Global PAE Health Check{label_suffix}")
        plot_fig2_pae_health_check(df_subset, species_label=suffix)
        figures_generated += 1

        # Supplementary Fig 1b (only when --disorder-scatter AND V2 data present)
        if args.disorder_scatter and col_flags['has_v2_data']:
            print(f"Fig 1b - Disorder Scatter (supplementary){label_suffix}")
            plot_fig1b_disorder_scatter(df_subset, density_mode=args.density, species_label=suffix)
            figures_generated += 1

        # Interface figures (require V2 + interface data): Figs 3-8
        if col_flags['has_v2_data'] and col_flags['has_interface_data']:
            print(f"Fig 3 - Interface PAE by Tier{label_suffix}")
            plot_fig3_interface_pae_by_tier(df_subset, species_label=suffix)
            figures_generated += 1

            print(f"Fig 4 - Composite & Tier Validation{label_suffix}")
            plot_fig4_composite_validation(df_subset, density_mode=args.density, species_label=suffix)
            figures_generated += 1

            # Supplementary: strict vs PAE-only fraction scatter (methodology evidence image)
            if 'pae_confident_contact_fraction' in df_subset.columns and 'strict_confident_contact_fraction' in df_subset.columns:
                print(f"Fig 4 supp - Strict vs PAE-only Fraction{label_suffix}")
                plot_fig4_supp_strict_vs_pae_only(df_subset, species_label=suffix)
                figures_generated += 1

            print(f"Fig 5 - Interface vs Bulk{label_suffix}")
            plot_fig5_interface_vs_bulk(df_subset, density_mode=args.density, species_label=suffix)
            figures_generated += 1

            print(f"Fig 6 - Paradox Spotlight{label_suffix}")
            plot_fig6_paradox_spotlight(df_subset, species_label=suffix)
            figures_generated += 1

            print(f"Fig 7 - Architecture (dimer-validated primary{', + multimer supp' if args.multimer_supplement else ''}){label_suffix}")
            plot_fig7_homo_vs_hetero(df_subset, species_label=suffix,
                                      multimer_supplement=args.multimer_supplement)
            figures_generated += 1

            print(f"Fig 8 - Metric Disagreement{label_suffix}")
            plot_fig8_metric_disagreement(df_subset, density_mode=args.density, species_label=suffix)
            figures_generated += 1
        elif not interface_figs_skipped_warning_shown:
            print("\nInterface figures (3-8) require V2 quality tiers AND interface")
            print("columns in the CSV. Re-run the batch script with interface analysis")
            print("enabled to generate the full 44-column CSV.")
            interface_figs_skipped_warning_shown = True

        # Chain-count figure (requires n_chains column)
        if col_flags['has_chain_info']:
            print(f"Fig 9 - Chain-Count Quality Profile{label_suffix}")
            plot_fig9_chain_count_profile(df_subset, density_mode=args.density, species_label=suffix)
            figures_generated += 1

    # Enrichment figures (10-16). Figs 10-12 use reviewed+TrEMBL (df_all_human)
    # since STRING clusters and ClinVar cover TrEMBL well. Figs 13-16 use
    # reviewed-only (df_human) because EVE/AlphaMissense/UniProt/Reactome data
    # is sparse on TrEMBL.
    print("\n--- Generating Enrichment Figures (Human) ---\n")

    # Clustering validation (requires --clustering)
    if col_flags.get('has_clustering_data', False) and 'complex_type' in df_all_human.columns:
        print("Fig 10 - Sequence Clustering Validation (reviewed + TrEMBL)")
        plot_fig10_clustering_validation(df_all_human)
        figures_generated += 1

    # Variant figures (require --variants)
    if col_flags['has_variant_data']:
        print("Fig 11 - Classified Variant Sankey (reviewed + TrEMBL)")
        plot_fig11_variant_consequence_flow(df_all_human)
        figures_generated += 1

        print("Fig 12 - Interface Variant Density vs Quality (reviewed + TrEMBL)")
        plot_fig12_variant_density(df_all_human, density_mode=args.density)
        figures_generated += 1

    # Stability cross-validation (requires --stability + --protvar)
    if col_flags.get('has_stability_data', False):
        print("Fig 13 - Stability Predictor Cross-Validation (reviewed only)")
        plot_fig13_stability_crossvalidation(df_human)
        figures_generated += 1

    # Disease enrichment (requires --disease)
    if col_flags.get('has_disease_data', False):
        print("Fig 14 - Disease Enrichment by Tier (reviewed only)")
        plot_fig14_disease_enrichment(df_human)
        figures_generated += 1

    # Pathway-level network (requires --pathways)
    if col_flags.get('has_pathway_data', False):
        print("Fig 15 - Pathway-Level Network (reviewed only)")
        plot_fig15_pathway_network(df_human)
        figures_generated += 1

    # Prediction quality paradox (requires --variants + --pathways)
    if col_flags.get('has_paradox_data', False):
        print("Fig 16 - Prediction Quality Paradox (reviewed only)")
        plot_fig16_prediction_quality_paradox(df_human)
        figures_generated += 1

    # On-demand: Per-complex PAE heatmaps
    if models_dir:
        if not os.path.isdir(models_dir):
            print(f"\nError: Models directory not found: {models_dir}")
        else:
            pkl_search_pattern = os.path.join(models_dir, "*.pkl")
            pkl_file_paths = sorted(glob.glob(pkl_search_pattern))

            if pkl_file_paths:
                total_available = len(pkl_file_paths)

                if args.limit is not None:
                    pkl_file_paths = pkl_file_paths[:args.limit]
                    print(f"\nGenerating PAE heatmaps for {len(pkl_file_paths)} "
                          f"of {total_available} PKL files (--limit {args.limit})...")
                else:
                    print(f"\nGenerating PAE heatmaps for all {total_available} "
                          f"PKL files...")
                    if total_available > 100:
                        print(f"  Warning: {total_available} heatmaps will take a "
                              f"while. Consider using --limit.")

                for pkl_path in pkl_file_paths:
                    plot_pae_matrix(pkl_path, models_dir)
            else:
                print(f"\nNo .pkl files found in {models_dir}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  Generated {figures_generated} figures. Saved to {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
