#!/usr/bin/env python3
"""AlphaFold2-Multimer analysis visualisation tool.

Reads the consolidated ``results.csv`` produced by ``toolkit.py`` and writes a
suite of PNG figures to the output directory. It performs no metric
computation: every value plotted is read from the CSV. The figure set is
selected automatically from the columns present, and each figure applies a
named population filter (from ``visualise_filters.py``) and skips gracefully,
with a message, when a required column is absent.

Project context
    Final stage of the pipeline: ``toolkit.py`` assembles the CSV, this module
    turns it into figures. It reads the population/scope columns
    (``tier_scope``, ``composite_is_calibrated``, ``partial_reason``,
    ``species_status``) to restrict each figure to the appropriate subset, and
    the optional annotation blocks (variants, disease, pathways, clustering,
    stability) to populate the corresponding figures when present.

Role in the submitted dissertation
    Dissertation-supporting and operational: this module generates the figures
    presented in the submitted Results. The output filenames use the
    dissertation's Results numbering (``Fig_1``-``Fig_8_*.png`` correspond to
    the dissertation's Figures 1-5 and 7-9); the dissertation's Figure 6, the
    worked single-complex examples, is produced separately by
    ``render_complex_summary.py``. The full filename-to-figure mapping is in
    ``Docs/Toolkit_Commands_List.md``. The internal ``plot_figN`` function
    names retain an older numbering that no longer matches the output
    filenames, so a function name should not be used to identify a figure.

Scope
    The figures visualise model-confidence patterns, the population structure
    of the dataset, and biological-annotation context. They support
    prioritisation and interpretation; they do not establish that a predicted
    interface occurs biologically or that a structure is experimentally
    correct. Calibrated interpretation applies to dimers
    (``tier_scope == 'dimer_validated'``); multimer and species-split panels
    are descriptive or exploratory and were not calibrated for the submitted
    claims.

Additional functionality
    Beyond the submitted figure set, the tool emits supplementary ``*_supp_*``
    figures, species-subset variants (``--human-supplement`` /
    ``--nonhuman-supplement``), a multimer-stoichiometry supplement, KDE
    density overlays (``--density``), a recoverability dashboard, and on-demand
    per-complex PAE heatmaps (``--pae-heatmaps``). At large N the scatter
    figures adapt point size and alpha and rasterise the artist so exported
    files stay bounded while still drawing one mark per complex.

Dependencies
    ``read_af2_nojax.py`` (JAX-free PKL loading), ``pdockq.py`` (chain offsets
    for PAE heatmaps), pandas, numpy, matplotlib and scipy; seaborn and
    networkx are optional.

The complete command reference is in ``Docs/Toolkit_Commands_List.md`` and the
output fields are defined in ``Docs/OUTPUT_SCHEMA.md``.
"""

import os
import glob
import re
import time
import argparse
from typing import Optional, Tuple
import numpy as np
import pandas as pd

# Canonical filter masks + parse_boolish / numeric helpers live in a sibling
# module so each filter is independently testable and reusable.
from visualise_filters import (
    parse_boolish,
    split_interface_flags,
    apply_filter,
    require_columns,
    FILTER_REGISTRY,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.ticker import MaxNLocator
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

# Composite-score cut-offs that reclassify v1 into quality_tier_v2 (must match
# the corresponding constants in toolkit.py). How these cut-offs were selected
# from the development set and checked for stability is documented in the
# dissertation's Appendix B (Table B.1).
UPGRADE_LOW_THRESHOLD = 0.64
UPGRADE_MEDIUM_THRESHOLD = 0.85
DOWNGRADE_HIGH_THRESHOLD = 0.63

# Composite screening bands. The classification axis is quality_tier_v2 and the
# screening axis is composite_screen_status; the two answer different questions
# and must never be substituted for one another.
COMPOSITE_SCREEN_BANDS = {
    "strong_screen_candidate":   (0.85, None),
    "moderate_screen_candidate": (0.63, 0.85),
    "weak_screen_candidate":     (None, 0.63),
    "unavailable":               (None, None),
}

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

# Tier-scope labels and the dimer-only default filter. Calibrated interpretation
# applies to dimers; multimer rows are treated as provisional.
TIER_SCOPE_DIMER = 'dimer_validated'
TIER_SCOPE_MULTIMER = 'multimer_provisional'
DIMER_STOICHIOMETRIES = ('A2', 'AB')
# Caption policy: every figure must advertise its scope as one of these literals.
CAPTION_SCOPE_DIMER = 'dimer-validated'
CAPTION_SCOPE_ALL_N = 'all-N descriptive'
CAPTION_SCOPE_MULTIMER = 'multimer exploratory'
# Corpus scope captions: each figure's title/filename includes one of these
# literals so the reader can tell at a glance which subset of the full corpus a
# figure speaks for.
CAPTION_SCOPE_CALIBRATED_DIMER = 'calibrated dimers'
CAPTION_SCOPE_CALIBRATED_A2_AB = 'calibrated A2/AB dimers'  # architecture figures only — narrower than calibrated_dimer because A2/AB-only.
CAPTION_SCOPE_RECOVERABLE_ALL_N = 'recoverable all-N descriptive'  # PAE-health and chain-count figures — the structurally usable subset, not every row.
CAPTION_SCOPE_CALIBRATED_HUMAN_BROAD = 'calibrated dimer x human'
CAPTION_SCOPE_CALIBRATED_HUMAN_STRICT = 'calibrated dimer x reviewed-human'
CAPTION_SCOPE_PARTIAL = 'input recoverability diagnostic'
CAPTION_SCOPE_SCREENING = 'screening / prioritisation'
CAPTION_SCOPE_CORPUS_FUNNEL = 'dataset audit / analysis population definition'

# -------------------------------------------------------------------------
# Human-readable display labels for code tokens.
# Canonical values stay unchanged in DataFrame logic and filter names.
# This dict only governs how those tokens are RENDERED in figure titles,
# axis labels, legends, and captions.
# -------------------------------------------------------------------------
DISPLAY_LABELS: dict[str, str] = {
    # Composite screen status tokens (4)
    'strong_screen_candidate':   'Strong screen candidate',
    'moderate_screen_candidate': 'Moderate screen candidate',
    'weak_screen_candidate':     'Weak screen candidate',
    'unavailable':               'Unavailable',
    # Filter / population tokens
    'all_rows':                  'All rows',
    'recoverable':               'Recoverable structural rows',
    'calibrated_dimer':          'Calibrated dimers',
    'calibrated_human_broad':    'Reviewed/TrEMBL human calibrated dimers',
    'calibrated_human_strict':   'Reviewed-human calibrated dimers',
    'composite_status_present':  'Composite-status-present rows',
    'composite_screenable':      'Composite-screenable rows',
    'partial_error':             'Partial/error rows',
    'multimer_exploratory':      'Multimer exploratory rows',
    # Species (5)
    'reviewed_human':            'Reviewed human',
    'trembl_human':              'TrEMBL human',
    'non_human':                 'Non-human',
    'ambiguous':                 'Ambiguous species',
    'unknown':                   'Unknown species',
    # tier_scope (2)
    'dimer_validated':           'Dimer-validated',
    'multimer_provisional':      'Multimer-provisional',
    # Full partial_reason vocabulary; the recoverability dashboard renders every value.
    '':                                  'Calibrated (no failure)',
    'pdb_io_error':                      'PDB I/O error',
    'pdb_decompression_error':           'PDB decompression failure',
    'pdb_parse_error':                   'PDB parse error',
    'pdb_no_chains':                     'PDB has no chains',
    'pkl_io_error':                      'PKL I/O error',
    'pkl_decompression_error':           'PKL decompression failure',
    'pkl_unpickle_error':                'PKL unpickle error',
    'pkl_loaded_missing_iptm':           'PKL loaded but missing ipTM',
    'pkl_loaded_missing_pae':            'PKL loaded but missing PAE',
    'no_positive_interface_contacts':    'No positive interface contacts',
    'missing_required_composite_inputs': 'Missing required composite inputs',
    'worker_exception':                  'Worker exception',
    'unreadable_pdb_or_structure_input': 'Unreadable PDB or structure input (legacy)',
    'missing_pkl_or_pkl_unreadable':     'PKL missing or unreadable (legacy)',
    'incomplete_input':                  'Incomplete input',
}

_unknown_display_warned: set = set()


def _display(token) -> str:
    """Return a human-readable label for a code token; fall back to title-case.

    Unknown tokens emit a one-time warning to stdout so missing coverage is
    surfaced during development. Pass-throughs to str() handle non-string
    inputs (numpy strings, etc.).
    """
    key = str(token) if token is not None else ''
    if key in DISPLAY_LABELS:
        return DISPLAY_LABELS[key]
    if key not in _unknown_display_warned:
        _unknown_display_warned.add(key)
        print(f"  Note: DISPLAY_LABELS missing entry for token '{key}'; "
              f"falling back to title-case.")
    return key.replace('_', ' ').strip().capitalize()


#-----------------------------------------------Infrastructure helpers--------------------------------------------------------

_LEGACY_CSV_WARNED = False


def _derive_tier_scope(df: pd.DataFrame) -> pd.Series:
    """Derive tier_scope for CSVs written before that column existed.

    Such CSVs lack `tier_scope` and `schema_version`. Rows with `n_chains == 2`
    are treated as dimer_validated and everything else as multimer_provisional.
    Emits a one-time warning the first time this is invoked.
    """
    global _LEGACY_CSV_WARNED
    if not _LEGACY_CSV_WARNED:
        print("  Warning: loaded CSV lacks the schema_version/tier_scope columns. "
              "Deriving tier_scope from n_chains for backward compatibility.")
        _LEGACY_CSV_WARNED = True
    if 'n_chains' in df.columns:
        n_chains = pd.to_numeric(df['n_chains'], errors='coerce')
        return np.where(n_chains == 2, TIER_SCOPE_DIMER, TIER_SCOPE_MULTIMER)
    return np.full(len(df), TIER_SCOPE_DIMER)


def _filter_dimer_validated(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows with tier_scope == 'dimer_validated'.

    The default population for figures whose thresholds were calibrated against
    dimers; calibrated interpretation does not extend to larger assemblies.
    """
    if 'tier_scope' not in df.columns:
        return df
    return df[df['tier_scope'] == TIER_SCOPE_DIMER].reset_index(drop=True)


def _boolish(series: pd.Series) -> pd.Series:
    """Vectorised parse_boolish over a Series; non-parseable cells become NaN.
    Use anywhere a boolean column round-tripped through CSV may arrive as the
    strings "True"/"False"/"0"/"1"/"yes"/"no" — `.astype(bool)` would coerce
    "False" to True (non-empty string is truthy).
    """
    return series.map(parse_boolish)


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


# Never let adaptive scatter PathCollection objects define tier legend
# markers. Adaptive markers are intentionally tiny/transparent at
# production scale; legends must use fixed-size opaque proxy artists from
# _build_tier_legend_handles() (or an equivalent Line2D/Patch proxy).
def _adaptive_scatter_params(n: int) -> Tuple[float, float, bool]:
    """Return (point_size, alpha, rasterize) scaled to dataset size.

    The third value indicates whether the scatter artist should be rasterised.
    At large N (>=50k) we rasterise so the saved PNG stays bounded in size even
    though we still draw one mark per complex: rasterising the matplotlib artist
    is a backend rendering choice, not a binning or density-raster decision (the
    figures remain one-mark-per-complex scatters, never hexbin or 2D-histogram).
    Args:
        n: Number of points to be plotted.
    Returns:
        Tuple of (size, alpha, rasterize) for use in axes.scatter().
    """
    if n < 1_000:
        return (40, 0.70, False)
    elif n < 10_000:
        return (16, 0.50, False)
    elif n < 50_000:
        return (6, 0.32, True)
    elif n < 200_000:
        return (3, 0.22, True)
    else:
        return (1.2, 0.14, True)

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
        # Corpus schema flags (population and calibration status)
        'has_partial_reason':       'partial_reason' in columns,
        'has_calibration_flag':     'composite_is_calibrated' in columns,
        'has_composite_screening':  'composite_screen_status' in columns,
        'has_species_status':       'species_status' in columns,
        'has_scope_columns':        {'tier_scope', 'composite_is_calibrated',
                                     'partial_reason'}.issubset(columns),
    }

def load_data(csv_path: str, legacy_mode: bool = False) -> pd.DataFrame:
    """Load the analysis CSV into a pandas DataFrame.

    Default mode: coerces numerics, normalises complex_type case, splits
    interface_flags on comma/pipe with exact-token matching, and derives
    tier_scope on CSVs that predate that column. NO rows are dropped and NO
    NaNs are filled — the per-figure filters in visualise_filters.py handle
    exclusion explicitly with logged before/after counts. This all-rows
    behaviour is the one used for the submitted analyses.

    Legacy mode (--legacy-mode): re-enables the older destructive drop of rows
    with missing/zero ipTM and the NaN->0 fill on pDockQ. It does NOT restore
    old v2 thresholds, captions, or figure filtering, and is retained only for
    reproducing the older load behaviour on older CSVs.

    Also:
      - Case normalisation on complex_type (Homodimer -> homodimer).
      - Splits interface_flags into individual boolean columns by re-splitting
        on both comma and pipe and matching exact tokens (no substring false
        positives).

    Args:
        csv_path: Path to the CSV produced by toolkit.py.
        legacy_mode: If True, drop rows with missing/zero ipTM and fill pDockQ
            NaNs with 0. Default False.
    Returns:
        Cleaned DataFrame.
    """
    df = pd.read_csv(csv_path)

    # Coerce key numeric columns.
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

    # Expand interface_flags into individual boolean columns using exact-token
    # matching (split on comma OR pipe, then `token in {flags}`). The audit
    # confirmed the final corpus is comma-delimited, but accepting pipes too
    # keeps the parser robust against older or hand-edited CSVs.
    ALL_KNOWN_FLAGS = [
        'small_interface', 'sparse_interface', 'asymmetric_interface',
        'interface_better_than_bulk', 'low_interface_confidence',
        'paradox_confident_disorder', 'paradox_artefactual', 'metric_disagreement',
    ]
    if 'interface_flags' in df.columns:
        flag_sets = df['interface_flags'].map(split_interface_flags)
        for flag_name in ALL_KNOWN_FLAGS:
            df[flag_name] = flag_sets.map(lambda flags, name=flag_name: name in flags)

    # Normalise complex_type to lowercase.
    if 'complex_type' in df.columns:
        df['complex_type'] = df['complex_type'].astype(str).str.lower()

    # Derive tier_scope for CSVs written before it existed (newer CSVs ship it).
    if 'tier_scope' not in df.columns:
        df['tier_scope'] = _derive_tier_scope(df)

    if legacy_mode:
        initial_count = len(df)
        df = df.dropna(subset=['iptm'])
        df = df[df['iptm'] > 0]
        if 'pdockq' in df.columns:
            df['pdockq'] = df['pdockq'].fillna(0)
        if 'n_chains' in df.columns and 'complex_type' in df.columns:
            missing_mask = df['n_chains'].isna()
            if missing_mask.any():
                dimer_mask = missing_mask & df['complex_type'].isin(
                    ['homodimer', 'heterodimer'])
                df.loc[dimer_mask, 'n_chains'] = 2
                filled = dimer_mask.sum()
                still_missing = df['n_chains'].isna().sum()
                if filled > 0:
                    print(f"  Inferred n_chains=2 for {filled} rows from complex_type "
                          f"({still_missing} still missing).")
        dropped = initial_count - len(df)
        if dropped > 0:
            print(f"  Dropped {dropped} rows with missing/zero ipTM (legacy mode).")

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

    Note: prefer `_scope_with_species()` for new code so the species qualifier
    lives inside the scope bracket (e.g. `[calibrated dimer x human; N=...]`)
    rather than as a trailing suffix.
    """
    if not species_label:
        return ''
    mapping = {'_human': ' - Human', '_nonhuman': ' - Non-Human'}
    return mapping.get(species_label, f' - {species_label.lstrip("_").replace("_", " ").title()}')


def _scope_with_species(base_scope: str, species_label: str) -> str:
    """Compose a scope caption that incorporates the species qualifier.

    Examples:
        _scope_with_species('calibrated dimer', '')         -> 'calibrated dimer'
        _scope_with_species('calibrated dimer', '_human')   -> 'calibrated dimer x human'
        _scope_with_species('calibrated dimer', '_nonhuman') -> 'calibrated dimer x non-human'
    """
    if species_label == '_human':
        return f'{base_scope} x human'
    if species_label == '_nonhuman':
        return f'{base_scope} x non-human'
    return base_scope


def _format_pvalue(p: float) -> str:
    """Format a p-value for display.

    Avoids the ugly 'p = 0.0e+00' that scipy returns when p underflows. Uses
    'p < 1e-300' for true zeros and 'p < 0.001' for very small but non-zero
    values, otherwise standard scientific or decimal notation.
    """
    if p is None or not isinstance(p, (int, float)):
        return 'p = n/a'
    try:
        import math
        if math.isnan(p):
            return 'p = nan'
        if p <= 0.0:
            return 'p < 1e-300'
        if p < 1e-300:
            return 'p < 1e-300'
        if p < 0.001:
            return f'p = {p:.1e}'
        return f'p = {p:.3f}'
    except Exception:
        return f'p = {p}'


# -------------------------------------------------------------------------
# Effect-size helpers.
# p-values become non-informative at very large N (everything is significant),
# so these helpers headline tier comparisons with effect-size measures instead.
# -------------------------------------------------------------------------

def _cramers_v(contingency) -> float:
    """Cramér's V from a chi-squared statistic on an r x c contingency table.
    Bounded [0, 1]; NaN if degenerate."""
    contingency = np.asarray(contingency, dtype=float)
    if contingency.size == 0:
        return float('nan')
    chi2, _, _, _ = chi2_contingency(contingency)
    n = contingency.sum()
    if n == 0:
        return float('nan')
    r, c = contingency.shape
    denom = n * (min(r, c) - 1)
    return float(np.sqrt(chi2 / denom)) if denom > 0 else float('nan')


def _epsilon_squared(h_stat: float, n: int, k: int) -> float:
    """Kruskal-Wallis epsilon-squared = (H - k + 1) / (n - k).
    Bounded [0, 1] at BOTH ends; NaN if n <= k."""
    if n <= k:
        return float('nan')
    value = (h_stat - k + 1) / (n - k)
    return float(min(1.0, max(0.0, value)))


def _cliffs_delta(x, y) -> float:
    """Cliff's delta in [-1, 1]. Positive means x tends to be larger than y.

    Rank-based (Mann-Whitney U) computation so this runs in
    O((n_x+n_y) log(n_x+n_y)) rather than the naive O(n_x * n_y)
    pairwise-comparison matrix - the latter would blow up memory at
    N >= ~10^5. Handles ties via average ranks.
    """
    from scipy.stats import rankdata
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return float('nan')
    combined = np.concatenate([x, y])
    ranks = rankdata(combined, method='average')
    rank_sum_x = ranks[:n_x].sum()
    u_x = rank_sum_x - n_x * (n_x + 1) / 2
    delta = (2 * u_x / (n_x * n_y)) - 1
    return float(delta)


def _odds_ratio_ci(a, b, c, d) -> tuple:
    """Odds ratio + 95% CI for a 2x2 contingency [[a, b], [c, d]].

    Uses Haldane-Anscombe 0.5 correction when any cell is zero.
    Returns (OR, CI_low, CI_high) - actual CI endpoints, NOT a half-width.
    """
    a, b, c, d = float(a), float(b), float(c), float(d)
    if min(a, b, c, d) == 0:
        a += 0.5
        b += 0.5
        c += 0.5
        d += 0.5
    or_ = (a * d) / (b * c)
    se = np.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)
    log_or = np.log(or_)
    lo = np.exp(log_or - 1.96 * se)
    hi = np.exp(log_or + 1.96 * se)
    return float(or_), float(lo), float(hi)

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

#---------------------------------Structure-prediction figures--------------------------------------------

def plot_fig1_quality_scatter(df: pd.DataFrame, col_flags: dict, density_mode: bool = False, species_label: str = '') -> None:
    """Fig 1: Overall prediction quality landscape (calibrated dimer).
    Colours by V2 quality tier when available, otherwise falls back to disorder-fraction colouring (RdYlGn_r colourmap).
    When density_mode is True, KDE contour overlays are added to show where complexes concentrate.
    """
    if len(df) == 0:
        print("  Skipping Fig 1: no complexes in subset.")
        return
    df, n_before, n_after = apply_filter(df, 'calibrated_dimer', fig_label='Fig 1')
    df, _, n_plot = require_columns(df, ['iptm', 'pdockq'], fig_label='Fig 1')
    if len(df) == 0:
        print("  Skipping Fig 1: 0 rows after calibrated_dimer + required columns.")
        return
    species_suffix = _species_display(species_label)
    use_tier_colouring = col_flags['has_v2_data']
    n_total = len(df)
    pt_size, pt_alpha, raster = _adaptive_scatter_params(n_total)
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
    scope = _scope_with_species(CAPTION_SCOPE_CALIBRATED_DIMER, species_label)
    title = (f"Calibrated dimer quality landscape: v2 tiers over ipTM/pDockQ space "
             f"[{scope}; n={n_total:,}]")
    _apply_common_style(axes, title, 'ipTM', 'pDockQ')
    # Inset note clarifying what colours and dashed lines mean.
    axes.text(0.02, 0.04,
              'Colours: quality_tier_v2\nDashed lines: v1 ipTM/pDockQ gates',
              transform=axes.transAxes, ha='left', va='bottom', fontsize=8,
              style='italic', color='#555',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor='none', alpha=0.85))
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
    plot_df, n_before, n_after = apply_filter(df, 'calibrated_dimer', fig_label='Fig 1b')
    plot_df, _, n_plot = require_columns(plot_df, required, fig_label='Fig 1b')
    if len(plot_df) == 0:
        print("  Skipping Fig 1b: 0 rows after calibrated_dimer + required columns.")
        return

    disorder = plot_df['plddt_below50_fraction'].fillna(0)
    x = plot_df['iptm'].values
    y = plot_df['pdockq'].values
    c = disorder.values
    n_points = len(x)
    figure, axes = plt.subplots(figsize=(10, 8))

    # Adaptive point sizing
    pt_size, pt_alpha, raster = _adaptive_scatter_params(n_points)

    # Scatter: colour = disorder fraction
    scatter = _timed_scatter(axes, x, y, n_points=n_points, fig_label='Fig 1b', c=c, cmap='RdYlGn_r', vmin=0, vmax=1, s=pt_size, alpha=pt_alpha, edgecolors='none', zorder=3, rasterized=raster)
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

    # Make the figure self-contained: unlike Fig 1, this scatter has no
    # tier legend, so a reader can't tell from the figure alone what the
    # dashed lines and colour scale represent.
    axes.text(0.02, 0.04,
              'Dashed lines: v1 ipTM/pDockQ gates\nColour: disorder fraction (pLDDT < 50)',
              transform=axes.transAxes, fontsize=7, va='bottom', ha='left',
              color='#555', style='italic')

    _apply_common_style(
        axes,
        f"Quality Scatter - Disorder Colouring (Supplementary) "
        f"[{_scope_with_species(CAPTION_SCOPE_CALIBRATED_DIMER, species_label)}; n={n_points:,}]",
        'ipTM', 'pDockQ')
    _save_figure(figure, f'1b_supp_Disorder_Scatter{species_label}.png')

def plot_fig2_pae_health_check(df: pd.DataFrame, species_label: str = '') -> None:
    """Fig 2: Is the dataset generally well-resolved? (all-N descriptive)
    Uses the `recoverable` scope so partial/error rows are excluded but every
    structurally readable complex is included regardless of calibration.
    """
    if 'pae_mean' not in df.columns:
        print("  Skipping Fig 2: no pae_mean column.")
        return
    sub, n_before, n_after = apply_filter(df, 'recoverable', fig_label='Fig 2')
    sub, _, _ = require_columns(sub, ['pae_mean'], fig_label='Fig 2')
    pae_values = sub['pae_mean'].dropna() if 'pae_mean' in sub.columns else pd.Series([], dtype=float)
    if len(pae_values) == 0:
        print("  Skipping Fig 2: no valid PAE values after recoverable filter.")
        return
    species_suffix = _species_display(species_label)
    median_pae = pae_values.median()
    below_threshold = (pae_values < PAE_CONFIDENT).sum()
    figure, axes = plt.subplots(figsize=(8, 5))
    axes.hist(pae_values, bins=30, color='#3498db', alpha=0.75, edgecolor='white', linewidth=0.5)

    # Median line
    axes.axvline(x=median_pae, color='red', linestyle='--', linewidth=1.5, label=f'Median: {median_pae:.1f} \u00c5')

    # Confident guideline
    axes.axvline(x=PAE_CONFIDENT, color='green', linestyle='--', linewidth=1.5, label=f'{PAE_CONFIDENT} \u00c5 reference used for confident contacts')

    axes.legend(fontsize=FONT_TICK, loc='upper right')
    # Scope label specifically calls out that this is the subset of
    # recoverable rows with a usable `pae_mean`, not every recoverable row.
    # A small number of recoverable rows lack a finite `pae_mean`, which would
    # otherwise raise an N-mismatch flag against the corpus funnel.
    scope_label = 'recoverable rows with global PAE available'
    title = (f"Supplementary: Global PAE distribution \u2014 descriptive input-confidence audit "
             f"(n={len(pae_values):,}, median {median_pae:.1f} \u00c5, {below_threshold} below {PAE_CONFIDENT} \u00c5) "
             f"[{_scope_with_species(scope_label, species_label)}]")
    _apply_common_style(axes, title, 'Mean PAE (\u00c5)', 'Count', grid=False)
    # Long single-line title sits flush against the top edge without
    # explicit top padding. Reserve room so the title can't clip on export.
    figure.subplots_adjust(top=0.88)
    _save_figure(figure, f'2_supp_PAE_Health_Check{species_label}.png')

def plot_fig3_interface_pae_by_tier(df: pd.DataFrame, species_label: str = '') -> None:
    """Fig 3: How confident are the contacts that matter for quality assessment?
    (calibrated dimer)"""
    required = ['interface_pae_mean', 'quality_tier_v2']
    plot_df, n_before, n_after = apply_filter(df, 'calibrated_dimer', fig_label='Fig 3')
    plot_df, _, n_plot = require_columns(plot_df, required, fig_label='Fig 3')
    if len(plot_df) == 0:
        print("  Skipping Fig 3: 0 rows after calibrated_dimer + required columns.")
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
    axes.axhline(y=PAE_CONFIDENT, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'{PAE_CONFIDENT} \u00c5 confident-contact reference')

    axes.set_xticks(positions)
    axes.set_xticklabels(tier_labels, fontsize=FONT_AXIS_LABEL)
    axes.legend(fontsize=FONT_TICK, loc='upper right')

    # Subtitle with medians
    median_text = " | ".join(
        f"{TIER_ORDER[i]} median: {tier_medians[i]:.1f} \u00c5"
        for i in range(len(tier_medians))
    )
    axes.text(0.5, -0.12, median_text, transform=axes.transAxes, ha='center', fontsize=FONT_TICK, style='italic', color='#555555')
    _apply_common_style(
        axes,
        f"Interface PAE by quality tier "
        f"[{_scope_with_species(CAPTION_SCOPE_CALIBRATED_DIMER, species_label)}; n={len(plot_df):,}]",
        '', 'Interface PAE (\u00c5)', grid=False)
    _save_figure(figure, f'Fig_2A_Interface_PAE_by_Quality_Tier{species_label}.png')

def plot_fig4_composite_validation(df: pd.DataFrame, density_mode: bool = False, species_label: str = '') -> None:
    """Fig 4: Why should I trust the quality tier assigned?
    Panel (a): Composite score distributions by tier (violin/boxplot).
    Panel (b): Composite vs STRICT confident contact fraction scatter - the fraction
    actually consumed by the composite score (PAE < 5A AND both residue pLDDT >= 70).
    Plotting the PAE-only fraction here would be circular w.r.t. the composite definition,
    since the composite uses the strict fraction post-revision.
    """
    required = ['interface_confidence_score', 'quality_tier_v2', 'strict_confident_contact_fraction']
    plot_df, n_before, n_after = apply_filter(df, 'calibrated_dimer', fig_label='Fig 4')
    plot_df, _, n_plot = require_columns(plot_df, required, fig_label='Fig 4')
    if len(plot_df) == 0:
        print("  Skipping Fig 4: 0 rows after calibrated_dimer + required columns.")
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

    # v2 reclassification boundaries on the composite-score axis (dissertation
    # Appendix B records how these cut-offs were selected):
    #   0.63 -> High downgrades to Medium
    #   0.64 -> Low rescues to Medium (moderate-composite band)
    #   0.85 -> Low/Medium promotes to High (strong-composite band)
    # The 0.63 and 0.64 lines sit so close together that two side-by-side
    # labels overlap; draw both lines but merge their labels into a single
    # bracket-style annotation on the boundary band.
    ax_a.axhline(y=DOWNGRADE_HIGH_THRESHOLD, color='#2ecc71',
                 linestyle=':', linewidth=1.2, alpha=0.7)
    ax_a.axhline(y=UPGRADE_LOW_THRESHOLD, color='#e74c3c',
                 linestyle=':', linewidth=1.2, alpha=0.7)
    ax_a.axhline(y=UPGRADE_MEDIUM_THRESHOLD, color='#f39c12',
                 linestyle=':', linewidth=1.2, alpha=0.7)
    x_right = ax_a.get_xlim()[1]
    band_mid = (DOWNGRADE_HIGH_THRESHOLD + UPGRADE_LOW_THRESHOLD) / 2
    ax_a.text(x_right, band_mid,
              f' 0.63-0.64 boundary\n  (High downgrade /\n  Low->Medium rescue)',
              va='center', fontsize=7, color='#555555', alpha=0.9)
    ax_a.text(x_right, UPGRADE_MEDIUM_THRESHOLD,
              f' Strong composite ({UPGRADE_MEDIUM_THRESHOLD})\n  Low/Medium -> High',
              va='center', fontsize=8, color='#f39c12', alpha=0.8)

    ax_a.set_xticks(positions_a)
    ax_a.set_xticklabels(tier_labels_a, fontsize=FONT_AXIS_LABEL)

    # Tier colour legend - upper-left to avoid threshold label overlap
    tier_legend = [mpatches.Patch(color=TIER_COLORS[t], alpha=0.6, label=t) for t in TIER_ORDER]
    ax_a.legend(handles=tier_legend, fontsize=FONT_TICK - 1, loc='upper left', framealpha=0.9)
    _apply_common_style(ax_a, "(a) Composite Score by Tier", '', 'Interface Confidence Score', grid=False)

    #====================================Panel (b): Composite vs strict confident contact fraction====================================
    n_panel_b = len(plot_df)
    pt_size_b, pt_alpha_b, raster_b = _adaptive_scatter_params(n_panel_b)

    for tier in reversed(TIER_ORDER):
        subset = plot_df[plot_df['quality_tier_v2'] == tier]
        axes_b_kwargs = dict(c=TIER_COLORS[tier], alpha=pt_alpha_b, s=pt_size_b, edgecolors='white', linewidths=0.3, label=tier, zorder=3, rasterized=raster_b)
        _timed_scatter(ax_b, subset['strict_confident_contact_fraction'], subset['interface_confidence_score'], n_points=n_panel_b, fig_label='Fig 4b', **axes_b_kwargs)
    # Explicit proxy handles: adaptive scatter markers are too small/transparent
    # for matplotlib's auto-legend to inherit.
    ax_b.legend(handles=_build_tier_legend_handles(plot_df),
                fontsize=FONT_TICK, title='Tier', title_fontsize=FONT_TICK)

    # Optional density contours
    if density_mode:
        valid_b = plot_df[['strict_confident_contact_fraction', 'interface_confidence_score']].dropna()
        _overlay_kde_contours(ax_b, valid_b['strict_confident_contact_fraction'].values, valid_b['interface_confidence_score'].values)

    # Correlation annotation
    valid_both = plot_df[['strict_confident_contact_fraction', 'interface_confidence_score']].dropna()
    if len(valid_both) > 2:
        r = valid_both['strict_confident_contact_fraction'].corr(valid_both['interface_confidence_score'])
        ax_b.text(0.05, 0.95, f'r = {r:.2f}', transform=ax_b.transAxes, fontsize=FONT_AXIS_LABEL, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    _apply_common_style(ax_b, "(b) Composite score tracks its strict confident-contact component", 'Strict Confident Contact Fraction (PAE < 5 Å & pLDDT ≥ 70)', 'Interface Confidence Score')
    figure.suptitle(
        f"Composite score behaviour across quality tiers "
        f"[{_scope_with_species(CAPTION_SCOPE_CALIBRATED_DIMER, species_label)}; n={len(plot_df):,}]",
        fontsize=14, fontweight='bold', y=1.02)
    # Figure-wide framing: panel (b)'s x-axis (strict confident-contact
    # fraction) is itself a direct input to the composite score on its
    # y-axis, and quality_tier_v2 is assigned from the composite-score
    # decision rules. This is a component-consistency check by construction,
    # not an independent external validation. (Non-visible note kept; the
    # diagonal trend itself is already obvious from the scatter.)
    figure.text(0.5, -0.04,
                'Panel (b) is a component-consistency check, not an '
                'independent validation: the strict confident-contact '
                'fraction (x-axis) is a direct input to the composite '
                'score (y-axis), and the tiers shown are themselves defined '
                'by the composite-score decision rules.',
                ha='center', va='top', fontsize=8, style='italic',
                color='#555555', wrap=True)
    _save_figure(figure, f'Fig_3_Composite_Score_Behaviour{species_label}.png')


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
    plot_df, n_before, n_after = apply_filter(df, 'calibrated_dimer', fig_label='Fig 4 supp')
    plot_df, _, n_plot = require_columns(plot_df, required, fig_label='Fig 4 supp')
    if len(plot_df) == 0:
        print("  Skipping Fig 4 supp: 0 rows after calibrated_dimer + required columns.")
        return

    species_suffix = _species_display(species_label)
    figure, axes = plt.subplots(figsize=(8, 8))
    n_points = len(plot_df)
    pt_size, pt_alpha, raster = _adaptive_scatter_params(n_points)

    for tier in reversed(TIER_ORDER):
        subset = plot_df[plot_df['quality_tier_v2'] == tier]
        if len(subset) == 0:
            continue
        axes.scatter(subset['pae_confident_contact_fraction'],
                     subset['strict_confident_contact_fraction'],
                     c=TIER_COLORS[tier], alpha=pt_alpha, s=pt_size,
                     edgecolors='white', linewidths=0.3, label=tier, zorder=3,
                     rasterized=raster)

    # y = x reference line: strict can never exceed PAE-only
    axes.plot([0, 1], [0, 1], color='#555555', linestyle='--', linewidth=1.0, alpha=0.7)

    # Summary stats annotation
    delta = (plot_df['pae_confident_contact_fraction']
             - plot_df['strict_confident_contact_fraction'])
    axes.text(0.05, 0.95,
              f"n = {len(plot_df)}\n"
              f"mean delta = {delta.mean():.3f}\n"
              f"median delta = {delta.median():.3f}",
              transform=axes.transAxes, fontsize=FONT_AXIS_LABEL, va='top',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Explicit proxy handles: tier swatches + y=x reference proxy.
    legend_handles = _build_tier_legend_handles(plot_df)
    legend_handles.append(Line2D([0], [0], color='#555555', linestyle='--',
                                 linewidth=1.0, alpha=0.7,
                                 label='y = x (upper bound)'))
    axes.legend(handles=legend_handles, fontsize=FONT_TICK, title='Tier',
                title_fontsize=FONT_TICK, loc='lower right')
    _apply_common_style(axes,
                        f"Strict vs PAE-only Confident Contact Fraction "
                        f"[{_scope_with_species(CAPTION_SCOPE_CALIBRATED_DIMER, species_label)}; n={len(plot_df):,}]",
                        'PAE-only Confident Contact Fraction (PAE < 5 Å)',
                        'Strict Confident Contact Fraction (PAE < 5 Å & pLDDT ≥ 70)')
    _save_figure(figure, f'4_supp_Strict_vs_PAE_Only_Fraction{species_label}.png')

def plot_fig5_interface_vs_bulk(df: pd.DataFrame, density_mode: bool = False, species_label: str = '') -> None:
    """Fig 5: Are interfaces special, or do they just reflect bulk quality?
    (calibrated dimer)"""
    required = ['interface_plddt_combined', 'bulk_plddt_combined', 'quality_tier_v2']
    plot_df, n_before, n_after = apply_filter(df, 'calibrated_dimer', fig_label='Fig 5')
    plot_df, _, n_plot = require_columns(plot_df, required, fig_label='Fig 5')
    if len(plot_df) == 0:
        print("  Skipping Fig 5: 0 rows after calibrated_dimer + required columns.")
        return

    species_suffix = _species_display(species_label)
    figure, axes = plt.subplots(figsize=(8, 8))

    # Identify paradox complexes for special marking
    paradox_mask = _get_paradox_mask(plot_df)

    # Plot non-paradox by tier with adaptive sizing
    non_paradox = plot_df[~paradox_mask] # to invert mask and get non-paradox complexes
    n_non_paradox = len(non_paradox)
    pt_size, pt_alpha, raster = _adaptive_scatter_params(n_non_paradox)

    for tier in reversed(TIER_ORDER):
        subset = non_paradox[non_paradox['quality_tier_v2'] == tier]
        _timed_scatter(axes, subset['bulk_plddt_combined'],
                       subset['interface_plddt_combined'],
                       n_points=n_non_paradox, fig_label='Fig 5',
                       c=TIER_COLORS[tier], s=pt_size,
                       alpha=pt_alpha, edgecolors='white',
                       linewidths=0.5, zorder=3, label=tier,
                       rasterized=raster)

    # Optional density contours
    if density_mode:
        _overlay_kde_contours(axes, non_paradox['bulk_plddt_combined'].values, non_paradox['interface_plddt_combined'].values)

    # Paradox complexes - outline-only triangles so they're visible but don't
    # dominate the very large non-paradox background. Adaptive sizing scales with paradox N.
    paradox_df = plot_df[paradox_mask]
    paradox_n = len(paradox_df)
    if paradox_n > 0:
        para_size = max(12, min(35, 800 // max(paradox_n, 1)))
        axes.scatter(paradox_df['bulk_plddt_combined'],
                     paradox_df['interface_plddt_combined'],
                     facecolors='none', edgecolors='#9b59b6',
                     s=para_size, alpha=0.65, marker='^',
                     linewidths=0.8, zorder=5)

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
    # Raised above the bottom-left legend to avoid overlap with the
    # `Paradox (n=...)` row at lower-left.
    axes.text(0.5, 0.14, f'{above_diagonal:,}/{total:,} ({pct:.0f}%) above diagonal',
              transform=axes.transAxes, ha='center', fontsize=FONT_TICK,
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # Explicit proxy handles: tier swatches + Paradox triangle proxy.
    # Auto-legend would inherit adaptive scatter s/alpha and render invisible.
    legend_handles = _build_tier_legend_handles(plot_df)
    if paradox_n > 0:
        legend_handles.append(Line2D([], [], marker='^', linestyle='',
                                     markersize=8, markerfacecolor='none',
                                     markeredgecolor='#9b59b6',
                                     markeredgewidth=1.2,
                                     label=f'Paradox (n={paradox_n:,})'))
    axes.legend(handles=legend_handles, fontsize=FONT_TICK,
                loc='lower left', framealpha=0.9)
    _apply_common_style(
        axes,
        f"Interface pLDDT versus bulk pLDDT "
        f"[{_scope_with_species(CAPTION_SCOPE_CALIBRATED_DIMER, species_label)}; n={len(plot_df):,}]",
        'Bulk pLDDT', 'Interface pLDDT')
    # Non-visible note: the legend marks the paradox subset by count only, so
    # spell out what defines it (the diagonal relationship itself is visible).
    axes.text(0.5, -0.10,
              f'Paradox triangles mark the subset with ipTM ≥ {IPTM_HIGH}, '
              f'pDockQ ≥ {PDOCKQ_HIGH} and ≥ {int(DISORDER_SUBSTANTIAL * 100)}% '
              f'of residues at pLDDT < 50: confident interfaces despite a substantial '
              f'low-pLDDT, disorder-associated bulk.',
              transform=axes.transAxes, ha='center', va='top', fontsize=7,
              style='italic', color='#777')
    _save_figure(figure, f'Fig_2B_Interface_pLDDT_vs_Bulk_pLDDT{species_label}.png')

def plot_fig6_paradox_spotlight(df: pd.DataFrame, species_label: str = '') -> None:
    """Fig 6: Can disordered proteins form confident interfaces?
    3-panel comparison of paradox vs non-paradox complexes - Paradox: ipTM >= 0.75, pDockQ >= 0.5, disorder fraction >= 0.30.
    """
    required = ['iptm', 'pdockq', 'plddt_below50_fraction', 'interface_vs_bulk_delta', 'pae_confident_contact_fraction', 'interface_symmetry']

    species_suffix = _species_display(species_label)

    # Dissertation-safe: paradox detection uses dimer-calibrated thresholds.
    scoped_df, n_before, n_after = apply_filter(df, 'calibrated_dimer', fig_label='Fig 6')
    if len(scoped_df) == 0:
        print("  Skipping Fig 6: 0 rows after calibrated_dimer filter.")
        return

    # Count paradox complexes before dropping rows with missing panel data so we can report how many are lost to incomplete interface metrics
    n_paradox_before_dropna = int(_get_paradox_mask(scoped_df).sum())

    plot_df, _, n_plot = require_columns(scoped_df, required, fig_label='Fig 6')
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
        bp = ax.boxplot(box_data, positions=positions, widths=0.5, patch_artist=True, showfliers=False,
                        medianprops=dict(color='black', linewidth=2),
                        boxprops=dict(linewidth=1.4, edgecolor='black'),
                        whiskerprops=dict(linewidth=1.2, color='black'),
                        capprops=dict(linewidth=1.2, color='black'))
        bp['boxes'][0].set_facecolor(colour_paradox)
        bp['boxes'][0].set_alpha(0.5)
        bp['boxes'][1].set_facecolor(colour_non_paradox)
        bp['boxes'][1].set_alpha(0.5)

        # Force box bodies and whiskers above the large strip cloud (zorder=3).
        # Median lines stay at zorder=10 so they remain the topmost element.
        for patch in bp['boxes']:
            patch.set_zorder(8)
        for line in bp['whiskers'] + bp['caps']:
            line.set_zorder(9)
        for median_line in bp['medians']:
            median_line.set_zorder(10)

        # Jittered strip. The non-paradox group can be very large (hundreds of
        # thousands of points), so use much smaller markers + lower alpha to
        # avoid a solid blue cloud that hides the box and medians. Paradox stays
        # clearly visible.
        para_pts_size, para_pts_alpha = 18, 0.45
        nonpara_pts_size, nonpara_pts_alpha, _ = _adaptive_scatter_params(len(data_non_paradox))
        # Floor alpha further so the strip doesn't dominate at HPC scale.
        nonpara_pts_alpha = min(nonpara_pts_alpha, 0.15)
        strip_styles = [
            (data_paradox, colour_paradox, para_pts_size, para_pts_alpha),
            (data_non_paradox, colour_non_paradox, nonpara_pts_size, nonpara_pts_alpha),
        ]
        for i, (data, colour, s, alpha) in enumerate(strip_styles):
            jitter = np.random.normal(0, 0.06, size=len(data))
            ax.scatter(positions[i] + jitter, data, c=colour, alpha=alpha, s=s,
                       zorder=3, edgecolors='none')

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
        f"Prediction-quality paradox interface characteristics "
        f"[{_scope_with_species(CAPTION_SCOPE_CALIBRATED_DIMER, species_label)}; n={len(plot_df):,}]",
        fontsize=14, fontweight='bold', y=1.06)

    # Non-visible note: the panels show the comparison and the per-group counts
    # (in the tick labels), but not the paradox definition itself; state it
    # here so the dissertation caption can reference exact thresholds.
    subtitle = (f"Paradox: ipTM ≥ {IPTM_HIGH} and pDockQ ≥ {PDOCKQ_HIGH} (v1 High) "
                f"with ≥ {int(DISORDER_SUBSTANTIAL * 100)}% of residues at pLDDT < 50")
    if n_paradox_missing_data > 0:
        subtitle += (f"  ({n_paradox_missing_data} paradox complexes excluded: "
                     f"incomplete interface data)")

    # Subtitle sits below the suptitle; both anchored above the axes to avoid
    # the prior ghost-overlap at y=0.99 / y=1.04.
    figure.text(0.5, 1.00, subtitle, ha='center', fontsize=FONT_AXIS_LABEL, style='italic', color='#555555')
    _save_figure(figure, f'Fig_6_Prediction_Quality_Paradox{species_label}.png')

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
    required = ['complex_type', 'quality_tier_v2', 'stoichiometry']
    # Keep full df available so the multimer supplement can re-filter from scratch.
    full_df = df

    # Primary panel (dissertation-safe): calibrated_dimer AND stoichiometry in {A2, AB}.
    primary_df, n_before, n_after = apply_filter(df, 'calibrated_dimer', fig_label='Fig 7')
    primary_df, _, n_plot = require_columns(primary_df, required, fig_label='Fig 7')
    if len(primary_df) == 0:
        print("  Skipping Fig 7: 0 rows after calibrated_dimer + required columns.")
        return
    primary_df = primary_df[primary_df['stoichiometry'].isin(DIMER_STOICHIOMETRIES)]
    if len(primary_df) == 0:
        print("  Skipping Fig 7: no calibrated dimers with stoichiometry A2/AB.")
        return

    species_suffix = _species_display(species_label)

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

    _apply_common_style(ax_b, "(b) Interface Symmetry (calibrated dimer)", '', 'Symmetry Score', grid=False)
    figure.suptitle(
        f"Prediction Quality by Complex Architecture "
        f"[{_scope_with_species(CAPTION_SCOPE_CALIBRATED_A2_AB, species_label)}; n={len(primary_df):,}]",
        fontsize=14, fontweight='bold', y=1.02)
    figure.text(0.5, -0.01,
                'Note: Restricted to A2/AB stoichiometry; calibrated dimers with ambiguous '
                'or other stoichiometry labels are excluded here only.',
                ha='center', fontsize=7, style='italic', color='#777')
    _save_figure(figure, f'7_Homo_vs_Hetero{species_label}.png')

    if multimer_supplement:
        _plot_fig7_multimer_supplement(full_df, species_label=species_label)


def _plot_fig7_multimer_supplement(df: pd.DataFrame, species_label: str = '') -> None:
    """Fig 7 supplementary: tier proportions and symmetry by multimer stoichiometry.
    Shows multimer_exploratory rows bucketed into A2B, ABC, A2B2, ABCD, and
    'Other' (anything else with n_chains > 2). Gated by --multimer-supplement
    - these are exploratory, not dissertation claims.
    """
    multimer_df, n_before, n_after = apply_filter(df, 'multimer_exploratory', fig_label='Fig 7 supp')
    multimer_df, _, _ = require_columns(
        multimer_df, ['quality_tier_v2', 'stoichiometry'], fig_label='Fig 7 supp')
    if len(multimer_df) == 0:
        print("  Fig 7 supp: no multimer_exploratory rows - skipping supplement.")
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
        f"Multimer Architecture Supplement "
        f"[{_scope_with_species(CAPTION_SCOPE_MULTIMER, species_label)}; n={len(multimer_df):,}]",
        fontsize=14, fontweight='bold', y=1.02)
    _save_figure(figure, f'7_supp_Multimer_Stoichiometry{species_label}.png')

def plot_fig8_iptm_pdockq_delta_histogram(df: pd.DataFrame, density_mode: bool = False, species_label: str = '') -> None:
    """Fig 5: ipTM and pDockQ categorical agreement (calibrated dimers).

    A single categorical agreement matrix - there are no other panels. ipTM and
    pDockQ are each classified INDEPENDENTLY into Low/Medium/High using their own
    thresholds (ipTM 0.50 and 0.75; pDockQ 0.23 and 0.50). The 3x3
    matrix cross-tabulates the two independent classifications: the diagonal is
    categorical agreement and the off-diagonal is categorical disagreement. The
    banner reports the overall agreement/disagreement split (the dissertation's
    41.4% categorical disagreement result for calibrated dimers).

    ipTM and pDockQ are on different numerical scales, so this figure makes NO
    raw ipTM - pDockQ subtraction, and uses no continuous Δ histogram, no Δ
    threshold, no composite score and no metric_disagreement flag. Neither metric
    is treated as ground truth. The descriptive ipTM-vs-pDockQ scatter remains
    available as `plot_fig8_supp_metric_disagreement_scatter`.

    density_mode is accepted for dispatch-signature compatibility and is unused.
    """
    required = ['iptm', 'pdockq']
    plot_df, _, _ = apply_filter(df, 'calibrated_dimer', fig_label='Fig 5')
    plot_df, _, n_plot = require_columns(plot_df, required, fig_label='Fig 5')
    if len(plot_df) == 0:
        print("  Skipping Fig 5: 0 rows after calibrated_dimer + required columns.")
        return

    # Classify ipTM and pDockQ INDEPENDENTLY into Low/Medium/High using their own
    # thresholds. 0 = Low, 1 = Medium, 2 = High (display order on
    # both axes). No raw subtraction, no Δ, no composite tier and no
    # metric_disagreement flag are used.
    IPTM_MED_MIN = 0.50      # v1 medium gate for ipTM (high gate is IPTM_HIGH)
    PDOCKQ_MED_MIN = 0.23    # v1 medium gate for pDockQ (high gate is PDOCKQ_HIGH)
    iptm_vals = plot_df['iptm'].astype(float).values
    pdockq_vals = plot_df['pdockq'].astype(float).values
    cat_finite = np.isfinite(iptm_vals) & np.isfinite(pdockq_vals)
    iptm_vals = iptm_vals[cat_finite]
    pdockq_vals = pdockq_vals[cat_finite]
    n_cat = len(iptm_vals)
    if n_cat == 0:
        print("  Skipping Fig 5: no rows with finite ipTM and pDockQ.")
        return

    iptm_idx = np.where(iptm_vals >= IPTM_HIGH, 2,
                        np.where(iptm_vals >= IPTM_MED_MIN, 1, 0))
    pdockq_idx = np.where(pdockq_vals >= PDOCKQ_HIGH, 2,
                          np.where(pdockq_vals >= PDOCKQ_MED_MIN, 1, 0))

    matrix = np.zeros((3, 3), dtype=int)  # rows = ipTM-only cat, cols = pDockQ-only cat
    for r in range(3):
        for c in range(3):
            matrix[r, c] = int(np.sum((iptm_idx == r) & (pdockq_idx == c)))
    pct_matrix = matrix / n_cat * 100.0

    agree_pct = float(np.trace(pct_matrix))   # diagonal = categorical agreement
    disagree_pct = 100.0 - agree_pct          # off-diagonal = categorical disagreement

    # Single-axis figure: the categorical matrix is now the whole figure.
    fig, ax = plt.subplots(figsize=(8.5, 7.0))

    # Colour: hue encodes agreement (green diagonal) vs disagreement (orange
    # off-diagonal); intensity within each hue scales with the cell's share of
    # the population so the dominant cells read darkest. Built as an explicit
    # RGBA grid.
    green_cmap = plt.cm.Greens
    orange_cmap = plt.cm.Oranges
    max_pct = float(pct_matrix.max()) if pct_matrix.max() > 0 else 1.0
    rgba = np.zeros((3, 3, 4))
    intensity_grid = np.zeros((3, 3))
    for r in range(3):
        for c in range(3):
            intensity = 0.20 + 0.70 * (pct_matrix[r, c] / max_pct)
            intensity_grid[r, c] = intensity
            rgba[r, c] = green_cmap(intensity) if r == c else orange_cmap(intensity)
    ax.imshow(rgba, aspect='auto', origin='upper')

    # Cell annotations: count + share of all calibrated dimers (full-population %).
    for r in range(3):
        for c in range(3):
            txt_colour = 'white' if intensity_grid[r, c] > 0.62 else '#2c3e50'
            ax.text(c, r, f'{matrix[r, c]:,}\n({pct_matrix[r, c]:.1f}%)',
                    ha='center', va='center', fontsize=FONT_AXIS_LABEL + 1,
                    fontweight='bold', color=txt_colour)

    cat_labels = ['Low', 'Medium', 'High']
    ax.set_xticks(range(3))
    ax.set_xticklabels(cat_labels)
    ax.set_yticks(range(3))
    ax.set_yticklabels(cat_labels)
    ax.set_xlabel('pDockQ-only category', fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel('ipTM-only category', fontsize=FONT_AXIS_LABEL)
    ax.tick_params(which='major', labelsize=FONT_AXIS_LABEL)
    # White cell borders for clear separation.
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=2)
    ax.tick_params(which='minor', bottom=False, left=False)

    # Headline split banner (derived from the matrix, not hardcoded), placed
    # above the matrix so it is read first.
    ax.text(0.5, 1.05,
            f'Categorical agreement: {agree_pct:.1f}%        '
            f'Disagreement: {disagree_pct:.1f}%',
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=FONT_AXIS_LABEL, fontweight='bold', color='#34495e',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#f7f7f7',
                      edgecolor='#cccccc', alpha=0.95))

    fig.suptitle(
        f"ipTM and pDockQ categorical agreement "
        f"[{_scope_with_species(CAPTION_SCOPE_CALIBRATED_DIMER, species_label)}; "
        f"n={n_cat:,}]",
        fontsize=14, fontweight='bold', y=1.00)
    # Non-visible note for caption writing: how the two independent
    # classifications were thresholded, and that neither metric is ground truth.
    fig.text(0.5, -0.04,
             f'Categories were assigned independently using ipTM thresholds of '
             f'{IPTM_MED_MIN:.2f} and {IPTM_HIGH:.2f} and pDockQ thresholds of '
             f'{PDOCKQ_MED_MIN:.2f} and {PDOCKQ_HIGH:.2f}. Neither metric was '
             f'treated as ground truth.',
             ha='center', va='top', fontsize=8, style='italic',
             color='#555555', wrap=True)
    _save_figure(fig, f'Fig_5_ipTM_pDockQ_Metric_Disagreement{species_label}.png')


def plot_fig8_supp_metric_disagreement_scatter(df: pd.DataFrame, density_mode: bool = False, species_label: str = '') -> None:
    """Fig 8 supplementary: descriptive ipTM vs pDockQ scatter (calibrated dimer).

    NOTE: the dashed y=x line is a visual reference only; ipTM and pDockQ are
    not calibrated to the same numerical scale, so distance from y=x is NOT a
    direct agreement metric. The main figure (dissertation Fig 5,
    `plot_fig8_iptm_pdockq_delta_histogram`) is the ipTM/pDockQ categorical
    agreement matrix, which avoids any raw-scale comparison entirely.
    """
    required = ['iptm', 'pdockq', 'quality_tier_v2']
    plot_df, n_before, n_after = apply_filter(df, 'calibrated_dimer', fig_label='Fig 8 supp')
    plot_df, _, n_plot = require_columns(plot_df, required, fig_label='Fig 8 supp')
    if len(plot_df) == 0:
        print("  Skipping Fig 8 supp: 0 rows after calibrated_dimer + required columns.")
        return

    species_suffix = _species_display(species_label)
    figure, axes = plt.subplots(figsize=(10, 8))
    n_plot = len(plot_df)
    pt_size, pt_alpha, raster = _adaptive_scatter_params(n_plot)

    for tier in reversed(TIER_ORDER):
        subset = plot_df[plot_df['quality_tier_v2'] == tier]
        _timed_scatter(axes, subset['iptm'], subset['pdockq'],
                       n_points=n_plot, fig_label='Fig 8 supp',
                       c=TIER_COLORS[tier], s=pt_size,
                       alpha=pt_alpha, edgecolors='white',
                       linewidths=0.5, zorder=3, label=tier,
                       rasterized=raster)
    # Explicit proxy handles: adaptive scatter markers are too small/transparent
    # for matplotlib's auto-legend to inherit.
    axes.legend(handles=_build_tier_legend_handles(plot_df),
                fontsize=FONT_TICK, title='Tier',
                title_fontsize=FONT_TICK, loc='upper left')

    if density_mode:
        _overlay_kde_contours(axes, plot_df['iptm'].values, plot_df['pdockq'].values)

    # Visual reference y=x line (NOT a calibrated agreement line).
    axes.plot([0, 1.1], [0, 1.1], 'k--', linewidth=1.2, alpha=0.6, zorder=1)

    # Descriptive highlight of large positive-Δ cases (ipTM >> pDockQ), using
    # METRIC_DISAGREEMENT_GAP only as a display cut-off for the highlight. This
    # is NOT the dissertation's metric-disagreement definition, which is
    # categorical (ipTM-only vs pDockQ-only tiers, no Δ threshold) — the footer
    # states that boundary so the two are not conflated.
    extreme_mask = (plot_df['iptm'] - plot_df['pdockq']) > METRIC_DISAGREEMENT_GAP
    n_extreme = int(extreme_mask.sum())
    pct_extreme = n_extreme / len(plot_df) * 100 if len(plot_df) > 0 else 0
    axes.text(0.85, 0.10,
              f'{n_extreme:,} complexes ({pct_extreme:.1f}%)\n'
              f'ipTM >> pDockQ\n(Δ > {METRIC_DISAGREEMENT_GAP})',
              transform=axes.transAxes, ha='center', va='center',
              fontsize=FONT_AXIS_LABEL, fontweight='bold', color='#c0392b',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    axes.text(0.5, -0.10,
              f'ipTM >> pDockQ is a descriptive highlight of a large positive '
              f'gap (Δ = ipTM - pDockQ > {METRIC_DISAGREEMENT_GAP}); the reported '
              f'metric-disagreement rate is a separate categorical result '
              f'(ipTM-only vs pDockQ-only tiers). Dashed y = x is a visual '
              f'reference; ipTM and pDockQ are not on the same calibrated scale.',
              transform=axes.transAxes, ha='center', va='top', fontsize=8,
              style='italic', color='#777')

    axes.set_xlim(0.2, 1.05)
    axes.set_ylim(-0.02, 0.8)

    _apply_common_style(
        axes,
        f"Supplementary: ipTM vs pDockQ scatter (descriptive) "
        f"[{_scope_with_species(CAPTION_SCOPE_CALIBRATED_DIMER, species_label)}; n={len(plot_df):,}]",
        'ipTM', 'pDockQ')
    _save_figure(figure, f'8_supp_iptm_pdockq_scatter{species_label}.png')


# Alias retained for backward compatibility with external callers; redirects to the supplementary scatter.
plot_fig8_metric_disagreement = plot_fig8_supp_metric_disagreement_scatter

def plot_fig9_chain_count_profile(df: pd.DataFrame, density_mode: bool = False, species_label: str = '') -> None:
    """Fig 9: Chain-count profile - exposes order-statistic bias honestly.
    Four violin panels by n_chains group (2, 3, 4+):
        (a) pdockq (best pair) - order-statistic biased upward for large N.
        (b) pdockq_mean        - unbiased aggregate across all inter-chain pairs.
        (c) pdockq_min         - worst-pair lower bound; surfaces dangling chains.
        (d) coherence gap (pdockq - pdockq_min) - 0 for every N=2 row by construction.
    The figure is descriptive across all N (CAPTION_SCOPE_ALL_N). Requires the
    multimer-safe aggregates (pdockq_mean/pdockq_min); falls back gracefully
    otherwise.
    """
    base_required = ['n_chains', 'pdockq']
    if not all(col in df.columns for col in base_required):
        print("  Skipping Fig 9: missing required columns.")
        return

    aggregate_cols = ['pdockq_mean', 'pdockq_min']
    if not all(col in df.columns for col in aggregate_cols):
        print("  Skipping Fig 9: aggregate columns (pdockq_mean/pdockq_min) missing. "
              "Regenerate the CSV with the multimer-aware schema that includes them.")
        return

    fig9_required = base_required + aggregate_cols
    # Fig 9 is descriptive across all chain counts (all-N), so it uses
    # `recoverable` rather than `calibrated_dimer` — calibration is irrelevant
    # to a per-chain-count distribution shape.
    plot_df, n_before, n_after = apply_filter(df, 'recoverable', fig_label='Fig 9')
    plot_df, _, n_plot = require_columns(plot_df, fig9_required, fig_label='Fig 9')
    if len(plot_df) == 0:
        print("  Skipping Fig 9: 0 rows after recoverable + required columns.")
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

        # Coherence gap is identically 0 for every N=2 row by construction
        # (pdockq == pdockq_min when only one inter-chain pair exists).
        # The collapsed violin then disappears under the scatter cloud and
        # reads as missing data. Add a zero baseline and an in-panel
        # annotation to disambiguate.
        if col == 'coherence_gap':
            ax.axhline(0, color='#999', linewidth=0.8, linestyle=':',
                       zorder=0)

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

        if col == 'coherence_gap' and '2 chains' in present_groups:
            two_chain_x = present_groups.index('2 chains')
            ymax = ax.get_ylim()[1]
            ax.text(two_chain_x, ymax * 0.92,
                    'N=2 coherence gap\n≡ 0 by construction',
                    ha='center', va='top', fontsize=8, style='italic',
                    color='#555555',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                              edgecolor='#cccccc', alpha=0.85))

        _apply_common_style(ax, title, '', ylabel, grid=False)

    figure.suptitle(
        f"Supplementary: Chain-count quality profile (order-statistic bias) "
        f"[{_scope_with_species(CAPTION_SCOPE_RECOVERABLE_ALL_N, species_label)}; n={len(plot_df):,}]",
        fontsize=14, fontweight='bold', y=1.02)
    _save_figure(figure, f'9_supp_Chain_Count_Profile{species_label}.png')

#--------------------------------------------Sequence-similarity figure---------------------------------------------------

def plot_fig10_clustering_validation(df: pd.DataFrame) -> None:
    """Fig 10: sequence-cluster sharing by architecture and quality tier.
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
    # Calibrated dimers with A2/AB stoichiometry only — calibration is required
    # for tier comparisons in Panel B; cluster scatter in Panel A is described
    # against the same scope for consistency.
    plot_df, n_before, n_after = apply_filter(df, 'calibrated_dimer', fig_label='Fig 10')
    plot_df, _, _ = require_columns(plot_df, required, fig_label='Fig 10')
    if len(plot_df) == 0:
        print("  Skipping Fig 10: 0 rows after calibrated_dimer + required columns.")
        return
    # Restrict to positive sequence_cluster_count and present shared_cluster_count.
    plot_df = plot_df[(plot_df['sequence_cluster_count'] > 0)
                      & plot_df['shared_cluster_count'].notna()].copy()
    if 'stoichiometry' in plot_df.columns:
        plot_df = plot_df[plot_df['stoichiometry'].isin(DIMER_STOICHIOMETRIES)].copy()
        if len(plot_df) == 0:
            print("  Skipping Fig 10: no calibrated A2/AB rows with cluster data.")
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
                kw_p_str = _format_pvalue(kw_p)
                n_total_kw = int(sum(len(g) for g in all_groups))
                eps_sq = _epsilon_squared(h_stat, n_total_kw, len(all_groups))
                # Headline effect size; K-W p as a small caption.
                stat_lines = [
                    f'Kruskal-Wallis ε² = {eps_sq:.3f}',
                    f'  (H = {h_stat:.1f}, {kw_p_str})',
                ]

                if 'High' in tier_data and 'Low' in tier_data:
                    _, mw_p = mannwhitneyu(tier_data['High'], tier_data['Low'], alternative='two-sided')
                    med_h = np.median(tier_data['High'])
                    med_l = np.median(tier_data['Low'])
                    mw_p_str = _format_pvalue(mw_p)
                    # Low/High median fold-change as a tier-comparison effect.
                    fold = (med_h / med_l) if med_l > 0 else float('inf')
                    fold_txt = f'High/Low fold = {fold:.2f}' if np.isfinite(fold) else 'High/Low fold = inf (Low median 0)'
                    stat_lines.append(f'{fold_txt}')
                    stat_lines.append(f'High median: {med_h:.3f}')
                    stat_lines.append(f'Low median: {med_l:.3f}')
                    stat_lines.append(f'High vs Low MWU: {mw_p_str}')

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
    figure.suptitle(
        f"Supplementary: Sequence-cluster consistency across calibrated dimer quality tiers "
        f"[{CAPTION_SCOPE_CALIBRATED_DIMER}; n={len(plot_df):,}]",
        fontsize=14, fontweight='bold', y=1.02)
    figure.text(0.5, -0.01,
                f'Note: N may be lower than the calibrated_dimer headline; requires '
                f'non-null sequence_cluster_count > 0 AND non-null shared_cluster_count.',
                ha='center', fontsize=7, style='italic', color='#777')
    plt.tight_layout()
    _save_figure(figure, '10_supp_Clustering_Validation.png')

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

#-------------------------------------Genome-variation figures---------------------------------------------

def plot_fig11_variant_consequence_flow(df: pd.DataFrame) -> None:
    """Fig 11: Where do clinically classified variants land structurally?
    Sankey (alluvial) diagram. 
    Left nodes: clinical significance categories (Unknown excluded). 
    Right nodes: 4 structural contexts. 
    Flow bands show how many variants of each significance land in each context.
    """
    df, n_before, n_after = apply_filter(df, 'calibrated_human_broad', fig_label='Fig 11')
    if len(df) == 0:
        print("  Skipping Fig 11: 0 rows after calibrated_human_broad filter.")
        return
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
    ax.text(0.50, 1.06,
            f"Supplementary: Classified Variant Flow from Available Per-chain Detail Fields "
            f"[{CAPTION_SCOPE_CALIBRATED_HUMAN_BROAD}]",
            ha='center', va='bottom', fontsize=14, fontweight='bold', transform=ax.transAxes)

    # Footer annotation. variant_details_a/b is capped at 20 per chain by
    # toolkit.py, so this figure is a structural-context illustration of
    # the AVAILABLE per-chain detail fields — not an enumeration of every
    # coding variant. The earlier "X shown / ~Y total" line was removed
    # because "total" had no consistent definition (it summed UniProt +
    # ClinVar + ExAC overlap and dwarfed the shown count by ~25x, which
    # consistently distracted readers from the actual structural-context
    # point of the figure).
    footer_parts = [
        f'n = {n_classified:,} classified variants from per-chain detail fields '
        f'({pct_unknown:.1f}% Unknown excluded, {n_unknown:,} variants)',
        'Per-chain detail fields are capped at 20 variants/chain; this figure '
        'shows the structural-context distribution from the available capped '
        'detail fields, not an exhaustive variant enumeration.',
        'VUS = variant of uncertain significance.',
    ]
    ax.text(0.50, -0.06, '\n'.join(footer_parts), ha='center', va='top', fontsize=9, style='italic', color='#444444', transform=ax.transAxes, linespacing=1.4)
    _save_figure(figure, '11_supp_Variant_Consequence_Flow.png')

def plot_fig12_variant_density(df: pd.DataFrame, density_mode: bool = False) -> None:
    """Fig 12: interface variant density versus composite confidence.
    Scatter plot of interface variant density (variants per interface residue) against composite score, coloured by quality tier.
    Spearman and partial correlations annotated. Tier-stratified median densities in text box.
    """
    df, n_before, n_after = apply_filter(df, 'calibrated_human_broad', fig_label='Fig 12')
    if len(df) == 0:
        print("  Skipping Fig 12: 0 rows after calibrated_human_broad filter.")
        return
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
    base_size, base_alpha, raster = _adaptive_scatter_params(len(x_vals))
    # Fig 7 readability: at HPC scale _adaptive_scatter_params returns s~1.2 and
    # alpha~0.14 which, combined with white edges, washes the tier colours out
    # almost completely. Enforce a visible floor on size/opacity and drop the
    # white edge so the coloured cloud reads clearly in the exported figure.
    # Rasterisation is retained for bounded file size at large N.
    point_size = max(base_size, 6.0)
    point_alpha = max(base_alpha, 0.38)
    colors = df.loc[valid_mask, tier_col].map(TIER_COLORS).fillna('#bdc3c7').values if tier_col in df.columns else '#3498db'

    _timed_scatter(ax, x_vals, y_vals, len(x_vals), fig_label='Fig 12', c=colors, s=point_size, alpha=point_alpha, edgecolors='none', rasterized=raster)

    if density_mode:
        _overlay_kde_contours(ax, x_vals, y_vals)

    #==============================Spearman correlation===================================
    stat_lines = []
    if len(x_vals) >= 5 and np.std(x_vals) > 1e-9 and np.std(y_vals) > 1e-9:
        rho, pval = spearmanr(x_vals, y_vals)
        p_str = _format_pvalue(pval)
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
                    p_str2 = _format_pvalue(pval_partial)
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

    # Cap the y-axis at the 99th percentile so a handful of high-density
    # outliers do not visually compress the bulk of the distribution.
    if len(y_vals) > 0:
        y99 = float(np.percentile(y_vals, 99))
        y_max = max(y99 * 1.05, y99 + 0.01)
        n_above = int((y_vals > y_max).sum())
        ax.set_ylim(bottom=0, top=y_max)
        if n_above > 0:
            ax.text(0.99, 0.01,
                    f'y-axis capped at 99th percentile ({y99:.2f}); '
                    f'{n_above} outlier(s) above range',
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=7, style='italic', color='#777777',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='none', alpha=0.85))

    _apply_common_style(ax, '',
                        'Interface confidence score',
                        'Variant density per interface residue')
    figure.suptitle(
        f"Variant density versus composite confidence "
        f"[broad-human calibrated dimers; n={len(df):,}]",
        fontsize=14, fontweight='bold', y=1.02)
    figure.text(0.5, -0.01,
                'This is annotation-supported correlation, not causal validation.',
                ha='center', fontsize=7, style='italic', color='#777')
    _save_figure(figure, 'Fig_7_Variant_Density_Versus_Composite_Confidence.png')

#--------------------------------------Stability-score figure-----------------------------------------------

def plot_fig13_stability_crossvalidation(df: pd.DataFrame) -> None:
    """Fig 13: concordance between EVE, AlphaMissense and monomeric FoldX stability scores.
    Panel A - EVE vs AlphaMissense concordance scatter (pooled both chains).
    Panel B - AlphaMissense vs monomeric FoldX DDG scatter (chain A).
    Panel C - Coverage landscape grouped bar chart by quality tier.
    Requires stability + ProtVar columns: eve_score_mean_a/b, protvar_am_mean_a/b, protvar_foldx_mean_a/b, eve_coverage_a/b, quality_tier_v2.
    """
    required = ['eve_score_mean_a', 'protvar_am_mean_a', 'quality_tier_v2']
    if not all(c in df.columns for c in required):
        print("  Skipping Fig 13 - missing stability/ProtVar columns")
        return

    # Dissertation-safe: stability comparisons use dimer-calibrated quality tiers
    # restricted to reviewed human (EVE/AlphaMissense assume canonical sequences).
    df, n_before, n_after = apply_filter(df, 'calibrated_human_strict', fig_label='Fig 13')
    if len(df) == 0:
        print("  Skipping Fig 13 - no calibrated reviewed-human complexes.")
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
        s, alpha, raster = _adaptive_scatter_params(len(eve_arr))
        ax_a.scatter(eve_arr, am_arr, c=colors_a, s=s, alpha=alpha, edgecolors='none', rasterized=raster)

        try:
            from scipy.stats import spearmanr
            rho, p = spearmanr(eve_arr, am_arr)
            p_str = _format_pvalue(p)
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
        s, alpha, raster = _adaptive_scatter_params(len(am_b_arr))
        ax_b.scatter(am_b_arr, fx_arr, c=colors_b, s=s, alpha=alpha, edgecolors='none', rasterized=raster)

        try:
            from scipy.stats import spearmanr
            rho, p = spearmanr(am_b_arr, fx_arr)
            p_str = _format_pvalue(p)
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
    # Cap FoldX y-axis at the 99th percentile so a handful of extreme \u0394\u0394G
    # values do not visually compress the bulk of the correlation cloud.
    if len(am_b) >= 10 and len(foldx_b) > 0:
        fy99 = float(np.percentile(foldx_b, 99))
        fy_max = max(fy99 * 1.05, fy99 + 0.5)
        n_above = int(sum(1 for v in foldx_b if v > fy_max))
        ax_b.set_ylim(top=fy_max)
        if n_above > 0:
            ax_b.text(0.99, 0.01,
                      f'FoldX y-axis capped at 99th pct ({fy99:.1f}); '
                      f'{n_above} outlier(s) above range',
                      transform=ax_b.transAxes, ha='right', va='bottom',
                      fontsize=7, style='italic', color='#777777',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor='none', alpha=0.85))
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
        f"Supplementary: Stability predictor cross-validation (coverage-limited) "
        f"[{CAPTION_SCOPE_CALIBRATED_HUMAN_STRICT}]",
        fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save_figure(figure, '13_supp_Stability_CrossValidation.png')

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
    """Fig 14: Disease annotation burden.
    Emits TWO separate figures (split for layout breathing room):
      14A_Disease_Prevalence_by_Tier.png - grouped bar chart of disease
            prevalence by tier with chi-square test and drug-target annotation.
      14B_Top_Disease_Categories_by_Tier.png - top 10 diseases as horizontal
            stacked bars segmented by quality tier.
    Requires 'n_diseases_a' column.
    """
    if 'n_diseases_a' not in df.columns:
        print("  Skipping Fig 14 - no disease data available")
        return

    tier_col = 'quality_tier_v2' if 'quality_tier_v2' in df.columns else 'quality_tier'
    if tier_col not in df.columns:
        print("  Skipping Fig 14 - no quality tier column")
        return

    df, n_before, n_after = apply_filter(df, 'calibrated_human_strict', fig_label='Fig 14')
    if len(df) == 0:
        print("  Skipping Fig 14 - no calibrated reviewed-human rows")
        return

    total_diseases = df['n_diseases_a'].fillna(0).astype(int)
    if 'n_diseases_b' in df.columns:
        total_diseases = total_diseases + df['n_diseases_b'].fillna(0).astype(int)

    has_disease = total_diseases > 0

    # --- Figure 14A: disease prevalence by tier (Panel A only) ---
    figure_a, ax_a = plt.subplots(figsize=(8, 6))

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
        p_str = _format_pvalue(p_val)
        cv = _cramers_v(contingency)
        # Tier fold-change: (Low disease rate) / (High disease rate). Headline.
        try:
            n_low_dis = float(tier_stats['Low']['n_disease'])
            n_low_tot = float(tier_stats['Low']['n_tier'])
            n_high_dis = float(tier_stats['High']['n_disease'])
            n_high_tot = float(tier_stats['High']['n_tier'])
            rate_low = n_low_dis / n_low_tot if n_low_tot > 0 else float('nan')
            rate_high = n_high_dis / n_high_tot if n_high_tot > 0 else float('nan')
            fold = rate_low / rate_high if rate_high > 0 else float('inf')
            fold_str = f'Low/High disease-rate fold = {fold:.2f}'
        except Exception:
            fold_str = ''
        annotation_lines = [f"Cram\u00e9r's V = {cv:.3f}"]
        if fold_str:
            annotation_lines.append(fold_str)
        annotation_lines.append(f'(\u03c7\u00b2 = {chi2:.1f}, {p_str})')
        ax_a.text(0.98, 0.95, '\n'.join(annotation_lines),
                  transform=ax_a.transAxes, ha='right', va='top',
                  fontsize=9, bbox=dict(boxstyle='round,pad=0.3',
                                        facecolor='white', edgecolor='grey', alpha=0.85))

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
            fisher_p_str = _format_pvalue(fisher_p)
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

    figure_a.suptitle(
        f"Supplementary: Disease annotation rate by quality tier "
        f"[{CAPTION_SCOPE_CALIBRATED_HUMAN_STRICT}]",
        fontsize=FONT_TITLE + 1, fontweight='bold', y=1.00)
    figure_a.text(
        0.5, -0.02,
        'IMPORTANT: this is ANNOTATION BURDEN, not disease causality. Tier '
        'imbalance and UniProt annotation bias (well-studied disordered/disease '
        'proteins are over-annotated) drive most tier-rank differences.',
        ha='center', fontsize=7, style='italic', color='#777777', wrap=True)
    figure_a.tight_layout()
    _save_figure(figure_a, '14A_supp_Disease_Prevalence_by_Tier.png')

    # --- Figure 14B: top disease categories (Panel B only) ---
    figure_b, ax_b = plt.subplots(figsize=(10, 6))

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

    figure_b.suptitle(
        f"Supplementary: Top UniProt disease annotations among reviewed-human calibrated dimers "
        f"[{CAPTION_SCOPE_CALIBRATED_HUMAN_STRICT}]",
        fontsize=FONT_TITLE + 1, fontweight='bold', y=1.00)
    figure_b.text(
        0.5, -0.02,
        'Top diseases ranked by unique-protein count. Reads as descriptive of '
        'the annotation database, not as biology of the prediction quality. '
        'Hub-protein effects can inflate annotation counts in proteins with many '
        'disease links.',
        ha='center', fontsize=7, style='italic', color='#777777', wrap=True)
    figure_b.tight_layout()
    _save_figure(figure_b, '14B_supp_Top_Disease_Categories_by_Tier.png')

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

def plot_fig15_pathway_bar_chart(df: pd.DataFrame, top_n: int = 20) -> None:
    """Fig 15 (main): top Reactome pathways as a horizontal bar chart.

    Replaces the network as the main-text Fig 15 — easier to read at
    dissertation scale than a 20-node graph. Each bar shows the number of
    calibrated reviewed-human complexes annotated to that pathway, coloured by
    the fraction of those complexes in the High quality_tier_v2 bucket.

    The network rendering remains available as `15_supp_Pathway_Network.png`
    via plot_fig15_pathway_network().
    """
    if 'reactome_pathways_a' not in df.columns:
        print("  Skipping Fig 15 - no pathway data available")
        return

    tier_col = 'quality_tier_v2' if 'quality_tier_v2' in df.columns else 'quality_tier'
    if tier_col not in df.columns:
        print("  Skipping Fig 15 - no quality tier column")
        return

    df, n_before, n_after = apply_filter(df, 'calibrated_human_strict', fig_label='Fig 15')
    if len(df) == 0:
        print("  Skipping Fig 15 - no calibrated reviewed-human rows")
        return

    pathway_counts: dict[str, int] = {}
    pathway_high: dict[str, int] = {}
    pathway_names: dict[str, str] = {}

    for _, row in df.iterrows():
        tier = str(row.get(tier_col, ''))
        seen_pids: set[str] = set()
        for suffix in ('a', 'b'):
            pathways_str = row.get(f'reactome_pathways_{suffix}', '')
            if not pathways_str or pd.isna(pathways_str):
                continue
            for entry in str(pathways_str).split('|'):
                if entry.startswith('...('):
                    continue
                parts = entry.split(':', 1)
                if len(parts) != 2:
                    continue
                pid, pname = parts
                pathway_names[pid] = pname
                seen_pids.add(pid)
        for pid in seen_pids:
            pathway_counts[pid] = pathway_counts.get(pid, 0) + 1
            if tier == 'High':
                pathway_high[pid] = pathway_high.get(pid, 0) + 1

    if not pathway_counts:
        print("  Skipping Fig 15 - no parseable pathway data")
        return

    top_pids = sorted(pathway_counts, key=pathway_counts.get, reverse=True)[:top_n]
    counts = np.array([pathway_counts[p] for p in top_pids], dtype=float)
    high_counts = np.array([pathway_high.get(p, 0) for p in top_pids], dtype=float)
    high_frac = np.where(counts > 0, high_counts / counts, 0.0)
    labels = [pathway_names.get(p, p) for p in top_pids]

    # Truncate long pathway names so bars stay readable.
    def _truncate(name: str, max_len: int = 60) -> str:
        return name if len(name) <= max_len else (name[:max_len].rsplit(' ', 1)[0] + '...')
    labels = [_truncate(n) for n in labels]

    figure, ax = plt.subplots(figsize=(11, max(5.5, 0.35 * len(top_pids) + 1)))
    cmap = plt.cm.RdYlGn
    vmax = max(0.05, np.ceil(max(high_frac.max(), 0.05) * 20) / 20)
    norm = plt.Normalize(vmin=0.0, vmax=vmax)
    colors = [cmap(norm(f)) for f in high_frac]
    y_pos = np.arange(len(top_pids))[::-1]
    bars = ax.barh(y_pos, counts, color=colors, edgecolor='white')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=FONT_TICK - 1)

    # Annotate each bar with N complexes and High%.
    x_max = counts.max() if len(counts) else 1.0
    for i, (c, hf) in enumerate(zip(counts, high_frac)):
        ax.text(c + x_max * 0.01, y_pos[i],
                f' n={int(c):,}, {hf*100:.0f}% High',
                va='center', fontsize=7, color='#555555')

    ax.set_xlim(0, x_max * 1.25)
    ax.set_xlabel('Calibrated reviewed-human complexes annotated to pathway',
                  fontsize=FONT_AXIS_LABEL)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    # Bars span ~2–5% High-tier; the colorbar previously stretched the full
    # figure height which dwarfed the data range. Shrink to 0.45 keeps the
    # gradient legible without dominating the panel.
    cbar = figure.colorbar(sm, ax=ax, shrink=0.45, pad=0.02)
    cbar.set_label('Fraction High tier (v2)', fontsize=FONT_TICK)
    _despine(ax)
    ax.set_title(
        f"Supplementary: Top {len(top_pids)} Reactome pathways by complex count "
        f"[{CAPTION_SCOPE_CALIBRATED_HUMAN_STRICT}]",
        fontsize=FONT_TITLE, fontweight='bold')
    figure.tight_layout()
    _save_figure(figure, '15_supp_Pathway_Bar_Chart.png')


def plot_fig15_pathway_network(df: pd.DataFrame,
                                max_pathways: int = 20,
                                min_shared_complexes: int = 20,
                                hierarchy_file: Optional[str] = 'data/pathways/ReactomePathwaysRelation.txt',
                                filter_hierarchy: bool = True,
                                depth_level: int = 1) -> None:
    """Fig 15 (supplementary): pathway network visualisation.
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

    df, n_before, n_after = apply_filter(df, 'calibrated_human_strict', fig_label='Fig 15')
    if len(df) == 0:
        print("  Skipping Fig 15 - no calibrated reviewed-human rows")
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

    # Label ALL nodes - centred inside node with collision avoidance.
    # Tighter trim (28 chars) and wider nudge to reduce centre-of-graph
    # label collisions observed in the round-4 render.
    node_list = list(G.nodes())
    node_size_map = dict(zip(node_list, node_sizes))
    label_data = []
    for n in node_list:
        full_name = pathway_names.get(n, n)
        name = full_name[:28] + '\u2026' if len(full_name) > 28 else full_name
        wrapped = _textwrap.fill(name, width=18)
        x, y = pos[n]
        sz = node_size_map[n]
        fs = 5 + 3 * (sz - 300) / max(1, 2700)
        label_data.append({'x': x, 'y': y, 'text': wrapped, 'fontsize': fs})

    # Collision nudge - sort top-to-bottom, push overlapping labels apart.
    label_data.sort(key=lambda d: -d['y'])
    nudge = 0.045
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
    figure.suptitle(
        f"Supplementary: Reactome Pathway Network by Structural Quality "
        f"[{CAPTION_SCOPE_CALIBRATED_HUMAN_STRICT}]",
        fontsize=FONT_TITLE + 1, fontweight='bold')

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
    _save_figure(figure, '15_supp_Pathway_Network.png')

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
      Top row  - "Interface-level corroboration":
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
    wdf, n_before, n_after = apply_filter(df, 'calibrated_human_broad', fig_label='Fig 16')
    wdf = wdf.copy()
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

    def _annotate_panel(ax, res, is_binary=False, direction='up',
                        effect_lines=None):
        """Add effect-size headline + omnibus p-value note + arrow."""
        lines = list(effect_lines) if effect_lines else []
        _, p_omni = res['omnibus']
        p_str = _format_pvalue(p_omni)
        test_name = 'Chi-sq' if is_binary else 'K-W'
        lines.append(f'({test_name}: {p_str})')
        ax.text(0.97, 0.95, '\n'.join(lines), transform=ax.transAxes,
                ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          edgecolor='grey', alpha=0.85))
        if direction == 'up':
            ax.text(0.97, 0.70, '\u2191 Signal strengthens',
                    transform=ax.transAxes, ha='right', va='top', fontsize=8,
                    color='#27AE60', fontweight='bold')
        else:
            ax.text(0.97, 0.70, '\u2193 Prediction bias',
                    transform=ax.transAxes, ha='right', va='top', fontsize=8,
                    color='#E74C3C', fontweight='bold')

    def _binary_effect_lines(counts, totals):
        """High vs Low odds ratio + 95% CI; format for the annotation box."""
        a = counts.get('High', 0)
        b = max(totals.get('High', 0) - a, 0)
        c = counts.get('Low', 0)
        d = max(totals.get('Low', 0) - c, 0)
        if (a + b) == 0 or (c + d) == 0:
            return ["OR (High vs Low) = n/a"]
        or_, lo, hi = _odds_ratio_ci(a, b, c, d)
        return [f"OR (High vs Low) = {or_:.2f}",
                f"  [95% CI {lo:.2f}\u2013{hi:.2f}]"]

    def _continuous_effect_lines(data_by_tier):
        """Cliff's delta (High vs Low) + tier medians as headline."""
        med_lines = []
        for t in TIER_ORDER:
            vals = np.asarray(data_by_tier.get(t, []), dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                med_lines.append(f"  {t} median: {float(np.median(vals)):.3f}")
        high_vals = np.asarray(data_by_tier.get('High', []), dtype=float)
        low_vals = np.asarray(data_by_tier.get('Low', []), dtype=float)
        if len(high_vals) > 0 and len(low_vals) > 0:
            cd = _cliffs_delta(high_vals, low_vals)
            return [f"Cliff's \u03b4 (High vs Low) = {cd:.3f}"] + med_lines
        return med_lines or ["effect: insufficient data"]

    #========================Panel A: Pathogenic Interface Variant Rate (grouped bar)=============================
    ax_a = axes[0, 0]
    counts_a = {t: int(wdf.loc[wdf['quality_tier_v2'] == t, '_has_path_iface'].sum()) for t in tiers}
    totals_a = {t: int((wdf['quality_tier_v2'] == t).sum()) for t in tiers}
    _grouped_bar_panel(ax_a, counts_a, totals_a, ylabel='Fraction with Pathogenic\nInterface Variants', title='Pathogenic Interface Variants by Tier')
    res_a = _run_pairwise_tests_binary(counts_a, totals_a)
    _annotate_panel(ax_a, res_a, is_binary=True, direction='up',
                    effect_lines=_binary_effect_lines(counts_a, totals_a))

    #=============================Panel B: PPI Enrichment Ratio (violin+box, log10(x+1))=============================
    ax_b = axes[0, 1]
    col_b = 'ppi_enrichment_ratio'
    sub_b = wdf.dropna(subset=[col_b])
    # Plot log10(ratio + 1) to handle zero values cleanly. The raw
    # ratio is extremely right-skewed and contains zeros, both of which break
    # set_yscale('log').
    raw_b = {t: sub_b.loc[sub_b['quality_tier_v2'] == t, col_b].astype(float).values for t in tiers}
    data_b = {t: np.log10(np.asarray(v, dtype=float) + 1.0) for t, v in raw_b.items()}
    _violin_box_panel(ax_b, data_b, ylabel='log10(PPI enrichment ratio + 1)', title='PPI Enrichment Ratio by Tier (log10)')
    res_b = _run_pairwise_tests_continuous(data_b)
    _annotate_panel(ax_b, res_b, direction='up',
                    effect_lines=_continuous_effect_lines(data_b))
    # STRING saturation note moved to figure footer

    #=============================Panel C: LoF-Intolerant Genes, pLI >= 0.9 (grouped bar)=============================
    ax_c = axes[1, 0]
    sub_c = wdf.dropna(subset=['_pli_constrained'])
    counts_c = {t: int(sub_c.loc[sub_c['quality_tier_v2'] == t, '_pli_constrained'].sum()) for t in tiers}
    totals_c = {t: int((sub_c['quality_tier_v2'] == t).sum()) for t in tiers}
    _grouped_bar_panel(ax_c, counts_c, totals_c, ylabel='Fraction with pLI \u2265 0.9', title='LoF-Intolerant Genes (pLI \u2265 0.9) by Tier')
    res_c = _run_pairwise_tests_binary(counts_c, totals_c)
    _annotate_panel(ax_c, res_c, is_binary=True, direction='down',
                    effect_lines=_binary_effect_lines(counts_c, totals_c))

    #=============================Panel D: Disorder Fraction (violin+box)=============================
    ax_d = axes[1, 1]
    col_d = 'plddt_below50_fraction'
    sub_d = wdf.dropna(subset=[col_d])
    data_d = {t: sub_d.loc[sub_d['quality_tier_v2'] == t, col_d].values for t in tiers}
    _violin_box_panel(ax_d, data_d, ylabel='Disorder Fraction (pLDDT < 50)', title='Disorder Fraction by Tier')
    res_d = _run_pairwise_tests_continuous(data_d)
    _annotate_panel(ax_d, res_d, direction='down',
                    effect_lines=_continuous_effect_lines(data_d))


    #=============================================Figure footer===================================================
    footer_parts = [
        'Panel B note: STRING p-values saturate at a numerical floor for '
        'large pathways (shown as p < 1e-300 where applicable), so the '
        'observed-to-expected enrichment ratio is the discriminative metric.'
    ]
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

    fig.suptitle(
        f"Biological corroboration and prediction bias "
        f"[reviewed-human calibrated dimers; n={len(wdf):,}]",
        fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save_figure(fig, 'Fig_8_Biological_Corroboration_and_Prediction_Bias.png')

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

#---------------------------------------Dataset-funnel, screening and recoverability figures-----------------------------------------------

def plot_fig0_corpus_funnel(df: pd.DataFrame) -> None:
    """Fig 0: Final corpus analysis funnel.

    Two-part layout:
      * Main funnel (true subset chain): all_rows -> recoverable -> calibrated_dimer
        -> calibrated_human_broad -> calibrated_human_strict.
      * Side callouts (parallel bars, contextual populations - not a partition):
        partial_error, multimer_exploratory, strong/moderate/weak_screen_candidate.

    `composite_screenable` is intentionally NOT on the main funnel because it
    equals `calibrated_dimer` in the audit; presenting it as a downstream step
    would imply a linear filtering relationship that does not exist.
    """
    MAIN_FUNNEL_ORDER = [
        'all_rows',
        'recoverable',
        'calibrated_dimer',
        'calibrated_human_broad',
        'calibrated_human_strict',
    ]
    SIDE_CALLOUT_ORDER = [
        'partial_error',
        'multimer_exploratory',
        'strong_screen_candidate',
        'moderate_screen_candidate',
        'weak_screen_candidate',
    ]

    counts = {}
    for name in MAIN_FUNNEL_ORDER + SIDE_CALLOUT_ORDER:
        try:
            counts[name] = int(FILTER_REGISTRY[name](df).sum())
        except Exception:
            counts[name] = 0

    total = max(counts.get('all_rows', 1), 1)

    figure, (ax_main, ax_side) = plt.subplots(
        1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [3, 2]})

    # Main funnel: horizontal bars, descending. Uses _display() for the
    # bucket name and appends a "(X% of <parent>)" line so the relationship
    # between consecutive funnel steps is visible at a glance.
    main_labels = []
    main_values = []
    for idx, name in enumerate(MAIN_FUNNEL_ORDER):
        n = counts.get(name, 0)
        pct_of_total = 100.0 * n / total
        if idx == 0:
            parent_label = ''
        else:
            parent_name = MAIN_FUNNEL_ORDER[idx - 1]
            parent_n = counts.get(parent_name, 0)
            if parent_n > 0:
                pct_of_parent = 100.0 * n / parent_n
                parent_label = f"\n({pct_of_parent:.1f}% of {_display(parent_name)})"
            else:
                parent_label = ''
        main_labels.append(
            f"{_display(name)}\n{n:,} ({pct_of_total:.1f}% of all rows){parent_label}"
        )
        main_values.append(n)
    main_colours = ['#2c3e50', '#34495e', '#16a085', '#27ae60', '#229954']
    y_positions = np.arange(len(MAIN_FUNNEL_ORDER))[::-1]
    ax_main.barh(y_positions, main_values, color=main_colours, edgecolor='white')
    ax_main.set_yticks(y_positions)
    ax_main.set_yticklabels(main_labels, fontsize=FONT_TICK - 1)
    ax_main.set_xlabel('Number of complexes', fontsize=FONT_AXIS_LABEL)
    ax_main.set_title('Main funnel (true subsets)', fontsize=FONT_TITLE, fontweight='bold')
    _despine(ax_main)

    # Side callouts: contextual populations (NOT a partition). Use _display().
    # strong/moderate/weak screen candidates partition the calibrated dimer
    # set, so for those rows we annotate BOTH the full-corpus denominator
    # AND the calibrated-dimer denominator. partial_error and
    # multimer_exploratory don't partition anything, so they get the
    # single full-corpus percentage only.
    side_labels = []
    side_values = []
    calibrated_n = counts.get('calibrated_dimer', 0)
    screen_candidates = {
        'strong_screen_candidate',
        'moderate_screen_candidate',
        'weak_screen_candidate',
    }
    for name in SIDE_CALLOUT_ORDER:
        n = counts.get(name, 0)
        pct = 100.0 * n / total
        if name in screen_candidates and calibrated_n > 0:
            pct_calib = 100.0 * n / calibrated_n
            side_labels.append(
                f"{_display(name)}\n{n:,} "
                f"({pct:.1f}% of all rows;\n{pct_calib:.1f}% of calibrated dimers)"
            )
        else:
            side_labels.append(f"{_display(name)}\n{n:,} ({pct:.1f}%)")
        side_values.append(n)
    side_colours = ['#c0392b', '#8e44ad', '#16a085', '#f39c12', '#7f8c8d']
    side_y = np.arange(len(SIDE_CALLOUT_ORDER))[::-1]
    ax_side.barh(side_y, side_values, color=side_colours, edgecolor='white')
    ax_side.set_yticks(side_y)
    ax_side.set_yticklabels(side_labels, fontsize=FONT_TICK - 1)
    ax_side.set_xlabel('Number of complexes', fontsize=FONT_AXIS_LABEL)
    ax_side.set_title('Side callouts (contextual)', fontsize=FONT_TITLE, fontweight='bold')
    # 5000-step default ticks crowd into illegibility on the narrow side
    # panel; cap to ~6 evenly spaced labels.
    ax_side.xaxis.set_major_locator(MaxNLocator(nbins=6))
    _despine(ax_side)

    figure.suptitle(
        f"Dataset and analysis populations "
        f"[full dataset; n={total:,}]",
        fontsize=14, fontweight='bold', y=1.02)
    figure.text(
        0.5, -0.02,
        "Side callouts are contextual populations, not a partition of the "
        "dataset. The strong/moderate/weak screen bands do partition the "
        "calibrated dimer set, so those rows carry both a full-dataset and a "
        "calibrated-dimer percentage; partial_error and multimer_exploratory "
        "overlap neither the main funnel nor the screen bands by construction.",
        ha='center', fontsize=8, style='italic', color='#555555', wrap=True)
    plt.tight_layout()
    _save_figure(figure, 'Fig_1_Dataset_and_Analysis_Population_Funnel.png')


def plot_fig17_screening_landscape(df: pd.DataFrame) -> None:
    """Fig 17: Composite screening landscape (classification vs prioritisation).

    Within calibrated_dimer:
      Panel A - histogram of interface_confidence_score, hue=composite_screen_status,
                vertical lines at 0.63 and 0.85.
      Panel B - stacked bar quality_tier_v2 x composite_screen_status. Expected
                bands are strong/moderate/weak only; `unavailable` is absent here
                by construction (it lives outside the calibrated screenable
                population).

    Classification axis is `quality_tier_v2`; screening axis is
    `composite_screen_status`. These are NOT interchangeable.
    """
    required = ['interface_confidence_score', 'composite_screen_status', 'quality_tier_v2']
    sub, n_before, n_after = apply_filter(df, 'calibrated_dimer', fig_label='Fig 17')
    sub, _, n_plot = require_columns(sub, required, fig_label='Fig 17')
    if len(sub) == 0:
        print("  Skipping Fig 17: 0 rows after calibrated_dimer + required columns.")
        return

    figure, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(15, 6))
    status_order = ['weak_screen_candidate', 'moderate_screen_candidate',
                    'strong_screen_candidate']
    status_colours = {
        'weak_screen_candidate':     '#7f8c8d',
        'moderate_screen_candidate': '#f39c12',
        'strong_screen_candidate':   '#27ae60',
        'unavailable':               '#bdc3c7',
    }

    # Panel A: histogram of interface_confidence_score, coloured by screen status.
    bins = np.linspace(0.0, 1.0, 41)
    for status in status_order:
        vals = sub.loc[sub['composite_screen_status'] == status,
                       'interface_confidence_score'].dropna().values
        if len(vals) == 0:
            continue
        ax_a.hist(vals, bins=bins, alpha=0.55, color=status_colours[status],
                  label=f"{status}\n(n={len(vals):,})", edgecolor='white', linewidth=0.3)
    ax_a.axvline(0.63, color='#e74c3c', linestyle='--', linewidth=1.2,
                 alpha=0.8, label='weak/moderate cutoff (0.63)')
    ax_a.axvline(0.85, color='#16a085', linestyle='--', linewidth=1.2,
                 alpha=0.8, label='moderate/strong cutoff (0.85)')
    ax_a.set_xlabel('Interface Confidence Score', fontsize=FONT_AXIS_LABEL)
    ax_a.set_ylabel('Count', fontsize=FONT_AXIS_LABEL)
    ax_a.set_title('(a) Continuous screening signal', fontsize=FONT_TITLE,
                   fontweight='bold')
    ax_a.legend(fontsize=FONT_TICK - 1, loc='upper left', framealpha=0.9)
    _despine(ax_a)

    # Panel B: heatmap of quality_tier_v2 x composite_screen_status.
    # Stacked bars were dominated by the Low|weak cell; a heatmap with cells
    # showing raw N + row % makes the per-tier composition readable.
    crosstab = pd.crosstab(sub['quality_tier_v2'], sub['composite_screen_status'])
    crosstab = crosstab.reindex(index=TIER_ORDER, columns=status_order, fill_value=0)
    row_sums = crosstab.sum(axis=1).replace(0, np.nan)
    row_pct = crosstab.div(row_sums, axis=0).fillna(0) * 100
    im = ax_b.imshow(row_pct.values, cmap='YlOrRd', aspect='auto',
                     vmin=0, vmax=100)
    for i, tier in enumerate(TIER_ORDER):
        for j, status in enumerate(status_order):
            n = int(crosstab.iat[i, j])
            pct = float(row_pct.iat[i, j])
            ax_b.text(j, i, f"{n:,}\n({pct:.1f}%)",
                      ha='center', va='center',
                      fontsize=9, fontweight='bold',
                      color='white' if pct > 50 else '#333')
    ax_b.set_xticks(range(len(status_order)))
    ax_b.set_xticklabels([_display(s) for s in status_order],
                         fontsize=FONT_AXIS_LABEL, rotation=12)
    ax_b.set_yticks(range(len(TIER_ORDER)))
    ax_b.set_yticklabels(
        [f"{t}\n(n={int(crosstab.loc[t].sum()):,})" for t in TIER_ORDER],
        fontsize=FONT_AXIS_LABEL)
    ax_b.set_title('(b) Classification by screening status (row %)',
                   fontsize=FONT_TITLE, fontweight='bold')
    cbar = figure.colorbar(im, ax=ax_b, shrink=0.6, pad=0.04)
    cbar.set_label('Row %', fontsize=FONT_TICK)

    figure.suptitle(
        f"Classification versus continuous screening "
        f"[{CAPTION_SCOPE_CALIBRATED_DIMER}; n={len(sub):,}]",
        fontsize=14, fontweight='bold', y=1.02)
    figure.text(
        0.5, -0.02,
        "Classification (quality_tier_v2) and continuous screening "
        "(composite_screen_status) are separate outputs, but both derive from "
        "the same adopted composite thresholds, so panel (b) is a "
        "policy-consistency view, not an independent validation. The "
        "unavailable screen status is excluded here by construction: it lies "
        "outside the calibrated screenable population.",
        ha='center', fontsize=8, style='italic', color='#555555', wrap=True)
    plt.tight_layout()
    _save_figure(figure, 'Fig_4_Classification_Versus_Screening.png')


def plot_fig18_partial_reason_dashboard(df: pd.DataFrame) -> None:
    """Fig 18: Input recoverability dashboard (gated on --include-partial-diagnostics).

    Three panels to keep the dominant `pdb_decompression_error` bucket from
    hiding the smaller failure modes:
      Panel A - dominant failure (single bar; usually pdb_decompression_error).
      Panel B - minor partial_reason categories on their own scale.
      Panel C - calibrated vs non-calibrated row counts within recoverable.
    """
    if 'partial_reason' not in df.columns:
        print("  Skipping Fig 18: partial_reason column not present.")
        return

    partial, n_before, n_after = apply_filter(df, 'partial_error', fig_label='Fig 18')
    if len(partial) == 0:
        print("  Skipping Fig 18: 0 rows in partial_error.")
        return

    reason_counts = (
        partial['partial_reason']
        .astype(str)
        .str.strip()
        .replace('', '__BLANK__')
        .value_counts()
    )

    # Split dominant (largest) reason from the remaining minor reasons so the
    # secondary scale is readable. Falls back gracefully when there's only one
    # bucket.
    if len(reason_counts) >= 2:
        dominant_label = reason_counts.index[0]
        dominant_count = int(reason_counts.iloc[0])
        minor_counts = reason_counts.iloc[1:]
    else:
        dominant_label = reason_counts.index[0] if len(reason_counts) else 'none'
        dominant_count = int(reason_counts.iloc[0]) if len(reason_counts) else 0
        minor_counts = reason_counts.iloc[1:]

    figure = plt.figure(figsize=(16, 6))
    gs = figure.add_gridspec(1, 3, width_ratios=[1.2, 3, 1])
    ax_a = figure.add_subplot(gs[0])
    ax_b = figure.add_subplot(gs[1])
    ax_c = figure.add_subplot(gs[2])

    # Panel A: dominant failure (single tall bar).
    ax_a.bar([0], [dominant_count], color='#c0392b', edgecolor='white')
    ax_a.set_xticks([0])
    ax_a.set_xticklabels([f"{dominant_label}\n({dominant_count:,})"],
                         fontsize=FONT_TICK)
    ax_a.set_ylabel('Number of rows', fontsize=FONT_AXIS_LABEL)
    ax_a.set_title('(a) Dominant failure', fontsize=FONT_TITLE, fontweight='bold')
    _despine(ax_a)

    # Panel B: minor partial_reason categories.
    if len(minor_counts) > 0:
        y_pos = np.arange(len(minor_counts))[::-1]
        ax_b.barh(y_pos, minor_counts.values, color='#e67e22', edgecolor='white')
        ax_b.set_yticks(y_pos)
        ax_b.set_yticklabels(
            [f"{r} ({c:,})" for r, c in minor_counts.items()],
            fontsize=FONT_TICK)
        ax_b.set_xlabel('Number of rows', fontsize=FONT_AXIS_LABEL)
        ax_b.set_title(f'(b) Minor failure modes (excluding {dominant_label})',
                       fontsize=FONT_TITLE, fontweight='bold')
    else:
        ax_b.text(0.5, 0.5, 'No other partial_reason values present',
                  transform=ax_b.transAxes, ha='center', va='center',
                  fontsize=FONT_AXIS_LABEL, color='grey')
        ax_b.set_title('(b) Minor failure modes',
                       fontsize=FONT_TITLE, fontweight='bold')
    _despine(ax_b)

    # Panel C: calibrated vs uncalibrated within recoverable.
    recoverable, _, _ = apply_filter(df, 'recoverable', fig_label='Fig 18c')
    if 'composite_is_calibrated' in recoverable.columns:
        calibrated = recoverable['composite_is_calibrated'].map(
            lambda v: parse_boolish(v) is True)
        n_cal = int(calibrated.sum())
        n_uncal = len(recoverable) - n_cal
    else:
        n_cal, n_uncal = 0, len(recoverable)
    bar_labels = [f'Calibrated\n({n_cal:,})', f'Uncalibrated\n({n_uncal:,})']
    bar_values = [n_cal, n_uncal]
    bars = ax_c.bar([0, 1], bar_values, color=['#27ae60', '#7f8c8d'],
                    edgecolor='white')
    ax_c.set_xticks([0, 1])
    ax_c.set_xticklabels(bar_labels, fontsize=FONT_AXIS_LABEL)
    ax_c.set_ylabel('Number of rows (recoverable, log)',
                    fontsize=FONT_AXIS_LABEL)
    ax_c.set_title('(c) Calibrated vs uncalibrated\n(recoverable)',
                   fontsize=FONT_TITLE, fontweight='bold')
    # Calibrated outnumbers uncalibrated ~100x; a linear axis flattens the
    # smaller bar to 0 px. Log keeps both bars visible while honouring the
    # real magnitude difference.
    if min(v for v in bar_values if v > 0) > 0:
        ax_c.set_yscale('log')
    for bar, value in zip(bars, bar_values):
        if value > 0:
            ax_c.text(bar.get_x() + bar.get_width() / 2,
                      value, f'{value:,}',
                      ha='center', va='bottom', fontsize=FONT_TICK)
    _despine(ax_c)

    figure.suptitle(
        f"Supplementary: Input recoverability diagnostic - partial rows excluded from calibrated analyses "
        f"[{CAPTION_SCOPE_PARTIAL}; partial_error n={len(partial):,}]",
        fontsize=14, fontweight='bold', y=1.02)
    # Explicit framing: this dashboard is for audit/recoverability QA only.
    # The dominant pdb_decompression_error bar is an input-handling
    # diagnostic, not a confidence-score class or a biological result.
    figure.text(
        0.5, -0.04,
        'Partial/error rows are retained for auditability but excluded '
        'from calibrated structural analyses. The dominant failure mode '
        'reflects unreadable or failed decompression of compressed PDB '
        'inputs in the final corpus, not a confidence-score class.',
        ha='center', va='top', fontsize=8, style='italic',
        color='#555555', wrap=True)
    plt.tight_layout()
    _save_figure(figure, '18_supp_Partial_Reason_Dashboard.png')


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

    #======================Optional: load behaviour & final-corpus figures=========
    parser.add_argument(
        '--legacy-mode', action='store_true',
        help='Re-enable old load_data() row-drop on missing/zero ipTM only. Does '
             'NOT restore old v2 thresholds, old captions, or old figure filtering. '
             'For reproducing the older load behaviour only.')
    parser.add_argument(
        '--screening-figures', action=argparse.BooleanOptionalAction, default=True,
        help='Render Fig 17 screening landscape (use --no-screening-figures to disable).')
    parser.add_argument(
        '--corpus-funnel', action=argparse.BooleanOptionalAction, default=True,
        help='Render Fig 0 corpus funnel (use --no-corpus-funnel to disable).')
    parser.add_argument(
        '--include-partial-diagnostics', action='store_true',
        help='Render Fig 18 (partial-reason dashboard). Specialist flag; '
             'also implied by --full-figure-pack.')
    parser.add_argument(
        '--skip-diagnostics', action='store_true',
        help='Skip the warn_missing_required_rows() summaries.')

    #======================Main vs supplement / species variants==========
    parser.add_argument(
        '--full-figure-pack', action='store_true',
        help='Render every supplementary figure (*_supp_*) in addition to the '
             'main-text bundle. Default output is the main-text figure pack '
             'only (Figs 0, 1, 3, 4, 5, 7, 8 delta-histogram, 12, 16, 17). '
             'Implies --disorder-scatter and --include-partial-diagnostics.')
    parser.add_argument(
        '--human-supplement', action='store_true',
        help='Also render structural figures on the human subset; output '
             'filenames suffixed _human. Alone: main structural figures only. '
             'With --full-figure-pack: main + supplementary structural figures.')
    parser.add_argument(
        '--nonhuman-supplement', action='store_true',
        help='Also render structural figures on the non-human subset; output '
             'filenames suffixed _nonhuman. Fig 9 is force-skipped. Alone: '
             'main structural figures only. With --full-figure-pack: main + '
             'supplementary structural figures.')
    parser.add_argument(
        '--species-supplements', action='store_true',
        help='DEPRECATED: equivalent to --human-supplement --nonhuman-supplement. '
             'Use the explicit flags instead.')

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
    # Resolve flag interactions BEFORE printing.
    # --species-supplements is a deprecated alias that fans out to the two
    # explicit flags. --full-figure-pack implies the specialist supplementary
    # toggles (--disorder-scatter and --include-partial-diagnostics).
    if args.species_supplements:
        print("  Note: --species-supplements is deprecated; use "
              "--human-supplement and/or --nonhuman-supplement.")
        args.human_supplement = True
        args.nonhuman_supplement = True

    render_supplements = bool(getattr(args, 'full_figure_pack', False))
    render_disorder_scatter = args.disorder_scatter or render_supplements
    render_partial_diagnostics = args.include_partial_diagnostics or render_supplements

    active_flags = []
    if args.density: active_flags.append('density (KDE contours)')
    if args.legacy_mode: active_flags.append('legacy-mode (load-time ipTM drop)')
    if args.multimer_supplement: active_flags.append('multimer-supplement')
    if render_supplements: active_flags.append('full-figure-pack (*_supp_* figures)')
    if args.human_supplement: active_flags.append('human-supplement (_human variants)')
    if args.nonhuman_supplement: active_flags.append('nonhuman-supplement (_nonhuman variants; Fig 9 skipped)')
    if render_disorder_scatter and not render_supplements: active_flags.append('disorder-scatter (Fig 1b)')
    if render_partial_diagnostics and not render_supplements: active_flags.append('partial-diagnostics (Fig 18)')
    if not args.screening_figures: active_flags.append('screening-figures DISABLED')
    if not args.corpus_funnel: active_flags.append('corpus-funnel DISABLED')
    if active_flags:
        print(f"Active flags:     {', '.join(active_flags)}")
    print()

    # Load CSV
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print(f"Loading data from {csv_path}...")
    df = load_data(csv_path, legacy_mode=args.legacy_mode)
    print(f"Loaded {len(df):,} complex records.")

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

    # Console summary of pLDDT source counts.
    if 'plddt_source' in df.columns:
        source_counts = df['plddt_source'].value_counts()
        source_parts = [f"{count} {source}" for source, count in source_counts.items()]
        print(f"\n  pLDDT source: {', '.join(source_parts)}")

    figures_generated = 0
    interface_figs_skipped_warning_shown = False

    # Fig 0 - dataset/analysis population funnel (runs once on the full df).
    if args.corpus_funnel:
        print("\nFig 0 - Final Corpus Analysis Funnel")
        plot_fig0_corpus_funnel(df)
        figures_generated += 1

    # Primary pass runs structural figures on the full calibrated-dimer
    # population, unsplit by species. Each figure applies `calibrated_dimer`
    # internally, so passing the full df produces the dissertation-default
    # scope. Species-specific variants are opt-in via --human-supplement /
    # --nonhuman-supplement. The fourth tuple element (run_supp) tells the
    # per-pass loop whether to also emit supplementary figures for this pass.
    structural_passes = [(df, '', 'Primary (all species)', render_supplements)]
    if args.human_supplement and 'species_status' in df.columns:
        structural_passes.append((df_all_human, '_human', 'Human', render_supplements))
        print("\n--- Human supplement requested; will also generate _human variants ---")
    if args.nonhuman_supplement and 'species_status' in df.columns:
        structural_passes.append((df_nonhuman, '_nonhuman', 'Non-Human', render_supplements))
        print("\n--- Non-human supplement requested; will also generate _nonhuman variants ---")

    for df_subset, suffix, display_label, run_supp_for_this_pass in structural_passes:
        if len(df_subset) == 0:
            print(f"\n  Skipping {display_label} figures - empty subset.")
            continue
        header = f" ({display_label}, n={len(df_subset):,})" if display_label else ""
        label_suffix = f" - {display_label}" if display_label else ""
        print(f"\n--- Generating Figures{header} ---\n")

        # Main: Fig 1 (always).
        print(f"Fig 1 - Quality Scatter (ipTM vs pDockQ){label_suffix}")
        plot_fig1_quality_scatter(df_subset, col_flags, density_mode=args.density, species_label=suffix)
        figures_generated += 1

        # Supp: Fig 2 (PAE health check).
        if run_supp_for_this_pass:
            print(f"Fig 2 (supp) - Global PAE Health Check{label_suffix}")
            plot_fig2_pae_health_check(df_subset, species_label=suffix)
            figures_generated += 1

        # Supp: Fig 1b (only when --disorder-scatter or --full-figure-pack AND V2 data).
        if render_disorder_scatter and col_flags['has_v2_data']:
            print(f"Fig 1b (supp) - Disorder Scatter{label_suffix}")
            plot_fig1b_disorder_scatter(df_subset, density_mode=args.density, species_label=suffix)
            figures_generated += 1

        # Interface figures (require V2 + interface data): Figs 3-8
        if col_flags['has_v2_data'] and col_flags['has_interface_data']:
            # Main: Fig 3.
            print(f"Fig 3 - Interface PAE by Tier{label_suffix}")
            plot_fig3_interface_pae_by_tier(df_subset, species_label=suffix)
            figures_generated += 1

            # Main: Fig 4.
            print(f"Fig 4 - Composite & Tier Validation{label_suffix}")
            plot_fig4_composite_validation(df_subset, density_mode=args.density, species_label=suffix)
            figures_generated += 1

            # Supp: Fig 4 supp (strict vs PAE-only fraction).
            if run_supp_for_this_pass and 'pae_confident_contact_fraction' in df_subset.columns and 'strict_confident_contact_fraction' in df_subset.columns:
                print(f"Fig 4 supp - Strict vs PAE-only Fraction{label_suffix}")
                plot_fig4_supp_strict_vs_pae_only(df_subset, species_label=suffix)
                figures_generated += 1

            # Main: Fig 5.
            print(f"Fig 5 - Interface vs Bulk{label_suffix}")
            plot_fig5_interface_vs_bulk(df_subset, density_mode=args.density, species_label=suffix)
            figures_generated += 1

            # Supp: Fig 6 (paradox spotlight).
            if run_supp_for_this_pass:
                print(f"Fig 6 (supp) - Paradox Spotlight{label_suffix}")
                plot_fig6_paradox_spotlight(df_subset, species_label=suffix)
                figures_generated += 1

            # Main: Fig 7.
            print(f"Fig 7 - Architecture (calibrated A2/AB primary{', + multimer supp' if args.multimer_supplement else ''}){label_suffix}")
            plot_fig7_homo_vs_hetero(df_subset, species_label=suffix,
                                      multimer_supplement=args.multimer_supplement)
            figures_generated += 1

            # Main: Fig 5 - ipTM/pDockQ categorical agreement matrix.
            print(f"Fig 5 - ipTM/pDockQ categorical agreement matrix{label_suffix}")
            plot_fig8_iptm_pdockq_delta_histogram(df_subset, density_mode=args.density, species_label=suffix)
            figures_generated += 1
            # Supp: old Fig 8 scatter (descriptive).
            if run_supp_for_this_pass:
                print(f"Fig 8 (supp) - ipTM vs pDockQ scatter (descriptive){label_suffix}")
                plot_fig8_supp_metric_disagreement_scatter(df_subset, density_mode=args.density, species_label=suffix)
                figures_generated += 1
        elif not interface_figs_skipped_warning_shown:
            print("\nInterface figures (3-8) require V2 quality tiers AND interface")
            print("columns in the CSV. Re-run the batch script with interface analysis")
            print("enabled to include the interface columns.")
            interface_figs_skipped_warning_shown = True

        # Supp: Fig 9 (chain-count profile). Hard-skip non-human pass per plan A.3.
        if col_flags['has_chain_info'] and run_supp_for_this_pass and suffix != '_nonhuman':
            print(f"Fig 9 (supp) - Chain-Count Quality Profile{label_suffix}")
            plot_fig9_chain_count_profile(df_subset, density_mode=args.density, species_label=suffix)
            figures_generated += 1

    # Enrichment figures (10-16). Figs 10-12 use reviewed+TrEMBL (df_all_human)
    # since STRING clusters and ClinVar cover TrEMBL well. Figs 13-16 use
    # reviewed-only (df_human) because EVE/AlphaMissense/UniProt/Reactome data
    # is sparse on TrEMBL.
    print("\n--- Generating Enrichment Figures (Human) ---\n")

    # Supp: Fig 10 (clustering validation).
    if render_supplements and col_flags.get('has_clustering_data', False) and 'complex_type' in df_all_human.columns:
        print("Fig 10 (supp) - Sequence Clustering Validation (reviewed + TrEMBL)")
        plot_fig10_clustering_validation(df_all_human)
        figures_generated += 1

    # Variant figures (require --variants). Fig 11 is supplementary, Fig 12 is main.
    if col_flags['has_variant_data']:
        if render_supplements:
            print("Fig 11 (supp) - Classified Variant Sankey from Per-chain Detail Fields")
            plot_fig11_variant_consequence_flow(df_all_human)
            figures_generated += 1

        print("Fig 12 - Interface Variant Density vs Quality (reviewed + TrEMBL)")
        plot_fig12_variant_density(df_all_human, density_mode=args.density)
        figures_generated += 1

    # Supp: Fig 13 stability cross-validation (requires --stability + --protvar).
    if render_supplements and col_flags.get('has_stability_data', False):
        print("Fig 13 (supp) - Stability Predictor Cross-Validation (reviewed only)")
        plot_fig13_stability_crossvalidation(df_human)
        figures_generated += 1

    # Supp: Figs 14A + 14B disease annotation burden (requires --disease).
    if render_supplements and col_flags.get('has_disease_data', False):
        print("Fig 14A + 14B (supp) - Disease Annotation Burden (reviewed only)")
        plot_fig14_disease_enrichment(df_human)
        figures_generated += 2

    # Supp: Fig 15 pathway bar chart + network (require --pathways).
    if render_supplements and col_flags.get('has_pathway_data', False):
        print("Fig 15 (supp) - Top Reactome Pathways (bar chart) [reviewed-human]")
        plot_fig15_pathway_bar_chart(df_human)
        figures_generated += 1
        print("Fig 15 (supp) - Reactome Pathway Network [reviewed-human]")
        plot_fig15_pathway_network(df_human)
        figures_generated += 1

    # Main: Fig 16 prediction quality paradox (requires --variants + --pathways).
    if col_flags.get('has_paradox_data', False):
        print("Fig 16 - Prediction Quality Paradox (reviewed only)")
        plot_fig16_prediction_quality_paradox(df_human)
        figures_generated += 1

    # Main: Fig 17 screening landscape (always render when CSV supports it).
    if args.screening_figures and col_flags.get('has_composite_screening', False):
        print("\nFig 17 - Composite Screening Landscape")
        plot_fig17_screening_landscape(df)
        figures_generated += 1
    elif args.screening_figures:
        print("\n  Fig 17 skipped: composite_screen_status column missing.")

    # Supp: Fig 18 partial-reason dashboard.
    if render_partial_diagnostics:
        print("\nFig 18 (supp) - Input Recoverability Diagnostic")
        plot_fig18_partial_reason_dashboard(df)
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
