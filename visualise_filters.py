"""Canonical row filters for visualise_results.py.

Populations validated against results_516744.csv on 2026-05-11; see
Output/final_csv_schema_audit/audit_summary.json.

Ownership: this module owns parse_boolish and the numeric-presence helpers.
visualise_results.py imports from here. Never import visualise_results from here.

Canonical filters (14 total):

    Filter                          Audit N (516,744 corpus)
    ---------------------------     ------------------------
    all_rows                        516,744 (100.00%)
    recoverable                     406,244 ( 78.62%)
    calibrated_dimer                402,846 ( 77.96%)
    composite_status_present        516,744 (100.00%)
    composite_screenable            402,846 ( 77.96%)
    strong_screen_candidate           8,635 (  1.67%)
    moderate_screen_candidate        21,998 (  4.26%)
    weak_screen_candidate           372,213 ( 72.03%)
    human_broad                     474,444 ( 91.81%)
    human_strict                    465,655 ( 90.11%)
    multimer_exploratory              5,644 (  1.09%)
    partial_error                   110,500 ( 21.38%)
    calibrated_human_broad          364,357 ( 70.51%)
    calibrated_human_strict         357,073 ( 69.10%)

`mask_degenerate_empty_rows` is a diagnostic helper only and is not part of the
FILTER_REGISTRY because subtracting it from `recoverable` would invalidate the
audit row counts above.
"""
import re
from typing import Iterable

import pandas as pd

# ---------------------------------------------------------------------------
# Token sets
# ---------------------------------------------------------------------------

EMPTY_TOKENS = {"", "nan", "none", "null", "na", "n/a"}
_TRUE_STRS = {"true", "1", "1.0", "yes", "y", "t"}
_FALSE_STRS = {"false", "0", "0.0", "no", "n", "f"}

HUMAN_BROAD_STATUSES = ("reviewed_human", "trembl_human")
HUMAN_STRICT_STATUS = "reviewed_human"
TIER_SCOPE_DIMER = "dimer_validated"
TIER_SCOPE_MULTIMER = "multimer_provisional"

SCREEN_STATUS_STRONG = "strong_screen_candidate"
SCREEN_STATUS_MODERATE = "moderate_screen_candidate"
SCREEN_STATUS_WEAK = "weak_screen_candidate"
SCREEN_STATUS_UNAVAILABLE = "unavailable"

CALIBRATED_DIMER_NUMERIC_REQUIREMENTS = (
    "iptm",
    "pdockq",
    "interface_confidence_score",
    "strict_confident_contact_fraction",
    "interface_pae_mean",
    "n_interface_contacts",
)


# ---------------------------------------------------------------------------
# Primitive helpers
# ---------------------------------------------------------------------------

def parse_boolish(value):
    """Map a single CSV-stringified boolean to a real bool.

    Empty / "nan" / "none" / "null" / "na" / "n/a" -> None
    "true" / "1" / "1.0" / "yes" / "y" / "t"        -> True
    "false" / "0" / "0.0" / "no" / "n" / "f"        -> False
    Anything else                                   -> None
    """
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in EMPTY_TOKENS:
        return None
    if s in _TRUE_STRS:
        return True
    if s in _FALSE_STRS:
        return False
    return None


def _numeric_present(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a boolean Series: True where the coerced column is not NaN."""
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").notna()


def _numeric_gt(df: pd.DataFrame, col: str, value) -> pd.Series:
    """Return a boolean Series: True where the coerced column > value."""
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").gt(value)


def split_interface_flags(value) -> set:
    """Split an `interface_flags` cell on both comma and pipe and return a set
    of stripped, non-empty tokens. Exact-token matching avoids the substring
    false positives a `str.contains` approach would introduce.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return set()
    return {
        token.strip()
        for token in re.split(r"[|,]", str(value))
        if token.strip()
    }


# ---------------------------------------------------------------------------
# Canonical filters
# ---------------------------------------------------------------------------

def mask_all_rows(df: pd.DataFrame) -> pd.Series:
    """Every row."""
    return pd.Series(True, index=df.index)


def mask_recoverable(df: pd.DataFrame) -> pd.Series:
    """Rows whose `partial_reason` is empty/NaN — the structurally usable subset.

    The audit defines this as `partial_reason` NaN OR a lowercase-stripped
    value in EMPTY_TOKENS. Mirroring the audit ensures the count of 406,244
    is reproduced.
    """
    if "partial_reason" not in df.columns:
        return pd.Series(True, index=df.index)
    col = df["partial_reason"]
    is_na = col.isna()
    is_blank = col.astype(str).str.strip().str.lower().isin(EMPTY_TOKENS)
    return is_na | is_blank


def mask_calibrated_dimer(df: pd.DataFrame) -> pd.Series:
    """Dissertation-safe calibrated dimer rows.

    Six clauses (all required):
      1. tier_scope == 'dimer_validated'
      2. parse_boolish(composite_is_calibrated) == True
      3. mask_recoverable
      4. all six numeric metrics present
      5. n_interface_contacts > 0
    """
    if "tier_scope" not in df.columns or "composite_is_calibrated" not in df.columns:
        return pd.Series(False, index=df.index)

    scope = df["tier_scope"].astype(str) == TIER_SCOPE_DIMER
    calibrated = df["composite_is_calibrated"].map(lambda v: parse_boolish(v) is True)
    recoverable = mask_recoverable(df)

    numerics_present = pd.Series(True, index=df.index)
    for col in CALIBRATED_DIMER_NUMERIC_REQUIREMENTS:
        numerics_present &= _numeric_present(df, col)

    contacts_positive = _numeric_gt(df, "n_interface_contacts", 0)
    return scope & calibrated & recoverable & numerics_present & contacts_positive


def mask_composite_status_present(df: pd.DataFrame) -> pd.Series:
    """Rows whose `composite_screen_status` is non-empty and not NaN."""
    if "composite_screen_status" not in df.columns:
        return pd.Series(False, index=df.index)
    col = df["composite_screen_status"]
    is_na = col.isna()
    is_blank = col.astype(str).str.strip().str.lower().isin(EMPTY_TOKENS)
    return ~(is_na | is_blank)


def mask_composite_screenable(df: pd.DataFrame) -> pd.Series:
    """Status present AND != 'unavailable' AND numeric `interface_confidence_score`."""
    present = mask_composite_status_present(df)
    if "composite_screen_status" not in df.columns:
        return pd.Series(False, index=df.index)
    not_unavailable = df["composite_screen_status"].astype(str) != SCREEN_STATUS_UNAVAILABLE
    has_ics = _numeric_present(df, "interface_confidence_score")
    return present & not_unavailable & has_ics


def mask_strong_screen_candidate(df: pd.DataFrame) -> pd.Series:
    if "composite_screen_status" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["composite_screen_status"].astype(str) == SCREEN_STATUS_STRONG


def mask_moderate_screen_candidate(df: pd.DataFrame) -> pd.Series:
    if "composite_screen_status" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["composite_screen_status"].astype(str) == SCREEN_STATUS_MODERATE


def mask_weak_screen_candidate(df: pd.DataFrame) -> pd.Series:
    if "composite_screen_status" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["composite_screen_status"].astype(str) == SCREEN_STATUS_WEAK


def mask_human_broad(df: pd.DataFrame) -> pd.Series:
    """species_status in {reviewed_human, trembl_human}."""
    if "species_status" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["species_status"].astype(str).isin(HUMAN_BROAD_STATUSES)


def mask_human_strict(df: pd.DataFrame) -> pd.Series:
    """species_status == 'reviewed_human'."""
    if "species_status" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["species_status"].astype(str) == HUMAN_STRICT_STATUS


def mask_multimer_exploratory(df: pd.DataFrame) -> pd.Series:
    """tier_scope == 'multimer_provisional' OR n_chains > 2."""
    has_scope = "tier_scope" in df.columns
    scope_match = (
        df["tier_scope"].astype(str) == TIER_SCOPE_MULTIMER
        if has_scope
        else pd.Series(False, index=df.index)
    )
    chains_gt2 = _numeric_gt(df, "n_chains", 2)
    return scope_match | chains_gt2


def mask_partial_error(df: pd.DataFrame) -> pd.Series:
    """Complement of `recoverable`."""
    return ~mask_recoverable(df)


def mask_calibrated_human_broad(df: pd.DataFrame) -> pd.Series:
    return mask_calibrated_dimer(df) & mask_human_broad(df)


def mask_calibrated_human_strict(df: pd.DataFrame) -> pd.Series:
    return mask_calibrated_dimer(df) & mask_human_strict(df)


def mask_degenerate_empty_rows(df: pd.DataFrame) -> pd.Series:
    """Diagnostic mask flagging rows that are blank/NaN across the key identity
    columns. The audit observed 7 such rows. Used for reporting only;
    `recoverable` retains its audit-exact definition, so these rows are caught
    by `require_columns()` at figure-time.
    """
    key_cols = [
        "complex_name",
        "quality_tier_v2",
        "tier_scope",
        "composite_is_calibrated",
    ]
    present = [c for c in key_cols if c in df.columns]
    if not present:
        return pd.Series(False, index=df.index)
    str_view = df[present].astype(str).apply(lambda col: col.str.strip().str.lower())
    all_empty = str_view.isin(EMPTY_TOKENS).all(axis=1)
    all_na = df[present].isna().all(axis=1)
    return all_na | all_empty


# ---------------------------------------------------------------------------
# Registry and apply_filter
# ---------------------------------------------------------------------------

FILTER_REGISTRY = {
    "all_rows": mask_all_rows,
    "recoverable": mask_recoverable,
    "calibrated_dimer": mask_calibrated_dimer,
    "composite_status_present": mask_composite_status_present,
    "composite_screenable": mask_composite_screenable,
    "strong_screen_candidate": mask_strong_screen_candidate,
    "moderate_screen_candidate": mask_moderate_screen_candidate,
    "weak_screen_candidate": mask_weak_screen_candidate,
    "human_broad": mask_human_broad,
    "human_strict": mask_human_strict,
    "multimer_exploratory": mask_multimer_exploratory,
    "partial_error": mask_partial_error,
    "calibrated_human_broad": mask_calibrated_human_broad,
    "calibrated_human_strict": mask_calibrated_human_strict,
}


def apply_filter(df: pd.DataFrame, name: str, fig_label: str = ""):
    """Return (filtered_df, n_before, n_after) and log the transition to stdout.

    Raises KeyError if `name` is not in FILTER_REGISTRY.
    """
    if name not in FILTER_REGISTRY:
        raise KeyError(f"Unknown filter '{name}'. Available: {sorted(FILTER_REGISTRY)}")
    mask = FILTER_REGISTRY[name](df)
    n_before = len(df)
    n_after = int(mask.sum())
    prefix = f"  {fig_label} | " if fig_label else "  "
    print(f"{prefix}filter={name}: {n_before:,} -> {n_after:,} rows")
    return df.loc[mask].reset_index(drop=True), n_before, n_after


def require_columns(df: pd.DataFrame, cols: Iterable[str], fig_label: str = ""):
    """Drop rows where any column in `cols` is NaN. Skip cleanly when any
    required column is missing entirely — visualisation contract is graceful
    degradation, not silent pass-through.

    Returns (filtered_df, n_before, n_after) and logs the transition on its own
    line so per-figure column shrinkage is visible separately from the named
    filter's audit-aligned count.
    """
    cols = list(cols)
    missing = [c for c in cols if c not in df.columns]
    present = [c for c in cols if c in df.columns]
    prefix = f"  {fig_label} | " if fig_label else "  "
    n_before = len(df)
    if missing:
        print(f"{prefix}missing required columns: {', '.join(missing)}")
        return df.iloc[0:0].copy(), n_before, 0
    if not present:
        return df.reset_index(drop=True), n_before, n_before
    sub = df.dropna(subset=present)
    n_after = len(sub)
    print(f"{prefix}required columns ({','.join(present)}): {n_before:,} -> {n_after:,} rows")
    return sub.reset_index(drop=True), n_before, n_after
