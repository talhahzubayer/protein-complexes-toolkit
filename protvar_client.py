#!/usr/bin/env python3
"""
ProtVar API client for the protein-complexes-toolkit.

Cross-validates the toolkit's independent interface predictions against ProtVar
(EBI), which provides pre-computed pathogenicity scores (AlphaMissense, EVE,
ESM, conservation), interaction predictions, and FoldX ΔΔG values.

Architecture:
    - API-backed with aggressive caching: every response is stored as a JSON
      file so subsequent runs are instant.
    - Lazy querying: only positions from variants in the current pipeline run
      are fetched.
    - Rate-limited with exponential backoff on transient failures.

Endpoints used (3 per position):
    GET /score/{acc}/{pos}                  — multi-tool pathogenicity scores
    GET /prediction/interaction/{acc}/{pos}  — ProtVar interface predictions
    GET /prediction/foldx/{acc}/{pos}        — pre-computed FoldX ΔΔG

Usage (as importable module):
    from protvar_client import build_protvar_index, annotate_results_with_protvar

    acc_positions = {'P61981': {4, 10}, 'P24534': {81}}
    index = build_protvar_index(acc_positions, verbose=True)
    annotate_results_with_protvar(results, index, verbose=True)

Usage (standalone):
    python protvar_client.py summary
    python protvar_client.py lookup --protein P61981 --position 4
"""

import sys
import json
import time
import hashlib
import re
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import urllib.request
import urllib.error


# ── Constants ────────────────────────────────────────────────────────

# ProtVar API connection
PROTVAR_API_BASE_URL = "https://www.ebi.ac.uk/ProtVar/api"

# Rate limiting and retry
PROTVAR_API_RATE_LIMIT_PAUSE = 1.0   # Minimum seconds between consecutive requests
PROTVAR_API_MAX_RETRIES = 3          # Max retry attempts on transient failure
PROTVAR_API_BACKOFF_FACTOR = 2.0     # Exponential backoff multiplier
PROTVAR_API_TIMEOUT = 30             # Seconds per HTTP request

# Default cache directory (auto-caching enabled by default)
PROTVAR_API_DEFAULT_CACHE_DIR = Path(__file__).parent / "data" / "protvar_cache"

# Display limit for variant detail strings
PROTVAR_DETAILS_DISPLAY_LIMIT = 20

# FoldX destabilisation threshold (kcal/mol)
FOLDX_DESTABILISING_THRESHOLD = 1.6

# Score tool names from ProtVar /score endpoint
PROTVAR_TOOL_AM = "AM"           # AlphaMissense
PROTVAR_TOOL_EVE = "EVE"         # EVE evolutionary predictions
PROTVAR_TOOL_ESM = "ESM"         # ESM-1v language model
PROTVAR_TOOL_CONSERV = "CONSERV"  # Conservation score

# Variant detail parsing pattern (shared with stability_scorer.py)
_VARIANT_DETAIL_PATTERN = re.compile(r'^([A-Z*])(\d+)([A-Z*]):')

# Module-level rate limiting state
_last_request_time: float = 0.0

# CSV column names added by this module (8 columns, per-chain a/b)
CSV_FIELDNAMES_PROTVAR = [
    'protvar_am_mean_a', 'protvar_am_mean_b',
    'protvar_foldx_mean_a', 'protvar_foldx_mean_b',
    'protvar_interface_agreement_a', 'protvar_interface_agreement_b',
    'protvar_details_a', 'protvar_details_b',
]


# ── Section 1: Custom Exception ──────────────────────────────────────

class ProtVarAPIError(RuntimeError):
    """Raised when a ProtVar API request fails after all retries are exhausted,
    or when a non-retryable HTTP error is received.

    Callers should catch this and fall back gracefully:

        try:
            scores = get_scores("P61981", 4)
        except ProtVarAPIError as e:
            warnings.warn(f"ProtVar API unavailable: {e}")
            scores = []
    """


# ── Section 2: Internal HTTP Helpers ─────────────────────────────────

def _resolve_cache_dir(cache_dir: Optional[Union[str, bool]]) -> Optional[Path]:
    """Resolve cache_dir parameter to a Path or None.

    Args:
        cache_dir: Cache directory specification.
            None  -> use PROTVAR_API_DEFAULT_CACHE_DIR (auto-caching).
            str/Path -> use that path.
            False -> disable caching entirely.

    Returns:
        Path to cache directory, or None if caching is disabled.
    """
    if cache_dir is False:
        return None
    if cache_dir is None:
        return PROTVAR_API_DEFAULT_CACHE_DIR
    return Path(cache_dir)


def _cache_key(endpoint: str, accession: str, position: int) -> str:
    """Generate a deterministic cache key from endpoint and parameters.

    Args:
        endpoint: API endpoint path (e.g. 'score', 'prediction/interaction').
        accession: UniProt accession.
        position: Residue position.

    Returns:
        Hex SHA256 digest string.
    """
    key_data = {
        "endpoint": endpoint,
        "accession": accession,
        "position": position,
    }
    return hashlib.sha256(
        json.dumps(key_data, sort_keys=True).encode()
    ).hexdigest()


def _read_cache(cache_dir: Path, key: str) -> Optional[Union[dict, list]]:
    """Read a cached API response if it exists.

    Args:
        cache_dir: Directory containing cache files.
        key: Cache key (SHA256 hex digest).

    Returns:
        Parsed JSON data, or None if cache miss.
    """
    cache_file = cache_dir / f"{key}.json"
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, encoding="utf-8") as f:
            cached = json.load(f)
        return cached.get("data")
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def _write_cache(cache_dir: Path, key: str, endpoint: str,
                 data: Union[dict, list]) -> None:
    """Write an API response to the cache.

    Args:
        cache_dir: Directory for cache files (created if it does not exist).
        key: Cache key (SHA256 hex digest).
        endpoint: API endpoint (stored as metadata).
        data: Parsed JSON response to cache.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{key}.json"
    payload = {
        "_timestamp": datetime.now(timezone.utc).isoformat(),
        "_endpoint": endpoint,
        "data": data,
    }
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _rate_limit() -> None:
    """Enforce minimum pause between consecutive API requests."""
    global _last_request_time
    now = time.monotonic()
    elapsed = now - _last_request_time
    if elapsed < PROTVAR_API_RATE_LIMIT_PAUSE:
        time.sleep(PROTVAR_API_RATE_LIMIT_PAUSE - elapsed)


def _make_request(url: str) -> Union[dict, list]:
    """Execute a ProtVar API GET request with rate limiting, retry, and backoff.

    Unlike STRING API (POST with form data), ProtVar uses GET with path
    parameters. This function handles rate limiting, retry on transient
    failures, and error propagation.

    Args:
        url: Full API URL (e.g. 'https://www.ebi.ac.uk/ProtVar/api/score/P61981/4').

    Returns:
        Parsed JSON response (dict or list).

    Raises:
        ProtVarAPIError: On non-retryable HTTP errors or exhausted retries.
    """
    global _last_request_time

    last_error: Optional[Exception] = None

    for attempt in range(PROTVAR_API_MAX_RETRIES + 1):
        # Rate limiting
        _rate_limit()

        try:
            req = urllib.request.Request(url)
            req.add_header("Accept", "application/json")
            with urllib.request.urlopen(req, timeout=PROTVAR_API_TIMEOUT) as response:
                _last_request_time = time.monotonic()
                body = response.read().decode("utf-8")
                return json.loads(body)

        except urllib.error.HTTPError as e:
            _last_request_time = time.monotonic()
            status = e.code
            last_error = e

            # Retryable errors: 429 (rate limited) and 5xx (server error)
            if status == 429 or 500 <= status < 600:
                if attempt < PROTVAR_API_MAX_RETRIES:
                    wait = PROTVAR_API_BACKOFF_FACTOR ** attempt
                    print(
                        f"  ProtVar API: HTTP {status} on {url}, "
                        f"retrying in {wait:.1f}s "
                        f"(attempt {attempt + 1}/{PROTVAR_API_MAX_RETRIES})",
                        file=sys.stderr,
                    )
                    time.sleep(wait)
                    continue
                else:
                    raise ProtVarAPIError(
                        f"ProtVar API: HTTP {status} after "
                        f"{PROTVAR_API_MAX_RETRIES} retries: {url}"
                    ) from e

            # Non-retryable error (4xx except 429)
            try:
                error_body = e.read().decode("utf-8")
            except Exception:
                error_body = ""
            raise ProtVarAPIError(
                f"ProtVar API: HTTP {status}: {error_body} ({url})"
            ) from e

        except urllib.error.URLError as e:
            _last_request_time = time.monotonic()
            last_error = e
            if attempt < PROTVAR_API_MAX_RETRIES:
                wait = PROTVAR_API_BACKOFF_FACTOR ** attempt
                print(
                    f"  ProtVar API: connection error on {url}, "
                    f"retrying in {wait:.1f}s "
                    f"(attempt {attempt + 1}/{PROTVAR_API_MAX_RETRIES})",
                    file=sys.stderr,
                )
                time.sleep(wait)
                continue
            raise ProtVarAPIError(
                f"ProtVar API: connection failed after "
                f"{PROTVAR_API_MAX_RETRIES} retries: {e}"
            ) from e

    # Safety net (should not reach here)
    raise ProtVarAPIError(
        f"ProtVar API: unexpected failure after retries"
    ) from last_error


def _query_endpoint(endpoint: str, accession: str, position: int,
                    cache_dir: Optional[Union[str, bool]] = None) -> Union[dict, list]:
    """Internal helper: query a ProtVar endpoint with caching.

    Args:
        endpoint: API path segment (e.g. 'score', 'prediction/interaction',
                  'prediction/foldx').
        accession: UniProt accession.
        position: Residue position (1-based).
        cache_dir: Cache directory specification (see _resolve_cache_dir).

    Returns:
        Parsed JSON response.
    """
    resolved = _resolve_cache_dir(cache_dir)

    # Check cache first
    if resolved is not None:
        key = _cache_key(endpoint, accession, position)
        cached = _read_cache(resolved, key)
        if cached is not None:
            return cached

    # Make live request
    url = f"{PROTVAR_API_BASE_URL}/{endpoint}/{accession}/{position}"
    data = _make_request(url)

    # Write to cache
    if resolved is not None:
        key = _cache_key(endpoint, accession, position)
        _write_cache(resolved, key, endpoint, data)

    return data


# ── Section 3: Public API Query Functions ────────────────────────────

def get_scores(accession: str, position: int,
               cache_dir: Optional[Union[str, bool]] = None) -> list:
    """Query ProtVar /score/{acc}/{pos} for pathogenicity scores.

    Returns scores from multiple tools (AlphaMissense, EVE, ESM, conservation)
    for all possible amino acid substitutions at the given position.

    Args:
        accession: UniProt accession (canonical, no isoform suffix).
        position: Residue position (1-based).
        cache_dir: Cache specification (None=default, False=disabled, str=custom).

    Returns:
        List of score dicts. Each has 'name' (tool), 'mt' (mutant AA),
        and tool-specific keys like 'amPathogenicity', 'amClass', 'score',
        'eveClass'.
    """
    result = _query_endpoint("score", accession, position, cache_dir)
    return result if isinstance(result, list) else []


def get_interactions(accession: str, position: int,
                     cache_dir: Optional[Union[str, bool]] = None) -> list:
    """Query ProtVar /prediction/interaction/{acc}/{pos} for interaction data.

    Returns predicted interaction partners with binding residue lists and
    pDockQ scores for the given residue position.

    Args:
        accession: UniProt accession (canonical, no isoform suffix).
        position: Residue position (1-based).
        cache_dir: Cache specification (None=default, False=disabled, str=custom).

    Returns:
        List of interaction dicts. Each has 'a' (accession A), 'b' (accession B),
        'aresidues' (list[int]), 'bresidues' (list[int]), 'pdockq' (float),
        'pdbModel' (str). Empty list if no interactions.
    """
    result = _query_endpoint("prediction/interaction", accession, position, cache_dir)
    return result if isinstance(result, list) else []


def get_foldx(accession: str, position: int,
              cache_dir: Optional[Union[str, bool]] = None) -> list:
    """Query ProtVar /prediction/foldx/{acc}/{pos} for FoldX ΔΔG predictions.

    Returns pre-computed FoldX ΔΔG values for all possible amino acid
    substitutions at the given position. Values >1.6 kcal/mol are
    conventionally considered destabilising.

    Args:
        accession: UniProt accession (canonical, no isoform suffix).
        position: Residue position (1-based).
        cache_dir: Cache specification (None=default, False=disabled, str=custom).

    Returns:
        List of FoldX dicts. Each has 'proteinAcc', 'position', 'wildType',
        'mutatedType', 'foldxDdg' (float), 'plddt' (float). Empty list if
        no data.
    """
    result = _query_endpoint("prediction/foldx", accession, position, cache_dir)
    return result if isinstance(result, list) else []


# ── Section 4: Index Building ────────────────────────────────────────

def build_protvar_index(
    acc_positions: dict[str, set[int]],
    cache_dir: Optional[Union[str, bool]] = None,
    verbose: bool = False,
) -> dict[str, dict[int, dict]]:
    """Query ProtVar for all (accession, position) pairs and build a lookup index.

    Each position is queried once for all 3 endpoints (scores, interactions,
    foldx). Results are cached so subsequent runs are instant.

    Args:
        acc_positions: Dict mapping UniProt accession to set of positions.
            Example: {'P61981': {4, 10}, 'P24534': {81}}.
        cache_dir: Cache specification (None=default, False=disabled, str=custom).
        verbose: Print progress to stderr.

    Returns:
        Nested dict: {accession: {position: {'scores': list, 'interactions': list,
        'foldx': list}}}.
    """
    index: dict[str, dict[int, dict]] = {}

    # Count total positions for progress
    total = sum(len(positions) for positions in acc_positions.values())
    done = 0
    errors = 0

    if verbose and total > 0:
        print(f"ProtVar: querying {total} positions across "
              f"{len(acc_positions)} proteins...", file=sys.stderr)

    for accession in sorted(acc_positions):
        positions = sorted(acc_positions[accession])
        if accession not in index:
            index[accession] = {}

        for pos in positions:
            done += 1
            try:
                scores = get_scores(accession, pos, cache_dir=cache_dir)
                interactions = get_interactions(accession, pos, cache_dir=cache_dir)
                foldx = get_foldx(accession, pos, cache_dir=cache_dir)
                index[accession][pos] = {
                    'scores': scores,
                    'interactions': interactions,
                    'foldx': foldx,
                }
            except ProtVarAPIError as e:
                errors += 1
                if verbose:
                    print(f"  ProtVar: error for {accession}/{pos}: {e}",
                          file=sys.stderr)
                index[accession][pos] = {
                    'scores': [],
                    'interactions': [],
                    'foldx': [],
                }

            if verbose and done % 10 == 0:
                print(f"  ProtVar: {done}/{total} positions queried...",
                      file=sys.stderr)

    if verbose:
        n_with_scores = sum(
            1 for acc in index for pos in index[acc]
            if index[acc][pos]['scores']
        )
        print(f"  ProtVar index: {total} positions, {n_with_scores} with scores, "
              f"{errors} errors", file=sys.stderr)

    return index


# ── Section 5: Score Extraction Helpers ──────────────────────────────

def extract_am_score(scores: list[dict], alt_aa: str) -> Optional[float]:
    """Extract AlphaMissense pathogenicity score for a specific substitution.

    Args:
        scores: List of score dicts from get_scores().
        alt_aa: Single-letter mutant amino acid.

    Returns:
        AlphaMissense pathogenicity float, or None if not found.
    """
    for entry in scores:
        if entry.get('name') == PROTVAR_TOOL_AM and entry.get('mt') == alt_aa:
            return entry.get('amPathogenicity')
    return None


def extract_am_class(scores: list[dict], alt_aa: str) -> Optional[str]:
    """Extract AlphaMissense classification for a specific substitution.

    Args:
        scores: List of score dicts from get_scores().
        alt_aa: Single-letter mutant amino acid.

    Returns:
        Classification string ('PATHOGENIC', 'BENIGN', 'AMBIGUOUS'), or None.
    """
    for entry in scores:
        if entry.get('name') == PROTVAR_TOOL_AM and entry.get('mt') == alt_aa:
            return entry.get('amClass')
    return None


def extract_foldx_ddg(foldx_data: list[dict], alt_aa: str) -> Optional[float]:
    """Extract FoldX ΔΔG for a specific substitution.

    Args:
        foldx_data: List of FoldX dicts from get_foldx().
        alt_aa: Single-letter mutant amino acid.

    Returns:
        ΔΔG in kcal/mol, or None if not found.
    """
    for entry in foldx_data:
        if entry.get('mutatedType') == alt_aa:
            return entry.get('foldxDdg')
    return None


def check_protvar_interface(interactions: list[dict], accession: str,
                            position: int) -> bool:
    """Check if ProtVar considers a position to be at a binding interface.

    A position is at the interface if it appears in any interaction's
    'aresidues' or 'bresidues' list for the given accession.

    Args:
        interactions: List of interaction dicts from get_interactions().
        accession: UniProt accession to check.
        position: Residue position.

    Returns:
        True if position is at an interface, False otherwise.
    """
    if not interactions:
        return False

    base_acc = accession.split('-')[0] if '-' in accession else accession

    for entry in interactions:
        a_acc = str(entry.get('a', ''))
        b_acc = str(entry.get('b', ''))
        a_base = a_acc.split('-')[0] if '-' in a_acc else a_acc
        b_base = b_acc.split('-')[0] if '-' in b_acc else b_acc

        if a_base == base_acc and position in entry.get('aresidues', []):
            return True
        if b_base == base_acc and position in entry.get('bresidues', []):
            return True
    return False


def compute_interface_agreement(
    toolkit_interface_positions: set[int],
    protvar_index_for_protein: dict[int, dict],
    accession: str,
) -> Union[float, str]:
    """Compute fraction of toolkit interface positions confirmed by ProtVar.

    For each position the toolkit classified as interface (interface_core or
    interface_rim), checks whether ProtVar also considers it a binding
    interface position.

    Args:
        toolkit_interface_positions: Set of positions the toolkit identified
            as interface residues.
        protvar_index_for_protein: Position-level index for one protein.
        accession: UniProt accession.

    Returns:
        Float in [0, 1] if there are interface positions to compare,
        empty string if no data.
    """
    if not toolkit_interface_positions:
        return ''

    agree = 0
    checked = 0
    for pos in toolkit_interface_positions:
        if pos in protvar_index_for_protein:
            checked += 1
            interactions = protvar_index_for_protein[pos].get('interactions', [])
            if check_protvar_interface(interactions, accession, pos):
                agree += 1

    if checked == 0:
        return ''
    return round(agree / checked, 4)


# ── Section 6: Annotation ────────────────────────────────────────────

def _parse_variant_details_for_protvar(details_str: str) -> list[tuple[str, int, str]]:
    """Parse variant_details string to (ref_aa, position, alt_aa) tuples.

    Parses the pipe-separated variant detail strings produced by
    variant_mapper.format_variant_details(). Format:
        K81P:interface_core:pathogenic|E82K:interface_rim:VUS

    Args:
        details_str: Pipe-separated variant detail string.

    Returns:
        List of (ref_aa, position, alt_aa) tuples.
    """
    if not details_str:
        return []

    variants = []
    for part in details_str.split('|'):
        part = part.strip()
        if part.startswith('...(+'):
            continue
        match = _VARIANT_DETAIL_PATTERN.match(part)
        if match:
            ref = match.group(1)
            pos = int(match.group(2))
            alt = match.group(3)
            variants.append((ref, pos, alt))
    return variants


def _extract_context_from_detail(detail_part: str) -> str:
    """Extract structural context from a variant detail part.

    Args:
        detail_part: Single variant string, e.g. 'K81P:interface_core:pathogenic'.

    Returns:
        Context string (e.g. 'interface_core') or empty string.
    """
    parts = detail_part.split(':')
    if len(parts) >= 2:
        return parts[1]
    return ''


def format_protvar_details(
    scored_variants: list[dict],
    limit: int = PROTVAR_DETAILS_DISPLAY_LIMIT,
) -> str:
    """Format ProtVar-scored variants into a pipe-separated summary string.

    Format: REF{POS}ALT:am={score}:{class}:foldx={ddg}

    Args:
        scored_variants: List of dicts with keys 'ref_aa', 'position', 'alt_aa',
            'am_score', 'am_class', 'foldx_ddg'.
        limit: Maximum number of variants to display.

    Returns:
        Pipe-separated string, e.g. 'R4A:am=0.91:PATHOGENIC:foldx=3.18'.
        Empty string if no variants.
    """
    if not scored_variants:
        return ''

    details = []
    for var in scored_variants[:limit]:
        ref = var.get('ref_aa', '?')
        pos = var.get('position', '?')
        alt = var.get('alt_aa', '?')

        am = var.get('am_score')
        am_str = f"{am:.2f}" if am is not None else '-'

        am_cls = var.get('am_class', '-')
        if not am_cls:
            am_cls = '-'

        ddg = var.get('foldx_ddg')
        ddg_str = f"{ddg:.2f}" if ddg is not None else '-'

        details.append(f"{ref}{pos}{alt}:am={am_str}:{am_cls}:foldx={ddg_str}")

    result = '|'.join(details)

    remaining = len(scored_variants) - limit
    if remaining > 0:
        result += f"|...(+{remaining} more)"

    return result


def _score_chain_variants_protvar(
    accession: str,
    details_str: str,
    protvar_index: dict[str, dict[int, dict]],
) -> dict:
    """Score variants for one chain using the ProtVar index.

    Args:
        accession: UniProt accession for this chain.
        details_str: Pipe-separated variant detail string from Phase C.
        protvar_index: Full ProtVar index from build_protvar_index().

    Returns:
        Dict with keys:
            'am_mean': float or '' (mean AlphaMissense score),
            'foldx_mean': float or '' (mean FoldX ΔΔG),
            'n_interactions': int (number of ProtVar interaction partners),
            'details': str (formatted detail string),
            'n_scored': int (number of variants with AM scores),
            'interface_positions': set[int] (positions classified as interface by toolkit).
    """
    base_acc = accession.split('-')[0] if '-' in accession else accession
    protein_data = protvar_index.get(base_acc, {})
    variants = _parse_variant_details_for_protvar(details_str)

    am_scores = []
    foldx_ddgs = []
    scored_list = []
    interaction_partners: set[str] = set()
    interface_positions: set[int] = set()

    # Extract interface positions from variant detail contexts
    if details_str:
        for part in details_str.split('|'):
            part = part.strip()
            if part.startswith('...(+'):
                continue
            ctx = _extract_context_from_detail(part)
            if ctx.startswith('interface'):
                match = _VARIANT_DETAIL_PATTERN.match(part)
                if match:
                    interface_positions.add(int(match.group(2)))

    for ref_aa, pos, alt_aa in variants:
        pos_data = protein_data.get(pos, {})
        scores = pos_data.get('scores', [])
        foldx_data = pos_data.get('foldx', [])
        interactions = pos_data.get('interactions', [])

        am = extract_am_score(scores, alt_aa)
        am_cls = extract_am_class(scores, alt_aa)
        ddg = extract_foldx_ddg(foldx_data, alt_aa)

        if am is not None:
            am_scores.append(am)
        if ddg is not None:
            foldx_ddgs.append(ddg)

        # Collect interaction partners
        for entry in interactions:
            a_acc = str(entry.get('a', ''))
            b_acc = str(entry.get('b', ''))
            a_base = a_acc.split('-')[0] if '-' in a_acc else a_acc
            b_base = b_acc.split('-')[0] if '-' in b_acc else b_acc
            if a_base == base_acc:
                interaction_partners.add(b_acc)
            elif b_base == base_acc:
                interaction_partners.add(a_acc)

        scored_list.append({
            'ref_aa': ref_aa,
            'position': pos,
            'alt_aa': alt_aa,
            'am_score': am,
            'am_class': am_cls,
            'foldx_ddg': ddg,
        })

    return {
        'am_mean': round(sum(am_scores) / len(am_scores), 4) if am_scores else '',
        'foldx_mean': round(sum(foldx_ddgs) / len(foldx_ddgs), 4) if foldx_ddgs else '',
        'n_interactions': len(interaction_partners),
        'details': format_protvar_details(scored_list),
        'n_scored': len(am_scores),
        'interface_positions': interface_positions,
    }


def annotate_results_with_protvar(
    results: list[dict],
    protvar_index: dict[str, dict[int, dict]],
    verbose: bool = False,
) -> None:
    """Annotate result rows with ProtVar data (in-place).

    Main entry point from toolkit.py. For each complex:
    1. Parses variant_details_a/b to extract variants
    2. Looks up ProtVar scores, interactions, and FoldX data
    3. Computes AlphaMissense mean, FoldX mean, interface agreement
    4. Formats detail strings

    Args:
        results: List of per-complex result dicts. Modified in-place.
        protvar_index: Index from build_protvar_index().
        verbose: Print progress to stderr.
    """
    annotated = 0

    for row in results:
        for suffix in ('a', 'b'):
            acc = row.get(f'protein_{suffix}', '')
            details_str = row.get(f'variant_details_{suffix}', '')

            if acc and details_str:
                chain_result = _score_chain_variants_protvar(
                    acc, details_str, protvar_index,
                )
                row[f'protvar_am_mean_{suffix}'] = chain_result['am_mean']
                row[f'protvar_foldx_mean_{suffix}'] = chain_result['foldx_mean']

                # Compute interface agreement for this chain
                base_acc = acc.split('-')[0] if '-' in acc else acc
                protein_data = protvar_index.get(base_acc, {})
                agreement = compute_interface_agreement(
                    chain_result['interface_positions'],
                    protein_data,
                    acc,
                )
                row[f'protvar_interface_agreement_{suffix}'] = agreement
                row[f'protvar_details_{suffix}'] = chain_result['details']

                if chain_result['n_scored'] > 0:
                    annotated += 1
            else:
                row[f'protvar_am_mean_{suffix}'] = ''
                row[f'protvar_foldx_mean_{suffix}'] = ''
                row[f'protvar_interface_agreement_{suffix}'] = ''
                row[f'protvar_details_{suffix}'] = ''

    if verbose:
        print(f"  ProtVar: annotated {annotated} chains with scores "
              f"across {len(results)} complexes", file=sys.stderr)


# ── Section 7: Standalone CLI ────────────────────────────────────────

def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for standalone use.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description="ProtVar API client — cross-validate interface predictions "
                    "with ProtVar pathogenicity scores, interactions, and FoldX ΔΔG.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cache-dir",
        default=str(PROTVAR_API_DEFAULT_CACHE_DIR),
        help=f"Cache directory for API responses (default: {PROTVAR_API_DEFAULT_CACHE_DIR}).",
    )

    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # summary sub-command
    subparsers.add_parser("summary", help="Print cache statistics.")

    # lookup sub-command
    lookup_parser = subparsers.add_parser(
        "lookup",
        help="Look up ProtVar data for a protein position.",
    )
    lookup_parser.add_argument(
        "--protein", required=True,
        help="UniProt accession (e.g. P61981).",
    )
    lookup_parser.add_argument(
        "--position", required=True, type=int,
        help="Residue position (1-based).",
    )

    return parser


def _cli_summary(cache_dir: Path) -> None:
    """Print cache statistics to stdout.

    Args:
        cache_dir: Path to cache directory.
    """
    if not cache_dir.exists():
        print(f"Cache directory does not exist: {cache_dir}")
        print("No cached ProtVar responses.")
        return

    cache_files = list(cache_dir.glob("*.json"))
    print(f"ProtVar API cache: {cache_dir}")
    print(f"  Cached responses: {len(cache_files)}")

    if cache_files:
        # Count unique proteins (approximate from cache metadata)
        endpoints: dict[str, int] = {}
        for f in cache_files:
            try:
                with open(f, encoding="utf-8") as fh:
                    meta = json.load(fh)
                ep = meta.get("_endpoint", "unknown")
                endpoints[ep] = endpoints.get(ep, 0) + 1
            except (json.JSONDecodeError, OSError):
                pass
        for ep, count in sorted(endpoints.items()):
            print(f"    {ep}: {count} cached")


def _cli_lookup(accession: str, position: int,
                cache_dir: Optional[str] = None) -> None:
    """Look up and display ProtVar data for a protein position.

    Args:
        accession: UniProt accession.
        position: Residue position.
        cache_dir: Cache directory path.
    """
    print(f"Looking up ProtVar data for {accession} position {position}...")
    print()

    # Scores
    scores = get_scores(accession, position, cache_dir=cache_dir)
    am_entries = [s for s in scores if s.get('name') == PROTVAR_TOOL_AM]
    eve_entries = [s for s in scores if s.get('name') == PROTVAR_TOOL_EVE]
    conserv_entries = [s for s in scores if s.get('name') == PROTVAR_TOOL_CONSERV]

    print(f"=== Scores ({len(scores)} entries) ===")
    print(f"  AlphaMissense: {len(am_entries)} substitutions")
    for s in sorted(am_entries, key=lambda x: x.get('amPathogenicity', 0), reverse=True)[:5]:
        print(f"    → {s.get('mt', '?')}: {s.get('amPathogenicity', '?')} ({s.get('amClass', '?')})")
    if len(am_entries) > 5:
        print(f"    ... and {len(am_entries) - 5} more")

    print(f"  EVE: {len(eve_entries)} substitutions")
    if conserv_entries:
        print(f"  Conservation: {conserv_entries[0].get('score', '?')}")
    print()

    # Interactions
    interactions = get_interactions(accession, position, cache_dir=cache_dir)
    print(f"=== Interactions ({len(interactions)} partners) ===")
    for entry in interactions[:5]:
        print(f"  {entry.get('a', '?')} ↔ {entry.get('b', '?')} "
              f"(pDockQ: {entry.get('pdockq', '?'):.4f})")
    is_interface = check_protvar_interface(interactions, accession, position)
    print(f"  Position {position} is at interface: {is_interface}")
    print()

    # FoldX
    foldx_data = get_foldx(accession, position, cache_dir=cache_dir)
    print(f"=== FoldX ΔΔG ({len(foldx_data)} substitutions) ===")
    destab = [f for f in foldx_data
              if f.get('foldxDdg', 0) > FOLDX_DESTABILISING_THRESHOLD]
    print(f"  Destabilising (>{FOLDX_DESTABILISING_THRESHOLD} kcal/mol): "
          f"{len(destab)}/{len(foldx_data)}")
    for f in sorted(foldx_data, key=lambda x: x.get('foldxDdg', 0), reverse=True)[:5]:
        wt = f.get('wildType', '?')
        mt = f.get('mutatedType', '?')
        ddg = f.get('foldxDdg', 0)
        flag = " ⚠ DESTABILISING" if ddg > FOLDX_DESTABILISING_THRESHOLD else ""
        print(f"    {wt}{position}{mt}: ΔΔG = {ddg:.3f}{flag}")


def main() -> None:
    """CLI entry point."""
    parser = build_argument_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    cache_dir = Path(args.cache_dir) if args.cache_dir else PROTVAR_API_DEFAULT_CACHE_DIR

    if args.command == "summary":
        _cli_summary(cache_dir)
    elif args.command == "lookup":
        _cli_lookup(args.protein, args.position, cache_dir=str(cache_dir))


if __name__ == "__main__":
    main()
