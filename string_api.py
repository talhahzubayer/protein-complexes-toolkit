#!/usr/bin/env python3
"""
Centralised STRING API client for the protein-complexes-toolkit.

Provides a single module through which all STRING database API interactions are
routed. The architecture is offline-first: local flat files (parsed by
database_loaders.py and id_mapper.py) remain the primary data source. This
module is an optional supplement invoked only when needed — for example, to
resolve identifiers that fail local lookup, to retrieve quantitative homology
scores, or to perform functional enrichment analysis.

Features:
    - Rate-limited requests with configurable pause between calls
    - Automatic retry with exponential backoff on HTTP 429 / 5xx errors
    - Optional response caching (JSON files with SHA256-keyed filenames)
    - Caller identity injection on every request (STRING API TOS requirement)
    - Custom StringAPIError exception for clean error propagation
    - 7 public functions covering ID resolution, interactions, homology,
      enrichment, network retrieval, and version checking

Usage (as importable module):
    from string_api import get_string_ids, get_version, StringAPIError

    try:
        ids_df = get_string_ids(["P04637", "Q9UKT4"], species=9606)
        version = get_version()
    except StringAPIError as e:
        print(f"STRING API unavailable: {e}")
        # fall back to local data

Usage (standalone):
    python string_api.py --resolve P04637,Q9UKT4 --species 9606
    python string_api.py --enrichment P04637,P12345,Q9UKT4 --species 9606
    python string_api.py --network TP53,MDM2,BRCA1 --network-type physical
    python string_api.py --version
"""

import sys
import json
import time
import hashlib
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import urllib.request
import urllib.parse
import urllib.error

import pandas as pd

# ---Constants----------------------------------------------------------

# STRING API connection
STRING_API_BASE_URL = "https://string-db.org/api"              # Central endpoint
STRING_API_CALLER_IDENTITY = "protein-complexes-toolkit-kcl"    # Required by STRING API TOS
STRING_API_OUTPUT_FORMAT = "json"                               # Also supports "tsv", "xml"

# Default query parameters
STRING_API_SPECIES = 9606                                       # Homo sapiens

# Rate limiting and retry
STRING_API_RATE_LIMIT_PAUSE = 1.0   # Minimum seconds between consecutive requests
STRING_API_MAX_RETRIES = 3          # Max retry attempts on transient failure
STRING_API_BACKOFF_FACTOR = 2.0     # Exponential backoff multiplier
STRING_API_TIMEOUT = 30             # Seconds per HTTP request

# Enrichment batch limit (STRING API constraint)
STRING_API_MAX_ENRICHMENT_BATCH = 2000  # Max proteins per enrichment call

# Default cache directory (auto-caching enabled by default)
STRING_API_DEFAULT_CACHE_DIR = Path(__file__).parent / "data" / "string_api_cache"

# Valid network types
_VALID_NETWORK_TYPES = {"functional", "physical"}

# Module-level rate limiting state
_last_request_time: float = 0.0


# ---Custom Exception---------------------------------------------------

class StringAPIError(RuntimeError):
    """Raised when a STRING API request fails after all retries are exhausted,
    or when a non-retryable HTTP error is received.

    Callers should catch this and fall back to local data with a warning:

        try:
            result = get_string_ids(ids)
        except StringAPIError as e:
            warnings.warn(f"STRING API unavailable: {e}")
            result = local_fallback(ids)
    """


# ---Internal Helpers---------------------------------------------------

def _resolve_cache_dir(cache_dir: Optional[Union[str, bool]]) -> Optional[Path]:
    """Resolve cache_dir parameter to a Path or None.

    Args:
        cache_dir: Cache directory specification.
            None  -> use STRING_API_DEFAULT_CACHE_DIR (auto-caching).
            str/Path -> use that path.
            False -> disable caching entirely.

    Returns:
        Path to cache directory, or None if caching is disabled.
    """
    if cache_dir is False:
        return None
    if cache_dir is None:
        return STRING_API_DEFAULT_CACHE_DIR
    return Path(cache_dir)


def _build_url(endpoint: str) -> str:
    """Build a full STRING API URL for the given endpoint.

    Args:
        endpoint: API endpoint name, e.g. 'get_string_ids', 'version'.

    Returns:
        Full URL string, e.g. 'https://string-db.org/api/json/get_string_ids'.
    """
    return f"{STRING_API_BASE_URL}/{STRING_API_OUTPUT_FORMAT}/{endpoint}"


def _cache_key(endpoint: str, params: dict) -> str:
    """Generate a deterministic cache key from endpoint and parameters.

    Args:
        endpoint: API endpoint name.
        params: Request parameters (caller_identity is excluded from the key).

    Returns:
        Hex SHA256 digest string.
    """
    key_data = {"endpoint": endpoint}
    for k, v in sorted(params.items()):
        if k != "caller_identity":
            key_data[k] = v
    return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()


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
        endpoint: API endpoint name (stored as metadata).
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


def _make_request(endpoint: str, params: dict) -> Union[dict, list]:
    """Execute a STRING API request with rate limiting, retry, and backoff.

    This is the sole function that performs HTTP calls. All public functions
    delegate to it. Responsibilities:

    1. Inject ``caller_identity`` into every request (STRING API TOS).
    2. Enforce a minimum pause between consecutive requests.
    3. Retry on HTTP 429 (rate-limited) and 5xx (server error) with
       exponential backoff.
    4. Raise ``StringAPIError`` on non-retryable errors or exhausted retries.

    Args:
        endpoint: API endpoint name (appended to base URL).
        params: Request parameters (identifiers, species, etc.).

    Returns:
        Parsed JSON response (dict or list).

    Raises:
        StringAPIError: On non-retryable HTTP errors or exhausted retries.
    """
    global _last_request_time

    # Inject caller identity
    params = dict(params)
    params["caller_identity"] = STRING_API_CALLER_IDENTITY

    url = _build_url(endpoint)
    encoded_data = urllib.parse.urlencode(params).encode("utf-8")

    last_error: Optional[Exception] = None

    for attempt in range(STRING_API_MAX_RETRIES + 1):
        # Rate limiting
        now = time.monotonic()
        elapsed = now - _last_request_time
        if elapsed < STRING_API_RATE_LIMIT_PAUSE:
            time.sleep(STRING_API_RATE_LIMIT_PAUSE - elapsed)

        try:
            req = urllib.request.Request(url, data=encoded_data)
            with urllib.request.urlopen(req, timeout=STRING_API_TIMEOUT) as response:
                _last_request_time = time.monotonic()
                body = response.read().decode("utf-8")
                return json.loads(body)

        except urllib.error.HTTPError as e:
            _last_request_time = time.monotonic()
            status = e.code
            last_error = e

            # Retryable errors: 429 (rate limited) and 5xx (server error)
            if status == 429 or 500 <= status < 600:
                if attempt < STRING_API_MAX_RETRIES:
                    wait = STRING_API_BACKOFF_FACTOR ** attempt
                    print(
                        f"  STRING API: HTTP {status} on {endpoint}, "
                        f"retrying in {wait:.1f}s (attempt {attempt + 1}/{STRING_API_MAX_RETRIES})",
                        file=sys.stderr,
                    )
                    time.sleep(wait)
                    continue
                else:
                    raise StringAPIError(
                        f"STRING API {endpoint}: HTTP {status} after "
                        f"{STRING_API_MAX_RETRIES} retries"
                    ) from e

            # Non-retryable error (4xx except 429)
            try:
                error_body = e.read().decode("utf-8")
            except Exception:
                error_body = ""
            raise StringAPIError(
                f"STRING API {endpoint}: HTTP {status}: {error_body}"
            ) from e

        except urllib.error.URLError as e:
            _last_request_time = time.monotonic()
            last_error = e
            if attempt < STRING_API_MAX_RETRIES:
                wait = STRING_API_BACKOFF_FACTOR ** attempt
                print(
                    f"  STRING API: connection error on {endpoint}, "
                    f"retrying in {wait:.1f}s (attempt {attempt + 1}/{STRING_API_MAX_RETRIES})",
                    file=sys.stderr,
                )
                time.sleep(wait)
                continue
            raise StringAPIError(
                f"STRING API {endpoint}: connection failed after "
                f"{STRING_API_MAX_RETRIES} retries: {e}"
            ) from e

    # Should not reach here, but safety net
    raise StringAPIError(
        f"STRING API {endpoint}: unexpected failure after retries"
    ) from last_error


# ---Public API Functions-----------------------------------------------

def get_string_ids(identifiers: list[str], species: int = STRING_API_SPECIES,
                   cache_dir: Optional[Union[str, bool]] = None) -> pd.DataFrame:
    """Resolve identifiers to STRING internal IDs.

    Maps protein names, UniProt accessions, or other identifiers to STRING
    identifiers (ENSP IDs) with preferred names and annotations.

    Args:
        identifiers: List of protein identifiers (UniProt, gene names, etc.).
        species: NCBI taxonomy ID (default: 9606 for human).
        cache_dir: Cache directory. None = auto-cache (default), False = no cache.

    Returns:
        DataFrame with columns: queryIndex, queryItem, stringId,
        ncbiTaxonId, taxonName, preferredName, annotation.

    Raises:
        StringAPIError: On API failure.
    """
    params = {
        "identifiers": "\r".join(identifiers),
        "species": species,
        "echo_query": 1,
    }

    resolved_cache = _resolve_cache_dir(cache_dir)

    if resolved_cache:
        key = _cache_key("get_string_ids", params)
        cached = _read_cache(resolved_cache, key)
        if cached is not None:
            return pd.DataFrame(cached)

    result = _make_request("get_string_ids", params)

    if resolved_cache:
        _write_cache(resolved_cache, key, "get_string_ids", result)

    return pd.DataFrame(result)


def get_interaction_partners(identifiers: list[str],
                             species: int = STRING_API_SPECIES,
                             required_score: int = 0,
                             limit: int = 10,
                             cache_dir: Optional[Union[str, bool]] = None) -> pd.DataFrame:
    """Retrieve interaction partners for the given proteins.

    Cross-validates local STRING interactions against the live API, or
    retrieves interaction data for proteins not in local flat files.

    Args:
        identifiers: List of protein identifiers.
        species: NCBI taxonomy ID (default: 9606).
        required_score: Minimum combined score (0-1000, default: 0).
        limit: Maximum number of interaction partners per query protein.
        cache_dir: Cache directory. None = auto-cache (default), False = no cache.

    Returns:
        DataFrame with columns: stringId_A, stringId_B, preferredName_A,
        preferredName_B, ncbiTaxonId, score, nscore, fscore, pscore, ascore,
        escore, dscore, tscore.

    Raises:
        StringAPIError: On API failure.
    """
    params = {
        "identifiers": "\r".join(identifiers),
        "species": species,
        "required_score": required_score,
        "limit": limit,
    }

    resolved_cache = _resolve_cache_dir(cache_dir)

    if resolved_cache:
        key = _cache_key("interaction_partners", params)
        cached = _read_cache(resolved_cache, key)
        if cached is not None:
            return pd.DataFrame(cached)

    result = _make_request("interaction_partners", params)

    if resolved_cache:
        _write_cache(resolved_cache, key, "interaction_partners", result)

    return pd.DataFrame(result)


def query_homology(identifiers: list[str],
                   species: int = STRING_API_SPECIES,
                   cache_dir: Optional[Union[str, bool]] = None) -> pd.DataFrame:
    """Retrieve homology (paralogy/orthology) scores for proteins.

    Returns continuous bitscore values for homologous pairs, supplementing
    the discrete cluster IDs from STRING flat files.

    Args:
        identifiers: List of protein identifiers.
        species: NCBI taxonomy ID (default: 9606).
        cache_dir: Cache directory. None = auto-cache (default), False = no cache.

    Returns:
        DataFrame with columns: ncbiTaxonId_A, stringId_A, ncbiTaxonId_B,
        stringId_B, bitscore.

    Raises:
        StringAPIError: On API failure.
    """
    params = {
        "identifiers": "\r".join(identifiers),
        "species": species,
    }

    resolved_cache = _resolve_cache_dir(cache_dir)

    if resolved_cache:
        key = _cache_key("homology", params)
        cached = _read_cache(resolved_cache, key)
        if cached is not None:
            return pd.DataFrame(cached)

    result = _make_request("homology", params)

    if resolved_cache:
        _write_cache(resolved_cache, key, "homology", result)

    return pd.DataFrame(result)


def query_enrichment(identifiers: list[str],
                     species: int = STRING_API_SPECIES,
                     cache_dir: Optional[Union[str, bool]] = None) -> pd.DataFrame:
    """Perform functional enrichment analysis (GO, KEGG, Reactome).

    Args:
        identifiers: List of protein identifiers (max 2000).
        species: NCBI taxonomy ID (default: 9606).
        cache_dir: Cache directory. None = auto-cache (default), False = no cache.

    Returns:
        DataFrame with columns: category, term, number_of_genes,
        number_of_genes_in_background, ncbiTaxonId, inputGenes,
        preferredNames, p_value, fdr, description.

    Raises:
        StringAPIError: On API failure or if batch size exceeds 2000.
    """
    if len(identifiers) > STRING_API_MAX_ENRICHMENT_BATCH:
        raise StringAPIError(
            f"Enrichment batch size ({len(identifiers)}) exceeds STRING API "
            f"limit of {STRING_API_MAX_ENRICHMENT_BATCH} proteins. Split your "
            f"input into smaller batches."
        )

    params = {
        "identifiers": "\r".join(identifiers),
        "species": species,
    }

    resolved_cache = _resolve_cache_dir(cache_dir)

    if resolved_cache:
        key = _cache_key("enrichment", params)
        cached = _read_cache(resolved_cache, key)
        if cached is not None:
            return pd.DataFrame(cached)

    result = _make_request("enrichment", params)

    if resolved_cache:
        _write_cache(resolved_cache, key, "enrichment", result)

    return pd.DataFrame(result)


def query_ppi_enrichment(identifiers: list[str],
                         species: int = STRING_API_SPECIES,
                         required_score: int = 0,
                         cache_dir: Optional[Union[str, bool]] = None) -> dict:
    """Test whether a set of proteins has more interactions than expected.

    Returns network statistics including observed vs expected edge counts
    and a p-value for enrichment.

    Args:
        identifiers: List of protein identifiers.
        species: NCBI taxonomy ID (default: 9606).
        required_score: Minimum combined score (0-1000, default: 0).
        cache_dir: Cache directory. None = auto-cache (default), False = no cache.

    Returns:
        Dict with keys: number_of_nodes, number_of_edges,
        average_node_degree, local_clustering_coefficient,
        expected_number_of_edges, p_value.

    Raises:
        StringAPIError: On API failure.
    """
    params = {
        "identifiers": "\r".join(identifiers),
        "species": species,
        "required_score": required_score,
    }

    resolved_cache = _resolve_cache_dir(cache_dir)

    if resolved_cache:
        key = _cache_key("ppi_enrichment", params)
        cached = _read_cache(resolved_cache, key)
        if cached is not None:
            return cached if isinstance(cached, dict) else cached[0]

    result = _make_request("ppi_enrichment", params)

    # STRING API returns a list with a single element for ppi_enrichment
    if isinstance(result, list) and len(result) == 1:
        result = result[0]

    if resolved_cache:
        _write_cache(resolved_cache, key, "ppi_enrichment", result)

    return result


def query_network(identifiers: list[str],
                  network_type: str = "functional",
                  species: int = STRING_API_SPECIES,
                  required_score: int = 0,
                  cache_dir: Optional[Union[str, bool]] = None) -> pd.DataFrame:
    """Retrieve network edges between the given proteins.

    Args:
        identifiers: List of protein identifiers.
        network_type: Type of network — 'functional' or 'physical'.
            'regulatory' is planned for future STRING releases but is not
            yet supported by the API.
        species: NCBI taxonomy ID (default: 9606).
        required_score: Minimum combined score (0-1000, default: 0).
        cache_dir: Cache directory. None = auto-cache (default), False = no cache.

    Returns:
        DataFrame with columns: stringId_A, stringId_B, preferredName_A,
        preferredName_B, ncbiTaxonId, score, nscore, fscore, pscore, ascore,
        escore, dscore, tscore.

    Raises:
        ValueError: If network_type is not 'functional' or 'physical'.
        StringAPIError: On API failure.
    """
    if network_type not in _VALID_NETWORK_TYPES:
        raise ValueError(
            f"Invalid network_type '{network_type}'. "
            f"Supported types: {sorted(_VALID_NETWORK_TYPES)}. "
            f"'regulatory' is planned for future STRING releases."
        )

    params = {
        "identifiers": "\r".join(identifiers),
        "species": species,
        "network_type": network_type,
        "required_score": required_score,
    }

    resolved_cache = _resolve_cache_dir(cache_dir)

    if resolved_cache:
        key = _cache_key("network", params)
        cached = _read_cache(resolved_cache, key)
        if cached is not None:
            return pd.DataFrame(cached)

    result = _make_request("network", params)

    if resolved_cache:
        _write_cache(resolved_cache, key, "network", result)

    return pd.DataFrame(result)


def get_version() -> dict:
    """Retrieve the current STRING database version.

    Returns:
        Dict with keys: string_version, stable_address.

    Raises:
        StringAPIError: On API failure.
    """
    result = _make_request("version", {})

    # STRING API returns a list with a single element for version
    if isinstance(result, list) and len(result) == 1:
        result = result[0]

    return result


# ---CLI----------------------------------------------------------------

def build_argument_parser() -> argparse.ArgumentParser:
    """Create and return the argument parser for the STRING API CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "STRING API client — query the STRING database for protein "
            "identifiers, interactions, homology, enrichment, and network data."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage (as importable module):
    from string_api import get_string_ids, get_version, StringAPIError

    try:
        ids_df = get_string_ids(["P04637", "Q9UKT4"])
        version = get_version()
    except StringAPIError as e:
        print(f"STRING API unavailable: {e}")

Usage (standalone):
    python string_api.py --resolve P04637,Q9UKT4 --species 9606
    python string_api.py --enrichment P04637,P12345,Q9UKT4 --species 9606
    python string_api.py --network TP53,MDM2,BRCA1 --network-type physical
    python string_api.py --version
""",
    )

    # Query mode (mutually exclusive)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--resolve",
        metavar="IDS",
        help="Comma-separated identifiers to resolve to STRING IDs",
    )
    mode.add_argument(
        "--interaction-partners",
        metavar="IDS",
        help="Comma-separated identifiers to find interaction partners for",
    )
    mode.add_argument(
        "--homology",
        metavar="IDS",
        help="Comma-separated identifiers to find homologs for",
    )
    mode.add_argument(
        "--enrichment",
        metavar="IDS",
        help="Comma-separated identifiers for functional enrichment analysis",
    )
    mode.add_argument(
        "--ppi-enrichment",
        metavar="IDS",
        help="Comma-separated identifiers for PPI enrichment test",
    )
    mode.add_argument(
        "--network",
        metavar="IDS",
        help="Comma-separated identifiers to retrieve network edges for",
    )
    mode.add_argument(
        "--version",
        action="store_true",
        help="Print the STRING database version and exit",
    )

    # Shared options
    parser.add_argument(
        "--species",
        type=int,
        default=STRING_API_SPECIES,
        help=f"NCBI taxonomy ID (default: {STRING_API_SPECIES} for Homo sapiens)",
    )
    parser.add_argument(
        "--network-type",
        choices=sorted(_VALID_NETWORK_TYPES),
        default="functional",
        help="Network type for --network queries (default: functional)",
    )
    parser.add_argument(
        "--required-score",
        type=int,
        default=0,
        help="Minimum combined score 0-1000 (default: 0, no filter)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Max interaction partners per query protein (default: 10)",
    )
    parser.add_argument(
        "--cache-dir",
        metavar="PATH",
        help="Directory for caching API responses (created if needed)",
    )
    parser.add_argument(
        "--output", "-o",
        metavar="PATH",
        help="Write tabular output to CSV file (default: print to stdout)",
    )

    return parser


def main() -> None:
    """Run the STRING API CLI."""
    parser = build_argument_parser()
    args = parser.parse_args()

    cache_dir = args.cache_dir

    try:
        if args.version:
            result = get_version()
            print(json.dumps(result, indent=2))
            return

        # Parse comma-separated identifiers from whichever mode was chosen
        ids_str = (
            args.resolve or args.interaction_partners or args.homology
            or args.enrichment or args.ppi_enrichment or args.network
        )
        identifiers = [x.strip() for x in ids_str.split(",") if x.strip()]

        if not identifiers:
            print("Error: no identifiers provided", file=sys.stderr)
            sys.exit(1)

        print(f"  Querying STRING API with {len(identifiers)} identifier(s)...",
              file=sys.stderr)

        if args.resolve:
            result = get_string_ids(identifiers, args.species, cache_dir)
        elif args.interaction_partners:
            result = get_interaction_partners(
                identifiers, args.species, args.required_score,
                args.limit, cache_dir,
            )
        elif args.homology:
            result = query_homology(identifiers, args.species, cache_dir)
        elif args.enrichment:
            result = query_enrichment(identifiers, args.species, cache_dir)
        elif args.ppi_enrichment:
            result = query_ppi_enrichment(
                identifiers, args.species, args.required_score, cache_dir,
            )
            print(json.dumps(result, indent=2))
            return
        elif args.network:
            result = query_network(
                identifiers, args.network_type, args.species,
                args.required_score, cache_dir,
            )

        # Output tabular result
        if args.output:
            result.to_csv(args.output, index=False)
            print(f"  Written to {args.output}", file=sys.stderr)
        else:
            print(result.to_string(index=False))

    except StringAPIError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
