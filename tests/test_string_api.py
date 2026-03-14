"""
Tests for the STRING API client module (string_api.py).

All tests use mocked HTTP responses — no live network calls are made.
Pre-captured JSON responses are loaded from
tests/offline_test_data/databases/string_api_responses/ via the
string_api_responses fixture in conftest.py.

Test counts by class:
    TestConstants                   4
    TestStringAPIError              2
    TestBuildUrl                    2
    TestMakeRequest                 8
    TestCaching                     4
    TestGetStringIds                3
    TestGetInteractionPartners      3
    TestQueryHomology               2
    TestQueryEnrichment             3
    TestQueryPpiEnrichment          2
    TestQueryNetwork                3
    TestGetVersion                  2
    TestCLI                         3
    TestDefaultCacheDir             2
    TestResolveCacheDir             3
    TestResolveIdApiFallback        4
    TestEnrichResultsApiFallback    3
    TestValidateWithApi             3
    ─────────────────────────────────
    Total                          56
"""

import io
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Import the module under test
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import string_api
from string_api import (
    STRING_API_BASE_URL,
    STRING_API_CALLER_IDENTITY,
    STRING_API_OUTPUT_FORMAT,
    STRING_API_SPECIES,
    STRING_API_MAX_ENRICHMENT_BATCH,
    STRING_API_RATE_LIMIT_PAUSE,
    STRING_API_MAX_RETRIES,
    STRING_API_BACKOFF_FACTOR,
    STRING_API_TIMEOUT,
    STRING_API_DEFAULT_CACHE_DIR,
    StringAPIError,
    _build_url,
    _cache_key,
    _read_cache,
    _write_cache,
    _resolve_cache_dir,
    _make_request,
    get_string_ids,
    get_interaction_partners,
    query_homology,
    query_enrichment,
    query_ppi_enrichment,
    query_network,
    get_version,
)

pytestmark = pytest.mark.api


# ── Helper: mock urlopen response ─────────────────────────────────

def _mock_response(data, status=200):
    """Create a mock urllib response object returning JSON data."""
    response = MagicMock()
    response.read.return_value = json.dumps(data).encode("utf-8")
    response.status = status
    response.__enter__ = MagicMock(return_value=response)
    response.__exit__ = MagicMock(return_value=False)
    return response


# ── TestConstants ─────────────────────────────────────────────────

class TestConstants:
    """Verify module-level constants are correctly defined."""

    def test_base_url_format(self):
        assert STRING_API_BASE_URL.startswith("https://")
        assert "string-db.org" in STRING_API_BASE_URL

    def test_caller_identity_nonempty(self):
        assert isinstance(STRING_API_CALLER_IDENTITY, str)
        assert len(STRING_API_CALLER_IDENTITY) > 0

    def test_species_default(self):
        assert STRING_API_SPECIES == 9606

    def test_output_format_json(self):
        assert STRING_API_OUTPUT_FORMAT == "json"


# ── TestStringAPIError ────────────────────────────────────────────

class TestStringAPIError:
    """Verify StringAPIError exception class."""

    def test_inherits_runtime_error(self):
        assert issubclass(StringAPIError, RuntimeError)

    def test_message_preserved(self):
        msg = "test error message"
        error = StringAPIError(msg)
        assert str(error) == msg


# ── TestBuildUrl ──────────────────────────────────────────────────

class TestBuildUrl:
    """Verify URL construction."""

    def test_constructs_correct_url(self):
        url = _build_url("get_string_ids")
        assert url == f"{STRING_API_BASE_URL}/{STRING_API_OUTPUT_FORMAT}/get_string_ids"

    def test_includes_format(self):
        url = _build_url("version")
        assert "/json/" in url


# ── TestMakeRequest ───────────────────────────────────────────────

class TestMakeRequest:
    """Verify _make_request internals: caller_identity, rate limiting, retry."""

    @patch("string_api.urllib.request.urlopen")
    def test_injects_caller_identity(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response({"result": "ok"})
        # Reset rate limiter
        string_api._last_request_time = 0.0

        _make_request("version", {})

        # Verify caller_identity was in the encoded data
        call_args = mock_urlopen.call_args
        request_obj = call_args[0][0]
        posted_data = request_obj.data.decode("utf-8")
        assert "caller_identity" in posted_data
        assert STRING_API_CALLER_IDENTITY in posted_data

    @patch("string_api.time.sleep")
    @patch("string_api.urllib.request.urlopen")
    def test_rate_limiting_enforced(self, mock_urlopen, mock_sleep):
        mock_urlopen.return_value = _mock_response([])

        # Simulate very recent request
        string_api._last_request_time = string_api.time.monotonic()

        _make_request("version", {})

        # time.sleep should have been called for rate limiting
        assert mock_sleep.called
        sleep_duration = mock_sleep.call_args[0][0]
        assert sleep_duration > 0
        assert sleep_duration <= STRING_API_RATE_LIMIT_PAUSE

    @patch("string_api.time.sleep")
    @patch("string_api.urllib.request.urlopen")
    def test_retry_on_429(self, mock_urlopen, mock_sleep):
        import urllib.error
        error_429 = urllib.error.HTTPError(
            url="http://test", code=429, msg="Rate Limited",
            hdrs=None, fp=io.BytesIO(b"rate limited"),
        )
        mock_urlopen.side_effect = [
            error_429,
            _mock_response({"ok": True}),
        ]
        string_api._last_request_time = 0.0

        result = _make_request("version", {})
        assert result == {"ok": True}
        assert mock_urlopen.call_count == 2

    @patch("string_api.time.sleep")
    @patch("string_api.urllib.request.urlopen")
    def test_retry_on_500(self, mock_urlopen, mock_sleep):
        import urllib.error
        error_500 = urllib.error.HTTPError(
            url="http://test", code=500, msg="Server Error",
            hdrs=None, fp=io.BytesIO(b"internal error"),
        )
        mock_urlopen.side_effect = [
            error_500,
            _mock_response({"ok": True}),
        ]
        string_api._last_request_time = 0.0

        result = _make_request("version", {})
        assert result == {"ok": True}

    @patch("string_api.time.sleep")
    @patch("string_api.urllib.request.urlopen")
    def test_retry_on_503(self, mock_urlopen, mock_sleep):
        import urllib.error
        error_503 = urllib.error.HTTPError(
            url="http://test", code=503, msg="Service Unavailable",
            hdrs=None, fp=io.BytesIO(b"unavailable"),
        )
        mock_urlopen.side_effect = [
            error_503,
            _mock_response({"ok": True}),
        ]
        string_api._last_request_time = 0.0

        result = _make_request("version", {})
        assert result == {"ok": True}

    @patch("string_api.time.sleep")
    @patch("string_api.urllib.request.urlopen")
    def test_max_retries_exhausted_raises(self, mock_urlopen, mock_sleep):
        import urllib.error
        error_500 = urllib.error.HTTPError(
            url="http://test", code=500, msg="Server Error",
            hdrs=None, fp=io.BytesIO(b"error"),
        )
        # Fail on all attempts (initial + MAX_RETRIES retries)
        mock_urlopen.side_effect = [error_500] * (STRING_API_MAX_RETRIES + 1)
        string_api._last_request_time = 0.0

        with pytest.raises(StringAPIError, match="after.*retries"):
            _make_request("version", {})

        assert mock_urlopen.call_count == STRING_API_MAX_RETRIES + 1

    @patch("string_api.urllib.request.urlopen")
    def test_timeout_passed(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response([])
        string_api._last_request_time = 0.0

        _make_request("version", {})

        call_args = mock_urlopen.call_args
        assert call_args[1].get("timeout") == STRING_API_TIMEOUT or \
               (len(call_args[0]) > 1 and call_args[0][1] == STRING_API_TIMEOUT)

    @patch("string_api.urllib.request.urlopen")
    def test_non_retryable_4xx_raises_immediately(self, mock_urlopen):
        import urllib.error
        error_404 = urllib.error.HTTPError(
            url="http://test", code=404, msg="Not Found",
            hdrs=None, fp=io.BytesIO(b"not found"),
        )
        mock_urlopen.side_effect = error_404
        string_api._last_request_time = 0.0

        with pytest.raises(StringAPIError, match="404"):
            _make_request("version", {})

        # Should NOT retry on 404
        assert mock_urlopen.call_count == 1


# ── TestCaching ───────────────────────────────────────────────────

class TestCaching:
    """Verify cache key generation and read/write roundtrip."""

    def test_cache_key_deterministic(self):
        key1 = _cache_key("test", {"a": 1, "b": 2})
        key2 = _cache_key("test", {"a": 1, "b": 2})
        assert key1 == key2

    def test_cache_key_order_independent(self):
        key1 = _cache_key("test", {"a": 1, "b": 2})
        key2 = _cache_key("test", {"b": 2, "a": 1})
        assert key1 == key2

    def test_cache_write_and_read_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            data = [{"x": 1}, {"x": 2}]
            key = "test_key"

            _write_cache(cache_dir, key, "test_endpoint", data)
            result = _read_cache(cache_dir, key)

            assert result == data

    def test_cache_miss_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _read_cache(Path(tmpdir), "nonexistent_key")
            assert result is None


# ── TestGetStringIds ──────────────────────────────────────────────

class TestGetStringIds:
    """Verify get_string_ids function."""

    @patch("string_api._make_request")
    def test_happy_path_returns_dataframe(self, mock_request, string_api_responses):
        mock_request.return_value = string_api_responses["get_string_ids"]

        result = get_string_ids(["P04637", "Q9UKT4"])

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "stringId" in result.columns
        assert "preferredName" in result.columns

    @patch("string_api._make_request")
    def test_echo_query_included(self, mock_request, string_api_responses):
        mock_request.return_value = string_api_responses["get_string_ids"]

        get_string_ids(["P04637"], cache_dir=False)

        call_params = mock_request.call_args[0][1]
        assert call_params.get("echo_query") == 1

    @patch("string_api._make_request")
    def test_error_raises_string_api_error(self, mock_request):
        mock_request.side_effect = StringAPIError("test error")

        with pytest.raises(StringAPIError):
            get_string_ids(["INVALID_ID"])


# ── TestGetInteractionPartners ────────────────────────────────────

class TestGetInteractionPartners:
    """Verify get_interaction_partners function."""

    @patch("string_api._make_request")
    def test_happy_path_returns_dataframe(self, mock_request, string_api_responses):
        mock_request.return_value = string_api_responses["interaction_partners"]

        result = get_interaction_partners(["P04637"])

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "stringId_A" in result.columns
        assert "score" in result.columns

    @patch("string_api._make_request")
    def test_required_score_passed(self, mock_request, string_api_responses):
        mock_request.return_value = string_api_responses["interaction_partners"]

        get_interaction_partners(["P04637"], required_score=700, cache_dir=False)

        call_params = mock_request.call_args[0][1]
        assert call_params["required_score"] == 700

    @patch("string_api._make_request")
    def test_empty_result_returns_empty_df(self, mock_request):
        mock_request.return_value = []

        result = get_interaction_partners(["UNKNOWN_PROTEIN"])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# ── TestQueryHomology ─────────────────────────────────────────────

class TestQueryHomology:
    """Verify query_homology function."""

    @patch("string_api._make_request")
    def test_happy_path_returns_dataframe(self, mock_request, string_api_responses):
        mock_request.return_value = string_api_responses["homology"]

        result = query_homology(["TP53", "TP63"])

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "bitscore" in result.columns

    @patch("string_api._make_request")
    def test_error_handling(self, mock_request):
        mock_request.side_effect = StringAPIError("server error")

        with pytest.raises(StringAPIError):
            query_homology(["INVALID"])


# ── TestQueryEnrichment ───────────────────────────────────────────

class TestQueryEnrichment:
    """Verify query_enrichment function."""

    @patch("string_api._make_request")
    def test_happy_path_returns_dataframe(self, mock_request, string_api_responses):
        mock_request.return_value = string_api_responses["enrichment"]

        result = query_enrichment(["P04637", "P38398", "Q9UKT4"])

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_batch_limit_exceeded_raises(self):
        huge_list = [f"PROT{i:05d}" for i in range(STRING_API_MAX_ENRICHMENT_BATCH + 1)]

        with pytest.raises(StringAPIError, match="exceeds"):
            query_enrichment(huge_list)

    @patch("string_api._make_request")
    def test_fdr_column_present(self, mock_request, string_api_responses):
        mock_request.return_value = string_api_responses["enrichment"]

        result = query_enrichment(["P04637", "P38398"])

        assert "fdr" in result.columns


# ── TestQueryPpiEnrichment ────────────────────────────────────────

class TestQueryPpiEnrichment:
    """Verify query_ppi_enrichment function."""

    @patch("string_api._make_request")
    def test_happy_path_returns_dict(self, mock_request, string_api_responses):
        mock_request.return_value = string_api_responses["ppi_enrichment"]

        result = query_ppi_enrichment(["P04637", "P38398", "Q9UKT4"])

        assert isinstance(result, dict)

    @patch("string_api._make_request")
    def test_pvalue_field_present(self, mock_request, string_api_responses):
        mock_request.return_value = string_api_responses["ppi_enrichment"]

        result = query_ppi_enrichment(["P04637", "P38398"])

        assert "p_value" in result


# ── TestQueryNetwork ──────────────────────────────────────────────

class TestQueryNetwork:
    """Verify query_network function."""

    @patch("string_api._make_request")
    def test_happy_path_returns_dataframe(self, mock_request, string_api_responses):
        mock_request.return_value = string_api_responses["network"]

        result = query_network(["TP53", "MDM2", "BRCA1"])

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "stringId_A" in result.columns

    def test_invalid_network_type_raises(self):
        with pytest.raises(ValueError, match="regulatory"):
            query_network(["TP53"], network_type="regulatory")

    @patch("string_api._make_request")
    def test_physical_network_type_accepted(self, mock_request, string_api_responses):
        mock_request.return_value = string_api_responses["network"]

        result = query_network(["TP53", "MDM2"], network_type="physical", cache_dir=False)

        call_params = mock_request.call_args[0][1]
        assert call_params["network_type"] == "physical"


# ── TestGetVersion ────────────────────────────────────────────────

class TestGetVersion:
    """Verify get_version function."""

    @patch("string_api._make_request")
    def test_returns_dict_with_version(self, mock_request, string_api_responses):
        mock_request.return_value = string_api_responses["version"]

        result = get_version()

        assert isinstance(result, dict)
        assert "string_version" in result

    @patch("string_api._make_request")
    def test_returns_stable_address(self, mock_request, string_api_responses):
        mock_request.return_value = string_api_responses["version"]

        result = get_version()

        assert "stable_address" in result


# ── TestCLI ───────────────────────────────────────────────────────

class TestCLI:
    """Verify CLI entry points via subprocess."""

    @pytest.mark.cli
    def test_resolve_subcommand(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "string_api", "--resolve", "P04637",
             "--species", "9606"],
            capture_output=True, text=True, timeout=60,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        # Should succeed (exit 0) with output containing STRING ID
        assert result.returncode == 0
        assert "ENSP" in result.stdout or "TP53" in result.stdout

    @pytest.mark.cli
    def test_version_subcommand(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "string_api", "--version"],
            capture_output=True, text=True, timeout=60,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert result.returncode == 0
        assert "string_version" in result.stdout

    @pytest.mark.cli
    def test_enrichment_subcommand(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "string_api", "--enrichment",
             "P04637,P38398,Q9UKT4", "--species", "9606"],
            capture_output=True, text=True, timeout=60,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert result.returncode == 0
        assert "fdr" in result.stdout or "p_value" in result.stdout


# ── Default Cache Directory ──────────────────────────────────────

class TestDefaultCacheDir:
    """Tests for the STRING_API_DEFAULT_CACHE_DIR constant."""

    def test_constant_is_path(self):
        assert isinstance(STRING_API_DEFAULT_CACHE_DIR, Path)

    def test_constant_points_to_expected_location(self):
        assert STRING_API_DEFAULT_CACHE_DIR.name == "string_api_cache"
        assert STRING_API_DEFAULT_CACHE_DIR.parent.name == "data"


class TestResolveCacheDir:
    """Tests for _resolve_cache_dir() helper."""

    def test_none_returns_default(self):
        result = _resolve_cache_dir(None)
        assert result == STRING_API_DEFAULT_CACHE_DIR

    def test_false_returns_none(self):
        result = _resolve_cache_dir(False)
        assert result is None

    def test_string_returns_path(self):
        result = _resolve_cache_dir("/tmp/my_cache")
        assert result == Path("/tmp/my_cache")


# ── API Fallback Integration Tests ───────────────────────────────

class TestResolveIdApiFallback:
    """Tests for IDMapper.resolve_id() API fallback behaviour."""

    def test_local_miss_api_hit(self, test_aliases_path):
        """When local lookup fails, API resolves the identifier."""
        from id_mapper import IDMapper

        mapper = IDMapper(str(test_aliases_path), api_fallback=True)

        mock_df = pd.DataFrame([{
            'queryIndex': 0,
            'queryItem': 'UNKNOWN_PROTEIN',
            'stringId': '9606.ENSP00000269305',
            'preferredName': 'TP53',
            'ncbiTaxonId': 9606,
            'taxonName': 'Homo sapiens',
            'annotation': 'test',
        }])

        with patch('string_api._make_request', return_value=mock_df.to_dict('records')):
            result = mapper.resolve_id('UNKNOWN_PROTEIN', target='gene_symbol')
            assert result == 'TP53'

    def test_api_error_returns_none_gracefully(self, test_aliases_path):
        """StringAPIError disables API and returns None."""
        from id_mapper import IDMapper

        mapper = IDMapper(str(test_aliases_path), api_fallback=True)
        assert mapper._api_available is True

        # Patch get_string_ids at the module level to bypass caching
        with patch('string_api.get_string_ids', side_effect=StringAPIError("unavailable")):
            result = mapper.resolve_id('TOTALLY_FAKE_ID_XYZ', target='gene_symbol')
            assert result is None
            assert mapper._api_available is False

    def test_api_available_latches_off(self, test_aliases_path):
        """After one API failure, subsequent calls skip the API entirely."""
        from id_mapper import IDMapper

        mapper = IDMapper(str(test_aliases_path), api_fallback=True)
        mapper._api_available = False

        with patch('string_api._make_request') as mock_req:
            result = mapper.resolve_id('UNKNOWN_PROTEIN', target='gene_symbol')
            assert result is None
            mock_req.assert_not_called()

    def test_api_fallback_disabled(self, test_aliases_path):
        """When api_fallback=False, no API calls are made."""
        from id_mapper import IDMapper

        mapper = IDMapper(str(test_aliases_path), api_fallback=False)

        with patch('string_api._make_request') as mock_req:
            result = mapper.resolve_id('UNKNOWN_PROTEIN', target='gene_symbol')
            assert result is None
            mock_req.assert_not_called()


class TestEnrichResultsApiFallback:
    """Tests for toolkit.enrich_results() API fallback via mapper."""

    def test_missing_lookup_with_mapper_resolves(self, test_aliases_path):
        """When a protein is missing from lookup, mapper API fills the gap."""
        from id_mapper import IDMapper
        from toolkit import enrich_results

        mapper = IDMapper(str(test_aliases_path), api_fallback=True)

        results = [{'protein_a': 'P04637', 'protein_b': 'Q9UKT4'}]
        lookup = {
            'P04637': {
                'gene_symbol': 'TP53', 'protein_name': 'Tumor protein p53',
                'ensembl_protein_id': 'ENSP00000269305',
                'ensembl_gene_id': 'ENSG00000141510',
                'secondary_accessions': '',
            },
        }  # Q9UKT4 is missing from lookup

        # Mock the API to return nothing (we're testing the fallback path exists)
        with patch('string_api._make_request', return_value=[]):
            enrich_results(results, lookup, mapper=mapper)

        # protein_a should be enriched from lookup
        assert results[0]['gene_symbol_a'] == 'TP53'
        # protein_b should have empty fields (API returned nothing)
        assert results[0]['gene_symbol_b'] == ''

    def test_mapper_none_skips_api(self):
        """When mapper=None, no API fallback is attempted."""
        from toolkit import enrich_results

        results = [{'protein_a': 'MISSING_A', 'protein_b': 'MISSING_B'}]
        lookup = {}

        with patch('string_api._make_request') as mock_req:
            enrich_results(results, lookup, mapper=None)
            mock_req.assert_not_called()

        assert results[0]['gene_symbol_a'] == ''
        assert results[0]['gene_symbol_b'] == ''

    def test_api_error_leaves_empty_fields(self, test_aliases_path):
        """API errors produce empty enrichment fields, not exceptions."""
        from id_mapper import IDMapper
        from toolkit import enrich_results

        mapper = IDMapper(str(test_aliases_path), api_fallback=True)

        results = [{'protein_a': 'MISSING_A', 'protein_b': 'MISSING_B'}]
        lookup = {}

        with patch('string_api._make_request', side_effect=StringAPIError("fail")):
            enrich_results(results, lookup, mapper=mapper)

        assert results[0]['gene_symbol_a'] == ''
        assert results[0]['gene_symbol_b'] == ''


class TestValidateWithApi:
    """Tests for database_loaders.validate_with_api()."""

    def test_happy_path(self):
        """Validation returns correct match counts."""
        from database_loaders import validate_with_api

        df = pd.DataFrame({
            'protein_a': ['9606.ENSP00000269305', '9606.ENSP00000344818',
                          '9606.ENSP00000256474'],
            'protein_b': ['9606.ENSP00000344818', '9606.ENSP00000256474',
                          '9606.ENSP00000269305'],
        })

        mock_response = [
            {'queryItem': '9606.ENSP00000269305', 'stringId': '9606.ENSP00000269305',
             'preferredName': 'TP53', 'ncbiTaxonId': 9606},
            {'queryItem': '9606.ENSP00000344818', 'stringId': '9606.ENSP00000344818',
             'preferredName': 'MDM2', 'ncbiTaxonId': 9606},
            {'queryItem': '9606.ENSP00000256474', 'stringId': '9606.ENSP00000256474',
             'preferredName': 'BRCA1', 'ncbiTaxonId': 9606},
        ]

        with patch('string_api._make_request', return_value=mock_response):
            report = validate_with_api(df, 'STRING', sample_size=3)

        assert report['api_error'] is False
        assert report['total_checked'] == 3
        assert report['matched'] == 3
        assert report['match_rate'] == 1.0

    def test_api_error_returns_flag(self):
        """API errors are flagged, not raised."""
        from database_loaders import validate_with_api

        df = pd.DataFrame({
            'protein_a': ['ENSP00000269305'],
            'protein_b': ['ENSP00000344818'],
        })

        with patch('string_api._make_request', side_effect=StringAPIError("fail")):
            report = validate_with_api(df, 'STRING', sample_size=1)

        assert report['api_error'] is True

    def test_empty_dataframe(self):
        """Empty DataFrame returns clean report."""
        from database_loaders import validate_with_api

        df = pd.DataFrame(columns=['protein_a', 'protein_b'])
        report = validate_with_api(df, 'STRING')

        assert report['total_checked'] == 0
        assert report['api_error'] is False
