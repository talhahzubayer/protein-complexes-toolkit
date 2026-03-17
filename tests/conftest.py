"""
Shared fixtures for the protein-complexes-toolkit test suite.

All fixtures that load real test data are session-scoped to avoid
re-reading PDB/PKL files across tests. Test_Data path is hardcoded
to the repository's test data directory.

Test outputs (CSV files, figures, etc.) are written to tests/test_output/
to avoid cluttering the user-facing project directory.
"""

import sys
from pathlib import Path

import pytest
import numpy as np

# Ensure the project root is importable
PROJECT_ROOT = Path(r"C:\Users\Talhah Zubayer\Documents\protein-complexes-toolkit")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TEST_DATA_DIR = PROJECT_ROOT / "Test_Data"
TEST_DB_DIR = PROJECT_ROOT / "tests" / "offline_test_data" / "databases"
TEST_OUTPUT_DIR = PROJECT_ROOT / "tests" / "test_output"


# ── Path Fixtures ─────────────────────────────────────────────────

@pytest.fixture(scope="session")
def test_data_dir():
    """Return the path to the Test_Data directory, validated to exist."""
    assert TEST_DATA_DIR.exists(), f"Test_Data directory not found: {TEST_DATA_DIR}"
    return TEST_DATA_DIR


# --- Reference Complex 1: heterodimer, old naming convention ---
# A0A0B4J2C3_P24534 - ~269 KB PDB, fast to parse

@pytest.fixture(scope="session")
def ref_pdb_1(test_data_dir):
    """PDB path for reference complex 1 (old naming convention)."""
    path = test_data_dir / "A0A0B4J2C3_P24534.pdb"
    assert path.exists(), f"Reference PDB 1 not found: {path}"
    return path


@pytest.fixture(scope="session")
def ref_pkl_1(test_data_dir):
    """PKL path for reference complex 1 (old naming convention)."""
    path = test_data_dir / "A0A0B4J2C3_P24534.results.pkl"
    assert path.exists(), f"Reference PKL 1 not found: {path}"
    return path


# --- Reference Complex 2: heterodimer, new naming convention ---
# A0A0A0MQZ0_P40933 - _relaxed_model_ / _result_model_ naming

@pytest.fixture(scope="session")
def ref_pdb_2(test_data_dir):
    """PDB path for reference complex 2 (new naming convention)."""
    path = test_data_dir / "A0A0A0MQZ0_P40933_relaxed_model_1_multimer_v3_pred_0.pdb"
    assert path.exists(), f"Reference PDB 2 not found: {path}"
    return path


@pytest.fixture(scope="session")
def ref_pkl_2(test_data_dir):
    """PKL path for reference complex 2 (new naming convention)."""
    path = test_data_dir / "A0A0A0MQZ0_P40933_result_model_1_multimer_v3_pred_0.pkl"
    assert path.exists(), f"Reference PKL 2 not found: {path}"
    return path


# --- Reference Complex 3: homodimer ---
# A0A0H3C8Q1_A0A0H3C8Q1 - protein_a == protein_b

@pytest.fixture(scope="session")
def ref_pdb_homodimer(test_data_dir):
    """PDB path for homodimer reference complex."""
    path = test_data_dir / "A0A0H3C8Q1_A0A0H3C8Q1.pdb"
    assert path.exists(), f"Homodimer PDB not found: {path}"
    return path


@pytest.fixture(scope="session")
def ref_pkl_homodimer(test_data_dir):
    """PKL path for homodimer reference complex."""
    path = test_data_dir / "A0A0H3C8Q1_A0A0H3C8Q1.results.pkl"
    assert path.exists(), f"Homodimer PKL not found: {path}"
    return path


# --- Reference Complex 4: isoform dash ID, cross-convention ---
# P63208-1_Q6PJ61 - old PDB name, new PKL name

@pytest.fixture(scope="session")
def ref_pdb_isoform(test_data_dir):
    """PDB path for isoform-dash reference complex (old naming)."""
    path = test_data_dir / "P63208-1_Q6PJ61.pdb"
    assert path.exists(), f"Isoform PDB not found: {path}"
    return path


@pytest.fixture(scope="session")
def ref_pkl_isoform(test_data_dir):
    """PKL path for isoform-dash reference complex (new naming)."""
    path = test_data_dir / "P63208-1_Q6PJ61_result_model_1_multimer_v3_pred_0.pkl"
    assert path.exists(), f"Isoform PKL not found: {path}"
    return path


# --- Reference Complex 5: doubled-name edge case ---
# P0C0L2_P0C0L2 PDB + P0C0L2_P0C0L2_P0C0L2_P0C0L2.results.pkl

@pytest.fixture(scope="session")
def ref_pdb_doubled(test_data_dir):
    """PDB path for doubled-name homodimer."""
    path = test_data_dir / "P0C0L2_P0C0L2.pdb"
    assert path.exists(), f"Doubled-name PDB not found: {path}"
    return path


@pytest.fixture(scope="session")
def ref_pkl_doubled(test_data_dir):
    """PKL path for doubled-name homodimer (name repeated twice)."""
    path = test_data_dir / "P0C0L2_P0C0L2_P0C0L2_P0C0L2.results.pkl"
    assert path.exists(), f"Doubled-name PKL not found: {path}"
    return path


# ── Loaded Data Fixtures (Complex 1) ─────────────────────────────

@pytest.fixture(scope="session")
def loaded_pkl_1(ref_pkl_1):
    """Loaded prediction result dict for reference complex 1."""
    from read_af2_nojax import load_pkl_without_jax
    return load_pkl_without_jax(ref_pkl_1)


@pytest.fixture(scope="session")
def extracted_metrics_1(loaded_pkl_1):
    """Extracted metrics dict for reference complex 1."""
    from read_af2_nojax import extract_metrics
    return extract_metrics(loaded_pkl_1)


@pytest.fixture(scope="session")
def pae_matrix_1(loaded_pkl_1):
    """PAE matrix (2D numpy array) for reference complex 1."""
    return np.asarray(loaded_pkl_1['predicted_aligned_error'])


@pytest.fixture(scope="session")
def chain_coords_plddt_1(ref_pdb_1):
    """(chain_coords, chain_plddt) from read_pdb_Edited for reference complex 1."""
    from pdockq import read_pdb_Edited
    return read_pdb_Edited(str(ref_pdb_1))


@pytest.fixture(scope="session")
def chain_info_1(ref_pdb_1):
    """ChainInfo_New from read_pdb_with_chain_info_New for reference complex 1."""
    from pdockq import read_pdb_with_chain_info_New
    return read_pdb_with_chain_info_New(str(ref_pdb_1))


@pytest.fixture(scope="session")
def contact_result_1(chain_coords_plddt_1):
    """ContactResult_New for reference complex 1."""
    from pdockq import calc_pdockq_and_contacts_New
    chain_coords, chain_plddt = chain_coords_plddt_1
    return calc_pdockq_and_contacts_New(chain_coords, chain_plddt, t=8)


@pytest.fixture(scope="session")
def best_pair_1(chain_info_1):
    """(chain_a, chain_b, ContactResult_New) from find_best_chain_pair for complex 1."""
    from pdockq import find_best_chain_pair_New
    return find_best_chain_pair_New(chain_info_1, t=8)


@pytest.fixture(scope="session")
def chain_offsets_1(chain_info_1):
    """Chain PAE offsets dict for reference complex 1."""
    from pdockq import compute_pae_chain_offsets_New
    return compute_pae_chain_offsets_New(chain_info_1)


# ── Loaded Data Fixtures (Complex 2 - new naming) ────────────────

@pytest.fixture(scope="session")
def loaded_pkl_2(ref_pkl_2):
    """Loaded prediction result dict for reference complex 2."""
    from read_af2_nojax import load_pkl_without_jax
    return load_pkl_without_jax(ref_pkl_2)


@pytest.fixture(scope="session")
def chain_info_2(ref_pdb_2):
    """ChainInfo_New for reference complex 2."""
    from pdockq import read_pdb_with_chain_info_New
    return read_pdb_with_chain_info_New(str(ref_pdb_2))


# ── Loaded Data Fixtures (Homodimer) ──────────────────────────────

@pytest.fixture(scope="session")
def chain_coords_plddt_homodimer(ref_pdb_homodimer):
    """(chain_coords, chain_plddt) for homodimer reference complex."""
    from pdockq import read_pdb_Edited
    return read_pdb_Edited(str(ref_pdb_homodimer))


@pytest.fixture(scope="session")
def contact_result_homodimer(chain_coords_plddt_homodimer):
    """ContactResult_New for homodimer reference complex."""
    from pdockq import calc_pdockq_and_contacts_New
    chain_coords, chain_plddt = chain_coords_plddt_homodimer
    return calc_pdockq_and_contacts_New(chain_coords, chain_plddt, t=8)


# ── Test Output Directory ─────────────────────────────────────────

@pytest.fixture(scope="session")
def test_output_dir():
    """Return the tests/test_output/ directory, creating it if needed.

    All test-generated files (CSVs, figures, etc.) go here to avoid
    cluttering the user-facing project directory.
    """
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return TEST_OUTPUT_DIR


# ── Pipeline-Generated CSV Fixture ────────────────────────────────

@pytest.fixture(scope="session")
def pipeline_csv(test_output_dir, test_data_dir):
    """Generate a full 44-column CSV by running the pipeline on all test data.

    Processes every complex in Test_Data that has both PDB and PKL files
    with --interface --pae, writes the result to tests/test_output/test_results.csv.
    This CSV is then used by visualisation tests instead of the pre-existing
    results.csv, ensuring the visual pipeline is truly tested end-to-end.
    """
    from toolkit import find_paired_data_files, process_single_complex, write_results_csv

    csv_path = test_output_dir / "test_results.csv"

    # Only regenerate if the CSV doesn't exist yet (session-scoped)
    if csv_path.exists():
        return csv_path

    pairs = find_paired_data_files(str(test_data_dir))
    results = []
    for name, file_paths in sorted(pairs.items()):
        if 'pdb' in file_paths and 'pkl' in file_paths:
            row = process_single_complex(
                name, file_paths,
                run_interface=True, run_interface_pae=True,
            )
            results.append(row)

    write_results_csv(results, str(csv_path), include_interface=True, include_pae=True)
    return csv_path


# ── Database Test Data Fixtures ──────────────────────────────────

@pytest.fixture(scope="session")
def test_db_dir():
    """Return the path to the tests/offline_test_data/databases/ directory."""
    assert TEST_DB_DIR.exists(), f"Test database directory not found: {TEST_DB_DIR}"
    return TEST_DB_DIR


@pytest.fixture(scope="session")
def test_aliases_path(test_db_dir):
    """Return the path to the test STRING aliases excerpt."""
    path = test_db_dir / "test_aliases.txt"
    assert path.exists(), f"Test aliases file not found: {path}"
    return path


@pytest.fixture(scope="session")
def id_mapper(test_aliases_path):
    """Session-scoped IDMapper loaded from the test aliases excerpt."""
    from id_mapper import IDMapper
    return IDMapper(str(test_aliases_path))


# ── Variant Test Data Fixtures ───────────────────────────────────

@pytest.fixture(scope="session")
def test_uniprot_variants_path(test_db_dir):
    """Return the path to the test UniProt variants excerpt."""
    path = test_db_dir / "test_uniprot_variants.txt"
    assert path.exists(), f"Test UniProt variants file not found: {path}"
    return path


@pytest.fixture(scope="session")
def test_clinvar_path(test_db_dir):
    """Return the path to the test ClinVar variants excerpt."""
    path = test_db_dir / "test_clinvar_variants.txt"
    assert path.exists(), f"Test ClinVar variants file not found: {path}"
    return path


@pytest.fixture(scope="session")
def test_exac_path(test_db_dir):
    """Return the path to the test ExAC constraint excerpt."""
    path = test_db_dir / "test_exac_constraint.txt"
    assert path.exists(), f"Test ExAC constraint file not found: {path}"
    return path


# ── Clustering Test Data Fixtures ────────────────────────────────

@pytest.fixture(scope="session")
def test_clusters_path(test_db_dir):
    """Return the path to the test STRING clusters excerpt."""
    path = test_db_dir / "test_string_clusters.txt"
    assert path.exists(), f"Test clusters file not found: {path}"
    return path


# ── STRING API Test Data Fixtures ────────────────────────────────

@pytest.fixture(scope="session")
def string_api_responses(test_db_dir):
    """Load all pre-captured STRING API response files into a dict.

    Returns a dict keyed by endpoint name (filename stem), e.g.
    {'get_string_ids': [...], 'version': [...], ...}.
    """
    import json
    response_dir = test_db_dir / "string_api_responses"
    assert response_dir.exists(), f"STRING API response dir not found: {response_dir}"
    responses = {}
    for path in sorted(response_dir.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            responses[path.stem] = json.load(f)
    return responses


# ── Stability Scoring Test Data Fixtures ─────────────────────────

@pytest.fixture(scope="session")
def test_eve_map_path(test_db_dir):
    """Return the path to the test UniProt ID mapping excerpt."""
    path = test_db_dir / "test_idmapping.dat"
    assert path.exists(), f"Test ID mapping file not found: {path}"
    return path


@pytest.fixture(scope="session")
def test_eve_dir(test_db_dir):
    """Return a directory containing the test EVE CSV.

    The test EVE CSV is named after a known test accession's entry name.
    Creates a temporary symlink/copy so the EVE loader can find it by entry name.
    """
    import shutil
    eve_test_dir = test_db_dir / "test_eve_data"
    eve_test_dir.mkdir(exist_ok=True)
    # Copy test_eve_scores.csv as 1433G_HUMAN.csv (the entry name for P61981)
    src = test_db_dir / "test_eve_scores.csv"
    dst = eve_test_dir / "1433G_HUMAN.csv"
    if not dst.exists():
        shutil.copy2(src, dst)
    return eve_test_dir


@pytest.fixture(scope="session")
def test_protvar_responses_dir(test_db_dir):
    """Return the directory containing ProtVar API response test data.

    Contains pre-captured JSON responses for offline testing:
        score_P61981_4.json, interaction_P61981_4.json, foldx_P61981_4.json
        score_P24534_81.json, interaction_P24534_81.json, foldx_P24534_81.json
    """
    path = test_db_dir / "protvar_responses"
    assert path.exists(), f"Missing test data: {path}"
    return path
