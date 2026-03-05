"""
Tests for pdockq.py - PDB reading, pDockQ calculation, multi-chain support.

Regression values captured on 2026-03-04 from reference complex 1
(A0A0B4J2C3_P24534) on Windows 11.
"""

import numpy as np
import pytest

from pdockq import (
    parse_atm_record_Edited,
    read_pdb_Edited,
    read_pdb_with_residue_ids_New,
    read_pdb_with_chain_info_New,
    calc_pdockq_Edited,
    calc_pdockq_and_contacts_New,
    compute_pae_chain_offsets_New,
    find_best_chain_pair_New,
    _lookup_ppv_New,
    ContactResult_New,
    ChainInfo_New,
    DEFAULT_CONTACT_THRESHOLD_New,
    PDOCKQ_L_New,
    PDOCKQ_K_New,
    PDOCKQ_X0_New,
    PDOCKQ_B_New,
    PPV_VALUES_New,
    PDOCKQ_THRESHOLDS_New,
)

# ── Regression expected values (complex 1: A0A0B4J2C3_P24534) ────
EXPECTED_CA_COUNTS_1 = {'A': 197, 'B': 225}
EXPECTED_PDOCKQ_1 = 0.330137006509103
EXPECTED_PPV_1 = 0.83367787
EXPECTED_N_CONTACTS_1 = 92
EXPECTED_N_RESIDUES_A_1 = 197
EXPECTED_N_RESIDUES_B_1 = 225

# Hardcoded ATOM line for pure unit testing
ATOM_LINE = "ATOM      1  N   MET A   1       3.656  -2.938 -18.082  1.00 78.47           N  "


# ── Pure Unit Tests ───────────────────────────────────────────────

class TestParseAtmRecord:
    """Tests for PDB ATOM line parsing."""

    def test_returns_expected_keys(self):
        record = parse_atm_record_Edited(ATOM_LINE)
        expected_keys = {
            'name', 'atm_no', 'atm_name', 'atm_alt', 'res_name',
            'chain', 'res_no', 'insert', 'resid', 'x', 'y', 'z', 'occ', 'B',
        }
        assert set(record.keys()) == expected_keys

    def test_values_correct_types(self):
        record = parse_atm_record_Edited(ATOM_LINE)
        assert isinstance(record['atm_no'], int)
        assert isinstance(record['x'], float)
        assert isinstance(record['y'], float)
        assert isinstance(record['z'], float)
        assert isinstance(record['B'], float)
        assert isinstance(record['chain'], str)
        assert isinstance(record['res_name'], str)

    def test_parsed_values(self):
        record = parse_atm_record_Edited(ATOM_LINE)
        assert record['atm_no'] == 1
        assert record['res_name'].strip() == 'MET'
        assert record['chain'] == 'A'
        assert record['res_no'] == 1
        assert record['x'] == pytest.approx(3.656, abs=0.001)
        assert record['y'] == pytest.approx(-2.938, abs=0.001)
        assert record['z'] == pytest.approx(-18.082, abs=0.001)
        assert record['B'] == pytest.approx(78.47, abs=0.01)


class TestConstants:
    """Verify module-level constants."""

    def test_default_contact_threshold(self):
        assert DEFAULT_CONTACT_THRESHOLD_New == 8

    def test_sigmoid_params(self):
        assert PDOCKQ_L_New == 0.724
        assert PDOCKQ_K_New == 0.052
        assert PDOCKQ_X0_New == 152.611
        assert PDOCKQ_B_New == 0.018

    def test_ppv_and_threshold_arrays_same_length(self):
        assert len(PPV_VALUES_New) == len(PDOCKQ_THRESHOLDS_New)

    def test_ppv_values_in_range(self):
        assert np.all(PPV_VALUES_New >= 0)
        assert np.all(PPV_VALUES_New <= 1)


class TestContactResultDefaults:
    """Test ContactResult_New default values."""

    def test_default_construction(self):
        cr = ContactResult_New()
        assert cr.pdockq == 0.0
        assert cr.ppv == 0.0
        assert cr.n_interface_contacts == 0
        assert cr.contacts.shape == (0, 2)
        assert cr.contact_distances.shape == (0,)
        assert len(cr.interface_residues_a) == 0
        assert len(cr.interface_residues_b) == 0
        assert cr.avg_if_plddt is None


class TestLookupPPV:
    """Tests for _lookup_ppv_New."""

    def test_high_score(self):
        ppv = _lookup_ppv_New(0.7)
        assert 0 < ppv <= 1.0

    def test_low_score(self):
        ppv = _lookup_ppv_New(0.01)
        assert 0 < ppv <= 1.0

    def test_zero_score(self):
        ppv = _lookup_ppv_New(0.0)
        assert ppv >= 0


class TestPdockqNoContacts:
    """Test pDockQ with synthetic distant chains."""

    def test_no_contacts_returns_zero(self):
        chain_coords = {
            'A': np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            'B': np.array([[500.0, 500.0, 500.0], [501.0, 500.0, 500.0]]),
        }
        chain_plddt = {
            'A': np.array([80.0, 80.0]),
            'B': np.array([80.0, 80.0]),
        }
        pdockq, ppv = calc_pdockq_Edited(chain_coords, chain_plddt, t=8)
        assert pdockq == 0.0
        assert ppv == 0.0

    def test_no_contacts_contact_result(self):
        chain_coords = {
            'A': np.array([[0.0, 0.0, 0.0]]),
            'B': np.array([[999.0, 999.0, 999.0]]),
        }
        chain_plddt = {
            'A': np.array([90.0]),
            'B': np.array([90.0]),
        }
        cr = calc_pdockq_and_contacts_New(chain_coords, chain_plddt, t=8)
        assert cr.pdockq == 0.0
        assert cr.n_interface_contacts == 0


# ── Real Data Tests ───────────────────────────────────────────────

@pytest.mark.slow
class TestReadPdb:
    """Tests for read_pdb_Edited with real PDB files."""

    def test_returns_two_dicts(self, chain_coords_plddt_1):
        chain_coords, chain_plddt = chain_coords_plddt_1
        assert isinstance(chain_coords, dict)
        assert isinstance(chain_plddt, dict)

    def test_has_two_chains(self, chain_coords_plddt_1):
        chain_coords, chain_plddt = chain_coords_plddt_1
        assert len(chain_coords) == 2
        assert len(chain_plddt) == 2

    def test_chain_coords_shape(self, chain_coords_plddt_1):
        chain_coords, _ = chain_coords_plddt_1
        for ch, coords in chain_coords.items():
            assert coords.ndim == 2, f"Chain {ch} coords not 2D"
            assert coords.shape[1] == 3, f"Chain {ch} coords not (N, 3)"

    def test_plddt_shape_matches_coords(self, chain_coords_plddt_1):
        chain_coords, chain_plddt = chain_coords_plddt_1
        for ch in chain_coords:
            assert chain_plddt[ch].shape == (chain_coords[ch].shape[0],), \
                f"Chain {ch} pLDDT length mismatch"

    def test_plddt_range(self, chain_coords_plddt_1):
        _, chain_plddt = chain_coords_plddt_1
        for ch, plddt in chain_plddt.items():
            assert np.all(plddt >= 0), f"Chain {ch} has negative pLDDT"
            assert np.all(plddt <= 100), f"Chain {ch} has pLDDT > 100"


@pytest.mark.slow
class TestReadPdbWithResidueIds:
    """Tests for read_pdb_with_residue_ids_New."""

    def test_returns_four_items(self, ref_pdb_1):
        result = read_pdb_with_residue_ids_New(str(ref_pdb_1))
        assert len(result) == 4

    def test_consistent_lengths(self, ref_pdb_1):
        coords, plddt, res_nums, res_names = read_pdb_with_residue_ids_New(str(ref_pdb_1))
        for ch in coords:
            assert len(coords[ch]) == len(plddt[ch])
            assert len(coords[ch]) == len(res_nums[ch])
            assert len(coords[ch]) == len(res_names[ch])


@pytest.mark.slow
class TestChainInfo:
    """Tests for read_pdb_with_chain_info_New and ChainInfo_New."""

    def test_returns_dataclass(self, chain_info_1):
        assert isinstance(chain_info_1, ChainInfo_New)

    def test_has_two_chain_ids(self, chain_info_1):
        assert len(chain_info_1.chain_ids) == 2

    def test_ca_counts_positive(self, chain_info_1):
        for ch in chain_info_1.chain_ids:
            assert chain_info_1.ca_counts[ch] > 0, f"Chain {ch} has 0 CA atoms"

    def test_cb_coords_match_cb_plddt(self, chain_info_1):
        for ch in chain_info_1.chain_ids:
            assert len(chain_info_1.cb_coords[ch]) == len(chain_info_1.cb_plddt[ch]), \
                f"Chain {ch} CB coords/pLDDT mismatch"

    def test_cb_to_ca_map_valid(self, chain_info_1):
        for ch in chain_info_1.chain_ids:
            ca_count = chain_info_1.ca_counts[ch]
            for idx in chain_info_1.cb_to_ca_map[ch]:
                assert 0 <= idx < ca_count, \
                    f"Chain {ch} CB->CA index {idx} out of range [0, {ca_count})"

    @pytest.mark.regression
    def test_residue_counts_regression(self, chain_info_1):
        actual = dict(chain_info_1.ca_counts)
        assert actual == EXPECTED_CA_COUNTS_1


# ── pDockQ Calculation Tests ─────────────────────────────────────

@pytest.mark.slow
class TestCalcPdockq:
    """Tests for pDockQ calculation functions."""

    def test_calc_pdockq_returns_valid_range(self, chain_coords_plddt_1):
        chain_coords, chain_plddt = chain_coords_plddt_1
        pdockq, ppv = calc_pdockq_Edited(chain_coords, chain_plddt, t=8)
        assert 0 <= pdockq <= 0.75, f"pDockQ {pdockq} out of range"
        assert 0 <= ppv <= 1, f"PPV {ppv} out of range"

    def test_contact_result_populated(self, contact_result_1):
        cr = contact_result_1
        assert cr.n_interface_contacts > 0
        assert cr.avg_if_plddt is not None
        assert len(cr.interface_residues_a) > 0
        assert len(cr.interface_residues_b) > 0

    def test_contacts_shape(self, contact_result_1):
        assert contact_result_1.contacts.ndim == 2
        assert contact_result_1.contacts.shape[1] == 2
        assert contact_result_1.contacts.shape[0] == contact_result_1.n_interface_contacts

    def test_distances_positive(self, contact_result_1):
        assert np.all(contact_result_1.contact_distances > 0)

    def test_distances_within_threshold(self, contact_result_1):
        assert np.all(contact_result_1.contact_distances <= 8.0)

    def test_interface_residue_indices_within_bounds(self, contact_result_1):
        cr = contact_result_1
        if cr.interface_residues_a:
            assert max(cr.interface_residues_a) < cr.n_residues_a
        if cr.interface_residues_b:
            assert max(cr.interface_residues_b) < cr.n_residues_b

    def test_methods_agree(self, chain_coords_plddt_1):
        chain_coords, chain_plddt = chain_coords_plddt_1
        pdockq_simple, _ = calc_pdockq_Edited(chain_coords, chain_plddt, t=8)
        cr = calc_pdockq_and_contacts_New(chain_coords, chain_plddt, t=8)
        assert pdockq_simple == pytest.approx(cr.pdockq, abs=1e-10)

    def test_deterministic(self, chain_coords_plddt_1):
        chain_coords, chain_plddt = chain_coords_plddt_1
        cr1 = calc_pdockq_and_contacts_New(chain_coords, chain_plddt, t=8)
        cr2 = calc_pdockq_and_contacts_New(chain_coords, chain_plddt, t=8)
        assert cr1.pdockq == cr2.pdockq
        assert cr1.n_interface_contacts == cr2.n_interface_contacts
        assert np.array_equal(cr1.contacts, cr2.contacts)

    @pytest.mark.regression
    def test_pdockq_regression(self, contact_result_1):
        assert contact_result_1.pdockq == pytest.approx(EXPECTED_PDOCKQ_1, abs=1e-6)
        assert contact_result_1.ppv == pytest.approx(EXPECTED_PPV_1, abs=1e-4)
        assert contact_result_1.n_interface_contacts == EXPECTED_N_CONTACTS_1
        assert contact_result_1.n_residues_a == EXPECTED_N_RESIDUES_A_1
        assert contact_result_1.n_residues_b == EXPECTED_N_RESIDUES_B_1


# ── Multi-Chain & Chain Offset Tests ─────────────────────────────

@pytest.mark.slow
class TestChainOffsets:
    """Tests for compute_pae_chain_offsets_New."""

    def test_offsets_correct(self, chain_info_1, chain_offsets_1):
        ids = chain_info_1.chain_ids
        assert chain_offsets_1[ids[0]] == 0
        assert chain_offsets_1[ids[1]] == chain_info_1.ca_counts[ids[0]]


@pytest.mark.slow
class TestFindBestChainPair:
    """Tests for find_best_chain_pair_New."""

    def test_returns_valid_result(self, best_pair_1):
        ch_a, ch_b, cr = best_pair_1
        assert isinstance(ch_a, str)
        assert isinstance(ch_b, str)
        assert ch_a != ch_b
        assert isinstance(cr, ContactResult_New)

    def test_dimer_returns_only_pair(self, chain_info_1, best_pair_1):
        ch_a, ch_b, _ = best_pair_1
        assert set([ch_a, ch_b]) == set(chain_info_1.chain_ids)


@pytest.mark.slow
class TestHomodimer:
    """Tests specific to homodimer complexes."""

    def test_homodimer_produces_contacts(self, contact_result_homodimer):
        cr = contact_result_homodimer
        assert isinstance(cr, ContactResult_New)
        # Homodimer should still have distinct chain IDs (A, B)
        assert cr.chain_ids[0] != cr.chain_ids[1]
