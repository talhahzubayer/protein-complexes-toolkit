"""
Tests for interface_analysis.py - interface geometry, pLDDT, PAE, composite scoring.

Regression values captured on 2026-03-04 from reference complex 1
(A0A0B4J2C3_P24534) on Windows 11.
"""

import json

import numpy as np
import pytest

from interface_analysis import (
    compute_interface_geometry,
    compute_interface_plddt,
    extract_interface_pae,
    compute_interface_pae_features,
    identify_confident_interface_residues,
    compute_interface_confidence,
    compute_extended_flags,
    analyse_interface,
    analyse_interface_with_pae,
    analyse_interface_from_contact_result,
    build_interface_export_record,
    INTERFACE_PLDDT_HIGH,
    PAE_CONFIDENT_THRESHOLD,
    MIN_INTERFACE_CONTACTS,
    SPARSE_INTERFACE_DENSITY,
    ASYMMETRIC_INTERFACE_RATIO,
    SUBSTANTIAL_DISORDER_FRACTION,
    WEIGHT_PLDDT,
    WEIGHT_PAE,
    WEIGHT_SYMMETRY,
    WEIGHT_DENSITY,
    DENSITY_NORMALIZATION,
    PARADOX_IPTM_THRESHOLD,
    PARADOX_PDOCKQ_THRESHOLD,
    PARADOX_CONFIDENT_CONTACT_GENUINE,
    PARADOX_CONFIDENT_CONTACT_ARTEFACT,
    METRIC_DISAGREEMENT_THRESHOLD,
)
from pdockq import ContactResult_New as ContactResult

# ── Regression expected values (complex 1: A0A0B4J2C3_P24534) ────
EXPECTED_N_CONTACTS_1 = 92
EXPECTED_SYMMETRY_1 = 0.9654
EXPECTED_IF_PLDDT_COMBINED_1 = 75.0
EXPECTED_BULK_PLDDT_COMBINED_1 = 78.44
EXPECTED_DELTA_1 = -3.44
EXPECTED_HIGH_FRACTION_1 = 0.7195
EXPECTED_PAE_MEAN_1 = 9.79
EXPECTED_CONFIDENT_FRACTION_1 = 0.5978
EXPECTED_N_CONFIDENT_CONTACTS_1 = 55
EXPECTED_N_CONFIDENT_RES_A_1 = 20
EXPECTED_N_CONFIDENT_RES_B_1 = 14
EXPECTED_COMPOSITE_SCORE_1 = 0.7007


# ── Phase 1: Geometry Tests ──────────────────────────────────────

@pytest.mark.slow
class TestComputeInterfaceGeometry:
    """Tests for compute_interface_geometry."""

    def test_returns_all_keys(self, contact_result_1):
        geom = compute_interface_geometry(contact_result_1)
        expected_keys = {
            'n_interface_contacts', 'n_interface_residues_a', 'n_interface_residues_b',
            'n_interface_residues_total', 'interface_fraction_a', 'interface_fraction_b',
            'interface_symmetry', 'contacts_per_interface_residue',
            'mean_contact_distance', 'min_contact_distance',
        }
        assert set(geom.keys()) == expected_keys

    def test_fractions_in_range(self, contact_result_1):
        geom = compute_interface_geometry(contact_result_1)
        assert 0 <= geom['interface_fraction_a'] <= 1
        assert 0 <= geom['interface_fraction_b'] <= 1

    def test_symmetry_in_range(self, contact_result_1):
        geom = compute_interface_geometry(contact_result_1)
        assert 0 <= geom['interface_symmetry'] <= 1

    def test_density_positive(self, contact_result_1):
        geom = compute_interface_geometry(contact_result_1)
        assert geom['contacts_per_interface_residue'] > 0

    def test_residue_total_consistent(self, contact_result_1):
        geom = compute_interface_geometry(contact_result_1)
        assert geom['n_interface_residues_total'] == \
            geom['n_interface_residues_a'] + geom['n_interface_residues_b']

    def test_zero_contacts(self):
        cr = ContactResult()
        geom = compute_interface_geometry(cr)
        assert geom['n_interface_contacts'] == 0
        assert geom['mean_contact_distance'] is None
        assert geom['min_contact_distance'] is None
        assert geom['interface_symmetry'] == 0.0

    @pytest.mark.regression
    def test_geometry_regression(self, contact_result_1):
        geom = compute_interface_geometry(contact_result_1)
        assert geom['n_interface_contacts'] == EXPECTED_N_CONTACTS_1
        assert geom['interface_symmetry'] == pytest.approx(EXPECTED_SYMMETRY_1, abs=1e-3)


# ── Phase 1: Interface pLDDT Tests ───────────────────────────────

@pytest.mark.slow
class TestComputeInterfacePlddt:
    """Tests for compute_interface_plddt."""

    def test_returns_all_keys(self, contact_result_1):
        result = compute_interface_plddt(contact_result_1)
        expected_keys = {
            'interface_plddt_a', 'interface_plddt_b', 'interface_plddt_combined',
            'bulk_plddt_a', 'bulk_plddt_b', 'bulk_plddt_combined',
            'interface_vs_bulk_delta', 'interface_plddt_high_fraction',
        }
        assert set(result.keys()) == expected_keys

    def test_plddt_range(self, contact_result_1):
        result = compute_interface_plddt(contact_result_1)
        assert 0 <= result['interface_plddt_combined'] <= 100
        assert 0 <= result['bulk_plddt_combined'] <= 100

    def test_high_fraction_range(self, contact_result_1):
        result = compute_interface_plddt(contact_result_1)
        assert 0 <= result['interface_plddt_high_fraction'] <= 1

    def test_delta_consistency(self, contact_result_1):
        result = compute_interface_plddt(contact_result_1)
        expected_delta = result['interface_plddt_combined'] - result['bulk_plddt_combined']
        assert result['interface_vs_bulk_delta'] == pytest.approx(expected_delta, abs=0.01)

    def test_zero_contacts_all_none(self):
        cr = ContactResult()
        result = compute_interface_plddt(cr)
        for key, val in result.items():
            assert val is None, f"Key '{key}' should be None for zero contacts"

    @pytest.mark.regression
    def test_plddt_regression(self, contact_result_1):
        result = compute_interface_plddt(contact_result_1)
        assert result['interface_plddt_combined'] == pytest.approx(EXPECTED_IF_PLDDT_COMBINED_1, abs=0.01)
        assert result['bulk_plddt_combined'] == pytest.approx(EXPECTED_BULK_PLDDT_COMBINED_1, abs=0.01)
        assert result['interface_vs_bulk_delta'] == pytest.approx(EXPECTED_DELTA_1, abs=0.01)
        assert result['interface_plddt_high_fraction'] == pytest.approx(EXPECTED_HIGH_FRACTION_1, abs=1e-3)


# ── Phase 2: PAE Tests ───────────────────────────────────────────

@pytest.mark.slow
class TestExtractInterfacePae:
    """Tests for extract_interface_pae."""

    def test_returns_array(self, contact_result_1, pae_matrix_1, chain_info_1, chain_offsets_1):
        ch_a, ch_b = chain_info_1.chain_ids[0], chain_info_1.chain_ids[1]
        pae_vals = extract_interface_pae(
            contact_result_1, pae_matrix_1,
            chain_offsets=(chain_offsets_1[ch_a], chain_offsets_1[ch_b]),
            cb_to_ca_maps=(chain_info_1.cb_to_ca_map[ch_a], chain_info_1.cb_to_ca_map[ch_b]),
        )
        assert isinstance(pae_vals, np.ndarray)
        assert pae_vals.ndim == 1

    def test_length_matches_contacts(self, contact_result_1, pae_matrix_1, chain_info_1, chain_offsets_1):
        ch_a, ch_b = chain_info_1.chain_ids[0], chain_info_1.chain_ids[1]
        pae_vals = extract_interface_pae(
            contact_result_1, pae_matrix_1,
            chain_offsets=(chain_offsets_1[ch_a], chain_offsets_1[ch_b]),
            cb_to_ca_maps=(chain_info_1.cb_to_ca_map[ch_a], chain_info_1.cb_to_ca_map[ch_b]),
        )
        assert len(pae_vals) == contact_result_1.n_interface_contacts

    def test_none_for_no_contacts(self, pae_matrix_1):
        cr = ContactResult()
        result = extract_interface_pae(cr, pae_matrix_1, chain_lengths=(100, 100))
        assert result is None


@pytest.mark.slow
class TestComputeInterfacePaeFeatures:
    """Tests for compute_interface_pae_features."""

    def test_returns_all_keys(self, contact_result_1, pae_matrix_1, chain_info_1, chain_offsets_1):
        ch_a, ch_b = chain_info_1.chain_ids[0], chain_info_1.chain_ids[1]
        result = compute_interface_pae_features(
            contact_result_1, pae_matrix_1,
            chain_offsets=(chain_offsets_1[ch_a], chain_offsets_1[ch_b]),
            cb_to_ca_maps=(chain_info_1.cb_to_ca_map[ch_a], chain_info_1.cb_to_ca_map[ch_b]),
        )
        expected_keys = {
            'interface_pae_mean', 'interface_pae_median',
            'interface_pae_min', 'interface_pae_max',
            'n_confident_contacts', 'confident_contact_fraction',
            'cross_chain_pae_mean',
        }
        assert set(result.keys()) == expected_keys

    def test_ranges(self, contact_result_1, pae_matrix_1, chain_info_1, chain_offsets_1):
        ch_a, ch_b = chain_info_1.chain_ids[0], chain_info_1.chain_ids[1]
        result = compute_interface_pae_features(
            contact_result_1, pae_matrix_1,
            chain_offsets=(chain_offsets_1[ch_a], chain_offsets_1[ch_b]),
            cb_to_ca_maps=(chain_info_1.cb_to_ca_map[ch_a], chain_info_1.cb_to_ca_map[ch_b]),
        )
        assert result['interface_pae_mean'] > 0
        assert 0 <= result['confident_contact_fraction'] <= 1
        assert result['interface_pae_min'] <= result['interface_pae_mean'] <= result['interface_pae_max']

    @pytest.mark.regression
    def test_pae_features_regression(self, contact_result_1, pae_matrix_1, chain_info_1, chain_offsets_1):
        ch_a, ch_b = chain_info_1.chain_ids[0], chain_info_1.chain_ids[1]
        result = compute_interface_pae_features(
            contact_result_1, pae_matrix_1,
            chain_offsets=(chain_offsets_1[ch_a], chain_offsets_1[ch_b]),
            cb_to_ca_maps=(chain_info_1.cb_to_ca_map[ch_a], chain_info_1.cb_to_ca_map[ch_b]),
        )
        assert result['interface_pae_mean'] == pytest.approx(EXPECTED_PAE_MEAN_1, abs=0.01)
        assert result['confident_contact_fraction'] == pytest.approx(EXPECTED_CONFIDENT_FRACTION_1, abs=1e-3)
        assert result['n_confident_contacts'] == EXPECTED_N_CONFIDENT_CONTACTS_1


@pytest.mark.slow
class TestIdentifyConfidentResidues:
    """Tests for identify_confident_interface_residues."""

    def test_returns_expected_keys(self, contact_result_1, pae_matrix_1, chain_info_1, chain_offsets_1):
        ch_a, ch_b = chain_info_1.chain_ids[0], chain_info_1.chain_ids[1]
        result = identify_confident_interface_residues(
            contact_result_1, pae_matrix_1,
            chain_residue_numbers={ch_a: chain_info_1.chain_res_numbers[ch_a],
                                   ch_b: chain_info_1.chain_res_numbers[ch_b]},
            chain_offsets=(chain_offsets_1[ch_a], chain_offsets_1[ch_b]),
            cb_to_ca_maps=(chain_info_1.cb_to_ca_map[ch_a], chain_info_1.cb_to_ca_map[ch_b]),
        )
        assert 'n_confident_residues_a' in result
        assert 'n_confident_residues_b' in result
        assert 'confident_residue_indices_a' in result
        assert 'confident_residue_indices_b' in result

    def test_confident_subset_of_interface(self, contact_result_1, pae_matrix_1, chain_info_1, chain_offsets_1):
        ch_a, ch_b = chain_info_1.chain_ids[0], chain_info_1.chain_ids[1]
        result = identify_confident_interface_residues(
            contact_result_1, pae_matrix_1,
            chain_residue_numbers={ch_a: chain_info_1.chain_res_numbers[ch_a],
                                   ch_b: chain_info_1.chain_res_numbers[ch_b]},
            chain_offsets=(chain_offsets_1[ch_a], chain_offsets_1[ch_b]),
            cb_to_ca_maps=(chain_info_1.cb_to_ca_map[ch_a], chain_info_1.cb_to_ca_map[ch_b]),
        )
        confident_a = set(result['confident_residue_indices_a'])
        confident_b = set(result['confident_residue_indices_b'])
        assert confident_a <= contact_result_1.interface_residues_a
        assert confident_b <= contact_result_1.interface_residues_b

    @pytest.mark.regression
    def test_confident_residues_regression(self, contact_result_1, pae_matrix_1, chain_info_1, chain_offsets_1):
        ch_a, ch_b = chain_info_1.chain_ids[0], chain_info_1.chain_ids[1]
        result = identify_confident_interface_residues(
            contact_result_1, pae_matrix_1,
            chain_residue_numbers={ch_a: chain_info_1.chain_res_numbers[ch_a],
                                   ch_b: chain_info_1.chain_res_numbers[ch_b]},
            chain_offsets=(chain_offsets_1[ch_a], chain_offsets_1[ch_b]),
            cb_to_ca_maps=(chain_info_1.cb_to_ca_map[ch_a], chain_info_1.cb_to_ca_map[ch_b]),
        )
        assert result['n_confident_residues_a'] == EXPECTED_N_CONFIDENT_RES_A_1
        assert result['n_confident_residues_b'] == EXPECTED_N_CONFIDENT_RES_B_1


# ── Composite Score & Flags ──────────────────────────────────────

class TestComputeInterfaceConfidence:
    """Tests for compute_interface_confidence."""

    def test_weights_sum_to_one(self):
        total = WEIGHT_PLDDT + WEIGHT_PAE + WEIGHT_SYMMETRY + WEIGHT_DENSITY
        assert total == pytest.approx(1.0)

    def test_returns_float_in_range(self):
        metrics = {
            'interface_plddt_combined': 80.0,
            'confident_contact_fraction': 0.6,
            'interface_symmetry': 0.95,
            'contacts_per_interface_residue': 1.5,
        }
        score = compute_interface_confidence(metrics)
        assert isinstance(score, float)
        assert 0 <= score <= 1

    def test_none_when_missing_pae(self):
        metrics = {
            'interface_plddt_combined': 80.0,
            'confident_contact_fraction': None,
            'interface_symmetry': 0.95,
            'contacts_per_interface_residue': 1.5,
        }
        assert compute_interface_confidence(metrics) is None

    def test_none_when_missing_plddt(self):
        metrics = {
            'interface_plddt_combined': None,
            'confident_contact_fraction': 0.5,
            'interface_symmetry': 0.9,
            'contacts_per_interface_residue': 1.0,
        }
        assert compute_interface_confidence(metrics) is None

    @pytest.mark.slow
    @pytest.mark.regression
    def test_composite_score_regression(self, contact_result_1, pae_matrix_1, chain_info_1, chain_offsets_1):
        ch_a, ch_b = chain_info_1.chain_ids[0], chain_info_1.chain_ids[1]
        geom = compute_interface_geometry(contact_result_1)
        plddt = compute_interface_plddt(contact_result_1)
        pae_feats = compute_interface_pae_features(
            contact_result_1, pae_matrix_1,
            chain_offsets=(chain_offsets_1[ch_a], chain_offsets_1[ch_b]),
            cb_to_ca_maps=(chain_info_1.cb_to_ca_map[ch_a], chain_info_1.cb_to_ca_map[ch_b]),
        )
        all_metrics = {}
        all_metrics.update(geom)
        all_metrics.update(plddt)
        all_metrics.update(pae_feats)
        score = compute_interface_confidence(all_metrics)
        assert score == pytest.approx(EXPECTED_COMPOSITE_SCORE_1, abs=1e-3)


class TestComputeExtendedFlags:
    """Tests for compute_extended_flags."""

    def test_small_interface_flag(self):
        features = {'n_interface_contacts': 3}
        flags = compute_extended_flags(features)
        assert 'small_interface' in flags

    def test_metric_disagreement_flag(self):
        features = {'n_interface_contacts': 50}
        flags = compute_extended_flags(features, iptm=0.9, pdockq=0.3)
        assert 'metric_disagreement' in flags

    def test_no_metric_disagreement_when_close(self):
        features = {'n_interface_contacts': 50}
        flags = compute_extended_flags(features, iptm=0.6, pdockq=0.5)
        assert 'metric_disagreement' not in flags

    def test_paradox_genuine(self):
        features = {
            'n_interface_contacts': 50,
            'confident_contact_fraction': 0.8,  # > PARADOX_CONFIDENT_CONTACT_GENUINE (0.73)
        }
        flags = compute_extended_flags(
            features,
            iptm=0.8,      # > PARADOX_IPTM_THRESHOLD (0.75)
            pdockq=0.6,    # > PARADOX_PDOCKQ_THRESHOLD (0.5)
            disorder_fraction=0.4,  # > SUBSTANTIAL_DISORDER_FRACTION (0.3)
        )
        assert 'paradox_confident_disorder' in flags

    def test_paradox_artefactual(self):
        features = {
            'n_interface_contacts': 50,
            'confident_contact_fraction': 0.4,  # < PARADOX_CONFIDENT_CONTACT_ARTEFACT (0.50)
        }
        flags = compute_extended_flags(
            features,
            iptm=0.8,
            pdockq=0.6,
            disorder_fraction=0.4,
        )
        assert 'paradox_artefactual' in flags

    def test_no_paradox_below_disorder_threshold(self):
        features = {
            'n_interface_contacts': 50,
            'confident_contact_fraction': 0.7,
        }
        flags = compute_extended_flags(
            features,
            iptm=0.8, pdockq=0.6,
            disorder_fraction=0.1,  # below threshold
        )
        assert 'paradox_confident_disorder' not in flags
        assert 'paradox_artefactual' not in flags


# ── High-Level Entry Points ──────────────────────────────────────

@pytest.mark.slow
class TestAnalyseInterface:
    """Tests for top-level analyse_interface functions."""

    def test_analyse_interface_returns_dict(self, ref_pdb_1):
        result = analyse_interface(str(ref_pdb_1))
        assert isinstance(result, dict)
        assert 'n_interface_contacts' in result

    def test_analyse_interface_with_pae_adds_pae_keys(self, ref_pdb_1, pae_matrix_1):
        result = analyse_interface_with_pae(str(ref_pdb_1), pae_matrix_1)
        assert isinstance(result, dict)
        assert 'interface_pae_mean' in result

    def test_analyse_from_contact_result_phase1(self, contact_result_1):
        result = analyse_interface_from_contact_result(contact_result_1)
        assert 'n_interface_contacts' in result
        assert 'interface_plddt_combined' in result
        # No PAE keys without PAE matrix
        assert result.get('interface_pae_mean') is None

    def test_analyse_from_contact_result_phase2(self, contact_result_1, pae_matrix_1, chain_info_1, chain_offsets_1):
        ch_a, ch_b = chain_info_1.chain_ids[0], chain_info_1.chain_ids[1]
        result = analyse_interface_from_contact_result(
            contact_result_1,
            pae_matrix=pae_matrix_1,
            chain_offsets=(chain_offsets_1[ch_a], chain_offsets_1[ch_b]),
            cb_to_ca_maps=(chain_info_1.cb_to_ca_map[ch_a], chain_info_1.cb_to_ca_map[ch_b]),
        )
        assert 'interface_pae_mean' in result
        assert 'confident_contact_fraction' in result
        assert result['interface_pae_mean'] is not None


# ── Export Record ─────────────────────────────────────────────────

class TestBuildExportRecord:
    """Tests for build_interface_export_record."""

    def test_json_serialisable(self):
        record = build_interface_export_record(
            complex_name='P12345_Q67890',
            protein_a='P12345',
            protein_b='Q67890',
            quality_tier_v2='High',
            interface_confidence_score=0.82,
            confident_residue_numbers_a=[45, 46, 49],
            confident_residue_numbers_b=[112, 115],
            flags=['paradox_confident_disorder'],
            iptm=0.89,
            pdockq=0.61,
            n_interface_contacts=47,
            confident_contact_fraction=0.74,
            interface_plddt_combined=85.3,
        )
        # Should not raise
        serialised = json.dumps(record)
        assert isinstance(serialised, str)

    def test_contains_required_keys(self):
        record = build_interface_export_record(
            complex_name='X_Y',
            protein_a='X',
            protein_b='Y',
            quality_tier_v2='Medium',
            interface_confidence_score=0.5,
            confident_residue_numbers_a=[1, 2],
            confident_residue_numbers_b=[3],
            flags=[],
        )
        assert 'complex_name' in record
        assert 'protein_a' in record
        assert 'quality_tier_v2' in record
        assert 'confident_interface_residues_a' in record
        assert 'flags' in record
