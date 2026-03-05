"""
Cross-module integration tests spanning the full pipeline.

These tests verify that data flows correctly between modules:
  read_af2_nojax -> pdockq -> interface_analysis -> toolkit
"""

import csv
from pathlib import Path

import numpy as np
import pytest

from read_af2_nojax import load_pkl_without_jax, extract_metrics
from pdockq import (
    read_pdb_Edited,
    read_pdb_with_chain_info_New,
    calc_pdockq_and_contacts_New,
    compute_pae_chain_offsets_New,
    find_best_chain_pair_New,
    ContactResult_New,
)
from interface_analysis import (
    analyse_interface_from_contact_result,
    compute_interface_confidence,
    compute_interface_geometry,
    compute_interface_plddt,
    compute_interface_pae_features,
)
from toolkit import (
    parse_complex_name,
    process_single_complex,
    classify_prediction_quality_v2,
    write_results_csv,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]


class TestFullPipeline:
    """Test the complete pipeline from PKL+PDB to analysis output."""

    def test_pkl_to_metrics_to_pdockq_to_interface(
        self, ref_pdb_1, ref_pkl_1, loaded_pkl_1, pae_matrix_1,
        chain_info_1, chain_offsets_1, contact_result_1
    ):
        """Full pipeline for one complex - every module contributes."""
        # Step 1: PKL -> metrics
        metrics = extract_metrics(loaded_pkl_1)
        assert 'iptm' in metrics
        assert 'pae_mean' in metrics

        # Step 2: PDB -> chain info -> contacts
        assert contact_result_1.n_interface_contacts > 0

        # Step 3: Contacts + PAE -> interface analysis
        ch_a, ch_b = chain_info_1.chain_ids[0], chain_info_1.chain_ids[1]
        result = analyse_interface_from_contact_result(
            contact_result_1,
            pae_matrix=pae_matrix_1,
            chain_offsets=(chain_offsets_1[ch_a], chain_offsets_1[ch_b]),
            cb_to_ca_maps=(chain_info_1.cb_to_ca_map[ch_a], chain_info_1.cb_to_ca_map[ch_b]),
        )
        assert result['n_interface_contacts'] == contact_result_1.n_interface_contacts
        assert result['interface_pae_mean'] is not None
        assert result['interface_confidence_score'] is not None


class TestContactResultInterop:
    """Verify ContactResult flows correctly from pdockq to interface_analysis."""

    def test_pdockq_contact_result_accepted_by_interface(
        self, contact_result_1, pae_matrix_1, chain_info_1, chain_offsets_1
    ):
        ch_a, ch_b = chain_info_1.chain_ids[0], chain_info_1.chain_ids[1]
        result = analyse_interface_from_contact_result(
            contact_result_1,
            pae_matrix=pae_matrix_1,
            chain_offsets=(chain_offsets_1[ch_a], chain_offsets_1[ch_b]),
            cb_to_ca_maps=(chain_info_1.cb_to_ca_map[ch_a], chain_info_1.cb_to_ca_map[ch_b]),
        )
        assert isinstance(result, dict)
        assert 'confident_contact_fraction' in result


class TestCompositeToQualityV2:
    """Verify composite score correctly feeds quality v2 classification."""

    def test_interface_confidence_feeds_classify_v2(
        self, contact_result_1, pae_matrix_1, chain_info_1, chain_offsets_1
    ):
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
        assert score is not None

        tier = classify_prediction_quality_v2(0.6, 0.3, score)
        assert tier in ('High', 'Medium', 'Low')


class TestPaeDimensionsMatchChainInfo:
    """Verify PAE matrix dimensions are consistent with chain structure."""

    def test_pae_matches_ca_counts(self, pae_matrix_1, chain_info_1):
        total_ca = sum(chain_info_1.ca_counts.values())
        assert pae_matrix_1.shape[0] == total_ca, \
            f"PAE rows {pae_matrix_1.shape[0]} != total CA {total_ca}"
        assert pae_matrix_1.shape[1] == total_ca


class TestCsvRoundtrip:
    """Test process -> write CSV -> read back."""

    def test_roundtrip(self, test_output_dir, ref_pdb_1, ref_pkl_1):
        file_paths = {'pdb': ref_pdb_1, 'pkl': ref_pkl_1}
        row = process_single_complex(
            'A0A0B4J2C3_P24534', file_paths,
            run_interface=True, run_interface_pae=True
        )
        output = test_output_dir / "integration_roundtrip.csv"
        write_results_csv([row], str(output), include_interface=True, include_pae=True)

        import pandas as pd
        df = pd.read_csv(output)
        assert len(df) == 1
        assert df.iloc[0]['complex_name'] == 'A0A0B4J2C3_P24534'
        assert df.iloc[0]['iptm'] == pytest.approx(row['iptm'], abs=1e-4)
        assert df.iloc[0]['pdockq'] == pytest.approx(row['pdockq'], abs=1e-3)


class TestFindPairedAndProcess:
    """Test file discovery -> processing pipeline."""

    def test_find_and_process_two(self, test_data_dir):
        from toolkit import find_paired_data_files
        pairs = find_paired_data_files(str(test_data_dir))
        assert len(pairs) >= 2

        processed = 0
        for name, file_paths in sorted(pairs.items())[:2]:
            if 'pdb' in file_paths and 'pkl' in file_paths:
                row = process_single_complex(name, file_paths)
                assert row['complex_name'] == name
                assert 'quality_tier' in row
                processed += 1

        assert processed == 2


class TestNamingConventionsAgree:
    """Verify old and new naming conventions produce the same complex name."""

    def test_old_and_new_agree(self):
        name_old, _, _, _ = parse_complex_name("A0A0A0MQZ0_P40933.pdb")
        name_new, _, _, _ = parse_complex_name(
            "A0A0A0MQZ0_P40933_relaxed_model_1_multimer_v3_pred_0.pdb"
        )
        assert name_old == name_new


class TestCbToCaMapNonTrivial:
    """Verify CB->CA maps are functional in PAE extraction."""

    def test_cb_to_ca_map_used(self, chain_info_1, contact_result_1, pae_matrix_1, chain_offsets_1):
        ch_a, ch_b = chain_info_1.chain_ids[0], chain_info_1.chain_ids[1]
        map_a = chain_info_1.cb_to_ca_map[ch_a]
        map_b = chain_info_1.cb_to_ca_map[ch_b]
        # The maps should have entries for each CB atom
        assert len(map_a) > 0
        assert len(map_b) > 0
        # All entries should be valid CA indices
        for idx in map_a:
            assert 0 <= idx < chain_info_1.ca_counts[ch_a]
        for idx in map_b:
            assert 0 <= idx < chain_info_1.ca_counts[ch_b]
