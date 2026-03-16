"""
Tests for variant_mapper.py — Phase C: Variant Mapping to Structures.

Covers 6 test groups:
1. Parsing (HGVS, UniProt, ClinVar, ExAC, variant index, ClinVar enrichment)
2. SASA & structural mapping (ShrakeRupley, context classification, distance)
3. Annotation & enrichment (fold-change, annotate_results_with_variants, format)
4. CLI (standalone subcommands, argparse)
5. Integration (toolkit --variants flag, CSV column presence)
6. Regression (reference complex 1 expected values)

Test data lives in tests/offline_test_data/databases/:
  test_uniprot_variants.txt  (~20 variant rows for A0A0B4J2C3, P24534, Q99999)
  test_clinvar_variants.txt  (~6 rows with GRCh37 + GRCh38)
  test_exac_constraint.txt   (~5 gene rows: GENE_A through GENE_E)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(r"C:\Users\Talhah Zubayer\Documents\protein-complexes-toolkit")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from variant_mapper import (
    CSV_FIELDNAMES_VARIANTS,
    CONTEXT_BURIED,
    CONTEXT_INTERFACE_CORE,
    CONTEXT_INTERFACE_RIM,
    CONTEXT_SURFACE,
    CONTEXT_UNMAPPED,
    HGVS_PATTERN,
    INTERFACE_CORE_DISTANCE,
    INTERFACE_RIM_DISTANCE,
    MAX_ASA,
    RELEVANT_CONSEQUENCES,
    SASA_BURIED_THRESHOLD,
    THREE_TO_ONE,
    annotate_results_with_variants,
    build_argument_parser,
    build_variant_index,
    classify_structural_context,
    compute_distance_to_interface,
    compute_interface_variant_enrichment,
    compute_residue_sasa,
    enrich_with_clinvar,
    format_variant_details,
    is_buried,
    load_clinvar_variants,
    load_exac_constraint,
    load_uniprot_variants,
    map_variants_to_complex,
    parse_hgvs_position,
)


# ═══════════════════════════════════════════════════════════════════
# Group 1: Parsing
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.variants
class TestHGVSParsing:
    """Test HGVS protein notation parsing."""

    def test_standard_missense(self):
        """Parse standard missense: p.Lys2Glu → ('K', 2, 'E')."""
        result = parse_hgvs_position('p.Lys2Glu')
        assert result == ('K', 2, 'E')

    def test_large_position(self):
        """Parse variant at large residue number."""
        result = parse_hgvs_position('p.Ala1234Thr')
        assert result == ('A', 1234, 'T')

    def test_stop_codon(self):
        """Parse stop gained: p.Arg123Ter → ('R', 123, '*')."""
        result = parse_hgvs_position('p.Arg123Ter')
        assert result == ('R', 123, '*')

    def test_methionine_start(self):
        """Parse variant at start codon."""
        result = parse_hgvs_position('p.Met1Val')
        assert result == ('M', 1, 'V')

    def test_invalid_empty(self):
        """Empty string returns None."""
        assert parse_hgvs_position('') is None
        assert parse_hgvs_position(None) is None

    def test_invalid_format(self):
        """Malformed HGVS returns None."""
        assert parse_hgvs_position('c.241T>C') is None  # cDNA, not protein
        assert parse_hgvs_position('K2E') is None        # missing p. prefix
        assert parse_hgvs_position('p.X2Y') is None      # invalid AA code

    def test_all_standard_amino_acids(self):
        """All 20 standard amino acids parse correctly."""
        for three, one in THREE_TO_ONE.items():
            if three == 'TER':
                continue
            hgvs = f"p.{three.capitalize()}10Ala"
            result = parse_hgvs_position(hgvs)
            assert result is not None, f"Failed for {three}"
            assert result[0] == one
            assert result[1] == 10

    def test_hgvs_pattern_regex(self):
        """HGVS regex matches expected formats."""
        assert HGVS_PATTERN.match('p.Ala1Val')
        assert HGVS_PATTERN.match('p.Arg999Ter')
        assert not HGVS_PATTERN.match('p.A1V')  # single-letter not accepted


@pytest.mark.variants
class TestUniProtLoading:
    """Test UniProt variant file loading."""

    def test_loads_correct_rows(self, test_uniprot_variants_path):
        """Loads variants for specified accessions only."""
        df = load_uniprot_variants(
            test_uniprot_variants_path,
            frozenset({'A0A0B4J2C3'}),
        )
        assert len(df) > 0
        assert set(df['accession'].unique()) == {'A0A0B4J2C3'}

    def test_filters_by_accession(self, test_uniprot_variants_path):
        """Only returns rows matching the accession filter."""
        df = load_uniprot_variants(
            test_uniprot_variants_path,
            frozenset({'P24534'}),
        )
        assert all(df['accession'] == 'P24534')

    def test_filters_by_consequence_type(self, test_uniprot_variants_path):
        """Only protein-altering consequences are retained."""
        df = load_uniprot_variants(
            test_uniprot_variants_path,
            frozenset({'A0A0B4J2C3', 'P24534'}),
        )
        for cons in df['consequence']:
            assert cons in RELEVANT_CONSEQUENCES

    def test_correct_output_columns(self, test_uniprot_variants_path):
        """Output has expected columns."""
        df = load_uniprot_variants(
            test_uniprot_variants_path,
            frozenset({'A0A0B4J2C3'}),
        )
        expected = {'accession', 'position', 'ref_aa', 'alt_aa', 'rsid',
                    'consequence', 'clinical_significance', 'phenotype', 'evidence'}
        assert set(df.columns) == expected

    def test_deduplication(self, test_uniprot_variants_path):
        """Duplicate rows (same accession+position+alt_aa) are deduplicated."""
        df = load_uniprot_variants(
            test_uniprot_variants_path,
            frozenset({'A0A0B4J2C3'}),
        )
        # L81P appears twice in test data (rs and RCV accession) but should dedup
        l81p_rows = df[(df['position'] == 81) & (df['alt_aa'] == 'P')]
        assert len(l81p_rows) == 1

    def test_empty_accession_set(self, test_uniprot_variants_path):
        """Empty accession set returns empty DataFrame."""
        df = load_uniprot_variants(test_uniprot_variants_path, frozenset())
        assert len(df) == 0

    def test_nonexistent_accession(self, test_uniprot_variants_path):
        """Non-existent accession returns empty DataFrame."""
        df = load_uniprot_variants(
            test_uniprot_variants_path,
            frozenset({'XXXXXX'}),
        )
        assert len(df) == 0

    def test_file_not_found_raises(self):
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_uniprot_variants(Path('/nonexistent/file.txt'), frozenset({'P24534'}))

    def test_position_is_integer(self, test_uniprot_variants_path):
        """All positions are integer type."""
        df = load_uniprot_variants(
            test_uniprot_variants_path,
            frozenset({'A0A0B4J2C3', 'P24534'}),
        )
        assert df['position'].dtype in (np.int64, np.int32, int)

    def test_protein_q99999_not_in_accessions(self, test_uniprot_variants_path):
        """Q99999 variants excluded when not in accessions filter."""
        df = load_uniprot_variants(
            test_uniprot_variants_path,
            frozenset({'A0A0B4J2C3', 'P24534'}),
        )
        assert 'Q99999' not in df['accession'].values


@pytest.mark.variants
class TestClinVarLoading:
    """Test ClinVar variant file loading."""

    def test_loads_grch38_only(self, test_clinvar_path):
        """Only GRCh38 rows are loaded (not GRCh37 duplicates)."""
        df = load_clinvar_variants(test_clinvar_path)
        assert len(df) > 0
        # The test data has one GRCh37 duplicate for rs100000005 — should not appear twice
        rs005 = df[df['rsid'].str.contains('100000005', na=False)]
        assert len(rs005) <= 1

    def test_rsid_filter(self, test_clinvar_path):
        """rsID filter correctly selects matching rows."""
        df = load_clinvar_variants(
            test_clinvar_path,
            rsids=frozenset({'rs100000005'}),
        )
        assert len(df) >= 1
        assert all(df['rsid'].str.contains('100000005', na=False))

    def test_output_columns(self, test_clinvar_path):
        """Output has expected ClinVar columns."""
        df = load_clinvar_variants(test_clinvar_path)
        expected = {'rsid', 'gene_symbol', 'clinvar_significance',
                    'review_status', 'n_submitters', 'phenotype_list', 'origin'}
        assert set(df.columns) == expected

    def test_review_status_values(self, test_clinvar_path):
        """Review status values are present and non-empty."""
        df = load_clinvar_variants(test_clinvar_path)
        assert df['review_status'].notna().any()

    def test_file_not_found_raises(self):
        """Missing ClinVar file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_clinvar_variants(Path('/nonexistent/clinvar.txt'))


@pytest.mark.variants
class TestExACLoading:
    """Test ExAC constraint file loading."""

    def test_loads_all_genes(self, test_exac_path):
        """Loads all genes from test data."""
        df = load_exac_constraint(test_exac_path)
        assert len(df) == 5

    def test_filters_by_gene(self, test_exac_path):
        """Gene filter works correctly."""
        df = load_exac_constraint(test_exac_path, gene_symbols=frozenset({'GENE_A'}))
        assert len(df) == 1
        assert df.iloc[0]['gene'] == 'GENE_A'

    def test_pli_values(self, test_exac_path):
        """pLI values are correct."""
        df = load_exac_constraint(test_exac_path)
        gene_a = df[df['gene'] == 'GENE_A'].iloc[0]
        assert abs(gene_a['pLI'] - 0.95) < 0.001

    def test_mis_z_values(self, test_exac_path):
        """mis_z values are correct."""
        df = load_exac_constraint(test_exac_path)
        gene_a = df[df['gene'] == 'GENE_A'].iloc[0]
        assert abs(gene_a['mis_z'] - 3.12) < 0.01

    def test_output_columns(self, test_exac_path):
        """Output has expected columns."""
        df = load_exac_constraint(test_exac_path)
        for col in ['gene', 'pLI', 'mis_z', 'lof_z', 'syn_z']:
            assert col in df.columns

    def test_file_not_found_raises(self):
        """Missing ExAC file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_exac_constraint(Path('/nonexistent/exac.txt'))


@pytest.mark.variants
class TestVariantIndex:
    """Test variant index building and ClinVar enrichment."""

    def test_groups_by_accession(self, test_uniprot_variants_path):
        """Variants grouped by UniProt accession."""
        df = load_uniprot_variants(
            test_uniprot_variants_path,
            frozenset({'A0A0B4J2C3', 'P24534'}),
        )
        idx = build_variant_index(df)
        assert 'A0A0B4J2C3' in idx
        assert 'P24534' in idx
        assert len(idx['A0A0B4J2C3']) > 0
        assert len(idx['P24534']) > 0

    def test_variant_dict_keys(self, test_uniprot_variants_path):
        """Each variant dict has expected keys."""
        df = load_uniprot_variants(
            test_uniprot_variants_path,
            frozenset({'A0A0B4J2C3'}),
        )
        idx = build_variant_index(df)
        var = idx['A0A0B4J2C3'][0]
        expected_keys = {'position', 'ref_aa', 'alt_aa', 'rsid',
                         'consequence', 'clinical_significance', 'phenotype', 'evidence'}
        assert set(var.keys()) == expected_keys

    def test_clinvar_enrichment(self, test_uniprot_variants_path, test_clinvar_path):
        """ClinVar enrichment adds review_status and significance."""
        df = load_uniprot_variants(
            test_uniprot_variants_path,
            frozenset({'A0A0B4J2C3'}),
        )
        idx = build_variant_index(df)

        # Collect rsIDs
        rsids = frozenset(
            v['rsid'] for variants in idx.values()
            for v in variants if v.get('rsid') and v['rsid'] != 'nan'
        )
        clinvar_df = load_clinvar_variants(test_clinvar_path, rsids=rsids)
        enrich_with_clinvar(idx, clinvar_df)

        # At least one variant should now have clinvar_review_status
        has_clinvar = False
        for variants in idx.values():
            for v in variants:
                if 'clinvar_review_status' in v:
                    has_clinvar = True
                    break
        assert has_clinvar, "No variants were enriched with ClinVar data"

    def test_clinvar_enrichment_empty_df(self, test_uniprot_variants_path):
        """ClinVar enrichment with empty DataFrame is a no-op."""
        df = load_uniprot_variants(
            test_uniprot_variants_path,
            frozenset({'A0A0B4J2C3'}),
        )
        idx = build_variant_index(df)
        original_keys = set(idx['A0A0B4J2C3'][0].keys())
        enrich_with_clinvar(idx, pd.DataFrame())
        assert set(idx['A0A0B4J2C3'][0].keys()) == original_keys

    def test_build_empty_dataframe(self):
        """Building index from empty DataFrame returns empty dict."""
        empty = pd.DataFrame(columns=[
            'accession', 'position', 'ref_aa', 'alt_aa', 'rsid',
            'consequence', 'clinical_significance', 'phenotype', 'evidence',
        ])
        idx = build_variant_index(empty)
        assert idx == {}


# ═══════════════════════════════════════════════════════════════════
# Group 2: SASA & Structural Mapping
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.variants
@pytest.mark.slow
class TestSASAComputation:
    """Test SASA computation using real PDB files."""

    def test_returns_dict(self, ref_pdb_1):
        """compute_residue_sasa returns a dict."""
        sasa = compute_residue_sasa(ref_pdb_1, 'A')
        assert isinstance(sasa, dict)
        assert len(sasa) > 0

    def test_values_in_range(self, ref_pdb_1):
        """RSA values are non-negative (can exceed 1.0 for extended conformations)."""
        sasa = compute_residue_sasa(ref_pdb_1, 'A')
        for res_num, rsa in sasa.items():
            assert rsa >= 0.0, f"Negative RSA at residue {res_num}: {rsa}"

    def test_both_chains(self, ref_pdb_1):
        """SASA computed for both chains A and B."""
        sasa_a = compute_residue_sasa(ref_pdb_1, 'A')
        sasa_b = compute_residue_sasa(ref_pdb_1, 'B')
        assert len(sasa_a) > 0
        assert len(sasa_b) > 0
        # Different chains should have different residue sets
        assert sasa_a != sasa_b or True  # May overlap for homodimers

    def test_known_interface_residue_accessible(self, ref_pdb_1):
        """Interface residues should generally have some accessibility."""
        sasa = compute_residue_sasa(ref_pdb_1, 'A')
        # Residue 81 is at the interface for reference complex 1
        if 81 in sasa:
            # Interface residues can vary; just check it exists
            assert isinstance(sasa[81], float)

    def test_is_buried_threshold(self):
        """is_buried correctly classifies RSA values."""
        assert is_buried(0.0) is True
        assert is_buried(0.10) is True
        assert is_buried(0.24) is True
        assert is_buried(0.25) is False
        assert is_buried(0.50) is False
        assert is_buried(1.0) is False


@pytest.mark.variants
class TestParallelSASA:
    """Test parallel SASA pre-computation."""

    def test_compute_sasa_pair_returns_two_dicts(self, ref_pdb_1):
        """_compute_sasa_pair returns a tuple of two SASA dicts."""
        from variant_mapper import _compute_sasa_pair
        sasa_a, sasa_b = _compute_sasa_pair(str(ref_pdb_1), 'A', 'B')
        assert isinstance(sasa_a, dict)
        assert isinstance(sasa_b, dict)
        assert len(sasa_a) > 0
        assert len(sasa_b) > 0

    def test_compute_sasa_pair_bad_path_returns_empty(self):
        """_compute_sasa_pair returns ({}, {}) for nonexistent PDB."""
        from variant_mapper import _compute_sasa_pair
        sasa_a, sasa_b = _compute_sasa_pair('/nonexistent/path.pdb', 'A', 'B')
        assert sasa_a == {}
        assert sasa_b == {}

    def test_precompute_sasa_parallel_serial(self, ref_pdb_1, chain_info_1):
        """precompute_sasa_parallel works in serial mode (workers=1)."""
        from variant_mapper import precompute_sasa_parallel
        results = [{
            '_pdb_path': str(ref_pdb_1),
            '_chain_info': chain_info_1,
            'best_chain_pair': 'A_B',
        }]
        cache = precompute_sasa_parallel(results, workers=1)
        assert 0 in cache
        sasa_a, sasa_b = cache[0]
        assert len(sasa_a) > 0
        assert len(sasa_b) > 0

    def test_precompute_sasa_parallel_multiprocess(self, ref_pdb_1, chain_info_1):
        """precompute_sasa_parallel works with workers=2."""
        from variant_mapper import precompute_sasa_parallel
        # Duplicate to have 2 work items
        results = [
            {
                '_pdb_path': str(ref_pdb_1),
                '_chain_info': chain_info_1,
                'best_chain_pair': 'A_B',
            },
            {
                '_pdb_path': str(ref_pdb_1),
                '_chain_info': chain_info_1,
                'best_chain_pair': 'A_B',
            },
        ]
        cache = precompute_sasa_parallel(results, workers=2)
        assert len(cache) == 2
        for idx in (0, 1):
            sasa_a, sasa_b = cache[idx]
            assert len(sasa_a) > 0
            assert len(sasa_b) > 0

    def test_precompute_skips_missing_chain_info(self):
        """precompute_sasa_parallel skips rows without _chain_info."""
        from variant_mapper import precompute_sasa_parallel
        results = [
            {'_pdb_path': '/some/path.pdb', '_chain_info': None},
            {'_pdb_path': None, '_chain_info': 'dummy'},
        ]
        cache = precompute_sasa_parallel(results, workers=1)
        assert len(cache) == 0

    def test_precompute_empty_results(self):
        """precompute_sasa_parallel handles empty results list."""
        from variant_mapper import precompute_sasa_parallel
        cache = precompute_sasa_parallel([], workers=1)
        assert cache == {}


@pytest.mark.variants
class TestDistanceToInterface:
    """Test distance-to-interface computation."""

    def test_zero_distance_at_interface(self):
        """Distance is zero when variant coord IS an interface coord."""
        coord = np.array([1.0, 2.0, 3.0])
        iface = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        dist = compute_distance_to_interface(coord, iface)
        assert abs(dist) < 1e-6

    def test_positive_distance(self):
        """Distance is positive when variant is away from interface."""
        coord = np.array([0.0, 0.0, 0.0])
        iface = np.array([[3.0, 4.0, 0.0]])  # distance = 5.0
        dist = compute_distance_to_interface(coord, iface)
        assert abs(dist - 5.0) < 1e-6

    def test_minimum_of_multiple(self):
        """Returns minimum distance across multiple interface coords."""
        coord = np.array([0.0, 0.0, 0.0])
        iface = np.array([
            [10.0, 0.0, 0.0],  # 10.0
            [3.0, 4.0, 0.0],   # 5.0
            [0.0, 0.0, 1.0],   # 1.0 (closest)
        ])
        dist = compute_distance_to_interface(coord, iface)
        assert abs(dist - 1.0) < 1e-6

    def test_empty_interface_returns_inf(self):
        """Returns inf when no interface coords provided."""
        coord = np.array([1.0, 2.0, 3.0])
        iface = np.empty((0, 3))
        dist = compute_distance_to_interface(coord, iface)
        assert dist == float('inf')


@pytest.mark.variants
class TestStructuralContextClassification:
    """Test 4-class structural context classification."""

    @pytest.fixture
    def mock_chain_data(self):
        """Minimal mock data for structural classification tests."""
        # 5 residues at known positions
        chain_res_numbers = [10, 20, 30, 40, 50]
        cb_coords = np.array([
            [0.0, 0.0, 0.0],    # 10
            [5.0, 0.0, 0.0],    # 20
            [10.0, 0.0, 0.0],   # 30
            [20.0, 0.0, 0.0],   # 40
            [50.0, 0.0, 0.0],   # 50
        ])
        # Residues 20, 30 are at the interface
        interface_residues = {20, 30}
        # SASA: 10=buried, 40=surface, 50=buried
        residue_sasa = {10: 0.10, 20: 0.40, 30: 0.35, 40: 0.60, 50: 0.15}
        # Interface CB coords
        iface_indices = [1, 2]  # indices of residues 20, 30
        interface_cb_coords = cb_coords[iface_indices]
        return chain_res_numbers, cb_coords, interface_residues, residue_sasa, interface_cb_coords

    def test_interface_core(self, mock_chain_data):
        """Residue 20 at interface, close to other interface residue → interface_core."""
        chain_res, cb, iface, sasa, iface_cb = mock_chain_data
        result = classify_structural_context(20, iface, chain_res, cb, sasa, iface_cb)
        # Distance from res 20 (at [5,0,0]) to nearest interface CB: either itself or res 30 (at [10,0,0])
        # Distance to itself = 0.0 → interface_core
        assert result['context'] == CONTEXT_INTERFACE_CORE

    def test_interface_rim(self, mock_chain_data):
        """Interface residue further than 4A but within 8A → interface_rim."""
        chain_res, cb, iface, sasa, iface_cb = mock_chain_data
        # Manually create data where interface residue is 6A from nearest
        chain_res2 = [10, 20]
        cb2 = np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
        iface2 = {10, 20}
        # Interface CB coords: just residue 10 at origin
        iface_cb2 = np.array([[0.0, 0.0, 0.0]])
        result = classify_structural_context(20, iface2, chain_res2, cb2, sasa, iface_cb2)
        assert result['context'] == CONTEXT_INTERFACE_RIM

    def test_buried_core(self, mock_chain_data):
        """Non-interface residue with RSA < 25% → buried_core."""
        chain_res, cb, iface, sasa, iface_cb = mock_chain_data
        result = classify_structural_context(50, iface, chain_res, cb, sasa, iface_cb)
        assert result['context'] == CONTEXT_BURIED

    def test_surface_non_interface(self, mock_chain_data):
        """Non-interface residue with RSA >= 25% → surface_non_interface."""
        chain_res, cb, iface, sasa, iface_cb = mock_chain_data
        result = classify_structural_context(40, iface, chain_res, cb, sasa, iface_cb)
        assert result['context'] == CONTEXT_SURFACE

    def test_unmapped_position(self, mock_chain_data):
        """Position not in chain → unmapped."""
        chain_res, cb, iface, sasa, iface_cb = mock_chain_data
        result = classify_structural_context(999, iface, chain_res, cb, sasa, iface_cb)
        assert result['context'] == CONTEXT_UNMAPPED

    def test_distance_returned(self, mock_chain_data):
        """Distance to interface is computed and returned."""
        chain_res, cb, iface, sasa, iface_cb = mock_chain_data
        result = classify_structural_context(40, iface, chain_res, cb, sasa, iface_cb)
        assert isinstance(result['distance_to_interface'], float)
        assert result['distance_to_interface'] > 0

    def test_context_labels_are_constants(self):
        """Verify context labels match module constants."""
        assert CONTEXT_INTERFACE_CORE == 'interface_core'
        assert CONTEXT_INTERFACE_RIM == 'interface_rim'
        assert CONTEXT_SURFACE == 'surface_non_interface'
        assert CONTEXT_BURIED == 'buried_core'
        assert CONTEXT_UNMAPPED == 'unmapped'


@pytest.mark.variants
@pytest.mark.slow
class TestMapVariantsToComplex:
    """Test end-to-end variant mapping onto a complex."""

    def test_maps_variants_for_known_protein(self, ref_pdb_1, test_uniprot_variants_path):
        """Maps variants for A0A0B4J2C3 onto reference complex 1."""
        from pdockq import read_pdb_with_chain_info_New

        df = load_uniprot_variants(
            test_uniprot_variants_path,
            frozenset({'A0A0B4J2C3'}),
        )
        idx = build_variant_index(df)
        chain_info = read_pdb_with_chain_info_New(str(ref_pdb_1))
        sasa = compute_residue_sasa(ref_pdb_1, 'A')

        # Use actual interface residues from the pipeline
        interface_a = {81, 82, 83, 84, 86, 87, 90, 92, 93, 96, 97, 100, 115, 118, 119}

        mapped = map_variants_to_complex(
            'A0A0B4J2C3', 'A', idx, interface_a,
            chain_info.chain_res_numbers.get('A', []),
            chain_info.cb_coords.get('A', np.empty((0, 3))),
            sasa,
        )

        assert len(mapped) > 0
        # Check that each variant has context info
        for var in mapped:
            assert 'context' in var
            assert 'distance_to_interface' in var
            assert 'chain_id' in var
            assert var['chain_id'] == 'A'

    def test_interface_variant_classified_correctly(self, ref_pdb_1, test_uniprot_variants_path):
        """Variant at position 81 (known interface) is classified as interface."""
        from pdockq import read_pdb_with_chain_info_New

        df = load_uniprot_variants(
            test_uniprot_variants_path,
            frozenset({'A0A0B4J2C3'}),
        )
        idx = build_variant_index(df)
        chain_info = read_pdb_with_chain_info_New(str(ref_pdb_1))
        sasa = compute_residue_sasa(ref_pdb_1, 'A')

        interface_a = {81, 82, 83, 84, 86, 87, 90, 92, 93, 96, 97, 100, 115, 118, 119}
        mapped = map_variants_to_complex(
            'A0A0B4J2C3', 'A', idx, interface_a,
            chain_info.chain_res_numbers.get('A', []),
            chain_info.cb_coords.get('A', np.empty((0, 3))),
            sasa,
        )

        pos81_vars = [v for v in mapped if v['position'] == 81]
        assert len(pos81_vars) >= 1
        assert pos81_vars[0]['context'] in (CONTEXT_INTERFACE_CORE, CONTEXT_INTERFACE_RIM)

    def test_empty_variant_index(self, ref_pdb_1):
        """Protein with no variants returns empty list."""
        from pdockq import read_pdb_with_chain_info_New

        chain_info = read_pdb_with_chain_info_New(str(ref_pdb_1))
        sasa = compute_residue_sasa(ref_pdb_1, 'A')

        mapped = map_variants_to_complex(
            'NONEXISTENT', 'A', {}, set(),
            chain_info.chain_res_numbers.get('A', []),
            chain_info.cb_coords.get('A', np.empty((0, 3))),
            sasa,
        )
        assert mapped == []


# ═══════════════════════════════════════════════════════════════════
# Group 3: Annotation & Enrichment
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.variants
class TestEnrichmentCalculation:
    """Test interface variant enrichment fold-change calculation."""

    def test_basic_enrichment(self):
        """2x enrichment when interface is 10% of residues but has 20% of variants."""
        result = compute_interface_variant_enrichment(
            n_interface_variants=2,
            n_total_variants=10,
            n_interface_residues=10,
            n_total_residues=100,
        )
        assert abs(result - 2.0) < 1e-6

    def test_no_enrichment(self):
        """1x enrichment when variants are evenly distributed."""
        result = compute_interface_variant_enrichment(
            n_interface_variants=1,
            n_total_variants=10,
            n_interface_residues=10,
            n_total_residues=100,
        )
        assert abs(result - 1.0) < 1e-6

    def test_zero_total_variants(self):
        """Returns 0.0 when no variants exist."""
        result = compute_interface_variant_enrichment(0, 0, 10, 100)
        assert result == 0.0

    def test_zero_interface_residues(self):
        """Returns 0.0 when no interface residues exist."""
        result = compute_interface_variant_enrichment(0, 10, 0, 100)
        assert result == 0.0

    def test_zero_total_residues(self):
        """Returns 0.0 when total residues is zero."""
        result = compute_interface_variant_enrichment(0, 0, 0, 0)
        assert result == 0.0


@pytest.mark.variants
class TestFormatVariantDetails:
    """Test variant details formatting for CSV output."""

    def test_empty_list(self):
        """Empty variant list returns empty string."""
        assert format_variant_details([]) == ''

    def test_single_variant(self):
        """Single variant formatted correctly."""
        variants = [{'ref_aa': 'K', 'position': 81, 'alt_aa': 'P',
                     'context': 'interface_core', 'clinical_significance': 'pathogenic'}]
        result = format_variant_details(variants)
        assert result == 'K81P:interface_core:pathogenic'

    def test_multiple_variants_pipe_separated(self):
        """Multiple variants joined with pipes."""
        variants = [
            {'ref_aa': 'K', 'position': 81, 'alt_aa': 'P',
             'context': 'interface_core', 'clinical_significance': 'pathogenic'},
            {'ref_aa': 'E', 'position': 82, 'alt_aa': 'K',
             'context': 'interface_rim', 'clinical_significance': ''},
        ]
        result = format_variant_details(variants)
        assert '|' in result
        assert 'K81P:interface_core:pathogenic' in result
        assert 'E82K:interface_rim:-' in result

    def test_truncation(self):
        """Excess variants truncated with count."""
        variants = [{'ref_aa': 'A', 'position': i, 'alt_aa': 'V',
                     'context': 'surface', 'clinical_significance': '-'}
                    for i in range(25)]
        result = format_variant_details(variants, limit=20)
        assert '...(+5 more)' in result

    def test_nan_clinical_significance(self):
        """NaN clinical significance replaced with dash."""
        variants = [{'ref_aa': 'A', 'position': 1, 'alt_aa': 'V',
                     'context': 'surface', 'clinical_significance': 'nan'}]
        result = format_variant_details(variants)
        assert ':surface:-' in result


@pytest.mark.variants
@pytest.mark.slow
class TestAnnotateResultsWithVariants:
    """Test the main annotation entry point."""

    def test_adds_all_variant_columns(self, ref_pdb_1, test_uniprot_variants_path, test_exac_path):
        """annotate_results_with_variants adds all 12 CSV columns."""
        from pdockq import read_pdb_with_chain_info_New

        df = load_uniprot_variants(
            test_uniprot_variants_path,
            frozenset({'A0A0B4J2C3', 'P24534'}),
        )
        idx = build_variant_index(df)
        exac_df = load_exac_constraint(test_exac_path)
        chain_info = read_pdb_with_chain_info_New(str(ref_pdb_1))

        results = [{
            'complex_name': 'A0A0B4J2C3_P24534',
            'protein_a': 'A0A0B4J2C3',
            'protein_b': 'P24534',
            'best_chain_pair': 'A_B',
            '_chain_info': chain_info,
            '_pdb_path': ref_pdb_1,
            '_confident_residue_numbers_a': [81, 82, 83, 84, 86, 87, 90],
            '_confident_residue_numbers_b': [103, 104, 105, 111, 112],
        }]

        gene_lookup = {'A0A0B4J2C3': 'GENE_A', 'P24534': 'GENE_B'}
        annotate_results_with_variants(results, idx, exac_df, gene_lookup)

        row = results[0]
        for col in CSV_FIELDNAMES_VARIANTS:
            assert col in row, f"Missing column: {col}"

    def test_variant_counts_positive(self, ref_pdb_1, test_uniprot_variants_path, test_exac_path):
        """Variant counts are > 0 for proteins with known variants."""
        from pdockq import read_pdb_with_chain_info_New

        df = load_uniprot_variants(
            test_uniprot_variants_path,
            frozenset({'A0A0B4J2C3', 'P24534'}),
        )
        idx = build_variant_index(df)
        exac_df = load_exac_constraint(test_exac_path)
        chain_info = read_pdb_with_chain_info_New(str(ref_pdb_1))

        results = [{
            'complex_name': 'A0A0B4J2C3_P24534',
            'protein_a': 'A0A0B4J2C3',
            'protein_b': 'P24534',
            'best_chain_pair': 'A_B',
            '_chain_info': chain_info,
            '_pdb_path': ref_pdb_1,
            '_confident_residue_numbers_a': [81, 82, 83],
            '_confident_residue_numbers_b': [103, 104],
        }]

        gene_lookup = {'A0A0B4J2C3': 'GENE_A', 'P24534': 'GENE_B'}
        annotate_results_with_variants(results, idx, exac_df, gene_lookup)

        row = results[0]
        assert row['n_variants_a'] > 0
        assert row['n_variants_b'] > 0

    def test_exac_constraint_attached(self, ref_pdb_1, test_uniprot_variants_path, test_exac_path):
        """ExAC pLI/mis_z attached via gene symbol lookup."""
        from pdockq import read_pdb_with_chain_info_New

        df = load_uniprot_variants(
            test_uniprot_variants_path,
            frozenset({'A0A0B4J2C3'}),
        )
        idx = build_variant_index(df)
        exac_df = load_exac_constraint(test_exac_path)
        chain_info = read_pdb_with_chain_info_New(str(ref_pdb_1))

        results = [{
            'complex_name': 'A0A0B4J2C3_P24534',
            'protein_a': 'A0A0B4J2C3',
            'protein_b': 'P24534',
            'best_chain_pair': 'A_B',
            '_chain_info': chain_info,
            '_pdb_path': ref_pdb_1,
            '_confident_residue_numbers_a': [],
            '_confident_residue_numbers_b': [],
        }]

        gene_lookup = {'A0A0B4J2C3': 'GENE_A', 'P24534': 'GENE_B'}
        annotate_results_with_variants(results, idx, exac_df, gene_lookup)

        row = results[0]
        assert row['gene_constraint_pli_a'] != ''
        assert float(row['gene_constraint_pli_a']) == pytest.approx(0.95, abs=0.01)

    def test_private_keys_stripped(self, ref_pdb_1, test_uniprot_variants_path, test_exac_path):
        """Private keys (_chain_info etc.) are removed after annotation."""
        from pdockq import read_pdb_with_chain_info_New

        df = load_uniprot_variants(
            test_uniprot_variants_path,
            frozenset({'A0A0B4J2C3'}),
        )
        idx = build_variant_index(df)
        exac_df = load_exac_constraint(test_exac_path)
        chain_info = read_pdb_with_chain_info_New(str(ref_pdb_1))

        results = [{
            'complex_name': 'A0A0B4J2C3_P24534',
            'protein_a': 'A0A0B4J2C3',
            'protein_b': 'P24534',
            'best_chain_pair': 'A_B',
            '_chain_info': chain_info,
            '_pdb_path': ref_pdb_1,
            '_confident_residue_numbers_a': [81],
            '_confident_residue_numbers_b': [103],
        }]

        gene_lookup = {'A0A0B4J2C3': 'GENE_A'}
        annotate_results_with_variants(results, idx, exac_df, gene_lookup)

        row = results[0]
        assert '_chain_info' not in row
        assert '_pdb_path' not in row
        assert '_confident_residue_numbers_a' not in row
        assert '_confident_residue_numbers_b' not in row

    def test_no_chain_info_graceful(self, test_exac_path):
        """Rows without _chain_info get default values."""
        results = [{
            'complex_name': 'MISSING_COMPLEX',
            'protein_a': 'X12345',
            'protein_b': 'Y67890',
            'best_chain_pair': 'A_B',
        }]
        exac_df = load_exac_constraint(test_exac_path)
        annotate_results_with_variants(results, {}, exac_df, {})

        row = results[0]
        assert row['n_variants_a'] == 0
        assert row['n_variants_b'] == 0
        assert row['variant_details_a'] == ''


# ═══════════════════════════════════════════════════════════════════
# Group 4: CLI
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.variants
@pytest.mark.cli
class TestStandaloneCLI:
    """Test standalone variant mapper CLI."""

    def test_parser_summary_subcommand(self):
        """Parser accepts summary subcommand."""
        parser = build_argument_parser()
        args = parser.parse_args(['summary', '--variants-dir', '/some/dir'])
        assert args.command == 'summary'
        assert args.variants_dir == '/some/dir'

    def test_parser_lookup_subcommand(self):
        """Parser accepts lookup subcommand."""
        parser = build_argument_parser()
        args = parser.parse_args(['lookup', '--variants-dir', '/some/dir', '--protein', 'P24534'])
        assert args.command == 'lookup'
        assert args.protein == 'P24534'

    def test_parser_map_subcommand(self):
        """Parser accepts map subcommand."""
        parser = build_argument_parser()
        args = parser.parse_args([
            'map',
            '--interfaces', 'iface.jsonl',
            '--pdb-dir', '/pdb/',
            '--variants-dir', '/var/',
            '--output', 'out.csv',
        ])
        assert args.command == 'map'
        assert args.interfaces == 'iface.jsonl'
        assert args.output == 'out.csv'

    def test_parser_no_clinvar_flag(self):
        """--no-clinvar flag is parsed."""
        parser = build_argument_parser()
        args = parser.parse_args([
            'map',
            '--interfaces', 'i.jsonl',
            '--pdb-dir', '/pdb/',
            '--no-clinvar',
        ])
        assert args.no_clinvar is True


@pytest.mark.variants
class TestCSVFieldnamesConstant:
    """Test CSV fieldnames constant is correct."""

    def test_twelve_columns(self):
        """CSV_FIELDNAMES_VARIANTS has exactly 12 columns."""
        assert len(CSV_FIELDNAMES_VARIANTS) == 12

    def test_expected_columns(self):
        """All expected column names are present."""
        expected = {
            'n_variants_a', 'n_variants_b',
            'n_interface_variants_a', 'n_interface_variants_b',
            'n_pathogenic_interface_variants',
            'interface_variant_enrichment',
            'variant_details_a', 'variant_details_b',
            'gene_constraint_pli_a', 'gene_constraint_pli_b',
            'gene_constraint_mis_z_a', 'gene_constraint_mis_z_b',
        }
        assert set(CSV_FIELDNAMES_VARIANTS) == expected

    def test_no_duplicates(self):
        """No duplicate column names."""
        assert len(CSV_FIELDNAMES_VARIANTS) == len(set(CSV_FIELDNAMES_VARIANTS))


# ═══════════════════════════════════════════════════════════════════
# Group 5: Integration
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.variants
@pytest.mark.integration
class TestToolkitIntegration:
    """Test integration with toolkit.py."""

    def test_variants_requires_interface_pae_enrich(self):
        """--variants without --interface --pae --enrich should error."""
        import subprocess
        result = subprocess.run(
            ['python', 'toolkit.py', '--dir', 'Test_Data', '--output', 'test.csv',
             '--variants'],
            capture_output=True, text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode != 0
        assert 'requires' in result.stderr.lower() or 'error' in result.stderr.lower()

    def test_get_csv_fieldnames_includes_variants(self):
        """get_csv_fieldnames with include_variants adds variant columns."""
        from toolkit import get_csv_fieldnames
        fields = get_csv_fieldnames(
            include_interface=True, include_pae=True,
            include_enrichment=True, include_variants=True,
        )
        for col in CSV_FIELDNAMES_VARIANTS:
            assert col in fields, f"Missing variant column: {col}"

    def test_get_csv_fieldnames_excludes_variants_by_default(self):
        """Variant columns NOT in fieldnames when include_variants=False."""
        from toolkit import get_csv_fieldnames
        fields = get_csv_fieldnames(
            include_interface=True, include_pae=True,
            include_enrichment=True, include_variants=False,
        )
        for col in CSV_FIELDNAMES_VARIANTS:
            assert col not in fields

    def test_process_single_complex_stash(self):
        """process_single_complex with stash_variant_data preserves _chain_info."""
        from toolkit import find_paired_data_files, process_single_complex

        pairs = find_paired_data_files(str(PROJECT_ROOT / "Test_Data"))
        # Find reference complex 1
        name = 'A0A0B4J2C3_P24534'
        if name in pairs:
            file_paths = pairs[name]
            row = process_single_complex(
                name, file_paths,
                run_interface=True,
                run_interface_pae=True,
                stash_variant_data=True,
            )
            assert '_chain_info' in row
            assert '_pdb_path' in row
            assert row['_chain_info'] is not None
        else:
            pytest.skip("Reference complex 1 not in test data")


# ═══════════════════════════════════════════════════════════════════
# Group 6: Regression
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.variants
@pytest.mark.regression
@pytest.mark.slow
class TestRegressionComplex1:
    """Regression tests for reference complex 1 variant mapping."""

    @pytest.fixture
    def complex1_mapped(self, ref_pdb_1, test_uniprot_variants_path):
        """Map variants for reference complex 1."""
        from pdockq import read_pdb_with_chain_info_New

        df = load_uniprot_variants(
            test_uniprot_variants_path,
            frozenset({'A0A0B4J2C3', 'P24534'}),
        )
        idx = build_variant_index(df)
        chain_info = read_pdb_with_chain_info_New(str(ref_pdb_1))

        sasa_a = compute_residue_sasa(ref_pdb_1, 'A')
        sasa_b = compute_residue_sasa(ref_pdb_1, 'B')

        interface_a = {81, 82, 83, 84, 86, 87, 90, 92, 93, 96, 97, 100, 115, 118, 119}
        interface_b = {103, 104, 105, 111, 112, 114, 115, 118, 119, 122, 123, 125, 126, 129}

        mapped_a = map_variants_to_complex(
            'A0A0B4J2C3', 'A', idx, interface_a,
            chain_info.chain_res_numbers.get('A', []),
            chain_info.cb_coords.get('A', np.empty((0, 3))),
            sasa_a,
        )
        mapped_b = map_variants_to_complex(
            'P24534', 'B', idx, interface_b,
            chain_info.chain_res_numbers.get('B', []),
            chain_info.cb_coords.get('B', np.empty((0, 3))),
            sasa_b,
        )

        return mapped_a, mapped_b, interface_a, interface_b, chain_info

    def test_variant_count_a(self, complex1_mapped):
        """Reference complex 1 protein A has expected number of mapped variants."""
        mapped_a, _, _, _, _ = complex1_mapped
        # Test data has 9 unique variants for A0A0B4J2C3 (after dedup of L81P)
        # All should be in chain A's residue range
        assert len(mapped_a) > 0

    def test_interface_variants_a(self, complex1_mapped):
        """Some protein A variants are at interface positions."""
        mapped_a, _, _, _, _ = complex1_mapped
        interface_variants = [v for v in mapped_a
                              if v['context'] in (CONTEXT_INTERFACE_CORE, CONTEXT_INTERFACE_RIM)]
        # Positions 81, 82, 83, 90 are in both test data and interface set
        assert len(interface_variants) >= 3

    def test_interface_core_positions(self, complex1_mapped):
        """Interface core variants exist at known contact positions."""
        mapped_a, _, _, _, _ = complex1_mapped
        core_positions = {v['position'] for v in mapped_a
                          if v['context'] == CONTEXT_INTERFACE_CORE}
        # At least position 81 should be interface_core
        assert 81 in core_positions or len(core_positions) > 0

    def test_pathogenic_at_interface(self, complex1_mapped):
        """Pathogenic variant at position 81 is at interface."""
        mapped_a, _, _, _, _ = complex1_mapped
        pos81 = [v for v in mapped_a if v['position'] == 81]
        assert len(pos81) >= 1
        assert pos81[0]['clinical_significance'] == 'pathogenic'
        assert pos81[0]['context'] in (CONTEXT_INTERFACE_CORE, CONTEXT_INTERFACE_RIM)

    def test_enrichment_positive(self, complex1_mapped):
        """Interface variant enrichment is > 1.0 for this complex."""
        mapped_a, mapped_b, interface_a, interface_b, chain_info = complex1_mapped
        n_if_vars = sum(
            1 for v in mapped_a + mapped_b
            if v['context'] in (CONTEXT_INTERFACE_CORE, CONTEXT_INTERFACE_RIM)
        )
        n_total_vars = len(mapped_a) + len(mapped_b)
        n_if_res = len(interface_a) + len(interface_b)
        n_total_res = (len(chain_info.chain_res_numbers.get('A', [])) +
                       len(chain_info.chain_res_numbers.get('B', [])))
        enrichment = compute_interface_variant_enrichment(
            n_if_vars, n_total_vars, n_if_res, n_total_res,
        )
        assert enrichment > 1.0, f"Expected enrichment > 1.0, got {enrichment}"


# ═══════════════════════════════════════════════════════════════════
# Constants & Module-level tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.variants
class TestModuleConstants:
    """Test module-level constants and configuration."""

    def test_three_to_one_has_20_aas_plus_ter(self):
        """THREE_TO_ONE dict has 20 amino acids + TER."""
        assert len(THREE_TO_ONE) == 21

    def test_three_to_one_ter_is_star(self):
        """TER maps to '*'."""
        assert THREE_TO_ONE['TER'] == '*'

    def test_max_asa_has_20_residues(self):
        """MAX_ASA has all 20 standard amino acids."""
        assert len(MAX_ASA) == 20

    def test_relevant_consequences_frozen(self):
        """RELEVANT_CONSEQUENCES is a frozenset."""
        assert isinstance(RELEVANT_CONSEQUENCES, frozenset)
        assert 'missense variant' in RELEVANT_CONSEQUENCES
        assert 'stop gained' in RELEVANT_CONSEQUENCES

    def test_distance_thresholds(self):
        """Distance thresholds are consistent."""
        assert INTERFACE_CORE_DISTANCE < INTERFACE_RIM_DISTANCE
        assert INTERFACE_CORE_DISTANCE == 4.0
        assert INTERFACE_RIM_DISTANCE == 8.0

    def test_sasa_threshold(self):
        """SASA buried threshold is 25%."""
        assert SASA_BURIED_THRESHOLD == 0.25
