# Output schema

This document defines the columns emitted by the full `toolkit.py` pipeline. It records field availability, missing-value behaviour, serialisation formats and the population filters used by the visualisation workflow. The principal workflow is described in [`../README.md`](../README.md); the commands that produce each column are in [`Toolkit_Commands_List.md`](Toolkit_Commands_List.md).

## Schema summary

### Column counts by stage

The base output has **41 columns**. Each optional stage appends its own block, up to **155 columns** under `--full-pipeline`. Counts are derived from the field registries in the code (`toolkit.py::CSV_FIELDNAMES_*` and the `CSV_FIELDNAMES_*` constant in each downstream module).

| Stage | Flag | Owning registry | Fields | Cumulative |
| ----- | ---- | --------------- | :----: | :--------: |
| Base | (always) | `toolkit.py::CSV_FIELDNAMES_BASE` | 41 | 41 |
| Enrichment | `--enrich` | `CSV_FIELDNAMES_ENRICHMENT` | 12 | 53 |
| Interface | `--interface` | `CSV_FIELDNAMES_INTERFACE` | 23 | 76 |
| Flags | `--interface` | `CSV_FIELDNAMES_FLAGS` | 1 | 77 |
| Interface PAE | `--interface --pae` | `CSV_FIELDNAMES_INTERFACE_PAE` | 19 | 96 |
| Clustering | `--clustering` | `protein_clustering.py` | 7 | 103 |
| Variants | `--variants` | `variant_mapper.py` | 12 | 115 |
| Stability | `--stability` | `stability_scorer.py` | 8 | 123 |
| ProtVar | `--protvar` | `protvar_client.py` | 8 | 131 |
| Disease | `--disease` | `disease_annotations.py` | 14 | 145 |
| Pathways | `--pathways` | `pathway_network.py` | 10 | 155 |

### Column order

Columns are assembled in a fixed order (`toolkit.py::get_csv_fieldnames`) and a later stage never reorders earlier columns:

```text
base → enrichment → interface → flags → interface-PAE →
clustering → variants → stability → ProtVar → disease → pathways
```

### Optional stages

The 41 base columns are present in every run. Every later block appears only when its flag is active: the CSV writer uses `extrasaction='ignore'`, so a field absent from the active stage set is not written even if the pipeline computed it internally. The dependency chain is `--pae` → `--interface`; `--variants` → `--interface --pae --enrich`; `--stability`/`--protvar` → `--variants`; and `--databases`/`--clustering`/`--disease`/`--pathways` → `--enrich`.

### Missing values

| Value | Meaning |
| ----- | ------- |
| Empty / `NaN` | Not calculated, unavailable, or not applicable to this row |
| `0` | Calculated, and the result was zero |

An empty cell is not the same as zero. A variant count of `0` means the lookup ran and found none; an empty variant-count cell means the lookup did not run (for example, the stage was off, or the row is `non_human` and the stage is human-scoped). Field entries below note where a specific column deviates from this rule.

### Per-chain suffixes

A `_a` / `_b` suffix denotes the two chains of the best-scoring pair (see [best-pair versus all-pairs fields](#best-pair-versus-all-pairs-fields)). Paired columns are documented together; both are always emitted.

### Serialised fields

Several columns pack multiple items into one cell. Most detail columns use the pipe `|`; a few identity columns use commas; two summary columns use their own formats; and three columns are JSON.

| Column(s) | Separator | One item / format | Cap | Over-cap suffix |
| --------- | :-------: | ----------------- | --- | --------------- |
| `variant_details_a/b` | `\|` | `K81P:interface_core:pathogenic` | 20 | `\|...(+N more)` |
| `stability_details_a/b` | `\|` | `R4A:eve=0.77:Pathogenic` | 20 | `\|...(+N more)` |
| `protvar_details_a/b` | `\|` | `M1A:am=0.36:ambiguous:foldx=0.11` | 20 | `\|...(+N more)` |
| `disease_details_a/b` | `\|` | `OMIM:618428:Popov-Chang syndrome (POPCHAS)` | 50 | `\|...(+N more)` |
| `ptm_details_a/b` | `\|` | `Phosphoserine:S9` | 50 | `\|...(+N more)` |
| `go_biological_process_a/b`, `go_molecular_function_a/b` | `\|` | `GO:0005515:protein binding` | 50 | `\|...(+N more)` |
| `reactome_pathways_a/b` | `\|` | `R-HSA-109581:Apoptosis` | 20 | `\|...(+N more)` |
| `homologous_pairs` | `\|` | `P04637_Q00987` | 20 | `\|+N more` |
| `sequence_cluster_ids`, `shared_cluster_ids` | `\|` | one STRING cluster id | none | — |
| `interface_residues_a/b` | `\|` | one PDB residue number | none | — |
| `secondary_accessions_a/b` | `\|` | one UniProt accession | none | — |
| `unique_accessions`, `chain_ids` | `,` | one accession / one chain id | none | — |
| `pathway_quality_context` | `;` | `mean_pdockq=..;frac_high=..;n_complexes=..` | 3 fields (+ optional `enrichment_fdr=`) | — |
| `pair_metrics`, `accession_chain_map`, `complex_identity_json` | JSON | see the field entry | — | — |

Two conventions are not uniform: `homologous_pairs` uses `|+N more` rather than the `|...(+N more)` form used elsewhere, and the `go_*` over-cap count reflects only the terms of the relevant Gene Ontology aspect.

### Use tags

Each field carries one of four tags. A tag describes the field's role in the output, not its correctness or general usefulness. Tags are set at group level where every field shares the role, and per field where a group is mixed.

| Tag | Meaning |
| --- | ------- |
| `core` | Defines the analysis populations, or produces the interface confidence, quality classification or screening status. |
| `supporting` | Biological-annotation fields used for context and corroboration. |
| `audit` | Input discovery, recoverability, calibration eligibility, species scope and provenance. |
| `extended` | Fields from implemented modules that are peripheral to the primary interface assessment. |

## Population filters

The filters below reproduce the named populations used by `visualise_filters.py`. Apply them to the full results CSV unless a specific analysis defines another scope. Conditions are given as exact field tests so a population can be reconstructed with `pandas`.

| Filter | Condition (clauses joined by AND unless stated) | Use | Main exclusions |
| ------ | ----------------------------------------------- | --- | --------------- |
| `recoverable` | `partial_reason` empty or `NaN` | Any analysis of successfully assessed rows | Partial and error rows |
| `calibrated_dimer` | `tier_scope == "dimer_validated"`; `composite_is_calibrated` is `True`; recoverable; the six columns `iptm`, `pdockq`, `interface_confidence_score`, `strict_confident_contact_fraction`, `interface_pae_mean`, `n_interface_contacts` all non-NaN; `n_interface_contacts > 0` | Structural-quality, tier and composite analyses | Multimers, partial rows, dimers missing a required metric or with no interface contact |
| `calibrated_human_broad` | `calibrated_dimer` AND `species_status ∈ {reviewed_human, trembl_human}` | Human analyses that tolerate unreviewed entries | Non-human rows, and everything `calibrated_dimer` excludes |
| `calibrated_human_strict` | `calibrated_dimer` AND `species_status == reviewed_human` | Human analyses requiring curated records | TrEMBL-human and non-human rows |
| `multimer_exploratory` | `tier_scope == "multimer_provisional"` OR `n_chains > 2` | Descriptive multimer inspection only | Everything treated as calibrated |
| `partial_error` | `partial_reason` present and not empty | Auditing incomplete assessments | All recoverable rows |
| `strong_screen_candidate` | `composite_screen_status == "strong_screen_candidate"` | Highest-priority shortlist | Moderate, weak, unavailable |
| `moderate_screen_candidate` | `composite_screen_status == "moderate_screen_candidate"` | Mid-priority candidates | Strong, weak, unavailable |
| `weak_screen_candidate` | `composite_screen_status == "weak_screen_candidate"` | Lowest-priority candidates | Strong, moderate, unavailable |
| `human_broad` | `species_status ∈ {reviewed_human, trembl_human}` | Species scoping before calibration | Non-human rows |
| `human_strict` | `species_status == reviewed_human` | Reviewed-human scoping before calibration | TrEMBL-human and non-human rows |
| `all_rows` | (no condition) | The full CSV | None |

Two operational filters complete the set: `composite_status_present` (`composite_screen_status` non-empty) and `composite_screenable` (present, not `unavailable`, and with a numeric `interface_confidence_score`). The three screen-candidate filters partition the calibrated-dimer population by priority; they are strata, not additional quality categories. Each filter degrades gracefully — a filter whose required column is absent selects no rows, except `all_rows` (always all rows) and `recoverable` (all rows when `partial_reason` is absent).

## Important field distinctions

Compact comparisons of the fields most often misread. Each is grounded in the assignment functions named alongside it.

### `quality_tier` versus `quality_tier_v2`

`quality_tier` (v1, `toolkit.py::classify_prediction_quality`) uses ipTM and pDockQ only: `High` when `iptm ≥ 0.75` and `pdockq ≥ 0.50`; `Medium` when `iptm ≥ 0.50` and `pdockq ≥ 0.23` and not already High; `Low` otherwise. `quality_tier_v2` (`classify_prediction_quality_v2`) starts from v1 and reclassifies using the composite score under an asymmetric policy: a v1 Low is rescued to Medium at composite `≥ 0.64` and to High at `≥ 0.85`; a v1 Medium is promoted to High at `≥ 0.85`; a v1 High is demoted to Medium at composite `≤ 0.63`. A v1 Medium is never demoted to Low, and a missing or non-finite composite leaves the v1 tier unchanged. Use `quality_tier_v2` for interface-aware statements and `quality_tier` as the baseline.

### Classification versus `composite_screen_status`

`quality_tier_v2` is a classification anchored in the joint ipTM–pDockQ baseline; `composite_screen_status` (`classify_composite_screen_status`) is a prioritisation label from the composite alone (`strong_` at `≥ 0.85`, `moderate_` at `≥ 0.63`, `weak_` below, or `unavailable`). They can differ without contradiction: a v1 High with a composite between 0.63 and 0.85 stays High but is only a moderate screen candidate. The screening status sits beside the tier; it does not replace it.

### Composite score versus a probability

`interface_confidence_score` (`interface_analysis.py::compute_interface_confidence`) is a weighted heuristic on 0–1: `0.35·P + 0.35·C + 0.15·S + 0.15·D`, where **P** is normalised mean interface pLDDT (`(pLDDT − 50) / 40`, clipped to `[0.05, 1.0]`), **C** is the strict confident-contact fraction, **S** is interface symmetry and **D** is normalised contact density (`min(density / 2, 1)`). The weights are fixed constants, not learned from labelled outcomes. It ranks predictions; it is not a probability of correctness, a positive-predictive value, or an externally calibrated estimate of accuracy.

### Ranking confidence versus the composite score

`ranking_confidence` is AlphaFold-Multimer's own `0.8·ipTM + 0.2·pTM`, used upstream to select among predicted structures. It is distinct from `interface_confidence_score`, which is computed by this toolkit at the predicted interface.

### Dimer-validated versus multimer-provisional

`tier_scope` is `dimer_validated` for two-chain complexes and `multimer_provisional` otherwise. "Validated" refers only to the scope in which the composite and its thresholds are calibrated; it does not mean experimental validation. Complexes with more than two chains are scored structurally — every chain pair is measured and serialised in `pair_metrics`, with the largest-contact pair populating the canonical interface columns — but their composites and tiers are not calibrated and are excluded from calibrated analyses. Restrict headline confidence claims to `tier_scope == "dimer_validated"`.

### Numeric score versus calibrated eligibility

A row can carry a numeric `interface_confidence_score` without being calibrated. `composite_is_calibrated` (`toolkit.py::_finalise_calibration_flag`) is `True` only when the row is a dimer, has no `partial_reason`, and has every composite input present. Filter on this flag for a calibrated claim; do not filter merely on the score being present.

### Partial or unrecoverable versus low quality

A non-empty `partial_reason` marks a row whose assessment is incomplete; such rows are excluded from calibrated analyses. This is distinct from a low-tier row, which was fully assessed and classified as low confidence. See [Recoverability](#recoverability--audit) for the vocabulary and precedence.

### Best-pair versus all-pairs fields

For a two-chain complex the best-pair columns and the whole-complex aggregates are identical. For more than two chains they differ, and the aggregation method is not uniform:

| Field(s) | Aggregation |
| -------- | ----------- |
| `pdockq_mean`, `pdockq_min` | Unweighted mean / minimum over all pairs, including zero-contact pairs (which contribute `0.0`) |
| `interface_plddt_mean`, `symmetry_mean`, `pae_confident_fraction_mean`, `strict_confident_fraction_mean` | Contact-weighted mean, excluding zero-contact pairs |
| `symmetry_min` | Minimum over contact-bearing pairs only |
| `pdockq_whole_complex` | Recomputed once over the union of all inter-chain contacts (not a mean of per-pair values) |
| `contact_count_total` | Sum over all pairs |

### PAE-only versus strict confident contacts

Both are fractions of interface contacts over the same denominator. `pae_confident_contact_fraction` counts a contact when its bidirectional PAE — the larger of PAE(i,j) and PAE(j,i), so both directions must agree — is below 5 Å. `strict_confident_contact_fraction` additionally requires both residues at pLDDT ≥ 70. The strict fraction feeds the composite and is a required calibration input; the PAE-only fraction feeds the quality flags.

### Interface pLDDT versus whole-complex pLDDT

`interface_plddt_*` averages pLDDT over interface residues only; `plddt_mean`/`plddt_median`/etc. average over the whole complex. `interface_vs_bulk_delta` is `interface_plddt_combined` minus the mean pLDDT of non-interface residues; a positive value means confidence is concentrated at the interface.

### Monomeric FoldX ΔΔG versus binding ΔΔG

`protvar_foldx_mean_*` reports monomeric FoldX ΔΔG — the estimated change in a single protein's folding stability. It is not a binding or interface ΔΔG and does not directly measure disruption of the interaction.

### Annotation versus structural validation

The enrichment, clustering, variant, disease and pathway columns record annotation — that a protein or pair appears in an external resource. This is context and corroboration; it is not experimental evidence that the modelled interface is correct or that the interaction occurs.

### Species-agnostic processing versus human-centred annotation

Structural assessment runs for any species; species status does not affect interface geometry or confidence metrics. The lightweight enrichment fields (gene symbols, protein names, sequences, Ensembl and secondary accessions, database tags) are attempted for all rows. The six heavier modules (clustering, variants, stability, ProtVar, disease, pathways) run for `reviewed_human` and `trembl_human` rows and are skipped for `non_human` rows, whose columns are left at their empty defaults. The gate is `toolkit.py::is_annotatable`, which returns `False` only for `species_status == "non_human"`.

## Base columns

Present in every run. Tags are on the group headings; mixed groups carry a per-field tag.

### Identity and species — `audit`

| Column | Type | Values | Meaning | Tag |
| ------ | ---- | ------ | ------- | --- |
| `schema_version` | text | `multimer_v1` | Output-schema version tag. | audit |
| `complex_name` | text | — | Unique complex identifier and primary key. | core |
| `protein_a`, `protein_b` | text | — | UniProt accession of each chain of the best pair. | core |
| `complex_type` | categorical | `Homodimer` / `Heterodimer` / `Multi-chain` (`None` on a worker-exception row) | Coarse legacy type label. | audit |
| `n_chains` | integer | ≥ 1 | Number of chains; determines `tier_scope`. | core |
| `best_chain_pair` | text | e.g. `A_B` | The chain pair with the most interface contacts; drives the best-pair columns. | core |
| `species` | text | constant `Homo sapiens (9606)` | Constant provenance tag on every row; it does **not** carry the row's real species. The actual classification is `species_status`. | audit |
| `structure_source` | text | constant `AlphaFold2_prediction` | Constant provenance tag. | audit |
| `species_a`, `species_b` | categorical | `reviewed_human` / `trembl_human` / `non_human` | Per-chain species classification (isoform suffixes stripped before lookup). | audit |
| `species_status` | categorical | `reviewed_human` / `trembl_human` / `non_human` | Complex-level species tag: the more restrictive of the two chains. Governs which annotation stages run. | core |

### AlphaFold metrics — `core`

| Column | Type | Range / unit | Meaning |
| ------ | ---- | ------------ | ------- |
| `iptm` | float | 0–1 | AlphaFold interface predicted TM-score; a complex-level baseline. |
| `ptm` | float | 0–1 | AlphaFold predicted TM-score. |
| `ranking_confidence` | float | — | AlphaFold ranking confidence (`0.8·ipTM + 0.2·pTM`); see the [ranking-confidence distinction](#ranking-confidence-versus-the-composite-score). |
| `pae_mean` | float | Å | Mean of the whole-complex PAE matrix (all residue pairs). |
| `pdockq` | float | 0–1 | Reproduced pDockQ for the best pair (FoldDock sigmoid, 8 Å Cβ contacts); `0` when no inter-chain contacts are detected. |
| `ppv` | float | 0–1 | Positive-predictive-value companion returned with `pdockq`. |

### Whole-complex pLDDT — `core`

| Column | Type | Range | Meaning |
| ------ | ---- | ----- | ------- |
| `plddt_mean`, `plddt_median`, `plddt_min`, `plddt_max` | float | 0–100 | Summary statistics of per-residue pLDDT over the whole complex. |
| `plddt_below50_fraction`, `plddt_below70_fraction` | float | 0–1 | Fraction of residues below the pLDDT 50 / 70 bands. `plddt_below50_fraction ≥ 0.30` is an input to the paradox interface flags. |
| `num_residues` | integer | ≥ 1 | Total residue count. |

### Classification and screening — `core`

The base stage emits only the v1 tier; `quality_tier_v2` and `composite_screen_status` are added at the interface-PAE stage.

| Column | Type | Values | Meaning |
| ------ | ---- | ------ | ------- |
| `quality_tier` | categorical | `High` / `Medium` / `Low` (`Error` on a worker-exception row) | The v1 tier from ipTM and pDockQ. See the [tier distinction](#quality_tier-versus-quality_tier_v2). |

### Multimer identity — `audit`

| Column | Type | Values | Meaning | Tag |
| ------ | ---- | ------ | ------- | --- |
| `stoichiometry` | text | e.g. `A2`, `AB`, `A2B`, `A2B2`, `ABCD` | Chain stoichiometry. | core |
| `is_homomeric` | boolean | | Whether all chains share one accession. | audit |
| `unique_accessions` | text (`,`) | e.g. `P04637,Q00987` | Comma-separated distinct accessions, first-seen order (not a count). | audit |
| `chain_ids` | text (`,`) | e.g. `A,B` | Comma-separated PDB chain identifiers. | audit |
| `accession_chain_map` | JSON | `{chain: accession}` | JSON object mapping **chain id → accession** (not the reverse). | audit |
| `tier_scope` | categorical | `dimer_validated` / `multimer_provisional` (`None` on a worker-exception row) | Calibration scope. See the [scope distinction](#dimer-validated-versus-multimer-provisional). | core |
| `filename_n_chains` | integer | | Chain count inferred from the filename. | audit |
| `pdb_n_chains` | integer | | Chain count observed in the PDB. | audit |
| `chain_count_consistency` | categorical | `match` / `filename_only` / `pdb_only` / `mismatch` | Agreement between the filename- and PDB-derived chain counts. | audit |
| `complex_identity_json` | JSON | | Full structured identity record (accessions, stoichiometry, chain map, chain counts). | audit |

### Recoverability — `audit`

| Column | Type | Values | Meaning |
| ------ | ---- | ------ | ------- |
| `has_pdb`, `has_pkl` | boolean | | Whether each source file was **discovered** in the input paths — not whether it was readable or parsed. Usability is recorded by `partial_reason`; computability by `composite_is_calibrated`. |
| `geometry_available` | boolean | | Whether interface geometry could be calculated. |
| `plddt_source` | categorical | `pdb` / `pkl` | Whether pLDDT was read from the PKL output or the PDB B-factor column. |
| `partial_reason` | categorical | see below | Empty for a fully assessed row; otherwise the highest-priority failure code. See the [partial distinction](#partial-or-unrecoverable-versus-low-quality). |

**`partial_reason` vocabulary and precedence.** When more than one condition applies, only the highest-priority value is stored (`toolkit.py::PARTIAL_REASON_PRIORITY`); a new reason overwrites the current one only if the current is empty or the new priority is strictly higher.

| Value (highest priority first) | Meaning |
| ------------------------------ | ------- |
| `worker_exception` | The per-complex worker raised an unhandled exception; a minimal error row is emitted (`quality_tier = "Error"`, `quality_tier_v2 = None`, `tier_scope = None`). |
| `pdb_decompression_error` | A compressed PDB could not be decompressed. |
| `pdb_io_error` | The PDB could not be read (I/O failure). |
| `pdb_parse_error` | The PDB was read but structure/chain parsing failed. |
| `pdb_no_chains` | The PDB parsed but yielded fewer than two usable chains. |
| `unreadable_pdb_or_structure_input` | Legacy fallback for an unclassified PDB error. |
| `pkl_decompression_error` | A compressed PKL could not be decompressed. |
| `pkl_io_error` | The PKL could not be read (I/O failure). |
| `pkl_unpickle_error` | The PKL was present but could not be deserialised. |
| `missing_pkl_or_pkl_unreadable` | Legacy fallback for an unclassified PKL error. |
| `no_positive_interface_contacts` | The best pair had zero inter-chain contacts (`pdockq` recorded as `0`; composite unavailable). |
| `pkl_loaded_missing_iptm` | The PKL loaded but held no finite ipTM. |
| `pkl_loaded_missing_pae` | The PKL loaded but held no PAE matrix. |
| `missing_required_composite_inputs` | A dimer expected to calibrate, but a required composite input is missing. |
| `incomplete_input` | Generic incompleteness not matching a more specific reason. |
| `""` (empty) | No failure recorded; calibrated inclusion is then decided by `tier_scope` and `composite_is_calibrated`. |

## Enrichment columns — `--enrich`

Attempted for **all rows**, including `non_human`. Mostly `supporting`; the cross-reference fields are `extended`.

| Column | Type | Meaning | Tag |
| ------ | ---- | ------- | --- |
| `gene_symbol_a`, `gene_symbol_b` | text | HGNC gene symbol per chain. | supporting |
| `protein_name_a`, `protein_name_b` | text | Full protein name per chain. | supporting |
| `ensembl_id_a`, `ensembl_id_b` | text | Ensembl cross-reference per chain. | extended |
| `secondary_accessions_a`, `secondary_accessions_b` | text (`\|`) | Alternate UniProt accessions (merged or TrEMBL entries); no cap. | extended |
| `database_source` | text (`\|`) | Interaction database(s) in which the pair is found, when `--databases` is active (STRING / BioGRID / HuRI / HuMAP). Records membership, not validation. | supporting |
| `evidence_types` | text (`\|`) | Evidence-type tags from the matched databases. | supporting |
| `sequence_a`, `sequence_b` | text | Amino-acid sequence per chain (assigned during structural processing, so present for all processed complexes). | supporting |

## Interface columns — `--interface`

Adds interface geometry plus the `interface_flags` column.

### Best-pair fields — `core`

| Column | Type | Range / unit | Meaning |
| ------ | ---- | ------------ | ------- |
| `n_interface_contacts` | integer | ≥ 0 | Inter-chain residue contacts in the best pair (8 Å Cβ; Cα for glycine). A required calibration input; must exceed 0. |
| `n_interface_residues_a`, `n_interface_residues_b` | integer | ≥ 0 | Interface residue count contributed by each chain. |
| `interface_residues_a`, `interface_residues_b` | text (`\|`) | | PDB residue numbers at the interface, per chain, sorted ascending; no cap. |
| `interface_fraction_a`, `interface_fraction_b` | float | 0–1 | Fraction of each chain's residues at the interface. |
| `interface_symmetry` | float | 0–1 | Smaller chain interface fraction divided by the larger (1 = balanced). Composite component **S**. |
| `contacts_per_interface_residue` | float | ≥ 0 | Contact density (contacts per unique interface residue). Composite component **D** after `min(density/2, 1)`. |
| `interface_plddt_a`, `interface_plddt_b` | float | 0–100 | Mean interface pLDDT per chain. |
| `interface_plddt_combined` | float | 0–100 | Mean pLDDT over both chains' interface residues. Composite component **P** after band-normalisation. |
| `bulk_plddt_combined` | float | 0–100 | Mean pLDDT over non-interface residues. |
| `interface_vs_bulk_delta` | float | pLDDT units | `interface_plddt_combined` minus `bulk_plddt_combined`. See the [interface-vs-whole distinction](#interface-plddt-versus-whole-complex-plddt). |
| `interface_plddt_high_fraction` | float | 0–1 | Fraction of interface residues with high pLDDT. |

### Whole-complex and all-pairs fields — `supporting`

Best-pair and whole-complex values are identical for two-chain complexes; see the [best-pair distinction](#best-pair-versus-all-pairs-fields) for the aggregation rules.

| Column | Type | Range / unit | Meaning |
| ------ | ---- | ------------ | ------- |
| `pair_metrics` | JSON | list, length `C(M,2)` | One record per inter-chain pair (M = chains with Cβ coordinates), including zero-contact pairs. Each record holds the pair's chains and accessions, contact count, interface pLDDT, pDockQ, PPV, symmetry, the two PAE fractions and per-chain interface residues. |
| `pdockq_mean`, `pdockq_min` | float | 0–1 | Unweighted mean / minimum of per-pair pDockQ; zero-contact pairs contribute `0.0`. |
| `pdockq_whole_complex` | float | 0–1 | pDockQ recomputed once over the union of all inter-chain contacts. |
| `contact_count_total` | integer | ≥ 0 | Sum of inter-chain contacts across all pairs. |
| `interface_plddt_mean` | float | 0–100 | Contact-weighted mean interface pLDDT; zero-contact pairs excluded. |
| `symmetry_mean` | float | 0–1 | Contact-weighted mean interface symmetry; zero-contact pairs excluded. |
| `symmetry_min` | float | 0–1 | Minimum symmetry over contact-bearing pairs. |

### Interface flags — `core`

| Column | Type | Meaning |
| ------ | ---- | ------- |
| `interface_flags` | text (comma-joined) | Zero or more automated quality flags; empty when none apply. |

The eight flags are: `small_interface` (fewer than 5 contacts); `sparse_interface` (low contact density); `asymmetric_interface` (symmetry below 0.5); `interface_better_than_bulk` (interface pLDDT exceeds bulk by more than 10); `low_interface_confidence` (PAE-only confident fraction below 0.2); `paradox_confident_disorder` and `paradox_artefactual` (both set when a high-global-confidence prediction has substantial low-pLDDT structure, distinguished by whether the interface contacts are PAE-confident); and `metric_disagreement` (`|iptm − pdockq| > 0.52`).

## Interface PAE columns — `--interface --pae`

Mostly `core`; the directional PAE diagnostics and the two all-pairs PAE means are `supporting`.

| Column | Type | Range / unit | Meaning | Tag |
| ------ | ---- | ------------ | ------- | --- |
| `interface_pae_mean` | float | Å | Mean interface PAE over the bidirectional contact-level values. A required calibration input. | core |
| `interface_pae_median` | float | Å | Median interface PAE. | core |
| `n_pae_confident_contacts` | integer | ≥ 0 | Contacts with bidirectional PAE below 5 Å. | core |
| `pae_confident_contact_fraction` | float | 0–1 | PAE-only confident fraction; feeds the flags. See the [PAE distinction](#pae-only-versus-strict-confident-contacts). | core |
| `n_strict_confident_contacts` | integer | ≥ 0 | Contacts with PAE below 5 Å **and** both residues at pLDDT ≥ 70. | core |
| `strict_confident_contact_fraction` | float | 0–1 | Strict confident fraction; the PAE component of the composite and a required calibration input. | core |
| `cross_chain_pae_mean` | float | Å | Mean of the cross-chain PAE block. | core |
| `interface_pae_forward_mean`, `interface_pae_reverse_mean` | float | Å | Directional interface PAE means (not composite inputs). | supporting |
| `interface_pae_directional_delta_mean`, `interface_pae_directional_delta_max` | float | Å | Mean and maximum asymmetry between the two PAE directions. | supporting |
| `n_confident_residues_a`, `n_confident_residues_b` | integer | ≥ 0 | Confident interface residues per chain. | core |
| `interface_confidence_score` | float | 0–1 | The composite screening score; empty when any of its four inputs is unavailable. See the [composite distinction](#composite-score-versus-a-probability). | core |
| `quality_tier_v2` | categorical | `High` / `Medium` / `Low` (`None` on a worker-exception row) | The composite-informed final tier. | core |
| `composite_screen_status` | categorical | `strong_screen_candidate` / `moderate_screen_candidate` / `weak_screen_candidate` / `unavailable` | The prioritisation label. See the [screening distinction](#classification-versus-composite_screen_status). | core |
| `pae_confident_fraction_mean`, `strict_confident_fraction_mean` | float | 0–1 | Contact-weighted all-pairs PAE fractions; empty if any contact-bearing pair lacks PAE. | supporting |
| `composite_is_calibrated` | boolean | | Calibration-eligibility flag: `True` only for a dimer with no `partial_reason` and all composite inputs present. See the [eligibility distinction](#numeric-score-versus-calibrated-eligibility). | core |

## Clustering columns — `--clustering`

Requires `--enrich`; skipped for `non_human` rows.

| Column | Type | Meaning | Tag |
| ------ | ---- | ------- | --- |
| `sequence_cluster_ids` | text (`\|`) | Union of the two proteins' STRING cluster memberships; no cap. | extended |
| `sequence_cluster_count` | integer | Number of clusters spanned. | extended |
| `shared_cluster_ids` | text (`\|`) | Clusters containing both proteins (a homology signal). | supporting |
| `shared_cluster_count` | integer | Number of shared clusters. | supporting |
| `homologous_pairs` | text (`\|`) | Detected homologous protein pairs (`a_b`); cap 20 with the `\|+N more` suffix. | extended |
| `n_homologous_pairs` | integer | Count of homologous pairs. | extended |
| `homology_bitscore` | float | STRING pairwise homology bitscore where available; empty under `--no-api`. Not proof of paralogy. | extended |

## Variant columns — `--variants`

Requires `--interface --pae --enrich`; skipped for `non_human` rows. Tag: `supporting`.

| Column | Type | Range | Meaning |
| ------ | ---- | ----- | ------- |
| `n_variants_a`, `n_variants_b` | integer | ≥ 0 | Mapped variants per chain (UniProt humsavar, with ClinVar significance attached by rsID). |
| `n_interface_variants_a`, `n_interface_variants_b` | integer | ≥ 0 | Variants at interface residues, per chain. |
| `n_pathogenic_interface_variants` | integer | ≥ 0 | Count of pathogenic interface variants. |
| `interface_variant_enrichment` | float | fold-change | Interface variant enrichment relative to the rest of the protein. |
| `variant_details_a`, `variant_details_b` | text (`\|`) | | Per-variant records `REF{POS}ALT:context:significance`; context is `interface_core` (< 4 Å from a partner residue), `interface_rim` (4–8 Å), `surface_non_interface` (relative solvent accessibility ≥ 25 %) or `buried_core` (< 25 %). Cap 20. |
| `gene_constraint_pli_a`, `gene_constraint_pli_b` | float | 0–1 | ExAC pLI (loss-of-function intolerance) per gene; the complex meets the criterion when either chain has pLI ≥ 0.9. |
| `gene_constraint_mis_z_a`, `gene_constraint_mis_z_b` | float | z-score | ExAC missense constraint z-score per gene. |

## Stability columns — `--stability`

Requires `--variants`; skipped for `non_human` rows. Tag: `supporting`.

| Column | Type | Range | Meaning |
| ------ | ---- | ----- | ------- |
| `eve_score_mean_a`, `eve_score_mean_b` | float | 0–1 | Mean EVE pathogenicity score over the chain's mapped variants. |
| `eve_n_pathogenic_a`, `eve_n_pathogenic_b` | integer | ≥ 0 | Count of EVE-pathogenic variants per chain. |
| `eve_coverage_a`, `eve_coverage_b` | float | 0–1 | Fraction of the chain's variants with an EVE score. |
| `stability_details_a`, `stability_details_b` | text (`\|`) | | Per-variant EVE records `REF{POS}ALT:eve=score:class`; cap 20. |

## ProtVar columns — `--protvar`

Requires `--variants`; skipped for `non_human` rows. Offline AlphaMissense and monomeric FoldX. Tag: `supporting`.

| Column | Type | Range / unit | Meaning |
| ------ | ---- | ------------ | ------- |
| `protvar_am_mean_a`, `protvar_am_mean_b` | float | 0–1 | Mean AlphaMissense pathogenicity over the chain's mapped variants. |
| `protvar_foldx_mean_a`, `protvar_foldx_mean_b` | float | kcal/mol | Mean monomeric FoldX ΔΔG. See the [FoldX distinction](#monomeric-foldx-δδg-versus-binding-δδg). |
| `protvar_am_n_pathogenic_a`, `protvar_am_n_pathogenic_b` | integer | ≥ 0 | Count of AlphaMissense-pathogenic variants per chain. |
| `protvar_details_a`, `protvar_details_b` | text (`\|`) | | Per-variant records `REF{POS}ALT:am=score:class:foldx=ddg`; cap 20. Every mapped variant in the offline data is scored; no per-position pLDDT filter is applied during scoring. |

## Disease columns — `--disease`

Requires `--enrich`; skipped for `non_human` rows.

| Column | Type | Meaning | Tag |
| ------ | ---- | ------- | --- |
| `n_diseases_a`, `n_diseases_b` | integer | Disease associations per chain (UniProt); read as annotation burden, not causality. | supporting |
| `disease_details_a`, `disease_details_b` | text (`\|`) | Per-disease records (`OMIM:id:label` or `label`); cap 50. | supporting |
| `is_drug_target_a`, `is_drug_target_b` | boolean | Whether the protein is a known drug target. | supporting |
| `n_ptm_sites_a`, `n_ptm_sites_b` | integer | Post-translational-modification sites per chain. | extended |
| `ptm_details_a`, `ptm_details_b` | text (`\|`) | Per-PTM records (`description:position`); cap 50. | extended |
| `go_biological_process_a`, `go_biological_process_b` | text (`\|`) | GO biological-process terms (`GO:id:name`); cap 50. | extended |
| `go_molecular_function_a`, `go_molecular_function_b` | text (`\|`) | GO molecular-function terms (`GO:id:name`); cap 50. | extended |

## Pathway columns — `--pathways`

Requires `--enrich`; skipped for `non_human` rows.

| Column | Type | Range | Meaning | Tag |
| ------ | ---- | ----- | ------- | --- |
| `reactome_pathways_a`, `reactome_pathways_b` | text (`\|`) | | Reactome pathways per chain (`id:name`); cap 20. | supporting |
| `n_reactome_pathways_a`, `n_reactome_pathways_b` | integer | ≥ 0 | Pathway count per chain. | supporting |
| `n_shared_pathways` | integer | ≥ 0 | Pathways containing both proteins. | supporting |
| `pathway_quality_context` | text (`;`) | | Key–value summary `mean_pdockq=..;frac_high=..;n_complexes=..` for the pair's retained pathway; a fourth `enrichment_fdr=..` is appended when a STRING enrichment FDR is available. | extended |
| `ppi_enrichment_pvalue` | float | 0–1 | STRING per-pathway PPI-enrichment p-value for the retained pathway (smallest across shared pathways). Saturates at STRING's floor of `0.0` for large, well-connected pathways, so the ratio is the discriminative measure. Empty under `--no-api`. | supporting |
| `ppi_enrichment_ratio` | float | ≥ 0 | Observed-to-expected PPI-edge ratio for the pathway. Empty under `--no-api`. | supporting |
| `network_degree_a`, `network_degree_b` | integer | ≥ 0 | Degree of each protein in the constructed interaction network. | extended |

## Audit and manifest fields

The 155 columns above are the complete set emitted by `toolkit.py`. Separately, `complex_resolver.py` writes tab-separated audit manifests that are **not** part of the results CSV:

- `complex_manifest.tsv` — one row per complete pair, with columns `name`, `layout`, `shard`, `complex_dir`, `pdb_path`, `pkl_path`, `pdb_size_bytes`, `pkl_size_bytes`, `pdb_mtime_ns`, `pkl_mtime_ns`.
- `incomplete_inputs.tsv` — the same columns plus a trailing `reason` code (`missing_pdb`, `missing_pkl`, `missing_both`, `empty_pdb`, `empty_pkl`, `empty_both`, `ambiguous_pdb`, `ambiguous_pkl`, `duplicate_complex_name`).

These describe input discovery and provenance and are documented with the resolver in [`Toolkit_Commands_List.md`](Toolkit_Commands_List.md).
