# Output schema — data dictionary

This is the field-by-field reference for the per-complex CSV produced by `toolkit.py`. Every complex the toolkit processes becomes one row. The base output has **41 columns**, and each optional stage appends its own block, up to **155 columns** when the full pipeline is run. Reading a column correctly means knowing not only what it contains but when it is populated, how it should be interpreted, and what part it played in the submitted dissertation; this document records all four.

For the commands that produce each column, see [`Toolkit_Commands_List.md`](Toolkit_Commands_List.md). For a grouped overview and the principal workflow, see [`../README.md`](../README.md). For the scientific rationale and reported results, see the submitted dissertation, [`MSc Dissertation Final.pdf`](MSc%20Dissertation%20Final.pdf).

## How to read this document

**Column order.** Columns always appear in the order the stages are assembled: base → enrichment → interface → flags → interface-PAE → clustering → variants → stability → ProtVar → disease → pathways. Enabling a later stage never reorders the earlier columns.

**Empty versus zero.** An empty cell (or `NaN`) means the value was not computed for that row, usually because the stage was not run, the row is non-human and the stage is human-scoped, the row is a partial or error record, or the underlying datum was unavailable. Empty is therefore not the same as zero: a variant count of `0` means the lookup ran and found no variant, whereas an empty variant-count cell means the lookup did not run for that row.

**Per-chain suffixes.** A `_a` / `_b` suffix denotes the two chains of the best-scoring pair (see [best-pair versus all-pairs fields](#best-pair-versus-all-pairs-fields)). Paired columns are documented together below; both are always emitted.

**Serialised fields.** Several columns pack multiple items into one cell. All of them use the pipe character `|` as the item separator, with one exception: `pathway_quality_context` uses semicolons. The per-item sub-format and the truncation cap differ by column:

| Column(s) | Item separator | One item looks like | Cap | Over-cap suffix |
| --------- | :------------: | ------------------- | ---- | --------------- |
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
| `pathway_quality_context` | `;` | `mean_pdockq=0.450;frac_high=0.300;n_complexes=12` | 3 fields, plus optional `enrichment_fdr=` | — |
| `pair_metrics`, `accession_chain_map`, `complex_identity_json` | JSON | see the relevant field entry | — | — |

Note that `homologous_pairs` uses `|+N more` rather than the `|...(+N more)` form used by the other detail columns, and that the `go_*` truncation counts only the terms of the relevant Gene Ontology aspect.

**Thresholds.** The numeric thresholds and composite weights quoted below are fixed module constants; there is no runtime override. They are documented in the dissertation's Methods (§3.3–§3.5) and Appendix B.

### Dissertation-role labels

The submitted dissertation used only some of the emitted columns directly. To make that relationship explicit, each field group carries one of four labels, applied at group level where every field shares the role and marked at field level where a group is mixed:

- **Dissertation core** — fields that define the analysis populations or that directly produce the interface confidence, the tier classification, the screening status, the metric-disagreement analysis, the paradox diagnostic, or the central structural results.
- **Dissertation supporting** — fields used for biological corroboration, prediction-bias analysis, runtime and scaling evidence, structural examples, or supplementary interpretation.
- **Operational/audit** — fields used for input discovery, recoverability, calibration eligibility, species scope, provenance and run auditing.
- **Extended toolkit functionality** — fields emitted by implemented modules that were not material to the submitted argument.

A role label describes how a field was *used in the dissertation*; it says nothing about the field's correctness or general usefulness.

## Canonical population recipes

Every analysis in the dissertation is a filter over the rows of the output CSV, not a separate dataset. The named filters below are implemented in `visualise_filters.py` and applied by `visualise_results.py`; each figure records the filter it used. The conditions are given as the exact field tests so that a population can be reconstructed from a CSV with `pandas`. Row counts are dataset-specific and are not reproduced here.

| Filter | Condition (clauses joined by AND unless stated) | Appropriate for | Main exclusions |
| ------ | ----------------------------------------------- | --------------- | --------------- |
| `recoverable` | `partial_reason` empty or NaN | Any analysis of successfully assessed rows | Partial and error rows |
| `calibrated_dimer` | `tier_scope == "dimer_validated"`; `composite_is_calibrated` parses to `True`; recoverable; the six columns `iptm`, `pdockq`, `interface_confidence_score`, `strict_confident_contact_fraction`, `interface_pae_mean`, `n_interface_contacts` all non-NaN; `n_interface_contacts > 0` | The principal structural-quality, tier and composite analyses | Multimers, partial rows, dimers missing a required metric or with no interface contact |
| `calibrated_human_broad` | `calibrated_dimer` AND `species_status ∈ {reviewed_human, trembl_human}` | Human analyses that tolerate unreviewed entries (variant density) | Non-human rows, and everything `calibrated_dimer` excludes |
| `calibrated_human_strict` | `calibrated_dimer` AND `species_status == reviewed_human` | Human cross-validation requiring curated records (pathogenic-variant, pathway, constraint) | TrEMBL-human and non-human rows |
| `multimer_exploratory` | `tier_scope == "multimer_provisional"` OR `n_chains > 2` | Descriptive multimer inspection only | Everything treated as calibrated |
| `partial_error` | `partial_reason` present and not empty | Auditing incomplete assessments | All recoverable rows |
| `strong_screen_candidate` | `composite_screen_status == "strong_screen_candidate"` | The shortlist of highest-priority candidates | Moderate, weak and unavailable |
| `moderate_screen_candidate` | `composite_screen_status == "moderate_screen_candidate"` | Mid-priority candidates | Strong, weak and unavailable |
| `weak_screen_candidate` | `composite_screen_status == "weak_screen_candidate"` | Lowest-priority candidates | Strong, moderate and unavailable |
| `human_broad` | `species_status ∈ {reviewed_human, trembl_human}` | Species scoping before calibration | Non-human rows |
| `human_strict` | `species_status == reviewed_human` | Reviewed-human scoping before calibration | TrEMBL-human and non-human rows |
| `all_rows` | (no condition) | The full CSV | None |

Two operational filters complete the set: `composite_status_present` (rows whose `composite_screen_status` is non-empty) and `composite_screenable` (present, not `unavailable`, and with a numeric `interface_confidence_score`). The three screen-candidate filters partition the calibrated-dimer population by priority; they are prioritisation strata, not additional quality categories. Each filter degrades gracefully: when a required column is absent it selects no rows, except `all_rows` (always all rows) and `recoverable` (all rows when `partial_reason` is absent).

## Key distinctions

The following distinctions are the ones most often misread. Each is expanded from the definitions in the dissertation's Methods and Appendix C.

### `quality_tier` versus `quality_tier_v2`

`quality_tier` (v1) is assigned from ipTM and pDockQ alone: `High` when `iptm ≥ 0.75` and `pdockq ≥ 0.50`; `Medium` when `iptm ≥ 0.50` and `pdockq ≥ 0.23` and not already High; `Low` otherwise. `quality_tier_v2` starts from that baseline and then uses the composite score as additional interface-localised evidence, under an asymmetric policy: a v1 Low prediction is rescued to Medium at a composite of 0.64 and to High at 0.85, a v1 Medium is promoted to High at 0.85, and a v1 High is demoted to Medium when its composite is 0.63 or below. A v1 Medium is never demoted to Low, because it has already passed both baseline thresholds; and when the composite is unavailable, the v1 tier is retained unchanged. Use `quality_tier_v2` for interface-aware statements and `quality_tier` as the transparent baseline.

### Classification versus `composite_screen_status`

`quality_tier_v2` is a conservative classification anchored in the joint ipTM–pDockQ baseline; `composite_screen_status` is a prioritisation label derived from the composite score alone (`strong_` at 0.85, `moderate_` at 0.63, `weak_` below, or `unavailable`). The two answer different questions, so they can differ without contradiction. A v1 High complex with a composite between 0.63 and 0.85, for instance, remains High because its interface evidence is not weak enough to justify demotion, yet it is only a moderate screen candidate because it does not meet the stronger shortlist threshold. The screening status sits beside the tier; it does not replace it.

### Composite score versus a probability

`interface_confidence_score` is a weighted heuristic on a 0–1 scale: `0.35 P + 0.35 C + 0.15 S + 0.15 D`, where P is normalised mean interface pLDDT, C is the strict confident-contact fraction, S is interface symmetry and D is normalised contact density. Its weights were chosen by design rather than fitted against DockQ, pDockQ2 or experimentally resolved interfaces, and its normalisation and reclassification thresholds were informed by the development and production datasets rather than by labelled outcomes. It therefore ranks predictions; it is not a probability of correctness or an externally calibrated estimate of accuracy, and it carries no positive-predictive-value table.

### Dimer-validated versus multimer-provisional

`tier_scope` is `dimer_validated` for two-chain complexes and `multimer_provisional` otherwise. "Validated" refers only to the scope in which the composite and its reclassification thresholds were calibrated; it does not imply experimental validation. Complexes with more than two chains are still scored structurally — contact count, interface pLDDT and pDockQ are computed for every chain pair and serialised in `pair_metrics`, with the largest-contact pair populating the canonical interface columns — but because neither the composite weights nor the tier adjustments were calibrated for that best-pair representation, those rows are excluded from calibrated composite and tier analyses rather than treated as failures. Restrict any headline confidence claim to `tier_scope == "dimer_validated"`.

### Numeric score versus calibrated score

A row can carry a numeric `interface_confidence_score` without being calibrated. `composite_is_calibrated` is `True` only when the row is a dimer (`tier_scope == "dimer_validated"`), has no recorded `partial_reason`, and contains complete values for the fields required to treat the composite as calibrated. Filter on this flag for a calibrated claim; do not filter merely on the score being present.

### Empty `partial_reason` versus a partial or error row

An empty `partial_reason` marks a row for which no input or processing failure was recorded; whether it then enters a calibrated analysis is decided separately by `tier_scope` and `composite_is_calibrated`. Any non-empty value marks a row whose assessment is incomplete, and such rows are excluded from calibrated-dimer analyses. A partial row is distinct from a low-tier row: a low-tier row was assessed and classified as low confidence, whereas a partial row could not be fully assessed. When more than one condition applies, only the highest-priority reason is stored (the priority order is given in Appendix C of the dissertation). The most severe case, `worker_exception`, denotes a row for which the per-complex pipeline raised an unhandled error and produced a minimal record carrying only identifier and error metadata.

### Best-pair versus all-pairs fields

For a two-chain complex the best-pair columns and the whole-complex aggregates are identical. For a complex with more than two chains they differ, and the aggregation method is not uniform, so it should not be inferred from the column name:

- `pdockq_mean` and `pdockq_min` are an **unweighted** arithmetic mean and minimum over every enumerated pair, including zero-contact pairs, which contribute a pDockQ of `0.0`.
- `interface_plddt_mean`, `symmetry_mean`, `pae_confident_fraction_mean` and `strict_confident_fraction_mean` are **contact-weighted** means that exclude zero-contact pairs (each pair is weighted by its contact count).
- `pdockq_whole_complex` is **recomputed from scratch** — a single pDockQ over the union of all inter-chain contacts, not a mean of the per-pair values.
- `contact_count_total` is the **sum** of contacts across pairs, and `symmetry_min` is the minimum over contact-bearing pairs only.

### PAE-only versus strict confident contacts

Both are fractions of interface contacts over the same denominator. `pae_confident_contact_fraction` counts a contact as confident when its bidirectional PAE — the larger of PAE(i,j) and PAE(j,i), so that both alignment directions must agree — is below 5 Å. `strict_confident_contact_fraction` additionally requires both contacting residues to have pLDDT ≥ 70. The strict fraction is the one that enters the composite score and is a required input for `composite_is_calibrated`; the PAE-only fraction is used by the quality flags and the paradox diagnostic.

### Interface pLDDT versus whole-complex pLDDT

The `interface_plddt_*` columns average pLDDT over interface residues only, whereas `plddt_mean`, `plddt_median` and the like average over the whole complex. `interface_vs_bulk_delta` is `interface_plddt_combined` minus the mean pLDDT of non-interface residues; a positive value indicates that confidence is concentrated at the predicted interface relative to the surrounding structure.

### Monomeric FoldX ΔΔG versus binding ΔΔG

`protvar_foldx_mean_*` reports monomeric FoldX ΔΔG — the estimated change in a single protein's folding stability caused by a variant. It is not a binding or interface ΔΔG and does not directly measure disruption of the interaction.

### Annotation versus structural validation

The enrichment, clustering, variant, disease and pathway columns record annotation: that a protein or pair appears in an external resource, cluster, variant record, disease entry or pathway. This provides biological context and population-level corroboration, but it is not experimental evidence that the modelled interface is correct or that the interaction occurs.

### Species-agnostic structural processing versus human-centred annotation

Structural assessment runs for complexes of any species; species status does not affect whether interface geometry or confidence metrics are computed. The biological annotations, however, draw on human-centred resources. The lightweight enrichment fields — gene symbols, protein names, sequences, Ensembl and secondary accessions, and the interaction-database tags — are attempted for all rows. The six heavier modules — sequence clustering, variant mapping, EVE stability, AlphaMissense with AFDB FoldX, disease annotation and pathway annotation — are applied to reviewed-human and TrEMBL-human rows but skipped for non-human rows, whose corresponding columns are left at their empty defaults. `species_status` records which case applies (`reviewed_human`, `trembl_human` or `non_human`, taking the more restrictive of the two chains).

---

## Base columns (41)

Always emitted. The AlphaFold metrics, pLDDT statistics and `quality_tier` are **dissertation core**; the identity, species and audit fields are **operational/audit**, with the core exceptions noted in the tables.

### Identity and species scope — operational/audit

| Column | Type | Range / values | Description and interpretation |
| ------ | ---- | -------------- | ------------------------------ |
| `schema_version` | text | `multimer_v1` | Output-schema version tag. |
| `complex_name` | text | — | Unique complex identifier and primary key. **Core** — defines every population. |
| `protein_a`, `protein_b` | text | — | UniProt accession of each chain of the best pair. |
| `complex_type` | categorical | `Homodimer` / `Heterodimer` / `Multi-chain` (`None` on a worker-exception row) | Coarse legacy type label; separates homodimers from heterodimers in that analysis. |
| `n_chains` | integer | ≥ 1 | Number of chains; determines `tier_scope` and the multimer populations. |
| `best_chain_pair` | text | e.g. `A_B` | The chain pair with the most interface contacts; drives the best-pair columns. |
| `species` | text | constant tag | Provenance tag (constant column). |
| `structure_source` | text | constant tag | Structure-provenance tag (constant column). |
| `species_a`, `species_b` | categorical | `reviewed_human` / `trembl_human` / `non_human` | Per-chain species classification (isoform suffixes stripped before lookup). |
| `species_status` | categorical | `reviewed_human` / `trembl_human` / `non_human` | Complex-level species tag: the more restrictive of the two chains. Governs which annotation stages run. **Core** for the human analysis populations. |

### AlphaFold model metrics — dissertation core

| Column | Type | Range / unit | Description and interpretation |
| ------ | ---- | ------------ | ------------------------------ |
| `iptm` | float | 0–1 | AlphaFold interface predicted TM-score; a complex-level baseline. |
| `ptm` | float | 0–1 | AlphaFold predicted TM-score. |
| `ranking_confidence` | float | — | AlphaFold ranking confidence (0.8·ipTM + 0.2·pTM), used upstream to select among predictions; distinct from the composite score. |
| `pae_mean` | float | Å | Mean of the whole-complex PAE matrix (all residue pairs). |
| `pdockq` | float | 0–1 | Reproduced pDockQ for the best pair (FoldDock sigmoid, 8 Å Cβ contacts); recorded as `0` when no inter-chain contacts are detected. A contact-count-dependent baseline. |
| `ppv` | float | 0–1 | Positive-predictive-value companion returned with `pdockq`. |

### pLDDT statistics (whole complex) — dissertation core

| Column | Type | Range | Description and interpretation |
| ------ | ---- | ----- | ------------------------------ |
| `plddt_mean`, `plddt_median`, `plddt_min`, `plddt_max` | float | 0–100 | Summary statistics of per-residue pLDDT across the whole complex. |
| `plddt_below50_fraction`, `plddt_below70_fraction` | float | 0–1 | Fraction of residues below the pLDDT 50 / 70 bands. `plddt_below50_fraction` defines the prediction-quality paradox (≥ 0.30). |
| `num_residues` | integer | ≥ 1 | Total residue count. |

### Baseline classification — dissertation core

| Column | Type | Values | Description and interpretation |
| ------ | ---- | ------ | ------------------------------ |
| `quality_tier` | categorical | `High` / `Medium` / `Low` (`Error` on a worker-exception row) | The v1 tier from ipTM and pDockQ. See [distinction](#quality_tier-versus-quality_tier_v2). |

### Multimer identity — operational/audit

| Column | Type | Range / values | Description and interpretation |
| ------ | ---- | -------------- | ------------------------------ |
| `stoichiometry` | text | e.g. `A2`, `AB`, `A2B`, `A2B2`, `ABCD` | Chain stoichiometry; the homo/hetero analysis restricts to `A2` and `AB`. **Core** for that restriction. |
| `is_homomeric` | boolean | | Whether all chains share one accession. |
| `unique_accessions` | integer | ≥ 1 | Number of distinct accessions in the complex. |
| `chain_ids` | text | e.g. `A,B` | PDB chain identifiers. |
| `accession_chain_map` | JSON | | Mapping of accession to chain identifier(s). |
| `tier_scope` | categorical | `dimer_validated` / `multimer_provisional` (`None` on a worker-exception row) | Calibration scope. **Core** — defines the calibrated population. See [distinction](#dimer-validated-versus-multimer-provisional). |
| `filename_n_chains` | integer | | Chain count inferred from the filename. |
| `pdb_n_chains` | integer | | Chain count observed in the PDB. |
| `chain_count_consistency` | categorical | `match` / `filename_only` / `pdb_only` / `mismatch` | Agreement between the filename- and PDB-derived chain counts. |
| `complex_identity_json` | JSON | | Full structured identity record. |

### Audit and recoverability — operational/audit

| Column | Type | Values | Description and interpretation |
| ------ | ---- | ------ | ------------------------------ |
| `has_pdb`, `has_pkl` | boolean | | Whether each source file was found and read. |
| `geometry_available` | boolean | | Whether interface geometry could be calculated. |
| `plddt_source` | categorical | `pdb` / `pkl` | Whether pLDDT was read from the PKL output or the PDB B-factor column. |
| `composite_is_calibrated` | boolean | | Whether the composite is calibrated for this row. **Core** for population definition. See [distinction](#numeric-score-versus-calibrated-score). |
| `partial_reason` | categorical | see below | Empty for a fully assessed row; otherwise a failure code. **Core** for population definition. See [distinction](#empty-partial_reason-versus-a-partial-or-error-row). |

**`partial_reason` vocabulary** (highest to lowest priority; empty = no failure): `worker_exception`, `pdb_decompression_error`, `pdb_io_error`, `pdb_parse_error`, `pdb_no_chains`, `unreadable_pdb_or_structure_input` (legacy fallback), `pkl_decompression_error`, `pkl_io_error`, `pkl_unpickle_error`, `missing_pkl_or_pkl_unreadable` (legacy fallback), `no_positive_interface_contacts`, `pkl_loaded_missing_iptm`, `pkl_loaded_missing_pae`, `missing_required_composite_inputs`, `incomplete_input`. Each value's meaning, and the outputs still retained under it, are tabulated in the dissertation's Appendix C.

---

## Enrichment columns (12) — `--enrich`

**Role: dissertation supporting** (identity underpins the species scoping, the structural examples and the biological analyses), with the cross-reference fields marked **extended**. Unlike the six heavier annotation modules, enrichment is attempted for **all rows**, including non-human ones.

| Column | Type | Description and interpretation | Role |
| ------ | ---- | ------------------------------ | ---- |
| `gene_symbol_a`, `gene_symbol_b` | text | HGNC gene symbol per chain. | supporting |
| `protein_name_a`, `protein_name_b` | text | Full protein name per chain. | supporting |
| `ensembl_id_a`, `ensembl_id_b` | text | Ensembl cross-reference per chain. | extended |
| `secondary_accessions_a`, `secondary_accessions_b` | text (`\|`) | Alternate UniProt accessions (merged or TrEMBL entries); no cap. | extended |
| `database_source` | text (`\|`) | Interaction database(s) in which the pair is found, when `--databases` is active (STRING / BioGRID / HuRI / HuMAP). Records membership, not validation. | supporting |
| `evidence_types` | text (`\|`) | Evidence-type tags from the matched databases. | supporting |
| `sequence_a`, `sequence_b` | text | Amino-acid sequence of each chain (assigned during structural processing, so present for all processed complexes). | supporting |

---

## Interface geometry columns (24) — `--interface`

Includes the `interface_flags` column. **Role: dissertation core**, except the all-pairs aggregates, which are **dissertation supporting** (they underpin the multimer order-statistic diagnostic that justifies the `multimer_provisional` policy).

### Best-pair geometry — dissertation core

| Column | Type | Range / unit | Description and interpretation |
| ------ | ---- | ------------ | ------------------------------ |
| `n_interface_contacts` | integer | ≥ 0 | Inter-chain residue contacts in the best pair (8 Å Cβ, Cα for glycine). A required calibrated-dimer input; must exceed 0. |
| `n_interface_residues_a`, `n_interface_residues_b` | integer | ≥ 0 | Interface residue count contributed by each chain. |
| `interface_residues_a`, `interface_residues_b` | text (`\|`) | | PDB residue numbers at the interface, per chain, sorted ascending; no cap. |
| `interface_fraction_a`, `interface_fraction_b` | float | 0–1 | Fraction of each chain's residues at the interface. |
| `interface_symmetry` | float | 0–1 | Smaller chain interface fraction divided by the larger (1 = balanced). A composite component. |
| `contacts_per_interface_residue` | float | ≥ 0 | Contact density (contacts per unique interface residue). Normalised (÷2, capped at 1) as a composite component. |
| `interface_plddt_a`, `interface_plddt_b` | float | 0–100 | Mean interface pLDDT per chain. |
| `interface_plddt_combined` | float | 0–100 | Mean pLDDT over both chains' interface residues; a composite component after band-normalisation. |
| `bulk_plddt_combined` | float | 0–100 | Mean pLDDT over non-interface residues. |
| `interface_vs_bulk_delta` | float | pLDDT units | `interface_plddt_combined` minus `bulk_plddt_combined`. See [distinction](#interface-plddt-versus-whole-complex-plddt). |
| `interface_plddt_high_fraction` | float | 0–1 | Fraction of interface residues with high pLDDT. |

### All-pairs aggregates — dissertation supporting

| Column | Type | Range / unit | Description and interpretation |
| ------ | ---- | ------------ | ------------------------------ |
| `pair_metrics` | JSON | list, length `C(M,2)` | One record per inter-chain pair (M = chains with Cβ coordinates), including zero-contact pairs. Each record holds the pair's chains and accessions, contact count, interface pLDDT, pDockQ, PPV, symmetry, the two PAE fractions and per-chain interface residues. |
| `pdockq_mean`, `pdockq_min` | float | 0–1 | **Unweighted** mean and minimum of per-pair pDockQ across all pairs; zero-contact pairs contribute `0.0`. |
| `pdockq_whole_complex` | float | 0–1 | pDockQ **recomputed** once over the union of all inter-chain contacts, not a mean of the per-pair values. |
| `contact_count_total` | integer | ≥ 0 | Sum of inter-chain contacts across all pairs. |
| `interface_plddt_mean` | float | 0–100 | **Contact-weighted** mean interface pLDDT; zero-contact pairs excluded. |
| `symmetry_mean` | float | 0–1 | **Contact-weighted** mean interface symmetry; zero-contact pairs excluded. |
| `symmetry_min` | float | 0–1 | Minimum symmetry over contact-bearing pairs. |

### Flags — dissertation core

| Column | Type | Description and interpretation |
| ------ | ---- | ------------------------------ |
| `interface_flags` | text (comma-joined) | Zero or more automated quality flags; empty when none apply. |

The eight possible flags are: `small_interface` (fewer than 5 contacts); `sparse_interface` (low contact density); `asymmetric_interface` (symmetry below 0.5); `interface_better_than_bulk` (interface pLDDT exceeds bulk by more than 10); `low_interface_confidence` (PAE-only confident fraction below 0.2); `paradox_confident_disorder` and `paradox_artefactual` (the two branches of the high-confidence-with-disorder paradox, distinguished by whether the interface contacts are genuinely PAE-confident); and `metric_disagreement` (`|iptm − pdockq| > 0.52`).

---

## Interface PAE columns (19) — `--interface --pae`

**Role: dissertation core**, except the directional PAE diagnostics and the two all-pairs PAE means, which are **dissertation supporting**.

| Column | Type | Range / unit | Description and interpretation |
| ------ | ---- | ------------ | ------------------------------ |
| `interface_pae_mean` | float | Å | Mean interface PAE over the bidirectional contact-level values. A required calibrated-dimer input. |
| `interface_pae_median` | float | Å | Median interface PAE. |
| `n_pae_confident_contacts` | integer | ≥ 0 | Contacts with bidirectional PAE below 5 Å. |
| `pae_confident_contact_fraction` | float | 0–1 | PAE-only confident fraction; used by the flags and the paradox diagnostic. See [distinction](#pae-only-versus-strict-confident-contacts). |
| `n_strict_confident_contacts` | integer | ≥ 0 | Contacts with PAE below 5 Å **and** both residues at pLDDT ≥ 70. |
| `strict_confident_contact_fraction` | float | 0–1 | Strict confident fraction; the PAE component of the composite and a required calibrated-dimer input. |
| `cross_chain_pae_mean` | float | Å | Mean of the cross-chain PAE block. |
| `interface_pae_forward_mean`, `interface_pae_reverse_mean` | float | Å | Directional interface PAE means. *Supporting* — a Methods diagnostic, not a composite input. |
| `interface_pae_directional_delta_mean`, `interface_pae_directional_delta_max` | float | Å | Mean and maximum asymmetry between the two PAE directions. *Supporting.* |
| `n_confident_residues_a`, `n_confident_residues_b` | integer | ≥ 0 | Confident interface residues per chain. |
| `interface_confidence_score` | float | 0–1 | The composite screening score, or empty when any of its four inputs is unavailable. See [distinction](#composite-score-versus-a-probability). |
| `quality_tier_v2` | categorical | `High` / `Medium` / `Low` (`None` on a worker-exception row) | The composite-informed final tier. |
| `composite_screen_status` | categorical | `strong_` / `moderate_` / `weak_screen_candidate` / `unavailable` | The prioritisation label. See [distinction](#classification-versus-composite_screen_status). |
| `pae_confident_fraction_mean`, `strict_confident_fraction_mean` | float | 0–1 | **Contact-weighted** all-pairs PAE fractions; each is empty if any contact-bearing pair lacks PAE. *Supporting.* |
| `composite_is_calibrated` | boolean | | Whether the composite is calibrated for this row (also listed under Base). |

---

## Clustering columns (7) — `--clustering`

Requires `--enrich`; skipped for non-human rows. **Role: dissertation supporting** (the shared-cluster signal appears in the supplementary clustering-validation figure), with the remaining fields **extended**.

| Column | Type | Description and interpretation | Role |
| ------ | ---- | ------------------------------ | ---- |
| `sequence_cluster_ids` | text (`\|`) | Union of the two proteins' STRING cluster memberships; no cap. | extended |
| `sequence_cluster_count` | integer | Number of clusters spanned. | extended |
| `shared_cluster_ids` | text (`\|`) | Clusters containing both proteins (a homology signal). | supporting |
| `shared_cluster_count` | integer | Number of shared clusters. | supporting |
| `homologous_pairs` | text (`\|`) | Detected homologous protein pairs (`a_b`); cap 20 with the `\|+N more` suffix. | extended |
| `n_homologous_pairs` | integer | Count of homologous pairs. | extended |
| `homology_bitscore` | float | STRING pairwise homology bitscore where available; empty under `--no-api`. Not treated as proof of paralogy. | extended |

---

## Variant columns (12) — `--variants`

Requires `--interface --pae --enrich`; applied to human rows, skipped for non-human. **Role: dissertation supporting** — variant burden and constraint provide the biological corroboration and prediction-bias analyses.

| Column | Type | Range | Description and interpretation |
| ------ | ---- | ----- | ------------------------------ |
| `n_variants_a`, `n_variants_b` | integer | ≥ 0 | Mapped variants per chain (UniProt humsavar, with ClinVar significance attached by rsID). |
| `n_interface_variants_a`, `n_interface_variants_b` | integer | ≥ 0 | Variants at interface residues, per chain. |
| `n_pathogenic_interface_variants` | integer | ≥ 0 | Pathogenic interface variants; a value above 0 defines the binary outcome used in cross-validation. |
| `interface_variant_enrichment` | float | fold-change | Interface variant enrichment relative to the rest of the protein. |
| `variant_details_a`, `variant_details_b` | text (`\|`) | | Per-variant records `REF{POS}ALT:context:significance`; context is one of `interface_core` (< 4 Å from a partner residue), `interface_rim` (4–8 Å), `surface_non_interface` (relative solvent accessibility ≥ 25 %) or `buried_core` (< 25 %). Cap 20. |
| `gene_constraint_pli_a`, `gene_constraint_pli_b` | float | 0–1 | ExAC pLI (loss-of-function intolerance) per gene; the complex-level criterion is met when either chain has pLI ≥ 0.9. |
| `gene_constraint_mis_z_a`, `gene_constraint_mis_z_b` | float | z-score | ExAC missense constraint z-score per gene. |

---

## Stability columns (8) — `--stability`

Requires `--variants`; applied to human rows, skipped for non-human. **Role: dissertation supporting** (EVE appears in the supplementary stability cross-validation).

| Column | Type | Range | Description and interpretation |
| ------ | ---- | ----- | ------------------------------ |
| `eve_score_mean_a`, `eve_score_mean_b` | float | 0–1 | Mean EVE pathogenicity score over the chain's mapped variants. |
| `eve_n_pathogenic_a`, `eve_n_pathogenic_b` | integer | ≥ 0 | Count of EVE-pathogenic variants per chain. |
| `eve_coverage_a`, `eve_coverage_b` | float | 0–1 | Fraction of the chain's variants with an EVE score. |
| `stability_details_a`, `stability_details_b` | text (`\|`) | | Per-variant EVE records `REF{POS}ALT:eve=score:class`; cap 20. |

---

## ProtVar columns (8) — `--protvar`

Requires `--variants`; applied to human rows, skipped for non-human. Offline AlphaMissense and monomeric FoldX. **Role: dissertation supporting** (AlphaMissense and monomeric FoldX appear in the supplementary stability cross-validation).

| Column | Type | Range / unit | Description and interpretation |
| ------ | ---- | ------------ | ------------------------------ |
| `protvar_am_mean_a`, `protvar_am_mean_b` | float | 0–1 | Mean AlphaMissense pathogenicity over the chain's mapped variants. |
| `protvar_foldx_mean_a`, `protvar_foldx_mean_b` | float | kcal/mol | Mean **monomeric** FoldX ΔΔG. See [distinction](#monomeric-foldx-δδg-versus-binding-δδg). |
| `protvar_am_n_pathogenic_a`, `protvar_am_n_pathogenic_b` | integer | ≥ 0 | Count of AlphaMissense-pathogenic variants per chain. |
| `protvar_details_a`, `protvar_details_b` | text (`\|`) | | Per-variant records `REF{POS}ALT:am=score:class:foldx=ddg`; cap 20. Every mapped variant present in the offline data is scored; there is no per-position pLDDT filter applied during scoring. |

---

## Disease columns (14) — `--disease`

Requires `--enrich`; applied to human rows, skipped for non-human. **Role: dissertation supporting** for the disease-count and drug-target fields; **extended** for the PTM and Gene Ontology fields, which the submitted analyses did not use.

| Column | Type | Description and interpretation | Role |
| ------ | ---- | ------------------------------ | ---- |
| `n_diseases_a`, `n_diseases_b` | integer | Disease associations per chain (UniProt). Supports the disease-prevalence-by-tier analysis, read as annotation burden rather than causality. | supporting |
| `disease_details_a`, `disease_details_b` | text (`\|`) | Per-disease records (`OMIM:id:label` or `label`); cap 50. | supporting |
| `is_drug_target_a`, `is_drug_target_b` | boolean | Whether the protein is a known drug target. | supporting |
| `n_ptm_sites_a`, `n_ptm_sites_b` | integer | Post-translational-modification sites per chain. | extended |
| `ptm_details_a`, `ptm_details_b` | text (`\|`) | Per-PTM records (`description:position`); cap 50. | extended |
| `go_biological_process_a`, `go_biological_process_b` | text (`\|`) | GO biological-process terms (`GO:id:name`); cap 50. | extended |
| `go_molecular_function_a`, `go_molecular_function_b` | text (`\|`) | GO molecular-function terms (`GO:id:name`); cap 50. | extended |

---

## Pathway columns (10) — `--pathways`

Requires `--enrich`; applied to human rows, skipped for non-human. **Role: dissertation supporting** for the Reactome and PPI-enrichment fields (pathway enrichment is a corroboration signal, and the pathway figures are supplementary); **extended** for the network-topology field.

| Column | Type | Range | Description and interpretation | Role |
| ------ | ---- | ----- | ------------------------------ | ---- |
| `reactome_pathways_a`, `reactome_pathways_b` | text (`\|`) | | Reactome pathways per chain (`id:name`); cap 20. | supporting |
| `n_reactome_pathways_a`, `n_reactome_pathways_b` | integer | ≥ 0 | Pathway count per chain. | supporting |
| `n_shared_pathways` | integer | ≥ 0 | Pathways containing both proteins. | supporting |
| `pathway_quality_context` | text (`;`) | | Key–value summary `mean_pdockq=..;frac_high=..;n_complexes=..` for the pair's retained pathway; a fourth `enrichment_fdr=..` field is appended when a STRING enrichment FDR is available. | extended |
| `ppi_enrichment_pvalue` | float | 0–1 | STRING per-pathway PPI-enrichment p-value for the retained pathway (smallest across shared pathways). Saturates at STRING's floor of 0.0 for large, well-connected pathways, so the ratio is the discriminative measure. Empty under `--no-api`. | supporting |
| `ppi_enrichment_ratio` | float | ≥ 0 | Observed-to-expected PPI-edge ratio for the pathway; the measure used in the tier comparison. Empty under `--no-api`. | supporting |
| `network_degree_a`, `network_degree_b` | integer | ≥ 0 | Degree of each protein in the constructed interaction network. | extended |

---

## A note on audit-only fields

The 155 columns above are the complete set emitted by `toolkit.py`. Separately, `complex_resolver.py` writes tab-separated **audit manifests** (`complex_manifest.tsv`, `incomplete_inputs.tsv`) whose columns — for example `layout`, `shard`, `pdb_path`, `pdb_size_bytes`, `pdb_mtime_ns`, and the `reason` code on incomplete inputs — describe input discovery and provenance. Those manifest columns are not part of the results CSV and are documented with the resolver in [`Toolkit_Commands_List.md`](Toolkit_Commands_List.md).
