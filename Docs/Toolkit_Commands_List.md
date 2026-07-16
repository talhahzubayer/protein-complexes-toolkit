# Toolkit command reference

This is the complete operational reference for the toolkit. It documents the commands needed to reproduce the dissertation workflow, every implemented command-line option, the standalone module tools, and the functionality that exists in the codebase but was not used in the submitted dissertation. For the principal reproduction workflow in narrative form, see [`../README.md`](../README.md); for the output fields, see [`OUTPUT_SCHEMA.md`](OUTPUT_SCHEMA.md); for the scientific rationale and results, see [`MSc Dissertation Final.pdf`](MSc%20Dissertation%20Final.pdf).

## How to read this document

Commands use `python` (the Windows convention). On Linux or macOS, substitute `python3` where `python` does not resolve to Python 3, and use forward slashes in paths.

Each command group carries a label describing its relationship to the submitted dissertation, because the toolkit implements more than the dissertation used:

- **Dissertation workflow** — used to produce, or required to reproduce, the submitted analyses.
- **Operational support** — infrastructure for running and auditing the pipeline.
- **Additional functionality** — implemented and usable, but not part of the submitted analyses.
- **Exploratory / supplementary** — descriptive or diagnostic outputs kept outside the main results.
- **Legacy compatibility** — retained for older inputs or behaviour.

The presence of a command here does not imply that its output was evaluated or reported in the dissertation; the label states whether it was.

### Placeholder reference

Replace the angle-bracket placeholders with your own paths before running.

| Placeholder | Meaning | Default if omitted |
|-------------|---------|--------------------|
| `<MODELS_DIR>` | Directory of AlphaFold2-Multimer PDB and PKL outputs | Required |
| `<OUTPUT_CSV>` | Output CSV path | `batch_results.csv` |
| `<ALIASES_FILE>` | STRING aliases file | `data/ppi/9606.protein.aliases.v12.0.txt` |
| `<PPI_DIR>` | PPI database directory | `data/ppi/` |
| `<CLUSTERS_FILE>` | STRING clusters file | `data/clusters/9606.clusters.proteins.v12.0.txt` |
| `<VARIANTS_DIR>` | Variant database directory | `data/variants/` |
| `<STABILITY_DIR>` | EVE data directory | `data/stability/` |
| `<FOLDX_EXPORT>` | AFDB FoldX export CSV | `data/stability/afdb_foldx_export_20250210.csv` |
| `<AM_FILE>` | AlphaMissense TSV | `data/stability/AlphaMissense_aa_substitutions.tsv` |
| `<PATHWAYS_DIR>` | Disease/Reactome directory | `data/pathways/` |
| `<OUTPUT_DIR>` | Figure output directory | `Output/` |
| `<INTERFACES_JSONL>` | Interface JSONL export path | none (user-specified) |
| `<PYMOL_OUTPUT>` | PyMOL script output directory | `pymol_scripts/` |

## Dissertation reproduction workflow

**Dissertation workflow.** The submitted analyses were produced by the following sequence: validate the external data, audit the prediction inputs, run the full pipeline (locally for development, and on the HPC cluster for the full dataset), consolidate any incremental output, and generate the figures from the consolidated CSV.

```bash
# 1. Confirm the external data is present
python data_registry.py

# 2. Catalogue and audit the prediction inputs
python complex_resolver.py --root <MODELS_DIR>

# 3. Run the full pipeline (writes the results CSV and the interface JSONL)
python toolkit.py --full-pipeline --dir <MODELS_DIR> -w 8 \
    --output results.csv --export-interfaces interfaces.jsonl

# 4. Generate the dissertation figure suite from the consolidated CSV
python visualise_results.py results.csv --output-dir Output --full-figure-pack --human-supplement
```

The single-complex structural examples (the dissertation's Figure 6) are produced separately by `render_complex_summary.py` (see [Single-complex structural summary](#single-complex-structural-summary)). At full dataset scale, steps 3 and 4 are run through the SLURM wrappers described in [Initial HPC wrapper](#initial-hpc-wrapper) and [Incremental HPC wrapper](#incremental-hpc-wrapper).

## Main toolkit command

`toolkit.py` processes a directory of predictions and writes one CSV row per complex. Each feature flag adds a block of columns; flags compose, subject to the dependencies below. The default output is `batch_results.csv` and the default worker count is 1 (sequential).

### Flag dependencies and implications

The toolkit enforces these at start-up. An auto-enable prints a note and continues; a requirement failure prints an error and exits with status 1.

| Flag | Requires | Auto-enables |
|------|----------|--------------|
| `--pae` | — | `--interface` |
| `--export-interfaces` | — | `--interface --pae` |
| `--databases` | `--enrich` | — |
| `--clustering` | `--enrich` | — |
| `--variants` | `--interface --pae --enrich` | — |
| `--stability` | `--variants` | — |
| `--protvar` | `--variants` | — |
| `--disease` | `--enrich` | — |
| `--pathways` | `--enrich` | — |
| `--pymol` | `--interface --pae` | — |
| `--resume` | — | `--checkpoint` |
| `--full-pipeline` | `--dir` only | all feature flags + `--checkpoint` |

`--protvar` additionally checks that the FoldX export and the AlphaMissense file exist, and `--full-pipeline` validates every required data file before processing starts; a missing file causes an immediate exit.

### Core structural processing — Dissertation workflow

```bash
# Base metrics only (no interface features)
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV>

# With interface geometry and PAE features (the structural core; no external data needed)
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV> --interface --pae
```

| Flag | Type / default | Effect |
|------|----------------|--------|
| `--dir` | required | Directory of PDB/PKL inputs (loose, flat-dir or sharded layout). |
| `--output` | `batch_results.csv` | Output CSV path; also the base name for the checkpoint file. |
| `--interface` | off | Compute interface geometry and interface pLDDT. |
| `--pae` | off | Compute PAE-based interface features; auto-enables `--interface`. |
| `--verbose` / `-v` | off | Per-complex progress (suppressed when workers > 1). |
| `--workers` / `-w` | `1` | Parallel workers for the structural pass. |

### Enrichment stages — Dissertation workflow

The lightweight enrichment (`--enrich`) runs for all rows; the heavier stages are applied to human rows only.

```bash
# Enrichment: gene symbols, protein names, sequences, cross-references
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV> --interface --pae --enrich <ALIASES_FILE>

# + interaction-database source tagging
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV> --interface --pae --enrich <ALIASES_FILE> --databases <PPI_DIR>

# + sequence clustering, variant mapping, EVE, AlphaMissense/FoldX, disease, pathways
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV> --interface --pae --enrich <ALIASES_FILE> \
    --clustering --variants --stability --protvar --disease --pathways
```

| Flag | Type / default | Effect |
|------|----------------|--------|
| `--enrich` | path, off | Enrich with gene symbols, protein names and cross-references from a STRING aliases file. |
| `--databases` | path, off | Tag each pair with its interaction-database sources (STRING, BioGRID, HuRI, HuMAP). |
| `--string-min-score` | `700` | Minimum STRING confidence score for database matching; only used with `--databases`. |
| `--clustering` | `{string,foldseek,hybrid}`, off; bare flag → `string` | Sequence-cluster and homologous-pair detection. `foldseek` and `hybrid` are placeholders. |
| `--clusters-file` | path, default clusters file | Override the STRING clusters file. |
| `--variants` | path, off; bare flag → `data/variants/` | Map UniProt/ClinVar/ExAC variants and classify structural context. |
| `--no-clinvar` | off | Skip ClinVar loading (UniProt + ExAC only, faster). A modifier that only takes effect alongside `--variants`; it is ignored otherwise rather than reported as an error. |
| `--stability` | path, off; bare flag → `data/stability/` | EVE variant-effect scoring. |
| `--protvar` | path, off; bare flag → FoldX export default | Offline AlphaMissense + monomeric FoldX scoring. |
| `--am-file` | path, default AlphaMissense TSV | Override the AlphaMissense file (only with `--protvar`). |
| `--disease` | path, off; bare flag → `data/pathways/` | UniProt disease, PTM, GO and drug-target annotation. |
| `--pathways` | off | Reactome pathway mapping and per-pathway PPI enrichment. |
| `--no-api` | off | Disable the STRING API fallback; use offline data only (also skips STRING PPI enrichment under `--pathways`). |

### Output controls — Operational support

```bash
# Export confident interface residues to JSONL (implies --interface --pae; High/Medium v2 tier only)
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV> --export-interfaces <INTERFACES_JSONL>
```

`--export-interfaces <PATH>` writes one JSON record per complex containing the confident interface residues, PAE and per-residue pLDDT. It auto-enables `--interface --pae` and exports only complexes that reach the High or Medium v2 tier.

### Checkpoint and resume controls — Operational support

```bash
# Parallel run with periodic checkpointing (every 50 complexes to <output>.checkpoint.jsonl)
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV> --interface --pae -w 8 --checkpoint

# Resume an interrupted run from its checkpoint (implies --checkpoint)
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV> --interface --pae -w 8 --resume
```

| Flag | Type / default | Effect |
|------|----------------|--------|
| `--checkpoint` | off | Save progress every 50 complexes to `<output>.checkpoint.jsonl`. |
| `--resume` | off | Resume from that checkpoint, skipping already-processed complexes; auto-enables `--checkpoint`. |

### Incremental and chunked processing — Dissertation workflow

These flags supported the memory-bounded batching used to assemble the full dataset.

```bash
# Append-only incremental mode: process only complexes absent from the historical CSV
python toolkit.py --full-pipeline --dir <MODELS_DIR> \
    --skip-existing results.csv \
    --output results_incremental_<stamp>_<job>.csv \
    --export-interfaces interfaces_incremental_<stamp>_<job>.jsonl

# Chunk the incremental delta under a memory ceiling with --limit N
python toolkit.py --full-pipeline --dir <MODELS_DIR> \
    --skip-existing results.csv --limit 100000 \
    --output results_incremental_<stamp>_<job>.csv \
    --export-interfaces interfaces_incremental_<stamp>_<job>.jsonl
```

| Flag | Type / default | Effect |
|------|----------------|--------|
| `--skip-existing` | path, off | Read `complex_name` values from a historical results CSV and process only those not present. Reads that CSV only. |
| `--limit` | int, off | Process at most N complexes, taken from the alphabetically-sorted post-`--skip-existing` delta. Must be a positive integer. |

`--skip-existing` must point at the full historical `results.csv`, not a filtered subset, or partial and zero-contact rows are reprocessed on every run. The chunk membership is fixed *before* any `--resume` filtering, so a crashed-and-resumed chunk covers the same complexes as a clean run, provided the historical CSV is unchanged between attempts. Incremental output is written to separate files and must be consolidated before it becomes the reference for the next run (see [Consolidating incremental output](#consolidating-incremental-output)).

### PyMOL generation — Additional functionality

```bash
# Generate .pml scripts for High-tier complexes (requires --interface --pae)
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV> --interface --pae --pymol

# Lower the tier threshold, set an output directory, include ray-tracing commands
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV> --interface --pae \
    --pymol --pymol-min-tier Medium --pymol-output <PYMOL_OUTPUT> --pymol-render
```

| Flag | Type / default | Effect |
|------|----------------|--------|
| `--pymol` | off | Generate layered PyMOL `.pml` scripts for qualifying complexes. |
| `--pymol-output` | `pymol_scripts/` | Output directory (sharded into subdirectories of ≤ 1000 scripts). |
| `--pymol-render` | off | Include ray-tracing and PNG commands for `pymol -c` batch rendering. |
| `--pymol-min-tier` | `{High,Medium,Low}`, `High` | Minimum v2 tier for script generation. |

### Full pipeline — Dissertation workflow

```bash
python toolkit.py --full-pipeline --dir <MODELS_DIR> -w 8 --output results.csv --export-interfaces interfaces.jsonl
```

`--full-pipeline` activates `--interface --pae --enrich --databases --clustering --variants --stability --protvar --disease --pathways --pymol --checkpoint` with the registered default data paths, and validates that every required data file exists before starting.

## Figure generation

`visualise_results.py` reads a results CSV and writes PNG figures to `Output/` by default. Each figure records its population scope and denominator, and is skipped with a message when its required columns are absent.

### Dissertation analysis outputs — Dissertation workflow

The command that produced the dissertation figure suite is:

```bash
python visualise_results.py results.csv --output-dir Output --full-figure-pack --human-supplement
```

The output filenames use the dissertation's Results numbering. **The internal plotting-function names retain an older numbering and no longer match the filenames, so they must not be used to identify a figure.** The mapping between output file and dissertation figure is:

| Output filename | Dissertation figure | Content | Emitted by |
|-----------------|:-------------------:|---------|------------|
| `Fig_1_Dataset_and_Analysis_Population_Funnel.png` | Fig 1 | Nested population funnel with screening side-callouts | default |
| `Fig_2A_Interface_PAE_by_Quality_Tier.png` | Fig 2A | Interface PAE distribution across v2 tiers | default |
| `Fig_2B_Interface_pLDDT_vs_Bulk_pLDDT.png` | Fig 2B | Interface vs bulk pLDDT, with paradox subset marked | default |
| `Fig_3_Composite_Score_Behaviour.png` | Fig 3 | Composite by tier, and composite vs strict confident-contact fraction | default |
| `Fig_4_Classification_Versus_Screening.png` | Fig 4 | Composite screening bands, and tier × screen-status crosstab | default |
| `Fig_5_ipTM_pDockQ_Metric_Disagreement.png` | Fig 5 | Categorical agreement matrix of ipTM-only vs pDockQ-only classes | default |
| `Fig_6_Prediction_Quality_Paradox.png` | Fig 7 | Paradox vs non-paradox: interface-minus-bulk pLDDT, PAE-only fraction, symmetry | `--full-figure-pack` |
| `Fig_7_Variant_Density_Versus_Composite_Confidence.png` | Fig 8 | Interface variant density vs composite score (broad-human) | default |
| `Fig_8_Biological_Corroboration_and_Prediction_Bias.png` | Fig 9 | Pathogenic-variant, PPI-enrichment, pLI and disorder by tier (reviewed-human) | default |
| *(separate script)* | Fig 6 | Two structural examples | `render_complex_summary.py` |

Two points follow from the numbering. First, because the dissertation's Figure 6 is the structural-examples figure produced separately, the `Fig_6`, `Fig_7` and `Fig_8` output files correspond to the dissertation's Figures **7, 8 and 9**. Second, two filenames are produced by functions whose names describe different content: `Fig_5_ipTM_pDockQ_Metric_Disagreement.png` is generated by a function named for a delta histogram, but its current output is the categorical agreement matrix used in the dissertation; and `Fig_8_Biological_Corroboration_and_Prediction_Bias.png` is generated by a function named for the paradox, but its current output is the biological-corroboration panel. These are documentation-only observations; the code is not changed here.

The paradox figure (`Fig_6_Prediction_Quality_Paradox.png`, the dissertation's Figure 7) is emitted only under `--full-figure-pack`, which is why the reproduction command includes that flag. `--human-supplement` additionally re-renders the structural figures on the human subset, writing `_human`-suffixed copies.

### Additional and supplementary outputs — Exploratory / supplementary

The default run also emits two figures that use the same data but are not part of the submitted main figure set: `1_Quality_Scatter.png` (ipTM vs pDockQ coloured by v2 tier) and `7_Homo_vs_Hetero.png` (tier proportions and interface symmetry for homodimers versus heterodimers). Note that these two lack the `Fig_` filename prefix.

`--full-figure-pack` adds the supplementary (`*_supp_*`) figures, data permitting: `1b_supp_Disorder_Scatter`, `2_supp_PAE_Health_Check`, `4_supp_Strict_vs_PAE_Only_Fraction`, `8_supp_iptm_pdockq_scatter`, `9_supp_Chain_Count_Profile` (an order-statistic diagnostic supporting the multimer-provisional policy), `10_supp_Clustering_Validation`, `11_supp_Variant_Consequence_Flow`, `13_supp_Stability_CrossValidation`, `14A_supp_Disease_Prevalence_by_Tier`, `14B_supp_Top_Disease_Categories_by_Tier`, `15_supp_Pathway_Bar_Chart`, `15_supp_Pathway_Network` and `18_supp_Partial_Reason_Dashboard`. Each requires the columns from its stage.

| Flag | Default | Effect |
|------|---------|--------|
| `--output-dir` | `Output/` | Figure output directory. |
| `--full-figure-pack` | off | Emit every `*_supp_*` figure; implies `--disorder-scatter` and `--include-partial-diagnostics`. |
| `--human-supplement` | off | Re-render the structural figures on the reviewed+TrEMBL-human subset (`_human` suffix). |
| `--nonhuman-supplement` | off | Re-render on the non-human subset (`_nonhuman` suffix); the chain-count profile is skipped on this pass. |
| `--multimer-supplement` | off | Emit the multimer-exploratory stoichiometry panel (`7_supp_Multimer_Stoichiometry`), descriptive only. |
| `--disorder-scatter` | off | Emit the disorder-coloured scatter (`1b_supp_Disorder_Scatter`) on its own. |
| `--include-partial-diagnostics` | off | Emit the partial-reason dashboard (`18_supp_Partial_Reason_Dashboard`) on its own. |
| `--density` | off | Add KDE density contours to the scatter figures. |
| `--no-corpus-funnel` | funnel on | Suppress `Fig_1` (the population funnel). |
| `--no-screening-figures` | screening on | Suppress `Fig_4` (classification vs screening). |
| `--skip-diagnostics` | off | Suppress the per-figure missing-row summaries. |
| `--legacy-mode` | off | Restore the older data-loading behaviour that dropped rows with missing or zero ipTM (see [Legacy compatibility](#legacy-compatibility)). |

The species suffix is applied only to the structural figures and only when `--human-supplement` or `--nonhuman-supplement` is set and the CSV has a `species_status` column; a plain run emits unsuffixed files only.

### Per-complex PAE heatmaps — Exploratory / supplementary

```bash
# One PAE-matrix heatmap per complex, read from the PKL files in a models directory
python visualise_results.py results.csv --pae-heatmaps <MODELS_DIR>

# Cap the number generated
python visualise_results.py results.csv --pae-heatmaps <MODELS_DIR> --limit 10
```

`--pae-heatmaps <DIR>` reads each `*.pkl` in the given directory and writes a `<pkl_stem>_PAE.png` heatmap **into that directory** (not the figure output directory), with chain-boundary lines and the best-pair cross-chain block highlighted. `--limit N` caps the count and affects only this mode.

## Single-complex structural summary

**Dissertation workflow** (the dissertation's Figure 6). `render_complex_summary.py` renders one dimer into a single polished PNG, reusing the production metric path so that the metrics shown match the batch pipeline exactly.

```bash
python render_complex_summary.py \
    --input-dir <COMPLEX_DIR> \
    --output    complex_summary.png \
    --pymol-executable "$HOME/envs/pymol/bin/pymol"
```

| Flag | Type / default | Effect |
|------|----------------|--------|
| `--input-dir` | required | Directory containing exactly one PDB and one PKL (`.pdb`/`.pdb.gz`/`.pdb.bz2` and `.pkl`/`.pkl.gz`/`.pkl.bz2`). |
| `--output` | required | Output PNG path (must end in `.png`). |
| `--overwrite` | off | Replace an existing output; the default is to fail rather than overwrite. |
| `--width` | `2000` | Output width in pixels (positive integer). |
| `--height` | `1600` | Output height in pixels (positive integer). |
| `--pymol-executable` | none | Path to a headless PyMOL executable. |

The script is **dimer-only**: a non-dimer or non-calibrated input fails with a clear message and exit status 1. It requires no network access, writes the PNG atomically, and cleans up its intermediate files. It resolves the PyMOL executable from `--pymol-executable`, then the `PYMOL_EXECUTABLE` environment variable, then `pymol` on the `PATH`. Where no PyMOL module is available, install one once:

```bash
conda create -y -p "$HOME/envs/pymol" -c conda-forge pymol-open-source
"$HOME/envs/pymol/bin/pymol" -cq -d "print('PyMOL', cmd.get_version()[0]); quit"   # verify headless
export PYMOL_EXECUTABLE="$HOME/envs/pymol/bin/pymol"                                # optional: avoid the flag
```

Each invocation produces one PNG for one complex. The dissertation's two-example figure was assembled by rendering the two examples separately and composing them side by side; **that final composition is a manual step and is not performed by the script.**

## Input auditing and data validation — Operational support

### Data dependency validation

```bash
# Validate all required data files
python data_registry.py

# Validate specific groups only
python data_registry.py --groups ppi-databases variant-mapping

# Override the project root
python data_registry.py --root /path/to/project
```

`data_registry.py` checks the registered data files (16 required files; two further registry entries are output directories the pipeline creates, which are not checked) and prints a per-group report to standard error, exiting with status 1 if any required file is missing. The only arguments are `--groups` (choosing from `ppi-databases`, `clustering`, `variant-mapping`, `eve-stability`, `offline-scoring`, `disease-pathways`, `pymol`) and `--root`. The project root is resolved as: an explicit `--root`, then the `PROTEIN_TOOLKIT_PROJECT_ROOT` environment variable, then the repository directory.

### Input discovery and the audit manifest

```bash
# Audit a models directory (writes the forensic manifest)
python complex_resolver.py --root <MODELS_DIR>

# Capture a pre-expansion baseline, tagging the run id
python complex_resolver.py --root <MODELS_DIR> --purpose baseline
```

| Flag | Default | Effect |
|------|---------|--------|
| `--root` | `PROTEIN_COMPLEXES_ROOT` env var | Models directory to scan. |
| `--audit-dir` | `data/complex_manifest_audit/` | Override the audit output directory. |
| `--purpose` | `{baseline,incremental}`, `baseline` | Suffix embedded in the auto-generated run id. |
| `--run-id` | auto | Override the run id (intended for tests and reproducible fixtures). |

The resolver recognises two directory layouts: sharded (any child directory matching `^[A-Z0-9]{2}$`) and flat-dir (each child of the root is a complex directory). The loose layout, where PDB/PKL files sit directly in the root, is handled by `toolkit.py` itself and does not invoke the resolver. For each run it writes, under `data/complex_manifest_audit/runs/<run_id>/`, a `complex_manifest.tsv` of complete pairs, an `incomplete_inputs.tsv` with a `reason` column (`missing_pdb`, `missing_pkl`, `missing_both`, `empty_pdb`, `empty_pkl`, `empty_both`, `ambiguous_pdb`, `ambiguous_pkl`, `duplicate_complex_name`), and a run summary; a `latest/` mirror and a `latest_run_id.txt` pointer track the most recent run. It exits `0` when at least one complete pair is found and `1` otherwise. It accepts `.pdb`/`.pdb.bz2` structures and `.pkl`/`.pkl.bz2`/`.results.pkl`/`.results.pkl.bz2` confidence files (the long-form `*_relaxed_model_*` / `*_result_model_*` names are also matched); `.gz` inputs are recognised only in the loose layout handled by the toolkit.

## Initial HPC wrapper

**Dissertation workflow.** `hpc_dataset_run.sh` submits a clean, end-to-end production run to a SLURM cluster.

```bash
# Edit the two pinned paths near the top of hpc_dataset_run.sh for your site:
#   export PROTEIN_TOOLKIT_PROJECT_ROOT=/scratch/<project>/protein-complexes-toolkit-hpc
#   export PROTEIN_COMPLEXES_ROOT=/scratch/<project>/Protein_Complexes
sbatch hpc_dataset_run.sh
```

The wrapper sets those two paths itself (they are hard-coded `export` lines, so exporting them in the submitting shell has no effect — edit the script). It loads the Python module, activates the project virtual environment, caps the OpenMP, MKL, OpenBLAS and NumExpr thread counts at one to prevent CPU oversubscription across worker processes, and runs five phases: a `pip check`, `data_registry.py`, `complex_resolver.py`, `toolkit.py --full-pipeline` (writing `results.csv`, `interfaces.jsonl` and `pymol_scripts/`), and `visualise_results.py`. The figure step reads the `VISUALISE_ARGS` environment variable, which defaults to `--full-figure-pack --human-supplement`.

The `#SBATCH` header requests 16 CPUs, 80 GB of memory and a 48-hour walltime. These are the script's defaults, not general requirements. For reference, the dissertation's initial 41,196-complex run completed in under six hours at a peak of about 67 GB, but the completed dataset was assembled through memory-bounded batches of up to 100,000 complexes under a 128 GB allocation, because the current design accumulates results in memory.

## Incremental HPC wrapper

**Dissertation workflow.** `hpc_incremental_run.sh` processes only complexes absent from a cumulative historical CSV, and does not render figures. Capture a baseline manifest once before the corpus is expanded (`python complex_resolver.py --root "$PROTEIN_COMPLEXES_ROOT" --purpose baseline`), then submit:

```bash
sbatch --export=ALL,HISTORICAL_RESULTS_CSV=results.csv hpc_incremental_run.sh
```

It mirrors the initial wrapper's environment hardening and runs a four-step preflight (`pip check`; `data_registry.py`; a check that the baseline manifest exists at `data/complex_manifest_audit/latest/`; and a sanity check that the historical CSV and JSONL are consistent) before invoking `toolkit.py --full-pipeline --skip-existing`. It logs the historical CSV's SHA-256 at start so the baseline can be confirmed unchanged across a retry.

| Environment variable | Default | Purpose |
|----------------------|---------|---------|
| `HISTORICAL_RESULTS_CSV` | `results.csv` | The `--skip-existing` reference (the full historical CSV). |
| `HISTORICAL_INTERFACES_JSONL` | `interfaces.jsonl` | The paired historical JSONL. |
| `OUTPUT_CSV` | `results_incremental_<stamp>_<job>.csv` | Per-run incremental CSV. |
| `INTERFACES_JSONL` | `interfaces_incremental_<stamp>_<job>.jsonl` | Per-run incremental JSONL. |
| `RUN_STAMP`, `JOB_TAG` | date / job id | Override the stamp and tag in the output filenames. |
| `RESUME` | `0` | Set to `1` to add `--resume` for crash recovery. |
| `LIMIT` | unset | Positive integer; adds `--limit N` for memory-bounded chunking. |

**Chunked runs.** Set `LIMIT` to process the delta in memory-bounded batches; between batches, consolidate the incremental output into the cumulative CSV and submit the next batch against the updated file:

```bash
sbatch --export=ALL,HISTORICAL_RESULTS_CSV=results.csv,LIMIT=100000 hpc_incremental_run.sh
```

**Crash recovery.** Resubmit against the same partial output files with `RESUME=1`, keeping `HISTORICAL_RESULTS_CSV` identical to the original attempt so the batch boundary does not shift:

```bash
sbatch --export=ALL,RESUME=1,\
HISTORICAL_RESULTS_CSV=results.csv,\
OUTPUT_CSV=results_incremental_<stamp>_<job>.csv,\
INTERFACES_JSONL=interfaces_incremental_<stamp>_<job>.jsonl \
  hpc_incremental_run.sh
```

### Consolidating incremental output

**Operational support.** Incremental and chunked runs write to separate files and do not modify the historical CSV. Each output must be merged into the cumulative CSV and JSONL before the merged CSV is promoted as the next `HISTORICAL_RESULTS_CSV`; otherwise the wrapper reproduces the same delta. **There is no merge script in this repository** — this is a manual step, and the dissertation lists automatic merging as future work. When merging, use a CSV/JSON-aware tool (the Python standard library `csv` and `json` modules) rather than line-based text tools, because quoted fields can contain commas, and verify:

1. the incremental and historical headers are identical;
2. no `complex_name` value appears in both files;
3. the merged row count equals the sum of the two inputs;
4. every complex named in a JSONL record is present in the merged CSV;
5. a backup of the historical files exists before the merge;
6. the merged file replaces the historical one by an atomic rename.

Then run `visualise_results.py` on the merged CSV to regenerate the figures.

## Standalone module commands

Each downstream module has its own command-line interface. These are **additional functionality**: they expose stages of the pipeline for inspection and were not, individually, part of the submitted analyses.

**Database loading** (`database_loaders.py`): parse the PPI databases.

```bash
python database_loaders.py --data-dir <PPI_DIR>                         # load all, print a summary
python database_loaders.py --data-dir <PPI_DIR> --database string       # one database
python database_loaders.py --data-dir <PPI_DIR> --output all.csv        # export to CSV
python database_loaders.py --data-dir <PPI_DIR> --min-string-score 700  # STRING confidence filter
```

**ID mapping** (`id_mapper.py`): resolve identifiers via the STRING aliases.

```bash
python id_mapper.py --aliases <ALIASES_FILE> --stats
python id_mapper.py --aliases <ALIASES_FILE> --resolve P04637
python id_mapper.py --aliases <ALIASES_FILE> --export lookup_table.csv
```

**Overlap analysis** (`overlap_analysis.py`): cross-database overlap and Venn/UpSet diagrams (`--aliases` is required).

```bash
python overlap_analysis.py --data-dir <PPI_DIR> --aliases <ALIASES_FILE>
python overlap_analysis.py --data-dir <PPI_DIR> --aliases <ALIASES_FILE> --base-level --report <OUTPUT_DIR>/overlap_report.txt
```

**Protein clustering** (`protein_clustering.py`): STRING clusters and shared membership (the standalone CLI requires `--clusters-file`).

```bash
python protein_clustering.py --clusters-file <CLUSTERS_FILE> --aliases <ALIASES_FILE> --summary
python protein_clustering.py --clusters-file <CLUSTERS_FILE> --aliases <ALIASES_FILE> --pair P04637 Q00987
```

**Variant mapping** (`variant_mapper.py`): subcommands `summary`, `lookup`, `map`.

```bash
python variant_mapper.py summary --variants-dir <VARIANTS_DIR>
python variant_mapper.py lookup --variants-dir <VARIANTS_DIR> --protein P04637
python variant_mapper.py map --interfaces <INTERFACES_JSONL> --pdb-dir <MODELS_DIR> --variants-dir <VARIANTS_DIR> --output variant_analysis.csv
```

**Stability scoring** (`stability_scorer.py`): subcommands `summary`, `lookup`.

```bash
python stability_scorer.py --stability-dir <STABILITY_DIR> summary
python stability_scorer.py --stability-dir <STABILITY_DIR> lookup --protein P61981 --position 45
```

**Offline AlphaMissense + FoldX** (`protvar_client.py`): subcommands `summary`, `lookup`.

```bash
python protvar_client.py summary
python protvar_client.py lookup --protein P61981 --position 4
```

**Disease annotation** (`disease_annotations.py`): subcommands `summary`, `lookup`.

```bash
python disease_annotations.py summary --disease-dir <PATHWAYS_DIR>
python disease_annotations.py lookup --disease-dir <PATHWAYS_DIR> --protein P04637
```

**Pathway network** (`pathway_network.py`): subcommands `summary`, `network`, `enrichment`.

```bash
python pathway_network.py summary --csv <OUTPUT_CSV>
python pathway_network.py network --csv <OUTPUT_CSV> --output-dir <OUTPUT_DIR>
python pathway_network.py enrichment --csv <OUTPUT_CSV>
```

**PyMOL scripts** (`pymol_scripts.py`): subcommands `generate` (single PDB, written flat) and `batch` (from a CSV, sharded into `shard_NNNN/`).

```bash
python pymol_scripts.py generate --pdb <PDB_FILE> --render
python pymol_scripts.py batch --csv <OUTPUT_CSV> --pdb-dir <MODELS_DIR> --min-tier Medium --output <PYMOL_OUTPUT>
```

To open a sharded batch script for a named complex:

```bash
pymol "$(find <PYMOL_OUTPUT> -name '<complex>.pml' -print -quit)"
```

## Environment variables

| Variable | Read by | Purpose |
|----------|---------|---------|
| `PROTEIN_TOOLKIT_PROJECT_ROOT` | `data_registry.py`, toolkit | Project root when data and code live in different directories. The HPC wrappers set this themselves. |
| `PROTEIN_COMPLEXES_ROOT` | `complex_resolver.py`, HPC wrappers | Default models directory when `--root` is omitted. The HPC wrappers set this themselves. |
| `PYMOL_EXECUTABLE` | `render_complex_summary.py` | Headless PyMOL executable when `--pymol-executable` is not given. |
| `VISUALISE_ARGS` | `hpc_dataset_run.sh` | Flags forwarded to the figure step (default `--full-figure-pack --human-supplement`). |
| `HISTORICAL_RESULTS_CSV`, `OUTPUT_CSV`, `INTERFACES_JSONL`, `RESUME`, `LIMIT`, `RUN_STAMP`, `JOB_TAG` | `hpc_incremental_run.sh` | Incremental-run controls (see above). |

## Output locations

| Output | Location |
|--------|----------|
| Results CSV | `--output` path (default `batch_results.csv`; `results.csv` in the wrappers) |
| Interface JSONL | `--export-interfaces` path |
| Checkpoint | `<output>.checkpoint.jsonl` |
| PyMOL scripts | `pymol_scripts/shard_NNNN/*.pml` (batch/toolkit); flat under `--output` (single `generate`) |
| Batch figures | `Output/` (or `--output-dir`) |
| PAE heatmaps | inside the `--pae-heatmaps` directory |
| Single-complex summary | `--output` path |
| Audit manifests | `data/complex_manifest_audit/runs/<run_id>/` and `latest/` |
| STRING API cache | `data/string_api_cache/` (auto-created) |

## Common command combinations

```bash
# Structural core only, parallel, with interface export — no external data required
python toolkit.py --dir <MODELS_DIR> --output results.csv --interface --pae --export-interfaces interfaces.jsonl -w 8

# Structural core plus enrichment and variants (human annotation)
python toolkit.py --dir <MODELS_DIR> --output results.csv --interface --pae --enrich <ALIASES_FILE> --variants -w 8

# Everything, offline (no STRING API calls)
python toolkit.py --full-pipeline --dir <MODELS_DIR> --no-api -w 8 --output results.csv
```

## Troubleshooting

- **A data file is reported missing.** Run `python data_registry.py`; the `[MISSING]` message names the file and the source-file line whose filename constant to update if you have a newer release. `--full-pipeline` runs this check automatically and refuses to start if anything required is absent.
- **PDB and PKL are not paired.** Run `python complex_resolver.py --root <MODELS_DIR>` and read `incomplete_inputs.tsv`; the `reason` column distinguishes a missing file from an unreadable or ambiguous one.
- **The job runs out of memory at scale.** The current design accumulates results in memory. Use `--skip-existing` with `--limit N` to process the dataset in memory-bounded batches, consolidating between batches (see [Incremental HPC wrapper](#incremental-hpc-wrapper)).
- **A batch was interrupted.** Resubmit against the same partial output files with `RESUME=1`, keeping `HISTORICAL_RESULTS_CSV` unchanged.
- **PyMOL is not found.** `render_complex_summary.py` needs a headless PyMOL executable; install `pymol-open-source` via conda and set `PYMOL_EXECUTABLE`, or pass `--pymol-executable`.
- **A figure is skipped.** `visualise_results.py` skips any figure whose required columns are absent and prints the reason; generate the CSV with the stages that figure needs (for example `--variants` for the variant-density figure).
- **Population figures look wrong.** Generate figures from the consolidated CSV, not from an incremental-only file; an incremental fragment contains only the newly processed complexes.

## Functionality outside the submitted dissertation

The toolkit implements several capabilities that are usable but were not part of the submitted analyses. They are documented above under their command groups and summarised here so the boundary is explicit:

- **Standalone module tools** — the per-module command-line interfaces (`database_loaders.py`, `id_mapper.py`, `overlap_analysis.py`, and the annotation modules' `summary`/`lookup` subcommands) expose individual stages for inspection.
- **Database overlap analysis** (`overlap_analysis.py`) — cross-database Venn/UpSet comparison, used during development to characterise the interaction databases rather than to produce a dissertation result.
- **Deferred clustering modes** — `--clustering foldseek` and `--clustering hybrid` are placeholders; only `string` is implemented.
- **PyMOL script generation** (`--pymol`) — produces `.pml` scripts for structural inspection; the dissertation's structural figure used `render_complex_summary.py` instead.
- **py3Dmol viewer** — `pymol_scripts.py` provides an in-notebook py3Dmol fallback when the `py3Dmol` package is installed; it is not wired into the command-line interface.
- **Supplementary and diagnostic figures** — the `*_supp_*` figures, the multimer stoichiometry panel (`--multimer-supplement`), the non-human supplement (`--nonhuman-supplement`), the density overlays (`--density`) and the per-complex PAE heatmaps (`--pae-heatmaps`) are exploratory or diagnostic outputs kept outside the main results.

### Legacy compatibility

`visualise_results.py --legacy-mode` restores the older data-loading behaviour that dropped rows with missing or zero ipTM and filled missing pDockQ with zero. The current default keeps all rows and lets each figure's population filter handle exclusion, which is the behaviour used for the submitted analyses; `--legacy-mode` is retained only for compatibility with older CSVs and does not restore any other historical behaviour.
