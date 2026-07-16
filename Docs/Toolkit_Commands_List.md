# Toolkit command reference

This document lists the public commands, flags, defaults and output behaviour of the toolkit. The principal workflow is described in [`../README.md`](../README.md); the output fields are defined in [`OUTPUT_SCHEMA.md`](OUTPUT_SCHEMA.md).

Commands use `python` (the Windows convention); on Linux or macOS substitute `python3` where needed and use forward slashes. Each command or module carries a status label:

- **Workflow** — used to run the principal pipeline.
- **Operational** — infrastructure for running, validating and auditing a run.
- **Additional** — implemented and usable, but outside the principal workflow (mostly per-module inspection tools).
- **Legacy** — retained for older inputs or behaviour.
- **Accepted but unimplemented** — accepted by a parser but with no effect.

## Placeholder reference

Replace the angle-bracket placeholders before running.

| Placeholder | Meaning | Default if omitted |
|-------------|---------|--------------------|
| `<PROJECT_ROOT>` | Toolkit project root (code + `data/`) | repository directory |
| `<MODELS_DIR>` | Directory of AlphaFold-Multimer PDB/PKL outputs | required |
| `<COMPLEX_DIR>` | One complex's directory (one PDB + one PKL) | required |
| `<COMPLEX_NAME>` | A complex identifier | — |
| `<RESULTS_CSV>` | Results CSV path | `batch_results.csv` |
| `<INTERFACES_JSONL>` | Interface JSONL export path | none (user-specified) |
| `<OUTPUT_DIR>` | Figure output directory | `Output/` |
| `<N_WORKERS>` | Parallel worker count | `1` |
| `<PDB_FILE>` / `<PKL_FILE>` | A single PDB / PKL file | required |
| `<ALIASES_FILE>` | STRING aliases file | `data/ppi/9606.protein.aliases.v12.0.txt` |
| `<PPI_DIR>` | PPI database directory | `data/ppi/` |
| `<CLUSTERS_FILE>` | STRING clusters file | `data/clusters/9606.clusters.proteins.v12.0.txt` |
| `<VARIANTS_DIR>` | Variant database directory | `data/variants/` |
| `<STABILITY_DIR>` | EVE data directory | `data/stability/` |
| `<PATHWAYS_DIR>` | Disease/Reactome directory | `data/pathways/` |
| `<PYMOL_OUTPUT>` | PyMOL script output directory | `pymol_scripts/` |

## Quick command index

| Task | Command |
|------|---------|
| Validate registered data | `python data_registry.py` |
| Audit input pairs | `python complex_resolver.py --root <MODELS_DIR>` |
| Run the full pipeline | `python toolkit.py --full-pipeline --dir <MODELS_DIR> -w <N_WORKERS> --output <RESULTS_CSV> --export-interfaces <INTERFACES_JSONL>` |
| Generate figures | `python visualise_results.py <RESULTS_CSV> --output-dir <OUTPUT_DIR> --full-figure-pack --human-supplement` |
| Render one structural summary | `python render_complex_summary.py --input-dir <COMPLEX_DIR> --output summary.png --pymol-executable <PYMOL>` |
| Submit an initial HPC run | `sbatch hpc_dataset_run.sh` |
| Submit an incremental HPC run | `sbatch --export=ALL,HISTORICAL_RESULTS_CSV=results.csv hpc_incremental_run.sh` |

## `toolkit.py`

**Workflow.** Processes a directory of predictions and writes one CSV row per complex. Each feature flag adds a block of columns (defined in [`OUTPUT_SCHEMA.md`](OUTPUT_SCHEMA.md)); flags compose subject to the dependencies below. It is a flat parser (no subcommands). The default output is `batch_results.csv` and the default worker count is 1.

### Dependencies and auto-enabled flags

Enforced at start-up: an auto-enable prints a `Note:` and continues; a requirement failure prints an `Error:` and exits with status 1.

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

`--protvar` additionally checks that the FoldX export and AlphaMissense file exist (exit 1 if not), and `--full-pipeline` validates every required data file before processing starts.

### Core options

| Option | Type / default | Effect |
|--------|----------------|--------|
| `--dir` | required | Directory of PDB/PKL inputs (loose, flat-dir or sharded layout). |
| `--output` | `batch_results.csv` | Results CSV path; also the base name for the checkpoint file. |
| `--workers` / `-w` | int, `1` | Parallel workers for the structural pass (`>1` uses `ProcessPoolExecutor`). Must be `≥ 1`. |
| `--verbose` / `-v` | off | Per-complex progress (suppressed, with a note, when workers `> 1`). |

### Interface options

| Option | Type / default | Effect |
|--------|----------------|--------|
| `--interface` | off | Compute interface geometry and interface pLDDT. |
| `--pae` | off | Compute PAE-based interface features; auto-enables `--interface`. |
| `--export-interfaces` | `PATH`, off | Write one JSON record per complex (confident interface residues, PAE, per-residue pLDDT) for the High/Medium v2-tier complexes; auto-enables `--interface --pae`. |

### Enrichment options

Data requirements are listed in [`../README.md`](../README.md#setting-up-external-data). Enrichment (`--enrich`) runs for all rows; the six heavier stages skip non-human rows.

| Option | Type / default | Depends on | Data | Effect |
|--------|----------------|------------|------|--------|
| `--enrich` | `ALIASES_PATH`, off | — | STRING aliases | Gene symbols, protein names, sequences, cross-references. |
| `--databases` | `DATA_DIR`, off | `--enrich` | STRING/BioGRID/HuRI/HuMAP | Tag each pair with its interaction-database sources. |
| `--string-min-score` | int, `700` | `--databases` | — | Minimum STRING confidence for database matching. |
| `--clustering` | `{string}`, off; bare → `string` | `--enrich` | STRING clusters | Sequence-cluster / homologous-pair detection. Only `string` is accepted (see [legacy and unimplemented](#legacy-and-unimplemented-options)). |
| `--clusters-file` | `PATH`, default clusters file | `--clustering` | — | Override the STRING clusters file. |
| `--variants` | `VARIANTS_DIR`, off; bare → `data/variants/` | `--interface --pae --enrich` | UniProt/ClinVar/ExAC | Map variants and classify structural context. |
| `--no-clinvar` | off | `--variants` | — | Skip ClinVar (UniProt + ExAC only). Ignored if `--variants` is absent. |
| `--stability` | `STABILITY_DIR`, off; bare → `data/stability/` | `--variants` | EVE | EVE variant-effect scoring. |
| `--protvar` | `FOLDX_EXPORT`, off; bare → FoldX export | `--variants` | AlphaMissense + AFDB FoldX | Offline AlphaMissense + monomeric FoldX scoring. |
| `--am-file` | `AM_PATH`, default AlphaMissense TSV | `--protvar` | — | Override the AlphaMissense file. Ignored without `--protvar`. |
| `--disease` | `DIR`, off; bare → `data/pathways/` | `--enrich` | UniProt Swiss-Prot XML | Disease, PTM, GO and drug-target annotation. |
| `--pathways` | off | `--enrich` | Reactome | Reactome mapping and per-pathway PPI enrichment. |
| `--no-api` | off | — | — | Disable the STRING API fallback (offline only); also skips STRING PPI enrichment under `--pathways` and homology scores under `--clustering`. |

### Output options

| Option | Type / default | Effect |
|--------|----------------|--------|
| `--output` | `batch_results.csv` | Results CSV (also under Core options). |
| `--export-interfaces` | `PATH`, off | Interface JSONL (also under Interface options). |

### Checkpoint and incremental options

| Option | Type / default | Effect |
|--------|----------------|--------|
| `--checkpoint` | off | Save progress every 50 complexes to `<output>.checkpoint.jsonl`; removed on successful completion. |
| `--resume` | off | Resume from that checkpoint (skips completed complexes); auto-enables `--checkpoint`. |
| `--skip-existing` | `RESULTS_CSV`, off | Append-only mode: read `complex_name` values from a historical results CSV and process only those absent. Reads that CSV only; writes a fingerprinted audit snapshot under `data/complex_manifest_audit/runs/<run_id>/`. |
| `--limit` | int, off | Process at most N complexes from the alphabetically-sorted post-`--skip-existing` delta. Positive integer; chunk membership is fixed before `--resume` filtering. |

`--skip-existing` must point at the full historical `results.csv`, not a filtered subset. Chunk membership survives a crash-and-resume provided the historical CSV is unchanged between attempts.

### PyMOL options

| Option | Type / default | Effect |
|--------|----------------|--------|
| `--pymol` | off | Generate layered `.pml` scripts for qualifying complexes (requires `--interface --pae`). |
| `--pymol-output` | `DIR`, `pymol_scripts/` | Output directory (sharded into `shard_NNNN/`, ≤ 1000 scripts each). |
| `--pymol-render` | off | Include ray-tracing/PNG commands for `pymol -c` batch rendering. |
| `--pymol-min-tier` | `{High,Medium,Low}`, `High` | Minimum v2 tier for script generation. |

### Full-pipeline shortcut

`--full-pipeline` activates `--interface --pae --enrich --databases --clustering --variants --stability --protvar --disease --pathways --pymol --checkpoint` with the registered default data paths, and validates that every required data file exists before starting.

```bash
python toolkit.py --full-pipeline --dir <MODELS_DIR> -w <N_WORKERS> \
    --output <RESULTS_CSV> --export-interfaces <INTERFACES_JSONL>
```

### Legacy and unimplemented options

- `--clustering` accepts **only** `string` in the current code. Earlier `foldseek` and `hybrid` values are **no longer accepted** — argparse rejects them (`protein_clustering.py::VALID_CLUSTERING_MODES = ('string',)`).
- There are no accepted-but-unimplemented `toolkit.py` flags. (The one inert flag in the toolkit is in `visualise_results.py` — see [legacy mode](#legacy-mode).)

## `visualise_results.py`

**Workflow.** Reads a results CSV and writes PNG figures. No subcommands; a single positional argument (the CSV) plus flags. It degrades gracefully: any figure whose required columns are absent is skipped with a printed note.

### Main figure command

```bash
python visualise_results.py <RESULTS_CSV> --output-dir <OUTPUT_DIR> --full-figure-pack --human-supplement
```

`--output-dir` defaults to `./Output/`. Run figures from the consolidated CSV, not an incremental fragment.

### Output mapping

Filenames use the Results numbering. **The internal `plot_figN` function names use an older, non-matching numbering and must not be used to identify a figure**; the on-disk filename is authoritative. `{sfx}` is the species suffix (empty, `_human` or `_nonhuman`; structural figures only).

| Output filename | Content | Emitted |
|-----------------|---------|---------|
| `Fig_1_Dataset_and_Analysis_Population_Funnel.png` | Nested population funnel with screening side-callouts | default |
| `1_Quality_Scatter{sfx}.png` | ipTM vs pDockQ coloured by v2 tier | default |
| `Fig_2A_Interface_PAE_by_Quality_Tier{sfx}.png` | Interface PAE distribution by v2 tier | default |
| `Fig_2B_Interface_pLDDT_vs_Bulk_pLDDT{sfx}.png` | Interface vs bulk pLDDT, paradox subset marked | default |
| `Fig_3_Composite_Score_Behaviour{sfx}.png` | Composite by tier; composite vs strict confident-contact fraction | default |
| `Fig_4_Classification_Versus_Screening.png` | Composite screening bands; tier × screen-status crosstab | default (`--screening-figures`) |
| `Fig_5_ipTM_pDockQ_Metric_Disagreement{sfx}.png` | Categorical ipTM-only vs pDockQ-only agreement matrix | default |
| `7_Homo_vs_Hetero{sfx}.png` | Tier proportions and symmetry for homodimers vs heterodimers | default |
| `Fig_7_Variant_Density_Versus_Composite_Confidence.png` | Interface variant density vs composite (broad-human) | default (needs variant columns) |
| `Fig_8_Biological_Corroboration_and_Prediction_Bias.png` | Pathogenic-variant / PPI-enrichment / pLI / disorder by tier (reviewed-human) | default (needs variant + pathway columns) |
| `Fig_6_Prediction_Quality_Paradox{sfx}.png` | Paradox vs non-paradox: interface-minus-bulk pLDDT, PAE-only fraction, symmetry | `--full-figure-pack` |
| `1b_supp_Disorder_Scatter{sfx}.png` | ipTM vs pDockQ coloured by disorder fraction | `--disorder-scatter` / `--full-figure-pack` |
| `2_supp_PAE_Health_Check{sfx}.png` | Whole-complex PAE distribution | `--full-figure-pack` |
| `4_supp_Strict_vs_PAE_Only_Fraction{sfx}.png` | Strict vs PAE-only confident-contact fraction | `--full-figure-pack` |
| `8_supp_iptm_pdockq_scatter{sfx}.png` | ipTM vs pDockQ scatter by tier | `--full-figure-pack` |
| `9_supp_Chain_Count_Profile{sfx}.png` | Best-pair vs whole-complex pDockQ by chain count (skipped on the `_nonhuman` pass) | `--full-figure-pack` |
| `10_supp_Clustering_Validation.png` | Shared-cluster ratio by tier | `--full-figure-pack` |
| `11_supp_Variant_Consequence_Flow.png` | Clinical significance → structural context flow | `--full-figure-pack` |
| `13_supp_Stability_CrossValidation.png` | EVE / AlphaMissense / FoldX cross-comparison | `--full-figure-pack` |
| `14A_supp_Disease_Prevalence_by_Tier.png`, `14B_supp_Top_Disease_Categories_by_Tier.png` | Disease prevalence and top categories by tier | `--full-figure-pack` |
| `15_supp_Pathway_Bar_Chart.png`, `15_supp_Pathway_Network.png` | Top Reactome pathways; pathway network | `--full-figure-pack` |
| `18_supp_Partial_Reason_Dashboard.png` | Distribution of `partial_reason` values | `--include-partial-diagnostics` / `--full-figure-pack` |

The `_supp_` vs `Fig_` prefix is not a reliable main-vs-supplement indicator (e.g. `Fig_6_…` is supplement-gated); the deciding factor is `--full-figure-pack`.

### Figure flags

| Flag | Default | Effect |
|------|---------|--------|
| `--output-dir` | `./Output/` | Figure output directory. |
| `--full-figure-pack` | off | Emit every `*_supp_*` figure; implies `--disorder-scatter` and `--include-partial-diagnostics`. |
| `--dataset-funnel` / `--no-dataset-funnel` | on | Emit the population funnel (`Fig_1`). |
| `--screening-figures` / `--no-screening-figures` | on | Emit the classification-vs-screening figure (`Fig_4`). |
| `--disorder-scatter` | off | Emit the disorder-coloured scatter (`1b_supp`) on its own. |
| `--include-partial-diagnostics` | off | Emit the partial-reason dashboard (`18_supp`) on its own. |
| `--density` | off | Add KDE density contours to scatter figures. |
| `--multimer-supplement` | off | Emit the multimer-stoichiometry panel (`7_supp`); no rows dropped. |

### Population options

| Flag | Default | Effect |
|------|---------|--------|
| `--human-supplement` | off | Re-render the structural figures on the reviewed + TrEMBL-human subset (`_human` suffix). Needs a `species_status` column. |
| `--nonhuman-supplement` | off | Re-render on the non-human subset (`_nonhuman` suffix); the chain-count profile is skipped on this pass. |
| `--species-supplements` | off | **Legacy** deprecated alias that enables both of the above. |

`--include-multimers` has been removed; passing it exits with an error pointing to `--multimer-supplement`.

### PAE heatmaps

```bash
python visualise_results.py <RESULTS_CSV> --pae-heatmaps <MODELS_DIR>
python visualise_results.py <RESULTS_CSV> --pae-heatmaps <MODELS_DIR> --limit 10
```

`--pae-heatmaps <DIR>` reads each `*.pkl` in that directory and writes a `<pkl_stem>_PAE.png` heatmap **into that directory** (not `--output-dir`), with chain boundaries and the best-pair block highlighted. `--limit N` caps the count and **affects only this mode** — it does not touch the CSV-driven figures.

### Legacy mode

- `--legacy-mode` (**Legacy**) restores the old data-loading behaviour that dropped rows with missing or zero ipTM and filled missing pDockQ with zero. The default keeps all rows and lets each figure's population filter handle exclusion. It does not restore any other historical behaviour.
- `--skip-diagnostics` (**Accepted but unimplemented**) is a no-op: the flag is never read and the summary function it names is not called. It changes nothing whether set or not.

## `render_complex_summary.py`

**Operational.** Renders one dimer into a single PNG, reusing the production metric path so the metrics match the batch pipeline.

```bash
python render_complex_summary.py --input-dir <COMPLEX_DIR> --output summary.png \
    --pymol-executable "$HOME/envs/pymol/bin/pymol"
```

| Option | Type / default | Effect |
|--------|----------------|--------|
| `--input-dir` | required, `Path` | Directory with exactly one PDB and one PKL (`.pdb`/`.pdb.gz`/`.pdb.bz2`, `.pkl`/`.pkl.gz`/`.pkl.bz2`, and the `*_relaxed_model_*` / `*_result_model_*` / `.results.pkl` forms). |
| `--output` | required, `Path` | Output PNG path (must end in `.png`). |
| `--overwrite` | off | Replace an existing output; the default is to fail. |
| `--width` | int, `2000` | Output width in pixels (positive integer). |
| `--height` | int, `1600` | Output height in pixels (positive integer). |
| `--pymol-executable` | none | Headless PyMOL executable. |

- **Dimer-only:** a non-dimer or non-calibrated input raises a clear error and exits 1. It writes exactly one PNG (atomically, via a temp file), removing all intermediates; the two structural examples in a two-panel figure are composed separately, by hand.
- **PyMOL resolution order:** `--pymol-executable`, then `$PYMOL_EXECUTABLE`, then `pymol` on `PATH`. Install a headless build once with `conda create -y -p "$HOME/envs/pymol" -c conda-forge pymol-open-source`.
- No network access is used.

## Data and input utilities

### `data_registry.py`

**Operational.** Validates the registered external data before a run.

```bash
python data_registry.py                                   # all groups
python data_registry.py --groups ppi-databases variant-mapping
python data_registry.py --root <PROJECT_ROOT>
```

| Option | Type / default | Effect |
|--------|----------------|--------|
| `--groups` | choices, all | One or more of `ppi-databases`, `clustering`, `variant-mapping`, `eve-stability`, `offline-scoring`, `disease-pathways`, `pymol`. |
| `--root` | none | Override the project root. |

It checks the **16 required data files** (15 files + the `EVE_all_data` directory); two registry entries are auto-created output directories (`data/string_api_cache`, `pymol_scripts`) and are skipped. The report is printed to **stderr** as a per-group `[ OK ] / [MISSING]` table; a `[MISSING]` line names the source file and line whose filename constant to update for a newer release. Exit status is 1 if any required file is missing, else 0. The project root resolves as `--root` → `PROTEIN_TOOLKIT_PROJECT_ROOT` → repository directory. (`--groups pymol` selects a group whose only entry is an output directory, so it validates nothing.)

### `complex_resolver.py`

**Operational / Audit.** Discovers PDB/PKL pairs and writes a forensic manifest.

```bash
python complex_resolver.py --root <MODELS_DIR>
python complex_resolver.py --root <MODELS_DIR> --purpose baseline
```

| Option | Type / default | Effect |
|--------|----------------|--------|
| `--root` | none (`PROTEIN_COMPLEXES_ROOT`) | Models directory to scan. |
| `--audit-dir` | none | Override the audit output directory. |
| `--purpose` | `{baseline,incremental}`, `baseline` | Suffix embedded in the auto-generated run id. |
| `--run-id` | auto | Override the run id (intended for tests/fixtures; used verbatim). |

It recognises two directory layouts (sharded when a child directory matches `^[A-Z0-9]{2}$`, otherwise flat-dir); the loose layout is handled by `toolkit.py`. Per run it writes, under `data/complex_manifest_audit/runs/<run_id>/`, `complex_manifest.tsv` (complete pairs), `incomplete_inputs.tsv` (with a `reason` column) and `manifest_snapshot_summary.txt`, then mirrors them to `latest/` and updates `latest_run_id.txt`. Reason codes: `missing_pdb`, `missing_pkl`, `missing_both`, `empty_pdb`, `empty_pkl`, `empty_both`, `ambiguous_pdb`, `ambiguous_pkl`, `duplicate_complex_name`. Accepted suffixes are `.pdb`/`.pdb.bz2` and `.pkl`/`.pkl.bz2`/`.results.pkl`/`.results.pkl.bz2` (plus the `*_relaxed_model_*` / `*_result_model_*` globs); `.gz` is **not** accepted here. Exit status is 0 when at least one complete pair is found, else 1.

## HPC wrappers

Both SLURM wrappers set the runtime environment (module load, venv, BLAS/NumExpr thread caps, headless Matplotlib) and **hard-code the two cluster paths** near the top of the file — edit those lines for your site; a caller-exported value is overwritten.

### Initial run

```bash
sbatch hpc_dataset_run.sh
```

**Workflow.** Runs five phases: `pip check` → `data_registry.py` → `complex_resolver.py` → `toolkit.py --full-pipeline` (writes `results.csv`, `interfaces.jsonl`, `pymol_scripts/`) → `visualise_results.py`. The figure step reads `VISUALISE_ARGS` (default `--full-figure-pack --human-supplement`). SBATCH requests 16 CPU, 80 GB, 48 h — adjust for your cluster.

### Incremental run

```bash
sbatch --export=ALL,HISTORICAL_RESULTS_CSV=results.csv hpc_incremental_run.sh
```

**Workflow.** Processes only complexes absent from the historical CSV via `toolkit.py --full-pipeline --skip-existing`, and **does not render figures**. Preflight: `pip check`, `data_registry.py`, a check that the baseline manifest exists (`data/complex_manifest_audit/latest/`), and a historical CSV/JSONL sanity check (logs their SHA-256). SBATCH requests 16 CPU, 80 GB, 12 h. Overridable env vars:

| Variable | Default | Effect |
|----------|---------|--------|
| `HISTORICAL_RESULTS_CSV` | `results.csv` | Baseline CSV for `--skip-existing`. |
| `HISTORICAL_INTERFACES_JSONL` | `interfaces.jsonl` | Baseline JSONL (sanity-checked). |
| `OUTPUT_CSV` | `results_incremental_<stamp>_<job>.csv` | Per-run incremental CSV. |
| `INTERFACES_JSONL` | `interfaces_incremental_<stamp>_<job>.jsonl` | Per-run incremental JSONL. |
| `RUN_STAMP`, `JOB_TAG` | date / job id | Override the output filename stamp and tag. |
| `RESUME` | `0` | Set `1` to add `--resume`. |
| `LIMIT` | unset | Positive integer; adds `--limit N`. |

### Limited batches

```bash
sbatch --export=ALL,HISTORICAL_RESULTS_CSV=results.csv,LIMIT=100000 hpc_incremental_run.sh
```

Process the delta in memory-bounded batches: submit one batch, confirm completion, consolidate it into the cumulative CSV, then submit the next batch against the updated file.

### Resume

```bash
sbatch --export=ALL,RESUME=1,\
HISTORICAL_RESULTS_CSV=results.csv,\
OUTPUT_CSV=results_incremental_<stamp>_<job>.csv,\
INTERFACES_JSONL=interfaces_incremental_<stamp>_<job>.jsonl \
  hpc_incremental_run.sh
```

Resume against the **same** partial output files, with `HISTORICAL_RESULTS_CSV` unchanged from the interrupted run so the batch boundary does not shift.

### Consolidation

**Operational.** Incremental runs write to separate files and do not modify the historical CSV or JSONL. **There is no merge helper in the repository** — consolidation is manual. Merge each incremental output into the cumulative CSV/JSONL, then promote the merged CSV as the next `HISTORICAL_RESULTS_CSV`; otherwise the wrapper reproduces the same delta. Verify with a CSV/JSON-aware tool (not line-based text tools): identical headers; no `complex_name` present in both files; merged row count equals the sum; every JSONL record's complex present in the CSV; a backup before the merge; and an atomic rename. Then generate figures from the merged CSV.

## Standalone module commands

**Additional.** Each downstream module exposes a CLI for inspecting one stage. Default paths reference module constants (shown as directory placeholders here).

### `database_loaders.py`

Parse the PPI databases.

```bash
python database_loaders.py --data-dir <PPI_DIR>
python database_loaders.py --data-dir <PPI_DIR> --database string --output all.csv
```

| Option | Default | Effect |
|--------|---------|--------|
| `--data-dir` | `data/ppi/` | PPI directory. |
| `--database` | `all` | One of `string`, `biogrid`, `huri`, `humap`, `all`. |
| `--output` | none (summary only) | Export interactions to CSV. |
| `--min-string-score` | `0` | STRING confidence filter. |
| `--min-humap-prob` | `0.0` | HuMAP probability filter. |
| `--biogrid-physical-only` | on | Physical BioGRID interactions only. *(Cannot be turned off from the CLI — `store_true` with a `True` default.)* |
| `--no-api`, `--verbose`/`-v` | off | Disable STRING API validation; verbose output. |

### `id_mapper.py`

Resolve identifiers via the STRING aliases.

```bash
python id_mapper.py --aliases <ALIASES_FILE> --stats
python id_mapper.py --aliases <ALIASES_FILE> --resolve P04637
python id_mapper.py --aliases <ALIASES_FILE> --export lookup.csv
```

| Option | Default | Effect |
|--------|---------|--------|
| `--aliases` | STRING aliases file | Alias table. |
| `--stats` | off | Print mapping statistics. |
| `--resolve` | none | Resolve one identifier (UniProt / ENSP / ENSG / gene symbol). |
| `--export` | none | Export the full lookup table to CSV. |
| `--no-api`, `--verbose`/`-v` | off | Offline only; verbose. |
| `--validate-ids-api`, `--cross-validate-api N` | off / `0` | Optional STRING-API validation (network); only active within `--resolve` / when `N > 0`. |

### `overlap_analysis.py`

Cross-database overlap and Venn/UpSet diagrams (`--aliases` required).

```bash
python overlap_analysis.py --data-dir <PPI_DIR> --aliases <ALIASES_FILE>
python overlap_analysis.py --data-dir <PPI_DIR> --aliases <ALIASES_FILE> --base-level --report overlap.txt
```

| Option | Default | Effect |
|--------|---------|--------|
| `--data-dir` | `data/ppi/` | PPI directory. |
| `--aliases` | required | Alias table. |
| `--output` | `Output/venn_overlap.png` | Venn/UpSet figure. |
| `--string-min-score` | `700` | STRING confidence filter. |
| `--base-level` | off | Base-accession overlap (in addition to isoform-level). |
| `--threshold-comparison PATH` | none | Threshold-comparison figure. |
| `--report PATH` | none | Text overlap report. |
| `--verbose`/`-v` | off | Verbose. |

### `protein_clustering.py`

STRING clusters and shared membership. Exactly one of `--summary` / `--protein` / `--pair` is required.

```bash
python protein_clustering.py --clusters-file <CLUSTERS_FILE> --aliases <ALIASES_FILE> --summary
python protein_clustering.py --clusters-file <CLUSTERS_FILE> --aliases <ALIASES_FILE> --pair P04637 Q00987
```

| Option | Default | Effect |
|--------|---------|--------|
| `--clusters-file`, `--aliases` | required | Clusters file and alias table. |
| `--summary` \| `--protein ID` \| `--pair A B` | required (one) | Cluster statistics \| clusters for one protein \| shared clusters for a pair. |
| `--verbose`/`-v` | off | Verbose. |

### `variant_mapper.py`

Subcommands `summary`, `lookup`, `map`.

```bash
python variant_mapper.py summary --variants-dir <VARIANTS_DIR>
python variant_mapper.py lookup --variants-dir <VARIANTS_DIR> --protein P04637
python variant_mapper.py map --interfaces <INTERFACES_JSONL> --pdb-dir <MODELS_DIR> --variants-dir <VARIANTS_DIR> --output variants.csv
```

| Subcommand | Key options |
|-----------|-------------|
| `summary` | `--variants-dir` (`data/variants/`) |
| `lookup` | `--variants-dir`; `--protein` (required) |
| `map` | `--interfaces` (required), `--pdb-dir` (required), `--variants-dir`, `--output` (`variant_analysis.csv`), `--no-clinvar` |

### `stability_scorer.py`

Subcommands `summary`, `lookup` (global `--stability-dir`, default `data/stability/`).

```bash
python stability_scorer.py --stability-dir <STABILITY_DIR> lookup --protein P61981 --position 45
```

| Subcommand | Key options |
|-----------|-------------|
| `summary` | — |
| `lookup` | `--protein` (required); `--position` (int, optional) |

### `protvar_client.py`

Subcommands `summary`, `lookup` (globals `--foldx-export`, `--am-file`, defaulting to the AFDB FoldX and AlphaMissense files).

```bash
python protvar_client.py lookup --protein P61981 --position 4
```

| Subcommand | Key options |
|-----------|-------------|
| `summary` | — |
| `lookup` | `--protein` (required); `--position` (int, optional) |

### `disease_annotations.py`

Subcommands `summary`, `lookup` (global `--disease-dir`, default `data/pathways/`).

```bash
python disease_annotations.py lookup --disease-dir <PATHWAYS_DIR> --protein P04637
```

| Subcommand | Key options |
|-----------|-------------|
| `summary` | — |
| `lookup` | `--protein` (required) |

### `pathway_network.py`

Subcommands `summary`, `network`, `enrichment` (global `--pathways-dir`, default `data/pathways/`).

```bash
python pathway_network.py summary --csv <RESULTS_CSV>
python pathway_network.py network --csv <RESULTS_CSV> --output-dir <OUTPUT_DIR>
```

| Subcommand | Key options |
|-----------|-------------|
| `summary` | `--csv` (required) |
| `network` | `--csv` (required), `--output-dir` (`Output/networks/`), `--min-pdockq` (float) |
| `enrichment` | `--csv` (required) |

### `pymol_scripts.py`

Subcommands `generate` (single PDB, written flat) and `batch` (from a CSV, sharded into `shard_NNNN/`).

```bash
python pymol_scripts.py generate --pdb <PDB_FILE> --render
python pymol_scripts.py batch --csv <RESULTS_CSV> --pdb-dir <MODELS_DIR> --min-tier Medium --output <PYMOL_OUTPUT>
```

| Subcommand | Key options |
|-----------|-------------|
| `generate` | `--pdb` (required), `--output` (`.`), `--render`, `--name` |
| `batch` | `--csv` (required), `--pdb-dir` (required), `--output` (`pymol_scripts/`), `--min-tier` (`{High,Medium,Low}`, `High`), `--render`, `--no-variants` |

The `generate_py3dmol_view` in-notebook viewer is a library function only (guarded by an optional `py3Dmol` import); it is not wired to any CLI subcommand. To open a sharded batch script:

```bash
pymol "$(find <PYMOL_OUTPUT> -name '<COMPLEX_NAME>.pml' -print -quit)"
```

### Core inspection tools

Single-file inspection utilities used during development and debugging.

**`read_af2_nojax.py`** — read one AlphaFold2 PKL without JAX.

```bash
python read_af2_nojax.py --pkl <PKL_FILE> --keys
python read_af2_nojax.py --pkl <PKL_FILE> --json metrics.json --extract-pae pae.npy
```

| Option | Effect |
|--------|--------|
| `--pkl` (required) | PKL file to read. |
| `--keys` | List all keys in the PKL. |
| `--json PATH` | Save extracted metrics to JSON. |
| `--extract-pae PATH`, `--extract-plddt PATH` | Save the PAE matrix / pLDDT array to `.npy`. |
| `--quiet`/`-q` | Minimal output. |

**`pdockq.py`** — compute pDockQ for one structure.

```bash
python pdockq.py --pdbfile <PDB_FILE>
```

`--pdbfile` takes one PDB whose B-factor column holds pLDDT; it prints the pDockQ score and PPV, and exits if the structure has fewer than two chains.

**`interface_analysis.py`** — analyse one complex's interface.

```bash
python interface_analysis.py --pdb <PDB_FILE> --pkl <PKL_FILE> --json result.json
```

| Option | Effect |
|--------|--------|
| `--pdb` (required) | PDB file. |
| `--pkl` | PKL file (enables PAE analysis). |
| `--threshold` | Contact distance threshold in Å. |
| `--json PATH` | Save results to JSON. |
| `--quiet`/`-q` | Minimal output. |

**`string_api.py`** — query the STRING API directly (**Additional / Audit**; needs network access). Exactly one query mode is required.

```bash
python string_api.py --resolve P04637,Q9UKT4 --species 9606
python string_api.py --network TP53,MDM2,BRCA1 --network-type physical
python string_api.py --version
```

| Option | Effect |
|--------|--------|
| `--resolve` / `--interaction-partners` / `--homology` / `--enrichment` / `--ppi-enrichment` / `--network` (one required) | Comma-separated identifiers for the chosen query; or `--version` to print the STRING version. |
| `--species` | NCBI taxonomy ID (default `9606`). |
| `--network-type` | `--network` type (default `functional`). |
| `--required-score`, `--limit` | Minimum combined score (`0`); max partners per protein (`10`). |
| `--cache-dir`, `--output`/`-o` | Response cache directory; write output to CSV instead of stdout. |

## Environment variables

| Variable | Consumer | Default | Effect | Required |
|----------|----------|---------|--------|----------|
| `PROTEIN_TOOLKIT_PROJECT_ROOT` | `data_registry.py`, `complex_resolver.py`, `toolkit.py` | repository dir | Project root when code and `data/` differ; resolves data deps and the audit root. | No |
| `PROTEIN_COMPLEXES_ROOT` | `complex_resolver.py` | none | Models directory when `--root` is omitted (error if unset). | Only without `--root` |
| `PYMOL_EXECUTABLE` | `render_complex_summary.py` | `pymol` on `PATH` | Headless PyMOL binary when `--pymol-executable` is not passed. | No |
| `VISUALISE_ARGS` | `hpc_dataset_run.sh` | `--full-figure-pack --human-supplement` | Flags forwarded to the figure step. | No |
| `HISTORICAL_RESULTS_CSV`, `HISTORICAL_INTERFACES_JSONL`, `OUTPUT_CSV`, `INTERFACES_JSONL`, `RUN_STAMP`, `JOB_TAG`, `RESUME`, `LIMIT` | `hpc_incremental_run.sh` | see [Incremental run](#incremental-run) | Incremental-run controls. | No |
| `SLURM_JOB_ID` | `complex_resolver.py`, both wrappers | none | Embedded in the auto run id and job tag. | No (set by SLURM) |
| `SLURM_CPUS_PER_TASK` | both wrappers | `16` | Sets the worker count. | No (set by SLURM) |

The two cluster roots are **hard-coded** in the wrappers (edit them in-file). The wrappers also export runtime knobs consumed by third-party libraries, not by toolkit code: `PYTHONUNBUFFERED`, `PYTHONNOUSERSITE`, `MPLBACKEND=Agg`, `MPLCONFIGDIR`, and `OMP_/MKL_/OPENBLAS_/NUMEXPR_NUM_THREADS=1`.

## Output locations

| Output | Location |
|--------|----------|
| Results CSV | `--output` (default `batch_results.csv`; `results.csv` in the wrappers) |
| Interface JSONL | `--export-interfaces` path |
| Checkpoint | `<output>.checkpoint.jsonl` |
| Audit manifests | `data/complex_manifest_audit/runs/<run_id>/` (mirrored to `latest/`; pointer `latest_run_id.txt`) |
| Batch figures | `--output-dir` (default `Output/`) |
| PAE heatmaps | inside the `--pae-heatmaps` directory |
| PyMOL scripts | `pymol_scripts/shard_NNNN/*.pml` (batch/toolkit); flat under `--output` (single `generate`) |
| Single-complex summary | `render_complex_summary.py --output` path |
| STRING API cache | `data/string_api_cache/` (auto-created) |
| Wrapper logs | `hpc_dataset_run_%j.out/.err`, `hpc_incremental_%j.out/.err` |

## Common command patterns

```bash
# Minimal local run (base metrics only)
python toolkit.py --dir <MODELS_DIR> --output <RESULTS_CSV>

# Structural core, parallel, with interface export (no external data required)
python toolkit.py --dir <MODELS_DIR> --output <RESULTS_CSV> --interface --pae \
    --export-interfaces <INTERFACES_JSONL> -w <N_WORKERS>

# Structural core plus enrichment and variants (human annotation)
python toolkit.py --dir <MODELS_DIR> --output <RESULTS_CSV> --interface --pae \
    --enrich <ALIASES_FILE> --variants -w <N_WORKERS>

# Full pipeline, offline (no STRING API calls)
python toolkit.py --full-pipeline --dir <MODELS_DIR> --no-api -w <N_WORKERS> --output <RESULTS_CSV>

# Incremental batch, then figures from the consolidated CSV
sbatch --export=ALL,HISTORICAL_RESULTS_CSV=results.csv,LIMIT=100000 hpc_incremental_run.sh
python visualise_results.py <RESULTS_CSV> --output-dir <OUTPUT_DIR> --full-figure-pack --human-supplement

# One structural summary
python render_complex_summary.py --input-dir <COMPLEX_DIR> --output summary.png --pymol-executable "$HOME/envs/pymol/bin/pymol"
```

## Troubleshooting

| Problem | Action |
|---------|--------|
| A registered data file is missing | Run `python data_registry.py`; the `[MISSING]` line names the file and the constant to update for a newer release. |
| PDB/PKL pair incomplete | Run `python complex_resolver.py --root <MODELS_DIR>` and read `incomplete_inputs.tsv`; the `reason` column distinguishes missing, empty and ambiguous inputs. |
| Job runs out of memory at scale | Process in memory-bounded batches with `--skip-existing` + `LIMIT`, consolidating between batches. |
| Batch interrupted | Resubmit against the same output/checkpoint files with `RESUME=1` and an unchanged `HISTORICAL_RESULTS_CSV`. |
| A figure is skipped | The CSV lacks that figure's columns; re-run the toolkit with the stages it needs (the run prints the missing-column reason). |
| PyMOL not found | Install a headless `pymol-open-source` and set `$PYMOL_EXECUTABLE`, or pass `--pymol-executable`. |
| Population figures look wrong | Generate figures from the consolidated CSV, not an incremental-only fragment. |
| `--clustering foldseek`/`hybrid` rejected | Only `--clustering string` is supported. |
