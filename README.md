# Protein Complexes Toolkit

A Python toolkit to facilitate the analysis of protein complexes and target drug discovery

MSc Applied Bioinformatics Research Project - King's College London

**Student:** Talhah Zubayer | **Supervisor:** David Burke

## Usage

The simplest way to run the full analysis is with `--full-pipeline`, which activates every module using default data paths. It validates that all required data files exist before processing starts, so you get a clear report of anything missing up front rather than a crash mid-run.

```bash
# Full pipeline - only --dir and -w are needed
python toolkit.py --full-pipeline --dir <MODELS_DIR> -w 8 --output results.csv
```

This is equivalent to manually specifying `--interface --pae --enrich --databases --clustering --variants --stability --protvar --disease --pathways --pymol --checkpoint` with all their default file paths.

You can also check data dependencies independently before starting a run:

```bash
python data_registry.py
```

For individual flag control, progressive flag-stacking examples, and standalone module CLIs, see **[Toolkit_Commands_List.md](Toolkit_Commands_List.md)**.

## Repository Structure

```
protein-complexes-toolkit/
├── read_af2_nojax.py         # JAX-free AlphaFold2 PKL reader
├── pdockq.py                 # pDockQ score calculator
├── interface_analysis.py     # Interface analysis module
├── toolkit.py                # Batch processing orchestrator
├── visualise_results.py      # Visualisation engine
├── visualise_filters.py      # Audit-aligned 14-filter population registry consumed by visualise_results.py (single source of truth for `calibrated_dimer`, screening-status, species subsets)
├── database_loaders.py       # PPI database parsers (STRING, BioGRID, HuRI, HuMAP)
├── id_mapper.py              # Protein ID cross-referencing (ENSP/ENSG/UniProt/gene symbol)
├── overlap_analysis.py       # Database overlap computation and UpSet diagrams
├── string_api.py             # Centralised STRING API client (rate limiting, caching, retry)
├── protein_clustering.py     # Protein sequence clustering and homology detection
├── variant_mapper.py         # Genetic variant mapping and structural context classification
├── stability_scorer.py       # EVE stability scoring and variant effect predictions
├── protvar_client.py         # Offline AlphaMissense + monomeric FoldX scorer (local data files)
├── disease_annotations.py    # UniProt disease/PTM/GO/drug-target annotation
├── pathway_network.py        # Reactome pathway mapping, PPI enrichment, NetworkX networks
├── pymol_scripts.py          # PyMOL .pml script generation and py3Dmol fallback
├── data_registry.py          # Data dependency registry and pre-run validation
├── complex_resolver.py       # PDB/PKL pair discovery (flat + sharded, .bz2-aware) + fingerprinted manifest with runs/<id>/ + latest/ + latest_run_id.txt audit layout
├── file_io.py                # Transparent open() for plain / .gz / .bz2 inputs
├── hpc_dataset_run.sh        # SLURM wrapper for full-pipeline production HPC submission (see HPC Submission)
├── hpc_incremental_run.sh    # SLURM wrapper for append-only incremental runs (see HPC Submission > Append-only incremental mode)
├── Toolkit_Commands_List.md  # Full CLI command reference (all flags, defaults, examples)
├── requirements.txt          # Python dependencies
├── .gitignore
└── data/                            # External databases (not included in repo)
    ├── complex_manifest_audit/      # Fingerprinted manifest from complex_resolver (auto-generated)
    ├── ppi/                         # PPI databases (see "Setting Up Data")
    ├── clusters/                    # STRING sequence clusters (see "Setting Up Data")
    ├── variants/                    # Variant databases (see "Setting Up Data")
    ├── stability/                   # Stability prediction data (see "Setting Up Data")
    ├── pathways/                    # Disease & pathway databases (see "Setting Up Data")
    └── string_api_cache/            # STRING API response cache (auto-generated; sharded <key[:2]>/<key>.json)
```

## Installation

### Prerequisites

- Python 3.11+
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** JAX is **not** required. The toolkit uses module-level mocking to read AlphaFold2 PKL files without a JAX installation.


## Setting Up Data

The `data/` directory is **not included** in this repository due to the large size of external database files. These files are required for Database Ingestion & ID Mapping and later phases. To set up:

1. Create the directory structure:
```bash
mkdir -p data/ppi data/clusters data/variants data/stability data/pathways
```

2. Download the following files into `data/ppi/`:

| File | Source | Download |
|------|--------|----------|
| `9606.protein.links.v12.0.txt` | STRING | [string-db.org/cgi/download](https://string-db.org/cgi/download?sessionId=bqpmZGj7RlXV&species_text=Homo+sapiens) - select *Homo sapiens*, download `9606.protein.links.v12.0.txt.gz`, decompress |
| `9606.protein.aliases.v12.0.txt` | STRING | Same page - download `9606.protein.aliases.v12.0.txt.gz`, decompress |
| `BIOGRID-ALL-5.0.253.tab3.txt` | BioGRID | [downloads.thebiogrid.org](https://downloads.thebiogrid.org/File/BioGRID/Release-Archive/BIOGRID-5.0.253/BIOGRID-ALL-5.0.253.tab3.zip) - extract the `.tab3.txt` file from the zip |
| `HuRI.tsv` | HuRI | [interactome-atlas.org/download](https://interactome-atlas.org/download) - download `HuRI.tsv` |
| `humap2_ppis_ACC_20200821.pairsWprob` | HuMAP 2.0 | [humap2.proteincomplexes.org/download](https://humap2.proteincomplexes.org/download) - download "Protein Interaction Network with probability scores (Uniprot gzip)", decompress |

3. Download the STRING protein clusters file into `data/clusters/`:

| File | Source | Download |
|------|--------|----------|
| `9606.clusters.proteins.v12.0.txt` | STRING | [string-db.org/cgi/download](https://string-db.org/cgi/download?sessionId=bqpmZGj7RlXV&species_text=Homo+sapiens) - select *Homo sapiens*, download `9606.clusters.proteins.v12.0.txt.gz`, decompress |

4. Download variant database files into `data/variants/`:

| File | Source | Download |
|------|--------|----------|
| `homo_sapiens_variation.txt` | UniProt | [ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/variants/](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/variants/) - download `homo_sapiens_variation.txt.gz`, decompress |
| `variant_summary.txt` | ClinVar | [ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/) - download `variant_summary.txt.gz`, decompress |
| `forweb_cleaned_exac_r03_march16_z_data_pLI_CNV-final.txt` | ExAC/gnomAD | [gnomad.broadinstitute.org/downloads](https://gnomad.broadinstitute.org/downloads) - under "Gene constraint scores TSV", download and decompress |

5. Download EVE variant effect scores and UniProt ID mapping into `data/stability/`:

| File | Source | Download |
|------|--------|----------|
| `EVE_all_data/` (3,211 CSVs) | EVE | [evemodel.org/download/bulk](https://evemodel.org/download/bulk) - download "All variant files" CSV archive, extract into `data/stability/` |
| `HUMAN_9606_idmapping.dat` | UniProt | [ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/) - download `HUMAN_9606_idmapping.dat.gz`, decompress into `data/stability/` |

> **Note:** The EVE bulk download page offers several archives (MSAs, VCF files, PRC/ROC curves). Only the **variant files** archive is needed - the others are model training inputs or diagnostic plots not used by the pipeline. The `HUMAN_9606_idmapping.dat` file maps UniProt accessions to entry names (e.g. `P61981` -> `1433G_HUMAN`) which are used as EVE CSV filenames.

6. Download AlphaMissense and AFDB monomeric FoldX data into `data/stability/` (required for `--protvar`):

| File | Source | Download |
|------|--------|----------|
| `AlphaMissense_aa_substitutions.tsv` | Zenodo | [zenodo.org/records/10813168](https://zenodo.org/records/10813168) - download `AlphaMissense_aa_substitutions.tsv.gz`, decompress into `data/stability/` |
| `afdb_foldx_export_20250210.csv` | EBI | [ftp.ebi.ac.uk/pub/databases/ProtVar/predictions/stability/](https://ftp.ebi.ac.uk/pub/databases/ProtVar/predictions/stability/) - Pre-computed monomeric FoldX DDG + pLDDT for all human protein positions. Download `2025.02.10_foldx_energy.csv.gz`, decompress into `data/stability/` |

> **Note:** These 2 files (~14 GB total) provide offline AlphaMissense pathogenicity scores and monomeric FoldX stability predictions. They replace the previous ProtVar API dependency, eliminating the need for internet access during `--protvar` scoring.

7. Download disease and pathway annotation databases into `data/pathways/`:

| File | Source | Download |
|------|--------|----------|
| `uniprot_sprot_human.xml` | UniProt | [ftp.uniprot.org/pub/databases/uniprot/knowledgebase/taxonomic_divisions/](https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/taxonomic_divisions/) - download `uniprot_sprot_human.xml.gz`, decompress |
| `UniProt2Reactome_All_Levels.txt` | Reactome | [reactome.org/download/current/](https://reactome.org/download/current/) - download `UniProt2Reactome_All_Levels.txt` |
| `ReactomePathwaysRelation.txt` | Reactome | [reactome.org/download/current/](https://reactome.org/download/current/) - download `ReactomePathwaysRelation.txt` |

> **Note:** The UniProt XML file (~1.02 GB) contains all reviewed human protein entries with disease, PTM, GO, and drug target annotations. The Reactome files provide pathway-to-protein mappings (~110 MB) and pathway hierarchy (~611 KB) for network analysis.

8. Verify the directory contents:
```
data/
├── ppi/
│   ├── 9606.protein.links.v12.0.txt          (~616 MB)
│   ├── 9606.protein.aliases.v12.0.txt        (~195 MB)
│   ├── BIOGRID-ALL-5.0.253.tab3.txt          (~1.48 GB)
│   ├── HuRI.tsv                              (~1.6 MB)
│   └── humap2_ppis_ACC_20200821.pairsWprob   (~439 MB)
├── clusters/
│   └── 9606.clusters.proteins.v12.0.txt      (~40 MB)
├── variants/
│   ├── homo_sapiens_variation.txt             (~2.2 GB)
│   ├── variant_summary.txt                    (~1.1 GB)
│   └── forweb_cleaned_exac_r03_march16_z_data_pLI_CNV-final.txt  (~2 MB)
├── stability/
│   ├── HUMAN_9606_idmapping.dat               # UniProt ID mapping (~145 MB)
│   ├── EVE_all_data/                          # 3,211 per-protein EVE score CSVs (~10 GB)
│   │   ├── 1433G_HUMAN.csv
│   │   ├── 1433Z_HUMAN.csv
│   │   └── ...
│   ├── AlphaMissense_aa_substitutions.tsv     # AlphaMissense pathogenicity (~6.3 GB)
│   └── afdb_foldx_export_20250210.csv         # AFDB FoldX DDG + pLDDT (~7.7 GB)
└─── pathways/
    ├── uniprot_sprot_human.xml                 # UniProt reviewed human entries (~1.02 GB)
    ├── UniProt2Reactome_All_Levels.txt         # UniProt-Reactome mappings (~110 MB)
    └── ReactomePathwaysRelation.txt            # Reactome pathway hierarchy (~611 KB)
```


## HPC Submission

For cluster runs, `hpc_dataset_run.sh` is a hardened SLURM wrapper that orchestrates the full pipeline end-to-end. The minimum invocation:

```bash
export PROTEIN_TOOLKIT_PROJECT_ROOT=/scratch/<project>/protein-complexes-toolkit-hpc
export PROTEIN_COMPLEXES_ROOT=/scratch/<project>/Protein_Complexes
sbatch hpc_dataset_run.sh
```

The wrapper sets `module purge && module load python/3.11.6-gcc-13.2.0`, activates the project venv, applies the environment-hardening below, and runs 5 phases: `[0/4] pip check -> [1/4] data_registry.py -> [2/4] complex_resolver.py -> [3/4] toolkit.py --full-pipeline -> [4/4] visualise_results.py`.

### Resource allocation

| Resource | Allocation | Note |
|---|---|---|
| CPUs | 16 | Matches `ProcessPoolExecutor(max_workers=16)`. |
| Memory | 64 GB | Run 1 measured MaxRSS 67 GB on the 41,196-complex Run 1 corpus - bumped up to **80 GB** for subsequent runs. The corpus has since grown via incremental expansions to **516,744 rows** (`results_516744.csv`, 11 May 2026); at this scale the in-memory `results` list exceeds the 80 GB ceiling in a single shot (HPC job `33774663` OOM-killed at 71 % of 356,933 active complexes), so chunked runs via `--limit N` (see Append-only incremental mode below) are now the production path. |
| Walltime | 48 h | Run 1 finished in 5h 57m; 48 h gives ~8× safety. |

### Why the wrapper sets BLAS thread caps

NumPy / SciPy / BioPython transitively call BLAS, which by default tries to use all available cores per call. Combined with `ProcessPoolExecutor(max_workers=16)` on a 16-CPU allocation, this would oversubscribe to 16 × 16 = 256 threads competing for 16 cores - a 5-10× slowdown. The wrapper exports `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1` so `ProcessPoolExecutor`'s parallelism is the only level of concurrency.

### Why the wrapper sets matplotlib environment

Compute nodes have no DISPLAY, so the default Tk backend would crash when `visualise_results.py` runs. The wrapper exports `MPLBACKEND=Agg` and `MPLCONFIGDIR="$PROTEIN_TOOLKIT_PROJECT_ROOT/.matplotlib"` (the second redirects the font/style cache off the user's home directory, which is often a quota-restricted shared filesystem).

### Pre-flight gates

The wrapper's `[0/4]`–`[2/4]` steps fail in <30 s if anything is wrong, so a missing data file fails fast at minute 0 instead of at hour 12:

- `[0/4] pip check` - dependency consistency in the venv.
- `[1/4] data_registry.py` - all 18 registered data files exist and are non-empty.
- `[2/4] complex_resolver.py` - PDB/PKL pairs in the input tree, audit manifest written.

### Append-only incremental mode

When the input dataset is incrementally expanded (new complexes added to an existing tree), `toolkit.py --skip-existing results.csv` processes only the newly-added complexes against a previously-completed historical run, skipping the ~6 h full-pipeline cost. The companion SLURM wrapper `hpc_incremental_run.sh` mirrors `hpc_dataset_run.sh`'s environment hardening and adds a 4-step preflight (pip -> data registry -> baseline manifest existence -> historical CSV/JSONL sanity).

A fingerprinted manifest layout - `data/complex_manifest_audit/runs/<id>/` (immutable per-run snapshot) + `latest/` (convenience mirror) + `latest_run_id.txt` (atomic single-line pointer) - gives every resolver call a forensic trail and lets incremental runs detect file-level changes via `(size, mtime_ns)` fingerprints. Any complex whose fingerprint differs from the previous baseline is recorded in `changed_existing.tsv` with a stderr warning and is **not** silently reprocessed (mixed-vintage rows would invalidate downstream figures).

Incremental outputs are written separately (`results_incremental_<stamp>_<job>.csv`, `interfaces_incremental_<stamp>_<job>.jsonl`) and must be merged with the historical CSV/JSONL pair before figure generation. After the merge, promote the combined CSV as the next `--skip-existing` reference (or pass it via `HISTORICAL_RESULTS_CSV` to the wrapper) - otherwise the wrapper is intentionally idempotent and will reproduce the same delta.

```bash
# One-time, before the dataset is expanded - captures the baseline:
python complex_resolver.py --root "$PROTEIN_COMPLEXES_ROOT" --purpose baseline

# After dataset expansion - runs only the delta:
sbatch hpc_incremental_run.sh

# Crash recovery - same OUTPUT_CSV, RESUME=1:
sbatch --export=ALL,RESUME=1,OUTPUT_CSV=...,INTERFACES_JSONL=... hpc_incremental_run.sh
```

See [Toolkit_Commands_List.md](Toolkit_Commands_List.md) for the full env-override table and per-flag reference.

### Chunked-runs via `--limit N`

When the corpus grows past ~100K rows the toolkit's in-memory `results` list (plus per-row stash data `_sasa_*` / `_cb_coords_*` / `_chain_res_numbers_*` consumed by the post-pass variant annotation) exceeds the 80 GB HPC RAM ceiling in a single shot. HPC job `33774663` OOM-killed at 71 % of an active processing scope of 356,933 complexes; this is what motivated `--limit N`. The flag caps the number of complexes processed per invocation, letting operators submit the corpus in chunks that fit under the RAM ceiling.

```bash
# Submit a 30,000-complex chunk:
sbatch --export=ALL,LIMIT=30000 hpc_incremental_run.sh

# After the job completes, check MaxRSS to size the next chunk:
sacct -j <jobid> --format=JobID,State,ExitCode,MaxRSS,Elapsed

# Decide chunk size for the rest:
#   MaxRSS ≤ 50 GB  -> bump to LIMIT=60000
#   MaxRSS 50–65 GB -> stay at LIMIT=30000–40000
#   MaxRSS ≥ 65 GB  -> drop to LIMIT=20000 and reconsider the stash refactor
```

`--limit N` rejects `0` and negative values at the argparse layer (`--limit must be a positive integer (omit the flag for unlimited)`). The `LIMIT` env-override in `hpc_incremental_run.sh` is validated against `^[1-9][0-9]*$` and logs `sha256sum "$HISTORICAL_RESULTS_CSV"` at job start so chunk membership can be audited across a failed-attempt / retry pair.

**Correctness-load-bearing filter ordering.** The full filter pipeline is `scan -> skip-existing -> sort -> limit -> resume`. `--limit` is applied **before** `--resume` so chunk membership survives a mid-chunk crash - if the limit were applied after resume, the resume filter would remove completed rows first, the limit would slide forward to fill the gap, and the resumed run would silently pull rows from the next chunk. Two regression tests pin this invariant (`test_select_chunk_applies_limit_before_resume_filter` and `test_limit_chunk_membership_survives_resume`).

**Operator workflow between chunks.** After each chunk completes, merge `results_incremental_<stamp>_<job>.csv` into `results.csv` (and the JSONL pair likewise) using an 8-step atomic-merge protocol with Python stdlib `csv` / `json` (not `awk -F,` - fragile with quoted fields): header equality, dup check on `complex_name`, overlap check, JSONL ⊆ CSV invariant, row count = historical_rows + chunk_rows, backup-before-merge, `mv` atomic rename. Then submit the next chunk. For the ~425K-complex residual that triggered the OOM, this is ~14 submissions at LIMIT=30000 or ~7 at LIMIT=60000.

**Deferred follow-up.** The durable fix is a stash-flow refactor that would strip per-row stash from the in-memory `results` list after checkpoint append and read on-demand from disk during variant annotation. Designed but not implemented; ~1–2 h of work. The chunked-runs workflow above is the production path until the refactor lands.

---

## Pipeline Architecture

```
PDB + PKL files
       │
       ▼
read_af2_nojax.py ──▶ pdockq.py ──▶ interface_analysis.py ──▶ toolkit.py ──▶ visualise_results.py
  (PKL metrics)     (pDockQ/PPV)    (interface geometry       (batch CSV     (generates figures)
                                     + PAE features)           output)
                                                                  │
                                                    ┌─────────────┤ (optional --enrich)
                                                    ▼             ▼
                                              id_mapper.py   database_loaders.py
                                            (gene symbols,   (source tagging,
                                             protein names)   evidence types)
                                                    │             │
                                                    └──────┬──────┘
                                                           ▼
                                                    string_api.py
                                                  (automatic API fallback
                                                   for unresolved IDs;
                                                   disable with --no-api)
                                                           │
                                                           ▼ (optional --clustering)
                                               protein_clustering.py
                                             (STRING sequence clusters,
                                              homologous pair detection,
                                              optional API homology scores)
                                                           │
                                                           ▼ (optional --variants)
                                                 variant_mapper.py
                                             (UniProt/ClinVar/ExAC variants,
                                              biotite SASA structural context,
                                              cross-chain interface classification,
                                              variant enrichment analysis)
                                                           │
                                                           ▼ (optional --stability)
                                               stability_scorer.py
                                             (EVE evolutionary variant
                                              effect predictions,
                                              pathogenicity classification)
                                                           │
                                                           ▼ (optional --protvar)
                                               protvar_client.py
                                             (offline AlphaMissense +
                                              monomeric FoldX scoring
                                              from local data files)
                                                           │
                                                           ▼ (optional --disease)
                                             disease_annotations.py
                                             (UniProt disease/PTM/GO/
                                              drug-target annotation,
                                              offline XML + API fallback)
                                                           │
                                                           ▼ (optional --pathways)
                                               pathway_network.py
                                             (Reactome pathway mapping,
                                              per-pathway PPI enrichment,
                                              NetworkX network analysis)
                                                           │
                                                           ▼ (optional --pymol)
                                                pymol_scripts.py
                                             (PyMOL .pml script generation,
                                              chain/pLDDT/interface/variant
                                              colouring, py3Dmol fallback)
```

### Database Ingestion & ID Mapping Pipeline

```
PPI Database Files                    STRING Aliases File
  (STRING, BioGRID,                   (9606.protein.aliases)
   HuRI, HuMAP)                              │
       │                                      ▼
       ▼                               id_mapper.py
database_loaders.py ──────────▶    (ENSP/ENSG/UniProt
  (standardised DataFrames)         cross-referencing)
       │                                      │
       ▼                                      ▼
              overlap_analysis.py
        (pair normalisation, Venn/UpSet diagrams,
         --base-level dual analysis, --report)
```

### Script Descriptions

The pipeline produces a 41-column base CSV, progressively expandable to 155 columns by stacking optional flags (`--enrich`, `--interface --pae`, `--clustering`, `--variants`, `--stability`, `--protvar`, `--disease`, `--pathways`). JSONL interface export is also available. STRING API validation is on by default across all modules; disable with `--no-api`. Each downstream module also provides a standalone CLI. Compressed inputs (`.pdb.bz2`, `.pkl.bz2`) and the sharded HPC layout are supported transparently.

#### Core Analysis

**read_af2_nojax.py** - Loads AlphaFold2 result PKL files without requiring a JAX installation. Extracts ipTM, pTM, pLDDT arrays, and PAE matrices from `.pkl`, `.pkl.gz`, and `.pkl.bz2` formats.

**pdockq.py** - Calculates predicted DockQ scores using the FoldDock sigmoid model. Automatically selects the best interacting chain pair in multi-chain complexes and returns full contact geometry.

**interface_analysis.py** - 2-phase interface characterisation. Phase 1 derives structural geometry from PDB alone (contact count, interface fractions, symmetry, density, interface vs bulk pLDDT). Phase 2 adds PAE-aware confident contact identification, composite confidence scoring, and automated quality flags including paradox detection and metric disagreement.

**toolkit.py** - Batch orchestrator that processes directories of AlphaFold2 predictions with multiprocessing, periodic checkpointing, resume from interruption, and implements 2 quality classification schemes (v1 ipTM/pDockQ gating; v2 composite-informed reclassification) plus a separate `composite_screen_status` screening / prioritisation label (`strong_screen_candidate` / `moderate_screen_candidate` / `weak_screen_candidate` / `unavailable`) - the screen is a ranking heuristic, not a tier and not a calibrated probability. Reads paired PDB/PKL files via `complex_resolver.py` and decompresses `.bz2` inputs in-place via `file_io.py` (no staging mirror). Each optional flag activates a downstream module: `--enrich` (gene symbols, protein names, sequences, database source tagging, species classification), `--clustering` (sequence clusters, homologous pairs), `--variants` (variant mapping and structural context), `--stability` (EVE scores), `--protvar` (AlphaMissense + FoldX), `--disease` (UniProt annotations), `--pathways` (Reactome + network analysis), `--pymol` (PyMOL script generation). `--full-pipeline` activates all phases with default data paths and validates all data dependencies before processing starts.

**visualise_results.py** - Generates 24 distinct figure titles emitted as up to 36 PNG files at full settings - 12 dual-emit pairs (`*.png` + `*_human.png` covering calibrated dimer vs calibrated dimer × human) plus 12 single-emit figures, including 5 new figures (Fig 0 Corpus Funnel, Fig 4-supp Strict-vs-PAE-only Confident Contact Fraction, Fig 16 Prediction Quality Paradox, Fig 17 Screening Landscape, Fig 18-supp Partial Reason Dashboard). Pairs with `visualise_filters.py` for population filtering - a 14-filter registry whose row counts are audit-aligned against `results_516744.csv` so every figure's `[scope; N=…]` subtitle reproduces the dissertation citation numbers exactly. Adaptive scatter sizing, rasterisation for N ≥ 50k datasets (no hexbin/2-D-hist/density downsampling), optional KDE density contour overlays via `--density`. **Default invocation produces the main-text pack** on the calibrated-dimer scope (Figs 0, 1, 3, 4, 5, 7, 8 delta-histogram, 12, 16, 17); supplementary figures (`*_supp_*`) are opt-in via `--full-figure-pack` (implies `--disorder-scatter` + `--include-partial-diagnostics`), species dual-emit via `--human-supplement` / `--nonhuman-supplement` (Fig 9 force-skipped on non-human). Live composite thresholds `0.63 / 0.64 / 0.85` are drawn on tier-boundary annotations. The old `--include-multimers` flag has been removed - multimer-exploratory panels are opt-in via `--multimer-supplement` and never make dissertation claims.

**visualise_filters.py** - Audit-aligned 14-filter population registry consumed exclusively by `visualise_results.py`. Exposes `FILTER_REGISTRY` (14 named filters: `all_rows` / `recoverable` / `calibrated_dimer` / `composite_status_present` / `composite_screenable` / `strong_screen_candidate` / `moderate_screen_candidate` / `weak_screen_candidate` / `human_broad` / `human_strict` / `multimer_exploratory` / `partial_error` / `calibrated_human_broad` / `calibrated_human_strict`), `apply_filter(df, name, fig_label="")` which returns `(filtered_df, n_before, n_after)` and emits an audit-aligned stdout line `<fig_label> | filter=<name>: n_before -> n_after rows` for runtime traceability, `require_columns()` for graceful-degradation schema guards, and `parse_boolish` / `split_interface_flags` helpers. The registry's row counts are validated against `results_516744.csv` (`calibrated_dimer` 402,846; `calibrated_human_broad` 364,357; `strong/moderate/weak_screen_candidate` 8,635 / 21,998 / 372,213; `partial_error` 110,500) - these reproduce the dissertation citation numbers exactly, making the module the single source of truth backing the figure-subtitle `[scope; N=…]` annotations. `mask_calibrated_dimer` codifies the six-clause dissertation-safe definition: `tier_scope == 'dimer_validated'` AND `composite_is_calibrated == True` AND recoverable AND six required numeric metrics non-NaN AND `n_interface_contacts > 0`. **Dependency contract is one-way**: `visualise_filters.py` owns `parse_boolish` and the numeric-presence helpers; `visualise_results.py` imports from here, never the reverse. The diagnostic helper `mask_degenerate_empty_rows` is deliberately excluded from `FILTER_REGISTRY` because subtracting the 7 audited degenerate rows from `recoverable` would invalidate the documented 406,244 audit row count.

#### Database & Enrichment

**database_loaders.py** - Parsers for STRING, BioGRID, HuRI, and HuMAP protein interaction databases. All return standardised DataFrames (`protein_a`, `protein_b`, `source`, `confidence_score`, `evidence_type`) with optional API spot-check validation.

**id_mapper.py** - Isoform-aware protein identifier cross-referencing (ENSP, ENSG, UniProt, gene symbol) using STRING aliases as a single source of truth. Resolves any identifier type to a target namespace with automatic API fallback for local misses. Also provides `SpeciesClassifier`, which tags each accession as `reviewed_human`, `trembl_human`, or `non_human` using Swiss-Prot and `HUMAN_9606_idmapping.dat`; the toolkit uses this to skip human-only database lookups on non-human rows.

**overlap_analysis.py** - Computes pairwise interaction overlaps across databases with UpSet-style visualisation. Supports dual-level analysis (isoform-specific and base-accession) via `--base-level` and report generation via `--report`.

**string_api.py** - Centralised STRING API client through which all API interactions are routed. Offline-first architecture with rate limiting, automatic retry/backoff, and SHA256-keyed response caching.

**protein_clustering.py** - Parses STRING sequence clusters, maps them to UniProt accessions, and detects homologous protein pairs with optional API-based paralogy bitscores (`--clustering`). Caps pair enumeration for oversized clusters to avoid combinatorial explosion.

#### Variant & Stability

**variant_mapper.py** - Maps variants from UniProt, ClinVar, and ExAC onto complex interface residues (`--variants`). Computes SASA via biotite (with BioPython fallback) to classify each variant into 4 structural contexts: `interface_core`, `interface_rim`, `surface_non_interface`, or `buried_core`. Adds per-complex variant burden, enrichment fold-change, and ExAC constraint scores. Databases loaded via chunked streaming for memory efficiency.

**stability_scorer.py** - Integrates EVE evolutionary pathogenicity predictions with the variant pipeline (`--stability`). Lazy-loads only EVE score CSVs for proteins in the current run, mapping pipeline accessions to entry names via `HUMAN_9606_idmapping.dat`.

**protvar_client.py** - Offline pathogenicity and stability scoring from pre-computed AlphaMissense (216M variants) and AFDB monomeric FoldX DDG (209M substitutions) data files (`--protvar`). No API dependency; both files are streamed with accession/position filtering for memory efficiency.

#### Disease & Pathways

**disease_annotations.py** - Annotates proteins with UniProt disease associations, PTM sites (phosphorylation, ubiquitination, glycosylation, lipidation), GO terms, and drug target status (`--disease`). Offline-first via streaming XML parsing of reviewed human entries, with API fallback for missing proteins.

**pathway_network.py** - Maps proteins to Reactome pathways and runs per-pathway PPI enrichment via the STRING API (`--pathways`). Builds NetworkX interaction graphs for network topology analysis (degree, centrality). Generates 2 pathway visualisation figures: **Fig 15-supp bar chart** (top 20 Reactome pathways by calibrated reviewed-human complex count with % High-tier overlay) and **Fig 15-supp network** (NetworkX spring layout of top 20 pathways, ≥ 40-overlap edges, coloured by % High-tier complexes).

#### Structural Visualisation

**pymol_scripts.py** - Generates scene-managed PyMOL `.pml` scripts with layered visualisation: chain colouring (10-chain palette, homodimer transparency), pLDDT confidence bands, interface residue sticks, pathogenicity-aware variant spheres coloured by structural context, and AlphaMissense transparency overlay (`--pymol`, `--pymol-min-tier`, `--pymol-render`). Includes metadata and biological annotation comments, pre-computed interface residue lookup to avoid redundant PDB I/O, and a `py3Dmol` fallback for in-notebook rendering. For `.pdb.bz2` inputs the generator emits an inline `bz2.open` + `cmd.read_pdbstr` block because PyMOL's CLI `load` does not transparently decompress. Toolkit output is sharded into `pymol_scripts/shard_NNNN/` subdirs (≤1000 scripts each) for filesystem scalability - the `--pymol-output` flag still controls the top-level directory; use `find <pymol_scripts> -name '<complex>.pml'` to locate a specific script.

#### Input Discovery & HPC Submission

**data_registry.py** - Centralises all data-file path references into a single registry of 16 entries, each recording expected path, source module, constant name, and whether the filename contains a version string. Resolves the project root dynamically with the precedence `explicit argument > PROTEIN_TOOLKIT_PROJECT_ROOT env var > repo fallback` so the toolkit (e.g. on HPC at `/scratch/<project>/protein-complexes-toolkit-hpc/`) and its data tree (e.g. at `/scratch/<project>/Protein_Complexes/`) can live in different locations. Provides `validate_data_dependencies()` for pre-run checks used by `--full-pipeline`, and a standalone CLI (`python data_registry.py`) for dependency checking.

**complex_resolver.py** - Discovers paired PDB/PKL inputs across three layouts (loose flat, flat directory-per-complex, sharded directory-per-complex) and writes a forensic manifest of complete pairs plus an audit of incomplete inputs with reason codes (`missing_pdb`, `missing_pkl`, `missing_both`, `empty_pkl`, `duplicate_complex_name`, `ambiguous_pdb`). Layout detection via a `^[A-Z0-9]{2}$` shard regex. Atomic manifest writes (`write .tmp -> Path.replace()`) so the audit file is never half-written. Public API `find_complexes(root, audit_dir=None, write_audit=True)` is consumed by `toolkit.py`'s main pipeline, the standalone forensic CLI (`python complex_resolver.py`), and the `--pymol` script-generator path.

**file_io.py** - Transparent compression-aware open helpers for the eight PDB-reading sites across `toolkit.py`, `pdockq.py`, `variant_mapper.py`, and `pymol_scripts.py`. Three exports: `open_text_maybe_compressed(path)` (text mode with `errors='replace'`), `open_binary_maybe_compressed(path)` (binary mode), and `decompressed_pdb_view(path)` (a context manager that materialises a `.pdb.bz2`/`.pdb.gz` into a per-complex tempfile once, yields the path, and deletes it on exit). The view is entered once per complex in `toolkit.process_single_complex` so the five sequential PDB readers (extract_pLDDT + three CA/CB passes in `read_pdb_with_chain_info_New` + SASA) all hit plain disk text after a single decompression.

**hpc_dataset_run.sh** - Production SLURM wrapper for cluster submission. Owns the entire environment so the run is reproducible across login-node sessions: `module purge && module load python/3.11.6-gcc-13.2.0`, `source .venv/bin/activate`, BLAS thread caps (`OMP_NUM_THREADS=1` + MKL/OpenBLAS/NumExpr - prevents `ProcessPoolExecutor`'s 16 workers from oversubscribing to 256 BLAS threads on 16 cores), `MPLBACKEND=Agg` + `MPLCONFIGDIR` (compute nodes have no DISPLAY and home directories are quota-restricted), `PYTHONUNBUFFERED=1` (real-time SLURM logs) and `PYTHONNOUSERSITE=1` (defensive against stray user-site installs). Runs 5 phases: `[0/4] pip check`, `[1/4] data_registry.py`, `[2/4] complex_resolver.py`, `[3/4] toolkit.py --full-pipeline`, `[4/4] visualise_results.py`. The visualisation phase is parameterised by a `VISUALISE_ARGS` env var, defaulting to `--full-figure-pack --human-supplement` (the dissertation figure pack - 24 titles / 36 PNGs); override at submit time via e.g. `sbatch --export=ALL,VISUALISE_ARGS="" hpc_dataset_run.sh` for the main-text-only pack, or `sbatch --export=ALL,VISUALISE_ARGS="--full-figure-pack --human-supplement --nonhuman-supplement" ...` to add non-human variants. See [HPC Submission](#hpc-submission) for required env vars, resource allocation, and reference performance numbers.


## Input Data Format

The toolkit expects a directory containing paired AlphaFold2-Multimer output files. **3 directory layouts and both compressed (`.bz2`) and uncompressed inputs are supported transparently** - no pre-processing or decompression step is required.

### Layout 1 - Loose flat (legacy local)

Files directly in the root, complex names parsed from filenames:

```
Protein_Complexes/
├── ProteinA_ProteinB.pdb
├── ProteinA_ProteinB.results.pkl
├── ProteinC_ProteinD_relaxed_model_1_multimer_v3_pred_0.pdb
├── ProteinC_ProteinD_result_model_1_multimer_v3_pred_0.pkl
└── ...
```

### Layout 2 - Flat directory-per-complex

Each child of the root is one complex's directory:

```
Protein_Complexes/
├── A0A0A0MQZ0_P40933/
│   ├── A0A0A0MQZ0_P40933.pdb
│   └── A0A0A0MQZ0_P40933.pkl
└── ...
```

### Layout 3 - Sharded directory-per-complex (HPC)

2-letter shard prefix groups complexes for filesystem performance:

```
Protein_Complexes/
└── A0/
    └── A0A0A0MQZ0_P40933/
        ├── A0A0A0MQZ0_P40933.pdb.bz2
        └── A0A0A0MQZ0_P40933.pkl.bz2
```

### Supported file formats

- **PDB**: `.pdb`, `.pdb.bz2`, `.pdb.gz`
- **PKL**: `.pkl`, `.pkl.bz2`, `.pkl.gz`, `.results.pkl`, `.results.pkl.bz2`
- **AF2 long-form names**: `*_relaxed_model_*.pdb[.bz2]` and `*_result_model_*.pkl[.bz2]`

Compressed inputs are read directly via a transparent compression-aware open helper (`file_io.py`); the per-complex tempfile is created once and reused across all readers in the same complex's processing window.

**Reader-API notes** (worth knowing if you patch a new reader to consume compressed inputs):

- BioPython `PDBParser.get_structure(name, source)` accepts a string path **OR** a file-like object. Passing a string `'foo.pdb.bz2'` makes it attempt to read raw bzip2 bytes as PDB lines (silent corruption, no exception). Always pass an open text handle from `file_io.open_text_maybe_compressed()` for compressed inputs.
- Biotite `PDBFile.read(source)` follows the same convention - file-like objects work, string paths to `.bz2` files don't.
- PyMOL's CLI `load` command does not transparently decompress. The toolkit's `.pml` generator emits an inline `bz2.open` + `cmd.read_pdbstr` block for `.pdb.bz2` inputs.

### Naming conventions

Each pair contains:
- A **PDB file** with ATOM records
- A **PKL file** with the AlphaFold2 result dictionary (ipTM, pTM, pLDDT, PAE)

Homodimer, isoform, and multi-chain naming patterns are also handled. Layouts 2 and 3 also produce a forensic manifest at `data/complex_manifest_audit/` listing complete pairs and an audit of skipped complexes with reason codes (`missing_pdb`, `empty_pkl`, `duplicate_complex_name`, `ambiguous_pdb`, ...).


## Output

### CSV (41 base columns, up to 155 with all features)

Progressive column counts as flags are stacked.

| Flag combination | Column count |
|---|---|
| Base (no flags) | 41 |
| `--interface` | 65 |
| `--interface --pae` | 84 |
| `--enrich` (alone) | 53 |
| `--interface --pae --enrich` | 96 |
| `+ --clustering` | 103 |
| `+ --variants` | 115 |
| `+ --stability` | 123 |
| `+ --protvar` | 131 |
| `+ --disease` | 145 |
| `+ --pathways` (= `--full-pipeline`) | **155** |

The main output CSV groups columns into:

| Category | Key Columns |
|----------|-------------|
| **Identity** | complex_name, protein_a, protein_b, complex_type (legacy coarse: Homodimer / Heterodimer / Multi-chain), n_chains, num_residues, species, structure_source, species_a, species_b, species_status (per-chain and complex-level tag: `reviewed_human` / `trembl_human` / `non_human`) |
| **Multimer Identity** | schema_version (`multimer_v1`), stoichiometry (`A2`, `AB`, `A2B`, `A2B2`, `ABCD`, `A3`…), is_homomeric, unique_accessions, chain_ids, accession_chain_map (JSON), tier_scope (`dimer_validated` \| `multimer_provisional`), filename_n_chains, pdb_n_chains, chain_count_consistency (`match` / `filename_only` / `pdb_only` / `mismatch`), complex_identity_json |
| **Core Metrics** | ipTM, pTM, ranking_confidence, pDockQ, ppv, pae_mean (global PAE matrix mean) |
| **pLDDT Statistics** | plddt_mean, plddt_median, plddt_min, plddt_max, plddt_below50/70_fraction |
| **Interface Geometry (best pair)** | best_chain_pair, n_interface_contacts, n_interface_residues_a/b, interface_residues_a/b, interface_fraction_a/b, interface_symmetry, contacts_per_interface_residue |
| **Interface pLDDT** | interface_plddt_a/b (per-chain), interface_plddt_combined, bulk_plddt_combined, interface_vs_bulk_delta, interface_plddt_high_fraction |
| **PAE Features (best pair)** | interface_pae_mean (bidirectional max), interface_pae_median, n_pae_confident_contacts, pae_confident_contact_fraction (PAE<5A), n_strict_confident_contacts, strict_confident_contact_fraction (PAE<5A AND both pLDDT>=70; used by composite), cross_chain_pae_mean, interface_pae_forward_mean, interface_pae_reverse_mean, interface_pae_directional_delta_mean/_max, n_confident_residues_a/b |
| **All-Pairs Aggregates** | pair_metrics (JSON list, length `N*(N-1)/2`), pdockq_mean, pdockq_min, pdockq_whole_complex (recomputed from all inter-chain contacts, not a mean), contact_count_total, interface_plddt_mean, symmetry_mean, symmetry_min, pae_confident_fraction_mean, strict_confident_fraction_mean (aggregates are contact-weighted; zero-contact pairs excluded from weighted means but still appear in `pair_metrics`) |
| **Composite Scoring** | interface_confidence_score, quality_tier, quality_tier_v2, composite_screen_status (screening / prioritisation label: `strong_screen_candidate` ≥ 0.85, `moderate_screen_candidate` 0.63–0.85, `weak_screen_candidate` < 0.63, `unavailable`) |
| **Audit / Data Availability** | has_pdb, has_pkl, geometry_available (`True` iff pair enumeration succeeded), composite_is_calibrated (`True` only when the composite was actually computable: `tier_scope == "dimer_validated"`, every composite input present, AND `partial_reason` empty), partial_reason (row-level recoverability diagnostic - empty for valid rows, otherwise one of **16 canonical values** covering granular PDB classes (`pdb_io_error`, `pdb_decompression_error`, `pdb_parse_error`, `pdb_no_chains`), granular PKL classes (`pkl_io_error`, `pkl_decompression_error`, `pkl_unpickle_error`, `pkl_loaded_missing_iptm`, `pkl_loaded_missing_pae`), the geometry/composite reasons (`no_positive_interface_contacts`, `missing_required_composite_inputs`), the parallel-worker `worker_exception` sentinel, and three legacy fallback aliases (`unreadable_pdb_or_structure_input`, `missing_pkl_or_pkl_unreadable`, `incomplete_input`). Values are stamped via a priority-gated helper (`_stamp_partial_reason`) so the dominant failure wins when several apply; `worker_exception` rows are produced by `_safe_process_single_complex` on uncaught exceptions and carry `quality_tier='Error'`, `quality_tier_v2=None`, `tier_scope=None`, `complex_type=None`, `_error=str(exc)`. Together with `composite_is_calibrated` these are the dissertation-safe filters: a calibrated quality claim sits inside `composite_is_calibrated == True`, and any excluded row is recoverable to a known reason), plddt_source (`pdb` / `pkl` - diagnostic for which input the pLDDT array was read from) |
| **Flags** | interface_flags (8 automated flags including paradox detection) |
| **Enrichment** (with `--enrich`) | gene_symbol_a/b, protein_name_a/b, ensembl_id_a/b, secondary_accessions_a/b, database_source, evidence_types, sequence_a/b |
| **Clustering** (with `--clustering`) | sequence_cluster_ids, sequence_cluster_count, shared_cluster_ids, shared_cluster_count, homologous_pairs, n_homologous_pairs, homology_bitscore |
| **Variants** (with `--variants`) | n_variants_a/b, n_interface_variants_a/b, n_pathogenic_interface_variants, interface_variant_enrichment, variant_details_a/b, gene_constraint_pli_a/b, gene_constraint_mis_z_a/b |
| **Stability** (with `--stability`) | eve_score_mean_a/b, eve_n_pathogenic_a/b, eve_coverage_a/b, stability_details_a/b |
| **ProtVar** (with `--protvar`) | protvar_am_mean_a/b, protvar_foldx_mean_a/b, protvar_am_n_pathogenic_a/b, protvar_details_a/b |
| **Disease** (with `--disease`) | n_diseases_a/b, disease_details_a/b, is_drug_target_a/b, n_ptm_sites_a/b, ptm_details_a/b, go_biological_process_a/b, go_molecular_function_a/b |
| **Pathways** (with `--pathways`) | reactome_pathways_a/b, n_reactome_pathways_a/b, n_shared_pathways, pathway_quality_context, ppi_enrichment_pvalue, ppi_enrichment_ratio, network_degree_a/b |

### JSONL Interface Export

When `--export-interfaces` is used, one JSON record per complex is written, containing confident interface residue sets, PAE values, and per-residue pLDDT for downstream analysis.

## Figures Generated

`visualise_results.py` script produces **24 distinct figure titles** emitted as **up to 36 PNG files** at full settings: 12 dual-emit pairs (`*.png` + `*_human.png` covering calibrated dimer vs calibrated dimer × human scopes) plus 12 single-emit figures. The default invocation produces only the main-text pack on the calibrated-dimer scope (10 figures); supplementary figures and the human dual-emit pass are opt-in via `--full-figure-pack` and `--human-supplement` respectively. Every figure carries an explicit `[scope; N=…]` subtitle annotation derived from the `species_status` and `tier_scope` taxonomies. Population filters are applied via `apply_filter()` from `visualise_filters.py` whose 14-filter registry reproduces the audit row counts exactly.

### Main narrative figures (default emit)

| # | Figure | Emit | Description |
|---|--------|------|-------------|
| **0** | Corpus Funnel *(NEW)* | single | Population definitions + screening side-callouts: 516,744 -> 406,244 recoverable -> 402,846 calibrated dimers -> 364,357 human -> 357,073 reviewed-human, with strong / moderate / weak screen-candidate side numbers. |
| 1 | Quality Scatter | dual | ipTM vs pDockQ coloured by `quality_tier_v2`. |
| 3 | Interface PAE by Tier | dual | Box + swarm of interface PAE across quality tiers (medians: High 3.4 Å / Medium 9.6 Å / Low 24.4 Å - three tiers separate cleanly). |
| 4 | Composite Tier Validation | dual | Composite score by tier + strict-confident-contact-fraction × composite Pearson r=0.84 component-consistency check. Tier-boundary lines drawn at the live thresholds `0.63 / 0.64 / 0.85`. |
| 5 | Interface vs Bulk pLDDT | dual | Scatter with diagonal + paradox-triangle overlay; 40 % of complexes above the diagonal. |
| 7 | Homo vs Hetero | dual | A2 / AB-restricted stoichiometry; ~10× High-tier divide between homodimer A2 (41.4 %) and heterodimer AB (4.3 %). |
| 8 | ipTM − pDockQ Histogram | dual | Raw + per-tier normalised distribution of the metric Δ (median 0.065, p99 0.581). **The relationship is bidirectional** - Low tier has a long negative tail where pDockQ exceeds ipTM, contradicting any "ipTM is always optimistic" framing. |
| 12 | Variant Density vs Composite Confidence | single | Interface variant density (per residue) vs composite score scatter with Spearman ρ = 0.4315 + partial ρ = 0.2809 (size-controlled). |
| **16** | Prediction Quality Paradox *(NEW)* | single | 4-panel diagnostic: pathogenic interface variants OR (69.07 High vs Low), PPI enrichment ratio (Cliff's δ 0.239), pLI ≥ 0.9 LoF-intolerant fraction (OR 0.63 - High tier depleted of LoF-intolerant genes, an honest prediction-bias acknowledgement), disorder fraction (Cliff's δ 0.005, no meaningful tier bias). |
| **17** | Screening Landscape *(NEW)* | single | Composite histogram with `0.63 / 0.85` cutoffs + `quality_tier_v2 × composite_screen_status` row-percent crosstab. Visualises the two-layer interpretation (classification × screening) in a single artefact. Under V2: High (n=13,235) -> 0 weak / 4,600 moderate / 8,635 strong, **65.24 % strong-screen purity**; Medium (n=41,111) -> 24,094 / 17,017 / 0; Low (n=348,500) -> 348,119 / 381 / 0. |

### Supplementary figures (`--full-figure-pack`)

| # | Figure | Emit | Description |
|---|--------|------|-------------|
| 1b-supp | Disorder Scatter | dual | Fig 1 coloured by `plddt_below50_fraction` overlay. Also enabled standalone via `--disorder-scatter`. |
| 2-supp | Global PAE Health Check | dual | Global PAE distribution + 5 Å reference. Median 21.3 Å; only 1,093 / 406,237 (0.27 %) below 5 Å - global PAE is broadly poor at corpus scale, hence the toolkit's focus on *interface-localised* PAE rather than the global median. |
| **4-supp** | Strict vs PAE-only Confident Contact Fraction *(NEW)* | dual | Quantifies the strict pLDDT ≥ 70 gate's contribution above PAE-only: mean delta **0.001**, median delta **0.000**. The strict gate is methodologically defensible but contributes essentially nothing to discrimination at corpus scale. |
| 6-supp | Paradox Spotlight | dual | Triptych of ΔpLDDT / PAE-only fraction / symmetry for paradox vs non-paradox cohorts (medians: 28 vs −3; 0.78 vs 0.00; 0.30 vs 0.59). 1,512 paradox cases - robust at corpus scale. |
| 7-supp | Multimer Stoichiometry | dual | **Opt-in** via `--multimer-supplement`. Multimer-exploratory buckets `A2B` / `ABC` / `A2B2` / `ABCD` / Other; descriptive only, never dissertation claims. |
| 8-supp | ipTM vs pDockQ Scatter | dual | Scatter with empirically-derived Δ > 0.52 disagreement callout (9,628 complexes, 2.4 % of calibrated dimers). The disagreement zone is biologically meaningful and is the population the paradox-triangle cut pulls from. |
| 9-supp | Chain-Count Quality Profile | dual | Four panels: best-pair pDockQ, `pdockq_mean`, `pdockq_min`, coherence gap (`pdockq − pdockq_min`) by chain count. **Order-statistic bias diagnostic** empirically justifying the `tier_scope = multimer_provisional` non-calibration policy. (Force-skipped on `--nonhuman-supplement`.) |
| 10-supp | Clustering Validation | single | Homodimer ground-truth scatter (2,952 / 2,952 = 100 % on y=x) + shared-cluster-ratio violin by tier (medians: High 0.773 / Medium 0.665 / Low 0.615; Kruskal-Wallis ε² = 0.014). |
| 11-supp | Classified Variant Sankey | single | Alluvial flow from clinical significance (Pathogenic / Likely Pathogenic / VUS / Benign) to 4-class structural context (`interface_core` / `interface_rim` / `surface_non_interface` / `buried_core`); Unknown excluded. 539,723 classified variants flow through the diagram. |
| 13-supp | Stability Predictor Cross-Validation | single | 3-panel: EVE × AlphaMissense ρ = 0.60, AlphaMissense × FoldX ΔΔG ρ = 0.42, coverage landscape by tier (EVE 16 % / AlphaMissense 80 % / FoldX 74 %). |
| 14A-supp | Disease Prevalence by Tier | single | Grouped bars by quality tier + chi-square. Cramér's V = 0.036 - the effect exists (p = 9.0e-102) but is vanishingly small. Figure footer carries the load-bearing methodological caveat: this is **annotation burden**, not disease causality. |
| 14B-supp | Top Disease Categories by Tier | single | Top 10 disease categories stacked by H/M/L. Annotation counts dominated by Low tier in every case - consistent with 14A's annotation-burden framing. |
| 15-supp (bar) | Reactome Pathway Bar Chart | single | Top 20 Reactome pathways by calibrated reviewed-human complex count with % High-tier overlay. % High-tier ranges 2 – 4 % across all 20 - no pathway with markedly elevated High-tier fraction. |
| 15-supp (network) | Reactome Pathway Network | single | NetworkX spring layout of top 20 pathways at depth level 1, 190 edges between pathways with ≥ 40 shared complexes (hierarchical parent-child links excluded). Coloured by % High-tier complexes. |
| **18-supp** | Partial Reason Dashboard *(NEW)* | single | Distribution over the 16-value `partial_reason` vocabulary across the 110,500 partial / error rows. Dominant failure: `pdb_decompression_error` = 110,397 (99.9 % - the mid-copy `.bz2` corruption pattern). Production-evidence figure for the vocabulary. |

### Per-complex PAE heatmaps (on-demand)

`--pae-heatmaps <MODELS_DIR>` generates one PAE matrix heatmap per complex with chain-boundary lines and best-pair cross-chain block highlighting. Cap the count with `--limit N`. Single-emit; outside the dual-emit / full-figure-pack framework.

### Column dependencies (per figure)

Figures 3 / 4 / 4-supp / 5 / 6-supp / 7 / 8 / 8-supp / 9-supp require both `quality_tier_v2` and interface columns (from `--interface --pae`). Fig 9-supp requires `n_chains`. Figs 11-supp / 12 require variant columns (`--interface --pae --enrich --variants`); Fig 16 additionally requires pathway columns + `pli` + `plddt_below50_fraction`. Figs 14A-supp / 14B-supp require disease columns (`--disease`). Fig 15-supp (bar + network) requires pathway columns (`--pathways`). Fig 13-supp requires stability + ProtVar columns (`--stability --protvar`). Fig 10-supp requires clustering columns (`--clustering`). Fig 17 requires `composite_screen_status` + `interface_confidence_score` + `quality_tier_v2`. Fig 18-supp requires `partial_reason`.

### Population scopes

Population filters are centralised in `visualise_filters.py` as a 14-filter registry: `all_rows` / `recoverable` / `calibrated_dimer` / `composite_status_present` / `composite_screenable` / `strong_screen_candidate` / `moderate_screen_candidate` / `weak_screen_candidate` / `human_broad` / `human_strict` / `multimer_exploratory` / `partial_error` / `calibrated_human_broad` / `calibrated_human_strict`. Each figure declares its scope by name; the chosen filter's audit-aligned `before -> after` row counts print to stdout per figure for runtime traceability. The main-narrative scope is `calibrated_dimer` (six-clause definition: `tier_scope == 'dimer_validated'` AND `composite_is_calibrated == True` AND recoverable AND six numeric metrics non-NaN AND `n_interface_contacts > 0`); the `_human` dual-emit pass adds `species_status ∈ {reviewed_human, trembl_human}` on top.

## Single-complex summary figure (`render_complex_summary.py`)

A deliberately narrow CLI that turns **one** AlphaFold2-Multimer complex directory into **one** polished summary PNG. It reuses the production metric path (`toolkit.process_single_complex`) — no scientific formulas or tier thresholds are reimplemented — renders two PyMOL views in the same orientation (chain/interface and pLDDT confidence), composes the generic figure (two legends, the two views, and one metric box with the baseline → interface-informed tier transition and three descriptive rows), writes exactly one PNG, and removes every intermediate file automatically.

- **Dimer-only** — the interface-informed assessment is calibrated on dimers; a non-dimer input fails with a clear message.
- **Supported inputs** — one PDB and one PKL in the directory, in any of `.pdb` / `.pdb.gz` / `.pdb.bz2` and `.pkl` / `.pkl.gz` / `.pkl.bz2` (and the `*_relaxed_model_*` / `*_result_model_*` / `.results.pkl` forms).
- **No network** — gene/protein display names come from the local alias table when present and fall back to accessions otherwise.

```bash
python render_complex_summary.py \
    --input-dir "/scratch/prj/chmi_msa/Protein_Complexes/P6/P61088_Q9H4P4" \
    --output    "P61088_Q9H4P4_summary.png" \
    --pymol-executable "$HOME/envs/pymol/bin/pymol"
# -> Wrote P61088_Q9H4P4_summary.png  (2000x1600)
```

Optional flags: `--overwrite` (replace an existing output; the default is to fail), `--width` / `--height` (positive integers), `--pymol-executable` (path to a headless PyMOL; defaults to `pymol` on `PATH`).

**Output replacement:** if the output already exists the run fails unless `--overwrite` is given; the final PNG is written atomically, so a failed run never leaves a truncated image.

### PyMOL runtime + dependency delta

`render_complex_summary.py` adds **no new pip dependency**: it imports only the standard library, the toolkit's existing modules, and packages already pinned in `requirements.txt` (`matplotlib` composes the figure and bundles the DejaVu Sans font; `numpy`/`pandas` are used by the reused metric path). **`requirements.txt` is unchanged.**

The one requirement beyond `requirements.txt` is a **headless PyMOL executable**, kept as an *external runtime* (invoked via `subprocess`, selected with `--pymol-executable`) rather than a pip dependency — `pip install pymol-open-source` is unreliable. On King's CREATE there is no PyMOL module (`module spider pymol` → not found), so install it once with the conda that is already on the login node, into a self-contained env that does not touch the toolkit's `.venv`:

```bash
# one-time install (login node is fine — it is only a download)
conda create -y -p "$HOME/envs/pymol" -c conda-forge pymol-open-source
# verify (headless CPU ray-tracing; no X server required) before submitting a SLURM job
"$HOME/envs/pymol/bin/pymol" -cq -d "print('PyMOL', cmd.get_version()[0]); quit"
```

The PyMOL env brings its own Python and libraries and is **completely independent** of the toolkit's environment (they only communicate through the executable path), so you never `conda activate` it. To avoid passing `--pymol-executable` every time, make PyMOL discoverable from your toolkit environment once — either **symlink it onto `PATH`**:

```bash
ln -s "$HOME/envs/pymol/bin/pymol" "$VIRTUAL_ENV/bin/pymol"    # with the toolkit venv active
```

**or set `$PYMOL_EXECUTABLE`** (the CLI checks `--pymol-executable`, then `$PYMOL_EXECUTABLE`, then `pymol` on `PATH`) — persist it by appending to the venv's activate script:

```bash
echo 'export PYMOL_EXECUTABLE="$HOME/envs/pymol/bin/pymol"' >> "$VIRTUAL_ENV/bin/activate"
```

After either one-time step, activating the toolkit environment is all that's needed and the CLI runs with no `--pymol-executable` flag. (Verified route: conda-forge `pymol-open-source` 3.1.0.)

**This output is an inspection and communication figure. It is not evidence that the predicted interaction is biologically real.**

## Acknowledgements
Developed by Talhah Zubayer under the supervision of David Burke as part of the MSc Applied Bioinformatics programme at King's College London.
