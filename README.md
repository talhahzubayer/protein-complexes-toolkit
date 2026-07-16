# Protein Complexes Toolkit

An interface-aware quality-assessment and triage toolkit for AlphaFold2-Multimer protein-complex predictions.

*MSc Applied Bioinformatics research project, King's College London.*
**Author:** Mohammad Talhah Zubayer · **Supervisor:** Dr David Burke.

## Overview

The toolkit operates downstream of AlphaFold2-Multimer. It reads the predicted structure (`.pdb`) and confidence (`.pkl`) files that AlphaFold2-Multimer produces and assesses the confidence of each predicted inter-chain interface, without generating or modifying the predictions themselves. The problem it addresses is one of scale: large prediction sets contain far more complexes than can be inspected by eye, and the two most relevant confidence metrics — ipTM, which summarises the predicted arrangement of the chains as a single complex-level value, and pDockQ, which reads the interface directly but depends on contact abundance — measure different properties and frequently disagree. Neither, on its own, describes confidence at the predicted contact site.

To fill that gap, the toolkit reads confidence locally at the predicted interface. It retains ipTM and pDockQ as transparent baselines and adds interface-localised measurements: mean interface pLDDT (per-residue local confidence), interface PAE (predicted aligned error between contacting residues on different chains), interface symmetry and contact density. These are combined into a single continuous composite score used for screening and prioritisation, which is reported separately from a conservative categorical quality tier. This dual output keeps classification and prioritisation distinct rather than forcing one score to serve both purposes.

Optional stages attach biological context — gene and protein identity, sequence-cluster homology, variant mapping, variant-effect and stability predictions, disease and pathway annotation — and generate PyMOL scripts and a batch figure suite. The same command-line pipeline runs on a handful of complexes locally and on hundreds of thousands of complexes on a high-performance computing (HPC) cluster, with checkpointing, incremental runs and memory-bounded batching. The submitted dissertation applied it to 516,744 predicted complexes.

## Scope and interpretation

The toolkit assesses confidence and supports prioritisation. It does not validate a structure or an interaction experimentally, and its outputs should be read within the following boundaries, which follow the dissertation's stated scope:

- The composite `interface_confidence_score` is a screening and prioritisation heuristic. Its weights were chosen by design rather than learned from labelled outcomes, so it is not a probability of correctness or an externally calibrated estimate of structural accuracy.
- Structural confidence does not establish biological reality. A confidently predicted interface is a candidate for closer examination, not evidence that the interaction forms in cells.
- Biological annotations provide corroboration and context, not proof. Membership of an interaction database, a disease association or a shared pathway supports interpretation but cannot confirm that a predicted interface is real.
- Calibrated classification is restricted to dimers. Complexes with more than two chains are processed structurally but labelled `multimer_provisional`, because the composite uses best-pair inputs that are not calibrated for larger assemblies.
- Because no native structures were available at this scale, external structural benchmarking against experimentally resolved complexes remains necessary to establish predictive accuracy.

## Repository structure

```text
protein-complexes-toolkit/
├── toolkit.py                 # Batch orchestrator and main command-line entry point
├── read_af2_nojax.py          # Reads AlphaFold2 PKL files without a JAX installation
├── pdockq.py                  # pDockQ baseline and best inter-chain pair selection
├── interface_analysis.py      # Interface geometry, PAE features, composite score, flags
│
├── complex_resolver.py        # PDB/PKL pair discovery and audit manifest
├── data_registry.py           # External-data registry and pre-run validation
├── file_io.py                 # Compression-aware file opening (plain / .gz / .bz2)
│
├── id_mapper.py               # Identifier cross-referencing and species classification
├── database_loaders.py        # STRING / BioGRID / HuRI / HuMAP interaction parsers
├── overlap_analysis.py        # Cross-database overlap and Venn/UpSet diagrams
├── string_api.py              # STRING API client (offline-first, cached)
├── protein_clustering.py      # STRING sequence clusters and homologous-pair detection
├── variant_mapper.py          # UniProt/ClinVar/ExAC variant mapping and structural context
├── stability_scorer.py        # EVE variant-effect scoring
├── protvar_client.py          # Offline AlphaMissense and monomeric FoldX scoring
├── disease_annotations.py     # UniProt disease / PTM / GO / drug-target annotation
├── pathway_network.py         # Reactome pathways, PPI enrichment and network analysis
│
├── visualise_results.py       # Batch figure suite generated from a results CSV
├── visualise_filters.py       # Named population-filter registry used by the figures
├── pymol_scripts.py           # PyMOL .pml script generation (with py3Dmol fallback)
├── render_complex_summary.py  # Single-complex structural summary PNG for one dimer
│
├── hpc_dataset_run.sh         # SLURM wrapper: full-pipeline production run
├── hpc_incremental_run.sh     # SLURM wrapper: incremental and chunked runs
│
├── requirements.txt           # Python dependencies
├── Docs/                      # Dissertation, full command reference and output schema
│   ├── MSc Dissertation Final.pdf
│   ├── Toolkit_Commands_List.md
│   └── OUTPUT_SCHEMA.md
└── data/                      # External databases (not included; see below)
```

The `data/` directory and the large external files it holds are not stored in this repository.

## Reproduction requirements

"Reproduce" can mean three different things here, and the repository supports them to different degrees.

**Run the method on your own predictions.** If you supply your own paired AlphaFold2-Multimer PDB and PKL files, you can run the same pipeline and obtain the same kinds of output. This reproduces the *method* on another dataset; it does not reproduce the dissertation's numbers.

**Reproduce the dissertation's population-level analyses and figures.** This requires the final results CSV (`results_516744.csv`). That file is **not included in this repository** — the dissertation reports it as stored on King's College London's CREATE cluster alongside an execution-ready copy of the code. Given the CSV, the figure commands in [Reproducing the dissertation workflow](#reproducing-the-dissertation-workflow) regenerate the population-level figures (the dissertation's Figures 1–5 and 7–9) from it. The dissertation's Figure 6 — the two worked structural examples — is rendered from the raw PDB/PKL inputs rather than from the CSV, so it belongs to the raw-to-results level below.

**Reproduce the complete raw-to-results workflow.** This additionally requires the original AlphaFold2-Multimer prediction corpus (the paired PDB and PKL files), the same external-data releases, and the same processing and consolidation steps. The source predictions were generated upstream and provided by the supervisor; the toolkit assesses them but does not generate them. **The prediction corpus is not part of this repository**, so a full raw-to-results reproduction depends on obtaining those inputs separately.

The scientific rationale, reported results and limitations are documented in the submitted dissertation, [`Docs/MSc Dissertation Final.pdf`](Docs/MSc%20Dissertation%20Final.pdf).

## Installation

The toolkit requires **Python 3.11 or newer** (the dissertation analyses ran under Python 3.11.6). Create a virtual environment and install the pinned dependencies:

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

JAX is **not** required. `read_af2_nojax.py` reads the confidence arrays from AlphaFold2 PKL files and converts them to NumPy without a JAX installation.

Once the external data is in place (next sections), confirm it before a run:

```bash
python data_registry.py
```

This checks the 16 required data files (two further registry entries are output directories that the pipeline creates itself) and prints a per-group `[ OK ] / [MISSING]` report, exiting non-zero if anything required is absent. `render_complex_summary.py` additionally needs a headless PyMOL executable, which is an external runtime rather than a Python dependency; see [Generate the structural examples](#generate-the-structural-examples).

## Input predictions

The pipeline expects a directory of paired AlphaFold2-Multimer outputs. Each complex is represented by a PDB structure file and a PKL confidence file; the PDB supplies the geometry and per-residue B-factor pLDDT, while the PKL supplies ipTM, pTM, pLDDT and the PAE matrix. Discovery, layout detection and decompression are automatic, so no manual preparation step is required.

`toolkit.py --dir <MODELS_DIR>` recognises three directory layouts:

**Loose flat** — files sit directly in the root and complex names are parsed from the filenames. This layout is handled in place and does not write a manifest.

```text
Protein_Complexes/
├── A0A0A0MQZ0_P40933.pdb
├── A0A0A0MQZ0_P40933.pkl
└── ...
```

**Directory per complex** — each child directory holds one complex's files.

```text
Protein_Complexes/
└── A0A0A0MQZ0_P40933/
    ├── A0A0A0MQZ0_P40933.pdb
    └── A0A0A0MQZ0_P40933.pkl
```

**Sharded directory per complex** — a two-character shard prefix (matching `^[A-Z0-9]{2}$`) groups complexes for filesystem performance. This is the layout used for the HPC dataset.

```text
Protein_Complexes/
└── A0/
    └── A0A0A0MQZ0_P40933/
        ├── A0A0A0MQZ0_P40933.pdb.bz2
        └── A0A0A0MQZ0_P40933.pkl.bz2
```

The accepted suffixes are `.pdb`, `.pdb.bz2` and `.pdb.gz` for structures, and `.pkl`, `.pkl.bz2`, `.pkl.gz`, `.results.pkl` and `.results.pkl.bz2` for confidence files; the long-form AlphaFold names `*_relaxed_model_*` and `*_result_model_*` are also matched. One detail is layout-dependent: the two directory-based layouts are resolved by `complex_resolver.py`, which recognises the uncompressed and `.bz2` forms, whereas `.gz` inputs are recognised only in the loose layout. Compressed files are decompressed once per complex and reused across the readers, so no separate decompression step is needed.

The directory-based layouts also produce a forensic manifest. Running the resolver directly is a useful way to catalogue and audit inputs before a full run:

```bash
python complex_resolver.py --root <MODELS_DIR>
```

This writes, under `data/complex_manifest_audit/runs/<run_id>/`, a manifest of the complete PDB/PKL pairs, an `incomplete_inputs.tsv` listing skipped inputs with a reason code (for example `missing_pdb`, `empty_pkl`, `ambiguous_pdb` or `duplicate_complex_name`), and a run summary; a `latest/` mirror and a `latest_run_id.txt` pointer track the most recent run. Because each pair is fingerprinted by file size and modification time, the manifest lets later incremental runs detect inputs that have changed. With no `--root`, the resolver reads the `PROTEIN_COMPLEXES_ROOT` environment variable. The command exits `0` when at least one complete pair is found and `1` otherwise, so it can gate a batch script.

## Setting up external data

The `data/` directory is **not included** in this repository because the external database files are large (about 35 GB in total). The structural core (`--interface --pae`) needs none of these files — it uses only the PDB/PKL inputs. External data is required only by the optional annotation stages, as summarised below:

| Stage / flag | Data required | Location |
| ------------ | ------------- | -------- |
| structural core (`--interface --pae`) | none (PDB/PKL only) | — |
| `--enrich` | STRING aliases | `data/ppi/` |
| `--databases` | STRING links, BioGRID, HuRI, HuMAP | `data/ppi/` |
| `--clustering` | STRING sequence clusters | `data/clusters/` |
| `--variants` | UniProt, ClinVar, ExAC | `data/variants/` |
| `--stability` | EVE scores and UniProt ID mapping | `data/stability/` |
| `--protvar` | AlphaMissense and AFDB FoldX | `data/stability/` |
| `--disease` | UniProt Swiss-Prot XML | `data/pathways/` |
| `--pathways` | Reactome mappings and hierarchy | `data/pathways/` |

Create the directory structure:

```bash
mkdir -p data/ppi data/clusters data/variants data/stability data/pathways
```

> **Version note.** The filenames below are the release versions used for the dissertation workflow, and are the defaults hard-coded in the modules. Newer upstream releases will have different filenames and may require corresponding path or parser changes; they may also prevent exact numerical reproduction of the dissertation results. If you download a newer version, update the corresponding constant in the source file named by the `[MISSING]` message that `python data_registry.py` prints.

**PPI databases — download into `data/ppi/`:**

| File | Source | Download |
|------|--------|----------|
| `9606.protein.aliases.v12.0.txt` | STRING | [string-db.org/cgi/download](https://string-db.org/cgi/download?species_text=Homo+sapiens) — select *Homo sapiens*, download `9606.protein.aliases.v12.0.txt.gz`, decompress *(used by `--enrich`)* |
| `9606.protein.links.v12.0.txt` | STRING | Same page — download `9606.protein.links.v12.0.txt.gz`, decompress |
| `BIOGRID-ALL-5.0.253.tab3.txt` | BioGRID | [downloads.thebiogrid.org](https://downloads.thebiogrid.org/File/BioGRID/Release-Archive/BIOGRID-5.0.253/BIOGRID-ALL-5.0.253.tab3.zip) — extract the `.tab3.txt` from the zip |
| `HuRI.tsv` | HuRI | [interactome-atlas.org/download](https://interactome-atlas.org/download) — download `HuRI.tsv` |
| `humap2_ppis_ACC_20200821.pairsWprob` | hu.MAP 2.0 | [humap2.proteincomplexes.org/download](https://humap2.proteincomplexes.org/download) — "Protein Interaction Network with probability scores (Uniprot gzip)", decompress |

The four link/interaction files are used by `--databases`; the aliases file is used by `--enrich`.

**STRING sequence clusters — download into `data/clusters/`:**

| File | Source | Download |
|------|--------|----------|
| `9606.clusters.proteins.v12.0.txt` | STRING | [string-db.org/cgi/download](https://string-db.org/cgi/download?species_text=Homo+sapiens) — select *Homo sapiens*, download `9606.clusters.proteins.v12.0.txt.gz`, decompress *(used by `--clustering`)* |

**Variant databases — download into `data/variants/` (used by `--variants`):**

| File | Source | Download |
|------|--------|----------|
| `homo_sapiens_variation.txt` | UniProt | [ftp.uniprot.org/…/variants/](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/variants/) — `homo_sapiens_variation.txt.gz`, decompress |
| `variant_summary.txt` | ClinVar | [ftp.ncbi.nlm.nih.gov/…/clinvar/tab_delimited/](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/) — `variant_summary.txt.gz`, decompress |
| `forweb_cleaned_exac_r03_march16_z_data_pLI_CNV-final.txt` | ExAC/gnomAD | [gnomad.broadinstitute.org/downloads](https://gnomad.broadinstitute.org/downloads) — "Gene constraint scores TSV", decompress |

**EVE scores and UniProt identifier mapping — download into `data/stability/` (used by `--stability`):**

| File | Source | Download |
|------|--------|----------|
| `EVE_all_data/` (per-protein CSVs) | EVE | [evemodel.org/download/bulk](https://evemodel.org/download/bulk) — download the **"All variant files"** archive, extract into `data/stability/` |
| `HUMAN_9606_idmapping.dat` | UniProt | [ftp.uniprot.org/…/idmapping/by_organism/](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/) — `HUMAN_9606_idmapping.dat.gz`, decompress |

Only the variant-files archive is needed from the EVE download page; the other archives (multiple sequence alignments, VCFs, diagnostic curves) are not used. `HUMAN_9606_idmapping.dat` maps UniProt accessions to entry names (for example `P61981` → `1433G_HUMAN`), which are the EVE CSV filenames.

**AlphaMissense and AFDB FoldX — download into `data/stability/` (used by `--protvar`):**

| File | Source | Download |
|------|--------|----------|
| `AlphaMissense_aa_substitutions.tsv` | Zenodo | [zenodo.org/records/10813168](https://zenodo.org/records/10813168) — `AlphaMissense_aa_substitutions.tsv.gz`, decompress |
| `afdb_foldx_export_20250210.csv` | EBI | [ftp.ebi.ac.uk/…/ProtVar/predictions/stability/](https://ftp.ebi.ac.uk/pub/databases/ProtVar/predictions/stability/) — download `2025.02.10_foldx_energy.csv.gz`, decompress, and rename to `afdb_foldx_export_20250210.csv` |

These two files provide offline AlphaMissense pathogenicity scores and monomeric FoldX stability predictions, so `--protvar` needs no internet access. The FoldX values estimate the change in monomeric folding stability of a variant; they are not a binding or interface energy.

**UniProt and Reactome — download into `data/pathways/`:**

| File | Source | Download |
|------|--------|----------|
| `uniprot_sprot_human.xml` | UniProt | [ftp.uniprot.org/…/taxonomic_divisions/](https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/taxonomic_divisions/) — `uniprot_sprot_human.xml.gz`, decompress *(used by `--disease`)* |
| `UniProt2Reactome_All_Levels.txt` | Reactome | [reactome.org/download/current/](https://reactome.org/download/current/) *(used by `--pathways`)* |
| `ReactomePathwaysRelation.txt` | Reactome | [reactome.org/download/current/](https://reactome.org/download/current/) *(used by `--pathways`)* |

**Expected directory contents:**

```text
data/
├── ppi/
│   ├── 9606.protein.links.v12.0.txt                              (~616 MB)
│   ├── 9606.protein.aliases.v12.0.txt                            (~195 MB)
│   ├── BIOGRID-ALL-5.0.253.tab3.txt                              (~1.48 GB)
│   ├── HuRI.tsv                                                  (~1.6 MB)
│   └── humap2_ppis_ACC_20200821.pairsWprob                       (~439 MB)
├── clusters/
│   └── 9606.clusters.proteins.v12.0.txt                          (~40 MB)
├── variants/
│   ├── homo_sapiens_variation.txt                                (~2.2 GB)
│   ├── variant_summary.txt                                       (~1.1 GB)
│   └── forweb_cleaned_exac_r03_march16_z_data_pLI_CNV-final.txt  (~2 MB)
├── stability/
│   ├── HUMAN_9606_idmapping.dat                                  (~145 MB)
│   ├── EVE_all_data/                                             (~10 GB, per-protein CSVs)
│   ├── AlphaMissense_aa_substitutions.tsv                        (~6.3 GB)
│   └── afdb_foldx_export_20250210.csv                            (~7.7 GB)
└── pathways/
    ├── uniprot_sprot_human.xml                                   (~1.02 GB)
    ├── UniProt2Reactome_All_Levels.txt                           (~110 MB)
    └── ReactomePathwaysRelation.txt                             (~611 KB)
```

`data/string_api_cache/` (STRING API response cache) and `pymol_scripts/` (PyMOL output) are created automatically.

## Reproducing the dissertation workflow

This section documents the operational sequence used in the submitted project. It covers running the complete pipeline locally, the initial and incremental HPC runs, memory-bounded batching, resumption and consolidation, and the figure and structural outputs. For selective execution, individual flags and the standalone module tools, see [`Docs/Toolkit_Commands_List.md`](Docs/Toolkit_Commands_List.md).

### Validate the external data

Confirm that every registered dataset is present before starting, so a missing file is reported in seconds rather than causing a failure part-way through a long run:

```bash
python data_registry.py
```

### Audit the prediction inputs

Catalogue the input predictions and record which PDB/PKL pairs are complete:

```bash
python complex_resolver.py --root <MODELS_DIR>
```

The manifest it writes is the record of exactly which complexes entered the run, and its file-size and modification-time fingerprints are what later incremental runs use to detect changed inputs. Capturing it once before a dataset is expanded gives a baseline against which additions can be identified.

### Run the complete pipeline locally

`--full-pipeline` activates every stage using the registered default data paths and validates that all required data files exist before processing begins. Only `--dir` is required; `-w`/`--workers` sets the number of parallel workers.

```bash
python toolkit.py --full-pipeline --dir <MODELS_DIR> -w 8 \
    --output results.csv --export-interfaces interfaces.jsonl
```

This is equivalent to enabling `--interface --pae --enrich --databases --clustering --variants --stability --protvar --disease --pathways --pymol --checkpoint` with default paths. To run only some stages, stack the individual flags instead; the full flag reference and dependencies are in [`Docs/Toolkit_Commands_List.md`](Docs/Toolkit_Commands_List.md).

### Run the initial HPC workflow

`hpc_dataset_run.sh` submits a clean, end-to-end production run to a SLURM cluster. It pins the two cluster paths near the top of the file, so edit those lines for your site before submitting:

```bash
# In hpc_dataset_run.sh, set these to your paths:
#   export PROTEIN_TOOLKIT_PROJECT_ROOT=/scratch/<project>/protein-complexes-toolkit-hpc
#   export PROTEIN_COMPLEXES_ROOT=/scratch/<project>/Protein_Complexes
sbatch hpc_dataset_run.sh
```

The wrapper loads the Python module, activates the project virtual environment, caps the BLAS and NumExpr thread counts so worker processes do not oversubscribe the allocated CPUs, and runs five phases: a dependency check, external-data validation, input resolution, the full pipeline (writing `results.csv`, `interfaces.jsonl` and `pymol_scripts/`), and figure generation. The figure step reads the `VISUALISE_ARGS` environment variable, which defaults to `--full-figure-pack --human-supplement`.

The script requests 16 CPUs, 80 GB of memory and a 48-hour walltime by default. These are starting points rather than fixed requirements: the dissertation's initial 41,196-complex run completed in under six hours at a peak of roughly 67 GB, but the completed dataset was ultimately assembled through memory-bounded batches of up to 100,000 complexes under a 128 GB allocation, because the current design accumulates results in memory (see [Process the dataset in memory-bounded batches](#process-the-dataset-in-memory-bounded-batches)).

### Process an expanded dataset

When new complexes are added to a corpus that has already been processed, `hpc_incremental_run.sh` processes only the complexes that are absent from a cumulative historical CSV. It runs `toolkit.py --full-pipeline --skip-existing <historical results.csv>` and does not render figures, because figures must be generated from the consolidated dataset rather than from an incremental fragment.

```bash
sbatch --export=ALL,HISTORICAL_RESULTS_CSV=results.csv hpc_incremental_run.sh
```

The wrapper reads the complex names already present in `HISTORICAL_RESULTS_CSV`, processes only the remainder, and writes the new rows to a **separate** `results_incremental_<stamp>_<job>.csv` (and a matching interfaces JSONL). It logs the historical CSV's SHA-256 checksum at the start so that the baseline can be confirmed unchanged across a retry.

### Process the dataset in memory-bounded batches

At the largest scale, memory rather than runtime is the limiting factor, because completed rows and intermediate annotations accumulate in memory over a single invocation. The dissertation therefore assembled the final dataset in batches. The `LIMIT` environment variable (which appends `--limit N` to the toolkit call) caps the number of complexes processed per submission, taken from the alphabetically-sorted set of not-yet-processed complexes:

```bash
sbatch --export=ALL,HISTORICAL_RESULTS_CSV=results.csv,LIMIT=100000 hpc_incremental_run.sh
```

Each batch is then handled as a short loop: submit one batch, confirm it completed, consolidate its output into the cumulative CSV, and submit the next batch against the updated cumulative file. Because the chunk is selected before any resumption filtering, a crashed and resumed batch processes the same set of complexes as a clean one, provided `HISTORICAL_RESULTS_CSV` is unchanged between attempts.

### Resume an interrupted batch

To recover an interrupted batch, resubmit against the **same** partial output files with `RESUME=1`, which adds `--resume` to the toolkit call so that complexes already written to the current checkpoint are skipped:

```bash
sbatch --export=ALL,RESUME=1,\
HISTORICAL_RESULTS_CSV=results.csv,\
OUTPUT_CSV=results_incremental_<stamp>_<job>.csv,\
INTERFACES_JSONL=interfaces_incremental_<stamp>_<job>.jsonl \
hpc_incremental_run.sh
```

`OUTPUT_CSV` and `INTERFACES_JSONL` must point at the interrupted batch's own partial files, and `HISTORICAL_RESULTS_CSV` must be identical to the original attempt; otherwise the batch boundary shifts. `--skip-existing` (an append-only filter against a completed historical CSV) and `--resume` (in-flight crash recovery from the current checkpoint) read independent state and can be combined.

### Consolidate incremental outputs

Incremental and batched runs write their rows to separate files by design; they do not modify the historical CSV in place. Each incremental output must therefore be merged into the cumulative CSV and JSONL, and the merged CSV promoted as the `HISTORICAL_RESULTS_CSV` reference for the next run — otherwise the wrapper will rediscover and reprocess the same complexes. **There is no automated merge helper in this repository**; consolidation is a manual step, and the dissertation lists automatic merging with column and duplicate checks as future work. The minimum safe requirements are that the merged file preserves the column order, contains no duplicate `complex_name` values, and keeps every interface JSONL record's complex present in the CSV. A safe step-by-step merge procedure is given in [`Docs/Toolkit_Commands_List.md`](Docs/Toolkit_Commands_List.md).

### Generate the dissertation analyses

Once a consolidated `results.csv` exists, `visualise_results.py` generates the figures from it. The default invocation produces the main-text figure pack; the dissertation figure suite additionally requires the supplementary figures, which are produced by `--full-figure-pack`, and the human-subset structural variants, produced by `--human-supplement`:

```bash
python visualise_results.py results.csv --output-dir Output --full-figure-pack --human-supplement
```

Figures are written as PNG files to `Output/`. Each figure degrades gracefully: one whose required columns are absent is skipped with a message rather than causing an error. One caveat is worth knowing when locating a specific figure. The output filenames use the dissertation's Results numbering (for example `Fig_2A_Interface_PAE_by_Quality_Tier.png` corresponds to the dissertation's Figure 2A), but the internal plotting-function names retain an older numbering that no longer matches, so the function names should not be used to identify a figure. The `Fig_1` to `Fig_5` filenames align with the dissertation's Figures 1 to 5; because the dissertation's Figure 6 is the structural-examples figure produced separately (below), the `Fig_6`, `Fig_7` and `Fig_8` output files correspond to the dissertation's Figures 7, 8 and 9 respectively. The full filename-to-figure mapping is documented in [`Docs/Toolkit_Commands_List.md`](Docs/Toolkit_Commands_List.md). The dissertation's Results narrative and the figures themselves are in the dissertation.

### Generate the structural examples

The dissertation's Figure 6 uses two worked examples to show how interface-localised evidence changed a prediction's classification — one dimer raised from low to high tier by strong local interface evidence, and one lowered from high to medium by weak contact-level support. Each example is a single-complex structural summary produced by `render_complex_summary.py`:

```bash
python render_complex_summary.py \
    --input-dir <COMPLEX_DIR> \
    --output    complex_summary.png \
    --pymol-executable "$HOME/envs/pymol/bin/pymol"
```

The script reuses the production metric path, so the metrics it displays are computed exactly as in the batch pipeline. It renders a chain-and-interface view and a pLDDT-confidence view of the same complex in one orientation, composes them with a metric panel into a single PNG (2000 × 1600 pixels by default), and removes its intermediate files. It is **dimer-only**: a non-dimer or non-calibrated input fails with a clear message. It requires no network access — display names come from the local alias table when present and fall back to accessions — and it writes the PNG atomically, failing rather than overwriting an existing file unless `--overwrite` is given.

`render_complex_summary.py` produces **one** PNG per complex. The dissertation's two-example figure was assembled by rendering the two examples separately and composing them; that final side-by-side composition is a manual step and is not performed by the script.

The script invokes a **headless PyMOL executable** through a subprocess, resolving it from `--pymol-executable`, then the `PYMOL_EXECUTABLE` environment variable, then `pymol` on the `PATH`. Where no PyMOL module is available, install one once with conda into a self-contained environment (this does not touch the toolkit's virtual environment):

```bash
conda create -y -p "$HOME/envs/pymol" -c conda-forge pymol-open-source
"$HOME/envs/pymol/bin/pymol" -cq -d "print('PyMOL', cmd.get_version()[0]); quit"   # verify headless
```

Setting `export PYMOL_EXECUTABLE="$HOME/envs/pymol/bin/pymol"` then lets the script run without the `--pymol-executable` flag. This output is an inspection and communication figure; it is not evidence that the predicted interaction is biologically real.

## Outputs

A full run produces the following artefacts.

| Output | Purpose |
| ------ | ------- |
| Results CSV | One row per predicted complex; up to 155 fields depending on the stages enabled |
| Interface JSONL | Confident interface residues, PAE and per-residue pLDDT, one record per complex |
| PyMOL scripts | `.pml` scripts for structural inspection of qualifying complexes |
| Batch PNG figures | Population-level analyses generated from the results CSV |
| Single-complex summary PNG | A structural example for one dimer |
| Audit manifests | Input-discovery and per-run traceability records |

The interface JSONL is written when `--export-interfaces <PATH>` is given (it implies `--interface --pae`) and contains only complexes reaching the High or Medium v2 tier. PyMOL scripts are written when `--pymol` is given, filtered by v2 tier (High by default) and sharded into subdirectories of up to 1000 scripts. Detailed flag behaviour, defaults and output locations are documented in [`Docs/Toolkit_Commands_List.md`](Docs/Toolkit_Commands_List.md).

## Understanding the output CSV

The results CSV holds one row per complex. The base output has 41 columns; each optional stage appends its own block, up to 155 columns with the full pipeline:

| Stage combination | Columns |
| ----------------- | :-----: |
| Base (no flags) | 41 |
| `--enrich` | 53 |
| `--interface` | 65 |
| `--interface --pae` | 84 |
| `--interface --pae --enrich` | 96 |
| `+ --clustering` | 103 |
| `+ --variants` | 115 |
| `+ --stability` | 123 |
| `+ --protvar` | 131 |
| `+ --disease` | 145 |
| `+ --pathways` (= `--full-pipeline`) | **155** |

At a high level the columns fall into a few groups: complex identity and species scope; the AlphaFold model metrics (ipTM, pTM, pDockQ, whole-complex pLDDT); interface geometry and interface-localised confidence; the classification and screening outputs; recoverability and calibration flags; whole-complex aggregates for multimers; and the optional biological-annotation blocks. Seven fields carry most of the interpretive weight, and reading them correctly matters more than any single metric:

- **`quality_tier`** — the baseline (v1) tier from ipTM and pDockQ alone.
- **`quality_tier_v2`** — the final tier, which reclassifies v1 using the composite score (strong interface evidence can rescue a low prediction; weak interface evidence can downgrade a high one). This is the tier to use for interface-aware statements.
- **`interface_confidence_score`** — the composite screening score on a 0–1 scale. It ranks predictions; it is not a probability of correctness.
- **`composite_screen_status`** — a prioritisation label (`strong_`/`moderate_`/`weak_screen_candidate`, or `unavailable`) derived from the composite score. It sits beside the tier rather than replacing it.
- **`tier_scope`** — `dimer_validated` (two chains) or `multimer_provisional`. Calibrated claims apply only within `dimer_validated`.
- **`composite_is_calibrated`** — true only when the composite was genuinely computable on a dimer with all inputs present. A numeric score alone is not sufficient; filter on this flag for any calibrated claim.
- **`partial_reason`** — empty for a fully assessed row; otherwise a code identifying why the row is incomplete. A non-empty value marks a diagnostic row, not a low-confidence result.

Every emitted field is documented individually — with its stage, type, range, interpretation, caveats and role in the submitted dissertation — in [`Docs/OUTPUT_SCHEMA.md`](Docs/OUTPUT_SCHEMA.md), which also gives the reproducible population filters used to define the dissertation's analysis groups.

## Further documentation

This README is deliberately limited to the principal reproduction workflow. The complete references are:

| Document | Purpose |
| -------- | ------- |
| [`Docs/MSc Dissertation Final.pdf`](Docs/MSc%20Dissertation%20Final.pdf) | Submitted scientific rationale, methods, results and limitations |
| [`Docs/Toolkit_Commands_List.md`](Docs/Toolkit_Commands_List.md) | Complete command-line reference and the broader toolkit capabilities, including functionality not used in the submitted dissertation |
| [`Docs/OUTPUT_SCHEMA.md`](Docs/OUTPUT_SCHEMA.md) | Complete CSV field reference and each field's role in the dissertation |

## Acknowledgements

Developed by Mohammad Talhah Zubayer under the supervision of Dr David Burke as part of the MSc Applied Bioinformatics programme at King's College London. The AlphaFold2-Multimer predictions that formed the input dataset were provided by the supervisor, and the large-scale analyses used the King's Computational Research, Engineering and Technology Environment (CREATE). The toolkit builds on public resources including AlphaFold2, STRING, BioGRID, HuRI, hu.MAP 2.0, UniProt, ClinVar, gnomAD/ExAC, EVE, AlphaMissense, ProtVar and the AFDB FoldX data, and Reactome; please cite those resources when using their data.
