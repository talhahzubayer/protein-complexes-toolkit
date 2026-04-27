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
├── complex_resolver.py       # PDB/PKL pair discovery (flat + sharded, .bz2-aware) + forensic manifest
├── file_io.py                # Transparent open() for plain / .gz / .bz2 inputs
├── hpc_dataset_run.sh        # SLURM wrapper for production HPC submission (see HPC Submission)
├── Toolkit_Commands_List.md  # Full CLI command reference (all flags, defaults, examples)
├── requirements.txt          # Python dependencies
├── .gitignore
└── data/                        # External databases (not included in repo)
    ├── complex_manifest_audit/  # Forensic manifest from complex_resolver (auto-generated)
    ├── ppi/                     # PPI databases (see "Setting Up Data")
    ├── clusters/                # STRING sequence clusters (see "Setting Up Data")
    ├── variants/                # Variant databases (see "Setting Up Data")
    ├── stability/               # Stability prediction data (see "Setting Up Data")
    ├── pathways/                # Disease & pathway databases (see "Setting Up Data")
    └── string_api_cache/        # STRING API response cache (auto-generated)
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

The `data/` directory is **not included** in this repository due to the large size of external database files. These files are required for Phase A (Database Ingestion & ID Mapping) and later phases. To set up:

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
| Memory | 64 GB | Run 1 measured MaxRSS 67 GB on a 41,196-complex corpus - bump to **80 GB** for headroom. |
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

### Reference run (sanity baseline)

Job `33556112`, 26 April 2026, host `erc-hpc-comp012`, 41,196 complexes, sharded HPC layout:

| Phase | Elapsed |
|---|---|
| Structural pass (16-worker pool) | 32.6 min |
| ProtVar offline | 10.5 min |
| Disease annotation | 3.7 min |
| Pathway + per-pathway PPI enrichment | **67.9 min (dominant cost)** |
| PyMOL `.pml` generation (12,629 High-tier) | 1.9 min |
| **Total** | **5h 57m** |

Output: `results.csv` (~344 MB, 41,196 rows × 153 cols), `interfaces.jsonl` (~22 MB, 28,203 complexes), 12,629 `.pml` files, and the forensic manifest (`data/complex_manifest_audit/complex_manifest.tsv`, `incomplete_inputs.tsv`).

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

The pipeline produces a 40-column base CSV, progressively expandable to ~153 columns by stacking optional flags (`--enrich`, `--clustering`, `--variants`, `--stability`, `--protvar`, `--disease`, `--pathways`). JSONL interface export is also available. STRING API validation is on by default across all modules; disable with `--no-api`. Each downstream module also provides a standalone CLI. Compressed inputs (`.pdb.bz2`, `.pkl.bz2`) and the sharded HPC layout are supported transparently.

#### Core Analysis

**read_af2_nojax.py** - Loads AlphaFold2 result PKL files without requiring a JAX installation. Extracts ipTM, pTM, pLDDT arrays, and PAE matrices from `.pkl`, `.pkl.gz`, and `.pkl.bz2` formats.

**pdockq.py** - Calculates predicted DockQ scores using the FoldDock sigmoid model. Automatically selects the best interacting chain pair in multi-chain complexes and returns full contact geometry.

**interface_analysis.py** - 2-phase interface characterisation. Phase 1 derives structural geometry from PDB alone (contact count, interface fractions, symmetry, density, interface vs bulk pLDDT). Phase 2 adds PAE-aware confident contact identification, composite confidence scoring, and automated quality flags including paradox detection and metric disagreement.

**toolkit.py** - Batch orchestrator that processes directories of AlphaFold2 predictions with multiprocessing, periodic checkpointing, resume from interruption, and implements 2 quality classification schemes (v1 ipTM/pDockQ gating; v2 composite-informed reclassification). Reads paired PDB/PKL files via `complex_resolver.py` and decompresses `.bz2` inputs in-place via `file_io.py` (no staging mirror). Each optional flag activates a downstream module: `--enrich` (gene symbols, protein names, sequences, database source tagging, species classification), `--clustering` (sequence clusters, homologous pairs), `--variants` (variant mapping and structural context), `--stability` (EVE scores), `--protvar` (AlphaMissense + FoldX), `--disease` (UniProt annotations), `--pathways` (Reactome + network analysis), `--pymol` (PyMOL script generation). `--full-pipeline` activates all phases with default data paths and validates all data dependencies before processing starts.

**visualise_results.py** - Generates up to 16 figures (+ 1b supplementary) with adaptive scatter sizing for large datasets and optional KDE density contour overlays. Figures are generated automatically based on which columns are present in the CSV (e.g., variant figures from `--variants`, pathway figures from `--pathways`). When `species_status` is present, structural figures (1-9) are emitted per species subset (`<n>_<name>_human.png`, `<n>_<name>_nonhuman.png`); enrichment figures use reviewed+TrEMBL (Figs 10-12) or reviewed-only (Figs 13-16) depending on database coverage.

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

**pathway_network.py** - Maps proteins to Reactome pathways and runs per-pathway PPI enrichment via the STRING API (`--pathways`). Builds NetworkX interaction graphs for network topology analysis (degree, centrality). Generates 2 pathway/disease visualisation figures (Figs 14-15).

#### Structural Visualisation

**pymol_scripts.py** - Generates scene-managed PyMOL `.pml` scripts with layered visualisation: chain colouring (10-chain palette, homodimer transparency), pLDDT confidence bands, interface residue sticks, pathogenicity-aware variant spheres coloured by structural context, and AlphaMissense transparency overlay (`--pymol`, `--pymol-min-tier`, `--pymol-render`). Includes metadata and biological annotation comments, pre-computed interface residue lookup to avoid redundant PDB I/O, and a `py3Dmol` fallback for in-notebook rendering. For `.pdb.bz2` inputs the generator emits an inline `bz2.open` + `cmd.read_pdbstr` block because PyMOL's CLI `load` does not transparently decompress.

#### Input Discovery & HPC Submission

**data_registry.py** - Centralises all data-file path references into a single registry of 16 entries, each recording expected path, source module, constant name, and whether the filename contains a version string. Resolves the project root dynamically with the precedence `explicit argument > PROTEIN_TOOLKIT_PROJECT_ROOT env var > repo fallback` so the toolkit (e.g. on HPC at `/scratch/<project>/protein-complexes-toolkit-hpc/`) and its data tree (e.g. at `/scratch/<project>/Protein_Complexes/`) can live in different locations. Provides `validate_data_dependencies()` for pre-run checks used by `--full-pipeline`, and a standalone CLI (`python data_registry.py`) for dependency checking.

**complex_resolver.py** - Discovers paired PDB/PKL inputs across three layouts (loose flat, flat directory-per-complex, sharded directory-per-complex) and writes a forensic manifest of complete pairs plus an audit of incomplete inputs with reason codes (`missing_pdb`, `missing_pkl`, `missing_both`, `empty_pkl`, `duplicate_complex_name`, `ambiguous_pdb`). Layout detection via a `^[A-Z0-9]{2}$` shard regex. Atomic manifest writes (`write .tmp -> Path.replace()`) so the audit file is never half-written. Public API `find_complexes(root, audit_dir=None, write_audit=True)` is consumed by `toolkit.py`'s main pipeline, the standalone forensic CLI (`python complex_resolver.py`), and the `--pymol` script-generator path.

**file_io.py** - Transparent compression-aware open helpers for the eight PDB-reading sites across `toolkit.py`, `pdockq.py`, `variant_mapper.py`, and `pymol_scripts.py`. Three exports: `open_text_maybe_compressed(path)` (text mode with `errors='replace'`), `open_binary_maybe_compressed(path)` (binary mode), and `decompressed_pdb_view(path)` (a context manager that materialises a `.pdb.bz2`/`.pdb.gz` into a per-complex tempfile once, yields the path, and deletes it on exit). The view is entered once per complex in `toolkit.process_single_complex` so the five sequential PDB readers (extract_pLDDT + three CA/CB passes in `read_pdb_with_chain_info_New` + SASA) all hit plain disk text after a single decompression.

**hpc_dataset_run.sh** - Production SLURM wrapper for cluster submission. Owns the entire environment so the run is reproducible across login-node sessions: `module purge && module load python/3.11.6-gcc-13.2.0`, `source .venv/bin/activate`, BLAS thread caps (`OMP_NUM_THREADS=1` + MKL/OpenBLAS/NumExpr - prevents `ProcessPoolExecutor`'s 16 workers from oversubscribing to 256 BLAS threads on 16 cores), `MPLBACKEND=Agg` + `MPLCONFIGDIR` (compute nodes have no DISPLAY and home directories are quota-restricted), `PYTHONUNBUFFERED=1` (real-time SLURM logs) and `PYTHONNOUSERSITE=1` (defensive against stray user-site installs). Runs 5 phases: `[0/4] pip check`, `[1/4] data_registry.py`, `[2/4] complex_resolver.py`, `[3/4] toolkit.py --full-pipeline`, `[4/4] visualise_results.py`. See [HPC Submission](#hpc-submission) for required env vars, resource allocation, and reference performance numbers.


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

### CSV (40 base columns, up to 153 with all features)

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
| **Composite Scoring** | interface_confidence_score, quality_tier, quality_tier_v2 |
| **Audit / Data Availability** | has_pdb, has_pkl, geometry_available (`True` iff pair enumeration succeeded - Decision #34 contract), composite_is_calibrated (`True` only for `tier_scope == "dimer_validated"` - paired with `geometry_available` and `has_pdb`/`has_pkl` as the canonical audit set), plddt_source (`pdb` / `pkl` - diagnostic for which input the pLDDT array was read from) |
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

| # | Figure | Description |
|---|--------|-------------|
| 1 | Quality Scatter | ipTM vs pDockQ coloured by quality tier |
| 1b | Disorder Scatter | Same as Fig 1, coloured by disorder fraction (optional, `--disorder-scatter`) |
| 2 | PAE Health Check | Global PAE distribution histogram |
| 3 | Interface PAE by Tier | [dimer-validated] Boxplot + strip of interface PAE across quality tiers |
| 4 | Composite Tier Validation | Violin + scatter of composite scores by tier |
| 5 | Interface vs Bulk pLDDT | Scatter with diagonal showing interface confidence gain/loss |
| 6 | Paradox Spotlight | Violin triptych of paradox complex metrics |
| 7 | Stoichiometry Architecture | Primary panel [dimer-validated]: `A2` vs `AB`. Supplementary panel (opt-in via `--multimer-supplement`, file `7_supp_Multimer_Stoichiometry*.png`): `A2B` / `ABC` / `A2B2` / `ABCD` / Other. |
| 8 | Metric Disagreement | [dimer-validated] Scatter highlighting complexes with conflicting quality signals |
| 9 | Chain-Count Profile | [all-N descriptive] Four panels: best-pair pDockQ, pdockq_mean, pdockq_min, coherence gap (`pdockq − pdockq_min`) by chain count. Exposes order-statistic bias in best-pair metrics. |
| 10 | Clustering Validation | Homodimer ground truth scatter (shared = total clusters), cluster ratio by quality tier |
| 11 | Classified Variant Sankey | [dimer-validated] Alluvial flow: clinical significance -> structural context. Where do clinically significant variants land structurally? |
| 12 | Variant Density | [dimer-validated] Interface variant density (per residue) vs composite score scatter with Spearman + partial correlation (size-controlled). Does the confidence metric predict variant biology? |
| 13 | Stability Cross-Validation | EVE vs AlphaMissense concordance, AlphaMissense vs FoldX DDG, coverage landscape by tier |
| 14 | Disease Annotation Prevalence | [dimer-validated] Disease prevalence by quality tier (grouped bars + chi-square) + top 10 diseases stacked bars. The drug-target panel keeps a Fisher-test enrichment annotation in its own subtitle. |
| 15 | Pathway Network | [dimer-validated] NetworkX spring layout of top Reactome pathways, coloured by % High-tier complexes |
| 16 | Prediction Quality Paradox | [dimer-validated] 2×2 panel: pathogenic interface variants and PPI density strengthen with quality (top row) while gene constraint and disorder fraction decline (bottom row), revealing systematic AF2-Multimer prediction bias toward ordered protein pairs |

Figures 1-2 are generated from base CSV columns. Figures 3-9 require `--interface --pae` columns. Figure 10 requires clustering columns from `--clustering`. Figures 11-12 require variant columns from `--variants`. Figure 13 requires stability + ProtVar columns from `--stability --protvar`. Figures 14-15 require disease and pathway columns from `--disease --pathways`. Figure 16 requires variant + pathway columns from `--variants --pathways`.

When the CSV contains `species_status`, Figs 1-9 are emitted per species subset (e.g. `1_Quality_Scatter_human.png`, `1_Quality_Scatter_nonhuman.png`). Figs 10-12 use the reviewed+TrEMBL human subset; Figs 13-16 use the reviewed-only subset (the databases behind them cover reviewed human entries best).

## Acknowledgements
Developed by Talhah Zubayer under the supervision of David Burke as part of the MSc Applied Bioinformatics programme at King's College London.
