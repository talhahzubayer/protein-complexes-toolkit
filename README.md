# Protein Complexes Toolkit

A command-line Python toolkit for interface-aware assessment of AlphaFold-Multimer protein-complex predictions.

*MSc Applied Bioinformatics Research Project, King's College London.*
**Student:** Talhah Zubayer · **Supervisor:** Dr David Burke

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
├── Docs/                      # Full toolkit command list and output schema
│   ├── Toolkit_Commands_List.md
│   └── OUTPUT_SCHEMA.md
└── data/                      # External databases (not included; see below)
```

The external datasets under `data/` are not included in the repository.

## Required inputs

Some inputs must be supplied separately:

- The AlphaFold-Multimer prediction dataset is **not included**. The toolkit assesses predictions but does not generate them.
- The final dissertation results CSV (`results_516744.csv`) is **not included**; it is stored on King's College London's CREATE cluster.
- The external annotation databases under `data/` are **not included** and must be downloaded (see [Setting up external data](#setting-up-external-data)).
- Each complex must be supplied as a paired **PDB** structure file and **PKL** confidence file.

## Installation

Requires **Python 3.11 or newer** (the dissertation analyses ran under Python 3.11.6).

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

JAX is not required. Once the external data is in place, validate it:

```bash
python data_registry.py
```

This reports any missing registered data file before a run starts. `render_complex_summary.py` additionally needs a headless PyMOL executable (see [Generate structural summaries](#generate-structural-summaries)).

## Input prediction layouts

Each complex is a paired PDB and PKL file. `toolkit.py --dir <MODELS_DIR>` accepts three layouts:

**Loose flat** — files directly in the root, complex names parsed from filenames:

```text
Protein_Complexes/
├── A0A0A0MQZ0_P40933.pdb
└── A0A0A0MQZ0_P40933.pkl
```

**Directory per complex** — each child directory holds one complex:

```text
Protein_Complexes/
└── A0A0A0MQZ0_P40933/
    ├── A0A0A0MQZ0_P40933.pdb
    └── A0A0A0MQZ0_P40933.pkl
```

**Sharded directory per complex** — a two-character shard prefix groups complexes (the HPC layout):

```text
Protein_Complexes/
└── A0/
    └── A0A0A0MQZ0_P40933/
        ├── A0A0A0MQZ0_P40933.pdb.bz2
        └── A0A0A0MQZ0_P40933.pkl.bz2
```

Accepted suffixes are `.pdb`, `.pdb.bz2` and `.pdb.gz` for structures and `.pkl`, `.pkl.bz2`, `.pkl.gz`, `.results.pkl` and `.results.pkl.bz2` for confidence files; the long-form `*_relaxed_model_*` and `*_result_model_*` names are also matched. The directory-based layouts recognise the uncompressed and `.bz2` forms; `.gz` is recognised only in the loose layout.

Audit a prediction directory before processing:

```bash
python complex_resolver.py --root <MODELS_DIR>
```

This writes a manifest of complete pairs and reports missing, empty or ambiguous inputs. The manifest format and reason codes are documented in [`Docs/Toolkit_Commands_List.md`](Docs/Toolkit_Commands_List.md).

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

## Running the dissertation workflow

### Validate external data

```bash
python data_registry.py
```

Run the registry check before processing so that missing external files are reported before the analysis starts.

### Audit prediction inputs

```bash
python complex_resolver.py --root <MODELS_DIR>
```

Writes the input manifest and reports incomplete PDB/PKL pairs.

### Run the full pipeline locally

```bash
python toolkit.py --full-pipeline --dir <MODELS_DIR> -w 8 \
    --output results.csv \
    --export-interfaces interfaces.jsonl
```

`--full-pipeline` enables the complete production workflow. Selective flags, dependencies and standalone commands are documented in [`Docs/Toolkit_Commands_List.md`](Docs/Toolkit_Commands_List.md).

### Run the initial HPC workflow

```bash
sbatch hpc_dataset_run.sh
```

Before submitting, edit the two pinned paths near the top of `hpc_dataset_run.sh` (`PROTEIN_TOOLKIT_PROJECT_ROOT` and `PROTEIN_COMPLEXES_ROOT`) for your cluster. The wrapper validates the dependencies and inputs, runs the full pipeline, and generates the figures. Adjust the `#SBATCH` resource requests in the script for the target cluster.

### Run an incremental workflow

```bash
sbatch --export=ALL,HISTORICAL_RESULTS_CSV=results.csv \
    hpc_incremental_run.sh
```

Processes only complexes absent from the historical CSV, writing separate incremental CSV and JSONL outputs; it does not modify the historical files.

### Process a limited batch

```bash
sbatch --export=ALL,HISTORICAL_RESULTS_CSV=results.csv,LIMIT=100000 \
    hpc_incremental_run.sh
```

Run one batch, verify that it completed, consolidate it into the cumulative CSV, then submit the next batch against the updated file.

### Resume an interrupted batch

```bash
sbatch --export=ALL,RESUME=1,\
HISTORICAL_RESULTS_CSV=results.csv,\
OUTPUT_CSV=results_incremental_<stamp>_<job>.csv,\
INTERFACES_JSONL=interfaces_incremental_<stamp>_<job>.jsonl \
    hpc_incremental_run.sh
```

Resume against the same partial output files and the unchanged historical CSV used by the interrupted run.

### Consolidate incremental outputs

Incremental runs do not modify the historical CSV or JSONL. Merge each completed output into the cumulative files before starting the next batch. The repository does not currently include an automated merge helper. The complete validation procedure is documented in [`Docs/Toolkit_Commands_List.md`](Docs/Toolkit_Commands_List.md).

### Generate population figures

```bash
python visualise_results.py results.csv \
    --output-dir Output \
    --full-figure-pack \
    --human-supplement
```

Run this on the final consolidated CSV; figures are written to `Output/`. The filename-to-dissertation-figure mapping is in [`Docs/Toolkit_Commands_List.md`](Docs/Toolkit_Commands_List.md).

### Generate structural summaries

```bash
python render_complex_summary.py \
    --input-dir <COMPLEX_DIR> \
    --output complex_summary.png \
    --pymol-executable "$HOME/envs/pymol/bin/pymol"
```

Generates one PNG for one calibrated dimer, and requires a headless PyMOL executable. The dissertation's two structural examples were rendered separately and combined afterwards. Where no PyMOL module is available, install one once:

```bash
conda create -y -p "$HOME/envs/pymol" -c conda-forge pymol-open-source
```

Full options are documented in [`Docs/Toolkit_Commands_List.md`](Docs/Toolkit_Commands_List.md).

## Outputs

| Output | Created by |
| ------ | ---------- |
| Results CSV | `toolkit.py` |
| Interface JSONL | `--export-interfaces` |
| PyMOL scripts | `--pymol` |
| Population figures | `visualise_results.py` |
| Single-complex summary PNG | `render_complex_summary.py` |
| Audit manifests | `complex_resolver.py` |

Output locations, tier filtering and other details are documented in [`Docs/Toolkit_Commands_List.md`](Docs/Toolkit_Commands_List.md).

## Output CSV

The base pipeline emits 41 columns. Optional stages append their own field blocks, producing 155 columns under `--full-pipeline`.

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

The complete field definitions, availability rules, serialisation formats and dissertation population filters are documented in [`Docs/OUTPUT_SCHEMA.md`](Docs/OUTPUT_SCHEMA.md).
