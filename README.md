# Protein Complexes Toolkit

A Python toolkit to facilitate the analysis of protein complexes and target drug discovery

MSc Applied Bioinformatics Research Project - King's College London

**Student:** Talhah Zubayer | **Supervisor:** David Burke


## Features

- JAX-free loading of AlphaFold2 result PKL files (no JAX installation required)
- pDockQ scoring using the FoldDock sigmoid parameterisation
- 2-phase interface analysis: structural geometry (Phase 1) and PAE-aware confident contacts (Phase 2)
- 46-column CSV output with automated quality flags and paradox detection
- JSONL interface export for downstream analysis
- Batch processing with multiprocessing, checkpointing, and resume from interruption
- Generate 10 figures with adaptive rendering for datasets from hundreds to millions of complexes
- Optional KDE density contour overlays and per-complex PAE heatmaps
- 177-test regression suite to handle real PDB/PKL data


## Repository Structure

```
protein-complexes-toolkit/
├── read_af2_nojax.py        # JAX-free AlphaFold2 PKL reader
├── pdockq.py                # pDockQ score calculator
├── interface_analysis.py    # Interface analysis module
├── toolkit.py               # Batch processing orchestrator
├── visualise_results.py     # Visualisation engine
├── pytest.ini               # Pytest configuration
├── requirements.txt         # Python dependencies
├── .gitignore
├── tests/                   # Test suite (177 tests)
│   ├── conftest.py          # Shared fixtures and path config
│   ├── test_read_af2_nojax.py
│   ├── test_pdockq.py
│   ├── test_interface_analysis.py
│   ├── test_toolkit.py
│   ├── test_visualise_results.py
│   ├── test_integration.py
│   └── test_future_aims.py
├── data/                    # External databases (not included in repo)
│   └── ppi/                 # Protein-protein interaction databases (see "Setting Up PPI Databases")
└── Test_Data/               # Not included in repo (see "Setting Up Test Data")
```


## Pipeline Architecture

```
PDB + PKL files
       │
       ▼
read_af2_nojax.py ──▶ pdockq.py ──▶ interface_analysis.py ──▶ toolkit.py ──▶ visualise_results.py
  (PKL metrics)     (pDockQ/PPV)    (interface geometry     (batch CSV output)  (generates figures)
                                     + PAE features)            
```

### Script Descriptions

**read_af2_nojax.py**: Loads AlphaFold2 result PKL files using module-level JAX mocking, so JAX does not need to be installed. Extracts ipTM, pTM, ranking_confidence, per-residue pLDDT arrays, and PAE matrices. Supports `.pkl`, `.pkl.gz`, and `.pkl.bz2` formats.

**pdockq.py**: Calculates predicted DockQ scores using the FoldDock sigmoid model (L=0.724, x0=152.611, k=0.052, b=0.018). Provides three PDB readers at increasing detail levels. Finds the best interacting chain pair in multi-chain complexes and returns a `ContactResult` dataclass with full contact geometry.

**interface_analysis.py**: 2-phase interface characterisation. Phase 1 (PDB only): contact count, interface fractions, symmetry, density, interface vs bulk pLDDT. Phase 2 (PDB + PKL): PAE mapping with multi-chain offsets, confident contact identification (PAE < 5 Angstrom and pLDDT >= 70), composite confidence scoring, and automated quality flags including paradox detection and metric disagreement.

**toolkit.py**: Batch orchestrator that processes directories of AlphaFold2 predictions using direct module imports. Supports multiprocessing via `ProcessPoolExecutor`, periodic checkpointing (every 50 complexes), and resume from interruption. Produces a 46-column CSV and optional JSONL interface export. Implements 2 quality classification schemes.

**visualise_results.py**: Generates up to 10 figures plus supplementary plots and on-demand per-complex PAE heatmaps. Features adaptive scatter sizing for large datasets and optional KDE density contour overlays.



## Installation

### Prerequisites

- Python 3.13+
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** JAX is **not** required. The toolkit uses module-level mocking to read AlphaFold2 PKL files without a JAX installation.


## Setting Up PPI Databases

The `data/` directory is **not included** in this repository due to the large size of external database files. These files are required for Phase A (Database Ingestion & ID Mapping) and later phases. To set up:

1. Create the directory structure:
```bash
mkdir -p data/ppi
```

2. Download the following files into `data/ppi/`:

| File | Source | Download |
|------|--------|----------|
| `9606.protein.links.v12.0.txt` | STRING | [string-db.org/cgi/download](https://string-db.org/cgi/download?sessionId=bqpmZGj7RlXV&species_text=Homo+sapiens) - select *Homo sapiens*, download `9606.protein.links.v12.0.txt.gz`, decompress |
| `9606.protein.aliases.v12.0.txt` | STRING | Same page - download `9606.protein.aliases.v12.0.txt.gz`, decompress |
| `BIOGRID-ALL-5.0.253.tab3.txt` | BioGRID | [downloads.thebiogrid.org](https://downloads.thebiogrid.org/File/BioGRID/Release-Archive/BIOGRID-5.0.253/BIOGRID-ALL-5.0.253.tab3.zip) - extract the `.tab3.txt` file from the zip |
| `HuRI.tsv` | HuRI | [interactome-atlas.org/download](https://interactome-atlas.org/download) - download `HuRI.tsv` |
| `humap2_ppis_ACC_20200821.pairsWprob` | HuMAP 2.0 | [humap2.proteincomplexes.org/download](https://humap2.proteincomplexes.org/download) - download "Protein Interaction Network with probability scores (Uniprot gzip)", decompress |

3. Verify the directory contents:
```
data/ppi/
├── 9606.protein.links.v12.0.txt          (~616 MB)
├── 9606.protein.aliases.v12.0.txt        (~195 MB)
├── BIOGRID-ALL-5.0.253.tab3.txt          (~1.48 GB)
├── HuRI.tsv                              (~1.6 MB)
└── humap2_ppis_ACC_20200821.pairsWprob   (~439 MB)
```


## Setting Up Test Data

The `Test_Data/` directory is **not included** in this repository due to its large size. To run the test suite on your local machine, you need to:

1. Create a `Test_Data/` folder at the project root:

```bash
mkdir Test_Data
```

2. Place your own AlphaFold2-Multimer prediction files inside it. Each complex requires a **paired PDB structure file and PKL result file**. The toolkit supports two naming conventions:

**Old naming convention:**
```
A0A0B4J2C3_P24534.pdb
A0A0B4J2C3_P24534.results.pkl
```

**New naming convention:**
```
A0A0A0MQZ0_P40933_relaxed_model_1_multimer_v3_pred_0.pdb
A0A0A0MQZ0_P40933_result_model_1_multimer_v3_pred_0.pkl
```

3. **Update `tests/conftest.py`:** Open `tests/conftest.py` and update the `PROJECT_ROOT` path on **line 19** to match the location of your local clone:

```python
# Line 19 - change this to your local project path
PROJECT_ROOT = Path(r"C:\your\path\to\protein-complexes-toolkit")
```

4. Verify the tests work:

```bash
python -m pytest tests/ -m "not future" -v
```


## Usage

### Quick Start

Process a directory of AlphaFold2 predictions and generate a results CSV:

```bash
python toolkit.py --dir /path/to/models --output results.csv
```

### Full Pipeline

```bash
# Full analysis with interface metrics, PAE features, 4 workers, and checkpointing
python toolkit.py --dir /path/to/models --output results.csv --interface --pae -w 4 --checkpoint

# Full analysis with JSONL interface export
python toolkit.py --dir /path/to/models --output results.csv --interface --pae --export-interfaces interfaces.jsonl -w 4 --checkpoint

# Resume an interrupted run
python toolkit.py --dir /path/to/models --output results.csv --interface --pae -w 4 --resume
```

### Visualisation

```bash
# Generate all figures
python visualise_results.py results.csv --output-dir ./Output

# With KDE density contours
python visualise_results.py results.csv --output-dir ./Output --density

# With disorder-coloured scatter (Fig 1b)
python visualise_results.py results.csv --output-dir ./Output --disorder-scatter

# Per-complex PAE heatmaps (limit to top 50)
python visualise_results.py results.csv --pae-heatmaps /path/to/models --limit 50
```

### Individual Scripts

```bash
# PKL reader
python read_af2_nojax.py --pkl result.pkl
python read_af2_nojax.py --pkl result.pkl --json metrics.json
python read_af2_nojax.py --pkl result.pkl --keys

# pDockQ calculator
python pdockq.py --pdbfile structure.pdb

# Interface analysis (Phase 1 only)
python interface_analysis.py --pdb structure.pdb --json output.json

# Interface analysis (Phase 1 + Phase 2)
python interface_analysis.py --pdb structure.pdb --pkl result.pkl --json output.json
```

### Running Tests

```bash
# Full test suite (excludes future placeholders)
python -m pytest tests/ -m "not future" -v

# Quick tests only (no file I/O)
python -m pytest tests/ -m "not slow and not future" -v

# Regression tests only
python -m pytest tests/ -m "regression" -v

# Integration tests
python -m pytest tests/ -m "integration" -v

# View future feature placeholders
python -m pytest tests/ -m "future" -v -o "addopts="
```


## Input Data Format

The toolkit expects a directory containing paired AlphaFold2-Multimer output files:

```
models/
├── ProteinA_ProteinB.pdb                                          # Old naming
├── ProteinA_ProteinB.results.pkl                                  # Old naming
├── ProteinC_ProteinD_relaxed_model_1_multimer_v3_pred_0.pdb       # New naming
├── ProteinC_ProteinD_result_model_1_multimer_v3_pred_0.pkl        # New naming
└── ...
```

Each complex requires:
- A **PDB file** containing the predicted structure with ATOM records
- A **PKL file** containing the AlphaFold2 result dictionary (ipTM, pTM, pLDDT, PAE)

Both old (`X_Y.pdb` / `X_Y.results.pkl`) and new (`X_Y_relaxed_model_*.pdb` / `X_Y_result_model_*.pkl`) naming conventions are supported. The toolkit also handles homodimer, isoform, and multi-chain naming patterns.


## Output

### CSV (46 columns)

The main output CSV groups columns into:

| Category | Key Columns |
|----------|-------------|
| **Identity** | complex_name, protein_a, protein_b, complex_type, n_chains |
| **Core Metrics** | ipTM, pTM, ranking_confidence, pDockQ, ppv |
| **pLDDT Statistics** | plddt_mean, plddt_median, plddt_min, plddt_max, plddt_below50/70_fraction |
| **Interface Geometry** | n_interface_contacts, interface_fraction_a/b, interface_symmetry, contacts_per_interface_residue |
| **Interface pLDDT** | interface_plddt_combined, bulk_plddt_combined, interface_vs_bulk_delta |
| **PAE Features** | interface_pae_mean, n_confident_contacts, confident_contact_fraction, cross_chain_pae_mean |
| **Composite Scoring** | interface_confidence_score, quality_tier, quality_tier_v2 |
| **Flags** | interface_flags (8 automated flags including paradox detection) |

### JSONL Interface Export

When `--export-interfaces` is used, one JSON record per complex is written, containing confident interface residue sets, PAE values, and per-residue pLDDT for downstream analysis.


## Figures Generated

| # | Figure | Description |
|---|--------|-------------|
| 1 | Quality Scatter | ipTM vs pDockQ coloured by quality tier |
| 1b | Disorder Scatter | Same as Fig 1, coloured by disorder fraction (optional, `--disorder-scatter`) |
| 2 | PAE Health Check | Global PAE distribution histogram |
| 3 | Interface PAE by Tier | Boxplot + strip of interface PAE across quality tiers |
| 4 | Composite Tier Validation | Violin + scatter of composite scores by tier |
| 5 | Interface vs Bulk pLDDT | Scatter with diagonal showing interface confidence gain/loss |
| 6 | Paradox Spotlight | Violin triptych of paradox complex metrics |
| 7 | Homo vs Hetero | Architecture comparison of homodimers and heterodimers |
| 8 | Metric Disagreement | Scatter highlighting complexes with conflicting quality signals |
| 9 | Correlation & Flags | Metric correlation heatmap with flag landscape |
| 10 | Chain-Count Profile | Violin + scatter of quality by chain count |

Figures 1-2 are generated from base CSV columns. Figures 3-9 require `--interface --pae` columns. Figure 10 requires the `n_chains` column.


## Roadmap

The following features are planned for future development:

- **Aim 1 - Database Ingestion:** Download and parse interaction databases (STRING, BioGRID, HuRI, HuMAP)
- **Aim 2 - ID Cross-Referencing:** Build mapping pipeline from Ensembl to UniProt to gene symbol to RefSeq
- **Aim 3 - Protein Clustering:** Integrate STRING sequence clusters and Foldseek structural similarity
- **Aim 4 - Variant Mapping:** Map ClinVar and gnomAD variants onto interface residues with structural context classification
- **Aim 6 - Stability Scoring:** Integrate ProtVar, EVE scores, and FoldX for predicted structural impact
- **Disease & Pathway Integration:** Map complexes to KEGG/Reactome pathways, build interaction networks
- **PyMOL Visualisation:** Generate `.pml` scripts for batch structure rendering with interface highlighting
- **Million-Complex Production Run:** Full pipeline validation on large-scale AlphaFold-Multimer dataset


## Testing

The test suite contains **177 tests** across 8 modules:

| Module | Tests | Scope |
|--------|-------|-------|
| test_read_af2_nojax.py | 21 | PKL loading, metric extraction |
| test_pdockq.py | 32 | PDB parsing, pDockQ calculation, multi-chain |
| test_interface_analysis.py | 35 | Interface geometry, pLDDT, PAE, composite |
| test_toolkit.py | 24 | File discovery, quality classification, CSV |
| test_visualise_results.py | 22 | Figure generation, data loading, CLI |
| test_integration.py | 8 | Cross-module pipeline, data flow |
| test_future_aims.py | 35 | Placeholders for future features |

**Results:** 170 passing, 1 skipped (Fig 10 - requires multi-chain data), 35 future placeholders (deselected by default)

**Markers:** `slow` (file I/O), `regression` (exact numerical values), `integration` (cross-module), `cli` (command-line), `future` (unimplemented features)


## Acknowledgements
Developed by Talhah Zubayer under the supervision of David Burke as part of the MSc Applied Bioinformatics programme at King's College London.
