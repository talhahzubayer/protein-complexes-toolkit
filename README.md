# Protein Complexes Toolkit

A Python toolkit to facilitate the analysis of protein complexes and target drug discovery

MSc Applied Bioinformatics Research Project - King's College London

**Student:** Talhah Zubayer | **Supervisor:** David Burke


## Features

- JAX-free loading of AlphaFold2 result PKL files (no JAX installation required)
- pDockQ scoring using the FoldDock sigmoid parameterisation
- 2-phase interface analysis: structural geometry (Phase 1) and PAE-aware confident contacts (Phase 2)
- 48-column base CSV output (60 with `--enrich`, 67 with `--clustering`, 79 with `--variants`) with automated quality flags, paradox detection, and optional enrichment
- JSONL interface export for downstream analysis
- Batch processing with multiprocessing, checkpointing, and resume from interruption
- Generate up to 13 figures with adaptive rendering for datasets from hundreds to millions of complexes
- Optional KDE density contour overlays and per-complex PAE heatmaps
- PPI database ingestion: parsers for STRING, BioGRID, HuRI, and HuMAP with standardised DataFrame output
- Protein ID cross-referencing: isoform-aware mapping between ENSP, ENSG, UniProt, and gene symbols using STRING aliases
- Database overlap analysis: dual-level (isoform-specific + base-accession) intersection computation with UpSet-style visualisation
- CSV enrichment: gene symbols, protein names, database source tagging, amino acid sequences, and cross-database evidence types
- Centralised STRING API client with rate limiting, retry/backoff, and response caching
- Automatic API validation: ID resolution, enrichment, and database loading fall back to the STRING API when local data is incomplete (disable with `--no-api`)
- Protein sequence clustering: STRING cluster parsing, UniProt-mapped cluster indexing, homologous pair detection, and optional API-based homology scores
- Genetic variant mapping: UniProt/ClinVar/ExAC variant parsing, biotite/BioPython SASA-based 4-class structural context classification (interface core/rim via cross-chain distance, surface, buried core), per-complex variant burden and enrichment analysis
- 592-test suite (570 real + 22 future placeholders) with real PDB/PKL data, offline database excerpts, and mocked API tests


## Repository Structure

```
protein-complexes-toolkit/
├── read_af2_nojax.py        # JAX-free AlphaFold2 PKL reader
├── pdockq.py                # pDockQ score calculator
├── interface_analysis.py    # Interface analysis module
├── toolkit.py               # Batch processing orchestrator
├── visualise_results.py     # Visualisation engine
├── database_loaders.py      # PPI database parsers (STRING, BioGRID, HuRI, HuMAP)
├── id_mapper.py             # Protein ID cross-referencing (ENSP/ENSG/UniProt/gene symbol)
├── overlap_analysis.py      # Database overlap computation and UpSet diagrams
├── string_api.py            # Centralised STRING API client (rate limiting, caching, retry)
├── protein_clustering.py    # Protein sequence clustering and homology detection
├── variant_mapper.py        # Genetic variant mapping and structural context classification
├── pytest.ini               # Pytest configuration
├── requirements.txt         # Python dependencies
├── .gitignore
├── tests/                   # Test suite (570 tests + 22 future placeholders)
│   ├── conftest.py          # Shared fixtures and path config
│   ├── test_read_af2_nojax.py
│   ├── test_pdockq.py
│   ├── test_interface_analysis.py
│   ├── test_toolkit.py
│   ├── test_visualise_results.py
│   ├── test_integration.py
│   ├── test_future_aims.py
│   ├── test_database_loaders.py
│   ├── test_id_mapper.py
│   ├── test_multiprocessing.py
│   ├── test_string_api.py
│   ├── test_protein_clustering.py
│   ├── test_variant_mapper.py
│   └── offline_test_data/
│       └── databases/                           # Small database excerpts for offline testing
│           ├── string_api_responses/            # Pre-captured API responses (7 JSON files)
│           ├── test_aliases.txt                 # STRING aliases excerpt
│           ├── test_biogrid.tab3.txt            # BioGRID excerpt
│           ├── test_humap.pairsWprob            # HuMAP excerpt
│           ├── test_humap_malformed.pairsWprob  # Malformed HuMAP for edge-case tests
│           ├── test_huri.tsv                    # HuRI excerpt
│           ├── test_string_clusters.txt         # STRING clusters excerpt
│           ├── test_string_links.txt            # STRING links excerpt
│           ├── test_uniprot_variants.txt        # UniProt variants excerpt (20 rows)
│           ├── test_clinvar_variants.txt        # ClinVar excerpt (6 rows)
│           └── test_exac_constraint.txt         # ExAC constraint excerpt (5 rows)
├── data/                                 # External databases (not included in repo)
│    ├── ppi/                             # PPI databases (see "Setting Up Data")
│    ├── clusters/                        # STRING sequence clusters (see "Setting Up Data")
│    ├── variants/                        # Variant databases (see "Setting Up Data")
│    └── stability/                       # Stability prediction data (see "Setting Up Data")
│         └── EVE_all_data/               # EVE variant effect scores (3,211 CSVs)
└── Test_Data/							  # Not included in repo (see "Setting Up Test Data")
```


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

**read_af2_nojax.py**: Loads AlphaFold2 result PKL files using module-level JAX mocking, so JAX does not need to be installed. Extracts ipTM, pTM, ranking_confidence, per-residue pLDDT arrays, and PAE matrices. Supports `.pkl`, `.pkl.gz`, and `.pkl.bz2` formats.

**pdockq.py**: Calculates predicted DockQ scores using the FoldDock sigmoid model (L=0.724, x0=152.611, k=0.052, b=0.018). Provides three PDB readers at increasing detail levels. Finds the best interacting chain pair in multi-chain complexes and returns a `ContactResult` dataclass with full contact geometry.

**interface_analysis.py**: 2-phase interface characterisation. Phase 1 (PDB only): contact count, interface fractions, symmetry, density, interface vs bulk pLDDT. Phase 2 (PDB + PKL): PAE mapping with multi-chain offsets, confident contact identification (PAE < 5 Angstrom and pLDDT >= 70), composite confidence scoring, and automated quality flags including paradox detection and metric disagreement.

**toolkit.py**: Batch orchestrator that processes directories of AlphaFold2 predictions using direct module imports. Supports multiprocessing via `ProcessPoolExecutor`, periodic checkpointing (every 50 complexes), and resume from interruption. Produces a 48-column base CSV (60 with `--enrich`, 67 with `--clustering`, 79 with `--variants`) and optional JSONL interface export. Implements 2 quality classification schemes. Optional enrichment adds gene symbols, protein names, database source tagging, amino acid sequences, and cross-database evidence types via `--enrich` and `--databases` flags. Optional clustering adds sequence cluster IDs, shared clusters, and homologous pairs via `--clustering` (requires `--enrich`). Optional variant mapping adds per-chain variant counts, interface variant enrichment, and constraint scores via `--variants` (requires `--interface --pae --enrich`). STRING API validation is on by default during enrichment (disable with `--no-api`).

**visualise_results.py**: Generates up to 13 figures plus supplementary plots and on-demand per-complex PAE heatmaps. Features adaptive scatter sizing for large datasets and optional KDE density contour overlays. Figures 11-13 are generated automatically when variant columns are present in the input CSV.

**database_loaders.py**: Parsers for 4 protein-protein interaction databases. `load_string()` strips `9606.ENSP` prefixes and normalises combined scores from 0 - 1000 to 0.0 - 1.0. `load_biogrid()` filters to human (taxonomy 9606) physical interactions with Swiss-Prot/TrEMBL fallback extraction. `load_huri()` parses binary Y2H interactions with ENSG identifiers. `load_humap()` reads pairwise probability-scored interactions with optional UniProt ID validation. All parsers return standardised DataFrames with columns: `protein_a`, `protein_b`, `source`, `confidence_score`, `evidence_type`. `validate_with_api()` spot-checks loaded IDs against the STRING API (disable with `--no-api`).

**id_mapper.py**: Protein identifier cross-referencing using the STRING aliases file as a single source of truth. `IDMapper` class builds bidirectional lookup tables for ENSP-to-UniProt, UniProt-to-gene-symbol, and ENSG-to-ENSP mappings. `resolve_id()` accepts any identifier type and resolves to a target namespace, with automatic STRING API fallback when local lookup fails (`api_fallback=True` by default; disable with `--no-api`). Isoform-aware: preserves full isoform accessions (e.g., `P22607-2`) and prioritises reviewed Swiss-Prot accessions over TrEMBL. Includes `map_dataframe_to_uniprot()` for batch DataFrame ID conversion, `build_uniprot_lookup()` for efficient enrichment, and `export_lookup_table()` for structured CSV export with primary/secondary accession columns.

**overlap_analysis.py**: Computes pairwise protein interaction overlaps across databases after ID normalisation. `normalise_pair()` and `normalise_pair_base()` create order-independent pair keys at isoform-specific and base-accession levels respectively. `extract_pair_set()` and `extract_pair_set_base()` convert DataFrames to normalised pair sets. `compute_overlaps()` returns per-database counts, pairwise overlaps, triple overlaps, all-database intersections, and unique-to-database sets. Supports UpSet-style intersection visualisation for 4+ databases. CLI supports dual-level analysis (`--base-level`), report generation (`--report`), and STRING threshold comparison.

**string_api.py**: Centralised STRING database API client. All STRING API interactions are routed through this module. Architecture is offline-first: local flat files remain the primary data source; the API is an automatic supplement for unresolved identifiers and validation. Features rate-limited requests (1s between calls), automatic retry with exponential backoff on HTTP 429/5xx, SHA256-keyed response caching (auto-enabled to `data/string_api_cache/`), and caller identity injection per STRING API TOS. Provides 7 public functions: `get_string_ids()`, `get_interaction_partners()`, `query_homology()`, `query_enrichment()`, `query_ppi_enrichment()`, `query_network()`, `get_version()`. Raises `StringAPIError` on failure for clean error propagation.

**protein_clustering.py**: Parses STRING pre-computed protein sequence clusters and maps them to UniProt accessions via `IDMapper`. Defaults to `data/clusters/9606.clusters.proteins.v12.0.txt` when no `--clusters-file` is specified. Builds bidirectional cluster indices (cluster-to-proteins and protein-to-clusters) in both ENSP and UniProt space. `find_shared_clusters()` identifies sequence family relationships between protein pairs. `find_homologous_pairs()` discovers other protein pairs that share the same cluster combinations, with optional filtering against known interaction databases. Clusters exceeding `MAX_CLUSTER_SIZE_FOR_PAIRS` (500) are skipped during pair generation to avoid O(n^2) explosion from STRING's hierarchical mega-clusters (up to 144K UniProt members after isoform expansion). `annotate_results_with_clustering()` adds 7 CSV columns: union and intersection cluster IDs/counts, homologous pairs with counts, and a continuous homology bitscore from the STRING API. `enrich_with_homology_scores()` optionally queries the STRING API for paralogy bitscores using chunked batched deduplication (`HOMOLOGY_API_BATCH_SIZE = 100` proteins per API call; reduced from 500 because the STRING homology endpoint times out at larger batch sizes). `validate_clustering_mode()` accepts `'string'` and raises `NotImplementedError` for deferred `'foldseek'` and `'hybrid'` modes. Standalone CLI supports `--summary`, `--protein`, and `--pair` lookup modes.

**variant_mapper.py**: Maps genetic variants from UniProt, ClinVar, and ExAC databases onto predicted protein complex interface residues. Loads variant databases via chunked streaming (UniProt 33M rows, ClinVar 8.9M rows) for memory efficiency. Computes per-residue solvent-accessible surface area (SASA) using biotite's Cython-accelerated engine as the primary backend (with BioPython ShrakeRupley as fallback) and classifies each variant into one of 4 structural contexts: `interface_core` (<4 Å cross-chain distance from partner chain interface residues), `interface_rim` (4-8 Å cross-chain distance), `surface_non_interface` (RSA ≥ 25%), or `buried_core` (RSA < 25%). `annotate_results_with_variants()` adds 12 CSV columns: per-chain variant counts, interface variant counts, pathogenic interface variant counts, enrichment fold-change, pipe-separated variant detail strings, and per-chain ExAC constraint scores (pLI, mis_z). Standalone CLI supports `summary`, `lookup`, and `map` subcommands. Requires biotite (primary) or BioPython (fallback) for SASA computation.


## Installation

### Prerequisites

- Python 3.13+
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
mkdir -p data/ppi data/clusters data/variants data/stability
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

5. Download EVE variant effect scores into `data/stability/`:

| File | Source | Download |
|------|--------|----------|
| `EVE_all_data/` (3,211 CSVs) | EVE | [evemodel.org/download/bulk](https://evemodel.org/download/bulk) - download "All variant files" CSV archive, extract into `data/stability/` |

> **Note:** The EVE bulk download page offers several archives (MSAs, VCF files, PRC/ROC curves). Only the **variant files** archive is needed - the others are model training inputs or diagnostic plots not used by the pipeline.

6. Verify the directory contents:
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
└── stability/
    └── EVE_all_data/                          # 3,211 per-protein EVE score CSVs (~10 GB)
        ├── 1433G_HUMAN.csv
        ├── 1433Z_HUMAN.csv
        └── ...
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

### Enriched Pipeline (Gene Symbols, Database Sources, Sequences)

```bash
# Full pipeline with enrichment (adds gene symbols, protein names, sequences, species, structure source) - STRING API validation is on by default so unresolved IDs are checked against the API
python toolkit.py --dir /path/to/models --output results.csv --interface --pae --enrich data/ppi/9606.protein.aliases.v12.0.txt -w 4

# Full pipeline with enrichment + database source tagging
python toolkit.py --dir /path/to/models --output results.csv --interface --pae --enrich data/ppi/9606.protein.aliases.v12.0.txt --databases data/ppi/ -w 4

# With protein clustering (requires --enrich; defaults to data/clusters/9606.clusters.proteins.v12.0.txt)
python toolkit.py --dir /path/to/models --output results.csv --interface --pae --enrich data/ppi/9606.protein.aliases.v12.0.txt --clustering

# With clustering + database source tagging (full pipeline)
python toolkit.py --dir /path/to/models --output results.csv --interface --pae --enrich data/ppi/9606.protein.aliases.v12.0.txt --databases data/ppi/ --clustering

# With clustering + custom clusters file (overrides default path)
python toolkit.py --dir /path/to/models --output results.csv --interface --pae --enrich data/ppi/9606.protein.aliases.v12.0.txt --clustering --clusters-file data/clusters/9606.clusters.proteins.v12.0.txt

# Offline-only mode (disable all STRING API calls)
python toolkit.py --dir /path/to/models --output results.csv --interface --pae --enrich data/ppi/9606.protein.aliases.v12.0.txt --no-api
```

### Protein Clustering

```bash
# Cluster summary statistics
python protein_clustering.py --clusters-file data/clusters/9606.clusters.proteins.v12.0.txt --aliases data/ppi/9606.protein.aliases.v12.0.txt --summary

# Look up clusters for a single protein
python protein_clustering.py --clusters-file data/clusters/9606.clusters.proteins.v12.0.txt --aliases data/ppi/9606.protein.aliases.v12.0.txt --protein P04637

# Find shared clusters between two proteins
python protein_clustering.py --clusters-file data/clusters/9606.clusters.proteins.v12.0.txt --aliases data/ppi/9606.protein.aliases.v12.0.txt --pair P04637 Q00987
```

### Variant Mapping

```bash
# With variant mapping (requires --interface --pae --enrich)
python toolkit.py --dir /path/to/models --output results.csv --interface --pae --enrich data/ppi/9606.protein.aliases.v12.0.txt --variants

# With custom variant database directory
python toolkit.py --dir /path/to/models --output results.csv --interface --pae --enrich data/ppi/9606.protein.aliases.v12.0.txt --variants data/variants/

# Without ClinVar enrichment (faster, UniProt + ExAC only)
python toolkit.py --dir /path/to/models --output results.csv --interface --pae --enrich data/ppi/9606.protein.aliases.v12.0.txt --variants --no-clinvar

# Full pipeline: enrichment + databases + clustering + variants
python toolkit.py --dir /path/to/models --output results.csv --interface --pae --enrich data/ppi/9606.protein.aliases.v12.0.txt --databases data/ppi/ --clustering --variants

# Standalone: variant database summary
python variant_mapper.py summary --variants-dir data/variants/

# Standalone: look up variants for a protein
python variant_mapper.py lookup --variants-dir data/variants/ --protein P04637

# Standalone: map variants to interfaces from JSONL export
python variant_mapper.py map --interfaces interfaces.jsonl --pdb-dir /path/to/models --variants-dir data/variants/ --output variant_analysis.csv
```

### Database Ingestion & ID Mapping

```bash
# Load a specific PPI database
python database_loaders.py --database string --data-dir data/ppi/
python database_loaders.py --database biogrid --data-dir data/ppi/
python database_loaders.py --database huri --data-dir data/ppi/
python database_loaders.py --database humap --data-dir data/ppi/

# Load all databases and export to CSV
python database_loaders.py --database all --data-dir data/ppi/ --output all_interactions.csv

# Resolve a protein identifier
python id_mapper.py --aliases data/ppi/9606.protein.aliases.v12.0.txt --resolve ENSP00000269305
python id_mapper.py --aliases data/ppi/9606.protein.aliases.v12.0.txt --resolve P04637

# Print mapping statistics
python id_mapper.py --aliases data/ppi/9606.protein.aliases.v12.0.txt --stats

# Export structured lookup table
python id_mapper.py --aliases data/ppi/9606.protein.aliases.v12.0.txt --export lookup.csv

# Compute database overlaps and generate Venn/UpSet diagram
python overlap_analysis.py --data-dir data/ppi/ --aliases data/ppi/9606.protein.aliases.v12.0.txt --output Output/venn_overlap.png

# Dual-level overlap (isoform-specific + base-accession) with report
python overlap_analysis.py --data-dir data/ppi/ --aliases data/ppi/9606.protein.aliases.v12.0.txt --output Output/venn_overlap.png --base-level --report Output/overlap_report.txt
```

### STRING API

```bash
# Resolve identifiers to STRING IDs
python string_api.py --resolve P04637,Q9UKT4 --species 9606

# Check STRING database version
python string_api.py --version

# Functional enrichment analysis
python string_api.py --enrichment P04637,P38398,Q9UKT4 --species 9606

# Retrieve interaction partners
python string_api.py --interaction-partners P04637 --species 9606

# Query protein network
python string_api.py --network TP53,MDM2,BRCA1 --network-type physical

# Use a custom cache directory
python string_api.py --resolve P04637 --cache-dir /tmp/my_cache
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

# Database loading and ID mapping tests
python -m pytest tests/ -m "database" -v

# STRING API tests (mocked, no network)
python -m pytest tests/ -m "api" -v

# Clustering tests
python -m pytest tests/ -m "clustering" -v

# Variant mapping tests
python -m pytest tests/ -m "variants" -v

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

### CSV (48 base columns, 60 with enrichment, 67 with clustering, 79 with variants)

The main output CSV groups columns into:

| Category | Key Columns |
|----------|-------------|
| **Identity** | complex_name, protein_a, protein_b, complex_type, n_chains, species, structure_source |
| **Core Metrics** | ipTM, pTM, ranking_confidence, pDockQ, ppv |
| **pLDDT Statistics** | plddt_mean, plddt_median, plddt_min, plddt_max, plddt_below50/70_fraction |
| **Interface Geometry** | n_interface_contacts, interface_fraction_a/b, interface_symmetry, contacts_per_interface_residue |
| **Interface pLDDT** | interface_plddt_combined, bulk_plddt_combined, interface_vs_bulk_delta |
| **PAE Features** | interface_pae_mean, n_confident_contacts, confident_contact_fraction, cross_chain_pae_mean |
| **Composite Scoring** | interface_confidence_score, quality_tier, quality_tier_v2 |
| **Flags** | interface_flags (8 automated flags including paradox detection) |
| **Enrichment** (with `--enrich`) | gene_symbol_a/b, protein_name_a/b, ensembl_id_a/b, secondary_accessions_a/b, database_source, evidence_types, sequence_a/b |
| **Clustering** (with `--clustering`) | sequence_cluster_ids, sequence_cluster_count, shared_cluster_ids, shared_cluster_count, homologous_pairs, n_homologous_pairs, homology_bitscore |
| **Variants** (with `--variants`) | n_variants_a/b, n_interface_variants_a/b, n_pathogenic_interface_variants, interface_variant_enrichment, variant_details_a/b, gene_constraint_pli_a/b, gene_constraint_mis_z_a/b |

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
| 11 | Classified Variant Sankey | Alluvial flow: clinical significance -> structural context (Unknown excluded). Where do clinically significant variants land structurally? |
| 12 | Enrichment Distribution | Histogram + KDE of interface variant enrichment by quality tier with Wilcoxon signed-rank test vs 1.0 (neutral). Are predicted interfaces under evolutionary constraint? |
| 13 | Variant Density vs Quality | Interface variant density (per residue) vs composite score scatter with Spearman + partial correlation (size-controlled). Does the confidence metric predict variant biology? |

Figures 1-2 are generated from base CSV columns. Figures 3-9 require `--interface --pae` columns. Figure 10 requires the `n_chains` column. Figures 11-13 require variant columns from `--variants`.


## Roadmap

### Completed

- **Aim 5 - Structure Prediction Quality Assessment:** JAX-free PKL extraction, pDockQ scoring, 2-phase interface analysis, 46-column CSV, 10-figure visualisation suite
- **Aim 1 - Database Ingestion:** Parsers for STRING, BioGRID, HuRI, and HuMAP with standardised DataFrame output
- **Aim 2 - ID Cross-Referencing:** Isoform-aware mapping pipeline using STRING aliases (ENSP/ENSG/UniProt/gene symbol) with dual-level cross-database overlap analysis, structured lookup table export, and toolkit CSV enrichment
- **STRING API Integration:** Centralised API client (`string_api.py`) with automatic validation fallback across ID resolution, enrichment, and database loading - on by default with `--no-api` opt-out
- **Aim 3 - Protein Clustering:** STRING sequence cluster parsing, UniProt-mapped indexing, homologous pair detection, optional API homology scores, toolkit integration with `--clustering` flag (Foldseek/hybrid modes deferred)
- **Aim 4 - Variant Mapping:** UniProt/ClinVar/ExAC variant parsing, biotite/BioPython SASA-based 4-class structural context classification (interface core/rim via cross-chain distance, surface, buried core), per-complex variant burden and enrichment analysis, 3 variant visualisation figures (Figs 11-13), toolkit integration with `--variants` and `--no-clinvar` flags

### Planned
- **Aim 6 - Stability Scoring:** Integrate ProtVar, EVE scores, and FoldX for predicted structural impact
- **Disease & Pathway Integration:** Map complexes to KEGG/Reactome pathways, build interaction networks
- **PyMOL Visualisation:** Generate `.pml` scripts for batch structure rendering with interface highlighting
- **Million-Complex Production Run:** Full pipeline validation on large-scale AlphaFold-Multimer dataset


## Testing

The test suite contains **592 tests** across 15 modules (570 real + 22 future placeholders):

| Module | Tests | Scope |
|--------|-------|-------|
| test_read_af2_nojax.py | 26 | PKL loading, metric extraction |
| test_pdockq.py | 39 | PDB parsing, pDockQ calculation, multi-chain |
| test_interface_analysis.py | 39 | Interface geometry, pLDDT, PAE, composite |
| test_toolkit.py | 54 | File discovery, quality classification, CSV, enrichment, sequences |
| test_visualise_results.py | 36 | Figure generation, variant detail parsing, Figs 11-13, data loading, CLI |
| test_integration.py | 8 | Cross-module pipeline, data flow |
| test_database_loaders.py | 70 | STRING/BioGRID/HuRI/HuMAP parsing, edge cases, cross-DB overlap, base-level overlap |
| test_id_mapper.py | 65 | ID validation, mapping, isoform handling, secondary accessions, lookup builder |
| test_multiprocessing.py | 6 | Pickling, subprocess import, parallel parity |
| test_string_api.py | 56 | STRING API client, caching, rate limiting, retry, API fallback integration, database validation |
| test_protein_clustering.py | 55 | Protein clustering, homology detection, oversized cluster handling, CLI |
| test_variant_mapper.py | 103 | HGVS parsing, variant loading, SASA, parallel SASA (incl. combined both-chains), structural context (cross-chain distance), enrichment, CLI, toolkit integration |
| test_future_aims.py | 7 + 22 | 7 real database tests + 22 future placeholders |

**Results:** 569 passing, 1 skipped (Fig 10 - all test complexes are dimers), 22 future placeholders (deselected by default)

**Markers:** `slow` (file I/O), `regression` (exact numerical values), `integration` (cross-module), `cli` (command-line), `database` (PPI database loading and ID mapping), `multiprocessing` (parallel processing), `api` (STRING API, mocked), `clustering` (protein clustering and homology), `variants` (variant mapping and structural context), `future` (unimplemented features)


## Acknowledgements
Developed by Talhah Zubayer under the supervision of David Burke as part of the MSc Applied Bioinformatics programme at King's College London.
