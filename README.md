# Protein Complexes Toolkit

A Python toolkit to facilitate the analysis of protein complexes and target drug discovery

MSc Applied Bioinformatics Research Project - King's College London

**Student:** Talhah Zubayer | **Supervisor:** David Burke


## Features

- JAX-free loading of AlphaFold2 result PKL files (no JAX installation required)
- pDockQ scoring using the FoldDock sigmoid parameterisation
- 2-phase interface analysis: structural geometry (Phase 1) and PAE-aware confident contacts (Phase 2)
- 25-column base CSV output (up to 119 with all features: `--enrich`, `--clustering`, `--variants`, `--stability`, `--protvar`, `--disease`, `--pathways`) with automated quality flags, paradox detection, and optional enrichment
- JSONL interface export for downstream analysis
- Batch processing with multiprocessing, checkpointing, and resume from interruption
- Generate up to 18 figures with adaptive rendering for datasets from hundreds to millions of complexes
- Optional KDE density contour overlays and per-complex PAE heatmaps
- PPI database ingestion: parsers for STRING, BioGRID, HuRI, and HuMAP with standardised DataFrame output
- Protein ID cross-referencing: isoform-aware mapping between ENSP, ENSG, UniProt, and gene symbols using STRING aliases
- Database overlap analysis: dual-level (isoform-specific + base-accession) intersection computation with UpSet-style visualisation
- CSV enrichment: gene symbols, protein names, database source tagging, amino acid sequences, and cross-database evidence types
- Centralised STRING API client with rate limiting, retry/backoff, and response caching
- Automatic API validation: ID resolution, enrichment, and database loading fall back to the STRING API when local data is incomplete (disable with `--no-api`)
- Protein sequence clustering: STRING cluster parsing, UniProt-mapped cluster indexing, homologous pair detection, and optional API-based homology scores
- Genetic variant mapping: UniProt/ClinVar/ExAC variant parsing, biotite/BioPython SASA-based 4-class structural context classification (interface core/rim via cross-chain distance, surface, buried core), per-complex variant burden and enrichment analysis
- EVE stability scoring: evolutionary variant effect predictions mapped to pipeline variants via UniProt ID mapping, with per-chain EVE score summaries and pathogenicity classification
- Offline AlphaMissense + monomeric FoldX scoring: pre-computed AlphaMissense pathogenicity scores (216M variants) and AFDB FoldX DDG values (209M substitutions) loaded from local files for instant variant scoring with no API dependency
- UniProt disease annotations: offline XML parsing of disease associations, PTM sites, GO terms, and drug target status with API fallback for missing proteins
- Reactome pathway mapping: per-pathway PPI enrichment via STRING API, NetworkX network analysis, and 3 pathway/disease visualisation figures (Figs 14-16)
- PyMOL script generation: layered `.pml` command files with chain colouring, pLDDT confidence bands, interface residue highlighting, and variant position colouring by structural context; `py3Dmol` fallback for in-notebook rendering
- 1002-test suite (984 passing + 1 skipped + 15 future placeholders) with real PDB/PKL data, offline database excerpts, and mocked API tests


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
├── stability_scorer.py      # EVE stability scoring and variant effect predictions
├── protvar_client.py        # Offline AlphaMissense + monomeric FoldX scorer (local data files)
├── disease_annotations.py   # UniProt disease/PTM/GO/drug-target annotation
├── pathway_network.py       # Reactome pathway mapping, PPI enrichment, NetworkX networks
├── pymol_scripts.py         # PyMOL .pml script generation and py3Dmol fallback
├── pytest.ini               # Pytest configuration
├── requirements.txt         # Python dependencies
├── .gitignore
├── tests/                   # Test suite (985 tests + 15 future placeholders)
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
│   ├── test_stability_scorer.py
│   ├── test_protvar_client.py
│   ├── test_disease_annotations.py
│   ├── test_pathway_network.py
│   ├── test_pymol_scripts.py
│   └── offline_test_data/
│       └── databases/                            # Small database excerpts for offline testing
├── data/                                         # External databases (not included in repo)
│    ├── ppi/                                     # PPI databases (see "Setting Up Data")
│    ├── clusters/                                # STRING sequence clusters (see "Setting Up Data")
│    ├── variants/                                # Variant databases (see "Setting Up Data")
│    ├── stability/                               # Stability prediction data (see "Setting Up Data")
│    ├── pathways/                                # Disease & pathway databases (see "Setting Up Data")
│    └── string_api_cache/                        # STRING API response cache (auto-generated)
└── Test_Data/							          # Not included in repo (see "Setting Up Test Data")
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

**read_af2_nojax.py**: Loads AlphaFold2 result PKL files using module-level JAX mocking, so JAX does not need to be installed. Extracts ipTM, pTM, ranking_confidence, per-residue pLDDT arrays, and PAE matrices. Supports `.pkl`, `.pkl.gz`, and `.pkl.bz2` formats.

**pdockq.py**: Calculates predicted DockQ scores using the FoldDock sigmoid model (L=0.724, x0=152.611, k=0.052, b=0.018). Provides three PDB readers at increasing detail levels. Finds the best interacting chain pair in multi-chain complexes and returns a `ContactResult` dataclass with full contact geometry.

**interface_analysis.py**: 2-phase interface characterisation. Phase 1 (PDB only): contact count, interface fractions, symmetry, density, interface vs bulk pLDDT. Phase 2 (PDB + PKL): PAE mapping with multi-chain offsets, confident contact identification (PAE < 5 Angstrom and pLDDT >= 70), composite confidence scoring, and automated quality flags including paradox detection and metric disagreement.

**toolkit.py**: Batch orchestrator that processes directories of AlphaFold2 predictions using direct module imports. Supports multiprocessing via `ProcessPoolExecutor`, periodic checkpointing (every 50 complexes), and resume from interruption. Produces a 25-column base CSV (up to 119 with all features) and optional JSONL interface export. Implements 2 quality classification schemes. Optional enrichment adds gene symbols, protein names, database source tagging, amino acid sequences, and cross-database evidence types via `--enrich` and `--databases` flags. Optional clustering adds sequence cluster IDs, shared clusters, and homologous pairs via `--clustering` (requires `--enrich`). Optional variant mapping adds per-chain variant counts, interface variant enrichment, and constraint scores via `--variants` (requires `--interface --pae --enrich`). Optional stability scoring adds EVE evolutionary pathogenicity predictions via `--stability` (requires `--variants`). Optional offline AlphaMissense + monomeric FoldX scoring adds pathogenicity scores, DDG values, and pathogenic variant counts via `--protvar` (requires `--variants`). Optional disease annotation adds UniProt disease/PTM/GO/drug-target data via `--disease` (requires `--enrich`). Optional pathway mapping adds Reactome pathways, per-pathway PPI enrichment, and NetworkX network stats via `--pathways` (requires `--enrich`). STRING API validation is on by default during enrichment (disable with `--no-api`).

**visualise_results.py**: Generates up to 18 figures plus supplementary plots and on-demand per-complex PAE heatmaps. Features adaptive scatter sizing for large datasets and optional KDE density contour overlays. Figures 11-13 are generated automatically when variant columns are present. Figures 14-16 are generated automatically when disease and pathway columns are present. Figure 17 requires stability + ProtVar columns from `--stability --protvar`. Figure 18 requires clustering columns from `--clustering`.

**database_loaders.py**: Parsers for 4 protein-protein interaction databases. `load_string()` strips `9606.ENSP` prefixes and normalises combined scores from 0 - 1000 to 0.0 - 1.0. `load_biogrid()` filters to human (taxonomy 9606) physical interactions with Swiss-Prot/TrEMBL fallback extraction. `load_huri()` parses binary Y2H interactions with ENSG identifiers. `load_humap()` reads pairwise probability-scored interactions with optional UniProt ID validation. All parsers return standardised DataFrames with columns: `protein_a`, `protein_b`, `source`, `confidence_score`, `evidence_type`. `validate_with_api()` spot-checks loaded IDs against the STRING API (disable with `--no-api`).

**id_mapper.py**: Protein identifier cross-referencing using the STRING aliases file as a single source of truth. `IDMapper` class builds bidirectional lookup tables for ENSP-to-UniProt, UniProt-to-gene-symbol, and ENSG-to-ENSP mappings. `resolve_id()` accepts any identifier type and resolves to a target namespace, with automatic STRING API fallback when local lookup fails (`api_fallback=True` by default; disable with `--no-api`). Isoform-aware: preserves full isoform accessions (e.g., `P22607-2`) and prioritises reviewed Swiss-Prot accessions over TrEMBL. Includes `map_dataframe_to_uniprot()` for batch DataFrame ID conversion, `build_uniprot_lookup()` for efficient enrichment, and `export_lookup_table()` for structured CSV export with primary/secondary accession columns.

**overlap_analysis.py**: Computes pairwise protein interaction overlaps across databases after ID normalisation. `normalise_pair()` and `normalise_pair_base()` create order-independent pair keys at isoform-specific and base-accession levels respectively. `extract_pair_set()` and `extract_pair_set_base()` convert DataFrames to normalised pair sets. `compute_overlaps()` returns per-database counts, pairwise overlaps, triple overlaps, all-database intersections, and unique-to-database sets. Supports UpSet-style intersection visualisation for 4+ databases. CLI supports dual-level analysis (`--base-level`), report generation (`--report`), and STRING threshold comparison.

**string_api.py**: Centralised STRING database API client. All STRING API interactions are routed through this module. Architecture is offline-first: local flat files remain the primary data source; the API is an automatic supplement for unresolved identifiers and validation. Features rate-limited requests (1s between calls), automatic retry with exponential backoff on HTTP 429/5xx, SHA256-keyed response caching (auto-enabled to `data/string_api_cache/`), and caller identity injection per STRING API TOS. Provides 7 public functions: `get_string_ids()`, `get_interaction_partners()`, `query_homology()`, `query_enrichment()`, `query_ppi_enrichment()`, `query_network()`, `get_version()`. Raises `StringAPIError` on failure for clean error propagation.

**protein_clustering.py**: Parses STRING pre-computed protein sequence clusters and maps them to UniProt accessions via `IDMapper`. Defaults to `data/clusters/9606.clusters.proteins.v12.0.txt` when no `--clusters-file` is specified. Builds bidirectional cluster indices (cluster-to-proteins and protein-to-clusters) in both ENSP and UniProt space. `find_shared_clusters()` identifies sequence family relationships between protein pairs. `find_homologous_pairs()` discovers other protein pairs that share the same cluster combinations, with optional filtering against known interaction databases. Clusters exceeding `MAX_CLUSTER_SIZE_FOR_PAIRS` (500) are skipped during pair generation to avoid O(n^2) explosion from STRING's hierarchical mega-clusters (up to 144K UniProt members after isoform expansion). `annotate_results_with_clustering()` adds 7 CSV columns: union and intersection cluster IDs/counts, homologous pairs with counts, and a continuous homology bitscore from the STRING API. `enrich_with_homology_scores()` optionally queries the STRING API for paralogy bitscores using chunked batched deduplication (`HOMOLOGY_API_BATCH_SIZE = 100` proteins per API call; reduced from 500 because the STRING homology endpoint times out at larger batch sizes). `validate_clustering_mode()` accepts `'string'` and raises `NotImplementedError` for deferred `'foldseek'` and `'hybrid'` modes. Standalone CLI supports `--summary`, `--protein`, and `--pair` lookup modes.

**stability_scorer.py**: Integrates EVE (Evolutionary model of Variant Effect) pathogenicity predictions with the variant mapping pipeline. Loads per-protein EVE score CSVs keyed by UniProt entry name (e.g. `1433G_HUMAN.csv`) using a `HUMAN_9606_idmapping.dat` mapping file to convert from pipeline accessions (e.g. `P61981`). Lazy-loads only EVE CSVs for proteins in the current run. `annotate_results_with_stability()` adds 8 CSV columns: per-chain mean EVE scores, pathogenic counts, coverage fractions, and pipe-separated stability detail strings. Isoform accessions are automatically stripped to canonical before EVE lookup. Standalone CLI supports `summary` (coverage stats) and `lookup` (per-protein/position score query) subcommands.

**protvar_client.py**: Offline pathogenicity and stability scoring using two pre-computed data files: AlphaMissense pathogenicity scores (`AlphaMissense_aa_substitutions.tsv`, 216M variants from DeepMind) and AFDB monomeric FoldX DDG values (`afdb_foldx_export_20250210.csv`, 209M substitutions from EBI). Both files are streamed with accession/position filtering for memory efficiency. No API dependency — all scoring is local. `annotate_results_with_protvar()` adds 8 CSV columns: per-chain AlphaMissense mean scores, monomeric FoldX mean DDG, AlphaMissense pathogenic variant counts, and pipe-separated detail strings. Isoform accessions are automatically stripped to canonical. Standalone CLI supports `summary` (data statistics) and `lookup` (per-protein/position score query) subcommands.

**disease_annotations.py**: UniProt disease, PTM, GO term, and drug target annotation. Offline-first: streaming `iterparse` of local `uniprot_sprot_human.xml` (20,431 reviewed human entries). API fallback via `fetch_uniprot_annotation_api()` for proteins missing from local XML. Extracts disease associations (with OMIM cross-references), PTM sites (phosphorylation, ubiquitination, glycosylation, lipidation), Gene Ontology terms (biological process + molecular function), and drug target status via Pharmaceutical keyword. `annotate_results_with_disease()` adds 14 CSV columns. Detail fields truncated at `DETAILS_DISPLAY_LIMIT=50` items. Standalone CLI supports `summary` and `lookup` subcommands.

**pathway_network.py**: Reactome pathway mapping with per-pathway PPI enrichment and NetworkX network analysis. Offline-first: parses `UniProt2Reactome_All_Levels.txt` for pathway mappings and `ReactomePathwaysRelation.txt` for hierarchy. Per-pathway PPI enrichment via `invert_reactome_index()` + `run_per_pathway_ppi_enrichment()` using the STRING API (skipped with `--no-api`). Builds NetworkX interaction graphs with configurable pDockQ threshold (`NETWORK_PDOCKQ_THRESHOLD=0.23`). `annotate_results_with_pathways()` adds 10 CSV columns. Optional NetworkX dependency (`_HAS_NETWORKX` guard). Standalone CLI supports `summary`, `network`, and `enrichment` subcommands.

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

### Stability Scoring

```bash
# With EVE stability scoring (requires --variants)
python toolkit.py --dir /path/to/models --output results.csv --interface --pae --enrich data/ppi/9606.protein.aliases.v12.0.txt --variants --stability

# With custom stability data directory
python toolkit.py --dir /path/to/models --output results.csv --interface --pae --enrich data/ppi/9606.protein.aliases.v12.0.txt --variants --stability data/stability/

# Standalone: EVE coverage summary
python stability_scorer.py summary --stability-dir data/stability/

# Standalone: look up EVE scores for a protein
python stability_scorer.py lookup --stability-dir data/stability/ --protein P61981
```

### ProtVar Cross-Validation

```bash
# With offline AlphaMissense + monomeric FoldX scoring (requires --variants)
python toolkit.py --dir /path/to/models --output results.csv --interface --pae --enrich data/ppi/9606.protein.aliases.v12.0.txt --variants --protvar

# With custom AlphaMissense file path
python toolkit.py --dir /path/to/models --output results.csv --interface --pae --enrich data/ppi/9606.protein.aliases.v12.0.txt --variants --protvar --am-file data/stability/AlphaMissense_aa_substitutions.tsv

# Standalone: look up scores for a protein/position
python protvar_client.py lookup --protein P61981 --position 4

# Standalone: data file summary
python protvar_client.py summary
```

### Disease Annotations

```bash
# With disease annotation (requires --enrich)
python toolkit.py --dir /path/to/models --output results.csv --interface --pae --enrich data/ppi/9606.protein.aliases.v12.0.txt --disease

# With custom disease data directory
python toolkit.py --dir /path/to/models --output results.csv --interface --pae --enrich data/ppi/9606.protein.aliases.v12.0.txt --disease data/pathways/

# Standalone: disease annotation summary
python disease_annotations.py summary --disease-dir data/pathways/

# Standalone: look up annotations for a protein
python disease_annotations.py lookup --disease-dir data/pathways/ --protein P04637
```

### Pathway Network Analysis

```bash
# With pathway mapping (requires --enrich)
python toolkit.py --dir /path/to/models --output results.csv --interface --pae --enrich data/ppi/9606.protein.aliases.v12.0.txt --pathways

# With pathways in offline mode (skip STRING API enrichment)
python toolkit.py --dir /path/to/models --output results.csv --interface --pae --enrich data/ppi/9606.protein.aliases.v12.0.txt --pathways --no-api

# Standalone: pathway summary statistics
python pathway_network.py summary --csv results.csv

# Standalone: build and plot pathway networks
python pathway_network.py network --csv results.csv --output-dir Output
```

### Full Pipeline

```bash
# Full pipeline with all features
python toolkit.py --dir /path/to/models --output results.csv --interface --pae --enrich data/ppi/9606.protein.aliases.v12.0.txt --databases data/ppi/ --clustering --variants --stability --protvar --disease --pathways --pymol
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

# Stability scoring tests
python -m pytest tests/ -m "stability" -v

# ProtVar API tests (mocked, no network)
python -m pytest tests/ -m "protvar" -v

# Disease annotation tests
python -m pytest tests/ -m "disease" -v

# Pathway network tests
python -m pytest tests/ -m "pathways" -v

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

### CSV (25 base columns, up to 119 with all features)

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
| 14 | Pathway Coherence | Pathway complexity histogram + enrichment scatter with per-pathway PPI statistics |
| 15 | Disease Enrichment | Disease prevalence by quality tier (grouped bars + chi-square) + top 10 diseases stacked bars |
| 16 | Pathway Network | NetworkX spring layout of top Reactome pathways, coloured by % High-tier complexes |
| 17 | Stability Cross-Validation | EVE vs AlphaMissense concordance, AlphaMissense vs FoldX DDG, coverage landscape by tier |
| 18 | Clustering Validation | Homodimer ground truth scatter (shared = total clusters), cluster ratio by quality tier |

Figures 1-2 are generated from base CSV columns. Figures 3-9 require `--interface --pae` columns. Figure 10 requires the `n_chains` column. Figures 11-13 require variant columns from `--variants`. Figures 14 and 16 require pathway columns from `--pathways`. Figure 15 requires disease columns from `--disease`. Figure 17 requires stability + ProtVar columns from `--stability --protvar`. Figure 18 requires clustering columns from `--clustering`.


## Roadmap

### Completed

- **Aim 5 - Structure Prediction Quality Assessment:** JAX-free PKL extraction, pDockQ scoring, 2-phase interface analysis, 46-column CSV, 10-figure visualisation suite
- **Aim 1 - Database Ingestion:** Parsers for STRING, BioGRID, HuRI, and HuMAP with standardised DataFrame output
- **Aim 2 - ID Cross-Referencing:** Isoform-aware mapping pipeline using STRING aliases (ENSP/ENSG/UniProt/gene symbol) with dual-level cross-database overlap analysis, structured lookup table export, and toolkit CSV enrichment
- **STRING API Integration:** Centralised API client (`string_api.py`) with automatic validation fallback across ID resolution, enrichment, and database loading - on by default with `--no-api` opt-out
- **Aim 3 - Protein Clustering:** STRING sequence cluster parsing, UniProt-mapped indexing, homologous pair detection, optional API homology scores, clustering validation figure (Fig 18), toolkit integration with `--clustering` flag (Foldseek/hybrid modes deferred)
- **Aim 4 - Variant Mapping:** UniProt/ClinVar/ExAC variant parsing, biotite/BioPython SASA-based 4-class structural context classification (interface core/rim via cross-chain distance, surface, buried core), per-complex variant burden and enrichment analysis, 3 variant visualisation figures (Figs 11-13), toolkit integration with `--variants` and `--no-clinvar` flags
- **EVE Stability Scoring (D.2):** EVE evolutionary pathogenicity predictions with lazy-loaded per-protein score CSVs, accession-to-entry-name mapping via UniProt ID mapping file, 8 CSV columns, stability cross-validation figure (Fig 17 Panel A+C), toolkit integration with `--stability` flag
- **Offline AlphaMissense + Monomeric FoldX Scoring (D.1):** Pre-computed AlphaMissense pathogenicity scores (216M variants) and AFDB FoldX DDG values (209M substitutions) from local data files; no API dependency; 8 CSV columns, stability cross-validation figure (Fig 17 Panel B+C); toolkit integration with `--protvar` flag
- **Disease & Pathway Integration (Phase E):** UniProt disease/PTM/GO/drug-target annotation (offline XML + API fallback), Reactome pathway mapping with per-pathway PPI enrichment, NetworkX network analysis, 3 visualisation figures (Figs 14-16), toolkit integration with `--disease` and `--pathways` flags, 24 new CSV columns
- **PyMOL Visualisation (Phase F):** Layered `.pml` script generation with chain colouring, pLDDT confidence bands (canonical AF2 4-band scheme), interface residue highlighting (sticks), variant position colouring by structural context (spheres), `py3Dmol` in-notebook fallback, tqdm progress bar, standalone CLI (`generate` + `batch` subcommands), toolkit integration with `--pymol` flag

### Planned
- **Million-Complex Production Run:** Full pipeline validation on large-scale AlphaFold-Multimer dataset


## Testing

The test suite contains **1002 tests** across 19 modules (985 active + 15 future placeholders):

| Module | Tests | Scope |
|--------|-------|-------|
| test_read_af2_nojax.py | 26 | PKL loading, metric extraction |
| test_pdockq.py | 39 | PDB parsing, pDockQ calculation, multi-chain |
| test_interface_analysis.py | 39 | Interface geometry, pLDDT, PAE, composite |
| test_toolkit.py | 54 | File discovery, quality classification, CSV, enrichment, sequences |
| test_visualise_results.py | 60 | Figure generation (Figs 1-18 incl. Phase E, stability, clustering), variant/disease detail parsing, data loading, CLI |
| test_integration.py | 8 | Cross-module pipeline, data flow |
| test_database_loaders.py | 70 | STRING/BioGRID/HuRI/HuMAP parsing, edge cases, cross-DB overlap, base-level overlap |
| test_id_mapper.py | 65 | ID validation, mapping, isoform handling, secondary accessions, lookup builder |
| test_multiprocessing.py | 6 | Pickling, subprocess import, parallel parity |
| test_string_api.py | 56 | STRING API client, caching, rate limiting, retry, API fallback integration, database validation |
| test_protein_clustering.py | 55 | Protein clustering, homology detection, oversized cluster handling, CLI |
| test_variant_mapper.py | 103 | HGVS parsing, variant loading, SASA, parallel SASA (incl. combined both-chains), structural context (cross-chain distance), enrichment, CLI, toolkit integration |
| test_stability_scorer.py | 59 | EVE score loading, entry-name mapping, index building, annotation, formatting, parsing, CSV columns, CLI, regression |
| test_protvar_client.py | 88 | Offline AlphaMissense TSV loading, AFDB FoldX CSV loading, AM variant parsing, combined index building, score lookup, chain scoring, detail formatting, annotation, CSV columns, CLI, regression |
| test_disease_annotations.py | 96 | UniProt disease/PTM/GO parsing, API fallback, formatting, annotation, CLI, regression |
| test_pathway_network.py | 83 | Reactome loading, pathway quality, NetworkX, annotation, per-pathway PPI enrichment, CLI |
| test_pymol_scripts.py | 60 | PyMOL .pml generation, PML syntax, chain/pLDDT/interface/variant colouring, py3Dmol fallback, CLI, regression |
| test_future_aims.py | 18 + 15 | 18 real tests (7 database + 6 variant + 1 EVE + 1 ProtVar + 3 pathway) + 15 future placeholders |

**Results:** 984 passing, 1 skipped (Fig 10 — all test complexes are dimers), 15 future placeholders (deselected by default)

**Markers:** `slow` (file I/O), `regression` (exact numerical values), `integration` (cross-module), `cli` (command-line), `database` (PPI database loading and ID mapping), `multiprocessing` (parallel processing), `api` (STRING API, mocked), `clustering` (protein clustering and homology), `variants` (variant mapping and structural context), `stability` (EVE stability scoring), `protvar` (offline AlphaMissense + monomeric FoldX scoring), `alphamissense` (AlphaMissense scoring), `disease` (UniProt disease annotation), `pathways` (pathway mapping and network), `phase_e` (Phase E figure tests), `pymol` (PyMOL script generation), `future` (unimplemented features)


## Acknowledgements
Developed by Talhah Zubayer under the supervision of David Burke as part of the MSc Applied Bioinformatics programme at King's College London.
