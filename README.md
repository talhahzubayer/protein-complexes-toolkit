# Protein Complexes Toolkit

A Python toolkit to facilitate the analysis of protein complexes and target drug discovery

MSc Applied Bioinformatics Research Project - King's College London

**Student:** Talhah Zubayer | **Supervisor:** David Burke


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
├── Toolkit_Commands_List.md  # Full CLI command reference (all flags, defaults, examples)
├── requirements.txt          # Python dependencies
├── .gitignore
├── data/                         # External databases (not included in repo)
│    ├── ppi/                     # PPI databases (see "Setting Up Data")
│    ├── clusters/                # STRING sequence clusters (see "Setting Up Data")
│    ├── variants/                # Variant databases (see "Setting Up Data")
│    ├── stability/               # Stability prediction data (see "Setting Up Data")
│    ├── pathways/                # Disease & pathway databases (see "Setting Up Data")
│    └── string_api_cache/        # STRING API response cache (auto-generated)
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

The pipeline produces a 25-column base CSV, progressively expandable to 121 columns by stacking optional flags (`--enrich`, `--clustering`, `--variants`, `--stability`, `--protvar`, `--disease`, `--pathways`). JSONL interface export is also available. STRING API validation is on by default across all modules; disable with `--no-api`. Each downstream module also provides a standalone CLI.

#### Core Analysis

**read_af2_nojax.py** - Loads AlphaFold2 result PKL files without requiring a JAX installation. Extracts ipTM, pTM, pLDDT arrays, and PAE matrices from `.pkl`, `.pkl.gz`, and `.pkl.bz2` formats.

**pdockq.py** - Calculates predicted DockQ scores using the FoldDock sigmoid model. Automatically selects the best interacting chain pair in multi-chain complexes and returns full contact geometry.

**interface_analysis.py** - 2-phase interface characterisation. Phase 1 derives structural geometry from PDB alone (contact count, interface fractions, symmetry, density, interface vs bulk pLDDT). Phase 2 adds PAE-aware confident contact identification, composite confidence scoring, and automated quality flags including paradox detection and metric disagreement.

**toolkit.py** - Batch orchestrator that processes directories of AlphaFold2 predictions with multiprocessing, periodic checkpointing, and resume from interruption. Each optional flag activates a downstream module: `--enrich` (gene symbols, protein names, sequences, database source tagging), `--clustering` (sequence clusters, homologous pairs), `--variants` (variant mapping and structural context), `--stability` (EVE scores), `--protvar` (AlphaMissense + FoldX), `--disease` (UniProt annotations), `--pathways` (Reactome + network analysis), `--pymol` (PyMOL script generation). Implements 2 quality classification schemes.

**visualise_results.py** - Generates up to 19 figures with adaptive scatter sizing for large datasets and optional KDE density contour overlays. Figures are generated automatically based on which columns are present in the CSV (e.g., variant figures from `--variants`, pathway figures from `--pathways`).

#### Database & Enrichment

**database_loaders.py** - Parsers for STRING, BioGRID, HuRI, and HuMAP protein interaction databases. All return standardised DataFrames (`protein_a`, `protein_b`, `source`, `confidence_score`, `evidence_type`) with optional API spot-check validation.

**id_mapper.py** - Isoform-aware protein identifier cross-referencing (ENSP, ENSG, UniProt, gene symbol) using STRING aliases as a single source of truth. Resolves any identifier type to a target namespace with automatic API fallback for local misses.

**overlap_analysis.py** - Computes pairwise interaction overlaps across databases with UpSet-style visualisation. Supports dual-level analysis (isoform-specific and base-accession) via `--base-level` and report generation via `--report`.

**string_api.py** - Centralised STRING API client through which all API interactions are routed. Offline-first architecture with rate limiting, automatic retry/backoff, and SHA256-keyed response caching.

**protein_clustering.py** - Parses STRING sequence clusters, maps them to UniProt accessions, and detects homologous protein pairs with optional API-based paralogy bitscores (`--clustering`). Caps pair enumeration for oversized clusters to avoid combinatorial explosion.

#### Variant & Stability

**variant_mapper.py** - Maps variants from UniProt, ClinVar, and ExAC onto complex interface residues (`--variants`). Computes SASA via biotite (with BioPython fallback) to classify each variant into 4 structural contexts: `interface_core`, `interface_rim`, `surface_non_interface`, or `buried_core`. Adds per-complex variant burden, enrichment fold-change, and ExAC constraint scores. Databases loaded via chunked streaming for memory efficiency.

**stability_scorer.py** - Integrates EVE evolutionary pathogenicity predictions with the variant pipeline (`--stability`). Lazy-loads only EVE score CSVs for proteins in the current run, mapping pipeline accessions to entry names via `HUMAN_9606_idmapping.dat`.

**protvar_client.py** - Offline pathogenicity and stability scoring from pre-computed AlphaMissense (216M variants) and AFDB monomeric FoldX DDG (209M substitutions) data files (`--protvar`). No API dependency; both files are streamed with accession/position filtering for memory efficiency.

#### Disease & Pathways

**disease_annotations.py** - Annotates proteins with UniProt disease associations, PTM sites (phosphorylation, ubiquitination, glycosylation, lipidation), GO terms, and drug target status (`--disease`). Offline-first via streaming XML parsing of reviewed human entries, with API fallback for missing proteins.

**pathway_network.py** - Maps proteins to Reactome pathways and runs per-pathway PPI enrichment via the STRING API (`--pathways`). Builds NetworkX interaction graphs for network topology analysis (degree, centrality). Generates 3 pathway/disease visualisation figures (Figs 14-16).

#### Structural Visualisation

**pymol_scripts.py** - Generates scene-managed PyMOL `.pml` scripts with layered visualisation: chain colouring (10-chain palette, homodimer transparency), pLDDT confidence bands, interface residue sticks, pathogenicity-aware variant spheres coloured by structural context, and AlphaMissense transparency overlay (`--pymol`, `--pymol-min-tier`, `--pymol-render`). Includes metadata and biological annotation comments, pre-computed interface residue lookup to avoid redundant PDB I/O, and a `py3Dmol` fallback for in-notebook rendering.


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


## Usage

For the complete command reference - including all CLI flags, their defaults, flag dependencies, progressive flag-stacking examples and standalone module CLIs - see **[Toolkit_Commands_List.md](Toolkit_Commands_List.md)**.


## Input Data Format

The toolkit expects a directory containing paired AlphaFold2-Multimer output files:

```
Protein_Complexes/
├── ProteinA_ProteinB.pdb                                          # Old naming
├── ProteinA_ProteinB.results.pkl                                  # Old naming
├── ProteinC_ProteinD_relaxed_model_1_multimer_v3_pred_0.pdb       # New naming
├── ProteinC_ProteinD_result_model_1_multimer_v3_pred_0.pkl        # New naming
└── ...
```

Each complex requires a **paired PDB structure file and PKL result file**. The toolkit supports two naming conventions:

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

Each file pair contains:
- A **PDB file** containing the predicted structure with ATOM records
- A **PKL file** containing the AlphaFold2 result dictionary (ipTM, pTM, pLDDT, PAE)

The toolkit also handles homodimer, isoform, and multi-chain naming patterns.


## Output

### CSV (25 base columns, up to 121 with all features)

The main output CSV groups columns into:

| Category | Key Columns |
|----------|-------------|
| **Identity** | complex_name, protein_a, protein_b, complex_type, n_chains, species, structure_source |
| **Core Metrics** | ipTM, pTM, ranking_confidence, pDockQ, ppv |
| **pLDDT Statistics** | plddt_mean, plddt_median, plddt_min, plddt_max, plddt_below50/70_fraction |
| **Interface Geometry** | n_interface_contacts, n_interface_residues_a/b, interface_residues_a/b, interface_fraction_a/b, interface_symmetry, contacts_per_interface_residue |
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
| 19 | Prediction Quality Paradox | 2x2 panel: pathogenic interface variants and PPI density strengthen with quality (top row) while gene constraint and disorder fraction decline (bottom row), revealing systematic AF2-Multimer prediction bias toward ordered protein pairs |

Figures 1-2 are generated from base CSV columns. Figures 3-9 require `--interface --pae` columns. Figure 10 requires the `n_chains` column. Figures 11-13 require variant columns from `--variants`. Figures 14 and 16 require pathway columns from `--pathways`. Figure 15 requires disease columns from `--disease`. Figure 17 requires stability + ProtVar columns from `--stability --protvar`. Figure 18 requires clustering columns from `--clustering`. Figure 19 requires variant + pathway columns from `--variants --pathways`.

## Acknowledgements
Developed by Talhah Zubayer under the supervision of David Burke as part of the MSc Applied Bioinformatics programme at King's College London.
