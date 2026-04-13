# Toolkit Commands List

Copy-paste-ready terminal commands for all toolkit features.

> **Cross-platform note:** All commands below use `python` (Windows convention). On Linux/macOS, substitute `python3` if `python` does not point to Python 3. Backslash paths (`\`) should be replaced with forward slashes (`/`) on Unix systems.

---

## Placeholder Reference

All paths in this document use angle-bracket placeholders. Replace them with your actual paths before running.

| Placeholder | Description | Default (if omitted) |
|---|---|---|
| `<MODELS_DIR>` | Directory containing AlphaFold2-Multimer PDB and PKL output files | **Required** — no default |
| `<OUTPUT_CSV>` | Output CSV file path | `results.csv` |
| `<ALIASES_FILE>` | STRING protein aliases file (`9606.protein.aliases.v12.0.txt`) | `data/ppi/9606.protein.aliases.v12.0.txt` |
| `<PPI_DIR>` | Directory containing PPI database files: STRING `9606.protein.links.v12.0.txt`, BioGRID `BIOGRID-ALL-5.0.253.tab3.txt`, `HuRI.tsv`, HuMAP `humap2_ppis_ACC_20200821.pairsWprob` | `data/ppi/` |
| `<CLUSTERS_FILE>` | STRING protein clusters file (`9606.clusters.proteins.v12.0.txt`) | `data/clusters/9606.clusters.proteins.v12.0.txt` |
| `<VARIANTS_DIR>` | Directory containing variant database files: UniProt `homo_sapiens_variation.txt`, ClinVar `variant_summary.txt`, ExAC `forweb_cleaned_exac_r03_march16_z_data_pLI_CNV-final.txt` | `data/variants/` |
| `<STABILITY_DIR>` | Directory containing EVE data: `HUMAN_9606_idmapping.dat` mapping file and `EVE_all_data/` subdirectory with per-protein CSV files | `data/stability/` |
| `<FOLDX_EXPORT>` | AFDB FoldX export CSV (`afdb_foldx_export_20250210.csv`) | `data/stability/afdb_foldx_export_20250210.csv` |
| `<AM_FILE>` | AlphaMissense substitution scores (`AlphaMissense_aa_substitutions.tsv`) | `data/stability/AlphaMissense_aa_substitutions.tsv` |
| `<PATHWAYS_DIR>` | Directory containing `uniprot_sprot_human.xml`, Reactome `UniProt2Reactome_All_Levels.txt`, and `ReactomePathwaysRelation.txt` | `data/pathways/` |
| `<OUTPUT_DIR>` | Figure and report output directory | `Output/` |
| `<INTERFACES_JSONL>` | JSONL interface residue export file path, eg: just saying the file name like `interfaces.jsonl`, will generate that file in the home project directory | No default (user-specified) |
| `<PDB_FILE>` | Path to a single PDB file | **Required** — no default |
| `<PDB_DIR>` | Directory of PDB files (for batch operations) | **Required** — no default |
| `<PYMOL_OUTPUT>` | PyMOL script output directory | `pymol_scripts/` |

---

## 1. Toolkit Pipeline

The main analysis pipeline (`toolkit.py`) processes AlphaFold2-Multimer predictions to compute quality scores, interface geometry, and optional enrichment features. It reads PDB and PKL files from a models directory and produces a CSV with per-complex metrics. Features are composable: each `--flag` adds columns to the output CSV, and flags can be stacked in any valid combination.

### Flag Dependencies

| Flag | Requires |
|---|---|
| `--pae` | `--interface` |
| `--export-interfaces` | `--interface --pae` |
| `--databases` | `--enrich` |
| `--clustering` | `--enrich` |
| `--variants` | `--interface --pae --enrich` |
| `--no-clinvar` | `--variants` |
| `--stability` | `--variants` |
| `--protvar` | `--variants` |
| `--disease` | `--enrich` |
| `--pathways` | `--enrich` |
| `--pymol` | `--interface --pae` |
| `--full-pipeline` | `--dir` only (activates all flags with default data paths) |

### Flag Defaults

| Flag | Default when omitted |
|---|---|
| `--output` | `results.csv` |
| `-w` / `--workers` | `1` (sequential processing) |
| `--string-min-score` | `700` |
| `--clusters-file` | `data/clusters/9606.clusters.proteins.v12.0.txt` |
| `--variants <path>` | `data/variants/` (when `--variants` is used without a path) |
| `--stability <path>` | `data/stability/` (when `--stability` is used without a path) |
| `--protvar <path>` | `data/stability/afdb_foldx_export_20250210.csv` (when `--protvar` is used without a path) |
| `--am-file` | `data/stability/AlphaMissense_aa_substitutions.tsv` |
| `--disease <path>` | `data/pathways/` (when `--disease` is used without a path) |
| `--pymol-output` | `pymol_scripts/` |
| `--pymol-min-tier` | `High` |
| `--full-pipeline` | Activates `--interface --pae --enrich --databases --clustering --variants --stability --protvar --disease --pathways --pymol --checkpoint` with module defaults |

### Progressive Command Build-up

Commands are listed from simplest to most complex. Each builds on the previous by adding one or more flags.

```bash
# Basic analysis (sequential, no interface features)
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV>

# With interface geometry and PAE features
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV> --interface --pae

# With enrichment (gene symbols, protein names, sequences)
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV> --interface --pae --enrich <ALIASES_FILE>

# With enrichment + database source tagging
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV> --interface --pae --enrich <ALIASES_FILE> --databases <PPI_DIR>

# With JSONL interface export (requires --interface --pae)
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV> --interface --pae --export-interfaces <INTERFACES_JSONL>

# Parallel workers + checkpointing (-w defaults to 1 if omitted)
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV> --interface --pae -w 4 --checkpoint

# Resume an interrupted run (implies --checkpoint)
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV> --interface --pae -w 4 --resume

# Combined: enrichment + databases + parallel workers + checkpointing + JSONL export
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV> --interface --pae --enrich <ALIASES_FILE> --databases <PPI_DIR> --export-interfaces <INTERFACES_JSONL> -w 8 --checkpoint

# With protein clustering (requires --enrich; --clusters-file defaults to data/clusters/9606.clusters.proteins.v12.0.txt)
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV> --interface --pae --enrich <ALIASES_FILE> --clustering

# With clustering + database source tagging
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV> --interface --pae --enrich <ALIASES_FILE> --databases <PPI_DIR> --clustering

# With clustering + custom clusters file (overrides default path)
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV> --interface --pae --enrich <ALIASES_FILE> --clustering --clusters-file <CLUSTERS_FILE>

# With variant mapping (requires --interface --pae --enrich; --variants defaults to data/variants/)
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV> --interface --pae --enrich <ALIASES_FILE> --variants

# With variant mapping + custom variant database directory
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV> --interface --pae --enrich <ALIASES_FILE> --variants <VARIANTS_DIR>

# Without ClinVar enrichment (faster, UniProt + ExAC only)
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV> --interface --pae --enrich <ALIASES_FILE> --variants --no-clinvar

# With EVE stability scoring (requires --variants; --stability defaults to data/stability/)
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV> --interface --pae --enrich <ALIASES_FILE> --variants --stability

# With stability scoring + custom stability directory
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV> --interface --pae --enrich <ALIASES_FILE> --variants --stability <STABILITY_DIR>

# With offline AlphaMissense + monomeric FoldX scoring (requires --variants; --protvar defaults to data/stability/afdb_foldx_export_20250210.csv)
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV> --interface --pae --enrich <ALIASES_FILE> --variants --protvar

# With protvar + custom AlphaMissense file path (--am-file defaults to data/stability/AlphaMissense_aa_substitutions.tsv)
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV> --interface --pae --enrich <ALIASES_FILE> --variants --protvar --am-file <AM_FILE>

# With disease annotation (requires --enrich; --disease defaults to data/pathways/)
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV> --interface --pae --enrich <ALIASES_FILE> --disease

# With pathway mapping (requires --enrich)
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV> --interface --pae --enrich <ALIASES_FILE> --pathways

# With disease + pathways combined
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV> --interface --pae --enrich <ALIASES_FILE> --disease --pathways

# Full Phase D: clustering + variants + stability + protvar
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV> --interface --pae --enrich <ALIASES_FILE> --clustering --variants --stability --protvar

# Full pipeline: ALL features (enrichment + databases + clustering + variants + stability + protvar + disease + pathways + PyMOL)
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV> --interface --pae --enrich <ALIASES_FILE> --databases <PPI_DIR> --clustering --variants --stability --protvar --disease --pathways --pymol --export-interfaces <INTERFACES_JSONL> -w 8 --checkpoint

# Full pipeline shorthand (--full-pipeline activates all flags with default data paths, validates all data files exist before starting)
python toolkit.py --full-pipeline --dir <MODELS_DIR> -w 8 --output <OUTPUT_CSV>

# Clustering without API calls (offline-only, skips STRING homology scores)
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV> --interface --pae --enrich <ALIASES_FILE> --clustering --no-api

# Verbose output (sequential only)
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV> --interface --pae -v
```

---

## 2. Visualisation

Generates publication-quality figures (Figs 1-13) from the toolkit CSV output, including quality scatter plots, score distributions, interface geometry plots, and PAE heatmaps. Reads the CSV produced by `toolkit.py` and outputs figures to a directory.

### Flag Defaults

| Flag | Default when omitted |
|---|---|
| `--output-dir` | `Output/` |
| `--limit` | No limit (all complexes) |

### Commands

```bash
# All figures (reads toolkit CSV output)
python visualise_results.py <OUTPUT_CSV>

# With KDE density contour overlays
python visualise_results.py <OUTPUT_CSV> --density

# With disorder-coloured scatter (Fig 1b)
python visualise_results.py <OUTPUT_CSV> --disorder-scatter

# With per-complex PAE heatmaps (requires models directory for PKL files)
python visualise_results.py <OUTPUT_CSV> --pae-heatmaps <MODELS_DIR>

# PAE heatmaps limited to first 10 complexes (--limit defaults to no limit)
python visualise_results.py <OUTPUT_CSV> --pae-heatmaps <MODELS_DIR> --limit 10

# Custom output directory (--output-dir defaults to Output/)
python visualise_results.py <OUTPUT_CSV> --output-dir <OUTPUT_DIR>

# All options combined
python visualise_results.py <OUTPUT_CSV> --density --disorder-scatter --pae-heatmaps <MODELS_DIR> --limit 20 --output-dir <OUTPUT_DIR>
```

---

## 3. Database Loading

Loads and parses PPI interaction databases (STRING, BioGRID, HuRI, HuMAP) from local files. Can load individual databases or all at once, apply confidence filters, and export results to CSV. Optionally validates loaded IDs against the STRING API.

### Flag Defaults

| Flag | Default when omitted |
|---|---|
| `--data-dir` | `data/ppi/` |
| `--database` | `all` (loads all four databases) |
| `--min-string-score` | `0` (no filter) |
| `--min-humap-prob` | `0.0` (no filter) |

### Commands

```bash
# Load all databases (prints summary; --data-dir defaults to data/ppi/)
python database_loaders.py --data-dir <PPI_DIR>

# Load individual databases
python database_loaders.py --data-dir <PPI_DIR> --database string
python database_loaders.py --data-dir <PPI_DIR> --database biogrid
python database_loaders.py --data-dir <PPI_DIR> --database huri
python database_loaders.py --data-dir <PPI_DIR> --database humap

# Export all to CSV (--output omitted = prints summary only)
python database_loaders.py --data-dir <PPI_DIR> --output all_interactions.csv

# With STRING minimum score filter (--min-string-score defaults to 0; tiers: 150=low, 400=medium, 700=high, 900=highest)
python database_loaders.py --data-dir <PPI_DIR> --min-string-score 700

# With HuMAP minimum probability filter (--min-humap-prob defaults to 0.0)
python database_loaders.py --data-dir <PPI_DIR> --min-humap-prob 0.5

# Verbose
python database_loaders.py --data-dir <PPI_DIR> -v
```

---

## 4. ID Mapping

Maps identifiers between UniProt accessions, Ensembl protein/gene IDs, and gene symbols using the STRING aliases file. Can resolve single identifiers, print mapping statistics, or export a full lookup table. Optionally falls back to the STRING API for unresolved IDs.

### Flag Defaults

| Flag | Default when omitted |
|---|---|
| `--aliases` | `data/ppi/9606.protein.aliases.v12.0.txt` |

### Commands

```bash
# Print mapping statistics (--aliases defaults to data/ppi/9606.protein.aliases.v12.0.txt)
python id_mapper.py --aliases <ALIASES_FILE> --stats

# Resolve a single identifier (UniProt, ENSP, ENSG, or gene symbol)
python id_mapper.py --aliases <ALIASES_FILE> --resolve P04637
python id_mapper.py --aliases <ALIASES_FILE> --resolve ENSP00000269305
python id_mapper.py --aliases <ALIASES_FILE> --resolve TP53

# Export full lookup table to CSV
python id_mapper.py --aliases <ALIASES_FILE> --export lookup_table.csv

# Verbose
python id_mapper.py --aliases <ALIASES_FILE> --stats -v
```

---

## 5. Overlap Analysis

Computes pairwise overlap between PPI databases (STRING, BioGRID, HuRI, HuMAP) and generates Venn/UpSet-style diagrams. Requires the STRING aliases file for ID mapping. Can perform both isoform-specific and base-accession-level analyses, and generate threshold comparison figures.

### Flag Defaults

| Flag | Default when omitted |
|---|---|
| `--data-dir` | `data/ppi/` |
| `--output` | `Output/venn_overlap.png` |
| `--string-min-score` | `700` |

> **Note:** `--aliases` is required and has no default for this script.

### Commands

```bash
# Basic Venn diagram (--string-min-score defaults to 700)
python overlap_analysis.py --data-dir <PPI_DIR> --aliases <ALIASES_FILE>

# Dual-level analysis (isoform-specific + base-accession)
python overlap_analysis.py --data-dir <PPI_DIR> --aliases <ALIASES_FILE> --base-level

# With full report (writes overlap statistics to text file)
python overlap_analysis.py --data-dir <PPI_DIR> --aliases <ALIASES_FILE> --base-level --report <OUTPUT_DIR>/overlap_report.txt

# Threshold comparison figure
python overlap_analysis.py --data-dir <PPI_DIR> --aliases <ALIASES_FILE> --threshold-comparison <OUTPUT_DIR>/threshold_comparison.png

# Custom output path (--output defaults to Output/venn_overlap.png)
python overlap_analysis.py --data-dir <PPI_DIR> --aliases <ALIASES_FILE> --output <OUTPUT_DIR>/venn_overlap.png

# Custom STRING threshold (--string-min-score defaults to 700; tiers: 150=low, 400=medium, 700=high, 900=highest)
python overlap_analysis.py --data-dir <PPI_DIR> --aliases <ALIASES_FILE> --string-min-score 900

# Verbose
python overlap_analysis.py --data-dir <PPI_DIR> --aliases <ALIASES_FILE> -v
```

---

## 6. Protein Clustering

Analyses STRING protein sequence clusters to detect homologous pairs and shared cluster membership between proteins. Requires the STRING clusters file and aliases file. In the toolkit pipeline, `--clustering` uses the default clusters file automatically; the standalone CLI requires `--clusters-file` explicitly.

> **Note:** `--clusters-file` defaults to `data/clusters/9606.clusters.proteins.v12.0.txt` in the toolkit pipeline. The standalone CLI requires it explicitly.

### Commands

```bash
# Cluster summary statistics
python protein_clustering.py --clusters-file <CLUSTERS_FILE> --aliases <ALIASES_FILE> --summary

# Look up clusters for a single protein
python protein_clustering.py --clusters-file <CLUSTERS_FILE> --aliases <ALIASES_FILE> --protein P04637

# Find shared clusters between two proteins
python protein_clustering.py --clusters-file <CLUSTERS_FILE> --aliases <ALIASES_FILE> --pair P04637 Q00987
```

In the toolkit pipeline (Section 1), add `--no-api` to any `--clustering` command to skip STRING API homology score queries:

```bash
# Clustering without API calls (offline-only, skips homology scores)
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV> --interface --pae --enrich <ALIASES_FILE> --clustering --no-api
```

---

## 7. Variant Mapping

Maps known protein variants (UniProt, ClinVar, ExAC) to interface residues on predicted complexes. Computes per-residue SASA to classify variants by structural context (interface core, interface rim, surface, buried). Requires BioPython (`pip install biopython`). In the toolkit pipeline, `--variants` requires `--interface --pae --enrich`.

> **Note:** `--variants` defaults to `data/variants/` when the path is omitted in the toolkit. The standalone CLI requires `--variants-dir` explicitly.

### Flag Defaults (Standalone CLI)

| Flag | Default when omitted |
|---|---|
| `--variants-dir` | `data/variants/` |
| `--output` (map subcommand) | `variant_analysis.csv` |

### Commands

```bash
# Variant database summary statistics (--variants-dir defaults to data/variants/)
python variant_mapper.py summary --variants-dir <VARIANTS_DIR>

# Look up variants for a single protein
python variant_mapper.py lookup --variants-dir <VARIANTS_DIR> --protein P04637

# Map variants to interfaces from a JSONL export
python variant_mapper.py map --interfaces <INTERFACES_JSONL> --pdb-dir <PDB_DIR> --variants-dir <VARIANTS_DIR> --output variant_analysis.csv
```

In the toolkit pipeline (Section 1), add `--no-clinvar` to any `--variants` command to skip ClinVar loading:

```bash
# Variants without ClinVar (UniProt + ExAC only, faster loading)
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV> --interface --pae --enrich <ALIASES_FILE> --variants --no-clinvar
```

---

## 8. Stability Scoring

Scores interface variants using EVE (Evolutionary model of Variant Effect) predictions. Maps UniProt accessions to EVE entry names via the UniProt ID mapping file, then loads per-protein EVE CSVs on demand. Requires `HUMAN_9606_idmapping.dat` and the `EVE_all_data/` directory in the stability directory. In the toolkit pipeline, `--stability` requires `--variants`.

> **Note:** `--stability` defaults to `data/stability/` when the path is omitted in the toolkit. The standalone CLI uses `--stability-dir` which also defaults to `data/stability/`.

### Data Setup

Download the UniProt ID mapping file (needed to convert EVE entry names to accessions):

1. Download `HUMAN_9606_idmapping.dat.gz` from [ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/)
2. Decompress and place at `data/stability/HUMAN_9606_idmapping.dat`

### Flag Defaults (Standalone CLI)

| Flag | Default when omitted |
|---|---|
| `--stability-dir` | `data/stability/` |

### Commands

```bash
# EVE coverage summary statistics (--stability-dir defaults to data/stability/)
python stability_scorer.py --stability-dir <STABILITY_DIR> summary

# Look up EVE scores for a single protein
python stability_scorer.py --stability-dir <STABILITY_DIR> lookup --protein P61981

# Look up EVE scores at a specific position
python stability_scorer.py --stability-dir <STABILITY_DIR> lookup --protein P61981 --position 45
```

---

## 9. Offline AlphaMissense + Monomeric FoldX Scoring

Scores variants using offline AlphaMissense pathogenicity predictions and AFDB monomeric FoldX stability data. No internet access required. Reads local TSV/CSV data files. In the toolkit pipeline, `--protvar` requires `--variants`.

> **Note:** `--protvar` reads local data files from `data/stability/`. When file paths are not specified, defaults are used: `data/stability/afdb_foldx_export_20250210.csv` for the FoldX export and `data/stability/AlphaMissense_aa_substitutions.tsv` for AlphaMissense. Use `--protvar <FOLDX_EXPORT>` and `--am-file <AM_FILE>` to override.

### Flag Defaults

| Flag | Default when omitted |
|---|---|
| `--foldx-export` (standalone) / `--protvar` (toolkit) | `data/stability/afdb_foldx_export_20250210.csv` |
| `--am-file` | `data/stability/AlphaMissense_aa_substitutions.tsv` |

### Standalone CLI

```bash
# View data file statistics
python protvar_client.py summary

# Look up AlphaMissense + FoldX scores for a specific protein
python protvar_client.py lookup --protein P61981

# Look up scores at a specific position
python protvar_client.py lookup --protein P61981 --position 4
```

### Toolkit Integration

```bash
# Add offline AlphaMissense + monomeric FoldX scoring to the pipeline
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV> --interface --pae --enrich <ALIASES_FILE> --variants --protvar

# With custom AlphaMissense file path (--am-file defaults to data/stability/AlphaMissense_aa_substitutions.tsv)
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV> --interface --pae --enrich <ALIASES_FILE> --variants --protvar --am-file <AM_FILE>

# Full pipeline with all Phase D features
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV> --interface --pae --enrich <ALIASES_FILE> --clustering --variants --stability --protvar
```

---

## 10. Disease Annotations

Annotates proteins with disease associations, post-translational modifications (PTMs), Gene Ontology terms, and drug-target status from UniProt. Uses offline XML parsing of `uniprot_sprot_human.xml` (20,431 reviewed human entries) with API fallback for proteins missing from the local file. In the toolkit pipeline, `--disease` requires `--enrich`.

> **Note:** `--disease` defaults to `data/pathways/` when the path is omitted in the toolkit. The standalone CLI uses `--disease-dir` which also defaults to `data/pathways/`. Requires `uniprot_sprot_human.xml` in the disease directory.

### Flag Defaults (Standalone CLI)

| Flag | Default when omitted |
|---|---|
| `--disease-dir` | `data/pathways/` |

### Standalone CLI

```bash
# Disease annotation summary statistics (--disease-dir defaults to data/pathways/)
python disease_annotations.py summary --disease-dir <PATHWAYS_DIR>

# Look up annotations for a single protein
python disease_annotations.py lookup --disease-dir <PATHWAYS_DIR> --protein P04637

# Look up annotations for a drug target protein
python disease_annotations.py lookup --disease-dir <PATHWAYS_DIR> --protein Q2M2I8
```

### Toolkit Integration

```bash
# Add disease annotation to the pipeline (requires --enrich)
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV> --interface --pae --enrich <ALIASES_FILE> --disease

# With custom disease data directory
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV> --interface --pae --enrich <ALIASES_FILE> --disease <PATHWAYS_DIR>
```

---

## 11. Pathway Network Analysis

Maps proteins to Reactome pathways using local pathway files and performs per-pathway PPI enrichment analysis via the STRING API. Builds NetworkX interaction networks for visualisation. Requires `UniProt2Reactome_All_Levels.txt` and `ReactomePathwaysRelation.txt` in the pathways directory. In the toolkit pipeline, `--pathways` requires `--enrich`.

> **Note:** `--pathways` enables Reactome pathway mapping. STRING API enrichment is skipped with `--no-api`.

### Flag Defaults (Standalone CLI)

| Flag | Default when omitted |
|---|---|
| `--pathways-dir` | `data/pathways/` |
| `--output-dir` (network subcommand) | `Output/networks/` |
| `--min-pdockq` (network subcommand) | `0.23` |

### Standalone CLI

```bash
# Pathway summary statistics
python pathway_network.py summary --csv <OUTPUT_CSV>

# Build and plot pathway networks (--output-dir defaults to Output/networks/)
python pathway_network.py network --csv <OUTPUT_CSV> --output-dir <OUTPUT_DIR>

# Run enrichment analysis
python pathway_network.py enrichment --csv <OUTPUT_CSV>
```

### Toolkit Integration

```bash
# Add pathway mapping to the pipeline (requires --enrich)
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV> --interface --pae --enrich <ALIASES_FILE> --pathways

# Pathways without STRING API calls (offline-only)
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV> --interface --pae --enrich <ALIASES_FILE> --pathways --no-api
```

---

## 12. PyMOL Script Generation

Generates layered PyMOL `.pml` scripts for visualising predicted complexes, with chain colouring, pLDDT colouring, interface highlighting (sticks), variant highlighting (spheres with pathogenicity-aware sizing), and protvar overlay (transparency). Supports standalone single-PDB generation, batch generation from CSV, and toolkit pipeline integration. In the toolkit pipeline, `--pymol` requires `--interface --pae`.

### Flag Defaults

| Flag | Default when omitted |
|---|---|
| `--output` (standalone) | Current directory |
| `--min-tier` (standalone batch) / `--pymol-min-tier` (toolkit) | `High` |
| `--pymol-output` (toolkit) | `pymol_scripts/` |

### Standalone CLI

```bash
# Generate PyMOL script for a single PDB
python pymol_scripts.py generate --pdb <PDB_FILE>

# Generate with custom output directory (--output defaults to current directory)
python pymol_scripts.py generate --pdb <PDB_FILE> --output <PYMOL_OUTPUT>

# Generate with rendering commands (for pymol -c headless batch mode)
python pymol_scripts.py generate --pdb <PDB_FILE> --render

# Batch generate for all High-tier complexes from CSV (--min-tier defaults to High)
python pymol_scripts.py batch --csv <OUTPUT_CSV> --pdb-dir <PDB_DIR>

# Batch with custom tier threshold and output directory
python pymol_scripts.py batch --csv <OUTPUT_CSV> --pdb-dir <PDB_DIR> --min-tier Medium --output <PYMOL_OUTPUT>

# Batch without variant highlighting
python pymol_scripts.py batch --csv <OUTPUT_CSV> --pdb-dir <PDB_DIR> --no-variants
```

### Toolkit Integration

```bash
# Add PyMOL script generation to pipeline (requires --interface --pae; --pymol-min-tier defaults to High)
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV> --interface --pae --pymol

# With custom output directory (--pymol-output defaults to pymol_scripts/)
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV> --interface --pae --pymol --pymol-output <PYMOL_OUTPUT>

# With headless rendering enabled
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV> --interface --pae --pymol --pymol-render

# Filter to Medium+ tier (--pymol-min-tier defaults to High)
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV> --interface --pae --pymol --pymol-min-tier Medium

# Full pipeline with all features including PyMOL
python toolkit.py --dir <MODELS_DIR> --output <OUTPUT_CSV> --interface --pae --enrich <ALIASES_FILE> --clustering --variants --stability --protvar --disease --pathways --pymol
```

### Running Generated Scripts in PyMOL

```bash
# Interactive (opens PyMOL GUI)
pymol <PYMOL_OUTPUT>/A0A0B4J2C3_P24534.pml

# Headless batch rendering (requires --pymol-render or --render)
pymol -c <PYMOL_OUTPUT>/A0A0B4J2C3_P24534.pml
```

---

## 13. Data Dependency Validation

Pre-flight check that all required data files exist before starting the pipeline. Used automatically by `--full-pipeline`, or run standalone to diagnose missing files.

```bash
# Validate all data dependencies (all groups)
python data_registry.py

# Validate specific groups only
python data_registry.py --groups ppi-databases variant-mapping

# List all files with version strings in their names (risk on data updates)
python data_registry.py --versioned

# Override project root directory
python data_registry.py --root /path/to/project
```

### Available Groups

| Group | Description |
|---|---|
| `ppi-databases` | STRING links/aliases, BioGRID, HuRI, HuMAP |
| `clustering` | STRING protein clusters |
| `variant-mapping` | UniProt variants, ClinVar, ExAC constraint |
| `eve-stability` | EVE ID mapping + score CSVs |
| `offline-scoring` | AlphaMissense + AFDB FoldX export |
| `disease-pathways` | UniProt XML, Reactome mappings + hierarchy |
| `pymol` | PyMOL output directory (auto-created) |
