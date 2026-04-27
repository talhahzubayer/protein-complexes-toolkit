"""
Data dependency registry for the protein-complexes-toolkit.

Centralises every data-file path that the full pipeline requires,
and provides a pre-flight validation function so that
``--full-pipeline`` can check all inputs exist before starting a
multi-hour run.

Usage (standalone)::

    python data_registry.py                                          # validate all groups
    python data_registry.py --groups ppi-databases variant-mapping   # validate specific groups

Usage (from toolkit.py)::

    from data_registry import validate_data_dependencies, get_default_path
    errors = validate_data_dependencies(groups={"ppi-databases", "variant-mapping"})
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# ── Project root (same pattern as other modules) ─────────────────────
PROJECT_ROOT = Path(__file__).parent


def _resolve_project_root(project_root: Path | str | None = None) -> Path:
    """explicit > PROTEIN_TOOLKIT_PROJECT_ROOT > PROJECT_ROOT (repo fallback)."""
    if project_root is not None:
        path = Path(project_root).expanduser().resolve()
        source = "explicit project_root"
    else:
        env = os.environ.get("PROTEIN_TOOLKIT_PROJECT_ROOT")
        if env:
            path = Path(env).expanduser().resolve()
            source = "PROTEIN_TOOLKIT_PROJECT_ROOT"
        else:
            return PROJECT_ROOT

    if not path.is_dir():
        raise FileNotFoundError(
            f"{source} points to {str(path)!r}, but directory does not exist"
        )
    return path


# ── Dataclass ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DataDependency:
    """Metadata for one data file or directory required by the pipeline."""

    expected_path: str                # Relative to project root
    phase: str                        # Dependency group (e.g. "ppi-databases")
    source_constant_name: str         # Constant/variable name in source
    source_file: str                  # Module filename
    source_line: int                  # Line number of constant definition
    is_directory: bool = False        # True for directories
    is_output: bool = False           # True for auto-created output dirs


# ── Registry ──────────────────────────────────────────────────────────

DATA_DEPENDENCIES: dict[str, DataDependency] = {

    # ── PPI Databases + ID Mapping ───────────────────────────────────

    "string_aliases": DataDependency(
        expected_path="data/ppi/9606.protein.aliases.v12.0.txt",
        phase="ppi-databases",
        source_constant_name="DEFAULT_ALIASES_PATH",
        source_file="id_mapper.py",
        source_line=77,
    ),
    "string_links": DataDependency(
        expected_path="data/ppi/9606.protein.links.v12.0.txt",
        phase="ppi-databases",
        source_constant_name="STRING_LINKS_FILE",
        source_file="database_loaders.py",
        source_line=40,
    ),
    "biogrid": DataDependency(
        expected_path="data/ppi/BIOGRID-ALL-5.0.253.tab3.txt",
        phase="ppi-databases",
        source_constant_name="BIOGRID_FILE",
        source_file="database_loaders.py",
        source_line=42,
    ),
    "huri": DataDependency(
        expected_path="data/ppi/HuRI.tsv",
        phase="ppi-databases",
        source_constant_name="HURI_FILE",
        source_file="database_loaders.py",
        source_line=43,
    ),
    "humap": DataDependency(
        expected_path="data/ppi/humap2_ppis_ACC_20200821.pairsWprob",
        phase="ppi-databases",
        source_constant_name="HUMAP_FILE",
        source_file="database_loaders.py",
        source_line=44,
    ),

    # ── Protein Clustering ──────────────────────────────────────────

    "string_clusters": DataDependency(
        expected_path="data/clusters/9606.clusters.proteins.v12.0.txt",
        phase="clustering",
        source_constant_name="STRING_CLUSTERS_FILE",
        source_file="protein_clustering.py",
        source_line=39,
    ),

    # ── Variant Mapping ─────────────────────────────────────────────

    "uniprot_variants": DataDependency(
        expected_path="data/variants/homo_sapiens_variation.txt",
        phase="variant-mapping",
        source_constant_name="UNIPROT_VARIANTS_FILENAME",
        source_file="variant_mapper.py",
        source_line=60,
    ),
    "clinvar_variants": DataDependency(
        expected_path="data/variants/variant_summary.txt",
        phase="variant-mapping",
        source_constant_name="CLINVAR_VARIANTS_FILENAME",
        source_file="variant_mapper.py",
        source_line=61,
    ),
    "exac_constraint": DataDependency(
        expected_path="data/variants/forweb_cleaned_exac_r03_march16_z_data_pLI_CNV-final.txt",
        phase="variant-mapping",
        source_constant_name="EXAC_CONSTRAINT_FILENAME",
        source_file="variant_mapper.py",
        source_line=62,
    ),

    # ── EVE Stability Scoring ────────────────────────────────────────

    "eve_idmapping": DataDependency(
        expected_path="data/stability/HUMAN_9606_idmapping.dat",
        phase="eve-stability",
        source_constant_name="EVE_IDMAPPING_FILENAME",
        source_file="stability_scorer.py",
        source_line=44,
    ),
    "eve_data_dir": DataDependency(
        expected_path="data/stability/EVE_all_data",
        phase="eve-stability",
        source_constant_name="DEFAULT_EVE_DIR",
        source_file="stability_scorer.py",
        source_line=43,
        is_directory=True,
    ),

    # ── Offline AlphaMissense + FoldX ────────────────────────────────

    "foldx_export": DataDependency(
        expected_path="data/stability/afdb_foldx_export_20250210.csv",
        phase="offline-scoring",
        source_constant_name="DEFAULT_FOLDX_EXPORT",
        source_file="protvar_client.py",
        source_line=43,
    ),
    "alphamissense": DataDependency(
        expected_path="data/stability/AlphaMissense_aa_substitutions.tsv",
        phase="offline-scoring",
        source_constant_name="DEFAULT_AM_FILE",
        source_file="protvar_client.py",
        source_line=46,
    ),

    # ── Disease + Pathway Annotation ─────────────────────────────────

    "uniprot_xml": DataDependency(
        expected_path="data/pathways/uniprot_sprot_human.xml",
        phase="disease-pathways",
        source_constant_name="UNIPROT_XML_FILENAME",
        source_file="disease_annotations.py",
        source_line=42,
    ),
    "reactome_mappings": DataDependency(
        expected_path="data/pathways/UniProt2Reactome_All_Levels.txt",
        phase="disease-pathways",
        source_constant_name="REACTOME_MAPPINGS_FILENAME",
        source_file="pathway_network.py",
        source_line=50,
    ),
    "reactome_hierarchy": DataDependency(
        expected_path="data/pathways/ReactomePathwaysRelation.txt",
        phase="disease-pathways",
        source_constant_name="REACTOME_HIERARCHY_FILENAME",
        source_file="pathway_network.py",
        source_line=52,
    ),

    # ── Output / Cache Directories (auto-created) ─────────────────────

    "string_api_cache": DataDependency(
        expected_path="data/string_api_cache",
        phase="ppi-databases",
        source_constant_name="STRING_API_DEFAULT_CACHE_DIR",
        source_file="string_api.py",
        source_line=73,
        is_directory=True,
        is_output=True,
    ),
    "pymol_output": DataDependency(
        expected_path="pymol_scripts",
        phase="pymol",
        source_constant_name="DEFAULT_PYMOL_OUTPUT_DIR",
        source_file="pymol_scripts.py",
        source_line=45,
        is_directory=True,
        is_output=True,
    ),
}

# All dependency groups (in pipeline execution order)
ALL_GROUPS = frozenset({
    "ppi-databases", "clustering", "variant-mapping",
    "offline-scoring", "eve-stability", "disease-pathways", "pymol",
})

# Ordered list for display
_GROUP_ORDER = (
    "ppi-databases", "clustering", "variant-mapping",
    "eve-stability", "offline-scoring", "disease-pathways", "pymol",
)

# Display headings for each group
_GROUP_HEADINGS = {
    "ppi-databases":   "PPI Databases + ID Mapping",
    "clustering":      "Protein Clustering",
    "variant-mapping": "Variant Mapping",
    "eve-stability":   "EVE Stability Scoring",
    "offline-scoring": "Offline AlphaMissense + FoldX",
    "disease-pathways": "Disease + Pathway Annotation",
    "pymol":           "PyMOL Script Generation",
}


# ── Public helpers ────────────────────────────────────────────────────

def get_default_path(key: str, project_root: Path | None = None) -> str:
    """Resolve a registry key to an absolute path string.

    Parameters
    ----------
    key : str
        Registry key (e.g. ``"string_aliases"``).
    project_root : Path, optional
        Override for the project root directory.  Defaults to the
        directory containing this file.

    Returns
    -------
    str
        Absolute path as a string, suitable for passing to argparse.
    """
    dep = DATA_DEPENDENCIES[key]
    root = _resolve_project_root(project_root)
    return str(root / dep.expected_path)


def validate_data_dependencies(
    project_root: Path | None = None,
    groups: set[str] | None = None,
    verbose: bool = True,
) -> list[str]:
    """Check that all required data files exist before pipeline start.

    Parameters
    ----------
    project_root : Path, optional
        Override for the project root.  Defaults to ``PROJECT_ROOT``.
    groups : set of str, optional
        Which dependency groups to check (e.g. ``{"ppi-databases",
        "variant-mapping"}``).  ``None`` means all groups.
    verbose : bool
        If ``True``, print a summary table to stderr.

    Returns
    -------
    list of str
        Error messages for missing required files.  Empty list = all OK.
    """
    root = _resolve_project_root(project_root)
    check_groups = groups or ALL_GROUPS
    errors: list[str] = []
    found: list[str] = []

    # Group entries by dependency group for structured output
    results_by_group: dict[str, list[tuple[DataDependency, str]]] = {}

    for dep in DATA_DEPENDENCIES.values():
        if dep.phase not in check_groups:
            continue

        if dep.is_output:
            continue

        full_path = root / dep.expected_path

        if dep.is_directory:
            exists = full_path.is_dir()
        else:
            exists = full_path.is_file()

        if exists:
            found.append(dep.expected_path)
            status = "OK"
        else:
            errors.append(
                f"Missing: {dep.expected_path}  "
                f"(defined as {dep.source_constant_name} in "
                f"{dep.source_file}:{dep.source_line})"
            )
            status = "MISSING"

        results_by_group.setdefault(dep.phase, []).append((dep, status))

    if verbose:
        # Print per-group table so the user sees the full picture
        for group in _GROUP_ORDER:
            entries = results_by_group.get(group)
            if not entries:
                continue
            heading = _GROUP_HEADINGS.get(group, group)
            print(f"\n  {heading}:", file=sys.stderr)
            for dep, status in entries:
                if status == "OK":
                    print(f"    [ OK ] {dep.expected_path}",
                          file=sys.stderr)
                else:
                    # Rich multi-line MISSING message
                    group_label = _GROUP_HEADINGS.get(dep.phase, dep.phase)
                    print(f"    [MISSING] {dep.expected_path}",
                          file=sys.stderr)
                    print(f"      Required by: {dep.source_file} "
                          f"({group_label})", file=sys.stderr)
                    print(f'      See README section: "Setting Up Data"',
                          file=sys.stderr)
                    filename = dep.expected_path.rsplit("/", 1)[-1]
                    print(f"      If you have a newer version of this "
                          f"file, update the filename in:",
                          file=sys.stderr)
                    print(f"        -> {dep.source_file}, line "
                          f"{dep.source_line}: "
                          f'{dep.source_constant_name} = "{filename}"',
                          file=sys.stderr)
                    print(file=sys.stderr)  # blank line between entries

        total_checked = len(found) + len(errors)
        print(f"\n  Summary: {len(found)}/{total_checked} found, "
              f"{len(errors)} missing",
              file=sys.stderr)
        if not errors:
            print("  All required data files present.", file=sys.stderr)

    return errors


# ── Standalone CLI ────────────────────────────────────────────────────

def main() -> None:
    """Validate data dependencies from the command line."""
    parser = argparse.ArgumentParser(
        description="Check that all required data files exist for the pipeline.",
    )
    parser.add_argument(
        "--groups", nargs="*", default=None,
        choices=sorted(ALL_GROUPS),
        help="Dependency groups to check (default: all). "
             "E.g. --groups ppi-databases variant-mapping",
    )
    parser.add_argument(
        "--root", default=None,
        help="Override project root directory (default: auto-detect).",
    )
    args = parser.parse_args()

    groups = set(args.groups) if args.groups else None
    errors = validate_data_dependencies(project_root=args.root, groups=groups)

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
