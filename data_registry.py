"""
Data dependency registry for the protein-complexes-toolkit.

Centralises every data-file path that the full pipeline (Phases A–F)
requires, and provides a pre-flight validation function so that
``--full-pipeline`` can check all inputs exist before starting a
multi-hour run.

Usage (standalone)::

    python data_registry.py                    # validate all phases
    python data_registry.py --phases A B C     # validate specific phases

Usage (from toolkit.py)::

    from data_registry import validate_data_dependencies, get_default_path
    errors = validate_data_dependencies(phases={"A", "B", "C", "D1", "D2", "E", "F"})
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ── Project root (same pattern as other modules) ─────────────────────
PROJECT_ROOT = Path(__file__).parent


# ── Dataclass ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DataDependency:
    """Metadata for one data file or directory required by the pipeline."""

    key: str                          # Short registry name
    expected_path: str                # Relative to project root
    required_by: tuple[str, ...]      # Module names that use it
    phase: str                        # Dependency group (e.g. "ppi-databases")
    has_version_in_name: bool         # True if filename has version string
    version_string: str | None        # The version portion, e.g. "v12.0"
    source_constant_name: str         # Constant/variable name in source
    source_file: str                  # Module filename
    source_line: int                  # Line number of constant definition
    is_directory: bool = False        # True for directories
    is_optional: bool = False         # True if pipeline continues without it
    is_output: bool = False           # True for auto-created output dirs


# ── Registry ──────────────────────────────────────────────────────────

DATA_DEPENDENCIES: dict[str, DataDependency] = {

    # ── PPI Databases + ID Mapping ───────────────────────────────────

    "string_aliases": DataDependency(
        key="string_aliases",
        expected_path="data/ppi/9606.protein.aliases.v12.0.txt",
        required_by=("id_mapper", "toolkit"),
        phase="ppi-databases",
        has_version_in_name=True,
        version_string="v12.0",
        source_constant_name="DEFAULT_ALIASES_PATH",
        source_file="id_mapper.py",
        source_line=77,
    ),
    "string_links": DataDependency(
        key="string_links",
        expected_path="data/ppi/9606.protein.links.v12.0.txt",
        required_by=("database_loaders", "toolkit"),
        phase="ppi-databases",
        has_version_in_name=True,
        version_string="v12.0",
        source_constant_name="STRING_LINKS_FILE",
        source_file="database_loaders.py",
        source_line=40,
    ),
    "biogrid": DataDependency(
        key="biogrid",
        expected_path="data/ppi/BIOGRID-ALL-5.0.253.tab3.txt",
        required_by=("database_loaders", "toolkit"),
        phase="ppi-databases",
        has_version_in_name=True,
        version_string="5.0.253",
        source_constant_name="BIOGRID_FILE",
        source_file="database_loaders.py",
        source_line=42,
    ),
    "huri": DataDependency(
        key="huri",
        expected_path="data/ppi/HuRI.tsv",
        required_by=("database_loaders", "toolkit"),
        phase="ppi-databases",
        has_version_in_name=False,
        version_string=None,
        source_constant_name="HURI_FILE",
        source_file="database_loaders.py",
        source_line=43,
    ),
    "humap": DataDependency(
        key="humap",
        expected_path="data/ppi/humap2_ppis_ACC_20200821.pairsWprob",
        required_by=("database_loaders", "toolkit"),
        phase="ppi-databases",
        has_version_in_name=True,
        version_string="20200821",
        source_constant_name="HUMAP_FILE",
        source_file="database_loaders.py",
        source_line=44,
    ),

    # ── Protein Clustering ──────────────────────────────────────────

    "string_clusters": DataDependency(
        key="string_clusters",
        expected_path="data/clusters/9606.clusters.proteins.v12.0.txt",
        required_by=("protein_clustering", "toolkit"),
        phase="clustering",
        has_version_in_name=True,
        version_string="v12.0",
        source_constant_name="STRING_CLUSTERS_FILE",
        source_file="protein_clustering.py",
        source_line=39,
    ),

    # ── Variant Mapping ─────────────────────────────────────────────

    "uniprot_variants": DataDependency(
        key="uniprot_variants",
        expected_path="data/variants/homo_sapiens_variation.txt",
        required_by=("variant_mapper", "toolkit"),
        phase="variant-mapping",
        has_version_in_name=False,
        version_string=None,
        source_constant_name="UNIPROT_VARIANTS_FILENAME",
        source_file="variant_mapper.py",
        source_line=60,
    ),
    "clinvar_variants": DataDependency(
        key="clinvar_variants",
        expected_path="data/variants/variant_summary.txt",
        required_by=("variant_mapper", "toolkit"),
        phase="variant-mapping",
        has_version_in_name=False,
        version_string=None,
        source_constant_name="CLINVAR_VARIANTS_FILENAME",
        source_file="variant_mapper.py",
        source_line=61,
        is_optional=True,
    ),
    "exac_constraint": DataDependency(
        key="exac_constraint",
        expected_path="data/variants/forweb_cleaned_exac_r03_march16_z_data_pLI_CNV-final.txt",
        required_by=("variant_mapper", "toolkit"),
        phase="variant-mapping",
        has_version_in_name=False,
        version_string=None,
        source_constant_name="EXAC_CONSTRAINT_FILENAME",
        source_file="variant_mapper.py",
        source_line=62,
        is_optional=True,
    ),

    # ── EVE Stability Scoring ────────────────────────────────────────

    "eve_idmapping": DataDependency(
        key="eve_idmapping",
        expected_path="data/stability/HUMAN_9606_idmapping.dat",
        required_by=("stability_scorer", "toolkit"),
        phase="eve-stability",
        has_version_in_name=False,
        version_string=None,
        source_constant_name="EVE_IDMAPPING_FILENAME",
        source_file="stability_scorer.py",
        source_line=44,
    ),
    "eve_data_dir": DataDependency(
        key="eve_data_dir",
        expected_path="data/stability/EVE_all_data",
        required_by=("stability_scorer", "toolkit"),
        phase="eve-stability",
        has_version_in_name=False,
        version_string=None,
        source_constant_name="DEFAULT_EVE_DIR",
        source_file="stability_scorer.py",
        source_line=43,
        is_directory=True,
    ),

    # ── Offline AlphaMissense + FoldX ────────────────────────────────

    "foldx_export": DataDependency(
        key="foldx_export",
        expected_path="data/stability/afdb_foldx_export_20250210.csv",
        required_by=("protvar_client", "toolkit"),
        phase="offline-scoring",
        has_version_in_name=True,
        version_string="20250210",
        source_constant_name="DEFAULT_FOLDX_EXPORT",
        source_file="protvar_client.py",
        source_line=43,
    ),
    "alphamissense": DataDependency(
        key="alphamissense",
        expected_path="data/stability/AlphaMissense_aa_substitutions.tsv",
        required_by=("protvar_client", "toolkit"),
        phase="offline-scoring",
        has_version_in_name=False,
        version_string=None,
        source_constant_name="DEFAULT_AM_FILE",
        source_file="protvar_client.py",
        source_line=46,
    ),

    # ── Disease + Pathway Annotation ─────────────────────────────────

    "uniprot_xml": DataDependency(
        key="uniprot_xml",
        expected_path="data/pathways/uniprot_sprot_human.xml",
        required_by=("disease_annotations", "toolkit"),
        phase="disease-pathways",
        has_version_in_name=False,
        version_string=None,
        source_constant_name="UNIPROT_XML_FILENAME",
        source_file="disease_annotations.py",
        source_line=42,
    ),
    "reactome_mappings": DataDependency(
        key="reactome_mappings",
        expected_path="data/pathways/UniProt2Reactome_All_Levels.txt",
        required_by=("pathway_network", "toolkit"),
        phase="disease-pathways",
        has_version_in_name=False,
        version_string=None,
        source_constant_name="REACTOME_MAPPINGS_FILENAME",
        source_file="pathway_network.py",
        source_line=50,
    ),
    "reactome_hierarchy": DataDependency(
        key="reactome_hierarchy",
        expected_path="data/pathways/ReactomePathwaysRelation.txt",
        required_by=("pathway_network",),
        phase="disease-pathways",
        has_version_in_name=False,
        version_string=None,
        source_constant_name="REACTOME_HIERARCHY_FILENAME",
        source_file="pathway_network.py",
        source_line=52,
        is_optional=True,
    ),

    # ── Output / Cache Directories (auto-created) ─────────────────────

    "string_api_cache": DataDependency(
        key="string_api_cache",
        expected_path="data/string_api_cache",
        required_by=("string_api",),
        phase="ppi-databases",
        has_version_in_name=False,
        version_string=None,
        source_constant_name="STRING_API_DEFAULT_CACHE_DIR",
        source_file="string_api.py",
        source_line=73,
        is_directory=True,
        is_output=True,
    ),
    "pymol_output": DataDependency(
        key="pymol_output",
        expected_path="pymol_scripts",
        required_by=("pymol_scripts", "toolkit"),
        phase="pymol",
        has_version_in_name=False,
        version_string=None,
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
    root = project_root or PROJECT_ROOT
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
    root = project_root or PROJECT_ROOT
    check_groups = groups or ALL_GROUPS
    errors: list[str] = []
    found: list[str] = []
    skipped: list[str] = []

    # Group entries by dependency group for structured output
    results_by_group: dict[str, list[tuple[DataDependency, str]]] = {}

    for key, dep in DATA_DEPENDENCIES.items():
        if dep.phase not in check_groups:
            continue

        if dep.is_output:
            skipped.append(key)
            continue

        full_path = root / dep.expected_path

        if dep.is_directory:
            exists = full_path.is_dir()
        else:
            exists = full_path.is_file()

        if exists:
            found.append(key)
            status = "OK"
        elif dep.is_optional:
            skipped.append(key)
            status = "SKIP"
        else:
            errors.append(
                f"Missing: {dep.expected_path}  "
                f"(required by {', '.join(dep.required_by)}, "
                f"defined as {dep.source_constant_name} in "
                f"{dep.source_file}:{dep.source_line})"
            )
            status = "FAIL"

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
                tag = {"OK": " OK ", "SKIP": "SKIP", "FAIL": "FAIL"}[status]
                suffix = ""
                if status == "SKIP":
                    suffix = "  (optional)"
                elif status == "FAIL":
                    suffix = (f"  <- required by {', '.join(dep.required_by)}"
                              f"  ({dep.source_constant_name} in "
                              f"{dep.source_file}:{dep.source_line})")
                print(f"    [{tag}] {dep.expected_path}{suffix}",
                      file=sys.stderr)

        total_checked = len(found) + len(errors)
        print(f"\n  Summary: {len(found)}/{total_checked} found, "
              f"{len(errors)} missing, {len(skipped)} skipped",
              file=sys.stderr)
        if not errors:
            print("  All required data files present.", file=sys.stderr)

    return errors


def get_versioned_files() -> list[DataDependency]:
    """Return all registry entries whose filenames contain version strings."""
    return [dep for dep in DATA_DEPENDENCIES.values() if dep.has_version_in_name]


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
    parser.add_argument(
        "--versioned", action="store_true",
        help="List all files with version strings in their names.",
    )
    args = parser.parse_args()

    root = Path(args.root) if args.root else PROJECT_ROOT

    if args.versioned:
        print("Files with version strings (risk on data updates):")
        for dep in get_versioned_files():
            print(f"  {dep.expected_path}  version={dep.version_string}  "
                  f"({dep.source_constant_name} in {dep.source_file}:{dep.source_line})")
        return

    groups = set(args.groups) if args.groups else None
    errors = validate_data_dependencies(project_root=root, groups=groups)

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
