"""
Disease & biological annotation module for the protein complexes toolkit.

Parses UniProt SwissProt XML to extract disease associations, post-translational
modifications (PTM), Gene Ontology terms, KEGG pathway identifiers, and drug
target status for proteins in the pipeline. Supports offline-first annotation
from a local XML file with optional UniProt REST API fallback for missing entries.

Data source
-----------
``data/pathways/uniprot_sprot_human.xml`` — full UniProt SwissProt reviewed human
proteome (20,431 entries, ~1.1 GB). Parsed via ``xml.etree.ElementTree.iterparse``
for streaming (low peak memory).

Usage
-----
Standalone CLI::

    python disease_annotations.py summary
    python disease_annotations.py lookup --protein P04637

Toolkit integration::

    python toolkit.py Test_Data/ -o results.csv --interface --pae --enrich aliases.txt --disease
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Union

# ── Constants ─────────────────────────────────────────────────────

DEFAULT_DISEASE_DIR = Path(__file__).parent / "data" / "pathways"
UNIPROT_XML_FILENAME = "uniprot_sprot_human.xml"

DETAILS_DISPLAY_LIMIT = 50  # Max items before "(+N more)" truncation

UNIPROT_XML_NAMESPACE = "https://uniprot.org/uniprot"

# PTM feature types to extract from UniProt XML
PTM_FEATURE_TYPES = frozenset({
    "modified residue",
    "cross-link",
    "glycosylation site",
    "lipidation",
})

# Drug target keyword in UniProt
DRUG_TARGET_KEYWORD = "Pharmaceutical"

# UniProt REST API base URL for fallback queries
UNIPROT_API_BASE = "https://rest.uniprot.org/uniprotkb"
UNIPROT_API_TIMEOUT = 15  # seconds

CSV_FIELDNAMES_DISEASE = [
    'n_diseases_a', 'n_diseases_b',
    'disease_details_a', 'disease_details_b',
    'is_drug_target_a', 'is_drug_target_b',
    'n_ptm_sites_a', 'n_ptm_sites_b',
    'ptm_details_a', 'ptm_details_b',
    'go_biological_process_a', 'go_biological_process_b',
    'go_molecular_function_a', 'go_molecular_function_b',
]


# ── XML Parsing Helpers ──────────────────────────────────────────

def _parse_entry_diseases(entry_elem: ET.Element, ns: str) -> list[dict]:
    """Extract disease associations from ``<comment type="disease">`` elements.

    Returns
    -------
    list[dict]
        Each dict has keys: ``disease_name``, ``acronym``, ``omim_id``.
    """
    diseases: list[dict] = []
    for comment in entry_elem.findall(f"{ns}comment"):
        if comment.get("type") != "disease":
            continue
        disease_elem = comment.find(f"{ns}disease")
        if disease_elem is None:
            continue
        name_elem = disease_elem.find(f"{ns}name")
        acronym_elem = disease_elem.find(f"{ns}acronym")
        dbref_elem = disease_elem.find(f"{ns}dbReference")
        omim_id = ""
        if dbref_elem is not None and dbref_elem.get("type") == "MIM":
            omim_id = dbref_elem.get("id", "")
        diseases.append({
            "disease_name": name_elem.text if name_elem is not None and name_elem.text else "",
            "acronym": acronym_elem.text if acronym_elem is not None and acronym_elem.text else "",
            "omim_id": omim_id,
        })
    return diseases


def _parse_entry_ptm_features(entry_elem: ET.Element, ns: str) -> list[dict]:
    """Extract PTM sites from ``<feature>`` elements.

    Returns
    -------
    list[dict]
        Each dict has keys: ``type``, ``position``, ``description``.
    """
    ptm_sites: list[dict] = []
    for feature in entry_elem.findall(f"{ns}feature"):
        ftype = feature.get("type", "")
        if ftype not in PTM_FEATURE_TYPES:
            continue
        description = feature.get("description", "")
        position = ""
        location = feature.find(f"{ns}location")
        if location is not None:
            pos_elem = location.find(f"{ns}position")
            if pos_elem is not None:
                position = pos_elem.get("position", "")
            else:
                # Range PTM (e.g. cross-links)
                begin = location.find(f"{ns}begin")
                end = location.find(f"{ns}end")
                if begin is not None and end is not None:
                    position = f"{begin.get('position', '?')}-{end.get('position', '?')}"
        ptm_sites.append({
            "type": ftype,
            "position": position,
            "description": description,
        })
    return ptm_sites


def _parse_entry_go_terms(entry_elem: ET.Element, ns: str) -> list[dict]:
    """Extract Gene Ontology terms from ``<dbReference type="GO">`` elements.

    Returns
    -------
    list[dict]
        Each dict has keys: ``go_id``, ``go_name``, ``aspect``
        (F=molecular function, P=biological process, C=cellular component).
    """
    go_terms: list[dict] = []
    for dbref in entry_elem.findall(f"{ns}dbReference"):
        if dbref.get("type") != "GO":
            continue
        go_id = dbref.get("id", "")
        go_name = ""
        aspect = ""
        for prop in dbref.findall(f"{ns}property"):
            if prop.get("type") == "term":
                value = prop.get("value", "")
                if value and len(value) > 2 and value[1] == ":":
                    aspect = value[0]  # F, P, or C
                    go_name = value[2:]
                else:
                    go_name = value
        go_terms.append({
            "go_id": go_id,
            "go_name": go_name,
            "aspect": aspect,
        })
    return go_terms


def _parse_entry_kegg_ids(entry_elem: ET.Element, ns: str) -> list[str]:
    """Extract KEGG pathway IDs from ``<dbReference type="KEGG">``."""
    kegg_ids: list[str] = []
    for dbref in entry_elem.findall(f"{ns}dbReference"):
        if dbref.get("type") == "KEGG":
            kid = dbref.get("id", "")
            if kid:
                kegg_ids.append(kid)
    return kegg_ids


def _detect_drug_target(entry_elem: ET.Element, ns: str) -> bool:
    """Check if the entry has the ``Pharmaceutical`` keyword."""
    for kw in entry_elem.findall(f"{ns}keyword"):
        if kw.text == DRUG_TARGET_KEYWORD:
            return True
    return False


def _parse_entry(entry_elem: ET.Element, ns: str) -> dict:
    """Parse a single UniProt XML ``<entry>`` into an annotation dict."""
    return {
        "diseases": _parse_entry_diseases(entry_elem, ns),
        "ptm_sites": _parse_entry_ptm_features(entry_elem, ns),
        "go_terms": _parse_entry_go_terms(entry_elem, ns),
        "kegg_ids": _parse_entry_kegg_ids(entry_elem, ns),
        "is_drug_target": _detect_drug_target(entry_elem, ns),
    }


# ── Annotation Loading ──────────────────────────────────────────

def load_uniprot_annotations(
    filepath: Union[str, Path],
    accessions: frozenset[str],
    verbose: bool = False,
) -> dict[str, dict]:
    """Stream-parse UniProt XML, filtering to pipeline accessions.

    Uses ``iterparse`` to avoid loading the entire 1.1 GB file into memory.
    Only entries whose primary or secondary accession is in *accessions*
    are parsed and retained.

    Parameters
    ----------
    filepath : str or Path
        Path to ``uniprot_sprot_human.xml``.
    accessions : frozenset[str]
        Set of UniProt accessions to retain.
    verbose : bool
        Print progress to stderr.

    Returns
    -------
    dict[str, dict]
        Mapping ``{accession: annotation_dict}``.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"UniProt XML not found: {filepath}")

    ns = f"{{{UNIPROT_XML_NAMESPACE}}}"
    index: dict[str, dict] = {}
    n_scanned = 0

    if verbose:
        print(f"  Loading UniProt annotations from: {filepath.name}", file=sys.stderr)
        print(f"  Filtering to {len(accessions):,} target accessions", file=sys.stderr)

    for event, elem in ET.iterparse(str(filepath), events=("end",)):
        if elem.tag != f"{ns}entry":
            continue
        n_scanned += 1

        # Check all accessions on this entry (primary + secondary)
        entry_accs = [a.text for a in elem.findall(f"{ns}accession") if a.text]
        matched = set(entry_accs) & accessions
        if matched:
            annotation = _parse_entry(elem, ns)
            # Index under all matched accessions
            for acc in matched:
                index[acc] = annotation

        elem.clear()

        if verbose and n_scanned % 5000 == 0:
            print(f"    ...scanned {n_scanned:,} entries, "
                  f"matched {len(index):,}", file=sys.stderr)

    if verbose:
        print(f"  Scanned {n_scanned:,} entries, "
              f"matched {len(index):,} accessions", file=sys.stderr)

    return index


# ── UniProt REST API Fallback ────────────────────────────────────

def fetch_uniprot_annotation_api(accession: str) -> Optional[dict]:
    """Query UniProt REST API for a single protein's annotations.

    Used as fallback when a protein is missing from the local XML file.

    Parameters
    ----------
    accession : str
        UniProt accession (e.g. ``P04637``).

    Returns
    -------
    dict or None
        Annotation dict matching ``_parse_entry`` format, or None on failure.
    """
    url = f"{UNIPROT_API_BASE}/{accession}.json"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=UNIPROT_API_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError,
            TimeoutError, OSError):
        return None

    # Parse JSON response into our standard format
    diseases: list[dict] = []
    for comment in data.get("comments", []):
        if comment.get("commentType") != "DISEASE":
            continue
        # Try structured disease object (legacy API schema)
        disease = comment.get("disease", {})
        disease_name = disease.get("diseaseId", "")
        acronym = disease.get("acronym", "")
        omim_id = ""
        for xref in disease.get("dbReferences", []):
            if xref.get("type") == "MIM":
                omim_id = xref.get("id", "")
        # Current API schema: disease info in note.texts[].value
        if not disease_name and not omim_id:
            note = comment.get("note", {})
            for text_entry in note.get("texts", []):
                val = text_entry.get("value", "").strip()
                if val:
                    disease_name = val
                    break  # take first note text as disease description
        # Only record if we found meaningful data
        if disease_name or omim_id:
            diseases.append({
                "disease_name": disease_name,
                "acronym": acronym,
                "omim_id": omim_id,
            })

    ptm_sites: list[dict] = []
    for feat in data.get("features", []):
        ftype = feat.get("type", "")
        # JSON uses different type names than XML
        if ftype in ("Modified residue", "Cross-link", "Glycosylation", "Lipidation"):
            location = feat.get("location", {})
            position = ""
            if "position" in location:
                raw_pos = location["position"]
                position = str(raw_pos.get("value", "")) if isinstance(raw_pos, dict) else str(raw_pos)
            elif "start" in location and "end" in location:
                start = location["start"]
                end = location["end"]
                if isinstance(start, dict):
                    start = start.get("value", "")
                if isinstance(end, dict):
                    end = end.get("value", "")
                position = str(start) if start == end else f"{start}-{end}"
            ptm_sites.append({
                "type": ftype.lower().replace(" ", "_"),
                "position": position,
                "description": feat.get("description", ""),
            })

    go_terms: list[dict] = []
    kegg_ids: list[str] = []
    for xref in data.get("uniProtKBCrossReferences", []):
        if xref.get("database") == "GO":
            go_id = xref.get("id", "")
            go_name = ""
            aspect = ""
            for prop in xref.get("properties", []):
                if prop.get("key") == "GoTerm":
                    val = prop.get("value", "")
                    if val and len(val) > 2 and val[1] == ":":
                        aspect = val[0]
                        go_name = val[2:]
                    else:
                        go_name = val
            go_terms.append({"go_id": go_id, "go_name": go_name, "aspect": aspect})
        elif xref.get("database") == "KEGG":
            kid = xref.get("id", "")
            if kid:
                kegg_ids.append(kid)

    is_drug_target = False
    for kw in data.get("keywords", []):
        if kw.get("name") == DRUG_TARGET_KEYWORD:
            is_drug_target = True
            break

    return {
        "diseases": diseases,
        "ptm_sites": ptm_sites,
        "go_terms": go_terms,
        "kegg_ids": kegg_ids,
        "is_drug_target": is_drug_target,
    }


# ── Formatting ───────────────────────────────────────────────────

def format_disease_details(
    diseases: list[dict],
    limit: int = DETAILS_DISPLAY_LIMIT,
) -> str:
    """Format disease associations as a pipe-separated string.

    Format: ``OMIM:618428:Popov-Chang syndrome (POPCHAS)|...``

    Parameters
    ----------
    diseases : list[dict]
        Disease dicts from ``_parse_entry_diseases``.
    limit : int
        Maximum entries before truncation.

    Returns
    -------
    str
        Pipe-separated disease details, or empty string.
    """
    if not diseases:
        return ""
    parts: list[str] = []
    for d in diseases[:limit]:
        omim = d.get("omim_id", "")
        name = d.get("disease_name", "")
        acronym = d.get("acronym", "")
        label = f"{name} ({acronym})" if acronym else name
        if omim:
            parts.append(f"OMIM:{omim}:{label}")
        else:
            parts.append(label)
    result = "|".join(parts)
    if len(diseases) > limit:
        result += f"|...(+{len(diseases) - limit} more)"
    return result


def format_ptm_details(
    ptm_sites: list[dict],
    limit: int = DETAILS_DISPLAY_LIMIT,
) -> str:
    """Format PTM sites as a pipe-separated string.

    Format: ``Phosphoserine:S9|Cross-link:K48|...``

    Parameters
    ----------
    ptm_sites : list[dict]
        PTM dicts from ``_parse_entry_ptm_features``.
    limit : int
        Maximum entries before truncation.

    Returns
    -------
    str
        Pipe-separated PTM details, or empty string.
    """
    if not ptm_sites:
        return ""
    parts: list[str] = []
    for p in ptm_sites[:limit]:
        desc = p.get("description", p.get("type", ""))
        pos = p.get("position", "")
        if pos:
            parts.append(f"{desc}:{pos}")
        else:
            parts.append(desc)
    result = "|".join(parts)
    if len(ptm_sites) > limit:
        result += f"|...(+{len(ptm_sites) - limit} more)"
    return result


def format_go_details(
    go_terms: list[dict],
    aspect_filter: str = "",
    limit: int = DETAILS_DISPLAY_LIMIT,
) -> str:
    """Format GO terms as a pipe-separated string.

    Format: ``GO:0005515:protein binding|GO:0006915:apoptotic process|...``

    Parameters
    ----------
    go_terms : list[dict]
        GO dicts from ``_parse_entry_go_terms``.
    aspect_filter : str
        If set, filter to this aspect only (``F``, ``P``, or ``C``).
    limit : int
        Maximum entries before truncation.

    Returns
    -------
    str
        Pipe-separated GO details, or empty string.
    """
    if not go_terms:
        return ""
    filtered = go_terms
    if aspect_filter:
        filtered = [g for g in go_terms if g.get("aspect") == aspect_filter]
    if not filtered:
        return ""
    parts: list[str] = []
    for g in filtered[:limit]:
        go_id = g.get("go_id", "")
        go_name = g.get("go_name", "")
        parts.append(f"{go_id}:{go_name}")
    result = "|".join(parts)
    if len(filtered) > limit:
        result += f"|...(+{len(filtered) - limit} more)"
    return result


# ── Annotation ───────────────────────────────────────────────────

def _empty_annotation() -> dict:
    """Return an empty annotation dict for proteins with no data."""
    return {
        "diseases": [],
        "ptm_sites": [],
        "go_terms": [],
        "kegg_ids": [],
        "is_drug_target": False,
    }


def _lookup_annotation(
    accession: str,
    annotation_index: dict[str, dict],
    api_fallback: bool = True,
    _api_cache: Optional[dict] = None,
) -> dict:
    """Look up annotations for a protein, with optional API fallback.

    Strips isoform suffix before lookup (e.g. ``P61981-2`` → ``P61981``).
    """
    # Try exact match first
    if accession in annotation_index:
        return annotation_index[accession]
    # Strip isoform suffix
    base = accession.split("-")[0] if "-" in accession else accession
    if base in annotation_index:
        return annotation_index[base]
    # API fallback
    if api_fallback:
        if _api_cache is not None and base in _api_cache:
            return _api_cache[base]
        result = fetch_uniprot_annotation_api(base)
        if result is not None:
            if _api_cache is not None:
                _api_cache[base] = result
            annotation_index[base] = result  # Cache for future lookups
            return result
    return _empty_annotation()


def annotate_results_with_disease(
    results: list[dict],
    annotation_index: dict[str, dict],
    api_fallback: bool = True,
    verbose: bool = False,
) -> None:
    """Annotate result rows with disease, PTM, GO, and drug target data.

    Modifies *results* **in-place** — no return value.

    Parameters
    ----------
    results : list[dict]
        Per-complex result dicts from the pipeline.
    annotation_index : dict[str, dict]
        Mapping ``{accession: annotation_dict}`` from ``load_uniprot_annotations``.
    api_fallback : bool
        If True, query UniProt REST API for proteins missing from the local index.
    verbose : bool
        Print progress to stderr.
    """
    if verbose:
        print(f"  Annotating {len(results)} complexes with disease data...",
              file=sys.stderr)
    api_cache: dict[str, dict] = {}
    api_misses = 0

    # Lazy import avoids circular dependency at module load time.
    from toolkit import is_annotatable

    for i, row in enumerate(results):
        protein_a = row.get("protein_a", "")
        protein_b = row.get("protein_b", "")

        # Non-human rows: leave all disease columns empty and skip UniProt
        # XML / API lookups (they'd return nothing anyway). TrEMBL-human rows
        # still run lookup because the XML/API carries records for them.
        if not is_annotatable(row):
            row["n_diseases_a"] = 0
            row["n_diseases_b"] = 0
            row["disease_details_a"] = ""
            row["disease_details_b"] = ""
            row["is_drug_target_a"] = False
            row["is_drug_target_b"] = False
            row["n_ptm_sites_a"] = 0
            row["n_ptm_sites_b"] = 0
            row["ptm_details_a"] = ""
            row["ptm_details_b"] = ""
            row["go_biological_process_a"] = ""
            row["go_biological_process_b"] = ""
            row["go_molecular_function_a"] = ""
            row["go_molecular_function_b"] = ""
            continue

        ann_a = _lookup_annotation(protein_a, annotation_index,
                                   api_fallback, api_cache)
        ann_b = _lookup_annotation(protein_b, annotation_index,
                                   api_fallback, api_cache)

        # Disease associations
        row["n_diseases_a"] = len(ann_a["diseases"])
        row["n_diseases_b"] = len(ann_b["diseases"])
        row["disease_details_a"] = format_disease_details(ann_a["diseases"])
        row["disease_details_b"] = format_disease_details(ann_b["diseases"])

        # Drug target status
        row["is_drug_target_a"] = ann_a["is_drug_target"]
        row["is_drug_target_b"] = ann_b["is_drug_target"]

        # PTM sites
        row["n_ptm_sites_a"] = len(ann_a["ptm_sites"])
        row["n_ptm_sites_b"] = len(ann_b["ptm_sites"])
        row["ptm_details_a"] = format_ptm_details(ann_a["ptm_sites"])
        row["ptm_details_b"] = format_ptm_details(ann_b["ptm_sites"])

        # GO terms (split by aspect)
        row["go_biological_process_a"] = format_go_details(
            ann_a["go_terms"], aspect_filter="P")
        row["go_biological_process_b"] = format_go_details(
            ann_b["go_terms"], aspect_filter="P")
        row["go_molecular_function_a"] = format_go_details(
            ann_a["go_terms"], aspect_filter="F")
        row["go_molecular_function_b"] = format_go_details(
            ann_b["go_terms"], aspect_filter="F")

        if verbose and (i + 1) % 1000 == 0:
            print(f"    ...annotated {i + 1:,}/{len(results):,} complexes",
                  file=sys.stderr)

    if verbose:
        n_with_disease = sum(
            1 for r in results
            if r.get("n_diseases_a", 0) > 0 or r.get("n_diseases_b", 0) > 0
        )
        n_drug_targets = sum(
            1 for r in results
            if r.get("is_drug_target_a") or r.get("is_drug_target_b")
        )
        print(f"  Disease annotation: {n_with_disease} complexes with disease "
              f"associations, {n_drug_targets} with drug targets", file=sys.stderr)
        if api_cache:
            print(f"  API fallback: {len(api_cache)} proteins resolved via API",
                  file=sys.stderr)


# ── Standalone CLI ───────────────────────────────────────────────

def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="disease_annotations",
        description="Disease and biological annotation for protein complexes.",
    )
    parser.add_argument(
        "--disease-dir", type=str,
        default=str(DEFAULT_DISEASE_DIR),
        help=f"Path to directory containing {UNIPROT_XML_FILENAME}. "
             f"Default: {DEFAULT_DISEASE_DIR}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    # summary
    subparsers.add_parser(
        "summary",
        help="Print annotation data coverage statistics.",
    )

    # lookup
    sub_lookup = subparsers.add_parser(
        "lookup",
        help="Query annotations for a specific protein.",
    )
    sub_lookup.add_argument(
        "--protein", required=True,
        help="UniProt accession (e.g. P04637).",
    )

    return parser


def _cli_summary(disease_dir: str) -> None:
    """Print summary statistics for the annotation data."""
    xml_path = Path(disease_dir) / UNIPROT_XML_FILENAME
    if not xml_path.exists():
        print(f"Error: {xml_path} not found", file=sys.stderr)
        sys.exit(1)

    ns = f"{{{UNIPROT_XML_NAMESPACE}}}"
    total = 0
    with_disease = 0
    with_pharma = 0
    with_ptm = 0
    with_go = 0

    print(f"Scanning {xml_path.name}...")
    for event, elem in ET.iterparse(str(xml_path), events=("end",)):
        if elem.tag != f"{ns}entry":
            continue
        total += 1
        if any(c.get("type") == "disease"
               for c in elem.findall(f"{ns}comment")):
            with_disease += 1
        if any(k.text == DRUG_TARGET_KEYWORD
               for k in elem.findall(f"{ns}keyword")):
            with_pharma += 1
        if any(f.get("type") in PTM_FEATURE_TYPES
               for f in elem.findall(f"{ns}feature")):
            with_ptm += 1
        if any(d.get("type") == "GO"
               for d in elem.findall(f"{ns}dbReference")):
            with_go += 1
        elem.clear()
        if total % 5000 == 0:
            print(f"  ...{total:,} entries", file=sys.stderr)

    print(f"\nUniProt annotation summary:")
    print(f"  Total entries:     {total:,}")
    print(f"  With disease:      {with_disease:,} ({100*with_disease/total:.1f}%)")
    print(f"  Drug targets:      {with_pharma:,} ({100*with_pharma/total:.1f}%)")
    print(f"  With PTM features: {with_ptm:,} ({100*with_ptm/total:.1f}%)")
    print(f"  With GO terms:     {with_go:,} ({100*with_go/total:.1f}%)")


def _cli_lookup(disease_dir: str, accession: str) -> None:
    """Look up annotations for a single protein."""
    xml_path = Path(disease_dir) / UNIPROT_XML_FILENAME
    if not xml_path.exists():
        print(f"Error: {xml_path} not found", file=sys.stderr)
        sys.exit(1)

    index = load_uniprot_annotations(xml_path, frozenset({accession}), verbose=True)

    if accession not in index:
        # Try API fallback
        print(f"  Not found in local XML, trying API...", file=sys.stderr)
        result = fetch_uniprot_annotation_api(accession)
        if result is not None:
            index[accession] = result
        else:
            print(f"  Protein {accession} not found in local XML or API.")
            return

    ann = index[accession]
    print(f"\nAnnotations for {accession}:")
    print(f"  Diseases ({len(ann['diseases'])}):")
    for d in ann["diseases"]:
        omim = f" [OMIM:{d['omim_id']}]" if d["omim_id"] else ""
        print(f"    - {d['disease_name']}{omim}")

    print(f"  Drug target: {ann['is_drug_target']}")

    print(f"  PTM sites ({len(ann['ptm_sites'])}):")
    for p in ann["ptm_sites"][:10]:
        print(f"    - {p['description']} at position {p['position']}")
    if len(ann["ptm_sites"]) > 10:
        print(f"    ... and {len(ann['ptm_sites']) - 10} more")

    go_bp = [g for g in ann["go_terms"] if g["aspect"] == "P"]
    go_mf = [g for g in ann["go_terms"] if g["aspect"] == "F"]
    go_cc = [g for g in ann["go_terms"] if g["aspect"] == "C"]
    print(f"  GO Biological Process ({len(go_bp)}):")
    for g in go_bp[:5]:
        print(f"    - {g['go_id']}: {g['go_name']}")
    print(f"  GO Molecular Function ({len(go_mf)}):")
    for g in go_mf[:5]:
        print(f"    - {g['go_id']}: {g['go_name']}")
    print(f"  GO Cellular Component ({len(go_cc)}):")
    for g in go_cc[:5]:
        print(f"    - {g['go_id']}: {g['go_name']}")

    if ann["kegg_ids"]:
        print(f"  KEGG: {', '.join(ann['kegg_ids'])}")


def main() -> None:
    """CLI entry point."""
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "summary":
        _cli_summary(args.disease_dir)
    elif args.command == "lookup":
        _cli_lookup(args.disease_dir, args.protein)


if __name__ == "__main__":
    main()
