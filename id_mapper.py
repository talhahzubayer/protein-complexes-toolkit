#!/usr/bin/env python3
"""
ID Cross-Reference and Mapping Module.
Parses the STRING aliases file to build in-memory lookup dictionaries for efficient cross-referencing between Ensembl protein IDs (ENSP), Ensembl gene IDs (ENSG), UniProt accessions, and gene symbols.
Isoform-aware: preserves full isoform-specific UniProt accessions (e.g., Q9UKT4-2) as primary keys and uses base accessions (e.g., Q9UKT4) as grouping fields.

Features:
    - ENSP -> UniProt cross-referencing via STRING aliases
    - ENSG -> UniProt cross-referencing via ENSP intermediary
    - UniProt -> gene symbol and protein name resolution
    - Secondary accession detection and canonical accession prioritisation
    - Master lookup table export (CSV) with all cross-references
    - Single-identifier resolution for interactive debugging

Usage (as importable module):
    from id_mapper import IDMapper
    mapper = IDMapper("data/ppi/9606.protein.aliases.v12.0.txt")

    # -> ['P04637']
    mapper.ensembl_to_uniprot("ENSP00000269305") 

    # -> 'TP53' 
    mapper.uniprot_to_gene_symbol("P04637")

    # -> ['P04637']        
    mapper.ensg_to_uniprot("ENSG00000141510")      

Usage (standalone):
    python id_mapper.py --aliases data/ppi/9606.protein.aliases.v12.0.txt --stats
    python id_mapper.py --aliases data/ppi/9606.protein.aliases.v12.0.txt --export lookup.csv
    python id_mapper.py --aliases data/ppi/9606.protein.aliases.v12.0.txt --resolve P04637
"""

import sys
import re
import argparse
import csv
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional
from collections import defaultdict
import pandas as pd

#------Constants----------------------------------------------------

# UniProt accession pattern (with optional isoform suffix)
# Matches: P12345, Q9UKT4-2, A0A0B4J2C3, A0A0B4J2C3-1
UNIPROT_ACCESSION_RE = re.compile(
    r'^[OPQ][0-9][A-Z0-9]{3}[0-9](-\d+)?$'
    r'|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}(-\d+)?$'
)

# Ensembl protein ID pattern
ENSP_RE = re.compile(r'^ENSP\d{11}$')

# Ensembl gene ID pattern
ENSG_RE = re.compile(r'^ENSG\d{11}$')

# STRING taxonomy prefix
STRING_TAXONOMY_PREFIX = '9606.'

# Alias sources to parse from the STRING aliases file (priority order)
ALIAS_SOURCES_UNIPROT = frozenset({'UniProt_AC', 'Ensembl_UniProt'})
ALIAS_SOURCES_GENE_SYMBOL = frozenset({'Ensembl_HGNC_symbol'})
ALIAS_SOURCES_ENSG = frozenset({'Ensembl_gene'})
ALIAS_SOURCES_PROTEIN_NAME = frozenset({'UniProt_DE_RecName_Full'})

# All relevant sources combined for fast filtering during parsing
ALL_RELEVANT_SOURCES = (
    ALIAS_SOURCES_UNIPROT
    | ALIAS_SOURCES_GENE_SYMBOL
    | ALIAS_SOURCES_ENSG
    | ALIAS_SOURCES_PROTEIN_NAME
)

# Default aliases file path
DEFAULT_ALIASES_PATH = Path(__file__).parent / "data" / "ppi" / "9606.protein.aliases.v12.0.txt"

# Swiss-Prot canonical accession characteristics (for sort priority)
# Canonical entries start with O/P/Q and are exactly 6 characters.
# TrEMBL entries use other letters or the longer 10-character A0A format.
CANONICAL_FIRST_CHARS = frozenset('OPQ')
CANONICAL_ACCESSION_LENGTH = 6

# Species classification status values.
SPECIES_REVIEWED_HUMAN = 'reviewed_human'
SPECIES_TREMBL_HUMAN   = 'trembl_human'
SPECIES_NON_HUMAN      = 'non_human'

# Priority used to combine two per-chain statuses into a per-complex status
# (higher number = worse; non-human dominates).
_SPECIES_PRIORITY = {
    SPECIES_REVIEWED_HUMAN: 0,
    SPECIES_TREMBL_HUMAN:   1,
    SPECIES_NON_HUMAN:      2,
}

# XML namespace used in Swiss-Prot flat-file downloads (uniprot_sprot_human.xml).
_UNIPROT_XML_NS = "{https://uniprot.org/uniprot}"


#-----------------------ID Validation Functions------------------------------------------------

def is_uniprot_accession(identifier: str) -> bool:
    """Check if a string matches the UniProt accession format.
    Accepts both canonical (P12345) and isoform-specific (Q9UKT4-2) accessions, including the longer format (A0A0B4J2C3).
    Args:
        identifier: String to test.
    Returns:
        True if it matches UniProt accession pattern.
    """
    return bool(UNIPROT_ACCESSION_RE.match(identifier))


def split_isoform(accession: str) -> tuple[str, Optional[str]]:
    """Split a UniProt accession into base accession and isoform number.
    Args:
        accession: UniProt accession, e.g. 'Q9UKT4-2' or 'P12345'.
    Returns:
        Tuple of (base_accession, isoform_number_or_None).
        For 'Q9UKT4-2' returns ('Q9UKT4', '2').
        For 'P12345' returns ('P12345', None).
    """
    if '-' in accession:
        parts = accession.rsplit('-', 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0], parts[1]
    return accession, None


#-----------------------Species Classification-------------------------------------------------

def load_sprot_reviewed_accessions(xml_path) -> set[str]:
    """Return every <accession> element (primary + secondary) from a Swiss-Prot XML dump.
    Uses a streaming iterparse so the 1.5GB human XML doesn't load fully into memory.
    Args:
        xml_path: Path or str to uniprot_sprot_human.xml.
    Returns:
        Set of reviewed human UniProt accessions (incl. secondary/obsolete).
    """
    accs: set[str] = set()
    ns = _UNIPROT_XML_NS
    ctx = ET.iterparse(str(xml_path), events=("end",))
    for _, elem in ctx:
        if elem.tag == ns + "entry":
            for a in elem.findall(ns + "accession"):
                if a.text:
                    accs.add(a.text.strip())
            elem.clear()
    return accs


def load_human_accessions_from_idmapping(dat_path) -> set[str]:
    """Return all human UniProt accessions (reviewed + TrEMBL) from HUMAN_9606_idmapping.dat.
    Each accession appears once under the UniProtKB-ID mapping-type key, so
    filtering on column 2 == 'UniProtKB-ID' yields exactly one row per accession.
    Args:
        dat_path: Path or str to HUMAN_9606_idmapping.dat.
    Returns:
        Set of all human UniProt accessions (both Swiss-Prot and TrEMBL).
    """
    human: set[str] = set()
    with open(dat_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[1] == "UniProtKB-ID":
                human.add(parts[0])
    return human


def combine_species_statuses(status_a: str, status_b: str) -> str:
    """Combine two per-chain statuses into a per-complex status (worst-of-two).
    non_human dominates trembl_human dominates reviewed_human. Unrecognised
    strings are treated as non_human.
    Args:
        status_a: Status of chain A (reviewed_human / trembl_human / non_human).
        status_b: Status of chain B (same vocabulary).
    Returns:
        The higher-priority status of the two.
    """
    non_human = _SPECIES_PRIORITY[SPECIES_NON_HUMAN]
    pa = _SPECIES_PRIORITY.get(status_a, non_human)
    pb = _SPECIES_PRIORITY.get(status_b, non_human)
    worst = max(pa, pb)
    if worst == _SPECIES_PRIORITY[SPECIES_NON_HUMAN]:
        return SPECIES_NON_HUMAN
    if worst == _SPECIES_PRIORITY[SPECIES_TREMBL_HUMAN]:
        return SPECIES_TREMBL_HUMAN
    return SPECIES_REVIEWED_HUMAN


class SpeciesClassifier:
    """Classify UniProt accessions as reviewed_human / trembl_human / non_human.

    Reference sets are loaded lazily on the first classify() call so the
    ~15s one-time load cost is only paid when the classifier is actually used.
    Isoform suffixes (e.g. '-2') are stripped before lookup — the reference
    sets only contain canonical accessions.
    """

    def __init__(
        self,
        sprot_xml_path: Optional[str] = None,
        idmapping_path: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        self._sprot_xml_path = sprot_xml_path
        self._idmapping_path = idmapping_path
        self._verbose = verbose
        self._sprot_reviewed: Optional[set[str]] = None
        self._human_all: Optional[set[str]] = None

    def _resolve_paths(self) -> tuple[Path, Path]:
        # Import here to avoid any circular-import risk with data_registry.
        from data_registry import get_default_path
        xml = Path(self._sprot_xml_path) if self._sprot_xml_path \
            else Path(get_default_path('uniprot_xml'))
        dat = Path(self._idmapping_path) if self._idmapping_path \
            else Path(get_default_path('eve_idmapping'))
        return xml, dat

    def _ensure_loaded(self) -> None:
        if self._sprot_reviewed is not None and self._human_all is not None:
            return
        xml_path, dat_path = self._resolve_paths()
        if not xml_path.is_file():
            raise FileNotFoundError(
                f"Swiss-Prot XML not found at {xml_path}. "
                f"Run 'python data_registry.py' to check data dependencies."
            )
        if not dat_path.is_file():
            raise FileNotFoundError(
                f"idmapping file not found at {dat_path}. "
                f"Run 'python data_registry.py' to check data dependencies."
            )
        if self._verbose:
            print(f"  Loading reviewed Swiss-Prot accessions from {xml_path.name} ...",
                  file=sys.stderr)
        self._sprot_reviewed = load_sprot_reviewed_accessions(xml_path)
        if self._verbose:
            print(f"  Loaded {len(self._sprot_reviewed):,} reviewed accessions",
                  file=sys.stderr)
            print(f"  Loading human idmapping from {dat_path.name} ...",
                  file=sys.stderr)
        self._human_all = load_human_accessions_from_idmapping(dat_path)
        if self._verbose:
            print(f"  Loaded {len(self._human_all):,} total human accessions",
                  file=sys.stderr)

    def classify(self, accession: str) -> str:
        """Classify a UniProt accession. Empty/None returns non_human."""
        if not accession:
            return SPECIES_NON_HUMAN
        self._ensure_loaded()
        base, _ = split_isoform(accession.strip())
        if base in self._sprot_reviewed:
            return SPECIES_REVIEWED_HUMAN
        if base in self._human_all:
            return SPECIES_TREMBL_HUMAN
        return SPECIES_NON_HUMAN


#-----------------------ID Type Detection-----------------------------------------------------

def detect_id_type(identifier: str) -> str:
    """Detect the type of a protein/gene identifier.
    Args:
        identifier: Any protein or gene identifier string.
    Returns:
        One of: 'uniprot_isoform', 'uniprot', 'ensp', 'ensg', 'unknown'.
    """
    if ENSP_RE.match(identifier):
        return 'ensp'
    if ENSG_RE.match(identifier):
        return 'ensg'
    if is_uniprot_accession(identifier):
        _, iso = split_isoform(identifier)
        return 'uniprot_isoform' if iso is not None else 'uniprot'
    return 'unknown'


#---------------------------------IDMapper Class---------------------------------------------------

class IDMapper:
    """Cross-reference mapper for protein identifiers.
    Parses the STRING aliases file once and builds in-memory lookup dictionaries for efficient ID resolution. Supports mappings between:
    - Ensembl protein IDs (ENSP) <-> UniProt accessions
    - Ensembl protein IDs (ENSP) <-> gene symbols
    - Ensembl gene IDs (ENSG) <-> Ensembl protein IDs (ENSP)
    - UniProt accessions <-> gene symbols (via ENSP as bridge)
    Isoform-aware: when mapping from databases that lack isoform specificity (STRING, BioGRID), the mapper flags results as base-accession matches.
    Args:
        aliases_filepath: Path to STRING aliases file. Defaults to data/ppi/9606.protein.aliases.v12.0.txt.
        verbose: Print progress during parsing.
        api_fallback: If True (default), attempt STRING API resolution when
            local lookup fails. Set to False to disable all API calls.
    """

    def __init__(self, aliases_filepath: Optional[str] = None, verbose: bool = False,
                 api_fallback: bool = True) -> None:
        """Parse the aliases file and build lookup dictionaries."""
        # ENSP -> list of UniProt accessions
        self._ensp_to_uniprot: dict[str, list[str]] = defaultdict(list)
        # UniProt base accession -> list of ENSP IDs
        self._uniprot_to_ensp: dict[str, list[str]] = defaultdict(list)
        # ENSP -> gene symbol (one-to-one, last wins)
        self._ensp_to_symbol: dict[str, str] = {}
        # gene symbol -> list of ENSP IDs
        self._symbol_to_ensp: dict[str, list[str]] = defaultdict(list)
        # ENSG -> list of ENSP IDs
        self._ensg_to_ensp: dict[str, list[str]] = defaultdict(list)
        # ENSP -> ENSG (one-to-one)
        self._ensp_to_ensg: dict[str, str] = {}
        # ENSP -> protein name (one-to-one, last wins)
        self._ensp_to_name: dict[str, str] = {}

        # API fallback state
        self._api_fallback = api_fallback
        self._api_available = True  # Latches to False after first StringAPIError

        filepath = Path(aliases_filepath) if aliases_filepath else DEFAULT_ALIASES_PATH
        if not filepath.exists():
            raise FileNotFoundError(f"Aliases file not found: {filepath}")

        self._parse_aliases(filepath, verbose)

    def _parse_aliases(self, filepath: Path, verbose: bool = False) -> None:
        """Parse the STRING aliases file line by line.
        Reads only rows with source types in the relevant sets.
        For Ensembl_UniProt rows, additionally filters by UniProt accession regex to exclude gene symbols mixed into this source.
        Args:
            filepath: Path to the aliases TSV file.
            verbose: Print progress.
        """
        lines_read = 0
        lines_used = 0

        with open(filepath, encoding='utf-8', errors='replace') as f:
            # Skip header line
            next(f)
            for line in f:
                lines_read += 1
                parts = line.rstrip('\n').split('\t')
                if len(parts) < 3:
                    continue

                ensp_raw, alias, source = parts[0], parts[1], parts[2]

                if source not in ALL_RELEVANT_SOURCES:
                    continue

                # Strip 9606. taxonomy prefix
                ensp = ensp_raw.removeprefix(STRING_TAXONOMY_PREFIX)

                # Route to appropriate dictionary based on source
                if source in ALIAS_SOURCES_UNIPROT:
                    # For Ensembl_UniProt, filter by regex (mixed content)
                    if source == 'Ensembl_UniProt' and not is_uniprot_accession(alias):
                        continue
                    if alias not in self._ensp_to_uniprot[ensp]:
                        self._ensp_to_uniprot[ensp].append(alias)
                        base, _ = split_isoform(alias)
                        if ensp not in self._uniprot_to_ensp[base]:
                            self._uniprot_to_ensp[base].append(ensp)

                elif source in ALIAS_SOURCES_GENE_SYMBOL:
                    self._ensp_to_symbol[ensp] = alias
                    if ensp not in self._symbol_to_ensp[alias]:
                        self._symbol_to_ensp[alias].append(ensp)

                elif source in ALIAS_SOURCES_ENSG:
                    if ensp not in self._ensg_to_ensp[alias]:
                        self._ensg_to_ensp[alias].append(ensp)
                    self._ensp_to_ensg[ensp] = alias

                elif source in ALIAS_SOURCES_PROTEIN_NAME:
                    self._ensp_to_name[ensp] = alias

                lines_used += 1

        # Sort UniProt accession lists to prioritize reviewed (Swiss-Prot) accessions. 
        # Swiss-Prot entries for humans typically start with P, Q, or O (6 chars). 
        # TrEMBL entries start with other letters or use the longer A0A0* format (10 chars).
        def _uniprot_sort_key(acc: str) -> tuple[int, int, str]:
            is_canonical = acc[0] in CANONICAL_FIRST_CHARS and len(acc) == CANONICAL_ACCESSION_LENGTH
            return (0 if is_canonical else 1, len(acc), acc)

        for ensp in self._ensp_to_uniprot:
            self._ensp_to_uniprot[ensp].sort(key=_uniprot_sort_key)

        # Convert defaultdicts to regular dicts to prevent accidental creation
        self._ensp_to_uniprot = dict(self._ensp_to_uniprot)
        self._uniprot_to_ensp = dict(self._uniprot_to_ensp)
        self._symbol_to_ensp = dict(self._symbol_to_ensp)
        self._ensg_to_ensp = dict(self._ensg_to_ensp)

        if verbose:
            print(f"  Aliases: parsed {lines_read:,} lines, used {lines_used:,} relevant entries", file=sys.stderr)
            stats = self.get_mapping_stats()
            for key, count in stats.items():
                print(f"    {key}: {count:,} entries", file=sys.stderr)

    def ensembl_to_uniprot(self, ensp_id: str) -> list[str]:
        """Map an Ensembl protein ID to UniProt accession(s).
        Args:
            ensp_id: Ensembl protein ID, e.g. 'ENSP00000269305'. Accepts with or without '9606.' prefix.
        Returns:
            List of UniProt accessions mapped to this ENSP ID.
            Empty list if no mapping found.
        """
        ensp = ensp_id.removeprefix(STRING_TAXONOMY_PREFIX)
        return list(self._ensp_to_uniprot.get(ensp, []))

    def uniprot_to_ensembl(self, uniprot_id: str) -> list[str]:
        """Map a UniProt accession to Ensembl protein ID(s).
        For isoform accessions (e.g. 'Q9UKT4-2'), looks up the base accession ('Q9UKT4') since STRING lacks isoform specificity.
        Args:
            uniprot_id: UniProt accession (base or isoform).
        Returns:
            List of ENSP IDs. Empty list if no mapping found.
        """
        base, _ = split_isoform(uniprot_id)
        return list(self._uniprot_to_ensp.get(base, []))

    def uniprot_to_gene_symbol(self, uniprot_id: str) -> Optional[str]:
        """Map a UniProt accession to its HGNC gene symbol.
        Chains: UniProt -> ENSP -> gene symbol.
        Args:
            uniprot_id: UniProt accession (base or isoform).
        Returns:
            Gene symbol string, or None if no mapping found.
        """
        ensp_list = self.uniprot_to_ensembl(uniprot_id)
        for ensp in ensp_list:
            symbol = self._ensp_to_symbol.get(ensp)
            if symbol:
                return symbol
        return None

    def ensg_to_uniprot(self, ensg_id: str) -> list[str]:
        """Map an Ensembl gene ID (ENSG) to UniProt accession(s).
        This is the bridge for HuRI data, which uses ENSG IDs.
        Chains: ENSG -> ENSP -> UniProt.
        Args:
            ensg_id: Ensembl gene ID, e.g. 'ENSG00000141510'.
        Returns:
            List of unique UniProt accessions. Empty list if no mapping found.
        """
        ensp_list = self._ensg_to_ensp.get(ensg_id, [])
        seen = set()
        result = []
        for ensp in ensp_list:
            for uniprot in self._ensp_to_uniprot.get(ensp, []):
                if uniprot not in seen:
                    seen.add(uniprot)
                    result.append(uniprot)
        return result

    def ensg_to_ensembl(self, ensg_id: str) -> list[str]:
        """Map an Ensembl gene ID (ENSG) to Ensembl protein ID(s) (ENSP).
        Args:
            ensg_id: Ensembl gene ID.
        Returns:
            List of ENSP IDs. Empty list if no mapping found.
        """
        return list(self._ensg_to_ensp.get(ensg_id, []))

    def resolve_id(self, identifier: str, target: str = 'uniprot') -> Optional[str]:
        """Master resolution function: accept any ID type, return the target type.
        Detects the input ID type automatically and resolves through the mapping chain to produce the requested target type.
        For isoform-specific UniProt accessions passed as input, the full isoform ID is preserved in the output when target is 'uniprot'.
        Falls back to the STRING API when local lookup fails and api_fallback is True.
        Args:
            identifier: Any protein/gene identifier.
            target: Target ID type: 'uniprot', 'ensp', 'gene_symbol'.
        Returns:
            Resolved identifier in the target namespace, or None if unmappable.
            For 'uniprot' target with one-to-many: returns the first accession.
        """
        result = self._resolve_id_local(identifier, target)
        if result is not None:
            return result

        # API fallback: if local resolution returned None, try STRING API
        if self._api_fallback and self._api_available:
            return self._resolve_via_api(identifier, target)
        return None

    def _resolve_id_local(self, identifier: str, target: str = 'uniprot') -> Optional[str]:
        """Local-only ID resolution (no API calls).
        Args:
            identifier: Any protein/gene identifier.
            target: Target ID type: 'uniprot', 'ensp', 'gene_symbol'.
        Returns:
            Resolved identifier or None if not found in local data.
        """
        id_type = detect_id_type(identifier)

        if target == 'uniprot':
            if id_type in ('uniprot', 'uniprot_isoform'):
                return identifier  # Already UniProt
            if id_type == 'ensp':
                results = self.ensembl_to_uniprot(identifier)
                return results[0] if results else None
            if id_type == 'ensg':
                results = self.ensg_to_uniprot(identifier)
                return results[0] if results else None

        elif target == 'ensp':
            if id_type == 'ensp':
                return identifier
            if id_type in ('uniprot', 'uniprot_isoform'):
                results = self.uniprot_to_ensembl(identifier)
                return results[0] if results else None
            if id_type == 'ensg':
                results = self.ensg_to_ensembl(identifier)
                return results[0] if results else None

        elif target == 'gene_symbol':
            if id_type in ('uniprot', 'uniprot_isoform'):
                return self.uniprot_to_gene_symbol(identifier)
            if id_type == 'ensp':
                return self._ensp_to_symbol.get(
                    identifier.removeprefix(STRING_TAXONOMY_PREFIX)
                )
            if id_type == 'ensg':
                ensp_list = self.ensg_to_ensembl(identifier)
                for ensp in ensp_list:
                    symbol = self._ensp_to_symbol.get(ensp)
                    if symbol:
                        return symbol

        return None

    def _resolve_via_api(self, identifier: str, target: str) -> Optional[str]:
        """Attempt to resolve an identifier via the STRING API.
        Called automatically when local resolution fails and api_fallback is True.
        On the first StringAPIError, sets _api_available to False to avoid
        repeated failed API calls within the same session.
        Args:
            identifier: Protein/gene identifier that failed local lookup.
            target: Target ID type: 'uniprot', 'ensp', or 'gene_symbol'.
        Returns:
            Resolved identifier in the target namespace, or None.
        """
        try:
            from string_api import get_string_ids, StringAPIError
        except ImportError:
            self._api_available = False
            return None

        try:
            api_result = get_string_ids([identifier])
            if api_result.empty:
                return None

            row = api_result.iloc[0]
            string_id = row.get('stringId', '')
            preferred_name = row.get('preferredName', '')

            # Extract ENSP from stringId (format: "9606.ENSP00000269305")
            ensp = string_id.removeprefix(STRING_TAXONOMY_PREFIX) if string_id else ''

            if target == 'ensp':
                return ensp if ENSP_RE.match(ensp) else None
            elif target == 'gene_symbol':
                return preferred_name if preferred_name else None
            elif target == 'uniprot':
                # Check if preferredName is a UniProt accession
                if preferred_name and is_uniprot_accession(preferred_name):
                    return preferred_name
                # Try local ENSP->UniProt mapping with the API-resolved ENSP
                if ensp and ENSP_RE.match(ensp):
                    results = self.ensembl_to_uniprot(ensp)
                    if results:
                        return results[0]
            return None

        except Exception as e:
            # Check if it's a StringAPIError to latch off API
            from string_api import StringAPIError
            if isinstance(e, StringAPIError):
                self._api_available = False
                warnings.warn(
                    f"STRING API unavailable, disabling API fallback: {e}",
                    stacklevel=2,
                )
            return None

    def resolve_pair_to_uniprot(self, id_a: str, id_b: str) -> Optional[tuple[str, str, bool]]:
        """Resolve a pair of identifiers to UniProt accessions.
        Args:
            id_a: Identifier for protein A (any type).
            id_b: Identifier for protein B (any type).
        Returns:
            Tuple of (uniprot_a, uniprot_b, is_base_accession_match) or None if either protein cannot be resolved.
            is_base_accession_match is True when the source database lacks isoform specificity (mapped via base accession only).
        """
        uniprot_a = self.resolve_id(id_a, target='uniprot')
        uniprot_b = self.resolve_id(id_b, target='uniprot')

        if uniprot_a is None or uniprot_b is None:
            return None

        # Check if either input was a non-UniProt type (base accession match)
        type_a = detect_id_type(id_a)
        type_b = detect_id_type(id_b)
        is_base = type_a not in ('uniprot', 'uniprot_isoform') or type_b not in ('uniprot', 'uniprot_isoform')
        return (uniprot_a, uniprot_b, is_base)

    def get_secondary_accessions(self, ensp_id: str) -> list[str]:
        """Return non-primary UniProt accessions mapped to an ENSP ID.
        Design decision: We use the STRING aliases file rather than the UniProt REST API or the 8 GB idmapping.dat download. 
        STRING aliases already contain secondary accessions as alternative ENSP mappings.
        Trade-off: proteins removed entirely from UniProt (not just merged) are not caught. 
        See Documentation/Research_Project_Roadmap.md for the full rationale and future alternatives.
        The primary accession is the first entry (prioritised by the Swiss-Prot-first sort key). 
        Secondary accessions are all remaining entries - these may be older UniProt accessions that have been merged into the primary or TrEMBL alternatives.
        Args:
            ensp_id: Ensembl protein ID.
        Returns:
            List of secondary UniProt accessions (excluding the primary).
            Empty list if ENSP has zero or one mapping.
        """
        ensp = ensp_id.removeprefix(STRING_TAXONOMY_PREFIX)
        accessions = self._ensp_to_uniprot.get(ensp, [])
        return list(accessions[1:]) if len(accessions) > 1 else []

    def get_protein_name(self, ensp_id: str) -> Optional[str]:
        """Return the protein name for an ENSP ID, or None.
        Args:
            ensp_id: Ensembl protein ID.
        Returns:
            Protein name string, or None if no name is recorded.
        """
        ensp = ensp_id.removeprefix(STRING_TAXONOMY_PREFIX)
        return self._ensp_to_name.get(ensp)

    def get_mapping_stats(self) -> dict[str, int]:
        """Return counts of entries in each lookup dictionary.
        Returns:
            Dict with keys like 'ensp_to_uniprot', 'uniprot_to_ensp', etc.
            Integer counts as values.
        """
        return {
            'ensp_to_uniprot': len(self._ensp_to_uniprot),
            'uniprot_to_ensp': len(self._uniprot_to_ensp),
            'ensp_to_symbol': len(self._ensp_to_symbol),
            'symbol_to_ensp': len(self._symbol_to_ensp),
            'ensg_to_ensp': len(self._ensg_to_ensp),
            'ensp_to_ensg': len(self._ensp_to_ensg),
            'ensp_to_name': len(self._ensp_to_name),
        }

#------------------------------------Convenience Functions------------------------------------------------

def map_dataframe_to_uniprot(df: pd.DataFrame, mapper: IDMapper, source_columns: tuple[str, str] = ('protein_a', 'protein_b'), verbose: bool = False) -> pd.DataFrame:
    """Map protein IDs in a DataFrame to UniProt accessions.
    Adds columns 'uniprot_a' and 'uniprot_b' to the DataFrame. Rows where either protein cannot be mapped are dropped and counted.
    This is used to normalise HuRI (ENSG IDs) and STRING (ENSP IDs) to UniProt for cross-database comparison. BioGRID and HuMAP already use UniProt accessions.
    Args:
        df: DataFrame with interaction data.
        mapper: Initialised IDMapper instance.
        source_columns: Column names containing protein IDs to map.
        verbose: Print mapping statistics.
    Returns:
        Copy of DataFrame with added 'uniprot_a', 'uniprot_b' columns.
        Rows where mapping failed are dropped.
    """
    col_a, col_b = source_columns
    result = df.copy()

    result['uniprot_a'] = result[col_a].map(
        lambda x: mapper.resolve_id(x, target='uniprot')
    )
    result['uniprot_b'] = result[col_b].map(
        lambda x: mapper.resolve_id(x, target='uniprot')
    )

    n_before = len(result)
    result = result.dropna(subset=['uniprot_a', 'uniprot_b'])
    n_after = len(result)
    n_dropped = n_before - n_after

    if verbose and n_dropped > 0:
        print(f"  ID mapping: {n_dropped:,} rows dropped "
              f"({n_dropped/n_before*100:.1f}%), "
              f"{n_after:,} rows retained",
              file=sys.stderr)

    return result

def export_lookup_table(mapper: IDMapper, output_path: str, verbose: bool = False) -> None:
    """Export the master ID lookup table as a CSV file.
    Produces a table with one row per unique ENSP ID, containing all available cross-references. 
    The primary UniProt accession is the reviewed Swiss-Prot entry (sorted first by _uniprot_sort_key).
    Args:
        mapper: Initialised IDMapper instance.
        output_path: Path for the output CSV file.
        verbose: Print progress.
    """
    fieldnames = [
        'ensembl_protein_id',
        'primary_uniprot',
        'base_accession',
        'secondary_accessions',
        'ensembl_gene_id',
        'gene_symbol',
        'protein_name',
    ]

    # Collect all ENSP IDs across all dictionaries
    all_ensp = set(mapper._ensp_to_uniprot.keys())
    all_ensp.update(mapper._ensp_to_symbol.keys())
    all_ensp.update(mapper._ensp_to_ensg.keys())
    all_ensp.update(mapper._ensp_to_name.keys())

    rows = []
    for ensp in sorted(all_ensp):
        uniprots = mapper._ensp_to_uniprot.get(ensp, [])
        primary = uniprots[0] if uniprots else ''
        base, _ = split_isoform(primary) if primary else ('', None)
        secondary = uniprots[1:] if len(uniprots) > 1 else []
        rows.append({
            'ensembl_protein_id': ensp,
            'primary_uniprot': primary,
            'base_accession': base,
            'secondary_accessions': '|'.join(secondary),
            'ensembl_gene_id': mapper._ensp_to_ensg.get(ensp, ''),
            'gene_symbol': mapper._ensp_to_symbol.get(ensp, ''),
            'protein_name': mapper._ensp_to_name.get(ensp, ''),
        })

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if verbose:
        print(f"  Exported {len(rows):,} protein entries to {output_path}", file=sys.stderr)

def build_uniprot_lookup(mapper: IDMapper) -> dict[str, dict]:
    """Build a fast UniProt-keyed lookup dict for enrichment.
    Returns a dict keyed by UniProt accession (primary and secondary) containing gene symbol, protein name, Ensembl IDs. 
    Used by the toolkit to enrich CSV rows without repeated resolve_id() calls.
    Args:
        mapper: Initialised IDMapper instance.
    Returns:
        Dict mapping UniProt accession -> {
            'gene_symbol': str or '',
            'protein_name': str or '',
            'ensembl_protein_id': str or '',
            'ensembl_gene_id': str or '',
            'secondary_accessions': str (pipe-separated) or '',
        }.
    """
    lookup: dict[str, dict] = {}

    for ensp, uniprots in mapper._ensp_to_uniprot.items():
        symbol = mapper._ensp_to_symbol.get(ensp, '')
        name = mapper._ensp_to_name.get(ensp, '')
        ensg = mapper._ensp_to_ensg.get(ensp, '')

        info = {
            'gene_symbol': symbol,
            'protein_name': name,
            'ensembl_protein_id': ensp,
            'ensembl_gene_id': ensg,
            # Pipe-separated alternate UniProt accessions for this ENSP.
            # One ENSP can map to multiple UniProt IDs (reviewed + unreviewed, or merged/legacy accessions).  
            # uniprots[0] is the primary and uniprots[1:] are secondary, joined with '|'.
            'secondary_accessions': '|'.join(uniprots[1:]) if len(uniprots) > 1 else '',
        }

        for acc in uniprots:
            # First entry for this accession wins (primary ENSP mapping)
            if acc not in lookup:
                lookup[acc] = info
            # Also index by base accession for isoform-agnostic lookup
            base, _ = split_isoform(acc)
            if base != acc and base not in lookup:
                lookup[base] = info

    return lookup

#-------------------CLI Entry Point-----------------------------------------

def build_argument_parser() -> argparse.ArgumentParser:
    """Create and return the argument parser for id_mapper."""
    parser = argparse.ArgumentParser(
        description="Protein ID cross-reference mapper using STRING aliases.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage (as importable module):
    from id_mapper import IDMapper
    mapper = IDMapper("data/ppi/9606.protein.aliases.v12.0.txt")

    # -> ['P04637']
    mapper.ensembl_to_uniprot("ENSP00000269305") 

    # -> 'TP53' 
    mapper.uniprot_to_gene_symbol("P04637")

    # -> ['P04637']        
    mapper.ensg_to_uniprot("ENSG00000141510")      

Usage (standalone):
    python id_mapper.py --aliases data/ppi/9606.protein.aliases.v12.0.txt --stats
    python id_mapper.py --aliases data/ppi/9606.protein.aliases.v12.0.txt --export lookup.csv
    python id_mapper.py --aliases data/ppi/9606.protein.aliases.v12.0.txt --resolve P04637
""",
)
    parser.add_argument(
        "--aliases",
        default=str(DEFAULT_ALIASES_PATH),
        help="Path to STRING aliases file (default: data/ppi/9606.protein.aliases.v12.0.txt)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print mapping statistics and exit",
    )
    parser.add_argument(
        "--export",
        metavar="OUTPUT_CSV",
        help="Export master lookup table to CSV",
    )
    parser.add_argument(
        "--resolve",
        metavar="IDENTIFIER",
        help="Resolve a single identifier and print all mappings",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress information",
    )
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="Disable STRING API fallback for ID resolution. Use only local "
             "aliases file data.",
    )
    parser.add_argument(
        "--validate-ids-api",
        action="store_true",
        help=(
            "After local resolution, pass any unresolved identifiers through "
            "the STRING API get_string_ids endpoint as a fallback. Requires "
            "network access. Used with --resolve to validate IDs that fail "
            "local lookup against the aliases file."
        ),
    )
    parser.add_argument(
        "--cross-validate-api",
        type=int,
        metavar="N",
        default=0,
        help=(
            "Cross-validate N randomly sampled local STRING interactions "
            "against the live STRING API interaction_partners endpoint. "
            "Reports agreement/disagreement counts. Requires network access."
        ),
    )

    return parser


def main() -> None:
    """Run the ID mapper CLI."""
    parser = build_argument_parser()
    args = parser.parse_args()

    print(f"Loading aliases from: {args.aliases}", file=sys.stderr)
    mapper = IDMapper(args.aliases, verbose=True, api_fallback=not args.no_api)

    if args.stats:
        stats = mapper.get_mapping_stats()
        print("\nMapping Statistics:")
        for key, count in stats.items():
            print(f"  {key}: {count:,}")

    if args.export:
        export_lookup_table(mapper, args.export, verbose=True)
        print(f"\nLookup table exported to: {args.export}")

    if args.resolve:
        identifier = args.resolve
        id_type = detect_id_type(identifier)
        print(f"\nResolving: {identifier} (detected type: {id_type})")

        uniprot = mapper.resolve_id(identifier, target='uniprot')
        ensp = mapper.resolve_id(identifier, target='ensp')
        symbol = mapper.resolve_id(identifier, target='gene_symbol')

        print(f"  UniProt:     {uniprot or 'not found'}")
        print(f"  ENSP:        {ensp or 'not found'}")
        print(f"  Gene symbol: {symbol or 'not found'}")

        if id_type == 'ensp':
            all_uniprots = mapper.ensembl_to_uniprot(identifier)
            if len(all_uniprots) > 1:
                print(f"  All UniProt:  {', '.join(all_uniprots)}")

        # A.4.1: Validate unresolved IDs against STRING API
        if args.validate_ids_api and not uniprot:
            try:
                import warnings
                from string_api import get_string_ids, get_version, StringAPIError

                print(f"\n  Local resolution failed. Querying STRING API...",
                      file=sys.stderr)
                version = get_version()
                print(f"  STRING API version: {version.get('string_version', 'unknown')}",
                      file=sys.stderr)

                api_result = get_string_ids([identifier])
                if len(api_result) > 0:
                    print(f"\n  STRING API result:")
                    for _, row in api_result.iterrows():
                        print(f"    stringId:      {row.get('stringId', 'N/A')}")
                        print(f"    preferredName: {row.get('preferredName', 'N/A')}")
                        print(f"    annotation:    {str(row.get('annotation', 'N/A'))[:100]}...")
                else:
                    print(f"  STRING API: identifier not found")
            except ImportError:
                print("  Warning: string_api module not available", file=sys.stderr)
            except Exception as e:
                print(f"  Warning: STRING API query failed: {e}", file=sys.stderr)

    # A.4.2: Cross-validate local interactions against STRING API
    if args.cross_validate_api > 0:
        try:
            import random
            from string_api import get_interaction_partners, get_version, StringAPIError

            version = get_version()
            print(f"\nCross-validation against STRING API v{version.get('string_version', '?')}",
                  file=sys.stderr)

            stats = mapper.get_mapping_stats()
            ensp_ids = list(mapper._ensp_to_uniprot.keys())
            if not ensp_ids:
                print("  No ENSP IDs available for cross-validation", file=sys.stderr)
            else:
                sample_size = min(args.cross_validate_api, len(ensp_ids))
                sampled = random.sample(ensp_ids, sample_size)
                agree, disagree, api_fail = 0, 0, 0

                for ensp in sampled:
                    full_id = f"9606.{ensp}"
                    try:
                        partners = get_interaction_partners(
                            [full_id], required_score=700, limit=5,
                        )
                        if len(partners) > 0:
                            agree += 1
                        else:
                            disagree += 1
                    except Exception:
                        api_fail += 1

                print(f"\n  Cross-validation results ({sample_size} proteins):")
                print(f"    Has API partners (score>=700): {agree}")
                print(f"    No API partners:               {disagree}")
                print(f"    API errors:                    {api_fail}")

        except ImportError:
            print("  Warning: string_api module not available", file=sys.stderr)
        except Exception as e:
            print(f"  Warning: cross-validation failed: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
