"""
FoldX local stability predictions for AlphaFold2-predicted protein-protein complexes.

Runs FoldX BuildModel locally on AF2 complex PDB structures to compute binding
affinity changes (DDG) for disease-associated variants at protein-protein
interfaces. Unlike ProtVar (Phase D.1), which provides pre-computed FoldX DDG
on monomeric AlphaFold structures, this module runs FoldX on the actual complex
to measure binding energy changes — the correct metric for interface impact.

Architecture:
    - Subprocess-based: runs foldx5_Windows/foldx_1_20270131.exe via subprocess
    - SHA256-keyed caching: every FoldX result is cached as JSON for instant reuse
    - Variant filtering: only pathogenic + VUS at confident interface contacts
      (PAE <5A + pLDDT >=70) in High/Medium tier complexes are processed
    - RepairPDB: AF2 structures are repaired once per PDB (cached) before mutation

Data sources:
    - AlphaFold2 PDB files from the pipeline (--dir)
    - FoldX binary: foldx5_Windows/foldx_1_20270131.exe + rotabase.txt

Usage (standalone):
    python foldx_scorer.py summary --cache-dir data/foldx_cache
    python foldx_scorer.py lookup --pdb complex.pdb --chain A --position 81 --wildtype K --mutant P

Usage (via toolkit.py):
    python toolkit.py --dir DIR --output results.csv --interface --pae --enrich ALIASES --variants --foldx
    python toolkit.py --dir DIR --output results.csv --interface --pae --enrich ALIASES --variants --foldx data/foldx_cache
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Optional, Union


# ── Constants ────────────────────────────────────────────────────────

# FoldX binary paths (relative to this file)
DEFAULT_FOLDX_BINARY = Path(__file__).parent / "foldx5_Windows" / "foldx_1_20270131.exe"
DEFAULT_ROTABASE = Path(__file__).parent / "foldx5_Windows" / "rotabase.txt"

# Default cache directory
DEFAULT_FOLDX_CACHE_DIR = Path(__file__).parent / "data" / "foldx_cache"

# FoldX execution parameters
FOLDX_NUMBER_OF_RUNS = 3       # Number of BuildModel runs (mean DDG reported)
FOLDX_TIMEOUT_SECONDS = 120    # Subprocess timeout per FoldX invocation

# Destabilisation threshold (kcal/mol) — convention from literature
FOLDX_DESTABILISING_THRESHOLD = 1.6

# Variant filtering thresholds — which variants to process
FOLDX_ELIGIBLE_CONTEXTS = frozenset({'interface_core', 'interface_rim'})
FOLDX_ELIGIBLE_CLINICAL = frozenset({
    'pathogenic', 'likely_pathogenic', 'likely pathogenic',
    'VUS', 'uncertain_significance', 'uncertain significance', '-',
})
FOLDX_MIN_PLDDT = 70.0
FOLDX_QUALITY_TIERS = frozenset({'High', 'Medium'})

# Display limit for variant detail strings in CSV cells
FOLDX_DETAILS_DISPLAY_LIMIT = 20

# Variant detail parsing pattern (shared with stability_scorer.py, protvar_client.py)
_VARIANT_DETAIL_PATTERN = re.compile(r'^([A-Z*])(\d+)([A-Z*]):')

# CSV column names added by this module (8 columns, per-chain a/b)
CSV_FIELDNAMES_FOLDX = [
    'foldx_ddg_mean_a', 'foldx_ddg_mean_b',
    'foldx_n_destabilising_a', 'foldx_n_destabilising_b',
    'foldx_coverage_a', 'foldx_coverage_b',
    'foldx_details_a', 'foldx_details_b',
]


# ── Section 1: Custom Exception ──────────────────────────────────────

class FoldXError(RuntimeError):
    """Raised when FoldX binary is not found, execution fails, or output
    cannot be parsed.

    Callers should catch this and fall back gracefully:

        try:
            ddg = compute_ddg_for_variant(...)
        except FoldXError as e:
            warnings.warn(f"FoldX failed: {e}")
            ddg = None
    """


# ── Section 2: Binary Validation ─────────────────────────────────────

def validate_foldx_binary(
    binary_path: Path = DEFAULT_FOLDX_BINARY,
    rotabase_path: Path = DEFAULT_ROTABASE,
) -> None:
    """Validate that the FoldX binary and rotabase.txt exist.

    Called by toolkit.py at startup when --foldx is used.

    Args:
        binary_path: Path to FoldX executable.
        rotabase_path: Path to rotabase.txt (rotamer library).

    Raises:
        FoldXError: If binary or rotabase is not found.
    """
    if not binary_path.exists():
        raise FoldXError(
            f"FoldX binary not found: {binary_path}\n"
            f"Please ensure the FoldX executable is at the expected location."
        )
    if not rotabase_path.exists():
        raise FoldXError(
            f"FoldX rotabase.txt not found: {rotabase_path}\n"
            f"The rotamer library is required for FoldX execution."
        )


# ── Section 3: Caching ───────────────────────────────────────────────

def _foldx_cache_key(
    pdb_path: Path,
    chain: str,
    position: int,
    wildtype: str,
    mutant: str,
) -> str:
    """Generate a deterministic cache key from mutation parameters.

    Args:
        pdb_path: Path to the PDB file.
        chain: Chain identifier (e.g. 'A', 'B').
        position: Residue position (PDB numbering).
        wildtype: Wildtype amino acid (single letter).
        mutant: Mutant amino acid (single letter).

    Returns:
        Hex SHA256 digest string.
    """
    # Use file name + size as PDB identity (avoids hashing multi-MB files)
    pdb_identity = f"{pdb_path.name}:{pdb_path.stat().st_size}"
    key_data = {
        "pdb": pdb_identity,
        "chain": chain,
        "position": position,
        "wildtype": wildtype,
        "mutant": mutant,
    }
    return hashlib.sha256(
        json.dumps(key_data, sort_keys=True).encode()
    ).hexdigest()


def _read_foldx_cache(cache_dir: Path, key: str) -> Optional[float]:
    """Read a cached DDG value if it exists.

    Args:
        cache_dir: Directory containing cache files.
        key: Cache key (SHA256 hex digest).

    Returns:
        DDG float value, or None if cache miss.
    """
    cache_file = cache_dir / f"{key}.json"
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, encoding="utf-8") as f:
            cached = json.load(f)
        ddg = cached.get("ddg")
        return float(ddg) if ddg is not None else None
    except (json.JSONDecodeError, KeyError, OSError, ValueError, TypeError):
        return None


def _write_foldx_cache(
    cache_dir: Path,
    key: str,
    ddg: float,
    mutation_str: str,
) -> None:
    """Write a DDG value to the cache as JSON.

    Args:
        cache_dir: Directory for cache files (created if needed).
        key: Cache key (SHA256 hex digest).
        ddg: Computed DDG value (kcal/mol).
        mutation_str: FoldX mutation string for metadata.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{key}.json"
    payload = {
        "_timestamp": datetime.now(timezone.utc).isoformat(),
        "_mutation": mutation_str,
        "ddg": ddg,
    }
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ── Section 4: Variant Filtering ─────────────────────────────────────

def _parse_variant_detail_parts(detail_str: str) -> list[dict]:
    """Parse variant_details string into list of dicts with full context.

    Parses pipe-separated variant detail strings produced by
    variant_mapper.format_variant_details(). Format:
        K81P:interface_core:pathogenic|E82K:interface_rim:VUS

    Args:
        detail_str: Pipe-separated variant detail string.

    Returns:
        List of dicts with keys: ref_aa, position, alt_aa, context, clinical.
    """
    if not detail_str:
        return []

    variants = []
    for part in detail_str.split('|'):
        part = part.strip()
        if part.startswith('...(+'):
            continue

        match = _VARIANT_DETAIL_PATTERN.match(part)
        if match:
            ref = match.group(1)
            pos = int(match.group(2))
            alt = match.group(3)

            # Extract context and clinical significance from remaining parts
            remaining = part[match.end():]
            parts = remaining.split(':')
            context = parts[0] if len(parts) >= 1 else ''
            clinical = parts[1] if len(parts) >= 2 else ''

            variants.append({
                'ref_aa': ref,
                'position': pos,
                'alt_aa': alt,
                'context': context,
                'clinical': clinical,
            })

    return variants


def _filter_variants_for_foldx(
    detail_str: str,
    quality_tier: str,
    interface_plddt: float,
) -> list[dict]:
    """Filter variants to only those eligible for FoldX analysis.

    Applies all filtering criteria:
    1. Complex quality tier must be High or Medium
    2. Variant context must be interface_core or interface_rim
    3. Clinical significance must be pathogenic, likely_pathogenic, VUS, or unknown
    4. Interface pLDDT must be >= 70
    5. Stop codons (alt_aa == '*') are excluded

    Args:
        detail_str: Pipe-separated variant detail string from variant_mapper.
        quality_tier: Complex quality tier ('High', 'Medium', 'Low').
        interface_plddt: Mean interface pLDDT value.

    Returns:
        List of eligible variant dicts (ref_aa, position, alt_aa, context, clinical).
    """
    # Gate 1: quality tier
    if quality_tier not in FOLDX_QUALITY_TIERS:
        return []

    # Gate 2: interface confidence
    try:
        if float(interface_plddt) < FOLDX_MIN_PLDDT:
            return []
    except (TypeError, ValueError):
        return []

    # Parse and filter individual variants
    all_variants = _parse_variant_detail_parts(detail_str)
    eligible = []
    for var in all_variants:
        # Skip stop codons — FoldX cannot model truncations
        if var['alt_aa'] == '*':
            continue
        # Context filter
        if var['context'] not in FOLDX_ELIGIBLE_CONTEXTS:
            continue
        # Clinical significance filter
        clinical_lower = var['clinical'].lower().strip()
        if clinical_lower not in {c.lower() for c in FOLDX_ELIGIBLE_CLINICAL}:
            continue
        eligible.append(var)

    return eligible


# ── Section 5: PDB Preparation ───────────────────────────────────────

def _strip_remarks(pdb_path: Path, output_path: Path) -> None:
    """Strip REMARK lines from a PDB file (FoldX crashes on them).

    Args:
        pdb_path: Input PDB file.
        output_path: Output PDB file with REMARK lines removed.
    """
    with open(pdb_path, 'r', encoding='utf-8', errors='replace') as infile:
        lines = infile.readlines()
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for line in lines:
            if not line.startswith('REMARK'):
                outfile.write(line)


def _run_repair_pdb(
    pdb_path: Path,
    binary_path: Path,
    rotabase_path: Path,
    cache_dir: Path,
) -> Path:
    """Run FoldX RepairPDB on an AF2 structure, returning path to repaired PDB.

    The repaired PDB is cached in cache_dir/repaired/ to avoid redundant repairs.

    Args:
        pdb_path: Path to the input PDB file.
        binary_path: Path to FoldX executable.
        rotabase_path: Path to rotabase.txt.
        cache_dir: Base cache directory.

    Returns:
        Path to the repaired PDB file.

    Raises:
        FoldXError: If RepairPDB fails.
    """
    # Check for cached repair
    repair_cache = cache_dir / "repaired"
    repair_cache.mkdir(parents=True, exist_ok=True)

    pdb_identity = f"{pdb_path.name}:{pdb_path.stat().st_size}"
    repair_key = hashlib.sha256(pdb_identity.encode()).hexdigest()[:16]
    cached_repair = repair_cache / f"{repair_key}_Repair.pdb"

    if cached_repair.exists():
        return cached_repair

    # Run RepairPDB in a temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Copy rotabase.txt to working directory (FoldX requirement)
        shutil.copy2(rotabase_path, tmpdir_path / "rotabase.txt")

        # Strip REMARK lines and copy PDB
        clean_pdb = tmpdir_path / pdb_path.name
        _strip_remarks(pdb_path, clean_pdb)

        # Write config
        config_path = tmpdir_path / "config_repair.cfg"
        with open(config_path, 'w') as f:
            f.write(f"command=RepairPDB\n")
            f.write(f"pdb={pdb_path.name}\n")
            f.write(f"output-dir={tmpdir}\n")
            f.write(f"noHeader=true\n")

        # Run FoldX
        try:
            result = subprocess.run(
                [str(binary_path), '-f', str(config_path)],
                cwd=str(tmpdir_path),
                capture_output=True,
                text=True,
                timeout=FOLDX_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            raise FoldXError(
                f"FoldX RepairPDB timed out after {FOLDX_TIMEOUT_SECONDS}s "
                f"for {pdb_path.name}"
            )

        if result.returncode != 0:
            raise FoldXError(
                f"FoldX RepairPDB failed (exit code {result.returncode}) "
                f"for {pdb_path.name}: {result.stderr[:500]}"
            )

        # Find repaired PDB
        repaired_name = pdb_path.stem + "_Repair.pdb"
        repaired_path = tmpdir_path / repaired_name
        if not repaired_path.exists():
            raise FoldXError(
                f"FoldX RepairPDB did not produce expected output: {repaired_name}"
            )

        # Cache the repaired PDB
        shutil.copy2(repaired_path, cached_repair)

    return cached_repair


# ── Section 6: FoldX Execution ───────────────────────────────────────

def _format_foldx_mutation(
    wildtype: str,
    chain: str,
    position: int,
    mutant: str,
) -> str:
    """Format a mutation in FoldX notation.

    FoldX format: {WT}{Chain}{Pos}{Mut} e.g. KA81P means
    K (wildtype) at chain A position 81 mutated to P.

    Args:
        wildtype: Wildtype amino acid (single letter).
        chain: Chain identifier (e.g. 'A').
        position: Residue position (PDB numbering).
        mutant: Mutant amino acid (single letter).

    Returns:
        FoldX mutation string, e.g. 'KA81P'.
    """
    return f"{wildtype}{chain}{position}{mutant}"


def _write_individual_list(mutations: list[str], output_path: Path) -> None:
    """Write FoldX individual_list.txt file.

    Each mutation is on its own line, terminated with a semicolon.

    Args:
        mutations: List of FoldX-format mutation strings.
        output_path: Path to write the individual list file.
    """
    with open(output_path, 'w') as f:
        for mutation in mutations:
            f.write(f"{mutation};\n")


def _run_buildmodel(
    pdb_path: Path,
    mutation_str: str,
    binary_path: Path,
    rotabase_path: Path,
    work_dir: Path,
    n_runs: int = FOLDX_NUMBER_OF_RUNS,
) -> Path:
    """Run FoldX BuildModel for a single mutation.

    Args:
        pdb_path: Path to the (repaired) PDB file.
        mutation_str: FoldX-format mutation string (e.g. 'KA81P').
        binary_path: Path to FoldX executable.
        rotabase_path: Path to rotabase.txt.
        work_dir: Working directory for FoldX files.
        n_runs: Number of BuildModel runs.

    Returns:
        Path to the Dif_*.fxout output file.

    Raises:
        FoldXError: On timeout, non-zero exit, or missing output.
    """
    # Ensure rotabase.txt is in the working directory
    rotabase_dest = work_dir / "rotabase.txt"
    if not rotabase_dest.exists():
        shutil.copy2(rotabase_path, rotabase_dest)

    # Copy PDB to working directory if not already there
    pdb_dest = work_dir / pdb_path.name
    if not pdb_dest.exists():
        shutil.copy2(pdb_path, pdb_dest)

    # Write individual_list.txt
    indiv_path = work_dir / "individual_list.txt"
    _write_individual_list([mutation_str], indiv_path)

    # Write config
    config_path = work_dir / "config_BM.cfg"
    with open(config_path, 'w') as f:
        f.write("command=BuildModel\n")
        f.write("mutant-file=individual_list.txt\n")
        f.write(f"pdb={pdb_path.name}\n")
        f.write(f"output-dir={work_dir}\n")
        f.write(f"numberOfRuns={n_runs}\n")
        f.write("noHeader=true\n")

    # Run FoldX
    try:
        result = subprocess.run(
            [str(binary_path), '-f', str(config_path)],
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=FOLDX_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        raise FoldXError(
            f"FoldX BuildModel timed out after {FOLDX_TIMEOUT_SECONDS}s "
            f"for mutation {mutation_str}"
        )

    if result.returncode != 0:
        raise FoldXError(
            f"FoldX BuildModel failed (exit code {result.returncode}) "
            f"for mutation {mutation_str}: {result.stderr[:500]}"
        )

    # Find Dif output file
    dif_pattern = f"Dif_{pdb_path.stem}.fxout"
    dif_path = work_dir / dif_pattern
    if not dif_path.exists():
        # Try alternate naming patterns
        dif_files = list(work_dir.glob("Dif_*.fxout"))
        if dif_files:
            dif_path = dif_files[0]
        else:
            raise FoldXError(
                f"FoldX BuildModel did not produce Dif_*.fxout "
                f"for mutation {mutation_str}"
            )

    return dif_path


def _parse_buildmodel_output(fxout_path: Path) -> list[float]:
    """Parse FoldX BuildModel difference output to extract DDG values.

    The Dif_*.fxout file is tab-separated. The 'total energy' column
    (column index 2, 0-based) contains the DDG value for each run.
    With noHeader=true, there may or may not be a header line — we
    skip any line that starts with 'Pdb' (header indicator).

    Args:
        fxout_path: Path to the Dif_*.fxout file.

    Returns:
        List of DDG float values (one per run).

    Raises:
        FoldXError: If file cannot be parsed.
    """
    if not fxout_path.exists():
        raise FoldXError(f"FoldX output file not found: {fxout_path}")

    ddg_values = []
    with open(fxout_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Skip header line
            if line.startswith('Pdb') or line.startswith('pdb'):
                continue
            parts = line.split('\t')
            if len(parts) >= 3:
                try:
                    ddg = float(parts[2])  # 'total energy' column
                    ddg_values.append(ddg)
                except ValueError:
                    continue

    return ddg_values


def _parse_analysecomplex_output(fxout_path: Path) -> dict:
    """Parse FoldX AnalyseComplex output for interaction energies.

    Args:
        fxout_path: Path to the Interaction_*_AC.fxout file.

    Returns:
        Dict with 'interaction_energy' and component energy keys.
        Empty dict if parsing fails.
    """
    if not fxout_path.exists():
        return {}

    with open(fxout_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('Pdb') or line.startswith('pdb'):
                continue
            parts = line.split('\t')
            if len(parts) >= 4:
                try:
                    return {
                        'interaction_energy': float(parts[3]),
                    }
                except (ValueError, IndexError):
                    continue

    return {}


# ── Section 7: DDG Computation ───────────────────────────────────────

def compute_ddg_for_variant(
    pdb_path: Path,
    chain: str,
    ref_aa: str,
    position: int,
    alt_aa: str,
    binary_path: Path = DEFAULT_FOLDX_BINARY,
    rotabase_path: Path = DEFAULT_ROTABASE,
    cache_dir: Optional[Path] = None,
) -> Optional[float]:
    """Compute DDG for a single variant using FoldX BuildModel.

    Checks cache first. On miss, runs BuildModel with N runs and returns
    the mean DDG across runs.

    Args:
        pdb_path: Path to the (repaired) PDB file.
        chain: Chain identifier (e.g. 'A').
        ref_aa: Wildtype amino acid (single letter).
        position: Residue position (PDB numbering).
        alt_aa: Mutant amino acid (single letter).
        binary_path: Path to FoldX executable.
        rotabase_path: Path to rotabase.txt.
        cache_dir: Cache directory (None = DEFAULT_FOLDX_CACHE_DIR).

    Returns:
        Mean DDG across runs (kcal/mol), or None on failure.
    """
    resolved_cache = cache_dir if cache_dir is not None else DEFAULT_FOLDX_CACHE_DIR
    mutation_str = _format_foldx_mutation(ref_aa, chain, position, alt_aa)

    # Check cache
    try:
        key = _foldx_cache_key(pdb_path, chain, position, ref_aa, alt_aa)
        cached = _read_foldx_cache(resolved_cache, key)
        if cached is not None:
            return cached
    except OSError:
        pass  # PDB file may not exist yet for cache key generation

    # Run FoldX BuildModel
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            dif_path = _run_buildmodel(
                pdb_path, mutation_str, binary_path, rotabase_path, work_dir,
            )
            ddg_values = _parse_buildmodel_output(dif_path)

        if not ddg_values:
            return None

        ddg = round(mean(ddg_values), 4)

        # Cache the result
        try:
            _write_foldx_cache(resolved_cache, key, ddg, mutation_str)
        except OSError:
            pass

        return ddg

    except FoldXError as e:
        warnings.warn(f"FoldX failed for {mutation_str}: {e}", stacklevel=2)
        return None


def compute_ddg_batch(
    repaired_pdb: Path,
    chain: str,
    variants: list[dict],
    binary_path: Path = DEFAULT_FOLDX_BINARY,
    rotabase_path: Path = DEFAULT_ROTABASE,
    cache_dir: Optional[Path] = None,
    verbose: bool = False,
) -> dict[tuple[str, int, str], Optional[float]]:
    """Compute DDG for a batch of variants on the same repaired PDB.

    Args:
        repaired_pdb: Path to the repaired PDB file.
        chain: Chain identifier.
        variants: List of variant dicts with ref_aa, position, alt_aa.
        binary_path: Path to FoldX executable.
        rotabase_path: Path to rotabase.txt.
        cache_dir: Cache directory.
        verbose: Print progress.

    Returns:
        Dict mapping (ref_aa, position, alt_aa) to DDG float or None.
    """
    results = {}

    for i, var in enumerate(variants):
        ref = var['ref_aa']
        pos = var['position']
        alt = var['alt_aa']

        if verbose and (i + 1) % 5 == 0:
            print(f"    FoldX: variant {i + 1}/{len(variants)} on chain {chain}",
                  file=sys.stderr)

        ddg = compute_ddg_for_variant(
            repaired_pdb, chain, ref, pos, alt,
            binary_path=binary_path,
            rotabase_path=rotabase_path,
            cache_dir=cache_dir,
        )
        results[(ref, pos, alt)] = ddg

    return results


# ── Section 8: Annotation ────────────────────────────────────────────

def format_foldx_details(
    scored_variants: list[dict],
    limit: int = FOLDX_DETAILS_DISPLAY_LIMIT,
) -> str:
    """Format FoldX-scored variants into a pipe-separated summary string.

    Format: REF{POS}ALT:ddg={value}:{label}

    Args:
        scored_variants: List of dicts with keys 'ref_aa', 'position', 'alt_aa',
            'ddg'.
        limit: Maximum number of variants to include.

    Returns:
        Pipe-separated string, e.g. 'K81P:ddg=2.34:destabilising|E82K:ddg=0.45:stable'.
        Empty string if no variants.
    """
    if not scored_variants:
        return ''

    details = []
    for var in scored_variants[:limit]:
        ref = var.get('ref_aa', '?')
        pos = var.get('position', '?')
        alt = var.get('alt_aa', '?')
        ddg = var.get('ddg')

        if ddg is not None:
            ddg_str = f"{ddg:.2f}"
            label = 'destabilising' if ddg > FOLDX_DESTABILISING_THRESHOLD else 'stable'
        else:
            ddg_str = '-'
            label = 'no_data'

        details.append(f"{ref}{pos}{alt}:ddg={ddg_str}:{label}")

    result = '|'.join(details)

    remaining = len(scored_variants) - limit
    if remaining > 0:
        result += f"|...(+{remaining} more)"

    return result


def _score_chain_variants_foldx(
    details_str: str,
    quality_tier: str,
    interface_plddt: float,
    pdb_path: Path,
    chain_id: str,
    binary_path: Path,
    rotabase_path: Path,
    cache_dir: Optional[Path],
    verbose: bool = False,
) -> dict:
    """Score variants for one chain using local FoldX.

    Args:
        details_str: Pipe-separated variant detail string from variant_mapper.
        quality_tier: Complex quality tier.
        interface_plddt: Mean interface pLDDT.
        pdb_path: Path to the (repaired) PDB file.
        chain_id: Chain identifier.
        binary_path: Path to FoldX executable.
        rotabase_path: Path to rotabase.txt.
        cache_dir: Cache directory.
        verbose: Print progress.

    Returns:
        Dict with keys: ddg_mean, n_destabilising, coverage, n_eligible, details.
    """
    eligible = _filter_variants_for_foldx(
        details_str, quality_tier, interface_plddt,
    )

    if not eligible:
        return {
            'ddg_mean': '',
            'n_destabilising': 0,
            'coverage': '',
            'n_eligible': 0,
            'details': '',
        }

    # Compute DDG for eligible variants
    ddg_results = compute_ddg_batch(
        pdb_path, chain_id, eligible,
        binary_path=binary_path,
        rotabase_path=rotabase_path,
        cache_dir=cache_dir,
        verbose=verbose,
    )

    # Aggregate results
    scored_variants = []
    ddg_values = []
    n_destabilising = 0

    for var in eligible:
        key = (var['ref_aa'], var['position'], var['alt_aa'])
        ddg = ddg_results.get(key)

        scored_variants.append({
            'ref_aa': var['ref_aa'],
            'position': var['position'],
            'alt_aa': var['alt_aa'],
            'ddg': ddg,
        })

        if ddg is not None:
            ddg_values.append(ddg)
            if ddg > FOLDX_DESTABILISING_THRESHOLD:
                n_destabilising += 1

    n_scored = len(ddg_values)
    n_eligible = len(eligible)
    coverage = round(n_scored / n_eligible, 4) if n_eligible > 0 else 0.0

    return {
        'ddg_mean': round(mean(ddg_values), 4) if ddg_values else '',
        'n_destabilising': n_destabilising,
        'coverage': round(coverage, 4) if n_eligible > 0 else '',
        'n_eligible': n_eligible,
        'details': format_foldx_details(scored_variants),
    }


def annotate_results_with_foldx(
    results: list[dict],
    pdb_dir: Path,
    binary_path: Path = DEFAULT_FOLDX_BINARY,
    rotabase_path: Path = DEFAULT_ROTABASE,
    cache_dir: Optional[Path] = None,
    verbose: bool = False,
) -> None:
    """Annotate result rows with local FoldX DDG values (in-place).

    Main entry point from toolkit.py. For each complex:
    1. Checks quality tier (skip Low)
    2. Finds PDB file via glob
    3. Runs RepairPDB (cached)
    4. Filters variants, computes DDG batch per chain
    5. Writes 8 CSV columns to result dict

    Args:
        results: List of per-complex result dicts. Modified in-place.
        pdb_dir: Directory containing AF2 PDB files.
        binary_path: Path to FoldX executable.
        rotabase_path: Path to rotabase.txt.
        cache_dir: Cache directory (None = DEFAULT_FOLDX_CACHE_DIR).
        verbose: Print progress to stderr.
    """
    resolved_cache = cache_dir if cache_dir is not None else DEFAULT_FOLDX_CACHE_DIR
    annotated = 0
    skipped = 0

    for row in results:
        complex_name = row.get('complex_name', '')
        quality_tier = row.get('quality_tier_v2', '')

        # Set defaults for all columns
        for suffix in ('a', 'b'):
            row[f'foldx_ddg_mean_{suffix}'] = ''
            row[f'foldx_n_destabilising_{suffix}'] = 0
            row[f'foldx_coverage_{suffix}'] = ''
            row[f'foldx_details_{suffix}'] = ''

        # Skip complexes that don't meet quality criteria
        if quality_tier not in FOLDX_QUALITY_TIERS:
            skipped += 1
            continue

        # Find PDB file
        pdb_matches = list(pdb_dir.glob(f"*{complex_name}*relaxed*.pdb"))
        if not pdb_matches:
            # Try without 'relaxed' in pattern
            pdb_matches = list(pdb_dir.glob(f"*{complex_name}*.pdb"))
        if not pdb_matches:
            if verbose:
                print(f"  FoldX: PDB not found for {complex_name}, skipping",
                      file=sys.stderr)
            skipped += 1
            continue

        pdb_path = sorted(pdb_matches)[0]  # Take first alphabetically (model_1)

        # RepairPDB (cached per structure)
        try:
            repaired_pdb = _run_repair_pdb(
                pdb_path, binary_path, rotabase_path, resolved_cache,
            )
        except FoldXError as e:
            if verbose:
                print(f"  FoldX: RepairPDB failed for {complex_name}: {e}",
                      file=sys.stderr)
            skipped += 1
            continue

        # Get chain pair
        best_pair = row.get('best_chain_pair', 'A_B')
        chain_a, chain_b = best_pair.split('_') if '_' in best_pair else ('A', 'B')
        interface_plddt = row.get('interface_plddt_mean', 0)

        # Score each chain
        for suffix, chain_id in [('a', chain_a), ('b', chain_b)]:
            details_str = row.get(f'variant_details_{suffix}', '')
            if not details_str:
                continue

            chain_result = _score_chain_variants_foldx(
                details_str, quality_tier, interface_plddt,
                repaired_pdb, chain_id, binary_path, rotabase_path,
                resolved_cache, verbose=verbose,
            )

            row[f'foldx_ddg_mean_{suffix}'] = chain_result['ddg_mean']
            row[f'foldx_n_destabilising_{suffix}'] = chain_result['n_destabilising']
            row[f'foldx_coverage_{suffix}'] = chain_result['coverage']
            row[f'foldx_details_{suffix}'] = chain_result['details']

        annotated += 1

    if verbose:
        print(f"  FoldX: annotated {annotated} complexes, "
              f"skipped {skipped} (Low tier or no PDB)", file=sys.stderr)


# ── Section 9: Standalone CLI ────────────────────────────────────────

def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for standalone use.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        prog='foldx_scorer',
        description="FoldX local stability predictions — compute DDG for "
                    "interface variants using the FoldX binary.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--foldx-binary",
        default=str(DEFAULT_FOLDX_BINARY),
        help=f"Path to FoldX executable (default: {DEFAULT_FOLDX_BINARY}).",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_FOLDX_CACHE_DIR),
        help=f"Cache directory for FoldX results (default: {DEFAULT_FOLDX_CACHE_DIR}).",
    )

    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # summary sub-command
    subparsers.add_parser("summary", help="Print FoldX cache statistics.")

    # lookup sub-command
    lookup_parser = subparsers.add_parser(
        "lookup",
        help="Compute DDG for a single mutation.",
    )
    lookup_parser.add_argument(
        "--pdb", required=True,
        help="Path to PDB file.",
    )
    lookup_parser.add_argument(
        "--chain", required=True,
        help="Chain identifier (e.g. A).",
    )
    lookup_parser.add_argument(
        "--position", required=True, type=int,
        help="Residue position.",
    )
    lookup_parser.add_argument(
        "--wildtype", required=True,
        help="Wildtype amino acid (single letter).",
    )
    lookup_parser.add_argument(
        "--mutant", required=True,
        help="Mutant amino acid (single letter).",
    )

    return parser


def _cli_summary(cache_dir: Path) -> None:
    """Print FoldX cache statistics to stdout.

    Args:
        cache_dir: Path to cache directory.
    """
    if not cache_dir.exists():
        print(f"Cache directory does not exist: {cache_dir}")
        print("No cached FoldX results.")
        return

    cache_files = list(cache_dir.glob("*.json"))
    print(f"FoldX cache: {cache_dir}")
    print(f"  Cached DDG results: {len(cache_files)}")

    # Count repaired PDBs
    repair_dir = cache_dir / "repaired"
    if repair_dir.exists():
        repaired = list(repair_dir.glob("*_Repair.pdb"))
        print(f"  Cached repaired PDBs: {len(repaired)}")

    if cache_files:
        # Count destabilising variants
        n_destab = 0
        n_total = 0
        for f in cache_files:
            try:
                with open(f, encoding="utf-8") as fh:
                    data = json.load(fh)
                ddg = data.get("ddg")
                if ddg is not None:
                    n_total += 1
                    if float(ddg) > FOLDX_DESTABILISING_THRESHOLD:
                        n_destab += 1
            except (json.JSONDecodeError, OSError, ValueError):
                pass
        print(f"  Destabilising (>{FOLDX_DESTABILISING_THRESHOLD} kcal/mol): "
              f"{n_destab}/{n_total}")


def _cli_lookup(
    pdb_path: str,
    chain: str,
    position: int,
    wildtype: str,
    mutant: str,
    binary_path: str,
    cache_dir: str,
) -> None:
    """Compute and display DDG for a single mutation.

    Args:
        pdb_path: Path to PDB file.
        chain: Chain identifier.
        position: Residue position.
        wildtype: Wildtype amino acid.
        mutant: Mutant amino acid.
        binary_path: Path to FoldX executable.
        cache_dir: Cache directory path.
    """
    pdb = Path(pdb_path)
    binary = Path(binary_path)
    rotabase = binary.parent / "rotabase.txt"
    cache = Path(cache_dir)

    print(f"Computing DDG for {wildtype}{chain}{position}{mutant} "
          f"in {pdb.name}...")

    validate_foldx_binary(binary, rotabase)

    # Repair PDB first
    print("  Running RepairPDB...")
    repaired = _run_repair_pdb(pdb, binary, rotabase, cache)
    print(f"  Repaired PDB: {repaired}")

    # Compute DDG
    ddg = compute_ddg_for_variant(
        repaired, chain, wildtype, position, mutant,
        binary_path=binary,
        rotabase_path=rotabase,
        cache_dir=cache,
    )

    if ddg is not None:
        label = "DESTABILISING" if ddg > FOLDX_DESTABILISING_THRESHOLD else "STABLE"
        print(f"\n  DDG = {ddg:.4f} kcal/mol ({label})")
        print(f"  Threshold: {FOLDX_DESTABILISING_THRESHOLD} kcal/mol")
    else:
        print("\n  DDG computation failed.")


def main() -> None:
    """CLI entry point."""
    parser = build_argument_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    cache_dir = Path(args.cache_dir)

    if args.command == "summary":
        _cli_summary(cache_dir)
    elif args.command == "lookup":
        _cli_lookup(
            args.pdb, args.chain, args.position,
            args.wildtype, args.mutant,
            args.foldx_binary, args.cache_dir,
        )


if __name__ == "__main__":
    main()
