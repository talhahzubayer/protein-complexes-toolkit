"""Discover paired AlphaFold2 PDB/PKL files in flat or sharded layouts.

Operational / audit infrastructure. ``toolkit.py`` calls ``find_complexes`` (via
``find_paired_data_files``) to enumerate the complexes to process in the
flat-dir and sharded HPC layouts, and the fingerprinted manifest it writes
underpins the incremental ``--skip-existing`` run mode. It discovers inputs and
records an audit trail; it is not itself a reported result.

Library API
-----------

    from complex_resolver import find_complexes
    pairs = find_complexes(root="/scratch/.../Protein_Complexes")
    # {complex_name: {"pdb": Path, "pkl": Path}}

Library mode is silent (no stdout, no stderr progress).

Script mode
-----------

    PROTEIN_COMPLEXES_ROOT=/scratch/.../Protein_Complexes \\
        python complex_resolver.py

Writes a forensic manifest (`complex_manifest.tsv`) and an audit of
incomplete inputs (`incomplete_inputs.tsv`) to
``data/complex_manifest_audit/`` (or to a directory derived from
``PROTEIN_TOOLKIT_PROJECT_ROOT``). Exits 0 if at least one complete pair is
found, 1 otherwise.

Layout
------

A directory child whose name matches ``^[A-Z0-9]{2}$`` is treated as a shard;
the resolver descends into shards and treats their grandchildren as complex
directories. Otherwise the layout is flat (root's children are complex
directories).

Discovery
---------

Inside each complex directory, candidate filenames are tried in priority
order — uncompressed wins if both compressed and uncompressed are present:

PDB:
    {complex_name}.pdb
    {complex_name}.pdb.bz2
    *_relaxed_model_*.pdb
    *_relaxed_model_*.pdb.bz2

PKL:
    {complex_name}.pkl
    {complex_name}.pkl.bz2
    {complex_name}.results.pkl
    {complex_name}.results.pkl.bz2
    *_result_model_*.pkl
    *_result_model_*.pkl.bz2

Glob patterns that match more than one file produce ``ambiguous_pdb`` /
``ambiguous_pkl`` audit reasons. Exact-name patterns never produce ambiguity.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

SHARD_RE = re.compile(r"^[A-Z0-9]{2}$")

PDB_EXACT_PATTERNS = (".pdb", ".pdb.bz2")
PDB_GLOB_PATTERNS = ("*_relaxed_model_*.pdb", "*_relaxed_model_*.pdb.bz2")

PKL_EXACT_PATTERNS = (
    ".pkl",
    ".pkl.bz2",
    ".results.pkl",
    ".results.pkl.bz2",
)
PKL_GLOB_PATTERNS = ("*_result_model_*.pkl", "*_result_model_*.pkl.bz2")

MANIFEST_DIR_NAME = "data/complex_manifest_audit"
MANIFEST_FILE = "complex_manifest.tsv"
INCOMPLETE_FILE = "incomplete_inputs.tsv"
SUMMARY_FILE = "manifest_snapshot_summary.txt"
RUNS_DIR_NAME = "runs"
LATEST_DIR_NAME = "latest"
LATEST_RUN_ID_FILE = "latest_run_id.txt"

VALID_PURPOSES = ("baseline", "incremental")

PROGRESS_INTERVAL = 5_000


@dataclass(frozen=True)
class ResolvedFiles:
    pdb: Path | None
    pkl: Path | None
    pdb_ambiguous: bool = False
    pkl_ambiguous: bool = False


@dataclass
class CandidateRecord:
    name: str
    layout: str          # "flat" | "sharded"
    shard: str           # "" if flat
    complex_dir: Path
    pdb: Path | None
    pkl: Path | None
    pdb_size: int
    pkl_size: int
    pdb_mtime: int       # st_mtime_ns; 0 when file missing/ambiguous/unstattable
    pkl_mtime: int
    reason: str | None   # None if complete


@dataclass(frozen=True)
class DiscoveryResult:
    """Returned from find_complexes(..., return_audit=True).

    Carries the discovered pairs plus the run-id and run-dir paths the resolver
    just wrote, so callers (toolkit incremental mode) don't have to glob
    runs/* or read latest_run_id.txt to back-derive them.
    """
    complexes: dict[str, dict[str, Path]]
    run_id: str
    run_dir: Path | None       # None when write_audit=False
    manifest_path: Path | None # None when write_audit=False


def _resolve_root(root: Path | str | None) -> Path:
    if root is not None:
        path = Path(root).expanduser().resolve()
    else:
        env = os.environ.get("PROTEIN_COMPLEXES_ROOT")
        if not env:
            raise RuntimeError(
                "complex_resolver.find_complexes() requires either an "
                "explicit `root` argument or the PROTEIN_COMPLEXES_ROOT "
                "environment variable."
            )
        path = Path(env).expanduser().resolve()

    if not path.is_dir():
        raise FileNotFoundError(
            f"PROTEIN_COMPLEXES_ROOT / root resolves to {str(path)!r}, "
            f"but directory does not exist"
        )
    return path


def _resolve_audit_dir(audit_dir: Path | str | None) -> Path:
    if audit_dir is not None:
        return Path(audit_dir).expanduser().resolve()

    env_root = os.environ.get("PROTEIN_TOOLKIT_PROJECT_ROOT")
    if env_root:
        base = Path(env_root).expanduser().resolve()
    else:
        base = Path.cwd()
    return base / MANIFEST_DIR_NAME


def _detect_layout(root: Path) -> str:
    for child in root.iterdir():
        if child.is_dir() and SHARD_RE.match(child.name):
            return "sharded"
    return "flat"


def _iter_complex_dirs(root: Path, layout: str):
    if layout == "flat":
        for child in sorted(root.iterdir()):
            if child.is_dir():
                yield "", child
        return

    for shard in sorted(root.iterdir()):
        if not (shard.is_dir() and SHARD_RE.match(shard.name)):
            continue
        for complex_dir in sorted(shard.iterdir()):
            if complex_dir.is_dir():
                yield shard.name, complex_dir


def _resolve_files(complex_dir: Path, complex_name: str) -> ResolvedFiles:
    pdb: Path | None = None
    pdb_ambiguous = False
    for suffix in PDB_EXACT_PATTERNS:
        candidate = complex_dir / f"{complex_name}{suffix}"
        if candidate.is_file():
            pdb = candidate
            break
    if pdb is None:
        for pattern in PDB_GLOB_PATTERNS:
            matches = sorted(complex_dir.glob(pattern))
            if len(matches) == 1:
                pdb = matches[0]
                break
            if len(matches) > 1:
                pdb_ambiguous = True
                break

    pkl: Path | None = None
    pkl_ambiguous = False
    for suffix in PKL_EXACT_PATTERNS:
        candidate = complex_dir / f"{complex_name}{suffix}"
        if candidate.is_file():
            pkl = candidate
            break
    if pkl is None:
        for pattern in PKL_GLOB_PATTERNS:
            matches = sorted(complex_dir.glob(pattern))
            if len(matches) == 1:
                pkl = matches[0]
                break
            if len(matches) > 1:
                pkl_ambiguous = True
                break

    return ResolvedFiles(
        pdb=pdb,
        pkl=pkl,
        pdb_ambiguous=pdb_ambiguous,
        pkl_ambiguous=pkl_ambiguous,
    )


def _stat_or_zero(path: Path | None) -> tuple[int, int]:
    """Return (size_bytes, mtime_ns) for ``path``, or (0, 0) if missing/unstattable.

    Single stat() call so size and mtime are read consistently together.
    """
    if path is None:
        return 0, 0
    try:
        st = path.stat()
    except OSError:
        return 0, 0
    return st.st_size, st.st_mtime_ns


def _classify(files: ResolvedFiles) -> tuple[str | None, int, int, int, int]:
    """Return (reason, pdb_size, pkl_size, pdb_mtime_ns, pkl_mtime_ns).

    ``reason=None`` means complete. Sizes/mtimes are zero when the corresponding
    file is missing or ambiguous.
    """
    pdb_size, pdb_mtime = _stat_or_zero(files.pdb)
    pkl_size, pkl_mtime = _stat_or_zero(files.pkl)

    if files.pdb_ambiguous and files.pkl_ambiguous:
        return "ambiguous_pdb", 0, 0, 0, 0
    if files.pdb_ambiguous:
        return "ambiguous_pdb", 0, pkl_size, 0, pkl_mtime
    if files.pkl_ambiguous:
        return "ambiguous_pkl", pdb_size, 0, pdb_mtime, 0

    pdb_missing = files.pdb is None
    pkl_missing = files.pkl is None
    if pdb_missing and pkl_missing:
        return "missing_both", 0, 0, 0, 0
    if pdb_missing:
        return "missing_pdb", 0, pkl_size, 0, pkl_mtime
    if pkl_missing:
        return "missing_pkl", pdb_size, 0, pdb_mtime, 0

    if pdb_size == 0 and pkl_size == 0:
        return "empty_both", 0, 0, 0, 0
    if pdb_size == 0:
        return "empty_pdb", 0, pkl_size, 0, pkl_mtime
    if pkl_size == 0:
        return "empty_pkl", pdb_size, 0, pdb_mtime, 0

    return None, pdb_size, pkl_size, pdb_mtime, pkl_mtime


def _atomic_write_tsv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(row) + "\n")
    tmp.replace(path)


def _row_for_manifest(rec: CandidateRecord) -> list[str]:
    return [
        rec.name,
        rec.layout,
        rec.shard,
        str(rec.complex_dir),
        str(rec.pdb) if rec.pdb else "",
        str(rec.pkl) if rec.pkl else "",
        str(rec.pdb_size),
        str(rec.pkl_size),
        str(rec.pdb_mtime),
        str(rec.pkl_mtime),
    ]


def _row_for_incomplete(rec: CandidateRecord) -> list[str]:
    return _row_for_manifest(rec) + [rec.reason or ""]


MANIFEST_HEADER = [
    "name", "layout", "shard", "complex_dir",
    "pdb_path", "pkl_path", "pdb_size_bytes", "pkl_size_bytes",
    "pdb_mtime_ns", "pkl_mtime_ns",
]
INCOMPLETE_HEADER = MANIFEST_HEADER + ["reason"]


def _make_run_id(purpose: str) -> str:
    """Build the auto run-id ``<YYYY-MM-DD>_<HHMMSS>[_job_<id>]_<purpose>``."""
    if purpose not in VALID_PURPOSES:
        raise ValueError(
            f"purpose must be one of {VALID_PURPOSES}, got {purpose!r}"
        )
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        return f"{stamp}_job_{job_id}_{purpose}"
    return f"{stamp}_{purpose}"


def _refresh_latest(audit_root: Path, run_dir: Path, run_id: str) -> None:
    """Mirror ``run_dir`` into ``audit_root/latest/`` and update the pointer.

    Best-effort. ``runs/<run_id>/`` is the authoritative snapshot; if the mirror
    swap fails, a stderr warning is emitted and the previous ``latest/`` is left
    intact. ``latest_run_id.txt`` is the durable single-line pointer that always
    resolves to the active run.
    """
    latest_dir = audit_root / LATEST_DIR_NAME
    latest_tmp = audit_root / (LATEST_DIR_NAME + ".tmp")
    latest_old = audit_root / (LATEST_DIR_NAME + ".old")
    pointer = audit_root / LATEST_RUN_ID_FILE

    try:
        if latest_tmp.exists():
            shutil.rmtree(latest_tmp)
        latest_tmp.mkdir(parents=True)
        for item in run_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, latest_tmp / item.name)

        if latest_dir.exists():
            if latest_old.exists():
                shutil.rmtree(latest_old)
            latest_dir.rename(latest_old)
        latest_tmp.rename(latest_dir)
        if latest_old.exists():
            shutil.rmtree(latest_old, ignore_errors=True)
    except OSError as exc:
        print(
            f"Warning: failed to refresh {latest_dir} ({exc}); "
            f"runs/{run_id}/ remains authoritative",
            file=sys.stderr,
        )
        return

    try:
        pointer_tmp = pointer.with_suffix(pointer.suffix + ".tmp")
        pointer_tmp.write_text(run_id + "\n", encoding="utf-8")
        pointer_tmp.replace(pointer)
    except OSError as exc:
        print(
            f"Warning: failed to update {pointer} ({exc}); "
            f"runs/{run_id}/ remains authoritative",
            file=sys.stderr,
        )


def _write_snapshot_summary(
    summary_path: Path,
    run_id: str,
    purpose: str,
    layout: str,
    resolved_root: Path,
    n_complete: int,
    n_incomplete: int,
) -> None:
    """Write a small human-readable summary alongside the run's TSVs."""
    lines = [
        f"run_id:           {run_id}",
        f"purpose:          {purpose}",
        f"timestamp:        {datetime.now().isoformat(timespec='seconds')}",
        f"slurm_job_id:     {os.environ.get('SLURM_JOB_ID', 'none')}",
        f"layout:           {layout}",
        f"dataset_root:     {resolved_root}",
        f"complete_pairs:   {n_complete}",
        f"incomplete:       {n_incomplete}",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def find_complexes(
    root: Path | str | None = None,
    audit_dir: Path | str | None = None,
    write_audit: bool = True,
    progress_stream=None,
    *,
    purpose: str = "baseline",
    run_id: str | None = None,
    return_audit: bool = False,
):
    """Scan a flat or sharded complex root and return complete pairs.

    Parameters
    ----------
    root
        Explicit root override. Defaults to ``PROTEIN_COMPLEXES_ROOT``;
        raises ``RuntimeError`` if neither is set.
    audit_dir
        Audit-root override. Defaults to
        ``$PROTEIN_TOOLKIT_PROJECT_ROOT/data/complex_manifest_audit/`` (else
        ``$CWD/data/complex_manifest_audit/``). When ``write_audit`` is true,
        manifests are written to ``<audit_root>/runs/<run_id>/`` and mirrored
        to ``<audit_root>/latest/`` with a ``latest_run_id.txt`` pointer.
    write_audit
        Set ``False`` to suppress manifest writes (tests / embedding).
    progress_stream
        File-like object to receive scan progress lines (one per
        ``PROGRESS_INTERVAL`` directories). ``None`` is silent.
    purpose
        ``"baseline"`` or ``"incremental"``. Drives the auto run-id suffix.
    run_id
        Override the auto-generated run id (test/dev only).
    return_audit
        When ``False`` (default), return the legacy bare dict for backward
        compatibility. When ``True``, return a :class:`DiscoveryResult` carrying
        the run id and run-dir paths the resolver just wrote.

    Returns
    -------
    dict | DiscoveryResult
        See ``return_audit``. The dict form is
        ``{complex_name: {"pdb": Path, "pkl": Path}}``, sorted alphabetically.
    """
    if purpose not in VALID_PURPOSES:
        raise ValueError(
            f"purpose must be one of {VALID_PURPOSES}, got {purpose!r}"
        )

    resolved_root = _resolve_root(root)
    layout = _detect_layout(resolved_root)

    seen: dict[str, CandidateRecord] = {}
    duplicate_names: set[str] = set()
    incomplete: list[CandidateRecord] = []
    complete_records: dict[str, CandidateRecord] = {}

    scanned = 0
    for shard, complex_dir in _iter_complex_dirs(resolved_root, layout):
        scanned += 1
        if progress_stream and scanned % PROGRESS_INTERVAL == 0:
            print(
                f"Scanned {scanned:,} complex directories...",
                file=progress_stream,
                flush=True,
            )

        name = complex_dir.name
        files = _resolve_files(complex_dir, name)
        reason, pdb_size, pkl_size, pdb_mtime, pkl_mtime = _classify(files)

        record = CandidateRecord(
            name=name,
            layout=layout,
            shard=shard,
            complex_dir=complex_dir,
            pdb=files.pdb,
            pkl=files.pkl,
            pdb_size=pdb_size,
            pkl_size=pkl_size,
            pdb_mtime=pdb_mtime,
            pkl_mtime=pkl_mtime,
            reason=reason,
        )

        if name in duplicate_names:
            # Already audited as a duplicate; ignore further sightings silently.
            continue

        if name in seen:
            duplicate_names.add(name)
            previous = seen[name]
            complete_records.pop(name, None)
            incomplete[:] = [r for r in incomplete if r.name != name]
            incomplete.append(
                CandidateRecord(
                    **{**previous.__dict__, "reason": "duplicate_complex_name"}
                )
            )
            incomplete.append(
                CandidateRecord(
                    **{**record.__dict__, "reason": "duplicate_complex_name"}
                )
            )
            continue

        seen[name] = record
        if reason is None:
            complete_records[name] = record
        else:
            incomplete.append(record)

    actual_run_id = run_id if run_id is not None else _make_run_id(purpose)
    run_dir: Path | None = None
    manifest_path: Path | None = None

    if write_audit:
        audit_root = _resolve_audit_dir(audit_dir)
        run_dir = audit_root / RUNS_DIR_NAME / actual_run_id
        manifest_path = run_dir / MANIFEST_FILE

        manifest_rows = [
            _row_for_manifest(complete_records[name])
            for name in sorted(complete_records)
        ]
        incomplete_rows = [_row_for_incomplete(rec) for rec in incomplete]
        _atomic_write_tsv(manifest_path, MANIFEST_HEADER, manifest_rows)
        _atomic_write_tsv(
            run_dir / INCOMPLETE_FILE, INCOMPLETE_HEADER, incomplete_rows,
        )
        _write_snapshot_summary(
            run_dir / SUMMARY_FILE,
            run_id=actual_run_id,
            purpose=purpose,
            layout=layout,
            resolved_root=resolved_root,
            n_complete=len(complete_records),
            n_incomplete=len(incomplete),
        )
        _refresh_latest(audit_root, run_dir, actual_run_id)

    complexes = {
        name: {"pdb": complete_records[name].pdb, "pkl": complete_records[name].pkl}
        for name in sorted(complete_records)
    }

    if return_audit:
        return DiscoveryResult(
            complexes=complexes,
            run_id=actual_run_id,
            run_dir=run_dir,
            manifest_path=manifest_path,
        )
    return complexes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan a complex root and write a forensic manifest.",
    )
    parser.add_argument(
        "--root", default=None,
        help="Override PROTEIN_COMPLEXES_ROOT.",
    )
    parser.add_argument(
        "--audit-dir", default=None,
        help="Override audit-root directory "
             "(default: data/complex_manifest_audit/).",
    )
    parser.add_argument(
        "--purpose", choices=list(VALID_PURPOSES), default="baseline",
        help="Run purpose suffix in the auto-generated run id "
             "(default: baseline).",
    )
    parser.add_argument(
        "--run-id", default=None,
        help="Override the auto-generated run id. Intended for tests and "
             "reproducible fixture generation only; production runs should "
             "use the auto-generated id.",
    )
    args = parser.parse_args()

    discovery = find_complexes(
        root=args.root,
        audit_dir=args.audit_dir,
        write_audit=True,
        progress_stream=sys.stderr,
        purpose=args.purpose,
        run_id=args.run_id,
        return_audit=True,
    )

    resolved_root = _resolve_root(args.root)
    layout = _detect_layout(resolved_root)
    audit_dir = _resolve_audit_dir(args.audit_dir)

    n_complete = len(discovery.complexes)
    print(f"Layout:               {layout}")
    print(f"Complete complexes:   {n_complete:,}")
    print(f"Run id:               {discovery.run_id}")
    print(f"Run dir:              {discovery.run_dir}")
    print(f"Latest mirror:        {audit_dir / LATEST_DIR_NAME}")
    print(f"Latest pointer:       {audit_dir / LATEST_RUN_ID_FILE}")

    sys.exit(0 if n_complete > 0 else 1)


if __name__ == "__main__":
    main()
