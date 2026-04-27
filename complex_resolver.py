"""Discover paired AlphaFold2 PDB/PKL files in flat or sharded layouts.

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
import sys
from dataclasses import dataclass
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
    reason: str | None   # None if complete


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


def _classify(files: ResolvedFiles) -> tuple[str | None, int, int]:
    """Return (reason, pdb_size, pkl_size). reason=None means complete."""
    if files.pdb_ambiguous and files.pkl_ambiguous:
        return "ambiguous_pdb", 0, 0
    if files.pdb_ambiguous:
        return "ambiguous_pdb", 0, (files.pkl.stat().st_size if files.pkl else 0)
    if files.pkl_ambiguous:
        return "ambiguous_pkl", (files.pdb.stat().st_size if files.pdb else 0), 0

    pdb_missing = files.pdb is None
    pkl_missing = files.pkl is None
    if pdb_missing and pkl_missing:
        return "missing_both", 0, 0
    if pdb_missing:
        return "missing_pdb", 0, files.pkl.stat().st_size
    if pkl_missing:
        return "missing_pkl", files.pdb.stat().st_size, 0

    pdb_size = files.pdb.stat().st_size
    pkl_size = files.pkl.stat().st_size
    if pdb_size == 0 and pkl_size == 0:
        return "empty_both", 0, 0
    if pdb_size == 0:
        return "empty_pdb", 0, pkl_size
    if pkl_size == 0:
        return "empty_pkl", pdb_size, 0

    return None, pdb_size, pkl_size


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
    ]


def _row_for_incomplete(rec: CandidateRecord) -> list[str]:
    return _row_for_manifest(rec) + [rec.reason or ""]


MANIFEST_HEADER = [
    "name", "layout", "shard", "complex_dir",
    "pdb_path", "pkl_path", "pdb_size_bytes", "pkl_size_bytes",
]
INCOMPLETE_HEADER = MANIFEST_HEADER + ["reason"]


def find_complexes(
    root: Path | str | None = None,
    audit_dir: Path | str | None = None,
    write_audit: bool = True,
    progress_stream=None,
) -> dict[str, dict[str, Path]]:
    """Scan a flat or sharded complex root and return complete pairs.

    Parameters
    ----------
    root
        Explicit root override. Defaults to ``PROTEIN_COMPLEXES_ROOT``;
        raises ``RuntimeError`` if neither is set.
    audit_dir
        Where to write ``complex_manifest.tsv`` and ``incomplete_inputs.tsv``.
        Defaults to ``$PROTEIN_TOOLKIT_PROJECT_ROOT/data/complex_manifest_audit/``,
        falling back to ``$CWD/data/complex_manifest_audit/``.
    write_audit
        Set ``False`` to suppress manifest writes (useful in tests / when
        embedding inside another tool).
    progress_stream
        File-like object to receive scan progress lines (one per
        ``PROGRESS_INTERVAL`` directories). ``None`` is silent.

    Returns
    -------
    dict
        ``{complex_name: {"pdb": Path, "pkl": Path}}`` for complete pairs only,
        keyed and ordered alphabetically.
    """
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
        reason, pdb_size, pkl_size = _classify(files)

        record = CandidateRecord(
            name=name,
            layout=layout,
            shard=shard,
            complex_dir=complex_dir,
            pdb=files.pdb,
            pkl=files.pkl,
            pdb_size=pdb_size,
            pkl_size=pkl_size,
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

    if write_audit:
        out_dir = _resolve_audit_dir(audit_dir)
        manifest_rows = [
            _row_for_manifest(complete_records[name])
            for name in sorted(complete_records)
        ]
        incomplete_rows = [_row_for_incomplete(rec) for rec in incomplete]
        _atomic_write_tsv(
            out_dir / MANIFEST_FILE, MANIFEST_HEADER, manifest_rows,
        )
        _atomic_write_tsv(
            out_dir / INCOMPLETE_FILE, INCOMPLETE_HEADER, incomplete_rows,
        )

    return {
        name: {"pdb": complete_records[name].pdb, "pkl": complete_records[name].pkl}
        for name in sorted(complete_records)
    }


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
        help="Override audit directory (default: data/complex_manifest_audit/).",
    )
    args = parser.parse_args()

    pairs = find_complexes(
        root=args.root,
        audit_dir=args.audit_dir,
        write_audit=True,
        progress_stream=sys.stderr,
    )

    resolved_root = _resolve_root(args.root)
    layout = _detect_layout(resolved_root)
    audit_dir = _resolve_audit_dir(args.audit_dir)

    n_complete = len(pairs)
    print(f"Layout:               {layout}")
    print(f"Complete complexes:   {n_complete:,}")
    print(f"Manifest:             {audit_dir / MANIFEST_FILE}")
    print(f"Incomplete audit:     {audit_dir / INCOMPLETE_FILE}")

    sys.exit(0 if n_complete > 0 else 1)


if __name__ == "__main__":
    main()
