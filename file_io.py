"""Transparent open() for plain, gzip, and bzip2 files.

The toolkit's PDB readers used to assume plain-text input. This module is
the single point of truth for opening any input file regardless of its
compression suffix. Use it everywhere a PDB or other text/binary input is
opened so that ``.pdb.bz2`` (HPC layout) and ``.pdb`` (local Test_Data)
both work without per-call dispatch.
"""

from __future__ import annotations

import bz2
import gzip
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator, TextIO


def open_text_maybe_compressed(
    path: Path | str,
    encoding: str = "utf-8",
) -> TextIO:
    """Open ``path`` as a text stream, transparently decompressing .gz/.bz2."""
    p = Path(path)
    if p.suffix == ".bz2":
        return bz2.open(p, mode="rt", encoding=encoding, errors="replace")
    if p.suffix == ".gz":
        return gzip.open(p, mode="rt", encoding=encoding, errors="replace")
    return p.open(mode="rt", encoding=encoding, errors="replace")


def open_binary_maybe_compressed(path: Path | str) -> BinaryIO:
    """Open ``path`` as a binary stream, transparently decompressing .gz/.bz2."""
    p = Path(path)
    if p.suffix == ".bz2":
        return bz2.open(p, mode="rb")
    if p.suffix == ".gz":
        return gzip.open(p, mode="rb")
    return p.open(mode="rb")


@contextmanager
def decompressed_pdb_view(pdb_path: Path | str) -> Iterator[Path]:
    """Yield a plain-text PDB path for the duration of the context.

    If ``pdb_path`` is ``.bz2`` or ``.gz``, decompress it **once** into a
    per-call temporary file and yield the temp path; otherwise yield
    ``pdb_path`` unchanged. The temp file is deleted on context exit.

    Rationale: bz2 is a slow, strictly sequential codec (~50–100 MB/s
    decoded text). ``process_single_complex`` opens the same PDB up to
    five times across different readers (``extract_plddt_from_pdb``,
    three passes inside ``read_pdb_with_chain_info_New``, and the SASA
    parser). Doing five separate bz2 decompressions per complex adds up
    at HPC scale (41k complexes × ~600 KB × 5 reads ≈ tens of minutes of
    redundant CPU work even before the rest of the pipeline runs).
    Decompressing once per complex collapses that to a single bz2 read
    and makes downstream ``open()`` calls plain-disk text IO.

    Use this at the top of any per-complex worker function that calls
    multiple PDB readers.
    """
    p = Path(pdb_path)
    if p.suffix not in (".bz2", ".gz"):
        yield p
        return

    with tempfile.NamedTemporaryFile(
        suffix=".pdb", mode="w", encoding="utf-8", delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
        with open_text_maybe_compressed(p) as src:
            shutil.copyfileobj(src, tmp)
    try:
        yield tmp_path
    finally:
        tmp_path.unlink(missing_ok=True)
