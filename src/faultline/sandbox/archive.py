"""Workspace trees and tar archives. Anything read from a sandbox is treated as hostile."""

from __future__ import annotations

import hashlib
import io
import tarfile
from dataclasses import dataclass
from pathlib import PurePosixPath

MAX_TREE_BYTES = 64 * 1024 * 1024
MAX_TREE_FILES = 5_000
_IGNORED_PARTS = {"__pycache__", ".pytest_cache"}


class ArchiveError(ValueError):
    """An archive is malformed, too large, or tries to escape its root."""


@dataclass(frozen=True, slots=True)
class FileEntry:
    data: bytes
    mode: int = 0o644


Tree = dict[str, FileEntry]


def safe_relative_path(name: str) -> str:
    """Normalize an archive member name, rejecting anything that could leave the root."""
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ArchiveError(f"unsafe path in archive: {name!r}")
    normalized = PurePosixPath(*[part for part in path.parts if part != "."])
    return str(normalized)


def read_tar(archive: bytes) -> Tree:
    """Parse a tar into a tree. Links and device files are refused rather than followed."""
    if len(archive) > MAX_TREE_BYTES * 2:
        raise ArchiveError("archive is too large")
    tree: Tree = {}
    total = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
            for member in tar:
                name = safe_relative_path(member.name)
                if name == ".":
                    continue
                if _IGNORED_PARTS.intersection(PurePosixPath(name).parts):
                    continue
                if member.isdir():
                    continue
                if not member.isreg():
                    raise ArchiveError(f"archive member is not a regular file: {name!r}")
                total += member.size
                if total > MAX_TREE_BYTES or len(tree) >= MAX_TREE_FILES:
                    raise ArchiveError("archive is too large")
                extracted = tar.extractfile(member)
                if extracted is None:
                    raise ArchiveError(f"cannot read archive member: {name!r}")
                tree[name] = FileEntry(extracted.read(), member.mode & 0o755)
    except tarfile.TarError as error:
        raise ArchiveError(f"malformed archive: {error}") from error
    return tree


def build_tar(tree: Tree) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:", format=tarfile.PAX_FORMAT) as tar:
        for name in sorted(tree):
            entry = tree[name]
            info = tarfile.TarInfo(safe_relative_path(name))
            info.size = len(entry.data)
            info.mode = entry.mode
            tar.addfile(info, io.BytesIO(entry.data))
    return buffer.getvalue()


def tree_hash(tree: Tree) -> str:
    digest = hashlib.sha256()
    for name in sorted(tree):
        entry = tree[name]
        digest.update(f"{name}\0{entry.mode:o}\0".encode())
        digest.update(hashlib.sha256(entry.data).digest())
    return digest.hexdigest()
