"""Content-addressed checkpoints.

A checkpoint is a versioned JSON manifest plus file blobs named by their SHA-256. Its ID is the
hash of the canonical manifest, so identical state gets an identical ID, and branches share every
blob they have in common. Everything is checked on load, and any mismatch fails closed.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from faultline.sandbox.archive import ArchiveError, FileEntry, Tree, safe_relative_path

MANIFEST_VERSION = 1
_HEX_DIGITS = frozenset("0123456789abcdef")


class CheckpointError(ValueError):
    """A checkpoint is missing, corrupt, or belongs to a different task."""


@dataclass(frozen=True, slots=True)
class CheckpointState:
    task_id: str
    seed: int
    protected_hashes: dict[str, str]
    controller: dict[str, Any]
    ledger: dict[str, Any]
    workspace: Tree
    data: Tree
    parent: str | None = None


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_hash(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX_DIGITS


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(data)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


class CheckpointStore:
    def __init__(self, root: Path):
        self.root = root

    def _blob_path(self, digest: str) -> Path:
        return self.root / "objects" / digest[:2] / digest[2:]

    def _manifest_path(self, checkpoint_id: str) -> Path:
        return self.root / "manifests" / f"{checkpoint_id}.json"

    def _store_tree(self, tree: Tree) -> list[dict[str, Any]]:
        entries = []
        for name in sorted(tree):
            entry = tree[name]
            digest = _digest(entry.data)
            blob = self._blob_path(digest)
            if not blob.exists():
                _write_atomic(blob, entry.data)
            entries.append({"path": name, "mode": entry.mode, "sha256": digest})
        return entries

    def save(self, state: CheckpointState) -> str:
        manifest = {
            "version": MANIFEST_VERSION,
            "task_id": state.task_id,
            "seed": state.seed,
            "parent": state.parent,
            "protected_hashes": state.protected_hashes,
            "controller": state.controller,
            "ledger": state.ledger,
            "workspace": self._store_tree(state.workspace),
            "data": self._store_tree(state.data),
        }
        checkpoint_id = _digest(_canonical(manifest))
        _write_atomic(self._manifest_path(checkpoint_id), _canonical(manifest))
        return checkpoint_id

    def _load_tree(self, entries: Any) -> Tree:
        tree: Tree = {}
        for entry in entries:
            digest = entry["sha256"]
            if not _is_hash(digest):
                raise CheckpointError(f"invalid blob reference: {digest!r}")
            try:
                data = self._blob_path(digest).read_bytes()
                name = safe_relative_path(entry["path"])
            except OSError as error:
                raise CheckpointError(f"missing blob {digest}") from error
            except ArchiveError as error:
                raise CheckpointError(str(error)) from error
            if _digest(data) != digest:
                raise CheckpointError(f"corrupt blob {digest}")
            tree[name] = FileEntry(data, int(entry["mode"]) & 0o755)
        return tree

    def load(self, checkpoint_id: str, *, task_id: str) -> CheckpointState:
        if not _is_hash(checkpoint_id):
            raise CheckpointError(f"invalid checkpoint ID: {checkpoint_id!r}")
        try:
            raw = self._manifest_path(checkpoint_id).read_bytes()
        except OSError as error:
            raise CheckpointError(f"unknown checkpoint: {checkpoint_id}") from error
        if _digest(raw) != checkpoint_id:
            raise CheckpointError(f"corrupt manifest for checkpoint {checkpoint_id}")
        try:
            manifest = json.loads(raw)
            if manifest["version"] != MANIFEST_VERSION:
                raise CheckpointError(f"unsupported manifest version: {manifest['version']}")
            if manifest["task_id"] != task_id:
                raise CheckpointError(
                    f"checkpoint belongs to task {manifest['task_id']!r}, not {task_id!r}"
                )
            return CheckpointState(
                task_id=manifest["task_id"],
                seed=manifest["seed"],
                protected_hashes=manifest["protected_hashes"],
                controller=manifest["controller"],
                ledger=manifest["ledger"],
                workspace=self._load_tree(manifest["workspace"]),
                data=self._load_tree(manifest["data"]),
                parent=manifest["parent"],
            )
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, CheckpointError):
                raise
            raise CheckpointError(f"malformed manifest: {error!r}") from error
