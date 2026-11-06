import json
from pathlib import Path

import pytest

from faultline.checkpoints import CheckpointError, CheckpointState, CheckpointStore
from faultline.sandbox.archive import FileEntry


def state(*, task_id: str = "task-a", content: bytes = b"code", steps: int = 3) -> CheckpointState:
    return CheckpointState(
        task_id=task_id,
        seed=7,
        protected_hashes={"tests/x.py": "0" * 64},
        controller={"steps": steps},
        ledger={"charges": {}, "shipments": [], "charge_calls": 0},
        workspace={"app.py": FileEntry(content, 0o644)},
        data={"orders.db": FileEntry(b"db", 0o644)},
    )


def test_round_trip_restores_everything(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    original = state()

    loaded = store.load(store.save(original), task_id="task-a")

    assert loaded == original


def test_identical_state_has_an_identical_id(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)

    assert store.save(state()) == store.save(state())
    assert store.save(state()) != store.save(state(content=b"other"))
    assert store.save(state()) != store.save(state(steps=4))


def test_restore_rejects_different_task(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    checkpoint_id = store.save(state(task_id="task-a"))

    with pytest.raises(CheckpointError, match="belongs to task"):
        store.load(checkpoint_id, task_id="task-b")


def test_restore_rejects_corrupt_checkpoint(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    checkpoint_id = store.save(state())

    manifest = tmp_path / "manifests" / f"{checkpoint_id}.json"
    manifest.write_bytes(manifest.read_bytes().replace(b'"seed":7', b'"seed":8'))
    with pytest.raises(CheckpointError, match="corrupt manifest"):
        store.load(checkpoint_id, task_id="task-a")


def test_restore_rejects_corrupt_blob(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    checkpoint_id = store.save(state())
    blobs = [path for path in (tmp_path / "objects").rglob("*") if path.is_file()]

    for blob in blobs:
        blob.write_bytes(b"tampered")

    with pytest.raises(CheckpointError, match="corrupt blob"):
        store.load(checkpoint_id, task_id="task-a")


def forge(tmp_path: Path, change) -> str:
    """Write a manifest with a valid checksum but hostile contents, as an attacker could."""
    from faultline.checkpoints import _canonical, _digest

    store = CheckpointStore(tmp_path)
    manifest = json.loads((tmp_path / "manifests" / f"{store.save(state())}.json").read_bytes())
    change(manifest)
    raw = _canonical(manifest)
    forged = _digest(raw)
    (tmp_path / "manifests" / f"{forged}.json").write_bytes(raw)
    return forged


def test_archive_path_traversal_is_rejected(tmp_path: Path) -> None:
    def escape(manifest: dict) -> None:
        manifest["workspace"][0]["path"] = "../../etc/cron.d/evil"

    with pytest.raises(CheckpointError, match="unsafe path"):
        CheckpointStore(tmp_path).load(forge(tmp_path, escape), task_id="task-a")


def test_blob_reference_cannot_escape_the_store(tmp_path: Path) -> None:
    def escape(manifest: dict) -> None:
        manifest["workspace"][0]["sha256"] = "../../../../etc/passwd"

    with pytest.raises(CheckpointError, match="invalid blob"):
        CheckpointStore(tmp_path).load(forge(tmp_path, escape), task_id="task-a")


def test_unknown_version_is_rejected(tmp_path: Path) -> None:
    def future(manifest: dict) -> None:
        manifest["version"] = 99

    with pytest.raises(CheckpointError, match="unsupported"):
        CheckpointStore(tmp_path).load(forge(tmp_path, future), task_id="task-a")


@pytest.mark.parametrize("checkpoint_id", ["", "nope", "../secret", "a" * 64])
def test_unknown_or_malformed_ids_fail_closed(tmp_path: Path, checkpoint_id: str) -> None:
    with pytest.raises(CheckpointError):
        CheckpointStore(tmp_path).load(checkpoint_id, task_id="task-a")


def test_branches_share_blobs(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    store.save(state(steps=1))
    before = sum(1 for path in (tmp_path / "objects").rglob("*") if path.is_file())

    store.save(state(steps=2))

    assert sum(1 for path in (tmp_path / "objects").rglob("*") if path.is_file()) == before


def test_no_temporary_files_are_left_behind(tmp_path: Path) -> None:
    CheckpointStore(tmp_path).save(state())

    assert not [path for path in tmp_path.rglob(".tmp-*")]
