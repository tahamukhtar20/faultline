import io
import tarfile

import pytest

from faultline.sandbox.archive import ArchiveError, FileEntry, build_tar, read_tar, tree_hash


def tar_with(member: tarfile.TarInfo, data: bytes = b"") -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:") as tar:
        member.size = len(data)
        tar.addfile(member, io.BytesIO(data))
    return buffer.getvalue()


def test_round_trip_keeps_content_and_mode() -> None:
    tree = {"a/b.py": FileEntry(b"x", 0o644), "run.sh": FileEntry(b"y", 0o755)}

    assert read_tar(build_tar(tree)) == tree


@pytest.mark.parametrize("name", ["../evil", "a/../../evil", "/etc/passwd"])
def test_path_traversal_is_rejected(name: str) -> None:
    with pytest.raises(ArchiveError):
        read_tar(tar_with(tarfile.TarInfo(name), b"x"))


@pytest.mark.parametrize(
    "kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE, tarfile.FIFOTYPE]
)
def test_links_and_devices_are_rejected(kind: bytes) -> None:
    member = tarfile.TarInfo("link")
    member.type = kind
    member.linkname = "/etc/passwd"
    with pytest.raises(ArchiveError):
        read_tar(tar_with(member))


def test_setuid_bits_are_dropped() -> None:
    member = tarfile.TarInfo("program")
    member.mode = 0o4755

    assert read_tar(tar_with(member, b"x"))["program"].mode == 0o755


def test_garbage_is_rejected() -> None:
    with pytest.raises(ArchiveError):
        read_tar(b"this is not a tar archive" * 100)


def test_hash_depends_on_content_path_and_mode() -> None:
    base = tree_hash({"a": FileEntry(b"1")})
    assert base != tree_hash({"a": FileEntry(b"2")})
    assert base != tree_hash({"b": FileEntry(b"1")})
    assert base != tree_hash({"a": FileEntry(b"1", 0o755)})
    assert base == tree_hash({"a": FileEntry(b"1")})
