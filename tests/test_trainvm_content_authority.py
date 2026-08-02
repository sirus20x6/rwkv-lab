from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import pytest

from rwkv_lab.trainvm_adapters.content_authority import (
    ContentAuthorityError,
    InputContentRootIdentity,
    measure_input_content_root,
    verify_input_content_roots,
)


def test_nested_tree_is_deterministic_and_strictly_verified(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    (root / "nested").mkdir(parents=True)
    (root / "z.txt").write_bytes(b"last")
    (root / "nested" / "a.txt").write_bytes(b"first")
    (root / "nested" / "zero.bin").write_bytes(b"")

    first = measure_input_content_root(root)
    second = measure_input_content_root(root)

    assert first == second
    assert isinstance(first, InputContentRootIdentity)
    assert first.kind == "directory"
    assert first.file_count == 3
    assert first.total_bytes == 9
    assert (
        first.tree_sha256
        == "sha256:1b094ce7b36a2ce144606c2c9028454bdde962191f2958dc5e5c9b8d90ed5f99"
    )
    assert verify_input_content_roots([asdict(first)]) == (first,)


def test_same_path_content_mutation_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "mutable"
    root.mkdir()
    content = root / "sample.txt"
    content.write_bytes(b"before")
    declared = measure_input_content_root(root)

    content.write_bytes(b"after!")

    with pytest.raises(ContentAuthorityError, match="no longer matches"):
        verify_input_content_roots([asdict(declared)])


def test_root_nested_symlinks_special_nodes_and_empty_roots_are_rejected(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (target / "data.bin").write_bytes(b"data")
    root_link = tmp_path / "root-link"
    root_link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ContentAuthorityError, match="symbolic link"):
        measure_input_content_root(root_link)

    nested = tmp_path / "nested-link"
    nested.mkdir()
    (nested / "data-link").symlink_to(target / "data.bin")
    with pytest.raises(ContentAuthorityError, match="symbolic link"):
        measure_input_content_root(nested)

    fifo_root = tmp_path / "fifo-root"
    fifo_root.mkdir()
    fifo = fifo_root / "pipe"
    fifo.parent.mkdir(exist_ok=True)
    os.mkfifo(fifo)
    with pytest.raises(ContentAuthorityError, match="special node"):
        measure_input_content_root(fifo_root)

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ContentAuthorityError, match="empty"):
        measure_input_content_root(empty)


def test_file_root_supports_zero_bytes_and_detects_substitution(tmp_path: Path) -> None:
    root = tmp_path / "single.bin"
    root.write_bytes(b"")
    identity = measure_input_content_root(root)
    assert identity.kind == "file"
    assert identity.file_count == 1
    assert identity.total_bytes == 0
    assert verify_input_content_roots([asdict(identity)]) == (identity,)

    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"replacement")
    replacement.replace(root)
    with pytest.raises(ContentAuthorityError, match="no longer matches"):
        verify_input_content_roots([asdict(identity)])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(extra=True),
        lambda value: value.pop("kind"),
        lambda value: value.update(api_version="trainvm.input-content-root/v2"),
        lambda value: value.update(file_count=True),
        lambda value: value.update(total_bytes=-1),
        lambda value: value.update(tree_sha256="sha256:" + "A" * 64),
    ],
)
def test_malformed_identity_is_rejected(
    tmp_path: Path, mutate: Callable[[dict[str, object]], object]
) -> None:
    root = tmp_path / "file.bin"
    root.write_bytes(b"x")
    raw = asdict(measure_input_content_root(root))
    mutate(raw)
    with pytest.raises(ContentAuthorityError):
        verify_input_content_roots([raw])


def test_paths_must_be_sorted_unique_and_nonoverlapping(tmp_path: Path) -> None:
    left = tmp_path / "a"
    right = tmp_path / "b"
    left.write_bytes(b"a")
    right.write_bytes(b"b")
    left_raw = asdict(measure_input_content_root(left))
    right_raw = asdict(measure_input_content_root(right))

    assert verify_input_content_roots([left_raw, right_raw])
    with pytest.raises(ContentAuthorityError, match="strictly path-sorted"):
        verify_input_content_roots([right_raw, left_raw])
    with pytest.raises(ContentAuthorityError, match="strictly path-sorted"):
        verify_input_content_roots([left_raw, left_raw])

    parent = tmp_path / "parent"
    parent.mkdir()
    child = parent / "child.bin"
    child.write_bytes(b"child")
    parent_raw = asdict(measure_input_content_root(parent))
    child_raw = asdict(measure_input_content_root(child))
    with pytest.raises(ContentAuthorityError, match="overlap"):
        verify_input_content_roots([parent_raw, child_raw])


def test_raw_container_and_canonical_path_are_strict(tmp_path: Path) -> None:
    root = tmp_path / "file.bin"
    root.write_bytes(b"x")
    raw = asdict(measure_input_content_root(root))

    with pytest.raises(ContentAuthorityError, match="nonempty list"):
        verify_input_content_roots([])
    with pytest.raises(ContentAuthorityError, match="nonempty list"):
        verify_input_content_roots((raw,))
    noncanonical = dict(raw)
    noncanonical["path"] = f"{root.parent}/./{root.name}"
    with pytest.raises(ContentAuthorityError, match="absolute, normalized"):
        verify_input_content_roots([noncanonical])
    double_slash = dict(raw)
    double_slash["path"] = "/" + str(root)
    with pytest.raises(ContentAuthorityError, match="absolute, normalized"):
        verify_input_content_roots([double_slash])
