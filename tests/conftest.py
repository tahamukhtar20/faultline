from __future__ import annotations

import json
from typing import Any

from faultline.env import ACTION_KINDS
from faultline.models import ActionKind
from faultline.sandbox.archive import FileEntry, Tree
from faultline.taskgen import TaskInstance


def action(kind: ActionKind, argument: str = "") -> dict[str, Any]:
    return {"kind": ACTION_KINDS.index(kind), "argument": argument}


def patch_action(patch: dict[str, str]) -> dict[str, Any]:
    return action(ActionKind.PATCH, json.dumps(patch))


def apply_patches(tree: Tree, patches: list[dict[str, str]] | tuple[dict[str, str], ...]) -> Tree:
    patched = dict(tree)
    for patch in patches:
        text = patched[patch["path"]].data.decode()
        assert text.count(patch["old"]) == 1, patch["old"]
        patched[patch["path"]] = FileEntry(text.replace(patch["old"], patch["new"]).encode())
    return patched


def gold_tree(instance: TaskInstance) -> Tree:
    return apply_patches(instance.files, instance.gold_patches)
