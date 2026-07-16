from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def load_alignment():
    path = Path(__file__).resolve().parents[1] / "scripts/midjourney_alignment.py"
    spec = importlib.util.spec_from_file_location("test_midjourney_alignment_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_four_way_assignment_recovers_permuted_siblings():
    alignment = load_alignment()
    source = [
        ["orange halves, blue pruning tool, white tray, green tiles"],
        ["coffee mug, orange baseball, cocoa powder, wooden bat"],
        ["coffee mug, white baseball, scissors, teal and orange court"],
        ["orange baseball, black scissors, brown paper, coffee bean"],
    ]
    i1 = [source[1], source[3], source[2], source[0]]
    result = alignment.align_group(source, i1)
    assert result.row_to_suffix == (3, 0, 2, 1)
    assert result.margin > 0


@pytest.mark.parametrize("source_count,i1_count", [(3, 4), (4, 3), (5, 4)])
def test_alignment_rejects_non_four_image_groups(source_count, i1_count):
    alignment = load_alignment()
    with pytest.raises(ValueError, match="exactly four"):
        alignment.align_group(
            [[f"source {index}"] for index in range(source_count)],
            [[f"target {index}"] for index in range(i1_count)],
        )


def test_alignment_rejects_missing_caption_evidence():
    alignment = load_alignment()
    with pytest.raises(ValueError, match="source image"):
        alignment.align_group(
            [["first"], ["second"], [], ["fourth"]],
            [["one"], ["two"], ["three"], ["four"]],
        )


def test_alignment_infers_one_missing_i1_suffix():
    alignment = load_alignment()
    source = [["red apple"], ["blue car"], ["green tree"], ["white horse"]]
    targets = [source[2], source[0], [], source[1]]
    result = alignment.align_group(source, targets)
    assert result.row_to_suffix == (1, 3, 0, 2)


def test_alignment_rejects_two_missing_i1_suffixes():
    alignment = load_alignment()
    with pytest.raises(ValueError, match="at least three"):
        alignment.align_group(
            [["first"], ["second"], ["third"], ["fourth"]],
            [["one"], ["two"], [], []],
        )
