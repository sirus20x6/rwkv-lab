"""The contract-resolution half of the schema gate must actually fire.

``scripts/validate_experiment_documents.py`` resolves every non-builtin
component/operation invocation in every example document against the registry
pin. It exists because two shipped example documents invoked
``rwkv_lab.mageflow.v1.{Train,PrepareCacheSpan,CacheEncoders}`` — a bare
``mageflow`` component no registry has ever declared — and nothing reported it,
so they sat in ``examples/`` looking authoritative.

A gate for that failure mode has to be held to its own standard, so every claim
it makes gets a test that fires in the direction it claims. Two of these matter
more than the rest:

``test_a_copied_id_does_not_inherit_its_exemption`` is the masking test. The
precedent is a gate in this repository that keyed its lookup by *filename* and
passed all three of its conjunct mutations, because one catalog held two recipes
against one contract and the healthy sibling hid the broken one. The analogue
here would be excusing a *document* rather than a *contract id*: a document
excused for one bad id could then acquire a second one, and a document that
copied a bad id out of an excused neighbour would be excused with it. The
exclusion catalog is therefore keyed by contract, and bounded by the documents
it names.

``test_both_invocation_sites_are_walked`` covers the other half. A spec names a
component and operation in ``spec.execution`` as well as in every
``workflow.nodes[*].invoke``; the mageflow documents declare the unresolvable
component in both, so a walk over only the nodes would still go red and would
still be missing half the sites.
"""

from __future__ import annotations

import json
import pathlib

import pytest

import scripts.validate_experiment_documents as gate

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent


def spec(components: dict, nodes: dict, execution: dict | None = None) -> dict:
    body = {
        "artifacts": {},
        "components": components,
        "workflow": {"entrypoint": next(iter(nodes), "n"), "nodes": nodes},
    }
    if execution is not None:
        body["execution"] = execution
    return body


def document(name: str, components: dict, nodes: dict,
             execution: dict | None = None) -> dict:
    return {
        "api_version": "trainvm.rwkv-lab/v1alpha1",
        "kind": "Experiment",
        "metadata": {"name": name},
        "spec": spec(components, nodes, execution),
    }


# A component/node pair that always resolves, so a test document is never
# entirely unresolvable. The gate fails a run in which nothing resolved -- that
# vacuity rule is tested on its own, and without this it would fire in every
# other test and blunt their assertions.
PINNED = "rwkv_lab.rwkv_scratch.v1.Train"


def resolving() -> tuple[dict, dict]:
    return ({"resolves": {"adapter": "a", "version": "1.0.0",
                          "runtime": "python_worker",
                          "operations": {"train": {"contract": PINNED}}}},
            {"resolves": {"invoke": {"component": "resolves",
                                     "operation": "train"}}})


def worker(operations: dict, runtime: str = "python_worker") -> dict:
    return {
        "adapter": "example.adapter",
        "version": "1.0.0",
        "runtime": runtime,
        "operations": {
            name: {"contract": contract} for name, contract in operations.items()
        },
    }


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A miniature docs/experiment-vm/ the gate is pointed at.

    Returns a callable taking (documents, pinned_contracts, exclusions) and
    running ``check_contract_resolution`` against them.
    """
    docs = tmp_path / "docs" / "experiment-vm"
    examples = docs / "examples"
    examples.mkdir(parents=True)
    monkeypatch.setattr(gate, "DOCS", docs)
    monkeypatch.setattr(gate, "EXAMPLES", examples)
    monkeypatch.setattr(gate, "REGISTRY_PIN", docs / "step-zero-arming.v1.json")
    monkeypatch.setattr(
        gate, "EXCLUSIONS", docs / "unresolved-contract-exclusions.v1.json")

    def run(documents: dict[str, dict], pinned: list[str],
            exclusions: list[dict] | None = None):
        for name, body in documents.items():
            (examples / name).write_text(json.dumps(body), encoding="utf-8")
        (docs / "step-zero-arming.v1.json").write_text(
            json.dumps({"profiles": [{"contract": c} for c in pinned]}),
            encoding="utf-8")
        (docs / "unresolved-contract-exclusions.v1.json").write_text(
            json.dumps({"exclusions": exclusions or []}), encoding="utf-8")
        return gate.check_contract_resolution()

    return run


def test_the_repository_resolves_and_counts_what_it_resolved(monkeypatch):
    """The baseline. Green, and green over a non-trivial number of sites.

    A gate that resolved nothing would also report no problems, so the count is
    part of the assertion rather than decoration.
    """
    monkeypatch.chdir(REPOSITORY)
    problems, counts = gate.check_contract_resolution()
    assert problems == []
    assert counts["resolved"] >= 5
    assert counts["sites"] >= counts["resolved"] + counts["builtin"]
    assert counts["registry"] >= 20


def test_the_two_mageflow_documents_are_the_recorded_exemptions(monkeypatch):
    """The exemption is exactly the three ids on exactly the two documents.

    Without this, the entries could be widened to a whole family or a whole
    directory and the gate would stay green while covering less.
    """
    monkeypatch.chdir(REPOSITORY)
    catalog = json.loads(
        (REPOSITORY / gate.EXCLUSIONS).read_text(encoding="utf-8"))
    mageflow = {
        entry["contract"]: set(entry["documents"])
        for entry in catalog["exclusions"]
        if entry["contract"].startswith("rwkv_lab.")
    }
    assert set(mageflow) == {
        "rwkv_lab.mageflow.v1.Train",
        "rwkv_lab.mageflow.v1.PrepareCacheSpan",
        "rwkv_lab.mageflow.v1.CacheEncoders",
    }
    for documents in mageflow.values():
        assert documents == {
            "examples/mageflow-cache-resume.json",
            "examples/rwkv-scratch-topologies.json",
        }


def test_an_unrecorded_contract_fails_naming_document_and_id(tree):
    components, nodes = resolving()
    problems, counts = tree(
        {"bad.json": document(
            "bad",
            {**components, "w": worker({"train": "rwkv_lab.rwkv_scratch.v2.Train"})},
            {**nodes, "t": {"invoke": {"component": "w", "operation": "train"}}})},
        [PINNED])
    assert len(problems) == 1
    assert "bad.json" in problems[0]
    assert "rwkv_lab.rwkv_scratch.v2.Train" in problems[0]
    assert counts["resolved"] == 1


def test_a_recorded_contract_passes_on_the_document_that_records_it(tree):
    components, nodes = resolving()
    problems, counts = tree(
        {"fixture.json": document(
            "fixture",
            {**components, "w": worker({"train": "coverage.x.v1.Train"})},
            {**nodes, "t": {"invoke": {"component": "w", "operation": "train"}}})},
        [PINNED],
        [{"contract": "coverage.x.v1.Train", "reason": "synthetic",
          "documents": ["examples/fixture.json"]}])
    assert problems == []
    assert counts["excluded"] == 1


def test_a_copied_id_does_not_inherit_its_exemption(tree):
    """The masking test. A second document copying the id is still red."""
    base, base_nodes = resolving()
    invoke = {**base_nodes,
              "t": {"invoke": {"component": "w", "operation": "train"}}}
    components = {**base, "w": worker({"train": "coverage.x.v1.Train"})}
    problems, _ = tree(
        {
            "fixture.json": document("fixture", components, invoke),
            "copy.json": document("copy", components, invoke),
        },
        [PINNED],
        [{"contract": "coverage.x.v1.Train", "reason": "synthetic",
          "documents": ["examples/fixture.json"]}])
    assert len(problems) == 1
    assert "copy.json" in problems[0]
    assert "does not inherit" in problems[0]


def test_an_exemption_that_now_resolves_fails(tree):
    problems, _ = tree(
        {"ok.json": document(
            "ok", {"w": worker({"train": "rwkv_lab.rwkv_scratch.v1.Train"})},
            {"t": {"invoke": {"component": "w", "operation": "train"}}})},
        ["rwkv_lab.rwkv_scratch.v1.Train"],
        [{"contract": "rwkv_lab.rwkv_scratch.v1.Train", "reason": "stale",
          "documents": ["examples/ok.json"]}])
    assert len(problems) == 1
    assert "delete the entry" in problems[0]
    assert "now declares it" in problems[0]


def test_an_exemption_no_document_invokes_fails(tree):
    problems, _ = tree(
        {"ok.json": document(
            "ok", {"w": worker({"train": "rwkv_lab.rwkv_scratch.v1.Train"})},
            {"t": {"invoke": {"component": "w", "operation": "train"}}})},
        ["rwkv_lab.rwkv_scratch.v1.Train"],
        [{"contract": "coverage.gone.v1.Train", "reason": "dead",
          "documents": ["examples/gone.json"]}])
    assert len(problems) == 1
    assert "no example document invokes it" in problems[0]


def test_an_exemption_without_a_reason_fails(tree):
    components, nodes = resolving()
    problems, _ = tree(
        {"fixture.json": document(
            "fixture",
            {**components, "w": worker({"train": "coverage.x.v1.Train"})},
            {**nodes, "t": {"invoke": {"component": "w", "operation": "train"}}})},
        [PINNED],
        [{"contract": "coverage.x.v1.Train", "reason": "  ",
          "documents": ["examples/fixture.json"]}])
    assert any("no reason" in problem for problem in problems)


def test_missing_component_operation_and_contract_read_differently(tree):
    """Three defects, three sentences.

    Folding them into one 'unresolvable contract' would send the reader to the
    registry when the fault is a typo in the document.
    """
    base, base_nodes = resolving()
    components = {**base, "w": worker({"train": "coverage.x.v1.Train"})}
    problems, _ = tree(
        {"bad.json": document("bad", components, {
            **base_nodes,
            "a": {"invoke": {"component": "absent", "operation": "train"}},
            "b": {"invoke": {"component": "w", "operation": "absent"}},
        })},
        [PINNED])
    joined = " | ".join(problems)
    assert "does not declare under spec.components" in joined
    assert "declares no such operation" in joined

    base, base_nodes = resolving()
    blank = {**base, "w": {"adapter": "a", "runtime": "python_worker",
                           "operations": {"train": {}}}}
    problems, _ = tree(
        {"blank.json": document("blank", blank, {
            **base_nodes,
            "a": {"invoke": {"component": "w", "operation": "train"}}})},
        [PINNED])
    assert any("declares no contract string" in problem for problem in problems)


def test_builtin_invocations_are_counted_not_resolved(tree):
    problems, counts = tree(
        {"core.json": document(
            "core",
            {"core": worker({"acquire": "trainvm.v1.AcquireResources"},
                            runtime="builtin"),
             "w": worker({"train": PINNED})},
            {"a": {"invoke": {"component": "core", "operation": "acquire"}},
             "t": {"invoke": {"component": "w", "operation": "train"}}})},
        ["rwkv_lab.rwkv_scratch.v1.Train"])
    assert problems == []
    assert counts["builtin"] == 1
    assert counts["resolved"] == 1


def test_both_invocation_sites_are_walked(tree):
    """spec.execution names a component too, and it is checked."""
    problems, counts = tree(
        {"exec.json": document(
            "exec", {"w": worker({"train": "rwkv_lab.rwkv_scratch.v1.Train"})},
            {"t": {"invoke": {"component": "w", "operation": "train"}}},
            execution={"component": "w", "operation": "absent"})},
        ["rwkv_lab.rwkv_scratch.v1.Train"])
    assert counts["sites"] == 2
    assert len(problems) == 1
    assert "execution" in problems[0]


def test_an_empty_example_tree_fails_rather_than_passing_vacuously(tree):
    problems, counts = tree({}, ["rwkv_lab.rwkv_scratch.v1.Train"])
    assert counts["sites"] == 0
    assert any("walked nothing" in problem for problem in problems)


def test_an_empty_registry_pin_fails(tree):
    problems, _ = tree(
        {"ok.json": document(
            "ok", {"w": worker({"train": "coverage.x.v1.Train"})},
            {"t": {"invoke": {"component": "w", "operation": "train"}}})},
        [],
        [{"contract": "coverage.x.v1.Train", "reason": "synthetic",
          "documents": ["examples/ok.json"]}])
    assert any("declares no contracts" in problem for problem in problems)
    assert any("proved nothing" in problem for problem in problems)


def test_a_missing_exclusion_catalog_fails(tree, monkeypatch):
    problems, _ = tree(
        {"ok.json": document(
            "ok", {"w": worker({"train": "rwkv_lab.rwkv_scratch.v1.Train"})},
            {"t": {"invoke": {"component": "w", "operation": "train"}}})},
        ["rwkv_lab.rwkv_scratch.v1.Train"])
    assert problems == []
    gate.EXCLUSIONS.unlink()
    problems, _ = gate.check_contract_resolution()
    assert any("every example contract would be excused" in problem
               for problem in problems)


def test_a_recipe_catalogs_nested_template_documents_are_walked(tree):
    """A catalog holds several specs; each is resolved on its own.

    Keyed by file, one recipe losing its contract would hide behind a sibling
    that still has one — the exact shape that made an earlier gate in this
    repository pass every mutation.
    """
    catalog = {
        "api_version": "trainvm.recipe-profiles/v1",
        "recipes": [
            {"key": {"name": "good", "version": "1"},
             "template_document": document(
                 "good", {"w": worker({"train": "rwkv_lab.rwkv_scratch.v1.Train"})},
                 {"t": {"invoke": {"component": "w", "operation": "train"}}})},
            {"key": {"name": "bad", "version": "1"},
             "template_document": document(
                 "bad", {"w": worker({"train": "rwkv_lab.rwkv_scratch.v9.Train"})},
                 {"t": {"invoke": {"component": "w", "operation": "train"}}})},
        ],
    }
    problems, counts = tree({"catalog.json": catalog},
                            ["rwkv_lab.rwkv_scratch.v1.Train"])
    assert counts["specs"] == 2
    assert len(problems) == 1
    assert "[bad]" in problems[0]
