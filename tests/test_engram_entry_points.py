"""The Engram entry points must be importable, and testable, without engram-ext.

``rwkv_lab.engram_integration`` used to import ``engram_ext`` at module scope.
That distribution is a separately attested runtime nothing in this repository or
in CI installs, so ``import rwkv_lab.load_mla_engram`` and
``import rwkv_lab.train_mla_engram`` raised on every machine CI runs on.

The consequence was not a failing test — it was the absence of one. Both modules
had zero coverage and nothing reported that they had none: either could be
changed, merged fully green, and be completely broken, with the first evidence a
failed training launch on the one host with the extension built. PR #103 had to
stub ``engram_ext`` to reach two of the ten defaults it was fixing; without that
stub those two would have been silently skipped while reported as covered.

This file is what stops that recurring, so it deliberately does two things that
a single "does it import" assertion would not:

* it BLOCKS ``engram_ext`` rather than assuming it is absent, so the assertions
  say the same thing on the training host as on a runner; and
* it exercises real behaviour in each module — the install/uninstall patching in
  ``engram_integration``, the pre-load runtime check and patch application in
  ``load_mla_engram``, the checkpoint writer and argument surface in
  ``train_mla_engram`` — so a break in any of them fails here rather than
  passing by virtue of the import having succeeded.

Honest scope: the with-runtime half runs against a STUB ``engram_ext`` module,
not the compiled extension. No CI machine can build the real one, so what is
verified is that the deferred import is resolved at the point of use and that
this package's own wiring around ``EngramModule`` still works. Parity with the
real kernel is not, and cannot be, asserted here.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.abc
import json
import sys
import types

import pytest
import torch
import torch.nn as nn


ENGRAM_MODULES = (
    "rwkv_lab.engram_integration",
    "rwkv_lab.load_mla_engram",
    "rwkv_lab.train_mla_engram",
    "rwkv_lab.gpu_engram_prefill",
)


# ---------------------------------------------------------------------------
# Import control. Both fixtures purge the cached rwkv_lab Engram modules so the
# import under test is a real one, and purge again afterwards so no later test
# inherits a module bound to a blocked or stubbed runtime.
# ---------------------------------------------------------------------------

class _BlockEngramExt(importlib.abc.MetaPathFinder):
    """Make ``import engram_ext`` fail the way a clean machine makes it fail."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "engram_ext" or fullname.startswith("engram_ext."):
            raise ModuleNotFoundError(
                f"No module named {fullname!r}", name=fullname)
        return None


def _purge_engram_modules() -> None:
    for name in [n for n in sys.modules if n in ENGRAM_MODULES]:
        del sys.modules[name]
    for name in [n for n in sys.modules if n.startswith("engram_ext")]:
        del sys.modules[name]


@pytest.fixture
def without_engram_ext(monkeypatch):
    """A machine on which the attested runtime is not installed."""
    _purge_engram_modules()
    finder = _BlockEngramExt()
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
    yield
    _purge_engram_modules()


class _StubEngramModule(nn.Module):
    """Stands in for engram_ext's EngramModule: same construction and call."""

    def __init__(self, layer_id: int, cfg, hidden_size: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.cfg = cfg
        self.hidden_size = hidden_size
        # A real parameter so device/dtype moves and param collection are real.
        self.value_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.constant_(self.value_proj.weight, 0.0)
        self.calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def forward(self, hidden_states: torch.Tensor,
                input_ids: torch.Tensor) -> torch.Tensor:
        self.calls.append((tuple(hidden_states.shape), tuple(input_ids.shape)))
        # A constant, non-zero contribution keyed to the layer, so a wiring
        # mistake shows up as a wrong value rather than a silent zero.
        return torch.full_like(hidden_states, float(self.layer_id + 1))


@dataclasses.dataclass
class _StubEngramConfig:
    num_hashes: int = 4
    embedding_dim: int = 8


@pytest.fixture
def stub_engram_ext(monkeypatch):
    """A machine on which SOMETHING importable answers to ``engram_ext``."""
    _purge_engram_modules()
    package = types.ModuleType("engram_ext")
    module = types.ModuleType("engram_ext.engram_module")
    module.EngramConfig = _StubEngramConfig
    module.EngramModule = _StubEngramModule
    package.engram_module = module
    monkeypatch.setitem(sys.modules, "engram_ext", package)
    monkeypatch.setitem(sys.modules, "engram_ext.engram_module", module)
    yield module
    _purge_engram_modules()


# ---------------------------------------------------------------------------
# 1. The absence itself.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module_name", ENGRAM_MODULES)
def test_the_engram_modules_import_without_the_runtime(
    without_engram_ext, module_name
):
    """The regression this file exists for.

    Restoring a module-scope ``from engram_ext...`` import in any of these
    modules fails this immediately, on every machine, instead of removing them
    from CI while the suite stays green.

    ``gpu_engram_prefill`` joined the list when its host paths were fixed: it
    reached the runtime by prepending a machine-local checkout to ``sys.path``
    and importing at module scope, which is the shape the README section "The
    ``engram-ext`` runtime dependency" tells callers not to use.
    """
    module = importlib.import_module(module_name)
    assert module.__name__ == module_name
    assert "engram_ext" not in sys.modules, (
        f"{module_name} imported engram_ext at module scope")


def test_the_missing_runtime_names_the_distribution_and_how_to_install_it(
    without_engram_ext,
):
    """The chosen run-time behaviour: a named error, not a stray ImportError."""
    integration = importlib.import_module("rwkv_lab.engram_integration")

    with pytest.raises(integration.EngramExtensionUnavailable) as failure:
        integration.require_engram_ext()

    message = str(failure.value)
    assert "engram_ext" in message
    assert "pip install" in message, "the error must say how to get the runtime"
    assert failure.value.name == "engram_ext"
    # Callers already writing `except ImportError` keep working.
    assert isinstance(failure.value, ImportError)
    # The real cause is preserved rather than replaced.
    assert isinstance(failure.value.__cause__, ImportError)


def test_installing_engram_without_the_runtime_reports_the_runtime(
    without_engram_ext,
):
    """The failure names engram_ext even though the caller called install_engram."""
    integration = importlib.import_module("rwkv_lab.engram_integration")

    model = _fake_causal_lm(hidden_size=8, layers=2)
    with pytest.raises(integration.EngramExtensionUnavailable) as failure:
        integration.install_engram(
            model, layer_indices=[0], engram_cfg=object(), hidden_size=8)
    assert "engram_ext" in str(failure.value)


def test_load_mla_engram_reports_the_runtime_before_loading_the_backbone(
    without_engram_ext, tmp_path, monkeypatch
):
    """Loading a 35B backbone first and only then discovering the runtime is
    missing costs the caller the whole load for nothing."""
    integration = importlib.import_module("rwkv_lab.engram_integration")
    loader = importlib.import_module("rwkv_lab.load_mla_engram")

    def _must_not_run(*args, **kwargs):  # pragma: no cover - asserts by raising
        raise AssertionError("the backbone was loaded before the runtime check")

    monkeypatch.setattr(loader, "load_converted_model", _must_not_run)
    for name in ("model", "mla", "engram"):
        (tmp_path / name).mkdir()

    with pytest.raises(integration.EngramExtensionUnavailable) as failure:
        loader.load_mla_engram(
            model_dir=str(tmp_path / "model"),
            mla_patch_dir=str(tmp_path / "mla"),
            engram_patch_dir=str(tmp_path / "engram"),
        )
    assert "engram_ext" in str(failure.value)


# ---------------------------------------------------------------------------
# 2. Real behaviour, so "it imports" is not the whole of the coverage.
# ---------------------------------------------------------------------------

class _FakeDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.eye_(self.projection.weight)
        self.received: list[torch.Tensor] = []

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        self.received.append(hidden_states)
        return self.projection(hidden_states)


class _FakeBackbone(nn.Module):
    def __init__(self, hidden_size: int, layers: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(16, hidden_size)
        self.layers = nn.ModuleList(
            [_FakeDecoderLayer(hidden_size) for _ in range(layers)])

    def forward(self, input_ids=None):
        hidden = self.embed(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class _FakeCausalLM(nn.Module):
    def __init__(self, hidden_size: int, layers: int) -> None:
        super().__init__()
        self.model = _FakeBackbone(hidden_size, layers)

    def forward(self, input_ids=None):
        return self.model(input_ids=input_ids)


def _fake_causal_lm(hidden_size: int, layers: int) -> nn.Module:
    """A ForCausalLM-shaped stand-in: ``model.model.layers``, ids-taking forward."""
    return _FakeCausalLM(hidden_size, layers)


def test_install_engram_threads_ids_and_adds_its_contribution(stub_engram_ext):
    """The live wiring: ids stashed on the backbone, delta added at layer top."""
    integration = importlib.import_module("rwkv_lab.engram_integration")

    hidden_size = 4
    model = _fake_causal_lm(hidden_size=hidden_size, layers=3)
    modules = integration.install_engram(
        model, layer_indices=[0, 2],
        engram_cfg=_StubEngramConfig(), hidden_size=hidden_size)

    assert len(modules) == 2
    assert [m.layer_id for m in modules] == [0, 2]
    assert model.model.layers[0].engram_module is modules[0]
    assert model.model.layers[2].engram_module is modules[1]
    assert not hasattr(model.model.layers[1], "engram_module")

    ids = torch.tensor([[1, 2, 3]])
    embedded = model.model.embed(ids)
    model(input_ids=ids)

    # Each patched layer saw its Engram contribution added before its own
    # forward; the identity projections make the expected value exact.
    assert modules[0].calls == [((1, 3, hidden_size), (1, 3))]
    torch.testing.assert_close(
        model.model.layers[0].received[0], embedded + 1.0)
    torch.testing.assert_close(
        model.model.layers[2].received[0],
        model.model.layers[1].received[0] + 3.0)


def test_uninstall_engram_restores_the_unpatched_model(stub_engram_ext):
    integration = importlib.import_module("rwkv_lab.engram_integration")

    model = _fake_causal_lm(hidden_size=4, layers=2)
    ids = torch.tensor([[1, 2]])
    before = model(input_ids=ids).clone()

    integration.install_engram(
        model, layer_indices=[1],
        engram_cfg=_StubEngramConfig(), hidden_size=4)
    assert not torch.allclose(model(input_ids=ids), before)

    integration.uninstall_engram(model)
    torch.testing.assert_close(model(input_ids=ids), before)
    assert not hasattr(model, "_engram_installed")
    assert not hasattr(model.model.layers[1], "engram_module")


def test_installing_engram_twice_is_refused(stub_engram_ext):
    integration = importlib.import_module("rwkv_lab.engram_integration")

    model = _fake_causal_lm(hidden_size=4, layers=2)
    integration.install_engram(
        model, layer_indices=[0],
        engram_cfg=_StubEngramConfig(), hidden_size=4)
    with pytest.raises(RuntimeError, match="already installed"):
        integration.install_engram(
            model, layer_indices=[1],
            engram_cfg=_StubEngramConfig(), hidden_size=4)


def test_load_mla_engram_applies_a_patch_and_rejects_unknown_keys(
    without_engram_ext,
):
    """``_apply_engram_patch`` is what puts trained Engram weights into a run."""
    loader = importlib.import_module("rwkv_lab.load_mla_engram")

    module = _StubEngramModule(layer_id=5, cfg=None, hidden_size=3)
    patch = {"layer_5.value_proj.weight": torch.ones(3, 3)}
    loader._apply_engram_patch([module], patch, [5])
    torch.testing.assert_close(module.value_proj.weight, torch.ones(3, 3))

    with pytest.raises(RuntimeError, match="unexpected keys"):
        loader._apply_engram_patch(
            [module], {"layer_5.not_a_parameter": torch.ones(1)}, [5])


def test_train_mla_engram_writes_a_checkpoint_atomically(tmp_path):
    """The trainer's own state writer, exercised end to end on CPU."""
    trainer = importlib.import_module("rwkv_lab.train_mla_engram")

    patch_dir = tmp_path / "engram_converted"
    patch_dir.mkdir()
    (patch_dir / "manifest.json").write_text(json.dumps({"layer_indices": [7]}))

    mla = nn.Linear(2, 2, bias=False)
    mla._save_key = "layer_3"
    engram = nn.Linear(2, 2, bias=False)
    optimizer = torch.optim.SGD(list(mla.parameters()), lr=0.1)

    config = trainer.EngramTrainConfig(
        model_dir="/supplied/by/the/caller",
        out_dir=str(tmp_path / "run"),
        engram_patch_dir=str(patch_dir),
    )
    (tmp_path / "run").mkdir()
    trainer.save_checkpoint(42, [mla], [engram], optimizer, None, config)

    written = tmp_path / "run" / "step_000042"
    assert written.is_dir()
    assert not list((tmp_path / "run").glob(".step_*.tmp")), (
        "the temporary directory survived the rename")
    payload = torch.load(written / "ckpt.pt", weights_only=False)
    assert payload["step"] == 42
    assert set(payload["mla_state_dicts"]) == {"layer_3"}
    assert set(payload["engram_state_dicts"]) == {"layer_7"}
    assert "optimizer_state_host" not in payload


def test_train_mla_engram_refuses_an_mla_module_with_no_save_key(tmp_path):
    """Saving a module whose key was lost would write an unloadable checkpoint."""
    trainer = importlib.import_module("rwkv_lab.train_mla_engram")

    config = trainer.EngramTrainConfig(
        model_dir="/m", out_dir=str(tmp_path / "run"), engram_patch_dir="/p")
    with pytest.raises(RuntimeError, match="_save_key"):
        trainer.save_checkpoint(
            1, [nn.Linear(2, 2)], [], torch.optim.SGD(
                list(nn.Linear(2, 2).parameters()), lr=0.1), None, config)


def test_train_mla_engram_demands_the_model_directory(monkeypatch, capsys):
    """main() builds its parser from the dataclass; --model-dir has no default.

    Reached through main() rather than by inspecting the dataclass, so a parser
    that stopped deriving its flags from the fields fails here.
    """
    trainer = importlib.import_module("rwkv_lab.train_mla_engram")

    assert issubclass(trainer.EngramTrainConfig, trainer.TrainConfig)
    assert {"mla_ckpt", "engram_patch_dir", "engram_lr_mult"} <= set(
        trainer.EngramTrainConfig.__dataclass_fields__)

    monkeypatch.setattr(sys, "argv", ["train_mla_engram"])
    with pytest.raises(SystemExit) as failure:
        trainer.main()
    assert failure.value.code == 2
    assert "--model-dir" in capsys.readouterr().err


def test_collect_engram_params_returns_the_optimizer_groups(stub_engram_ext):
    """train_mla_engram builds two optimizers from exactly this split."""
    integration = importlib.import_module("rwkv_lab.engram_integration")

    module = _StubEngramModule(layer_id=0, cfg=None, hidden_size=4)
    table = nn.Module()
    table.embedding = nn.Embedding(8, 4)   # not host-offloaded
    module.embedding = table

    # Nothing is on pinned host memory, so every parameter is a GPU-group one.
    assert integration.collect_host_embedding_params([module]) == []
    gpu = integration.collect_gpu_engram_params([module])
    assert any(parameter is module.value_proj.weight for parameter in gpu)
    assert any(parameter is table.embedding.weight for parameter in gpu)
