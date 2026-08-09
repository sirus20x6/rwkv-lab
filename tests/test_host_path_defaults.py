"""Host-specific path defaults must not be baked into the live trainers.

Dataclass fields, function signatures and argparse ``default=`` values across
the MLA trainers, loaders and Engram entry points were bare ``/thearray/...``
paths that exist only on the maintainer's host. A config built anywhere else
inherited a path its caller never chose, so the code did not work on a clean
machine at all.

The policy — environment variable, then the historical path *if this host has
it*, then nothing plus a message naming the field — is documented in the README
under "Host-specific path defaults" and implemented in ``rwkv_lab.host_paths``.

The clean machine is SIMULATED here: the environment variables are cleared and
each module's historical-path constant is pointed at an absent directory under
``tmp_path``. Depending on the host running the suite would make these tests
say different things on different machines, which is the defect, not the test.
"""

from __future__ import annotations

import dataclasses

import pytest

from rwkv_lab import host_paths


# Every module-level constant naming a historical path, so "clean machine"
# means all of them at once rather than whichever one a test remembered.
HISTORICAL_CONSTANTS = {
    "rwkv_lab.train_mla": (
        "PATCH_DIR_HISTORICAL_PATH",
        "TOKENS_BIN_HISTORICAL_PATH",
        "OUT_DIR_HISTORICAL_PATH",
        "OUT_DIR_HISTORICAL_ROOT",
    ),
    "rwkv_lab.train_mla_engram": ("ENGRAM_PATCH_DIR_HISTORICAL_PATH",),
    "rwkv_lab.load_converted": (
        "MODEL_DIR_HISTORICAL_PATH",
        "PATCH_DIR_HISTORICAL_PATH",
    ),
    "rwkv_lab.load_mla_engram": ("ENGRAM_PATCH_DIR_HISTORICAL_PATH",),
    "rwkv_lab.build_memory_targets": ("TEACHER_MODEL_DIR_HISTORICAL_PATH",),
    "rwkv_lab.gpu_engram_prefill": (
        "ENGRAM_PATCH_DIR_HISTORICAL_PATH",
        "PREFILL_OUT_HISTORICAL_PATH",
        "PREFILL_OUT_HISTORICAL_ROOT",
    ),
    "rwkv_lab.verify_engram": (
        "ENGRAM_PATCH_DIR_HISTORICAL_PATH",
        "MLA_CKPT_HISTORICAL_PATH",
    ),
}

ENVIRONMENT_VARIABLES = (
    "MOE_MLA_MODEL_DIR",
    "MOE_MLA_PATCH_DIR",
    "MOE_MLA_TOKENS_BIN",
    "MOE_MLA_OUT_DIR",
    "MOE_MLA_ENGRAM_PATCH_DIR",
    "MOE_MLA_ENGRAM_PREFILL_OUT",
    "MOE_MLA_TEACHER_MODEL_DIR",
    "MOE_MLA_CKPT",
)


def _import(name: str):
    """Import a module, skipping when its optional runtime is unavailable."""
    return pytest.importorskip(name)


# The Engram modules used to need an ``engram_ext`` stub here just to be
# importable, because rwkv_lab.engram_integration imported that separately
# attested runtime at module scope. The import is now deferred to its point of
# use, so the two Engram tests below exercise the real modules with no stub at
# all. tests/test_engram_entry_points.py is what keeps it that way.


@pytest.fixture
def clean_machine(monkeypatch, tmp_path):
    """A host with none of the environment variables and none of the paths."""
    import sys

    torchvision_was_imported = "torchvision" in sys.modules

    for variable in ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    absent = tmp_path / "absent"
    for module_name, constants in HISTORICAL_CONSTANTS.items():
        try:
            module = __import__(module_name, fromlist=["_"])
        except ImportError:
            continue
        for constant in constants:
            monkeypatch.setattr(module, constant, str(absent / constant))
    yield absent

    # build_memory_targets marks torchvision unavailable at import time, to get
    # past a torchvision/torch ABI mismatch in the training venv. That is a
    # global edit to sys.modules, and leaving it behind would make any later
    # test in the session see a torchvision that cannot be imported.
    if not torchvision_was_imported and sys.modules.get("torchvision") is None:
        sys.modules.pop("torchvision", None)


def test_the_resolver_prefers_the_environment_then_an_existing_path(
    monkeypatch, tmp_path
):
    """The three-step order, asserted directly on the shared resolver."""
    monkeypatch.delenv("MOE_MLA_TEST_PATH", raising=False)
    absent = tmp_path / "nowhere"
    present = tmp_path / "somewhere"
    present.mkdir()

    assert host_paths.resolve_host_path("MOE_MLA_TEST_PATH", str(absent)) is None
    assert host_paths.resolve_host_path(
        "MOE_MLA_TEST_PATH", str(present)) == str(present)

    monkeypatch.setenv("MOE_MLA_TEST_PATH", "/elsewhere")
    assert host_paths.resolve_host_path(
        "MOE_MLA_TEST_PATH", str(present)) == "/elsewhere"


def test_the_resolver_can_judge_an_output_by_the_root_it_lives_under(tmp_path):
    """A run directory does not exist yet, so its own absence says nothing."""
    root = tmp_path / "runs"
    root.mkdir()
    run = root / "mla_ft_v1"
    assert not run.exists()
    assert host_paths.resolve_host_path(
        "MOE_MLA_UNSET_FOR_THIS_TEST", str(run), probe=str(root)) == str(run)


def test_an_unconfigured_path_names_the_field_and_the_variable():
    with pytest.raises(ValueError) as failure:
        host_paths.require_host_path(
            None, field="tokens_bin", env_var="MOE_MLA_TOKENS_BIN",
            flag="tokens-bin",
        )
    message = str(failure.value)
    assert "tokens_bin" in message
    assert "MOE_MLA_TOKENS_BIN" in message
    assert "--tokens-bin" in message


def test_train_mla_requires_model_dir(clean_machine):
    """The one default deliberately not resolved: it must be supplied.

    The historical default named the 35B model, so a 9B launch that forgot
    --model-dir trained against the wrong weights and said nothing about it.
    """
    train_mla = _import("rwkv_lab.train_mla")

    model_dir_field = train_mla.TrainConfig.__dataclass_fields__["model_dir"]
    assert model_dir_field.default is dataclasses.MISSING
    assert model_dir_field.default_factory is dataclasses.MISSING

    with pytest.raises(TypeError) as failure:
        train_mla.TrainConfig()  # type: ignore[call-arg]
    assert "model_dir" in str(failure.value)


def test_train_config_defaults_are_not_one_machines_paths(clean_machine):
    """A config built on a clean machine inherits no absolute path at all."""
    train_mla = _import("rwkv_lab.train_mla")

    config = train_mla.TrainConfig(model_dir="/supplied/by/the/caller")
    assert config.patch_dir is None
    assert config.tokens_bin is None
    assert config.out_dir is None
    for value in dataclasses.asdict(config).values():
        assert not (isinstance(value, str) and value.startswith("/thearray")), (
            "a TrainConfig default still carries the maintainer's absolute path"
        )


@pytest.mark.parametrize(
    "field, variable",
    [
        ("patch_dir", "MOE_MLA_PATCH_DIR"),
        ("tokens_bin", "MOE_MLA_TOKENS_BIN"),
        ("out_dir", "MOE_MLA_OUT_DIR"),
    ],
)
def test_validation_names_the_unconfigured_field(clean_machine, field, variable):
    """Failing must say which knob to set, not echo a path nobody typed."""
    train_mla = _import("rwkv_lab.train_mla")

    settings = {"patch_dir": "/p", "tokens_bin": "/t", "out_dir": "/o"}
    settings[field] = None
    config = train_mla.TrainConfig(model_dir="/m", **settings)

    with pytest.raises(ValueError) as failure:
        train_mla.validate_train_config(config)
    message = str(failure.value)
    assert field in message
    assert variable in message


def test_validation_names_model_dir_when_it_is_blank(clean_machine):
    train_mla = _import("rwkv_lab.train_mla")

    config = train_mla.TrainConfig(
        model_dir="", patch_dir="/p", tokens_bin="/t", out_dir="/o")
    with pytest.raises(ValueError) as failure:
        train_mla.validate_train_config(config)
    assert "model_dir" in str(failure.value)


def test_a_fully_supplied_config_still_validates(clean_machine):
    """The clean machine is only fatal when nothing was supplied."""
    train_mla = _import("rwkv_lab.train_mla")

    config = train_mla.TrainConfig(
        model_dir="/m", patch_dir="/p", tokens_bin="/t", out_dir="/o")
    train_mla.validate_train_config(config)  # must not raise


def test_the_environment_still_supplies_the_defaults(clean_machine, monkeypatch):
    """A host that keeps the artifacts elsewhere configures them once."""
    train_mla = _import("rwkv_lab.train_mla")

    monkeypatch.setenv("MOE_MLA_PATCH_DIR", "/elsewhere/converted")
    monkeypatch.setenv("MOE_MLA_TOKENS_BIN", "/elsewhere/tokens.bin")
    monkeypatch.setenv("MOE_MLA_OUT_DIR", "/elsewhere/runs/current")

    config = train_mla.TrainConfig(model_dir="/m")
    assert config.patch_dir == "/elsewhere/converted"
    assert config.tokens_bin == "/elsewhere/tokens.bin"
    assert config.out_dir == "/elsewhere/runs/current"
    train_mla.validate_train_config(config)


def test_the_historical_path_is_still_used_where_it_exists(
    clean_machine, monkeypatch, tmp_path
):
    """The maintainer's host keeps behaving exactly as it did before."""
    train_mla = _import("rwkv_lab.train_mla")

    cached = tmp_path / "converted"
    cached.mkdir()
    monkeypatch.setattr(train_mla, "PATCH_DIR_HISTORICAL_PATH", str(cached))
    assert train_mla.TrainConfig(model_dir="/m").patch_dir == str(cached)


def test_engram_train_config_default_is_not_one_machines_path(clean_machine):
    engram = _import("rwkv_lab.train_mla_engram")

    config = engram.EngramTrainConfig(model_dir="/m")
    assert config.engram_patch_dir is None
    field = engram.EngramTrainConfig.__dataclass_fields__["engram_patch_dir"]
    assert field.default is dataclasses.MISSING, (
        "engram_patch_dir went back to a constant default"
    )


def test_load_converted_model_names_the_argument_on_a_clean_machine(
    clean_machine,
):
    loader = _import("rwkv_lab.load_converted")

    assert loader.default_model_dir() is None
    assert loader.default_patch_dir() is None
    with pytest.raises(ValueError) as failure:
        loader.load_converted_model()
    assert "model_dir" in str(failure.value)
    assert "MOE_MLA_MODEL_DIR" in str(failure.value)

    with pytest.raises(ValueError) as failure:
        loader.load_converted_model(model_dir="/supplied")
    assert "patch_dir" in str(failure.value)


def test_load_mla_engram_names_the_argument_on_a_clean_machine(clean_machine):
    loader = _import("rwkv_lab.load_mla_engram")

    assert loader.default_engram_patch_dir() is None
    with pytest.raises(ValueError) as failure:
        loader.load_mla_engram()
    assert "model_dir" in str(failure.value)

    with pytest.raises(ValueError) as failure:
        loader.load_mla_engram(model_dir="/m", mla_patch_dir="/p")
    assert "engram_patch_dir" in str(failure.value)


def _run_main(module, argv, monkeypatch):
    """Call ``module.main()`` with ``argv``, returning the failure message.

    The path resolution these tests are about happens immediately after
    ``parse_args``, so a clean machine fails there rather than anywhere near a
    GPU, a checkpoint or a model load.
    """
    monkeypatch.setattr("sys.argv", [module.__name__, *argv])
    with pytest.raises(ValueError) as failure:
        module.main()
    return str(failure.value)


def test_build_memory_targets_names_its_own_teacher_variable(
    clean_machine, monkeypatch, tmp_path
):
    """The 9B teacher must not resolve through the 35B model's variable.

    Both flags are spelled --model-dir, but they name different models. If one
    variable served both, a 9B extraction would silently read the 35B weights
    and look healthy — the same trap that made train_mla.py's --model-dir
    required.
    """
    targets = _import("rwkv_lab.build_memory_targets")

    assert targets.default_teacher_model_dir() is None
    message = _run_main(
        targets,
        ["--layer", "0", "--data", str(tmp_path), "--out", str(tmp_path)],
        monkeypatch,
    )
    assert "model_dir" in message
    assert "MOE_MLA_TEACHER_MODEL_DIR" in message

    # The 35B variable must not satisfy the 9B teacher.
    monkeypatch.setenv("MOE_MLA_MODEL_DIR", "/elsewhere/Qwen3.6-35B-A3B")
    assert targets.default_teacher_model_dir() is None

    monkeypatch.setenv("MOE_MLA_TEACHER_MODEL_DIR", "/elsewhere/Qwen3.5-9B-Base")
    assert targets.default_teacher_model_dir() == "/elsewhere/Qwen3.5-9B-Base"


def test_gpu_engram_prefill_defaults_are_not_one_machines_paths(
    clean_machine, monkeypatch
):
    prefill = _import("rwkv_lab.gpu_engram_prefill")

    assert prefill.default_engram_patch_dir() is None
    assert prefill.default_prefill_out() is None

    message = _run_main(prefill, [], monkeypatch)
    assert "model_dir" in message
    assert "MOE_MLA_MODEL_DIR" in message

    message = _run_main(prefill, ["--model-dir", "/m"], monkeypatch)
    assert "engram_patch_dir" in message
    assert "MOE_MLA_ENGRAM_PATCH_DIR" in message

    message = _run_main(
        prefill, ["--model-dir", "/m", "--engram-patch-dir", "/e"], monkeypatch
    )
    assert message.startswith("out is not configured")
    assert "--out" in message
    assert "MOE_MLA_ENGRAM_PREFILL_OUT" in message


def test_gpu_engram_prefill_judges_its_output_by_the_root_it_lives_under(
    clean_machine, monkeypatch, tmp_path
):
    """The prefill directory does not exist yet; its own absence proves nothing."""
    prefill = _import("rwkv_lab.gpu_engram_prefill")

    root = tmp_path / "moe-mla"
    root.mkdir()
    monkeypatch.setattr(
        prefill, "PREFILL_OUT_HISTORICAL_PATH", str(root / "engram_prefilled"))
    monkeypatch.setattr(prefill, "PREFILL_OUT_HISTORICAL_ROOT", str(root))
    assert prefill.default_prefill_out() == str(root / "engram_prefilled")


def test_verify_engram_defaults_are_not_one_machines_paths(
    clean_machine, monkeypatch
):
    verify = _import("rwkv_lab.verify_engram")

    assert verify.default_mla_ckpt() is None
    assert verify.default_engram_patch_dir() is None

    message = _run_main(verify, [], monkeypatch)
    assert "mla_ckpt" in message
    assert "MOE_MLA_CKPT" in message

    message = _run_main(verify, ["--mla-ckpt", "/c"], monkeypatch)
    assert "engram_patch_dir" in message
    assert "MOE_MLA_ENGRAM_PATCH_DIR" in message


def test_the_engram_patch_variable_serves_every_module_that_loads_one(
    clean_machine, monkeypatch
):
    """One artifact, one variable: a host configures the patch directory once.

    verify_engram and gpu_engram_prefill hand their flag to the same loader,
    so they share load_mla_engram's variable and keep only their own historical
    vintage of the path.
    """
    loader = _import("rwkv_lab.load_mla_engram")
    verify = _import("rwkv_lab.verify_engram")
    prefill = _import("rwkv_lab.gpu_engram_prefill")

    monkeypatch.setenv("MOE_MLA_ENGRAM_PATCH_DIR", "/elsewhere/engram")
    assert loader.default_engram_patch_dir() == "/elsewhere/engram"
    assert verify.default_engram_patch_dir() == "/elsewhere/engram"
    assert prefill.default_engram_patch_dir() == "/elsewhere/engram"


def test_the_historical_paths_still_win_where_this_host_has_them(
    clean_machine, monkeypatch, tmp_path
):
    """The maintainer's host keeps behaving exactly as it did before."""
    verify = _import("rwkv_lab.verify_engram")
    targets = _import("rwkv_lab.build_memory_targets")

    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_bytes(b"")
    teacher = tmp_path / "Qwen3.5-9B-Base"
    teacher.mkdir()
    monkeypatch.setattr(verify, "MLA_CKPT_HISTORICAL_PATH", str(ckpt))
    monkeypatch.setattr(
        targets, "TEACHER_MODEL_DIR_HISTORICAL_PATH", str(teacher))

    assert verify.default_mla_ckpt() == str(ckpt)
    assert targets.default_teacher_model_dir() == str(teacher)


def test_no_entry_point_argument_still_defaults_to_a_host_path():
    """The sweep for argparse defaults, across the modules that carry them.

    Scoped to ``default=`` rather than to every line, because these modules
    also carry a documented historical constant, a usage example naming the
    maintainer's corpora, and (in gpu_engram_prefill) the sys.path entry that
    makes engram_ext importable at all. Those are prose and runtime discovery,
    not configuration a caller inherits.
    """
    import pathlib
    import re

    baked = re.compile(r"""default\s*=\s*["']/thearray""")
    root = pathlib.Path(__file__).resolve().parents[1]
    for relative in (
        "src/rwkv_lab/gpu_engram_prefill.py",
        "src/rwkv_lab/build_memory_targets.py",
        "src/rwkv_lab/verify_engram.py",
    ):
        source = (root / relative).read_text(encoding="utf-8")
        for line in source.splitlines():
            assert not baked.search(line), (
                f"{relative} defaults an argument to a host-specific path: "
                f"{line.strip()}"
            )


def test_no_live_entry_point_still_carries_a_baked_in_host_path():
    """The sweep itself, so a reintroduced default fails here rather than on CI
    of whoever next clones the repository."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    for relative in (
        "src/rwkv_lab/train_mla.py",
        "src/rwkv_lab/train_mla_engram.py",
        "src/rwkv_lab/load_converted.py",
        "src/rwkv_lab/load_mla_engram.py",
    ):
        source = (root / relative).read_text(encoding="utf-8")
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or "HISTORICAL_PATH" in stripped:
                continue  # the documented constants, and prose about them
            if "HISTORICAL_ROOT" in stripped:
                continue
            assert "/thearray" not in stripped, (
                f"{relative} carries a host-specific path: {stripped}"
            )
