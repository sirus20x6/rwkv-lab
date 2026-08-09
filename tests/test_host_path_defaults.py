"""Host-specific path defaults must not be baked into the live trainers.

Ten dataclass and signature defaults in the MLA trainers and loaders were bare
``/thearray/...`` paths that exist only on the maintainer's host. A config built
anywhere else inherited a path its caller never chose, so the code did not work
on a clean machine at all.

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
}

ENVIRONMENT_VARIABLES = (
    "MOE_MLA_MODEL_DIR",
    "MOE_MLA_PATCH_DIR",
    "MOE_MLA_TOKENS_BIN",
    "MOE_MLA_OUT_DIR",
    "MOE_MLA_ENGRAM_PATCH_DIR",
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
    return absent


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
