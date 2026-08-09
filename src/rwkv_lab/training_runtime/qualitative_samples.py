from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .resolved import resolved_component_parts

_MAXIMUM_HELD_OUT_MANIFEST_BYTES = 256 * 1024 * 1024


class QualitativeSampleImplementation(str, Enum):
    FIXED_HELD_OUT_V1 = "rwkv_lab.qualitative_sample.fixed_held_out.v1"
    FIXED_HELD_OUT_V2 = "rwkv_lab.qualitative_sample.fixed_held_out.v2"
    FIXED_MANIFEST_V1 = "rwkv_lab.qualitative_sample.fixed_manifest.v1"


def _digest(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class FixedHeldOutConfiguration:
    identity_field: str
    identities_digest: str
    selector_digest: str
    sample_count: int

    def __post_init__(self) -> None:
        if (
            not self.identity_field
            or not self.identity_field[0].isascii()
            or not self.identity_field[0].isalpha()
            or any(
                not character.isascii()
                or not (character.isalnum() or character in {"_", "-", ".", ":"})
                for character in self.identity_field
            )
        ):
            raise ValueError("held-out identity field must be symbolic")
        for value, label in (
            (self.identities_digest, "identities_digest"),
            (self.selector_digest, "selector_digest"),
        ):
            if (
                len(value) != 71
                or not value.startswith("sha256:")
                or any(character not in "0123456789abcdef" for character in value[7:])
            ):
                raise ValueError(f"{label} must be a lowercase sha256 digest")
        if (
            not isinstance(self.sample_count, int)
            or isinstance(self.sample_count, bool)
            or not 1 <= self.sample_count <= 1_000_000
        ):
            raise ValueError("sample_count must be a positive bounded integer")

    @classmethod
    def from_resolved(
        cls, configuration: Mapping[str, Any]
    ) -> FixedHeldOutConfiguration:
        if set(configuration) != {
            "identity_field",
            "identities_digest",
            "selector_digest",
            "sample_count",
        }:
            raise ValueError("resolved held-out sample configuration is inexact")
        return cls(**configuration)


@dataclass(frozen=True, slots=True)
class FixedHeldOutSamples:
    configuration: FixedHeldOutConfiguration

    def bind(
        self, identities: Sequence[str], *, selector_digest: str
    ) -> tuple[str, ...]:
        if selector_digest != self.configuration.selector_digest:
            raise ValueError(
                "held-out sample selector differs from the locked split membership"
            )
        bound = tuple(identities)
        if len(bound) != self.configuration.sample_count:
            raise ValueError(
                "held-out sample count differs from its authority declaration"
            )
        if any(not isinstance(identity, str) or not identity for identity in bound):
            raise ValueError("held-out sample identities must be nonempty strings")
        if len(set(bound)) != len(bound):
            raise ValueError("held-out sample identities must be unique")
        if _digest(bound) != self.configuration.identities_digest:
            raise ValueError(
                "held-out sample identities differ from the locked identity digest"
            )
        return bound

    def select(
        self,
        population: Sequence[str],
        *,
        selector_digest: str,
        dataset_root: Path | None = None,
    ) -> HeldOutSampleBinding:
        del dataset_root
        bound = self.bind(
            tuple(population[: self.configuration.sample_count]),
            selector_digest=selector_digest,
        )
        return HeldOutSampleBinding(
            bound,
            self.configuration.identities_digest,
            self.configuration.selector_digest,
        )


@dataclass(frozen=True, slots=True)
class DerivedFixedHeldOutConfiguration:
    """Operator policy whose integrity receipts are measured at binding time."""

    identity_field: str
    sample_count: int

    def __post_init__(self) -> None:
        FixedHeldOutConfiguration(
            identity_field=self.identity_field,
            identities_digest="sha256:" + "0" * 64,
            selector_digest="sha256:" + "0" * 64,
            sample_count=self.sample_count,
        )

    @classmethod
    def from_resolved(
        cls, configuration: Mapping[str, Any]
    ) -> DerivedFixedHeldOutConfiguration:
        if set(configuration) != {"identity_field", "sample_count"}:
            raise ValueError("resolved derived held-out configuration is inexact")
        return cls(**configuration)


@dataclass(frozen=True, slots=True)
class HeldOutSampleBinding:
    identities: tuple[str, ...]
    identities_digest: str
    selector_digest: str


@dataclass(frozen=True, slots=True)
class DerivedFixedHeldOutSamples:
    configuration: DerivedFixedHeldOutConfiguration

    def bind(
        self, identities: Sequence[str], *, selector_digest: str
    ) -> HeldOutSampleBinding:
        bound = tuple(identities)
        if len(bound) != self.configuration.sample_count:
            raise ValueError("held-out sample count differs from declared policy")
        if any(not isinstance(identity, str) or not identity for identity in bound):
            raise ValueError("held-out sample identities must be nonempty strings")
        if len(set(bound)) != len(bound):
            raise ValueError("held-out sample identities must be unique")
        if (
            len(selector_digest) != 71
            or not selector_digest.startswith("sha256:")
            or any(
                character not in "0123456789abcdef" for character in selector_digest[7:]
            )
        ):
            raise ValueError("selector_digest must be a lowercase sha256 digest")
        return HeldOutSampleBinding(bound, _digest(bound), selector_digest)

    def select(
        self,
        population: Sequence[str],
        *,
        selector_digest: str,
        dataset_root: Path | None = None,
    ) -> HeldOutSampleBinding:
        del dataset_root
        return self.bind(
            tuple(population[: self.configuration.sample_count]),
            selector_digest=selector_digest,
        )


@dataclass(frozen=True, slots=True)
class FixedManifestConfiguration:
    """A frozen ordered held-out subset inside an authority-bound dataset root."""

    manifest_name: str
    manifest_sha256: str
    identity_field: str
    sample_count: int

    def __post_init__(self) -> None:
        if (
            not self.manifest_name.endswith(".jsonl")
            or Path(self.manifest_name).name != self.manifest_name
            or self.manifest_name in {".", ".."}
        ):
            raise ValueError("held-out manifest_name must be one JSONL basename")
        if (
            len(self.manifest_sha256) != 71
            or not self.manifest_sha256.startswith("sha256:")
            or any(
                character not in "0123456789abcdef"
                for character in self.manifest_sha256[7:]
            )
        ):
            raise ValueError("held-out manifest_sha256 must be a lowercase digest")
        DerivedFixedHeldOutConfiguration(
            identity_field=self.identity_field,
            sample_count=self.sample_count,
        )

    @classmethod
    def from_resolved(
        cls, configuration: Mapping[str, Any]
    ) -> FixedManifestConfiguration:
        if set(configuration) != {
            "identity_field",
            "manifest_name",
            "manifest_sha256",
            "sample_count",
        }:
            raise ValueError("resolved fixed-manifest configuration is inexact")
        return cls(**configuration)


@dataclass(frozen=True, slots=True)
class FixedManifestSamples:
    configuration: FixedManifestConfiguration

    def select(
        self,
        population: Sequence[str],
        *,
        selector_digest: str,
        dataset_root: Path | None = None,
    ) -> HeldOutSampleBinding:
        if dataset_root is None:
            raise ValueError("fixed held-out manifest requires a dataset root")
        if (
            len(selector_digest) != 71
            or not selector_digest.startswith("sha256:")
            or any(
                character not in "0123456789abcdef" for character in selector_digest[7:]
            )
        ):
            raise ValueError("selector_digest must be a lowercase sha256 digest")
        root_descriptor = -1
        descriptor = -1
        try:
            root_descriptor = os.open(
                dataset_root,
                os.O_RDONLY
                | os.O_DIRECTORY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            if not stat.S_ISDIR(os.fstat(root_descriptor).st_mode):
                raise OSError("dataset root is not a directory")
            descriptor = os.open(
                self.configuration.manifest_name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=root_descriptor,
            )
            info = os.fstat(descriptor)
        except OSError as error:
            if descriptor >= 0:
                os.close(descriptor)
                descriptor = -1
            raise ValueError("fixed held-out manifest is unavailable") from error
        finally:
            if root_descriptor >= 0:
                os.close(root_descriptor)
        try:
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("fixed held-out manifest must be a regular file")
            if not 0 < info.st_size <= _MAXIMUM_HELD_OUT_MANIFEST_BYTES:
                raise ValueError("fixed held-out manifest has an invalid size")
            with os.fdopen(descriptor, "rb") as source:
                descriptor = -1
                encoded = source.read(_MAXIMUM_HELD_OUT_MANIFEST_BYTES + 1)
            if len(encoded) != info.st_size:
                raise ValueError("fixed held-out manifest changed while reading")
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        actual = "sha256:" + hashlib.sha256(encoded).hexdigest()
        if actual != self.configuration.manifest_sha256:
            raise ValueError("fixed held-out manifest content digest changed")
        population_set = frozenset(population)
        selected: list[str] = []
        selected_set: set[str] = set()
        for line_number, line in enumerate(encoded.splitlines(), start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"fixed held-out manifest row {line_number} is malformed"
                ) from error
            identity = (
                row.get(self.configuration.identity_field)
                if isinstance(row, Mapping)
                else None
            )
            if not isinstance(identity, str) or not identity:
                raise ValueError("fixed held-out manifest has an empty identity")
            if identity not in population_set:
                raise ValueError(
                    "fixed held-out identity is outside the selected split"
                )
            if identity in selected_set:
                raise ValueError("fixed held-out manifest has a duplicate identity")
            selected.append(identity)
            selected_set.add(identity)
            if len(selected) == self.configuration.sample_count:
                break
        if len(selected) != self.configuration.sample_count:
            raise ValueError("fixed held-out manifest has too few records")
        identities = tuple(selected)
        return HeldOutSampleBinding(identities, _digest(identities), selector_digest)


def qualitative_sample_from_resolved_component(
    component: Mapping[str, Any],
) -> FixedHeldOutSamples | DerivedFixedHeldOutSamples | FixedManifestSamples:
    implementation, configuration = resolved_component_parts(
        component, "qualitative_sample"
    )
    if implementation == QualitativeSampleImplementation.FIXED_HELD_OUT_V1:
        return FixedHeldOutSamples(
            FixedHeldOutConfiguration.from_resolved(configuration)
        )
    if implementation == QualitativeSampleImplementation.FIXED_HELD_OUT_V2:
        return DerivedFixedHeldOutSamples(
            DerivedFixedHeldOutConfiguration.from_resolved(configuration)
        )
    if implementation == QualitativeSampleImplementation.FIXED_MANIFEST_V1:
        return FixedManifestSamples(
            FixedManifestConfiguration.from_resolved(configuration)
        )
    else:
        raise ValueError(
            "resolved qualitative sample implementation is not allowlisted"
        )


__all__ = [
    "DerivedFixedHeldOutConfiguration",
    "DerivedFixedHeldOutSamples",
    "FixedHeldOutConfiguration",
    "FixedHeldOutSamples",
    "FixedManifestConfiguration",
    "FixedManifestSamples",
    "HeldOutSampleBinding",
    "QualitativeSampleImplementation",
    "qualitative_sample_from_resolved_component",
]
