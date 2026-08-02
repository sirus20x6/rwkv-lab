from __future__ import annotations

import itertools
import json
import math
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

LINEAR = "rwkv_lab.schedule.linear_warmup_cosine.v1"
POWERCOOL = "rwkv_lab.schedule.powercool.v1"
RELATIVE_TOLERANCE = 1.0e-12


def native_values(
    trainvm: Path,
    implementation: str,
    configuration: Mapping[str, Any],
    max_step: int,
) -> list[float]:
    output = json.loads(
        subprocess.check_output(
            [
                str(trainvm),
                "inspect-training-schedule",
                implementation,
                json.dumps(configuration, sort_keys=True, separators=(",", ":")),
                str(max_step),
            ],
            text=True,
        )
    )
    if output.get("implementation_id") != implementation:
        raise SystemExit("native schedule output changed its implementation identity")
    if output.get("configuration") != configuration:
        raise SystemExit("native schedule output changed its resolved configuration")
    if output.get("max_step") != max_step:
        raise SystemExit("native schedule output changed its requested step bound")
    values = output.get("multipliers")
    if not isinstance(values, list):
        raise SystemExit("native schedule output has no multiplier list")
    return values


def compare_values(
    implementation: str,
    configuration: Mapping[str, Any],
    native: list[float],
    oracle: list[float],
    *,
    exact_steps: frozenset[int] = frozenset(),
) -> None:
    if len(native) != len(oracle):
        raise SystemExit(
            f"{implementation} returned {len(native)} values, expected {len(oracle)}"
        )
    for step, (actual, expected) in enumerate(zip(native, oracle, strict=True)):
        if not isinstance(actual, (int, float)) or not math.isfinite(actual):
            raise SystemExit(
                f"{implementation} step {step} returned non-finite native output"
            )
        if actual == expected:
            continue
        detail = json.dumps(
            {
                "implementation": implementation,
                "configuration": configuration,
                "step": step,
                "native": actual,
                "python": expected,
            },
            sort_keys=True,
        )
        if step in exact_steps or expected == 0.0:
            raise SystemExit("exact native/Python schedule mismatch: " + detail)
        relative_error = abs(actual - expected) / abs(expected)
        if relative_error > RELATIVE_TOLERANCE:
            raise SystemExit(
                "native/Python schedule mismatch above relative tolerance: "
                + detail
                + f", relative_error={relative_error}"
            )


def rejected_by_native(
    trainvm: Path, implementation: str, configuration: Mapping[str, Any]
) -> bool:
    completed = subprocess.run(
        [
            str(trainvm),
            "inspect-training-schedule",
            implementation,
            json.dumps(configuration, sort_keys=True, separators=(",", ":")),
            "2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode != 0


def require_rejection_parity(
    trainvm: Path,
    implementation: str,
    invalid_configurations: list[dict[str, Any]],
    python_configuration: Callable[[Mapping[str, Any]], object],
) -> None:
    for configuration in invalid_configurations:
        try:
            python_configuration(configuration)
        except (TypeError, ValueError):
            python_rejected = True
        else:
            python_rejected = False
        native_rejected = rejected_by_native(trainvm, implementation, configuration)
        if not python_rejected or not native_rejected:
            raise SystemExit(
                "schedule rejection parity failed: "
                + json.dumps(
                    {
                        "implementation": implementation,
                        "configuration": configuration,
                        "native_rejected": native_rejected,
                        "python_rejected": python_rejected,
                    },
                    sort_keys=True,
                )
            )


def require_command_rejection(
    trainvm: Path, arguments: list[str], message: str
) -> None:
    completed = subprocess.run(
        [str(trainvm), "inspect-training-schedule", *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        raise SystemExit(message)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_training_schedule_parity.py TRAINVM")
    trainvm = Path(sys.argv[1]).resolve(strict=True)

    from rwkv_lab.training_runtime import schedules

    implementations: dict[
        str,
        tuple[
            Callable[[Mapping[str, Any]], object],
            Callable[[int, object], float],
        ],
    ] = {
        LINEAR: (
            schedules.LinearWarmupCosineConfiguration.from_resolved,
            schedules.linear_warmup_cosine_multiplier,
        ),
        POWERCOOL: (
            schedules.PowerCoolConfiguration.from_resolved,
            schedules.powercool_multiplier,
        ),
    }

    linear_configurations = [
        {
            "warmup_steps": warmup_steps,
            "max_steps": max_steps,
            "minimum_ratio": minimum_ratio,
        }
        for warmup_steps, max_steps, minimum_ratio in itertools.product(
            (0, 1, 1_000), (1, 2, 10_000), (0.0, 0.1, 1.0)
        )
    ]
    powercool_configurations = [
        {
            "warmup_steps": warmup_steps,
            "max_steps": max_steps,
            "minimum_ratio": minimum_ratio,
            "cooldown_fraction": cooldown_fraction,
            "power": power,
        }
        for warmup_steps, max_steps, minimum_ratio, cooldown_fraction, power in itertools.product(
            (0, 1, 1_000),
            (1, 2, 10_000),
            (0.0, 0.1, 1.0),
            (0.2, 1.0),
            (0.5, 1.0, 2.0, 3.0),
        )
        if warmup_steps <= max_steps
    ]

    for implementation, configurations in (
        (LINEAR, linear_configurations),
        (POWERCOOL, powercool_configurations),
    ):
        build_configuration, multiplier = implementations[implementation]
        for configuration in configurations:
            typed = build_configuration(configuration)
            max_step = int(configuration["max_steps"]) + 2
            oracle = [multiplier(step, typed) for step in range(max_step + 1)]
            native = native_values(trainvm, implementation, configuration, max_step)
            compare_values(implementation, configuration, native, oracle)

    exact_cases = (
        (
            LINEAR,
            {"warmup_steps": 4, "max_steps": 8, "minimum_ratio": 0.0},
            frozenset({0, 1, 2, 4, 8}),
        ),
        (
            POWERCOOL,
            {
                "warmup_steps": 4,
                "max_steps": 8,
                "minimum_ratio": 0.0,
                "cooldown_fraction": 0.25,
                "power": 2.0,
            },
            frozenset({0, 1, 3, 6, 7, 8}),
        ),
    )
    for implementation, configuration, exact_steps in exact_cases:
        build_configuration, multiplier = implementations[implementation]
        typed = build_configuration(configuration)
        max_step = max(exact_steps)
        oracle = [multiplier(step, typed) for step in range(max_step + 1)]
        native = native_values(trainvm, implementation, configuration, max_step)
        compare_values(
            implementation,
            configuration,
            native,
            oracle,
            exact_steps=exact_steps,
        )

    linear_invalid = [
        {"warmup_steps": -1, "max_steps": 2, "minimum_ratio": 0.1},
        {"warmup_steps": 0, "max_steps": 0, "minimum_ratio": 0.1},
        {"warmup_steps": 0, "max_steps": 2, "minimum_ratio": -0.1},
        {"warmup_steps": 0, "max_steps": 2, "minimum_ratio": 1.1},
        {"warmup_steps": True, "max_steps": 2, "minimum_ratio": 0.1},
        {"warmup_steps": 0, "max_steps": 2, "minimum_ratio": True},
        {"warmup_steps": 0, "max_steps": 2},
        {
            "warmup_steps": 0,
            "max_steps": 2,
            "minimum_ratio": 0.1,
            "unknown": 1,
        },
    ]
    powercool_valid = {
        "warmup_steps": 0,
        "max_steps": 2,
        "minimum_ratio": 0.0,
        "cooldown_fraction": 0.2,
        "power": 2.0,
    }
    powercool_invalid = []
    for field, value in (
        ("warmup_steps", -1),
        ("max_steps", 0),
        ("minimum_ratio", -0.1),
        ("minimum_ratio", 1.1),
        ("cooldown_fraction", 0.0),
        ("cooldown_fraction", 1.1),
        ("power", 0.0),
        ("power", -1.0),
        ("warmup_steps", True),
        ("power", True),
    ):
        powercool_invalid.append({**powercool_valid, field: value})
    powercool_invalid.extend(
        [
            {**powercool_valid, "warmup_steps": 3},
            {key: value for key, value in powercool_valid.items() if key != "power"},
            {**powercool_valid, "unknown": 1},
        ]
    )

    require_rejection_parity(
        trainvm,
        LINEAR,
        linear_invalid,
        schedules.LinearWarmupCosineConfiguration.from_resolved,
    )
    require_rejection_parity(
        trainvm,
        POWERCOOL,
        powercool_invalid,
        schedules.PowerCoolConfiguration.from_resolved,
    )

    encoded_valid = json.dumps(powercool_valid, separators=(",", ":"))
    require_command_rejection(
        trainvm,
        ["unsupported.schedule.v1", encoded_valid, "2"],
        "unknown schedule implementation was accepted",
    )
    require_command_rejection(
        trainvm,
        [POWERCOOL, encoded_valid, "100001"],
        "schedule output bound above 100000 was accepted",
    )
    require_command_rejection(
        trainvm,
        [POWERCOOL, encoded_valid, "-1"],
        "negative schedule output bound was accepted",
    )

    print(
        "native/Python training schedule parity passed at "
        f"relative tolerance {RELATIVE_TOLERANCE:g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
