#!/usr/bin/env python3
"""Report which of a model's parameters a checkpoint actually populates.

Transformers reports missing and unexpected keys as flat lists, which is easy to
skim past: a checkpoint that nests a whole submodule one path segment away from
where the model class looks produces a large, uniform, undramatic list, and the
load still "succeeds" with that submodule randomly initialized. This groups both
sides by family so that case is obvious, and reports whether a pure prefix
rename reconciles them.

Nothing is written and no weights are loaded: the model is instantiated on the
meta device and the checkpoint is read from its safetensors index.

    python scripts/audit_checkpoint_keys.py /path/to/model
    python scripts/audit_checkpoint_keys.py /path/to/model --rename old. new.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path


def families(keys, depth: int = 3) -> collections.Counter:
    counter: collections.Counter = collections.Counter()
    for key in keys:
        counter[".".join(key.split(".")[:depth])] += 1
    return counter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path)
    parser.add_argument(
        "--rename",
        nargs=2,
        metavar=("FROM", "TO"),
        help="test whether renaming this checkpoint key prefix reconciles the sets",
    )
    parser.add_argument("--depth", type=int, default=3)
    arguments = parser.parse_args()

    import torch
    import transformers
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(str(arguments.model_path))
    architecture = config.architectures[0]
    model_class = getattr(transformers, architecture)
    print(f"transformers {transformers.__version__} | class {architecture}")

    with torch.device("meta"):
        model = model_class._from_config(config)
    model_keys = set(model.state_dict())

    index = arguments.model_path / "model.safetensors.index.json"
    checkpoint_keys = set(json.loads(index.read_text())["weight_map"])

    ignored = getattr(model_class, "_keys_to_ignore_on_load_unexpected", None) or []
    print(f"declared unexpected-key ignore patterns: {list(ignored)}")

    def is_declared_ignorable(key: str) -> bool:
        return any(re.search(pattern, key) for pattern in ignored)

    print(
        "declared checkpoint conversion mapping: "
        f"{getattr(model_class, '_checkpoint_conversion_mapping', '<absent>')}"
    )

    missing = model_keys - checkpoint_keys
    unexpected = checkpoint_keys - model_keys
    print(f"\nmodel tensors {len(model_keys)} | checkpoint tensors {len(checkpoint_keys)}")
    print(f"MISSING    (model wants, checkpoint lacks -> randomly initialized): {len(missing)}")
    print(f"UNEXPECTED (checkpoint has, model ignores -> dropped):              {len(unexpected)}")

    for label, keys in (("MISSING", missing), ("UNEXPECTED", unexpected)):
        if not keys:
            continue
        print(f"\n--- {label} by family ---")
        for family, count in sorted(
            families(keys, arguments.depth).items(), key=lambda item: -item[1]
        ):
            print(f"  {count:6d}  {family}")

    if arguments.rename:
        source, target = arguments.rename
        renamed = {
            target + key[len(source):] if key.startswith(source) else key
            for key in checkpoint_keys
        }
        still_missing = model_keys - renamed
        still_unexpected = renamed - model_keys
        # Keys the model class itself declares ignorable are not evidence of a
        # problem, so the verdict is about what remains unexplained.
        unexplained = {key for key in still_unexpected if not is_declared_ignorable(key)}
        print(f"\n--- after renaming {source} -> {target} ---")
        print(f"  still missing:            {len(still_missing)}")
        print(f"  still unexpected:         {len(still_unexpected)}")
        print(f"  of those, declared ignorable: {len(still_unexpected) - len(unexplained)}")
        print(f"  unexplained remaining:    {len(unexplained)}")
        print(f"  reconciles exactly: {not still_missing and not unexplained}")

    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
