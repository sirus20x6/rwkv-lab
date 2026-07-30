import json
from collections import Counter

from scripts.materialize_midjourney_v6_continuation import (
    additional_stage_contract,
    assign_resolution_buckets,
    fractional_targets,
    select_candidate_shards,
    select_training_records,
)


def _row(index: int, domain: str, shard: str = "train_000.parquet"):
    return {
        "source_shard": shard,
        "source_row": index,
        "domain": domain,
        "family_id": f"family-{index}",
        "sha256": f"sha-{index}",
    }


def test_fractional_targets_floor_each_domain_without_rebalancing():
    assert fractional_targets({"photo": 196_630, "animation": 269_178}, 0.30) == {
        "photo": 58_989,
        "animation": 80_753,
    }
    assert fractional_targets({"photo": 137_641, "animation": 188_425}, 0.30) == {
        "photo": 41_292,
        "animation": 56_527,
    }


def test_additional_stage_contract_excludes_prior_tranche(tmp_path):
    stage = tmp_path / "tranche-a"
    stage.mkdir()
    rows = [
        {
            "source_shard": "train_001.parquet",
            "source_row": 7,
            "sha256": "first",
        },
        {
            "source_shard": "train_002.parquet",
            "source_row": 9,
            "sha256": "second",
        },
    ]
    (stage / "metadata.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    (stage / "selection_plan.json").write_text(
        json.dumps(
            {
                "selected_shards": [
                    "train_001.parquet",
                    "train_002.parquet",
                ]
            }
        ),
        encoding="utf-8",
    )

    keys, hashes, shards = additional_stage_contract([stage])

    assert keys == {
        ("train_001.parquet", 7),
        ("train_002.parquet", 9),
    }
    assert hashes == {"first", "second"}
    assert shards == ["train_001.parquet", "train_002.parquet"]


def test_candidate_shards_reuse_existing_pool_before_new_downloads():
    rows = [
        *[_row(i, "photo", "cached.parquet") for i in range(4)],
        *[_row(100 + i, "animation", "cached.parquet") for i in range(4)],
        *[_row(200 + i, "photo", "new.parquet") for i in range(4)],
        *[_row(300 + i, "animation", "new.parquet") for i in range(4)],
    ]
    shards, counts = select_candidate_shards(
        rows,
        targets={"photo": 5, "animation": 5},
        preferred_shards=["cached.parquet"],
        revision="revision",
        reserve_fraction=0,
    )
    assert shards == ["cached.parquet", "new.parquet"]
    assert counts == {"photo": 8, "animation": 8}


def test_selection_and_resolution_buckets_are_exact_and_deterministic():
    rows = [
        *[_row(i, "photo") for i in range(13)],
        *[_row(100 + i, "animation") for i in range(17)],
    ]
    selected = select_training_records(rows, {"photo": 11, "animation": 14})
    first = assign_resolution_buckets(selected, [512, 640, 768, 896, 1024])
    second = assign_resolution_buckets(selected, [1024, 896, 768, 640, 512])
    assert first == second
    assert Counter(row["domain"] for row in first) == {
        "photo": 11,
        "animation": 14,
    }
    per_domain = {
        domain: Counter(
            row["resolution_bucket"] for row in first if row["domain"] == domain
        )
        for domain in ("photo", "animation")
    }
    assert max(per_domain["photo"].values()) - min(per_domain["photo"].values()) <= 1
    assert (
        max(per_domain["animation"].values()) - min(per_domain["animation"].values())
        <= 1
    )
