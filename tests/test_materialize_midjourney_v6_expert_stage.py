from scripts.materialize_midjourney_v6_expert_stage import (
    deduplicate,
    split_records,
)


def _row(index, domain, family, digest=None):
    return {
        "source_shard": "train_000.parquet",
        "source_row": index,
        "domain": domain,
        "family_id": family,
        "sha256": digest or f"sha-{index}",
    }


def test_dedup_and_family_disjoint_split():
    rows = [
        *[_row(i, "photo", f"photo-{i // 2}") for i in range(12)],
        *[_row(100 + i, "animation", f"animation-{i // 2}") for i in range(12)],
        _row(999, "photo", "duplicate", digest="sha-0"),
    ]
    unique = deduplicate(rows)
    assert len(unique) == 24
    train, eval_, heldout = split_records(
        unique, train_per_domain=6, eval_per_domain=2
    )
    assert len(train) == 12
    assert len(eval_) == 4
    assert {row["family_id"] for row in train}.isdisjoint(heldout)
    assert {row["family_id"] for row in eval_}.issubset(heldout)
