from scripts.prepare_mage_flow_expert_manifests import (
    split_balanced_animation_holdout,
)


def test_animation_holdout_is_balanced_deterministic_and_disjoint():
    rows = [
        {
            "image": f"/images/{content_class}-{index}.png",
            "image_sha256": f"{content_class}-{index}",
            "content_class": content_class,
        }
        for content_class in ("sfw", "nsfw")
        for index in range(10)
    ]
    train_a, eval_a, counts = split_balanced_animation_holdout(
        rows, count=8, seed=42
    )
    train_b, eval_b, _ = split_balanced_animation_holdout(
        list(reversed(rows)), count=8, seed=42
    )
    eval_ids = {row["image_sha256"] for row in eval_a}
    assert counts == {"sfw": 4, "nsfw": 4}
    assert len(train_a) == 12
    assert len(eval_a) == 8
    assert eval_ids.isdisjoint(row["image_sha256"] for row in train_a)
    assert eval_ids == {row["image_sha256"] for row in eval_b}
    assert {row["image_sha256"] for row in train_a} == {
        row["image_sha256"] for row in train_b
    }
