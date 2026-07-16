from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_next_lever_launchers_default_to_combined_ocr_cache_contract():
    expected = (
        "vision_next_shard_000_ocr10_train.jsonl",
        "vision_next_shard_000_ocr10_eval.jsonl",
        "moonvit_next_${PREFIX}_shard_000_ocr10",
        "fusion_so400m_next_${PREFIX}_shard_000_ocr10",
        "vision_next_so400m_${PREFIX}_shard_000_ocr10.cache.json",
    )
    for name in ("prepare_vision_next_levers.sh", "run_vision_next_levers.sh"):
        text = (ROOT / "scripts" / name).read_text()
        for value in expected:
            assert value in text
