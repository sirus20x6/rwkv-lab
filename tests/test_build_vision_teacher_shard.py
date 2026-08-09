import json
import subprocess
import sys
from pathlib import Path

# Anchored to this checkout: a bare relative script path runs whatever the
# caller's working directory happens to contain, which is a test grading source
# it was not pointed at.
REPOSITORY = Path(__file__).resolve().parents[1]


def test_builds_exact_nonoverlapping_slice(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    source.write_text("".join(
        json.dumps({"image": f"image-{index}.jpg", "text": str(index)}) + "\n"
        for index in range(9)))
    output = tmp_path / "shard.jsonl"
    subprocess.run([
        sys.executable, str(REPOSITORY / "scripts/build_vision_teacher_shard.py"),
        "--source", str(source), "--output", str(output),
        "--index", "1", "--size", "3"], check=True)
    rows = [json.loads(line) for line in output.open()]
    assert [row["image"] for row in rows] == [
        "image-3.jpg", "image-4.jpg", "image-5.jpg"]
    summary = json.loads(output.with_suffix(".summary.json").read_text())
    assert summary["start_row_inclusive"] == 3
    assert summary["stop_row_exclusive"] == 6
    assert summary["unique_images"] == 3
