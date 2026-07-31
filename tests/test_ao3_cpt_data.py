import io
import json

import zstandard as zstd

from rwkv_lab.ao3_cpt_data import (
    AO3FilterPolicy,
    deterministic_split,
    filter_record,
    prepare_ao3_corpus,
    render_document,
)


def record(*, text="A neutral prose passage. " * 30, metadata=None, record_id="123"):
    return {
        "id": record_id,
        "title": "Synthetic title",
        "metadata": metadata or {"Language": "English", "Rating": "Explicit"},
        "text": text,
    }


def write_zstd_jsonl(path, rows):
    with path.open("wb") as raw:
        with zstd.ZstdCompressor().stream_writer(raw, closefd=False) as stream:
            with io.TextIOWrapper(stream, encoding="utf-8") as text:
                for row in rows:
                    text.write(json.dumps(row) + "\n")


def read_zstd_jsonl(path):
    with path.open("rb") as raw, zstd.ZstdDecompressor().stream_reader(raw) as stream:
        return [json.loads(line) for line in io.TextIOWrapper(stream, encoding="utf-8")]


def test_rejects_singular_and_plural_underage_archive_warning():
    for key in ("Archive Warning", "Archive Warnings"):
        value = record(metadata={"Language": "English", key: "Underage Sex"})
        decision = filter_record(value)
        assert not decision.accepted
        assert decision.reason == "minor_archive_warning"


def test_rejects_explicit_minor_tag_but_not_negated_or_aged_up_tag():
    rejected = record(
        metadata={"Language": "English", "Additional Tags": "Fluff, Underage Sex - Freeform"}
    )
    assert filter_record(rejected).reason == "minor_metadata_tag"

    for tag in ("Not Underage", "Aged-Up Characters"):
        accepted = record(metadata={"Language": "English", "Additional Tags": tag})
        assert filter_record(accepted).accepted


def test_rejects_numeric_minor_age_only_in_sexual_context():
    unsafe = record(
        text=("The character was a 17-year-old student. " + "Naked sexual scene. ") * 20
    )
    assert filter_record(unsafe).reason == "minor_text_age_context"

    safe = record(
        text="A 17-year-old bridge was restored by engineers. " + "Historical notes. " * 30
    )
    assert filter_record(safe).accepted


def test_render_uses_title_and_prose_but_not_metadata_tags():
    value = record(metadata={"Language": "English", "Additional Tags": "Tag syntax"})
    rendered = render_document(value)
    assert rendered.startswith("Synthetic title\n\n")
    assert "Tag syntax" not in rendered


def test_split_is_stable():
    policy = AO3FilterPolicy(eval_fraction=0.5, split_seed=7)
    assert deterministic_split("same", policy) == deterministic_split("same", policy)


def test_prepare_keeps_source_immutable_and_logs_no_rejected_text(tmp_path):
    source = tmp_path / "source"
    selected = source / "selected"
    selected.mkdir(parents=True)
    shard = selected / "part.jsonl.zst"
    unsafe = record(
        record_id="unsafe-id",
        metadata={"Language": "English", "Archive Warnings": "Underage Sex"},
        text="private rejected excerpt " * 30,
    )
    safe = record(record_id="safe-id")
    write_zstd_jsonl(shard, [unsafe, safe])
    before = shard.read_bytes()

    output = tmp_path / "prepared"
    report = prepare_ao3_corpus(
        source,
        output,
        policy=AO3FilterPolicy(eval_fraction=0.0, output_shard_records=1),
    )
    assert shard.read_bytes() == before
    assert report["scanned"] == 2
    assert report["accepted"] == 1
    assert report["counts"]["minor_archive_warning"] == 1

    rejection = read_zstd_jsonl(next((output / "rejections").glob("*.zst")))[0]
    serialized = json.dumps(rejection)
    assert "unsafe-id" not in serialized
    assert "private rejected excerpt" not in serialized
    prepared = read_zstd_jsonl(next((output / "documents/train").glob("*.zst")))[0]
    assert prepared["id"] == "safe-id"
    assert prepared["kind"] == "pretrain"
