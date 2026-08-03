import json

import numpy as np

from rwkv_lab.ao3_cpt_tokens import _document_separator, pack_token_stream


class _FakeTokenizer:
    def id_to_token(self, token_id):
        return {7: "<raw-eos>"}.get(token_id)

    def token_to_id(self, token):
        return {"<chat-eos>": 9}.get(token)


def test_document_separator_prefers_model_config_raw_eos(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"text_config": {"eos_token_id": 7}})
    )
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"eos_token": "<chat-eos>"})
    )
    assert _document_separator(tmp_path, _FakeTokenizer()) == (
        "<raw-eos>",
        7,
        "model_config.text_config.eos_token_id",
    )


def test_pack_uses_every_full_context_once_and_drops_only_tail(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    tokens = np.arange(1, 24, dtype=np.uint32)
    tokens.tofile(cache / "train.tokens.bin")
    np.save(cache / "train.offsets.npy", np.asarray([0, 5, 12], dtype=np.uint64))
    np.save(cache / "train.lengths.npy", np.asarray([5, 7, 11], dtype=np.uint64))
    (cache / "manifest.json").write_text(
        json.dumps(
            {
                "splits": {
                    "train": {
                        "tokens_file": "train.tokens.bin",
                        "offsets_file": "train.offsets.npy",
                        "lengths_file": "train.lengths.npy",
                    }
                }
            }
        )
    )
    output = tmp_path / "packed"
    report = pack_token_stream(
        cache,
        "train",
        output,
        context_length=8,
        shuffle_documents=False,
    )
    packed = np.fromfile(output / "train.ctx8.bin", dtype=np.uint32).reshape(-1, 8)
    assert report["rows"] == 2
    assert report["packed_tokens"] == 16
    assert report["dropped_tail_tokens"] == 7
    assert packed.flatten().tolist() == list(range(1, 17))


def test_pack_document_shuffle_is_deterministic(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    np.asarray([10, 11, 20, 21, 30, 31], dtype=np.uint32).tofile(
        cache / "train.tokens.bin"
    )
    np.save(cache / "train.offsets.npy", np.asarray([0, 2, 4], dtype=np.uint64))
    np.save(cache / "train.lengths.npy", np.asarray([2, 2, 2], dtype=np.uint64))
    (cache / "manifest.json").write_text(
        json.dumps(
            {
                "splits": {
                    "train": {
                        "tokens_file": "train.tokens.bin",
                        "offsets_file": "train.offsets.npy",
                        "lengths_file": "train.lengths.npy",
                    }
                }
            }
        )
    )
    values = []
    for name in ("a", "b"):
        output = tmp_path / name
        pack_token_stream(cache, "train", output, context_length=3, seed=9)
        values.append((output / "train.ctx3.bin").read_bytes())
    assert values[0] == values[1]
