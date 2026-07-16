import os
from pathlib import Path
import subprocess

import pytest

from rwkv_lab.generate import WorldVocab


def test_world_vocab_in_process_encode_matches_ztok_for_common_text():
    ztok = Path(os.environ.get("ZTOK", "/thearray/git/ztok/zig-out/bin/ztok"))
    vocab_path = Path(os.environ.get(
        "VOCAB", "/thearray/git/ztok/bench/vocabs/rwkv_vocab_v20230424.txt"))
    if not ztok.is_file() or not vocab_path.is_file():
        pytest.skip("ztok parity reference is not installed")
    vocab = WorldVocab(str(vocab_path))
    samples = [
        "Describe this image:\n",
        "a red fox, standing in grass",
        "猫と犬 — café",
    ]
    for text in samples:
        external = subprocess.run(
            [str(ztok), "encode", "--model", vocab.path, text],
            capture_output=True, text=True, check=True)
        assert vocab.encode(text) == [int(x) for x in external.stdout.split()]
