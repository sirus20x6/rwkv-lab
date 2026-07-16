import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/fetch_doclingmatix.py"
SPEC = importlib.util.spec_from_file_location("doclingmatix_fetch", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


class FakeApi:
    def list_repo_files(self, *args, **kwargs):
        return [
            "README.md",
            "default/train/0002.parquet",
            "default/train/0000.parquet",
            "default/train/0001.parquet",
            "default/test/0000.parquet",
        ]


def test_shard_selection_is_sorted_and_bounded():
    assert module.shard_names(FakeApi(), 2) == [
        "default/train/0000.parquet", "default/train/0001.parquet"]


def test_pinned_revisions_and_defaults():
    assert len(module.PARQUET_REVISION) == 40
    assert len(module.MAIN_REVISION) == 40
    assert module.DEFAULT_SHARDS == 218
