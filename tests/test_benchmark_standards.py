"""Unit and integration tests for the Standardized Comparative Benchmark."""

import os
os.environ.setdefault("RWKV8_FORCE_PYREF", "1")

import json
from pathlib import Path
import pytest

from rwkv_lab.benchmark_standards import run_benchmark


def test_standardized_comparative_benchmark_execution():
    """Verify that the comparative benchmark executes correctly and writes its report."""
    # Ensure any existing benchmark report is removed
    report_path = Path("runs/benchmark_report.json")
    report_path.unlink(missing_ok=True)

    # Run the standardized benchmark (runs fast with epochs=15 on CPU)
    results = run_benchmark(
        epochs=1,
        d_model=16,
        n_layers=2,
        head_size=16,
        train_rows=1,
        sequence_length=16,
    )

    # Verify that the return dict structure is correct
    assert "multilingual_stories" in results
    assert "logic_niah" in results

    assert "Standard RWKV-8" in results["multilingual_stories"]
    assert "ROSA + BLT" in results["multilingual_stories"]
    assert "ROSA Latent Thinker" in results["multilingual_stories"]

    assert "1l_no_think_acc" in results["logic_niah"]
    assert "1l_think_acc" in results["logic_niah"]
    assert "2l_solver_acc" in results["logic_niah"]

    # Verify that the JSON report was written and is identical
    assert report_path.is_file()
    with report_path.open() as f:
        saved_results = json.load(f)

    assert saved_results == results

    # Verify exact logical reasoning accuracy assertions
    # 1. 1-Layer ROSA without thinking must fail on the 2-hop logic task (0% accuracy)
    assert results["logic_niah"]["1l_no_think_acc"] == 0.0

    # 2. 1-Layer ROSA with 1 thinking step must succeed on the 2-hop logic task (100% accuracy)
    assert results["logic_niah"]["1l_think_acc"] == 1.0

    # 3. 2-Layer ROSA physical solver must succeed on the 2-hop logic task (100% accuracy)
    assert results["logic_niah"]["2l_solver_acc"] == 1.0
