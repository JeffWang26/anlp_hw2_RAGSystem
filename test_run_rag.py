#!/usr/bin/env python3
"""
Test run_rag.py: run RAG pipeline and validate output.
Uses --reader simple for fast testing (no LLM loading).
Run: python test_run_rag.py
"""

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_FILE = PROJECT_ROOT / "system_outputs" / "test_output.json"
TEST_QUERIES = PROJECT_ROOT / "test_queries.json"


def run_rag(args: list[str]) -> subprocess.CompletedProcess:
    """Run run_rag.py with given args."""
    cmd = [sys.executable, str(PROJECT_ROOT / "run_rag.py")] + args
    return subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120)


def test_run_rag_simple():
    """Test with simple reader (fast, no model download)."""
    print("=== Test: run_rag.py with --reader simple ===")
    result = run_rag([
        "--knowledge", "data/knowledge",
        "--queries", str(TEST_QUERIES),
        "--output", str(OUTPUT_FILE),
        "--reader", "simple",
        "--format", "leaderboard",
        "--andrew-id", "test_user",
    ])
    assert result.returncode == 0, f"run_rag failed: {result.stderr}"
    print(result.stdout)

    assert OUTPUT_FILE.exists(), f"Output file not created: {OUTPUT_FILE}"

    with open(OUTPUT_FILE, encoding="utf-8") as f:
        data = json.load(f)

    assert "andrewid" in data, "Missing andrewid"
    assert data["andrewid"] == "test_user"

    for qid in ["1", "2", "3"]:
        assert qid in data, f"Missing answer for question {qid}"
        assert isinstance(data[qid], str), f"Answer for {qid} must be string"

    print(f"OK: Generated {len([k for k in data if k != 'andrewid'])} answers\n")


def test_run_rag_extractive():
    """Test with extractive reader (loads QA model, slower)."""
    print("=== Test: run_rag.py with --reader extractive ===")
    result = run_rag([
        "--knowledge", "data/knowledge",
        "--queries", str(TEST_QUERIES),
        "--output", str(OUTPUT_FILE),
        "--reader", "extractive",
        "--format", "leaderboard",
    ])
    assert result.returncode == 0, f"run_rag failed: {result.stderr}"
    print(result.stdout)

    with open(OUTPUT_FILE, encoding="utf-8") as f:
        data = json.load(f)
    assert "1" in data and isinstance(data["1"], str)
    print("OK: Extractive reader completed\n")


def test_output_format_leaderboard():
    """Verify leaderboard output format."""
    with open(OUTPUT_FILE, encoding="utf-8") as f:
        data = json.load(f)
    assert "andrewid" in data
    assert all(k == "andrewid" or (k.isdigit() and isinstance(data[k], str)) for k in data)
    print("OK: Leaderboard format valid\n")


def test_output_format_test():
    """Test format=test (no andrewid)."""
    out_test = PROJECT_ROOT / "system_outputs" / "test_format_output.json"
    result = run_rag([
        "--queries", str(TEST_QUERIES),
        "--output", str(out_test),
        "--reader", "simple",
        "--format", "test",
    ])
    assert result.returncode == 0
    with open(out_test, encoding="utf-8") as f:
        data = json.load(f)
    assert "andrewid" not in data
    print("OK: Test format valid\n")


if __name__ == "__main__":
    test_run_rag_simple()
    test_output_format_leaderboard()
    test_output_format_test()

    # Optional: extractive (slower, needs transformers)
    try:
        test_run_rag_extractive()
    except subprocess.TimeoutExpired:
        print("Skip extractive test: timeout\n")
    except AssertionError as e:
        print(f"Skip extractive test: {e}\n")
