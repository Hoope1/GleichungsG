"""Ensure code base passes ruff checks and formatting."""

from __future__ import annotations

import subprocess


def test_ruff_check() -> None:
    """ruff should report no lint errors."""
    result = subprocess.run(
        ["python", "-m", "ruff", "check", "."],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_ruff_format() -> None:
    """ruff should report no formatting changes."""
    result = subprocess.run(
        ["python", "-m", "ruff", "format", "--check", "."],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
