"""Ensure `src/` is on `sys.path` across all pytest versions."""
from __future__ import annotations

from pathlib import Path
import sys

SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
