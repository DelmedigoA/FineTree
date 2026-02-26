from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SRC_STR = str(SRC)
if SRC_STR not in sys.path:
    sys.path.insert(0, SRC_STR)
