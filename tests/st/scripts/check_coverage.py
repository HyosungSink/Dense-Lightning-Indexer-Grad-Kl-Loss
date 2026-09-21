#!/usr/bin/env python3
"""Check the test suite against documented operator requirements."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tests.common.coverage import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
