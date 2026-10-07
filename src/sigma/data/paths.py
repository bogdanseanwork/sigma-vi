"""Where bulk market data lives on the user's machine.

Bulk data (prices, SEC facts) is several GB, so it is kept as local Parquet files rather than in the
free 1 GB Neon database. On Windows it defaults to %LOCALAPPDATA%\\SIGMA-VI\\data so it never lands in
a OneDrive-synced folder. Override with SIGMA_DATA_DIR.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path


def data_dir(env: Mapping[str, str] | None = None, platform: str | None = None) -> Path:
    env = os.environ if env is None else env
    platform = sys.platform if platform is None else platform
    if env.get("SIGMA_DATA_DIR"):
        return Path(env["SIGMA_DATA_DIR"])
    if platform == "win32" and env.get("LOCALAPPDATA"):
        return Path(env["LOCALAPPDATA"]) / "SIGMA-VI" / "data"
    return Path.home() / ".sigma-vi" / "data"
