"""Local Parquet store for bulk data (needs pyarrow: installed by setup via the [data] extra)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd


class ParquetStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def exists(self, *parts: str) -> bool:
        return self.path(*parts).exists()

    def write(self, rows: Iterable[Mapping[str, Any]] | pd.DataFrame, *parts: str) -> int:
        df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
        dest = self.path(*parts)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(dest)  # atomic: a half-written file never looks complete
        return len(df)

    def read(self, *parts: str, columns: list[str] | None = None,
             filters: list[tuple[str, str, Any]] | None = None) -> pd.DataFrame:
        """Read one file or every part in a folder; ``filters`` (pyarrow syntax) skip rows on read."""
        target = self.path(*parts)
        kw: dict[str, Any] = {"columns": columns}
        if filters:
            kw["filters"] = filters
        if target.is_dir():
            files = sorted(target.glob("*.parquet"))
            if not files:
                return pd.DataFrame(columns=columns or [])
            return pd.concat((pd.read_parquet(f, **kw) for f in files), ignore_index=True)
        return pd.read_parquet(target, **kw)
