"""Freeze the methodology before the sealed holdout is opened, and gate the one allowed look.

    python -m sigma.freeze --note "why now"          # write frozen_model.json (refuses to overwrite)
    python -m sigma.freeze --check                   # has anything changed since?

The freeze records every number and every line of code that decides the ranking: factor list, weights,
data-quality bounds, filters, portfolio size, cost presets, period splits, and a hash of each source
file involved. The pass/fail criteria are written down *here*, before anyone has seen holdout results.
The holdout run (sigma.holdout) refuses to start unless the freeze exists and still matches, and it can
be opened once; a second look is recorded as such because by then the holdout has informed decisions.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from sigma import screen
from sigma.backtest import engine, splits
from sigma.core.config import DEFAULT_DOTENV
from sigma.data import sec
from sigma.factors import scoring

FROZEN_FILE = DEFAULT_DOTENV.parent / "frozen_model.json"
LEDGER_FILE = DEFAULT_DOTENV.parent / "holdout_ledger.txt"
TOP_N = 40
CODE_FILES = ("factors/scoring.py", "factors/fundamentals.py", "factors/market.py", "factors/sectors.py",
              "screen.py", "backtest/engine.py", "backtest/analysis.py", "backtest/splits.py",
              "backtest/run.py", "data/sec.py")
DD_LIMIT = -0.40

# Fixed before the holdout is opened. Changing them afterwards is visible in the freeze file.
CRITERIA = [
    "net CAGR beats the equal-weight universe",
    "mean rank IC is positive",
    "beats 95% of matched random portfolios",
    "worst drawdown within limit",
]


class AlreadyFrozen(RuntimeError):
    pass


class GateClosed(RuntimeError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest() -> dict[str, Any]:
    pkg = Path(__file__).resolve().parent
    sp = splits.Splits()
    return {
        "weights": scoring.effective_weights(),
        "categories": {k: list(v) for k, v in scoring.CATEGORIES.items()},
        "not_scored": dict(scoring.NOT_SCORED),
        "plausible": {k: list(v) for k, v in scoring.PLAUSIBLE.items()},
        "constants": {"MIN_COVERAGE": scoring.MIN_COVERAGE, "MIN_EV_TO_MCAP": scoring.MIN_EV_TO_MCAP,
                      "WORST_LEVERAGE": scoring.WORST_LEVERAGE, "TAX_RATE": scoring.TAX_RATE,
                      "PARSER_VERSION": sec.PARSER_VERSION},
        "filters": dataclasses.asdict(screen.DEFAULT_FILTERS),
        "top_n": TOP_N,
        "cost_bps": dict(engine.COST_BPS),
        "splits": {"train_end": sp.train_end.isoformat(), "validation_end": sp.validation_end.isoformat()},
        "criteria": {"checks": CRITERIA, "random_percentile": 95.0, "drawdown_limit": DD_LIMIT},
        "code": {f: _sha(pkg / f) for f in CODE_FILES},
    }


def digest(m: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(m, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def freeze(path: Path = FROZEN_FILE, note: str = "", replace: bool = False) -> dict[str, Any]:
    path = Path(path)
    if path.exists() and not replace:
        raise AlreadyFrozen(f"{path.name} already exists; re-freezing changes the methodology on record "
                            "(pass replace=True deliberately)")
    m = manifest()
    doc = {"frozen_at": datetime.now().isoformat(timespec="seconds"), "note": note,
           "digest": digest(m), "manifest": m}
    path.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    return doc


def _diff(a: Any, b: Any, prefix: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out: list[str] = []
        for k in sorted(set(a) | set(b)):
            out += _diff(a.get(k), b.get(k), f"{prefix}{k}.")
        return out
    return [] if a == b else [f"{prefix.rstrip('.')}: frozen {a!r} -> now {b!r}"]


def check(path: Path = FROZEN_FILE) -> tuple[bool, list[str]]:
    path = Path(path)
    if not path.exists():
        return False, ["not frozen: run `python -m sigma.freeze --note ...` first"]
    doc = json.loads(path.read_text(encoding="utf-8"))
    frozen = doc["manifest"]
    if digest(frozen) != doc["digest"]:
        return False, ["the freeze file itself was edited after it was written",
                       *_diff(frozen, json.loads(json.dumps(manifest())))]
    diffs = _diff(frozen, json.loads(json.dumps(manifest())))
    return (not diffs), diffs


def open_gate(frozen: Path, ledger: Path, reason: str, second_look: bool = False) -> None:
    """Raise unless the model is frozen and unchanged and the holdout has not already been opened."""
    ok, diffs = check(frozen)
    if not ok:
        if diffs and diffs[0].startswith("not frozen"):
            raise GateClosed("the model is not frozen yet - freeze it first")
        raise GateClosed("the model changed or the freeze file was edited since the freeze:\n  "
                         + "\n  ".join(diffs[:10]))
    ledger = Path(ledger)
    seen = ledger.read_text(encoding="utf-8") if ledger.exists() else ""
    if "\tFINAL\t" in seen and not second_look:
        raise GateClosed("the holdout was already opened once (see holdout_ledger.txt); a second look "
                         "means the holdout has informed decisions and must be labelled as such")
    tag = "SECOND LOOK" if second_look else "FINAL"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as fh:
        fh.write(f"{datetime.now().isoformat(timespec='seconds')}\t{tag}\t{reason}\n")


def judge(r: Mapping[str, float]) -> list[dict[str, Any]]:
    """Evaluate the pre-registered criteria. A missing number fails; it never passes by default."""
    def has(*keys: str) -> bool:
        return all(isinstance(r.get(k), int | float) and r[k] == r[k] for k in keys)

    rows = []
    rows.append((CRITERIA[0], has("cagr_net", "cagr_universe") and r["cagr_net"] > r["cagr_universe"],
                 f"{r.get('cagr_net')} vs {r.get('cagr_universe')}"))
    rows.append((CRITERIA[1], has("ic") and r["ic"] > 0, f"{r.get('ic')} (t {r.get('ic_t')})"))
    rows.append((CRITERIA[2], has("random_percentile") and r["random_percentile"] >= 95.0,
                 f"{r.get('random_percentile')}"))
    rows.append((CRITERIA[3], has("worst_dd", "dd_limit") and r["worst_dd"] >= r["dd_limit"],
                 f"{r.get('worst_dd')} vs limit {r.get('dd_limit')}"))
    return [{"criterion": c, "passed": bool(p), "value": v} for c, p, v in rows]


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - small CLI
    ap = argparse.ArgumentParser(prog="python -m sigma.freeze")
    ap.add_argument("--note", default="")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--replace", action="store_true", help="deliberately re-freeze (recorded in the file)")
    args = ap.parse_args(argv)
    if args.check:
        ok, diffs = check()
        sys.stdout.write("frozen model unchanged\n" if ok else "CHANGED:\n  " + "\n  ".join(diffs) + "\n")
        return 0 if ok else 1
    if not args.note:
        sys.stdout.write("Give a reason with --note, e.g. --note \"E2 reviewed; no further tuning\"\n")
        return 2
    doc = freeze(note=args.note, replace=args.replace)
    sys.stdout.write(f"Frozen {doc['frozen_at']}  digest {doc['digest'][:16]}...\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
