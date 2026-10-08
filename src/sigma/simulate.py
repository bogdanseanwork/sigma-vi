"""Simulation engine: the distribution of outcomes for the strategy over the next 1 to 3 years.

    python -m sigma.simulate --paths 1000000 --years 3

Simulates the strategy's net returns and a benchmark's, jointly, from the history the point-in-time
backtest produced (backtest_returns.csv). Five path generators, each stressing something different:

  bootstrap     stationary block bootstrap: keeps the clustering of good and bad periods in history
  student_t     fat-tailed multivariate Student-t fitted to history
  regime        two-state Markov chain (calm / stressed) with its own mean and covariance per state
  uncertainty   Student-t with the mean and volatility themselves drawn from their estimation error
  stress        bootstrap with a historical crisis window spliced in, sometimes scaled up

Every path also pays a randomly drawn trading-cost drag. WHAT THIS DOES AND DOES NOT TELL YOU: it
shows how wide the range of outcomes is, and how often drawdowns and underperformance happen, GIVEN
that the history is representative. It cannot show that the strategy has an edge - it inherits whatever
edge (or luck) is in the history. No language model is involved; results are reproducible from the seed.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

GENERATORS = ("bootstrap", "student_t", "regime", "uncertainty", "stress")
DEFAULT_SHARES = {"bootstrap": 0.25, "student_t": 0.25, "regime": 0.15, "uncertainty": 0.20, "stress": 0.15}
T_DF = 5.0
MIN_RETURN = -0.95
CHUNK = 20_000
CONVERGENCE_POINTS = (10_000, 100_000, 500_000, 1_000_000)
METRICS = ("cagr", "vol", "sharpe", "max_dd", "terminal", "bench_terminal", "excess")


@dataclass(frozen=True)
class Config:
    paths: int = 1_000_000
    years: float = 3.0
    periods_per_year: int = 12
    seed: int = 2026
    shares: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_SHARES))
    turnover_per_year: float = 2.0       # share of the portfolio replaced per year
    cost_bps_median: float = 30.0        # one-way, median of the drawn cost
    cost_bps_sigma: float = 0.5          # log-sd of the drawn cost: occasionally several times worse
    workers: int = 1

    @property
    def steps(self) -> int:
        return max(1, round(self.years * self.periods_per_year))


@dataclass(frozen=True)
class History:
    """Aligned per-period returns: column 0 strategy, column 1 benchmark."""
    returns: np.ndarray            # (T, 2)
    periods_per_year: int

    @classmethod
    def from_frame(cls, strategy: pd.Series, benchmark: pd.Series, periods_per_year: int) -> History:
        both = pd.concat([strategy, benchmark], axis=1).dropna()
        return cls(both.to_numpy(dtype=float), periods_per_year)


# ---------------------------------------------------------------------------------------------
# Generators: each returns an array (n, steps, 2)
# ---------------------------------------------------------------------------------------------
def _mvt(rng: np.random.Generator, n: int, steps: int, mean: np.ndarray, cov: np.ndarray, df: float
         ) -> np.ndarray:
    """Multivariate Student-t draws with the given mean and covariance (scale set so variance matches)."""
    k = len(mean)
    chol = np.linalg.cholesky(cov * (df - 2) / df + 1e-12 * np.eye(k))
    z = rng.standard_normal((n, steps, k)) @ chol.T
    w = rng.chisquare(df, size=(n, steps, 1)) / df
    return mean + z / np.sqrt(w)


def _stationary_indices(rng: np.random.Generator, n: int, steps: int, size: int, mean_block: float
                        ) -> np.ndarray:
    """Politis-Romano indices: continue to the next period with prob 1-1/b, otherwise jump anywhere."""
    jump = rng.random((n, steps)) < 1.0 / mean_block
    jump[:, 0] = True
    start = rng.integers(0, size, (n, steps))
    first = np.maximum.accumulate(np.where(jump, np.arange(steps), 0), axis=1)   # start of my block
    base = np.take_along_axis(start, first, axis=1)
    return (base + np.arange(steps)[None, :] - first) % size


def gen_bootstrap(rng: np.random.Generator, n: int, steps: int, h: History) -> np.ndarray:
    size = len(h.returns)
    mean_block = max(2.0, size ** (1 / 3))
    return h.returns[_stationary_indices(rng, n, steps, size, mean_block)]


def gen_student_t(rng: np.random.Generator, n: int, steps: int, h: History) -> np.ndarray:
    return _mvt(rng, n, steps, h.returns.mean(axis=0), np.cov(h.returns.T), T_DF)


def gen_uncertainty(rng: np.random.Generator, n: int, steps: int, h: History) -> np.ndarray:
    """Per path: draw the true mean from N(mean, cov/T) and scale volatility by a chi-square draw."""
    t = len(h.returns)
    mean, cov = h.returns.mean(axis=0), np.cov(h.returns.T)
    chol = np.linalg.cholesky(cov + 1e-12 * np.eye(2))
    mus = mean + (rng.standard_normal((n, 2)) @ chol.T) / math.sqrt(t)
    vol_scale = np.sqrt((t - 1) / rng.chisquare(t - 1, size=n))   # estimated sd is itself uncertain
    z = _mvt(rng, n, steps, np.zeros(2), cov, T_DF)
    return mus[:, None, :] + z * vol_scale[:, None, None]


def _regimes(h: History) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Label each historical period stressed (benchmark return below its lower third) or calm."""
    bench = h.returns[:, 1]
    stressed = bench <= np.quantile(bench, 1 / 3)
    trans = np.full((2, 2), 0.5)
    for a in (0, 1):
        idx = np.where(stressed[:-1] == bool(a))[0]
        if len(idx):
            nxt = stressed[idx + 1]
            trans[a] = [(~nxt).mean(), nxt.mean()]
    params = []
    for flag in (False, True):
        x = h.returns[stressed == flag]
        params.append((x.mean(axis=0), np.cov(x.T) if len(x) > 2 else np.cov(h.returns.T)))
    return stressed, trans, params


def gen_regime(rng: np.random.Generator, n: int, steps: int, h: History) -> np.ndarray:
    _, trans, params = _regimes(h)
    state = (rng.random(n) < 1 / 3).astype(int)
    out = np.empty((n, steps, 2))
    for t in range(steps):
        for s in (0, 1):
            idx = np.where(state == s)[0]
            if len(idx):
                out[idx, t, :] = _mvt(rng, len(idx), 1, params[s][0], params[s][1], T_DF)[:, 0, :]
        state = (rng.random(n) < trans[state, 1]).astype(int)
    return out


def worst_window(h: History, length: int) -> int:
    """Start index of the historical window with the lowest benchmark growth."""
    length = min(length, len(h.returns))
    g = np.array([np.prod(1 + h.returns[i:i + length, 1]) for i in range(len(h.returns) - length + 1)])
    return int(np.argmin(g))


def gen_stress(rng: np.random.Generator, n: int, steps: int, h: History) -> np.ndarray:
    """Bootstrap paths; in each, the worst historical window (scaled 1x-1.5x) is spliced in somewhere."""
    out = gen_bootstrap(rng, n, steps, h)
    length = min(max(2, h.periods_per_year // 2), steps, len(h.returns))
    crisis = h.returns[worst_window(h, length):][:length]
    start = rng.integers(0, steps - length + 1, n)
    scale = rng.uniform(1.0, 1.5, n)
    for i in range(n):
        out[i, start[i]:start[i] + length, :] = np.clip(crisis * scale[i], MIN_RETURN, None)
    return out


_GEN: dict[str, Callable[[np.random.Generator, int, int, History], np.ndarray]] = {
    "bootstrap": gen_bootstrap, "student_t": gen_student_t, "regime": gen_regime,
    "uncertainty": gen_uncertainty, "stress": gen_stress,
}


# ---------------------------------------------------------------------------------------------
# Costs and per-path metrics
# ---------------------------------------------------------------------------------------------
def apply_costs(rng: np.random.Generator, paths: np.ndarray, cfg: Config) -> np.ndarray:
    """Subtract a per-period trading-cost drag from the strategy column; the benchmark pays none."""
    n = paths.shape[0]
    bps = cfg.cost_bps_median * np.exp(cfg.cost_bps_sigma * rng.standard_normal(n))
    drag = cfg.turnover_per_year / cfg.periods_per_year * 2 * bps / 1e4   # buy and sell sides
    out = paths.copy()
    out[:, :, 0] -= drag[:, None]
    return out


def path_metrics(paths: np.ndarray, ppy: int) -> dict[str, np.ndarray]:
    p, b = paths[:, :, 0], paths[:, :, 1]
    steps = p.shape[1]
    nav = np.cumprod(1 + p, axis=1)
    peak = np.maximum.accumulate(np.concatenate([np.ones((len(p), 1)), nav], axis=1), axis=1)[:, 1:]
    sd = p.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = np.where(sd > 0, p.mean(axis=1) / sd * math.sqrt(ppy), np.nan)
    term = nav[:, -1] - 1
    bterm = np.prod(1 + b, axis=1) - 1
    return {"cagr": (1 + term) ** (ppy / steps) - 1, "vol": sd * math.sqrt(ppy), "sharpe": sharpe,
            "max_dd": (1 - nav / peak).max(axis=1), "terminal": term, "bench_terminal": bterm,
            "excess": term - bterm}


def _run_chunk(args: tuple[str, int, int, np.random.SeedSequence, Config, History]) -> dict[str, np.ndarray]:
    name, n, steps, seed, cfg, hist = args
    rng = np.random.default_rng(seed)
    # a fat-tailed draw cannot lose more than 95% in one period
    raw = np.clip(_GEN[name](rng, n, steps, hist), MIN_RETURN, None)
    paths = apply_costs(rng, raw, cfg)
    m = path_metrics(paths, cfg.periods_per_year)
    return {k: v.astype(np.float32) for k, v in m.items()}


# ---------------------------------------------------------------------------------------------
# Run, summarise, converge
# ---------------------------------------------------------------------------------------------
@dataclass
class SimResult:
    config: Config
    metrics: dict[str, np.ndarray]       # per path, all generators concatenated
    generator: np.ndarray                # generator index per path
    ledger: dict[str, int]               # completed paths per generator
    seconds: float

    @property
    def completed(self) -> int:
        return sum(self.ledger.values())


def run(hist: History, cfg: Config, progress: Callable[[str], None] | None = None) -> SimResult:
    """Run every generator in chunks, each chunk with its own seed stream, so results do not depend on
    how many workers ran them. Only chunks that finished are counted in the ledger."""
    t0 = time.time()
    root = np.random.SeedSequence(cfg.seed)
    jobs: list[tuple[str, int, int, np.random.SeedSequence, Config, History]] = []
    plan = {g: round(cfg.paths * cfg.shares.get(g, 0.0)) for g in GENERATORS}
    plan[GENERATORS[0]] += cfg.paths - sum(plan.values())     # rounding goes to the first generator
    seeds = iter(root.spawn(sum(math.ceil(v / CHUNK) for v in plan.values()) or 1))
    for g, total in plan.items():
        left = total
        while left > 0:
            m = min(CHUNK, left)
            jobs.append((g, m, cfg.steps, next(seeds), cfg, hist))
            left -= m
    if cfg.workers > 1:
        with ProcessPoolExecutor(cfg.workers) as pool:
            results = list(pool.map(_run_chunk, jobs, chunksize=4))
    else:
        results = []
        for i, j in enumerate(jobs, 1):
            results.append(_run_chunk(j))
            if progress and i % 10 == 0:
                progress(f"  {i}/{len(jobs)} chunks")
    ledger = dict.fromkeys(GENERATORS, 0)
    for j, r in zip(jobs, results, strict=True):
        if all(np.isfinite(r["terminal"])):          # a chunk with NaN/inf paths does not count
            ledger[j[0]] += j[1]
    metrics = {k: np.concatenate([r[k] for r in results]) for k in METRICS}
    gen = np.concatenate([np.full(len(r["terminal"]), GENERATORS.index(j[0]), dtype=np.int8)
                          for j, r in zip(jobs, results, strict=True)])
    return SimResult(cfg, metrics, gen, ledger, time.time() - t0)


def summarise(m: Mapping[str, np.ndarray]) -> dict[str, float]:
    t, dd = m["terminal"].astype(float), m["max_dd"].astype(float)
    var95, var99 = np.quantile(t, 0.05), np.quantile(t, 0.01)
    tail = t[t <= np.quantile(t, 0.025)]
    out = {"cagr_mean": float(np.mean(m["cagr"])), "cagr_p05": float(np.quantile(m["cagr"], 0.05)),
           "cagr_p25": float(np.quantile(m["cagr"], 0.25)), "cagr_median": float(np.median(m["cagr"])),
           "cagr_p75": float(np.quantile(m["cagr"], 0.75)), "cagr_p95": float(np.quantile(m["cagr"], 0.95)),
           "vol_median": float(np.median(m["vol"])), "sharpe_median": float(np.nanmedian(m["sharpe"])),
           "max_dd_median": float(np.median(dd)), "var95": float(-var95), "var99": float(-var99),
           "es975": float(-tail.mean()), "p_loss": float((t < 0).mean()),
           "p_underperform": float((m["excess"] < 0).mean()), "p_dd_20": float((dd > 0.20).mean()),
           "p_dd_30": float((dd > 0.30).mean()), "p_dd_50": float((dd > 0.50).mean())}
    return out


def batch_se(values: np.ndarray, stat: Callable[[np.ndarray], float], batches: int = 100) -> float:
    """Standard error of a statistic from the spread of its value across equal batches."""
    if len(values) < batches * 2:
        return math.nan
    parts = np.array_split(values, batches)
    est = np.array([stat(p) for p in parts])
    return float(est.std(ddof=1) / math.sqrt(batches))


def convergence(res: SimResult) -> pd.DataFrame:
    """Key statistics at 10k / 100k / 500k / all paths, with standard error and a converged flag
    (the last step moved by less than one standard error). Paths are shuffled so that every subset
    mixes the generators in their intended proportions."""
    rng = np.random.default_rng(0)
    order = rng.permutation(len(res.generator))
    stats: dict[str, tuple[str, Callable[[np.ndarray], float]]] = {
        "median CAGR": ("cagr", lambda x: float(np.median(x))),
        "P[loss]": ("terminal", lambda x: float((x < 0).mean())),
        "P[max DD > 30%]": ("max_dd", lambda x: float((x > 0.30).mean())),
        "P[underperform]": ("excess", lambda x: float((x < 0).mean())),
    }
    points = [p for p in CONVERGENCE_POINTS if p < len(order)] + [len(order)]
    rows = []
    for label, (key, fn) in stats.items():
        x = res.metrics[key].astype(float)[order]
        vals = [fn(x[:p]) for p in points]
        se = batch_se(x, fn)
        rows.append({"statistic": label, **{f"n={p:,}": v for p, v in zip(points, vals, strict=True)},
                     "std_error": se,
                     "converged": bool(len(vals) > 1 and se == se and abs(vals[-1] - vals[-2]) < se)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------
def render(res: SimResult, hist: History) -> str:
    cfg, s = res.config, summarise(res.metrics)
    pct = lambda x: f"{x * 100:6.1f}%"  # noqa: E731
    L = ["SIGMA VI simulation - EXPERIMENTAL", "=" * 78,
         f"Simulations completed: {res.completed:,} of {cfg.paths:,} requested "
         f"({res.seconds:,.0f} s, seed {cfg.seed}, {cfg.workers} worker(s))",
         "  " + ", ".join(f"{g} {n:,}" for g, n in res.ledger.items()),
         f"Horizon {cfg.years:g} years in {cfg.steps} steps of 1/{cfg.periods_per_year} year; "
         f"history = {len(hist.returns)} periods of strategy and benchmark returns.", ""]
    if res.completed < cfg.paths:
        L += [f"LIMITATION: only {res.completed:,} of the requested {cfg.paths:,} paths completed.", ""]
    if res.completed < 1_000_000:
        L += ["LIMITATION: fewer than 1,000,000 paths were run; this report does not claim otherwise.", ""]
    L += ["Annualised return over the horizon (net of drawn trading costs)",
          f"  mean {pct(s['cagr_mean'])}   median {pct(s['cagr_median'])}   5th {pct(s['cagr_p05'])}   "
          f"25th {pct(s['cagr_p25'])}   75th {pct(s['cagr_p75'])}   95th {pct(s['cagr_p95'])}",
          f"  volatility (median) {pct(s['vol_median'])}   Sharpe (median) {s['sharpe_median']:.2f}",
          "Risk",
          f"  median worst drawdown {pct(s['max_dd_median'])}   VaR 95% {pct(s['var95'])}   "
          f"VaR 99% {pct(s['var99'])}   expected shortfall 97.5% {pct(s['es975'])}",
          f"  P[loss over horizon] {pct(s['p_loss'])}   P[trail the benchmark] {pct(s['p_underperform'])}",
          f"  P[drawdown > 20%] {pct(s['p_dd_20'])}   > 30% {pct(s['p_dd_30'])}   > 50% {pct(s['p_dd_50'])}",
          "", "By generator (median annualised return / P[loss] / P[max DD > 30%])"]
    for i, g in enumerate(GENERATORS):
        idx = res.generator == i
        if idx.any():
            sub = {k: v[idx] for k, v in res.metrics.items()}
            loss, deep = float((sub["terminal"] < 0).mean()), float((sub["max_dd"] > 0.30).mean())
            L.append(f"  {g:<12} {pct(float(np.median(sub['cagr'])))}  {pct(loss)}  {pct(deep)}")
    conv = convergence(res)
    pts = [c for c in conv.columns if c.startswith("n=")]
    L += ["", "Convergence (stable when the last step moved less than its standard error)",
          f"  {'':<18} " + "  ".join(f"{c.replace('n=', ''):>9}" for c in pts) + "   std.err"]
    for _, r in conv.iterrows():
        L.append(f"  {r['statistic']:<18} " + "  ".join(f"{r[c]:9.4f}" for c in pts)
                 + f"   {r['std_error']:.4f}  {'converged' if r['converged'] else 'NOT converged'}")
    L += ["", "Reading this honestly",
          "  * The paths are drawn from the strategy's own history, so they show how wide the range of",
          "    outcomes is, not whether there is an edge. If the history was lucky, so are the paths.",
          "  * Not implemented yet: adversarial parameter search and a separate execution-delay model;",
          "    costs enter as one random drag per path. Correlation spikes appear only through the",
          "    student_t, regime and stress generators.",
          f"  * {len(hist.returns)} history periods is short; widen your error bars accordingly."]
    return "\n".join(L)


def load_history(path: str, use_holdout: bool = False) -> History:
    df = pd.read_csv(path, parse_dates=["date"])
    if not use_holdout:
        df = df[df["period"] != "holdout"]
    ppy = 12 if df["date"].diff().dt.days.median() < 45 else 4
    return History.from_frame(df["strategy"], df["benchmark"], ppy)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - reads the local project folder
    from sigma.core.config import DEFAULT_DOTENV

    ap = argparse.ArgumentParser(prog="python -m sigma.simulate")
    ap.add_argument("--paths", type=int, default=1_000_000)
    ap.add_argument("--years", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--history", default=str(DEFAULT_DOTENV.parent / "backtest_returns.csv"))
    args = ap.parse_args(argv)
    hist = load_history(args.history)
    cfg = Config(paths=args.paths, years=args.years, periods_per_year=hist.periods_per_year,
                 seed=args.seed, workers=args.workers)
    sys.stdout.write(f"[{datetime.now():%H:%M:%S}] simulating {cfg.paths:,} paths on {cfg.workers} workers\n")
    res = run(hist, cfg, progress=lambda m: sys.stdout.write(m + "\n"))
    text = render(res, hist)
    (DEFAULT_DOTENV.parent / "simulation_report.txt").write_text(text + "\n", encoding="utf-8")
    sys.stdout.write("\n" + text + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
