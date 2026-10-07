"""Logical model roles and the model registry (spec §4).

No other module may contain a model id. Everything goes through :class:`ModelRegistry`.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "config" / "models.toml"


class Role(StrEnum):
    FAST = "FAST"
    DATA = "DATA"
    RESEARCH = "RESEARCH"
    REASONING = "REASONING"
    CRITIC = "CRITIC"
    JUDGE = "JUDGE"
    CODING = "CODING"
    FALLBACK = "FALLBACK"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    id: str
    provider: str
    context_window: int
    input_per_mtok: float
    output_per_mtok: float
    price_verified: bool

    def cost_usd(self, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
        # Cached input is billed differently by each provider; conservatively bill it as input.
        billable_in = input_tokens + cached_tokens
        return (billable_in * self.input_per_mtok + output_tokens * self.output_per_mtok) / 1e6


class ModelRegistry:
    def __init__(self, models: dict[str, ModelSpec], chains: dict[Role, list[str]]):
        unknown = {k for chain in chains.values() for k in chain} - models.keys()
        if unknown:
            raise ValueError(f"role chains reference unknown models: {sorted(unknown)}")
        missing_roles = set(Role) - chains.keys()
        if missing_roles:
            raise ValueError(f"no chain configured for roles: {sorted(r.value for r in missing_roles)}")
        self.models = models
        self.chains = chains

    @classmethod
    def from_toml(cls, path: Path | str = DEFAULT_CONFIG) -> ModelRegistry:
        with open(path, "rb") as fh:
            raw = tomllib.load(fh)
        models = {
            key: ModelSpec(
                key=key,
                id=m["id"],
                provider=m["provider"],
                context_window=int(m["context_window"]),
                input_per_mtok=float(m.get("input_per_mtok", 0.0)),
                output_per_mtok=float(m.get("output_per_mtok", 0.0)),
                price_verified=bool(m.get("price_verified", False)),
            )
            for key, m in raw["models"].items()
        }
        chains = {Role(name): list(keys) for name, keys in raw["roles"].items()}
        return cls(models, chains)

    def chain(self, role: Role) -> list[ModelSpec]:
        return [self.models[k] for k in self.chains[role]]
