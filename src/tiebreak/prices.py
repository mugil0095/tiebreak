from __future__ import annotations

import os
from dataclasses import dataclass

# Prices that are not in LiteLLM's table. Source: https://docs.typesafe.ai/models.md (checked 2026-10-18).
# Jev is priced per input token only; output tokens are free.
_BUILTIN = {
    "jev-1.14.0": (0.042, 0.0),
    "jev-1.13.0": (0.042, 0.0),
}
BUILTIN_SOURCE = "typesafe docs"
TABLE_SOURCE = "litellm table"


@dataclass(frozen=True)
class PriceRow:
    model: str
    input_per_mtok: float
    output_per_mtok: float
    source: str


def _litellm_table() -> dict:
    """LiteLLM's bundled price table (no network access), or {} when litellm is not installed."""
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    try:
        import litellm
        table = litellm.model_cost
    except Exception:
        return {}
    return table if isinstance(table, dict) else {}


def _row(model: str, entry) -> PriceRow | None:
    if not isinstance(entry, dict):
        return None
    tin, tout = entry.get("input_cost_per_token"), entry.get("output_cost_per_token")
    if isinstance(tin, bool) or isinstance(tout, bool) or not isinstance(tin, (int, float)) or not isinstance(tout, (int, float)):
        return None
    return PriceRow(model, float(tin) * 1e6, float(tout) * 1e6, TABLE_SOURCE)


def lookup_price(provider: str | None, model: str) -> PriceRow | None:
    """Price of one model: built-in entries first, then LiteLLM's table (provider/model key, then model)."""
    if model in _BUILTIN:
        tin, tout = _BUILTIN[model]
        return PriceRow(model, tin, tout, BUILTIN_SOURCE)
    table = _litellm_table()
    keys = ([f"{provider}/{model}"] if provider else []) + [model]
    for key in keys:
        row = _row(key, table.get(key))
        if row is not None:
            return row
    return None


def search_prices(query: str = "", limit: int = 25) -> list[PriceRow]:
    """Models whose id contains `query` (case-insensitive), sorted by id, at most `limit`."""
    q = query.lower()
    rows = [PriceRow(m, tin, tout, BUILTIN_SOURCE) for m, (tin, tout) in _BUILTIN.items()]
    rows += [r for key, entry in _litellm_table().items() if (r := _row(key, entry)) is not None]
    return sorted((r for r in rows if q in r.model.lower()), key=lambda r: r.model)[:limit]
