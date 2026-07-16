"""
store/po_store.py
=================
In-memory backing store for PO / contract reference data.

Data is loaded once from ``data/pos_contracts.json`` on first access and
cached for the lifetime of the process.  All public access goes through
:func:`get_po`.

Schema of each record in pos_contracts.json::

    {
      "<PO_NUMBER>": {
        "vendor":                  str,
        "contract_id":             str,
        "line_qty":                int,
        "unit_price":              float,
        "expected_total":          float,
        "approval_required_over":  float,
        "approved_by":             str | null
      }
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from project_paths import find_project_root

# ── resolve data file relative to the repository root (works regardless of CWD) ────────
_DATA_FILE = find_project_root(__file__) / "data" / "pos_contracts.json"

# Module-level cache — loaded once, never mutated at runtime
_PO_CACHE: Optional[dict[str, dict]] = None


def _load() -> dict[str, dict]:
    """Load pos_contracts.json into the module-level cache."""
    global _PO_CACHE
    if _PO_CACHE is None:
        with open(_DATA_FILE, encoding="utf-8") as fh:
            _PO_CACHE = json.load(fh)
    return _PO_CACHE


def get_po(po_number: str) -> Optional[dict]:
    """
    Return the PO/contract record for *po_number*, or ``None`` if not found.

    Parameters
    ----------
    po_number : str
        The PO number to look up (e.g. ``"PO-5567"``).  Leading/trailing
        whitespace is stripped before the lookup.

    Returns
    -------
    dict or None
        A copy of the matching record dict, or ``None`` if the PO number is
        not present in the store.  Returning a copy prevents callers from
        accidentally mutating the cache.
    """
    store = _load()
    record = store.get(po_number.strip())
    return dict(record) if record is not None else None


def list_po_numbers() -> list[str]:
    """Return all PO numbers currently in the store (useful for tests)."""
    return list(_load().keys())


def get_all_pos() -> dict[str, dict]:
    """
    Return a shallow copy of the entire PO store as a dict keyed by PO number.

    Safe for display — mutations to the returned dict do not affect the cache.
    """
    return {k: dict(v) for k, v in _load().items()}


def add_po(po_number: str, record: dict) -> None:
    """
    Add or replace a PO/contract record in the in-memory store and persist it
    back to ``data/pos_contracts.json``.

    Parameters
    ----------
    po_number : str
        PO number key (e.g. ``"PO-9001"``).
    record : dict
        Record dict with keys: vendor, contract_id, line_qty, unit_price,
        expected_total, approval_required_over, approved_by.

    Raises
    ------
    ValueError
        If any required key is missing from *record*.
    """
    required = {"vendor", "contract_id", "line_qty", "unit_price",
                "expected_total", "approval_required_over"}
    missing = required - record.keys()
    if missing:
        raise ValueError(f"add_po: missing required fields: {sorted(missing)}")

    store = _load()
    store[po_number.strip()] = dict(record)
    _persist(store)


def delete_po(po_number: str) -> bool:
    """
    Remove a PO record from the in-memory store and persist the change.

    Returns
    -------
    bool
        ``True`` if the record was found and removed, ``False`` if not found.
    """
    store = _load()
    key = po_number.strip()
    if key not in store:
        return False
    del store[key]
    _persist(store)
    return True


def _persist(store: dict[str, dict]) -> None:
    """Write the current cache state back to ``pos_contracts.json``."""
    with open(_DATA_FILE, "w", encoding="utf-8") as fh:
        json.dump(store, fh, indent=2)
