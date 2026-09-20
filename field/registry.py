"""Which checkpoint is live, and how to put the previous one back.

    runs/_active.json
      {"edges":   {"active": "edges_all3_s7", "previous": null, "history": [...]},
       "columns": {"active": "columns_v1",    "previous": null, "history": [...]},
       "sites":   {"acme": {"edges": "edges_site_acme_20260920"}}}

A site entry overrides the global one for that site only; it never changes what
anybody else runs.
"""
from __future__ import annotations

import datetime as _dt
import json
import os

import paths

PATH = os.path.join(paths.RUNS, "_active.json")
DEFAULTS = {"edges": "edges_all3_s7", "columns": "columns_v1"}


def load() -> dict:
    if os.path.exists(PATH):
        return json.load(open(PATH))
    return {k: {"active": v, "previous": None, "history": []} for k, v in DEFAULTS.items()} | {"sites": {}}


def save(reg: dict) -> None:
    os.makedirs(paths.RUNS, exist_ok=True)
    json.dump(reg, open(PATH, "w"), indent=1)


def active(track: str, site: str | None = None) -> str:
    reg = load()
    if site and reg.get("sites", {}).get(site, {}).get(track):
        return reg["sites"][site][track]
    return reg[track]["active"]


def checkpoint(track: str, site: str | None = None) -> str:
    run = active(track, site)
    p = os.path.join(paths.RUNS, run, "last.pt")
    return p if os.path.exists(p) else os.path.join(paths.RUNS, run, "stage2", "last.pt")


def promote(track: str, run: str, note: str = "") -> None:
    """Make `run` the global active model, keeping the old one for rollback."""
    reg = load()
    prev = reg[track]["active"]
    reg[track] = {"active": run, "previous": prev,
                  "history": reg[track]["history"] + [
                      {"run": run, "replaced": prev, "at": _dt.datetime.now().isoformat(timespec="seconds"),
                       "note": note}]}
    save(reg)


def promote_site(track: str, site: str, run: str, note: str = "") -> None:
    reg = load()
    sites = reg.setdefault("sites", {})
    entry = sites.setdefault(site, {})
    entry["previous_" + track] = entry.get(track)
    entry[track] = run
    entry.setdefault("history", []).append(
        {"track": track, "run": run, "at": _dt.datetime.now().isoformat(timespec="seconds"), "note": note})
    save(reg)


def rollback(track: str, site: str | None = None) -> str:
    reg = load()
    if site:
        e = reg.get("sites", {}).get(site, {})
        prev = e.get("previous_" + track)
        if not prev:
            raise SystemExit(f"no previous {track} model for site {site}")
        e[track] = prev
        save(reg)
        return prev
    prev = reg[track]["previous"]
    if not prev:
        raise SystemExit(f"no previous {track} model to roll back to")
    reg[track] = {"active": prev, "previous": reg[track]["active"], "history": reg[track]["history"]}
    save(reg)
    return prev


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "rollback":
        track = sys.argv[2] if len(sys.argv) > 2 else "edges"
        site = sys.argv[3] if len(sys.argv) > 3 else None
        print("rolled back to", rollback(track, site))
    else:
        print(json.dumps(load(), indent=1))
