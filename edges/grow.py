"""Per-sheet masks by GROWING OUTWARD from the counted centre blobs.

Why not the flow readout: near the strip's edges neighbouring sheets compress to a
few px, the predicted field is smooth, and tail pixels drift into the neighbour's
basin -- one sheet's tail gets the neighbour's identity (measured: type2 1251
contact px on a val crop). That is produced by the field at inference; training on
verified-clean labels reproduced it, so it cannot be relabelled away.

This readout never asks the field who owns a pixel:

  seeds   connected components of the foreground inside the centre band -- the
          same evidence the shipped counter uses, so the instance count EQUALS
          the count by construction (no merges, no extra instances, ever)
  grow    walk one pixel-column at a time toward each edge; in each column the
          foreground forms vertical runs, and a run inherits the id of the run it
          overlaps in the previous column
  split   when two sheets' runs merge into one run, pixels go to the nearer of the
          two incoming ids (midpoint split) instead of one swallowing the other
  stop    a run that overlaps no previous id stays unassigned (a detached fragment
          that never reaches the band is not a sheet of this column)

Worst case is therefore a boundary a few px off, never a swapped identity.
"""
from __future__ import annotations

import cv2
import numpy as np

from edges.counter import MIN_AREA, band


def seeds_in_band(fg: np.ndarray, frac: float = 0.50, min_area: int = MIN_AREA,
                  rel_floor: float = 0.0) -> tuple[np.ndarray, int, tuple[int, int]]:
    """Label map holding only the band seeds (1..N, ordered top-down)."""
    a, b = band(fg, frac)
    n, cc = cv2.connectedComponents(fg[:, a:b].astype(np.uint8), connectivity=8)
    areas = np.bincount(cc.ravel())[1:] if n > 1 else np.zeros(0, int)
    keep = [i for i in range(1, n) if areas[i - 1] >= min_area]
    if rel_floor and keep:
        med = np.median([areas[i - 1] for i in keep])
        keep = [i for i in keep if areas[i - 1] >= rel_floor * med]
    out = np.zeros(fg.shape, np.int32)
    order = sorted(keep, key=lambda i: float(np.where(cc == i)[0].mean()))
    for new_id, i in enumerate(order, 1):
        out[:, a:b][cc == i] = new_id
    return out, len(order), (a, b)


def _runs(col: np.ndarray) -> list[tuple[int, int]]:
    """[(y0, y1)] of contiguous True runs in a boolean column."""
    idx = np.flatnonzero(np.diff(np.r_[0, col.view(np.int8), 0]))
    return list(zip(idx[::2], idx[1::2]))


def _propagate(lab: np.ndarray, fg: np.ndarray, xs: range, step: int) -> None:
    for x in xs:
        prev = lab[:, x - step]
        if not prev.any():
            continue
        for y0, y1 in _runs(fg[:, x]):
            seg = prev[y0:y1]
            ids = [i for i in np.unique(seg) if i]
            if not ids:
                continue
            if len(ids) == 1:
                lab[y0:y1, x] = ids[0]
                continue
            centres = {i: float(np.where(seg == i)[0].mean()) + y0 for i in ids}   # midpoint split
            ys = np.arange(y0, y1, dtype=np.float32)
            keys = np.array(list(centres))
            d = np.abs(ys[:, None] - np.array([centres[k] for k in keys])[None, :])
            lab[y0:y1, x] = keys[d.argmin(axis=1)]


def grow_instances(fg: np.ndarray, frac: float = 0.50, min_area: int = MIN_AREA,
                   rel_floor: float = 0.0) -> tuple[np.ndarray, int]:
    """(instance labels over the whole strip, count). Count == centre-band count."""
    lab, n, (a, b) = seeds_in_band(fg, frac, min_area, rel_floor)
    if n == 0:
        return lab, 0
    _propagate(lab, fg, range(b, fg.shape[1]), +1)          # rightwards
    _propagate(lab, fg, range(a - 1, -1, -1), -1)           # leftwards
    return lab, n
