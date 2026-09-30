"""Cellpose-style flow representation shared by both pipelines.

Target : per foreground pixel, a unit vector pointing at its instance's centroid.
Readout: advect every foreground pixel along the predicted field (40 steps,
         sub-pixel bilinear), bin the landing points, and cluster.

Why iterative unit flows (measured, column campaign): a one-shot centre-offset
head scored instIoU 0.43 vs 0.82 -- raw offsets need ~pixel precision when
instance centres are 10-20 px apart, while each advection step only needs the
local DIRECTION right and errors self-correct over 40 steps.
"""
from __future__ import annotations

import cv2
import numpy as np


def flow_targets(lab: np.ndarray, aniso: bool = False) -> np.ndarray:
    """(2, H, W) float32 target (dy, dx). 0 outside instances.

    aniso=True divides each offset by the instance's own half-extent before
    normalising. Use it for SHEETS (3.2x wider than tall: an isotropic target
    spends 74 % of its range on the uninformative across-width axis; measured
    MAE 2.26 -> 1.90). Columns use the isotropic target.
    """
    fl = np.zeros((2, *lab.shape), np.float32)
    for i in np.unique(lab):
        if i == 0:
            continue
        ys, xs = np.where(lab == i)
        cy, cx = ys.mean(), xs.mean()
        dy, dx = cy - ys, cx - xs
        if aniso:
            dy = dy / max((ys.max() - ys.min() + 1) / 2.0, 1.0)
            dx = dx / max((xs.max() - xs.min() + 1) / 2.0, 1.0)
        n = np.sqrt(dy * dy + dx * dx) + 1e-6
        fl[0, ys, xs] = dy / n
        fl[1, ys, xs] = dx / n
    return fl


def follow_flows(fg: np.ndarray, flow: np.ndarray, n_iter: int = 40, bin_px: int = 3,
                 min_area: int = 25, min_hits: int = 3) -> np.ndarray:
    """Instance labels (uint8/int32, 1..N) from a fg mask and a (2,H,W) flow field.

    Two safeguards make "tiny bridge" fusions structurally impossible (field
    report 2026-07-21): bins need >= min_hits landings, and every cluster is
    re-split at 25 % of its OWN peak landing density -- a real instance has one
    density core, a bridged pair has two joined by a sparse trail.
    """
    ys, xs = np.where(fg)
    H, W = fg.shape
    if len(ys) == 0:
        return np.zeros((H, W), np.int32)
    py, px = ys.astype(np.float32), xs.astype(np.float32)

    def samp(ch, yy, xx):
        y0 = np.clip(np.floor(yy).astype(np.int32), 0, H - 2)
        x0 = np.clip(np.floor(xx).astype(np.int32), 0, W - 2)
        wy, wx = yy - y0, xx - x0
        return (ch[y0, x0] * (1 - wy) * (1 - wx) + ch[y0 + 1, x0] * wy * (1 - wx)
                + ch[y0, x0 + 1] * (1 - wy) * wx + ch[y0 + 1, x0 + 1] * wy * wx)

    if H >= 2 and W >= 2:
        for _ in range(n_iter):
            py = np.clip(py + samp(flow[0], py, px), 0, H - 1)
            px = np.clip(px + samp(flow[1], py, px), 0, W - 1)
    by, bx = (py / bin_px).astype(np.int32), (px / bin_px).astype(np.int32)
    gh, gw = H // bin_px + 2, W // bin_px + 2
    cnt = np.zeros((gh, gw), np.int32)
    np.add.at(cnt, (by, bx), 1)
    occ = (cnt >= min_hits).astype(np.uint8)
    if occ.sum() == 0:
        occ = (cnt > 0).astype(np.uint8)
    n0, cc0 = cv2.connectedComponents(occ, connectivity=8)
    core = np.zeros_like(occ)
    for i in range(1, n0):
        m = cc0 == i
        core[m & (cnt >= max(min_hits, int(0.25 * cnt[m].max())))] = 1
    ncc, cc = cv2.connectedComponents(core, connectivity=8)
    cid = cc[by, bx]
    if (cid == 0).any() and ncc > 1:
        _, lblidx = cv2.distanceTransformWithLabels((core == 0).astype(np.uint8), cv2.DIST_L2, 3,
                                                    labelType=cv2.DIST_LABEL_PIXEL)
        core_flat = np.flatnonzero(core.ravel())
        idx_map = np.zeros(gh * gw + 1, np.int32)
        idx_map[lblidx.ravel()[core_flat]] = cc.ravel()[core_flat]
        orph = cid == 0
        cid[orph] = idx_map[lblidx[by[orph], bx[orph]]]
    lab = np.zeros((H, W), np.int32)
    lab[ys, xs] = cid
    out = np.zeros((H, W), np.int32)
    k = 0
    for i in range(1, ncc):
        m = lab == i
        if m.sum() >= min_area:
            k += 1
            out[m] = k
    return out
