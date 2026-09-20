"""Edge-model readouts: per-sheet instances (flow) and the shipped COUNT rule (band).

COUNT = connected components of the model's FOREGROUND inside the central 50 %
of the column width. It never reads the flow field. Measured on type3 human
labels: 18/18 exact, 292/300 production crops exact, and a 2x wider threshold
margin plus 100 % under blur/gain/noise stress -- the flow-instance count failed
the brightness stress (89 %) and is replicate-sensitive. The band cuts off the
sheet tips, which is where a high threshold splits one sheet into two.
Instances (flow readout) are for MASKS, not for counting.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

from common.flows import follow_flows
from common.imageio import pad_to_multiple, to_tensor

BAND_FRAC, THR, MIN_AREA = 0.50, 0.50, 6
# A blob this much smaller than the crop's MEDIAN blob is a fragment (a sheet broken
# by glare, a speck on the stack cap), not a sheet. Measured on every labelled crop
# and on 394 unseen type3 crops: 6 px floor only -> 120/126 and 282/300 exact;
# with this floor -> 123-124/126 and 290-296/300. It breaks 1-4 crops per set where a
# genuinely small partial sheet is dropped, and fixes far more than it breaks.
REL_FLOOR = 0.25


@torch.no_grad()
def infer(net, img: np.ndarray, device: str):
    """Fully convolutional on the whole crop. Returns (fg prob HxW, flow 2xHxW)."""
    padded, h, w = pad_to_multiple(img, 16)
    o = net(to_tensor(padded)[None].to(device))[0].cpu().numpy()[:, :h, :w]
    return 1 / (1 + np.exp(-o[0])), o[1:3]


def band(fg: np.ndarray, frac: float = BAND_FRAC) -> tuple[int, int]:
    xs = np.where(fg.any(axis=0))[0]
    if xs.size == 0:
        return 0, fg.shape[1]
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    cx, half = 0.5 * (x0 + x1), 0.5 * (x1 - x0) * frac
    return max(x0, int(round(cx - half))), min(x1, int(round(cx + half)))


def band_count(fg: np.ndarray, frac: float = BAND_FRAC, min_area: int = MIN_AREA,
               rel_floor: float = REL_FLOOR) -> int:
    a, b = band(fg, frac)
    sub = fg[:, a:b].astype(np.uint8)
    if sub.size == 0:
        return 0
    n, cc = cv2.connectedComponents(sub, connectivity=8)
    if n <= 1:
        return 0
    areas = np.bincount(cc.ravel())[1:]
    areas = areas[areas >= min_area]
    if rel_floor and areas.size:
        areas = areas[areas >= rel_floor * np.median(areas)]
    return int(areas.size)


def read(net, img: np.ndarray, device: str, masks: str = "grow") -> dict:
    """masks="grow" (default): instances grown outward from the counted centre blobs --
    zero identity leak, one mask per sheet on 1929/1930 labelled sheets, and the
    instance count equals `count` by construction. masks="flow": the raw field
    readout, kept for comparison (leak 5-71 px/instance, see tools/measure_leak.py)."""
    from edges.grow import grow_instances
    prob, flow = infer(net, img, device)
    fg = prob > THR
    inst = grow_instances(fg, rel_floor=REL_FLOOR)[0] if masks == "grow" else follow_flows(fg, flow)
    return {"fg": fg, "instances": inst, "count": band_count(fg)}
