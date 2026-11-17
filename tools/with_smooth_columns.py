"""Run any tools module with SMOOTH column outlines instead of the staircase ones, without editing the pipeline.
The shipped readout upsamples the 256-px label map to the frame with nearest neighbour (~7.5 px steps on a
1920x1080 frame). Here each column's 256-px mask is blurred (Gaussian, sigma SIGMA px at 256) and upsampled
bilinearly, then cut at 0.5: same shape, no steps.

    python tools/with_smooth_columns.py <sigma> tools.evaluate --models <dir> --vote 5
"""
import runpy
import sys

import cv2
import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "inference")
import sheetcount.pipeline as P  # noqa: E402

SIGMA = float(sys.argv[1])


def segment_columns(self, frame, typ="type3"):
    H, W = frame.shape[:2]
    imgL, s, px, py = P.letterbox(frame, P.COLUMN_SIZE)
    o = self.col.run(None, {"input": P._to_input(imgL)})[0][0]
    prob = P._sigmoid(o[0])
    lab = P.follow_flows(prob > 0.5, o[1:3])
    pc = P._sigmoid(o[3]) if o.shape[0] >= 4 else None
    pcN = P._unletter(pc, s, px, py, (H, W), cv2.INTER_LINEAR) if pc is not None else None
    ids = [i for i in range(1, int(lab.max()) + 1) if (lab == i).sum() >= P.COLUMN_MIN_AREA]
    cols = []
    for i in ids:
        soft = (lab == i).astype(np.float32)
        if SIGMA > 0:
            soft = cv2.GaussianBlur(soft, (0, 0), SIGMA)
        m = P._unletter(soft, s, px, py, (H, W), cv2.INTER_LINEAR) > 0.5
        if not m.any():
            continue
        p = float(pcN[m].mean()) if pcN is not None else 0.0
        cols.append(P.Column(mask=m, polygon=P._polygon(m), cropped=p > 0.5, p_cropped=p))
    if typ == "type10" and cols:
        yb = [np.where(c.mask.any(axis=1))[0] for c in cols]
        top, bot = min(y.min() for y in yb), max(y.max() for y in yb)
        for c, y in zip(cols, yb):
            c.cropped = bool(y.min() <= top + P.GEO_TOL or y.max() >= bot - P.GEO_TOL)
    cols.sort(key=lambda c: float(np.where(c.mask)[1].mean()))
    return cols


P.SheetCounter.segment_columns = segment_columns
mod = sys.argv[2]
sys.argv = [mod] + sys.argv[3:]
runpy.run_module(mod, run_name="__main__", alter_sys=True)
