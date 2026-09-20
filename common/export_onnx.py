"""Export any pipeline checkpoint to ONNX (dynamic H/W) and check parity.

    python -m common.export_onnx runs/columns_v1/stage2/last.pt column_flow.onnx
    python -m common.export_onnx runs/edges_v1/last.pt sheet_model.onnx

Input: float32 N x 3 x H x W, RGB, (x/255 - 0.45) / 0.23; H and W divisible by 16.
Columns are letterboxed to 256; edge crops run at native size (pad to /16).
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from common.unet import load_checkpoint


def main():
    ck, out = sys.argv[1], sys.argv[2]
    net = load_checkpoint(ck, "cpu")
    x = torch.randn(1, 3, 256, 128)
    torch.onnx.export(net, x, out, input_names=["input"], output_names=["output"], opset_version=17,
                      dynamic_axes={"input": {0: "n", 2: "h", 3: "w"}, "output": {0: "n", 2: "h", 3: "w"}})
    import onnxruntime as ort
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    for shape in ((1, 3, 256, 256), (1, 3, 640, 96)):
        t = torch.randn(*shape)
        with torch.no_grad():
            ref = net(t).numpy()
        got = sess.run(None, {"input": t.numpy()})[0]
        print(f"{shape}: max |torch - onnx| = {np.abs(ref - got).max():.2e}")
    print("wrote", out)


if __name__ == "__main__":
    main()
