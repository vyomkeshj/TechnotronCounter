"""Run any tools module with a different END_FLOOR (the first/last band blob must reach this fraction of the
crop's median blob to count), without editing the pipeline:

    python tools/with_floor.py 0.75 tools.evaluate --models <dir> --vote 5
"""
import runpy
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "inference")
import sheetcount.pipeline as P  # noqa: E402

P.END_FLOOR = float(sys.argv[1])
mod = sys.argv[2]
sys.argv = [mod] + sys.argv[3:]
runpy.run_module(mod, run_name="__main__", alter_sys=True)
