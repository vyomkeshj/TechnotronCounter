"""All locations, relative to this folder. Nothing in the pipeline hard-codes a path."""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
RUNS = os.path.join(ROOT, "runs")

COLUMNS_DATA = os.path.join(DATA, "columns")
COLUMNS_SPLIT = os.path.join(COLUMNS_DATA, "splits.json")
LEGACY_FG = os.path.join(COLUMNS_DATA, "legacy_fg")

EDGES_DATA = os.path.join(DATA, "edges")
EDGES_SPLIT = os.path.join(EDGES_DATA, "splits.json")
