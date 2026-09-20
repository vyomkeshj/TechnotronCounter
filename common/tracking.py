"""Experiment tracking: every write goes to TensorBoard AND the ExternalSoul monitor.

Mirroring, not replacing: runs/<name>/ keeps a complete local TB record, so "did
the mirror drop anything" stays answerable. esoul auth is resolved by the SDK
itself (explicit kwarg -> ESOUL_TOKEN -> ~/.config/esoul/credentials). NEVER put a
token in this repo and never re-add ESOUL_TOKEN to the environment (it would
shadow a rotated credentials file). No credentials -> TensorBoard only.

Telemetry can never take a run down: every esoul call is guarded.
"""
from __future__ import annotations

import sys
from typing import Any, Dict, Optional

import numpy as np
from torch.utils.tensorboard import SummaryWriter

for _s in (sys.stdout, sys.stderr):          # the SDK banner is non-ASCII; cp1252 consoles
    try:                                      # raise inside track.init() and silently drop the run
        if _s is not None and getattr(_s, "encoding", "").lower() not in ("utf-8", "utf8"):
            _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _have_credentials() -> bool:
    try:
        from esoul._auth import resolve_credentials
        resolve_credentials()
        return True
    except Exception:
        return False


def _hwc(img) -> np.ndarray:
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[0] in (1, 3, 4) and a.shape[2] not in (1, 3, 4):
        a = a.transpose(1, 2, 0)
    if a.dtype != np.uint8:
        a = np.clip(a * 255 if a.max() <= 1.0 else a, 0, 255).astype(np.uint8)
    return a


class Tracker:
    """add_scalar / add_image / log_audit / close -> TB + esoul."""

    def __init__(self, out_dir: str, run_name: str, monitor: str, group: str,
                 config: Optional[dict] = None, total_steps: Optional[int] = None,
                 hero: tuple[str, ...] = (), panels: tuple = ()):
        self.tb = SummaryWriter(out_dir)
        self.run = None
        self._pending: Dict[str, Any] = {}
        self._step: Optional[int] = None
        if not _have_credentials():
            print("[tracking] no esoul credentials -> TensorBoard only")
            return
        try:
            from esoul import track
            from esoul.track import Dashboard, Images, Line, System
            dash = Dashboard(title=run_name, hero=list(hero),
                             panels=list(panels) or [Line("Loss", ["loss/*"], span=1),
                                                     Line("Validation", ["val/*"], span=1),
                                                     Images("preview/val", title="Previews", per_row=1, span=2),
                                                     System(span=1)])
            self.run = track.init(monitor=monitor, name=run_name, group=group, config=config or {},
                                  total_steps=total_steps, dashboard=dash, quiet=True)
            print(f"[tracking] esoul monitor {monitor!r} group {group!r} run {run_name!r}")
        except Exception as e:
            print(f"[tracking] esoul init failed ({type(e).__name__}: {e}) -> TensorBoard only")
            self.run = None

    def add_scalar(self, tag: str, value, step: int) -> None:
        self.tb.add_scalar(tag, value, step)
        if self.run is None:
            return
        if self._step is not None and step != self._step:
            self._flush()
        self._step = step
        self._pending[tag] = float(value)

    def add_image(self, tag: str, img_chw, step: int) -> None:
        self.tb.add_image(tag, img_chw, step)
        if self.run is None:
            return
        if self._step == step:
            self._flush()
        try:
            self.run.log_images(tag, [_hwc(img_chw)], step=step, captions=[tag])
        except Exception as e:
            print(f"[tracking] log_images failed: {e}")

    def log_audit(self, metrics: Dict[str, float], step: int, note: str = "") -> None:
        """Attach the honest held-out numbers to THIS run, in-process (a later
        process resuming by id once attached the audit to the wrong run)."""
        for k, v in metrics.items():
            self.tb.add_scalar(k, v, step)
        self._flush()
        if self.run is not None:
            try:
                self.run.log(dict(metrics), step=step)
                if note:
                    self.run.note(note)
            except Exception as e:
                print(f"[tracking] log_audit failed: {e}")

    def _flush(self) -> None:
        if self.run is not None and self._pending:
            try:
                self.run.log(self._pending, step=self._step)
            except Exception as e:
                print(f"[tracking] log failed: {e}")
        self._pending = {}

    def close(self, status: str = "finished") -> None:
        self._flush()
        self.tb.close()
        if self.run is not None:
            try:
                self.run.finish(status=status)
            except Exception:
                pass
