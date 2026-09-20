"""Frame-level, leak-aware train/val assignment shared by both pipelines.

THE RULES (each one was learned by getting it wrong once):
  1. Split by SOURCE FRAME, never by crop/column: one photo yields 7-15 columns
     sharing lighting and geometry (a crop-level split leaked 49 % of an audit set).
  2. The rack camera is FIXED: neighbouring frames of a capture session are
     near-identical (thumbnail NCC 0.98-1.00). So whole capture SESSIONS go to val.
  3. A type with a single session gets a contiguous frame block as val, and the
     frames immediately around the block are EXCLUDED (buffer) from both sides.
  4. Existing assignments are PINNED and never re-rolled; only new frames are
     assigned. A new frame of an already-assigned session inherits the split of
     its nearest already-assigned frame (so a val session stays val).
  5. Val sessions are drawn per PRODUCT VARIANT (type + '-1_30pieces' etc.): ~20 %
     and at least one per variant with >= 2 sessions, never all of them. A variant
     with a single session trains. Sessions already held out by the other pipeline
     are preferred, so a frame is never train for one model and val for the other.

Frame names:  YYYYMMDD_HHMMSS_type<N>_..._<idx>   -> session = YYYYMMDD_HHMMSS_type<N>
Anything else (e.g. close-up photos) is treated as an independent photo.
"""
from __future__ import annotations

import hashlib
import re

SESSION_RE = re.compile(r"^(\d{8}_\d{6}_type\d+)_.*?_(\d+)$")
VARIANT_RE = re.compile(r"_type\d+_(-?\d+_\d+pieces)_")
VAL_EVERY = 5               # ~20 % of new sessions / independent photos go to val


def session_of(frame: str) -> str | None:
    m = SESSION_RE.match(frame)
    return m.group(1) if m else None


def index_of(frame: str) -> int:
    m = SESSION_RE.match(frame)
    return int(m.group(2)) if m else -1


def type_of(frame: str) -> str | None:
    s = session_of(frame)
    return s.split("_")[2] if s else None


def stratum_of(frame: str) -> str | None:
    """type + product variant, e.g. 'type2|-1_30pieces'. Sessions of different
    variants look different (type2 30- vs 50-piece stacks), so val is drawn per
    variant: holding out ALL sessions of a variant would leave it untrained."""
    t, m = type_of(frame), VARIANT_RE.search(frame)
    return f"{t}|{m.group(1) if m else ''}" if t else None


def val_sessions(split_path: str) -> set[str]:
    """Capture sessions held out in another pipeline's pinned split (empty if none)."""
    import json
    import os
    if not os.path.exists(split_path):
        return set()
    return {s for s in (session_of(f) for f in json.load(open(split_path)).get("val", [])) if s}


def _hash_val(key: str) -> bool:
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % VAL_EVERY == 0


def assign(frames: list[str], split: dict, prefer_val_sessions: set[str] = frozenset()) -> tuple[dict, list[str]]:
    """Extend a pinned split with any frames it has not seen yet.

    split = {"val": [...], "excluded": [...], "train": [...]}; all three are
    pinned. prefer_val_sessions: sessions another pipeline already holds out --
    chosen first, so no frame is train for one model and val for the other.
    Returns (updated split, human-readable decisions for new frames).
    """
    val, exc, train = (set(split.get(k, [])) for k in ("val", "excluded", "train"))
    known = val | exc | train
    new = sorted(set(frames) - known)
    notes: list[str] = []
    by_sess: dict[str, list[str]] = {}
    for f in new:
        s = session_of(f)
        if s is None:                                      # independent photo
            (val if _hash_val(f) else train).add(f)
            notes.append(f"{f}: independent photo -> {'val' if f in val else 'train'}")
        else:
            by_sess.setdefault(s, []).append(f)

    fresh_by_type: dict[str, list[str]] = {}
    for s, fs in sorted(by_sess.items()):
        old = [f for f in known if session_of(f) == s]
        if old:                                            # rule 4: inherit from nearest frame
            for f in fs:
                near = min(old, key=lambda o: abs(index_of(o) - index_of(f)))
                dst = val if near in val else exc if near in exc else train
                dst.add(f)
                notes.append(f"{f}: session already split, nearest {near} -> "
                             f"{'val' if dst is val else 'excluded' if dst is exc else 'train'}")
        else:
            fresh_by_type.setdefault(type_of(fs[0]), []).append(s)

    for typ, sessions in sorted(fresh_by_type.items()):
        all_sessions_of_type = {session_of(f) for f in known if type_of(f) == typ} | set(sessions)
        if len(all_sessions_of_type) == 1:                 # rule 3: single-session type
            s = sessions[0]
            fs = sorted(by_sess[s], key=index_of)
            k = max(1, round(0.2 * len(fs)))
            start = max(1, int(0.4 * len(fs)))
            for j, f in enumerate(fs):
                (val if start <= j < start + k else exc if j in (start - 1, start + k) else train).add(f)
            notes.append(f"session {s}: ONLY session of {typ} -> frames block of {k} to val, the frame on "
                         f"each side excluded as buffer. Report {typ} scores as SAME-SESSION.")
            continue
        chosen = set()                                     # rule 2: whole sessions, per variant
        strata: dict[str, list[str]] = {}
        for s in sessions:
            strata.setdefault(stratum_of(by_sess[s][0]), []).append(s)
        for st, ss in sorted(strata.items()):
            n_total = len({session_of(f) for f in known if stratum_of(f) == st} | set(ss))
            n_val_already = len({session_of(f) for f in val if stratum_of(f) == st})
            if n_total < 2:
                continue                                   # a lone session of a variant trains
            k = max(1, round(0.2 * n_total)) - n_val_already
            order = sorted(ss, key=lambda s: (s not in prefer_val_sessions, hashlib.md5(s.encode()).hexdigest()))
            chosen |= set(order[:max(0, min(k, n_total - 1 - n_val_already))])
        for s in sessions:
            (val if s in chosen else train).update(by_sess[s])
            notes.append(f"session {s} ({len(by_sess[s])} frames) -> {'val' if s in chosen else 'train'}")
    out = dict(split)
    out.update(val=sorted(val), excluded=sorted(exc), train=sorted(train))
    return out, notes
