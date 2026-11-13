# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Keep image and clip prompts positive.

A diffusion or video model has no notion of negation: "no people" pulls people
in, "no text" invites lettering. Image plates and clips therefore describe what
is there. The one place a negative is wanted is sound ("no music"), which the
clips do state, so audio clauses are allowed through untouched.
"""

from __future__ import annotations

import re

_NEGATIVE = re.compile(
    r"(?i)\b(?:no|not|never|without|avoid|forbid(?:den)?|exclude|cannot|"
    r"don't|do\s+not|must\s+not|nothing|none)\b"
)
_AUDIO = re.compile(
    r"(?i)\b(?:music|score|sound|audio|speech|dialogue|voice|silent|silence|"
    r"ambient|room\s+tone)\b|\u97f3\u4e50|\u914d\u4e50|\u58f0\u97f3|\u5bf9\u767d|"
    r"\u9759\u97f3|\u73af\u5883\u58f0"
)

# Phrases the plate writers already use, written as the positive they mean.
_REWRITES: tuple[tuple[str, str], ...] = (
    ("one single image, no scene, no environment", "one single image on a plain neutral studio background"),
    ("no scene, no environment", "plain neutral studio background"),
    ("no more, no fewer", "exactly this many"),
    ("no people anywhere in frame", "the place completely empty of people"),
    ("no characters or figures of any kind", "scenery only"),
    ("no people or characters", "empty of people"),
    ("no people", "empty of people"),
    ("no face repeated, no cloned features", "every face distinct, each with its own features"),
    ("no cloned features", "each with its own features"),
    ("no two alike", "each one different from the others"),
    ("no grass", "bare earth"),
    ("no text, no watermark", ""),
    ("no watermark", ""),
    ("no text", ""),
    ("empty of people", "empty of people"),
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?\u3002\uff1b;])\s*")


def _apply_rewrites(text: str) -> str:
    out = text
    for phrase, positive in _REWRITES:
        if phrase in out.lower():
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            out = pattern.sub(positive, out)
    return out


def scrub(text: str, *, allow_audio: bool = True) -> str:
    """Rewrite or drop negative clauses; audio negatives survive."""
    value = str(text or "")
    if not value.strip():
        return value
    value = _apply_rewrites(value)
    kept: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(value):
        if not sentence.strip():
            continue
        if _NEGATIVE.search(sentence) and not (allow_audio and _AUDIO.search(sentence)):
            continue
        kept.append(sentence.strip())
    out = " ".join(kept)
    out = re.sub(r"(\s*,\s*){2,}", ", ", out)
    out = re.sub(r"\s+([.,;:])", r"\1", out)
    return re.sub(r"\s+", " ", out).strip()
