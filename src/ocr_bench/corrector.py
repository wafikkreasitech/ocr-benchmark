"""OCR text post-processor — 3-stage pipeline.

Stage 1: Char normalization (full-width → ASCII, smart quotes, etc.)
Stage 2: KBBI-based DP word segmentation (re-insert lost spaces)
Stage 3: Conservative SymSpell (only for low-confidence / unmatched tokens)

Why this design:
  * OCR errors on Latin Indonesian are dominated by MISSING SPACES (e.g.
    "MENGURUSRUMAHTANGGA" instead of "MENGURUS RUMAH TANGGA") and full-width
    char substitution (":" → "："). SymSpell alone is the wrong tool — it
    tries to "fix" words that were never broken in the first place.
  * Word segmentation re-inserts spaces by finding the maximum-coverage
    partition of the input where each piece is a KBBI word. Unrecognized
    runs pass through untouched (no mangle-on-typo).
  * SymSpell runs LAST on segments that didn't match — catches true per-word
    typos like "kom petitif" → "kompetitif" where the OCR engine already
    inserted spaces.

Lazy: KBBI trie is built only on first ``get_corrector()`` call AND only
if ``settings.enable_symspell_correction`` is True.
"""
from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from symspellpy import SymSpell, Verbosity

from .config import Settings, get_settings
from .paths import PACKAGE_ROOT


# ─── Stage 1: char normalization table ──────────────────────────────
# Built once, applied per character. Full-width → ASCII + smart quotes +
# common OCR mis-reads. ponytail: literal table, no config knob.
_CHAR_MAP = {
    "：": ":", "，": ",", "．": ".", "／": "/", "；": ";",
    "（": "(", "）": ")", "［": "[", "］": "]", "｛": "{", "｝": "}",
    "！": "!", "？": "?", "＂": '"', "＇": "'", "＠": "@",
    "＃": "#", "＄": "$", "％": "%", "＆": "&", "＊": "*",
    "＋": "+", "＝": "=", "－": "-", "＿": "_",
    "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
    "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
    # Smart quotes
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-",
    # Misc
    "…": "...",
    " ": " ",  # non-breaking space → space
}
_CHAR_TRANS = str.maketrans(_CHAR_MAP)


# Pre-compiled patterns
_PUNCT_NO_SPACE_BEFORE = re.compile(r"\s*([,.;:!?])")
# Insert space between letter and digit when clearly joined: "abc123" stays
# (could be intentional), but "Jl. 123" is fine; we don't auto-add here —
# rely on Stage 2 segmentation.


@dataclass
class CorrectionResult:
    original: str
    corrected: str
    status: str   # "unchanged" | "exact" | "segmented" | "corrected" | "not_found"
    stages_applied: list[str] = None  # for debugging


class Corrector:
    """Lazy-loaded corrector. Singleton via get_corrector()."""

    # Max word length to try in segmentation. KBBI has short words ("a", "di")
    # up to long compounds ("menggunakannya"). 24 covers >99% of cases.
    MAX_SEG_LEN = 24

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._lock = Lock()
        self._loaded = False
        self._kbbi_set: set[str] | None = None
        self._symspell: SymSpell | None = None

    @property
    def enabled(self) -> bool:
        return self.settings.enable_symspell_correction

    @property
    def segmentation_enabled(self) -> bool:
        return self.settings.enable_word_segmentation

    def ensure_loaded(self) -> None:
        if not self.enabled or self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self._build()
            self._loaded = True

    def _build(self) -> None:
        csv_path = Path(self.settings.kbbi_csv_path)
        if not csv_path.is_absolute():
            csv_path = PACKAGE_ROOT / csv_path
        if not csv_path.exists():
            raise FileNotFoundError(f"KBBI CSV not found: {csv_path}")

        self._symspell = SymSpell(
            max_dictionary_edit_distance=self.settings.symspell_max_edit_distance,
            prefix_length=7,
        )
        self._kbbi_set = set()
        top_n = self.settings.kbbi_top_n if self.settings.kbbi_top_n > 0 else None
        loaded = 0
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if not row:
                    continue
                kata = row[0].strip().lower()
                if not kata:
                    continue
                self._kbbi_set.add(kata)
                self._symspell.create_dictionary_entry(kata, 1)
                loaded += 1
                if top_n is not None and loaded >= top_n:
                    break

    # ─── Stage 1 ────────────────────────────────────────────────────
    @staticmethod
    def _normalize_chars(text: str) -> str:
        return text.translate(_CHAR_TRANS)

    # ─── Stage 2: KBBI DP word segmentation ─────────────────────────
    def _segment(self, text: str) -> tuple[str, bool]:
        """Insert spaces using KBBI dictionary via forward DP.

        Cost function (maximized):
          * matched chars:   +1 each
          * unmatched chars: -2 each (heavy penalty — output must be mostly words)
          * matched segment: +2 bonus (reward for using the dictionary)

        DP finds the path to position n with max total cost. Backtrack
        reconstructs with spaces inserted between matched word runs.

        Non-alpha chars (digits, punct, spaces) are preserved as-is and
        don't participate in segmentation decisions (they're separators).
        """
        if not self._kbbi_set or not text:
            return text, False

        n = len(text)
        lower = text.lower()
        if lower in self._kbbi_set:
            return text, False

        # Build list of (start, end) spans of alpha-runs only; punct/digits
        # act as separators and pass through verbatim.
        alpha_spans: list[tuple[int, int]] = []
        i = 0
        while i < n:
            if text[i].isalpha():
                j = i
                while j < n and text[j].isalpha():
                    j += 1
                alpha_spans.append((i, j))
                i = j
            else:
                i += 1

        if not alpha_spans:
            return text, False

        # For each alpha span, do DP segmentation. Other chars emit verbatim.
        out_parts: list[str] = []
        any_inserted = False
        cursor = 0

        for span_start, span_end in alpha_spans:
            # Emit any non-alpha chars before this span
            if cursor < span_start:
                out_parts.append(text[cursor:span_start])

            seg_text = text[span_start:span_end]
            seg_lower = lower[span_start:span_end]
            m = len(seg_text)

            # dp[i] = (max_score, prev_j, matched_bool_of_segment_j_to_i)
            NEG_INF = float("-inf")
            dp_score = [NEG_INF] * (m + 1)
            dp_prev = [-1] * (m + 1)
            dp_matched = [False] * (m + 1)
            dp_score[0] = 0

            for i2 in range(1, m + 1):
                best_score = NEG_INF
                best_j = -1
                best_matched = False
                # Try matched segments ending at i2 — min length 4 to avoid
                # over-segmenting on common 2-3 letter words like "di", "ke".
                max_len = min(self.MAX_SEG_LEN, i2)
                for length in range(4, max_len + 1):
                    j2 = i2 - length
                    if dp_score[j2] == NEG_INF:
                        continue
                    if seg_lower[j2:i2] in self._kbbi_set:
                        score = dp_score[j2] + length + 2  # bonus
                        if score > best_score:
                            best_score = score
                            best_j = j2
                            best_matched = True
                # Try unmatched pass-through (length 1..3 only)
                for length in range(1, min(3, i2) + 1):
                    j2 = i2 - length
                    if dp_score[j2] == NEG_INF:
                        continue
                    score = dp_score[j2] + (-3 * length)  # heavy penalty
                    if score > best_score:
                        best_score = score
                        best_j = j2
                        best_matched = False
                dp_score[i2] = best_score
                dp_prev[i2] = best_j
                dp_matched[i2] = best_matched

            if dp_score[m] == NEG_INF:
                out_parts.append(seg_text)
                cursor = span_end
                continue

            # Backtrack to get segments
            segments: list[tuple[int, int, bool]] = []
            k = m
            while k > 0:
                j2 = dp_prev[k]
                if j2 < 0:
                    out_parts.append(seg_text)
                    segments = []
                    break
                segments.append((j2, k, dp_matched[k]))
                k = j2
            segments.reverse()

            if not segments:
                out_parts.append(seg_text)
                cursor = span_end
                continue

            # Reconstruct this span with spaces
            for idx, (s, e, matched) in enumerate(segments):
                piece = seg_text[s:e]
                if not piece:
                    continue
                if idx > 0:
                    # Insert space between consecutive letter-pieces
                    prev_piece = seg_text[segments[idx - 1][0]:segments[idx - 1][1]]
                    if piece[0].isalpha() and prev_piece and prev_piece[-1].isalpha():
                        out_parts.append(" ")
                        any_inserted = True
                out_parts.append(piece)
            cursor = span_end

        # Tail chars after last span
        if cursor < n:
            out_parts.append(text[cursor:])

        return "".join(out_parts), any_inserted

    # ─── Stage 3: conservative SymSpell ─────────────────────────────
    def _symspell_tokens(self, text: str) -> str:
        """Run SymSpell only on alphabetic tokens that don't match KBBI exactly.

        Skip digits — KBBI has 'a4', 'a5' entries that confuse OCR-number tokens.
        """
        if not self._symspell:
            return text
        parts = re.findall(r"\s+|\w+", text, flags=re.UNICODE)
        out: list[str] = []
        any_changed = False
        for p in parts:
            if not p or p.isspace():
                out.append(p)
                continue
            # Skip pure-digit and short tokens
            if p.isdigit() or len(p) < 3:
                out.append(p)
                continue
            # Skip if it contains a digit (mixed alpha-num, like "abc123")
            if any(c.isdigit() for c in p):
                out.append(p)
                continue
            low = p.lower()
            if low in (self._kbbi_set or set()):
                out.append(p)
                continue
            suggestions = self._symspell.lookup(
                low,
                verbosity=Verbosity.CLOSEST,
                max_edit_distance=self.settings.symspell_max_edit_distance,
            )
            if suggestions and suggestions[0].term != low:
                out.append(suggestions[0].term)
                any_changed = True
            else:
                out.append(p)
        return "".join(out) if any_changed else text

    # ─── Public ─────────────────────────────────────────────────────
    def correct(self, text: str, ocr_score: float | None = None) -> CorrectionResult:
        if not text or not text.strip():
            return CorrectionResult(original=text, corrected=text, status="unchanged", stages_applied=[])

        if not self.enabled:
            return CorrectionResult(original=text, corrected=text, status="unchanged", stages_applied=[])

        self.ensure_loaded()
        assert self._kbbi_set is not None

        stages: list[str] = []
        out = self._normalize_chars(text)
        if out != text:
            stages.append("normalize")

        # Stage 2: segmentation only when input is "joined up" AND feature enabled.
        # KBBI doesn't contain Indonesian affixed forms (me-NG-, me-N-, ber-AN, etc.),
        # so segmentation can split "MENGURUS" into "MENG" + "URUS" — WER goes up.
        # Opt-in only; default off. Useful for cases where the OCR dropped spaces
        # between root words (newspaper print, low-DPI scans).
        if self.segmentation_enabled:
            has_no_spaces = " " not in out and "\t" not in out
            if has_no_spaces and len(out) >= 6:
                segmented, was_seg = self._segment(out)
                if was_seg:
                    stages.append("segment")
                    out = segmented

        # Stage 3: conservative SymSpell. Runs on tokens that segmentation
        # left untouched (i.e. we suspect they might still be wrong). Skipped
        # by default for already-correctly-spaced input to avoid WER explosions
        # on proper nouns / Latin script.
        run_symspell = "segment" in stages or (ocr_score is not None and ocr_score < 0.7)
        if run_symspell:
            fixed = self._symspell_tokens(out)
            if fixed != out:
                stages.append("symspell")
                out = fixed

        status = "unchanged" if out == text else (
            "exact" if out.lower() in self._kbbi_set
            else "segmented" if "segment" in stages
            else "corrected"
        )
        return CorrectionResult(original=text, corrected=out, status=status, stages_applied=stages)


def get_corrector() -> Corrector:
    """Process-wide singleton."""
    global _corrector
    if _corrector is None:
        with _corrector_lock:
            if _corrector is None:
                _corrector = Corrector()
    return _corrector


# Module-level singleton state (lazy)
_corrector: Corrector | None = None
_corrector_lock = Lock()


if __name__ == "__main__":  # ponytail: self-check
    import os
    os.environ["ENABLE_SYMSPELL_CORRECTION"] = "true"
    c = Corrector(Settings(enable_symspell_correction=True))
    samples = [
        "MENGURUSRUMAHTANGGA",
        "Tempat/IglLahir:JEMBER,20-05-1974",
        "KOTABEKASI",
        "Rencana yang sangat matang",
        "kom petitif",
        "Hello world",
        "34.45",
        " mendorong pertumbuhan eko-",
        "untuk bertindak dengan itikad baik",
    ]
    for s in samples:
        r = c.correct(s, ocr_score=0.5)  # force symspell
        print(f"  {s!r:50} -> {r.corrected!r:50} [{r.status}] stages={r.stages_applied}")