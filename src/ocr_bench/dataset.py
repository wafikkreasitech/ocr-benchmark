"""Dataset loaders.

Two formats are supported; the loader is chosen per-root by inspecting the
directory layout.

labelme v4 (``IMG_OCR_IND_CN/<category>/image.{jpg,png}`` + sibling
``image.json`` with ``shapes[].label`` + ``shapes[].points``).

FUNSD-form (``<root>/<split>/images/*.png`` + ``<root>/<split>/annotations/*.json``
with ``form[]`` entries holding ``box [x1,y1,x2,y2]`` + ``text`` + ``label``).
The split subdir is reported as the synthetic category so the existing
category-based UI keeps working unchanged.

Both parsers yield the same ``GroundTruthPage`` / ``GroundTruthLine`` shape —
the matcher only cares about polygons + text.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .paths import DEFAULT_DATASET_ROOT


@dataclass
class GroundTruthLine:
    polygon: list[list[float]]
    text: str


@dataclass
class GroundTruthPage:
    image_path: Path
    category: str
    lines: list[GroundTruthLine]


# ─── labelme parser ────────────────────────────────────────────────────────

def _parse_labelme(json_path: Path) -> list[GroundTruthLine]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    out: list[GroundTruthLine] = []
    for shape in data.get("shapes", []):
        text = (shape.get("label") or "").strip()
        points = shape.get("points") or []
        if not text or len(points) < 2:
            continue
        out.append(GroundTruthLine(polygon=[[float(x), float(y)] for x, y in points], text=text))
    return out


def _is_labelme_category(dir_: Path) -> bool:
    """A labelme category dir has images directly inside (no ``images/`` subdir)."""
    if not dir_.is_dir():
        return False
    if (dir_ / "images").is_dir():
        return False
    return any(dir_.glob("*.jpg")) or any(dir_.glob("*.png"))


def _load_labelme_category(category_dir: Path) -> list[GroundTruthPage]:
    pages: list[GroundTruthPage] = []
    for ext in ("*.jpg", "*.png"):
        for image_path in sorted(category_dir.glob(ext)):
            json_path = image_path.with_suffix(".json")
            if not json_path.exists():
                continue
            try:
                lines = _parse_labelme(json_path)
            except (json.JSONDecodeError, OSError):
                continue
            pages.append(GroundTruthPage(image_path=image_path, category=category_dir.name, lines=lines))
    return pages


# ─── FUNSD-form parser ─────────────────────────────────────────────────────

def _box_to_polygon(box: list[float]) -> list[list[float]]:
    """Convert FUNSD ``box [x1,y1,x2,y2]`` to a 4-point polygon."""
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _parse_funsd(json_path: Path) -> list[GroundTruthLine]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    out: list[GroundTruthLine] = []
    for entry in data.get("form", []):
        text = (entry.get("text") or "").strip()
        box = entry.get("box") or []
        if not text or len(box) < 4:
            continue
        out.append(GroundTruthLine(polygon=_box_to_polygon(box), text=text))
    return out


def _is_funsd_split_dir(dir_: Path) -> bool:
    """A FUNSD split dir has ``images/`` and ``annotations/`` siblings."""
    return (dir_ / "images").is_dir() and (dir_ / "annotations").is_dir()


def _load_funsd_category(split_dir: Path) -> list[GroundTruthPage]:
    images_dir = split_dir / "images"
    annos_dir = split_dir / "annotations"
    pages: list[GroundTruthPage] = []
    for image_path in sorted(images_dir.glob("*.png")):
        json_path = annos_dir / f"{image_path.stem}.json"
        if not json_path.exists():
            continue
        try:
            lines = _parse_funsd(json_path)
        except (json.JSONDecodeError, OSError):
            continue
        pages.append(GroundTruthPage(image_path=image_path, category=split_dir.name, lines=lines))
    return pages


# ─── Dispatch ──────────────────────────────────────────────────────────────

def _pick_loader(dir_: Path):
    """Return the right ``load_<format>_category`` for a given category dir.

    Falls back to labelme if both heuristics fail — keeps the legacy flow safe.
    """
    if _is_funsd_split_dir(dir_):
        return _load_funsd_category
    if _is_labelme_category(dir_):
        return _load_labelme_category
    # No images found — return labelme (will yield []). Caller skips empty cats.
    return _load_labelme_category


def load_category(category_dir: Path) -> list[GroundTruthPage]:
    """Load all pages (image + GT) in one category directory."""
    loader = _pick_loader(category_dir)
    return loader(category_dir)


def list_categories(root: Path | None = None) -> list[Path]:
    """Return category subdirectories sorted by name.

    For labelme roots: returns each per-category dir as-is.
    For FUNSD roots: returns each split dir (e.g. ``testing_data``,
    ``training_data``) — the category reported to the UI is the split name.

    Skips hidden dirs, anything starting with ``_`` (drops ``__MACOSX/``),
    and any dir that contains no images.
    """
    root = root or DEFAULT_DATASET_ROOT
    if not root.exists():
        return []
    cats: list[Path] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir() or p.name.startswith("_") or p.name.startswith("."):
            continue
        if _is_funsd_split_dir(p):
            cats.append(p)
            continue
        has_images = any(p.glob("*.jpg")) or any(p.glob("*.png"))
        if has_images:
            cats.append(p)
    return cats


def iter_all_images(root: Path | None = None) -> list[GroundTruthPage]:
    """Iterate every page across every category, sorted by category then filename."""
    pages: list[GroundTruthPage] = []
    for cat in list_categories(root):
        pages.extend(load_category(cat))
    return pages


if __name__ == "__main__":  # ponytail: self-check
    cats = list_categories()
    print(f"{len(cats)} categories")
    total = 0
    for cat in cats:
        pages = load_category(cat)
        n_lines = sum(len(p.lines) for p in pages)
        print(f"  {cat.name}: {len(pages)} images, {n_lines} lines")
        total += len(pages)
    print(f"total: {total} images")