"""labelme JSON parser + category iterator.

Each image in ``IMG_OCR_IND_CN/<category>/`` has a sibling ``.json`` sidecar
in labelme v4 format::

    {"shapes": [{"label": "PROVINSI JAWA BARAT",
                 "points": [[x1,y1], [x2,y2], ...],
                 "shape_type": "rectangle"|"polygon"}, ...], ...}

We yield ``GroundTruthPage`` with polygons + transcripts. Images without a
matching JSON are skipped (logged once via stderr by the runner, not here).
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


def load_category(category_dir: Path) -> list[GroundTruthPage]:
    """Load all pages (image + GT) in one category directory."""
    pages: list[GroundTruthPage] = []
    for image_path in sorted(category_dir.glob("*.jpg")):
        json_path = image_path.with_suffix(".json")
        if not json_path.exists():
            continue
        try:
            lines = _parse_labelme(json_path)
        except (json.JSONDecodeError, OSError):
            continue
        pages.append(GroundTruthPage(image_path=image_path, category=category_dir.name, lines=lines))
    # PNG too — there are 4 PNGs in the dataset
    for image_path in sorted(category_dir.glob("*.png")):
        json_path = image_path.with_suffix(".json")
        if not json_path.exists():
            continue
        try:
            lines = _parse_labelme(json_path)
        except (json.JSONDecodeError, OSError):
            continue
        pages.append(GroundTruthPage(image_path=image_path, category=category_dir.name, lines=lines))
    return pages


def list_categories(root: Path | None = None) -> list[Path]:
    """Return category subdirectories sorted by name.

    Skips hidden dirs and any dir that contains no image+json pairs.
    """
    root = root or DEFAULT_DATASET_ROOT
    if not root.exists():
        return []
    cats: list[Path] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir() or p.name.startswith("_") or p.name.startswith("."):
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