"""Polygon IoU + greedy matcher.

Polygon → axis-aligned bounding box (lines are tight rectangles; rotated IoU
is overkill for line text). Greedy match by descending IoU; first match wins.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")
U = TypeVar("U")


def aabb(polygon: list[list[float]]) -> tuple[float, float, float, float]:
    """Return (xmin, ymin, xmax, ymax) of a polygon's AABB."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def iou_polygon(a: list[list[float]], b: list[list[float]]) -> float:
    """IoU of two polygons via their AABBs. Returns 0.0 on zero-area boxes."""
    ax1, ay1, ax2, ay2 = aabb(a)
    bx1, by1, bx2, by2 = aabb(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Match:
    gt_polygon: list[list[float]]
    pr_polygon: list[list[float]]
    iou: float


def match(
    gt_items: list[T],
    pr_items: list[U],
    gt_poly=lambda x: x.polygon,  # type: ignore[assignment]
    pr_poly=lambda x: x.polygon,  # type: ignore[assignment]
    iou_threshold: float = 0.5,
) -> tuple[list[tuple[T, U, float]], list[T], list[U]]:
    """Greedy IoU matcher. Returns (matches, unmatched_gt, unmatched_pr).

    Pairs sorted by IoU descending; first valid match wins. ponytail: global
    O(n*m) — fine for our scale (n lines per image < 50).
    """
    pairs: list[tuple[float, T, U]] = []
    for g in gt_items:
        for p in pr_items:
            iou_val = iou_polygon(gt_poly(g), pr_poly(p))
            if iou_val >= iou_threshold:
                pairs.append((iou_val, g, p))
    pairs.sort(key=lambda x: -x[0])

    used_g: set[int] = set()
    used_p: set[int] = set()
    matches: list[tuple[T, U, float]] = []
    for iou_val, g, p in pairs:
        if id(g) in used_g or id(p) in used_p:
            continue
        matches.append((g, p, iou_val))
        used_g.add(id(g))
        used_p.add(id(p))

    unmatched_gt = [g for g in gt_items if id(g) not in used_g]
    unmatched_pr = [p for p in pr_items if id(p) not in used_p]
    return matches, unmatched_gt, unmatched_pr


if __name__ == "__main__":  # ponytail: self-check
    # 2 GT, 2 PR overlapping perfectly → expect 2 matches, 0 unmatched
    g1 = [[0, 0], [10, 0], [10, 10], [0, 10]]
    g2 = [[20, 20], [30, 20], [30, 30], [20, 30]]
    p1 = [[1, 1], [11, 1], [11, 11], [1, 11]]  # matches g1
    p2 = [[19, 19], [31, 19], [31, 31], [19, 31]]  # matches g2
    m, ug, up = match([1, 2], [10, 20], lambda x: g1 if x == 1 else g2, lambda x: p1 if x == 10 else p2)
    assert len(m) == 2 and not ug and not up, f"got {len(m)} matches, {len(ug)} ug, {len(up)} up"
    # 1 GT, 1 PR overlapping poorly → no match
    m2, ug2, up2 = match([1], [99], lambda x: [[0,0],[10,0],[10,10],[0,10]], lambda x: [[100,100],[110,100],[110,110],[100,110]])
    assert not m2 and ug2 == [1] and up2 == [99]
    print("matcher self-check: OK")