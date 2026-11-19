"""
Perspective rectification from detected boxes.

Idea: objects standing in rows (e.g. products on shelves) share a common baseline. Under a
perspective view those baselines are no longer parallel - they converge towards a vanishing
point on the side of the image that is farther from the camera. We:

  1. take one anchor point per box (bottom / center / top of the box, horizontally centered),
  2. group the anchors into rows,
  3. fit a regression line y = slope * x + intercept per row,
  4. compare the vertical spread of the lines at the left vs. right image edge - the side where
     they are closer together is the side the image converges to (the far side), and is the
     side that must be stretched,
  5. build a homography that maps the outermost lines to horizontal lines, keeping the spacing of
     the near side and stretching the far side to match.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

ANCHORS = ('bottom', 'center', 'top')


@dataclass
class RowLine:
    """A regression line through the anchors of one row of boxes: y = slope * x + intercept."""
    slope: float
    intercept: float
    box_indices: List[int]
    residual_std: float

    def y_at(self, x):
        return self.slope * np.asarray(x, dtype=float) + self.intercept


@dataclass
class PerspectiveAnalysis:
    image_shape: Tuple[int, int]                 # (height, width)
    anchor: str
    lines: List[RowLine] = field(default_factory=list)
    # 'left' / 'right': the side the lines converge towards (= the side to stretch);
    # 'none': lines are (close to) parallel; None: not enough rows to decide.
    converges_to: Optional[str] = None
    spread_left: Optional[float] = None          # vertical distance top line -> bottom line at x=0
    spread_right: Optional[float] = None         # same at x=width
    stretch_ratio: Optional[float] = None        # near spread / far spread (>= 1)
    vanishing_point: Optional[Tuple[float, float]] = None
    homography: Optional[np.ndarray] = None      # 3x3, maps original -> rectified pixels
    output_size: Optional[Tuple[int, int]] = None  # (width, height) of the rectified image
    reason: str = ''

    def to_dict(self):
        d = asdict(self)
        d['homography'] = self.homography.tolist() if self.homography is not None else None
        return d


# --------------------------------------------------------------------------------------------
# Anchors and rows
# --------------------------------------------------------------------------------------------

def box_anchor(x1, y1, x2, y2, anchor='bottom'):
    """Anchor point of a box: horizontally centered, vertically at bottom / center / top."""
    if anchor not in ANCHORS:
        raise ValueError(f"anchor must be one of {ANCHORS}, got '{anchor}'")
    y = {'bottom': y2, 'center': (y1 + y2) / 2, 'top': y1}[anchor]
    return (x1 + x2) / 2, y


def group_into_rows(boxes: np.ndarray, anchor='bottom', row_tolerance=0.5) -> List[List[int]]:
    """
    Group boxes into horizontal rows.

    Boxes are visited left to right; each joins the row whose line (fitted on its members so
    far, or the last member's anchor while it has only one) passes closest to the box's anchor,
    provided that distance is under `row_tolerance` * median box height. Otherwise it starts a
    new row. Walking left to right lets a row follow its own tilt, which plain y-clustering
    can't do once rows are steep.

    Args:
        boxes: (N, 4) array of x1, y1, x2, y2
    Returns:
        list of rows, each a list of indices into `boxes`
    """
    if len(boxes) == 0:
        return []
    anchors = np.array([box_anchor(*b, anchor=anchor) for b in boxes])
    heights = boxes[:, 3] - boxes[:, 1]
    tol = row_tolerance * float(np.median(heights))

    rows: List[List[int]] = []
    for i in np.argsort(anchors[:, 0]):
        x, y = anchors[i]
        best, best_dist = None, tol
        for r, members in enumerate(rows):
            pts = anchors[members]
            if len(members) >= 2 and np.ptp(pts[:, 0]) > 1e-6:
                slope, intercept = np.polyfit(pts[:, 0], pts[:, 1], 1)
                pred = slope * x + intercept
            else:
                pred = pts[-1, 1]
            dist = abs(y - pred)
            if dist < best_dist:
                best, best_dist = r, dist
        if best is None:
            rows.append([int(i)])
        else:
            rows[best].append(int(i))
    return rows


def fit_row_line(points: np.ndarray, indices: Sequence[int], outlier_sigma=2.0) -> Optional[RowLine]:
    """Least-squares line through `points`, refit once without points beyond `outlier_sigma`."""
    idx = np.asarray(indices)
    pts = points[idx]
    if len(pts) < 2 or np.ptp(pts[:, 0]) < 1e-6:
        return None
    slope, intercept = np.polyfit(pts[:, 0], pts[:, 1], 1)
    resid = pts[:, 1] - (slope * pts[:, 0] + intercept)
    std = float(resid.std())
    if len(pts) > 3 and std > 0:
        keep = np.abs(resid) <= outlier_sigma * std
        if keep.sum() >= 2 and keep.sum() < len(pts) and np.ptp(pts[keep, 0]) > 1e-6:
            idx, pts = idx[keep], pts[keep]
            slope, intercept = np.polyfit(pts[:, 0], pts[:, 1], 1)
            std = float((pts[:, 1] - (slope * pts[:, 0] + intercept)).std())
    return RowLine(float(slope), float(intercept), [int(i) for i in idx], std)


def vanishing_point(lines: Sequence[RowLine]) -> Optional[Tuple[float, float]]:
    """Least-squares intersection of the lines (None if they are parallel)."""
    if len(lines) < 2:
        return None
    # each line: slope * x - y = -intercept
    A = np.array([[l.slope, -1.0] for l in lines])
    b = np.array([-l.intercept for l in lines])
    if np.linalg.matrix_rank(A, tol=1e-9) < 2:
        return None
    (x, y), *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(x), float(y)


# --------------------------------------------------------------------------------------------
# Analysis + rectification
# --------------------------------------------------------------------------------------------

def analyze_perspective(
    boxes,
    image_shape,
    anchor='bottom',
    min_boxes_per_row=3,
    row_tolerance=0.5,
    min_stretch_ratio=1.03,
    max_stretch_ratio=3.0,
    min_line_gap_frac=0.1,
) -> PerspectiveAnalysis:
    """
    Work out which side of the image the rows converge to and the homography that fixes it.

    Args:
        boxes: (N, 4) x1, y1, x2, y2 in pixels
        image_shape: (height, width)
        anchor: which part of each box to fit lines through - 'bottom', 'center' or 'top'
        min_boxes_per_row: rows with fewer boxes are ignored
        row_tolerance: see group_into_rows
        min_stretch_ratio: below this near/far ratio the image is considered already rectified
        max_stretch_ratio: refuse to correct beyond this (usually a bad row fit)
        min_line_gap_frac: the outer two lines must be at least this fraction of the image
            height apart, otherwise the spread estimate is too noisy to trust
    """
    h, w = image_shape
    result = PerspectiveAnalysis(image_shape=(int(h), int(w)), anchor=anchor)
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if len(boxes) == 0:
        result.reason = 'no detections'
        return result

    anchors = np.array([box_anchor(*b, anchor=anchor) for b in boxes])
    rows = [r for r in group_into_rows(boxes, anchor, row_tolerance) if len(r) >= min_boxes_per_row]
    lines = [l for l in (fit_row_line(anchors, r) for r in rows) if l is not None]
    lines.sort(key=lambda l: l.y_at(w / 2))
    result.lines = lines
    if len(lines) < 2:
        result.reason = f'need >= 2 rows with >= {min_boxes_per_row} boxes, found {len(lines)}'
        return result

    top, bottom = lines[0], lines[-1]
    spread_left = float(bottom.y_at(0) - top.y_at(0))
    spread_right = float(bottom.y_at(w) - top.y_at(w))
    result.spread_left, result.spread_right = spread_left, spread_right
    result.vanishing_point = vanishing_point(lines)

    if min(spread_left, spread_right) < min_line_gap_frac * h:
        result.reason = 'outer rows too close together (or crossing) to estimate perspective'
        return result

    near, far = max(spread_left, spread_right), min(spread_left, spread_right)
    ratio = near / far
    result.stretch_ratio = float(ratio)
    if ratio < min_stretch_ratio:
        result.converges_to = 'none'
    else:
        result.converges_to = 'right' if spread_right < spread_left else 'left'
    if ratio > max_stretch_ratio:
        result.reason = f'stretch ratio {ratio:.2f} exceeds max {max_stretch_ratio}; not correcting'
        return result

    # Map the outer lines' end points onto horizontal lines at the near side's heights.
    # If the lines are merely parallel-but-tilted this still levels them (shear correction).
    near_x = 0 if spread_left >= spread_right else w
    src = np.float32([
        [0, top.y_at(0)], [w, top.y_at(w)],
        [w, bottom.y_at(w)], [0, bottom.y_at(0)],
    ])
    y_top, y_bot = float(top.y_at(near_x)), float(bottom.y_at(near_x))
    dst = np.float32([[0, y_top], [w, y_top], [w, y_bot], [0, y_bot]])
    H = cv2.getPerspectiveTransform(src, dst)

    # Expand the canvas so nothing stretched off-frame is lost.
    corners = cv2.perspectiveTransform(np.float32([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]]), H)
    xmin, ymin = corners.reshape(-1, 2).min(axis=0)
    xmax, ymax = corners.reshape(-1, 2).max(axis=0)
    T = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=float)
    result.homography = T @ H
    result.output_size = (int(np.ceil(xmax - xmin)), int(np.ceil(ymax - ymin)))
    return result


def rectify_image(image: np.ndarray, analysis: PerspectiveAnalysis, border_value=(0, 0, 0)):
    """Warp `image` with the analysis' homography (returns the input unchanged if there is none)."""
    if analysis.homography is None:
        return image
    return cv2.warpPerspective(image, analysis.homography, analysis.output_size,
                               flags=cv2.INTER_LINEAR, borderValue=border_value)


def transform_boxes(boxes, H) -> np.ndarray:
    """Map (N, 4) xyxy boxes through homography H, returning axis-aligned xyxy boxes."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if H is None or len(boxes) == 0:
        return boxes
    x1, y1, x2, y2 = boxes.T
    corners = np.stack([np.stack([x1, y1], 1), np.stack([x2, y1], 1),
                        np.stack([x2, y2], 1), np.stack([x1, y2], 1)], 1)  # (N, 4, 2)
    warped = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H).reshape(-1, 4, 2)
    return np.concatenate([warped.min(axis=1), warped.max(axis=1)], axis=1)


def draw_analysis(image: np.ndarray, boxes, analysis: PerspectiveAnalysis) -> np.ndarray:
    """Debug view: boxes, anchors, one colored regression line per row, and the verdict."""
    out = image.copy()
    h, w = out.shape[:2]
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    palette = [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 200, 255), (255, 0, 255), (255, 255, 0)]
    row_of = {i: r for r, line in enumerate(analysis.lines) for i in line.box_indices}

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        color = palette[row_of[i] % len(palette)] if i in row_of else (160, 160, 160)
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
        ax, ay = box_anchor(x1, y1, x2, y2, analysis.anchor)
        cv2.circle(out, (int(ax), int(ay)), 4, color, -1)

    for r, line in enumerate(analysis.lines):
        color = palette[r % len(palette)]
        cv2.line(out, (0, int(round(line.y_at(0)))), (w, int(round(line.y_at(w)))), color, 2)

    label = f"converges: {analysis.converges_to}"
    if analysis.stretch_ratio:
        label += f"  stretch x{analysis.stretch_ratio:.2f}"
    if analysis.reason:
        label += f"  ({analysis.reason})"
    scale = max(0.5, w / 1600)
    cv2.putText(out, label, (10, int(30 * scale)), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(out, label, (10, int(30 * scale)), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1)
    return out
