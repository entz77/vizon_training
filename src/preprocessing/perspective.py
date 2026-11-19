"""
Perspective rectification from detected boxes.

Idea: objects standing in rows (e.g. products on shelves) share a common baseline. Under a
perspective view those baselines are no longer parallel - they converge towards a vanishing
point on the side of the image that is farther from the camera. We:

  1. take one anchor point per box (bottom / center / top of the box, horizontally centered),
  2. group the anchors into rows,
  3. fit a regression line y = slope * x + intercept per row,
  4. drop rows whose slope is an outlier: rows converging to one vanishing point have a slope
     that is linear in their height, so a robust slope-vs-height fit flags rows that don't share
     the vanishing point; the fit over the remaining rows is the perspective model,
  5. take the model's lines at the top-most and bottom-most kept row and compare their vertical
     spread at the left vs. right image edge - the side where they are closer together is the
     side the image converges to (the far side), and is the side that must be stretched,
  6. build a homography that maps those two lines to horizontal lines, keeping the spacing of
     the near side and stretching the far side to match.

Horizontal counterpart (analyze_shelf_edges): when the camera is tilted up / down, the shelf
unit's left and right sides converge towards the top or bottom. Each shelf box's left end (x1)
and right end (x2) lie on those sides, so a line is fitted through the left ends and one through
the right ends; their intersection is the vertical vanishing point, and the edge nearer to it is
stretched horizontally so the sides become parallel.
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
    # 'vertical': rows converge left / right, fixed by stretching vertically (this module).
    # 'horizontal': the shelf sides converge top / bottom, fixed by stretching horizontally
    #   (analyze_shelf_edges). lines are the left / right side, x = slope * y + intercept,
    #   with box_indices the shelves whose end is on that side; spread_left / spread_right are
    #   the side-to-side widths at the top / bottom edge.
    direction: str = 'vertical'
    lines: List[RowLine] = field(default_factory=list)            # rows used for the correction
    rejected_lines: List[RowLine] = field(default_factory=list)   # rows dropped as slope outliers
    rejected_boxes: List[int] = field(default_factory=list)       # horizontal: trimmed edge shelves
    # slope of a row as a linear function of its height at the image centre:
    # slope = a * y_mid + b (exact for rows converging to one vanishing point)
    slope_model: Optional[Tuple[float, float]] = None
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
    # Only trust a row's own tilt once its members span at least one box width horizontally;
    # boxes stacked at (almost) the same x would otherwise yield a near-vertical "row".
    min_span = float(np.median(boxes[:, 2] - boxes[:, 0]))

    rows: List[List[int]] = []
    for i in np.argsort(anchors[:, 0]):
        x, y = anchors[i]
        best, best_dist = None, tol
        for r, members in enumerate(rows):
            pts = anchors[members]
            if len(members) >= 2 and np.ptp(pts[:, 0]) >= min_span:
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


def fit_slope_model(lines: Sequence[RowLine], x_ref: float, slope_tolerance=0.03):
    """
    Robustly fit slope = a * y_ref + b over the row lines, with y_ref = line height at x_ref.

    Rows that converge to one vanishing point (vx, vy) satisfy this exactly, with
    a = 1 / (x_ref - vx) and b = -vy * a - so a row whose slope is far from the trend of the
    others doesn't go through the common vanishing point and is an outlier (mis-grouped boxes,
    a row mixing two shelves, too few boxes for a stable fit, ...).

    Every pair of lines proposes a model; the one with the most inliers (|slope - model| <=
    slope_tolerance) wins, ties broken by the inliers' total box count. The model is then refit
    on the inliers by least squares weighted by box count.

    Returns:
        (a, b), inlier mask (bool array, same order as `lines`)
    """
    y = np.array([l.y_at(x_ref) for l in lines], dtype=float)
    s = np.array([l.slope for l in lines], dtype=float)
    n = np.array([len(l.box_indices) for l in lines], dtype=float)

    best, best_score = np.ones(len(lines), bool), (-1, -1.0)
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if abs(y[j] - y[i]) < 1e-6:
                continue
            a = (s[j] - s[i]) / (y[j] - y[i])
            b = s[i] - a * y[i]
            inl = np.abs(s - (a * y + b)) <= slope_tolerance
            inl[[i, j]] = True
            score = (int(inl.sum()), float(n[inl].sum()))
            if score > best_score:
                best, best_score = inl, score

    a, b = np.polyfit(y[best], s[best], 1, w=np.sqrt(n[best])) if best.sum() >= 2 else (0.0, s.mean())
    return (float(a), float(b)), best


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
    max_row_slope=0.5,
    slope_tolerance=0.03,
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
        max_row_slope: rows steeper than this (|dy/dx|) are dropped as mis-grouped - a shelf
            row is never close to vertical
        slope_tolerance: rows whose slope is further than this from the slope-vs-height trend
            of the other rows are dropped (see fit_slope_model)
    """
    h, w = image_shape
    result = PerspectiveAnalysis(image_shape=(int(h), int(w)), anchor=anchor)
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if len(boxes) == 0:
        result.reason = 'no detections'
        return result

    anchors = np.array([box_anchor(*b, anchor=anchor) for b in boxes])
    rows = [r for r in group_into_rows(boxes, anchor, row_tolerance) if len(r) >= min_boxes_per_row]
    lines = [l for l in (fit_row_line(anchors, r) for r in rows)
             if l is not None and abs(l.slope) <= max_row_slope]
    lines.sort(key=lambda l: l.y_at(w / 2))
    result.lines = lines
    if len(lines) < 2:
        result.reason = f'need >= 2 rows with >= {min_boxes_per_row} boxes, found {len(lines)}'
        return result

    # Drop rows whose slope breaks the common-vanishing-point trend, then describe the remaining
    # rows by one slope model so all of them (not just the outer two) shape the correction.
    xc = w / 2
    (a, b), inliers = fit_slope_model(lines, xc, slope_tolerance)
    result.lines = [l for l, k in zip(lines, inliers) if k]
    result.rejected_lines = [l for l, k in zip(lines, inliers) if not k]
    result.slope_model = (a, b)
    if abs(a) > 1e-12:
        result.vanishing_point = (float(xc - 1 / a), float(-b / a))

    def model_line(y_mid):
        slope = a * y_mid + b
        return RowLine(float(slope), float(y_mid - slope * xc), [], 0.0)

    y_mids = [l.y_at(xc) for l in result.lines]
    top, bottom = model_line(min(y_mids)), model_line(max(y_mids))
    spread_left = float(bottom.y_at(0) - top.y_at(0))
    spread_right = float(bottom.y_at(w) - top.y_at(w))
    result.spread_left, result.spread_right = spread_left, spread_right

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
    result.homography, result.output_size = expand_canvas(cv2.getPerspectiveTransform(src, dst), w, h)
    return result


def expand_canvas(H, w, h):
    """Shift H so the warped w x h image starts at (0, 0); returns (H', (out_w, out_h))."""
    corners = cv2.perspectiveTransform(np.float32([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]]), H)
    xmin, ymin = corners.reshape(-1, 2).min(axis=0)
    xmax, ymax = corners.reshape(-1, 2).max(axis=0)
    T = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=float)
    return T @ H, (int(np.ceil(xmax - xmin)), int(np.ceil(ymax - ymin)))


def _fit_side(points: np.ndarray, indices, tol: float, min_points: int) -> Optional[RowLine]:
    """
    Robust line x = slope * y + intercept through shelf end points (x, y): every pair proposes a
    line, the one with the most points within `tol` px wins, refit by least squares on those.
    """
    if len(points) < max(2, min_points):
        return None
    x, y = points[:, 0], points[:, 1]
    best = None
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if abs(y[j] - y[i]) < 1e-6:
                continue
            slope = (x[j] - x[i]) / (y[j] - y[i])
            inl = np.abs(x - (x[i] + slope * (y - y[i]))) <= tol
            if best is None or inl.sum() > best.sum():
                best = inl
    if best is None or best.sum() < min_points or np.ptp(y[best]) < 1e-6:
        return None
    slope, intercept = np.polyfit(y[best], x[best], 1)
    resid = x[best] - (slope * y[best] + intercept)
    return RowLine(float(slope), float(intercept),
                   [int(k) for k, b in zip(indices, best) if b], float(resid.std()))


def analyze_shelf_edges(
    shelf_boxes,
    image_shape,
    edge_margin=0.01,
    tolerance=0.02,
    min_points=3,
    min_stretch_ratio=1.03,
    max_stretch_ratio=4.0,
) -> PerspectiveAnalysis:
    """
    Horizontal perspective from the ends of the shelf boxes.

    Trimmed edge shelves - boxes cut off by the left or right image border that don't reach the
    image's centre column, usually a neighbouring unit's shelf peeking in - are removed first:
    their ends are not on this unit's sides and would skew the fit. Of the remaining shelves,
    an end cut off by the border is skipped (the shelf's other end still counts).

    A line is fitted through the shelves' left ends and one through their right ends. With both,
    their intersection is the vertical vanishing point. With only one (the other side's ends
    run off the image), the vanishing point is taken where that line crosses the image's centre
    column - true when the camera is only tilted up / down, not rolled, which the vertical pass
    has already corrected. The homography then maps lines through the vanishing point and the
    corners of the near edge (the one farther from it) onto the image's left and right edges:
    the near edge keeps its width, the far edge is stretched to match.

    Args:
        shelf_boxes: (N, 4) x1, y1, x2, y2 of the shelf detections only
        image_shape: (height, width)
        edge_margin: an end within this fraction of the width of the image border is cut off by
            the frame, not a shelf end
        tolerance: max distance of an end from its side line, as a fraction of the width
        min_points: min shelf ends on a side to fit its line
        min_stretch_ratio: below this near/far width ratio the image is left as is
        max_stretch_ratio: refuse to correct beyond this
    """
    h, w = image_shape
    res = PerspectiveAnalysis(image_shape=(int(h), int(w)), anchor='shelf-ends',
                              direction='horizontal')
    boxes = np.asarray(shelf_boxes, dtype=float).reshape(-1, 4)
    yc = (boxes[:, 1] + boxes[:, 3]) / 2
    margin = edge_margin * w
    cut_left, cut_right = boxes[:, 0] <= margin, boxes[:, 2] >= w - margin
    trimmed = (cut_left & (boxes[:, 2] < w / 2)) | (cut_right & (boxes[:, 0] > w / 2))
    res.rejected_boxes = [int(i) for i in np.flatnonzero(trimmed)]
    left_ok = np.flatnonzero(~trimmed & ~cut_left)
    right_ok = np.flatnonzero(~trimmed & ~cut_right)
    tol = tolerance * w
    left = _fit_side(np.c_[boxes[left_ok, 0], yc[left_ok]], left_ok, tol, min_points)
    right = _fit_side(np.c_[boxes[right_ok, 2], yc[right_ok]], right_ok, tol, min_points)
    res.lines = [l for l in (left, right) if l is not None]
    if not res.lines:
        res.reason = (f'need >= {min_points} visible shelf ends on a side; found '
                      f'{len(left_ok)} left, {len(right_ok)} right')
        return res

    # vanishing point: intersection of x = s1 * y + c1 and x = s2 * y + c2
    if left is not None and right is not None:
        if abs(left.slope - right.slope) < 1e-9:
            res.converges_to, res.stretch_ratio = 'none', 1.0
            return res
        vy = (right.intercept - left.intercept) / (left.slope - right.slope)
        res.spread_left = float(right.y_at(0) - left.y_at(0))
        res.spread_right = float(right.y_at(h) - left.y_at(h))
    else:
        side = left or right
        if abs(side.slope) < 1e-9:
            res.converges_to, res.stretch_ratio = 'none', 1.0
            return res
        vy = (w / 2 - side.intercept) / side.slope
    vx = (left or right).y_at(vy)
    vx, vy = float(vx), float(vy)
    res.vanishing_point = (vx, vy)
    if 0 <= vy <= h:
        res.reason = 'shelf sides cross inside the image; not correcting'
        return res

    top = vy < 0
    near_y, far_y = (h, 0) if top else (0, h)
    ratio = abs(near_y - vy) / abs(far_y - vy)
    res.stretch_ratio = float(ratio)
    if ratio < min_stretch_ratio:
        res.converges_to = 'none'
        return res
    res.converges_to = 'top' if top else 'bottom'
    if ratio > max_stretch_ratio:
        res.reason = f'stretch ratio {ratio:.2f} exceeds max {max_stretch_ratio}; not correcting'
        return res

    def x_on(x_near, y):
        return vx + (x_near - vx) * (y - vy) / (near_y - vy)

    src = np.float32([[x_on(0, 0), 0], [x_on(w, 0), 0], [x_on(w, h), h], [x_on(0, h), h]])
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    res.homography, res.output_size = expand_canvas(cv2.getPerspectiveTransform(src, dst), w, h)
    return res


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
        if analysis.direction == 'vertical':
            ax, ay = box_anchor(x1, y1, x2, y2, analysis.anchor)
            cv2.circle(out, (int(ax), int(ay)), 4, color, -1)

    horizontal = analysis.direction == 'horizontal'
    for i in analysis.rejected_boxes:  # trimmed edge shelves, left out of the fit: crossed out
        x1, y1, x2, y2 = boxes[i].astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 255), max(2, w // 600))
        cv2.line(out, (x1, y1), (x2, y2), (255, 0, 255), max(2, w // 600))
        cv2.line(out, (x1, y2), (x2, y1), (255, 0, 255), max(2, w // 600))
    if horizontal:  # shelf sides: x = slope * y + intercept, through the used shelf ends
        for line in analysis.lines:
            for i in line.box_indices:
                x1, y1, x2, y2 = boxes[i]
                lx = line.y_at((y1 + y2) / 2)
                cx = x1 if abs(lx - x1) < abs(lx - x2) else x2
                cv2.circle(out, (int(cx), int((y1 + y2) / 2)), max(6, w // 200), (0, 220, 0), -1)

    def draw_line(line, color, thickness):
        if horizontal:
            p0, p1 = (line.y_at(0), 0), (line.y_at(h), h)
        else:
            p0, p1 = (0, line.y_at(0)), (w, line.y_at(w))
        cv2.line(out, tuple(int(round(v)) for v in p0), tuple(int(round(v)) for v in p1),
                 color, max(2, w // 600) if horizontal else thickness)

    for r, line in enumerate(analysis.lines):
        draw_line(line, palette[r % len(palette)], 2)
    for line in analysis.rejected_lines:  # slope outliers: thin grey
        draw_line(line, (128, 128, 128), 1)

    label = f"{'shelf sides' if horizontal else 'rows'} converge: {analysis.converges_to}"
    if analysis.stretch_ratio:
        label += f"  stretch x{analysis.stretch_ratio:.2f}"
    if analysis.rejected_lines:
        label += f"  rejected {len(analysis.rejected_lines)} row(s)"
    if analysis.rejected_boxes:
        label += f"  trimmed shelves {len(analysis.rejected_boxes)}"
    if analysis.reason:
        label += f"  ({analysis.reason})"
    scale = max(0.5, w / 1600)
    cv2.putText(out, label, (10, int(30 * scale)), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(out, label, (10, int(30 * scale)), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1)
    return out
