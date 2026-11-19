"""
Stitch a horizontally moving video, or a few spatially consecutive photos, into one wide still
image.

Photos (stitch_images) skip step 1: they are used as given, in their order along the scene
(file-name order in the CLI), and go straight to stitching and the border crop.

1. Keyframes (select_keyframes): every frame is downscaled to grey and its horizontal shift
   against the previous frame measured by phase correlation. Once the view has moved
   (1 - overlap) of a frame width since the last keyframe, the sharpest frame (variance of the
   Laplacian) among the last few is taken as the next keyframe, so motion-blurred frames are
   avoided. The first and last frame are always included, so the whole scene is covered.
2. Stitching (stitch_frames): OpenCV's stitching pipeline (as in its stitching_detailed sample),
   except that each keyframe / photo is matched only with its neighbours in order (--match-range), so
   repeated products / shelf sections are not linked to the wrong place: SIFT features,
   camera estimation + bundle adjustment, gain exposure compensation, graph-cut seams and
   multi-band blending. Mode 'scans' (affine) suits a camera moving sideways along a shelf and
   is also fine for moderate turning; 'panorama' (rotation, cylindrical projection) suits a
   camera turning on the spot over a wide angle, and curves straight lines a little.
3. Border crop (crop_to_valid): the empty border around the stitched area is cut off by
   greedily removing the border row / column with the most empty pixels until none is left.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

MODES = ('scans', 'panorama')
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v', '.3gp'}


@dataclass
class StitchResult:
    image: Optional[np.ndarray] = None           # BGR panorama, None if stitching failed
    inputs: List[str] = field(default_factory=list)   # photos: the image paths, in order
    frame_count: int = 0                         # frames in the video / number of photos
    fps: float = 0.0
    keyframes: List[int] = field(default_factory=list)   # indices of the frames / photos stitched
    pan_px: float = 0.0                          # total horizontal motion, in analysis pixels
    mode: str = ''                               # 'scans' / 'panorama'
    used: List[int] = field(default_factory=list)  # keyframes that made it into the panorama
    crop: Optional[Tuple[int, int, int, int]] = None   # x1, y1, x2, y2 of the border crop
    reason: str = ''

    def to_dict(self):
        d = {k: v for k, v in self.__dict__.items() if k != 'image'}
        d['output_size'] = None if self.image is None else [self.image.shape[1], self.image.shape[0]]
        return d


# --------------------------------------------------------------------------------------------
# Keyframes
# --------------------------------------------------------------------------------------------

_ROTATE = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def open_video(path):
    """
    Open a video and return (capture, read): read() -> (ok, frame) with the frame upright.
    Phone videos are often stored sideways with a rotation tag, which some OpenCV builds don't
    apply by default: auto-orientation is switched on, and if the backend can't do that the
    frames are rotated here from the rotation tag.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video {path}")
    rotate = None
    if not (cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1) and cap.get(cv2.CAP_PROP_ORIENTATION_AUTO)):
        rotate = _ROTATE.get(int(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0) % 360)

    def read():
        ok, frame = cap.read()
        if ok and rotate is not None:
            frame = cv2.rotate(frame, rotate)
        return ok, frame
    return cap, read


def _small_grey(frame, width):
    s = width / frame.shape[1]
    g = cv2.cvtColor(cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA),
                     cv2.COLOR_BGR2GRAY)
    return np.float32(g)


def _sharpness(grey):
    return float(cv2.Laplacian(grey, cv2.CV_32F).var())


def select_keyframes(video_path, overlap=0.6, analysis_width=480, max_frames=60, frame_step=1):
    """
    Read the video and return (frames, indices, info): the keyframes (full resolution, BGR),
    their frame indices and {'frame_count', 'fps', 'pan_px'}.
    """
    cap, read = open_video(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

    step_px = (1 - overlap) * analysis_width       # motion between keyframes
    window = None                                  # Hanning window for phase correlation
    frames, indices = [], []
    candidates = []                                # (sharpness, idx, frame, shift_at_frame)
    prev, since_key, total, idx, last = None, 0.0, 0.0, -1, None

    while True:
        ok, frame = read()
        if not ok:
            break
        idx += 1
        if idx % frame_step:
            continue
        grey = _small_grey(frame, analysis_width)
        if window is None:
            window = cv2.createHanningWindow(grey.shape[::-1], cv2.CV_32F)
        last = (idx, frame)
        if prev is None:                           # first frame is always a keyframe
            frames.append(frame); indices.append(idx)
            prev = grey
            continue
        (dx, _), _ = cv2.phaseCorrelate(prev, grey, window)
        prev = grey
        since_key += abs(dx)
        total += abs(dx)
        if since_key >= 0.75 * step_px:            # close to the next keyframe: collect candidates
            candidates.append((_sharpness(grey), idx, frame, since_key))
        if since_key >= step_px and candidates:
            _, k_idx, k_frame, k_shift = max(candidates, key=lambda c: c[0])
            frames.append(k_frame); indices.append(k_idx)
            since_key -= k_shift
            candidates = []
    cap.release()

    # Cover the end of the scene too
    if last is not None and last[0] != indices[-1] and since_key > 0.1 * analysis_width:
        frames.append(last[1]); indices.append(last[0])

    if len(frames) > max_frames:                   # keep an even spread
        keep = np.unique(np.linspace(0, len(frames) - 1, max_frames).round().astype(int))
        frames = [frames[i] for i in keep]
        indices = [indices[i] for i in keep]
    return frames, indices, {'frame_count': idx + 1, 'fps': fps, 'pan_px': total}


# --------------------------------------------------------------------------------------------
# Stitching
# --------------------------------------------------------------------------------------------

def _scaled(frame, scale):
    return frame if scale == 1.0 else cv2.resize(frame, None, fx=scale, fy=scale,
                                                 interpolation=cv2.INTER_AREA)


def stitch_frames(frames, mode='scans', match_range=2, confidence=0.4, work_megapix=0.6,
                  seam_megapix=0.1, max_canvas_ratio=8.0, skip_confidence=1.0, names=None):
    """
    Stitch keyframes (in video order) into one image with OpenCV's stitching pipeline, matching
    each frame only with the `match_range` frames before and after it. A pair only counts when
    its match confidence (inliers / (8 + 0.3 * matches)) reaches `confidence` for direct
    neighbours, which are known to overlap, and `skip_confidence` for pairs further apart, where
    a weak match is more likely wrong (and one wrong pair distorts the whole result). Frames
    without such a link to the rest are left out. `names` label the frames in the reason text.
    Returns (image, valid_mask, used_positions, reason); image is None on failure.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got '{mode}'")
    n = len(frames)
    if n == 1:
        return frames[0], np.full(frames[0].shape[:2], 255, np.uint8), [0], 'the video does not move'
    h, w = frames[0].shape[:2]               # scales from the first frame; sizes may differ
    work = min(1.0, np.sqrt(work_megapix * 1e6 / (h * w)))
    seam = min(1.0, np.sqrt(seam_megapix * 1e6 / (h * w)))
    scans = mode == 'scans'

    # 1. features and matches between neighbouring frames only
    finder = cv2.SIFT_create()
    feats = [cv2.detail.computeImageFeatures2(finder, _scaled(f, work)) for f in frames]
    matcher = (cv2.detail_AffineBestOf2NearestMatcher(False, False, 0.3) if scans
               else cv2.detail_BestOf2NearestMatcher(False, 0.3))
    i, j = np.indices((n, n))
    pairs = matcher.apply2(feats, ((abs(i - j) <= match_range) & (i != j)).astype(np.uint8))
    matcher.collectGarbage()

    # frames that don't connect confidently to the rest are dropped (largest connected group)
    best = np.zeros(n)                                  # best neighbour confidence per frame
    for pm in pairs:
        a, b = pm.src_img_idx, pm.dst_img_idx
        if a < 0 or b < 0:
            continue
        if abs(a - b) == 1:
            best[a] = max(best[a], pm.confidence)
        elif pm.confidence < skip_confidence:
            pm.confidence = 0.0                         # weak non-neighbour pair: ignore it
    used = [int(k) for k in np.ravel(cv2.detail.leaveBiggestComponent(feats, pairs, confidence))]
    names = names or [str(k) for k in range(n)]
    dropped = [f'{names[k]} (best match confidence {best[k]:.2f})' for k in range(n) if k not in used]
    if len(used) < 2:
        return None, None, used, (f'frames could not be matched to each other: '
                                  f'{", ".join(dropped)} < --confidence {confidence}')
    reason = '' if not dropped else (f'left out {", ".join(dropped)}: below --confidence '
                                     f'{confidence} - too little overlap or detail in common')
    frames = [frames[k] for k in used]

    # 2. camera parameters: estimate, then refine with bundle adjustment
    estimator = cv2.detail_AffineBasedEstimator() if scans else cv2.detail_HomographyBasedEstimator()
    ok, cams = estimator.apply(feats, pairs, None)
    if not ok:
        return None, None, used, 'camera estimation failed'
    for c in cams:
        c.R = c.R.astype(np.float32)
    adjuster = cv2.detail_BundleAdjusterAffinePartial() if scans else cv2.detail_BundleAdjusterRay()
    adjuster.setConfThresh(confidence)
    refine = np.zeros((3, 3), np.uint8)
    refine[0, :] = 1
    refine[1, 1:] = 1
    adjuster.setRefinementMask(refine)
    ok, cams = adjuster.apply(feats, pairs, cams)
    if not ok:
        return None, None, used, 'bundle adjustment failed'
    if not scans:  # keep the horizon straight
        for c, R in zip(cams, cv2.detail.waveCorrect([np.copy(c.R) for c in cams],
                                                     cv2.detail.WAVE_CORRECT_HORIZ)):
            c.R = R
    focal = float(np.median([c.focal for c in cams]))
    warp_type = 'affine' if scans else 'cylindrical'

    # 3. seams and exposure, estimated on low-resolution warped frames
    sw = seam / work
    warper = cv2.PyRotationWarper(warp_type, focal * sw)
    corners, imgs, masks = [], [], []
    for c, f in zip(cams, frames):
        K = c.K().astype(np.float32)
        K[0, 0] *= sw; K[0, 2] *= sw; K[1, 1] *= sw; K[1, 2] *= sw
        small = _scaled(f, seam)
        corner, img = warper.warp(small, K, c.R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
        _, mask = warper.warp(np.full(small.shape[:2], 255, np.uint8), K, c.R, cv2.INTER_NEAREST,
                              cv2.BORDER_CONSTANT)
        corners.append(corner); imgs.append(img); masks.append(mask)
    compensator = cv2.detail.ExposureCompensator_createDefault(cv2.detail.ExposureCompensator_GAIN_BLOCKS)
    compensator.feed(corners=corners, images=imgs, masks=masks)
    masks = cv2.detail_GraphCutSeamFinder('COST_COLOR').find(
        [im.astype(np.float32) for im in imgs], corners, masks)

    # 4. compose at full resolution with multi-band blending
    cw = 1.0 / work
    warper = cv2.PyRotationWarper(warp_type, focal * cw)
    corners, sizes = [], []
    for c, f in zip(cams, frames):
        c.focal *= cw; c.ppx *= cw; c.ppy *= cw
        roi = warper.warpRoi((f.shape[1], f.shape[0]), c.K().astype(np.float32), c.R)
        corners.append(roi[0:2]); sizes.append(roi[2:4])
    dst = cv2.detail.resultRoi(corners=corners, sizes=sizes)
    if dst[2] * dst[3] > max_canvas_ratio * sum(f.shape[0] * f.shape[1] for f in frames):
        return None, None, used, (f'alignment diverged (canvas {dst[2]}x{dst[3]}); '
                                  f'try the other --mode')
    blender = cv2.detail_MultiBandBlender()
    blender.setNumBands(max(1, int(np.log2(np.sqrt(dst[2] * dst[3]) * 0.05) - 1)))
    blender.prepare(dst)
    for k, (c, f) in enumerate(zip(cams, frames)):
        K = c.K().astype(np.float32)
        _, img = warper.warp(f, K, c.R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
        _, mask = warper.warp(np.full(f.shape[:2], 255, np.uint8), K, c.R, cv2.INTER_NEAREST,
                              cv2.BORDER_CONSTANT)
        compensator.apply(k, corners[k], img, mask)
        seam_mask = cv2.resize(cv2.dilate(masks[k], None), (mask.shape[1], mask.shape[0]),
                               interpolation=cv2.INTER_LINEAR_EXACT)
        blender.feed(cv2.UMat(img.astype(np.int16)), cv2.bitwise_and(seam_mask, mask), corners[k])
    pano, valid = blender.blend(None, None)
    pano = np.clip(pano, 0, 255).astype(np.uint8)
    pano[valid == 0] = 0
    return pano, valid, used, reason


def crop_to_valid(image, valid=None):
    """
    Cut off the empty border: repeatedly drop the outermost row or column with the largest
    fraction of empty pixels until the border has none. `valid` is the mask of stitched pixels
    (default: non-black pixels). Returns (image, (x1, y1, x2, y2)).
    """
    valid = (image.max(axis=2) > 0) if valid is None else valid > 0
    # work on a reduced mask for speed, then scale the box back
    s = min(1.0, 2000 / max(valid.shape))
    m = cv2.resize(valid.astype(np.uint8), None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST) > 0
    m = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0   # stay clear of the edge
    y1, x1, y2, x2 = 0, 0, m.shape[0], m.shape[1]
    while y2 - y1 > 1 and x2 - x1 > 1:
        sub = m[y1:y2, x1:x2]
        bad = {'top': 1 - sub[0].mean(), 'bottom': 1 - sub[-1].mean(),
               'left': 1 - sub[:, 0].mean(), 'right': 1 - sub[:, -1].mean()}
        side = max(bad, key=bad.get)
        if bad[side] == 0:
            break
        if side == 'top':
            y1 += 1
        elif side == 'bottom':
            y2 -= 1
        elif side == 'left':
            x1 += 1
        else:
            x2 -= 1
    box = (int(np.ceil(x1 / s)), int(np.ceil(y1 / s)), int(x2 / s), int(y2 / s))
    return image[box[1]:box[3], box[0]:box[2]], box


def stitch_video(video_path, overlap=0.6, mode='scans', match_range=2, crop=True,
                 max_frames=60, frame_step=1, confidence=0.4) -> StitchResult:
    """Keyframes -> stitched panorama -> border crop."""
    frames, indices, info = select_keyframes(video_path, overlap=overlap, max_frames=max_frames,
                                             frame_step=frame_step)
    res = StitchResult(frame_count=info['frame_count'], fps=info['fps'],
                       keyframes=indices, pan_px=info['pan_px'], mode=mode)
    if not frames:
        res.reason = 'no frames could be read'
        return res
    pano, valid, used, res.reason = stitch_frames(frames, mode=mode, match_range=match_range,
                                                  confidence=confidence,
                                                  names=[f'frame {i}' for i in indices])
    res.used = [indices[i] for i in used]
    if pano is not None and crop:
        pano, res.crop = crop_to_valid(pano, valid)
    res.image = pano
    return res


def stitch_images(paths, mode='scans', match_range=2, crop=True, confidence=0.4) -> StitchResult:
    """
    Stitch spatially consecutive photos, given in their order along the scene, into one image.
    cv2.imread applies the EXIF orientation, so phone photos come in upright.
    """
    res = StitchResult(inputs=[str(p) for p in paths], mode=mode)
    frames, positions = [], []
    for k, path in enumerate(paths):
        img = cv2.imread(str(path))
        if img is None:
            res.reason = f'could not read {path}; '
            continue
        frames.append(img); positions.append(k)
    res.frame_count, res.keyframes = len(paths), positions
    if len(frames) < 2:
        res.reason += 'need at least two readable images'
        return res
    pano, valid, used, reason = stitch_frames(
        frames, mode=mode, match_range=match_range, confidence=confidence,
        names=[f'{k} ({Path(paths[k]).name})' for k in positions])
    res.reason += reason
    res.used = [positions[i] for i in used]
    if pano is not None and crop:
        pano, res.crop = crop_to_valid(pano, valid)
    res.image = pano
    return res


def draw_contact_sheet(images, labels, height=240):
    """Images side by side at one height, each labelled (debug)."""
    tiles = []
    for img, label in zip(images, labels):
        t = cv2.resize(img, (max(1, int(img.shape[1] * height / img.shape[0])), height))
        cv2.putText(t, str(label), (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        tiles.append(t)
    return cv2.hconcat(tiles) if tiles else None


def draw_keyframes(video_path, indices, height=240):
    """Contact sheet of a video's keyframes, labelled with their frame numbers (debug)."""
    cap, read = open_video(video_path)
    frames, idx, want = {}, -1, set(indices)
    while want:
        ok, frame = read()
        if not ok:
            break
        idx += 1
        if idx in want:
            want.discard(idx)
            frames[idx] = frame
    cap.release()
    return draw_contact_sheet([frames[i] for i in sorted(frames)], sorted(frames), height)
