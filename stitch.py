"""
Stitcher: turn a horizontally moving video, or a few spatially consecutive photos, into one wide
still image.

Run this before preprocess.py: the output folder holds one image per video / photo set and can
be passed to preprocess.py as --source for perspective correction and cropping.

Inputs (--source, one or more paths):
  - a video file                    -> one stitched image per video
  - image files                     -> all of them together are one photo set, in the order
                                       given (--name sets the output name)
  - a folder (searched recursively) -> every video in it is stitched on its own, and every
                                       folder that directly holds two or more images is one
                                       photo set, in file-name order (img2 before img10)
  Photos must overlap their neighbours in that order (about a third of the frame or more).

Steps (see src/preprocessing/stitch.py):
  1. keyframes (videos only) - the horizontal motion between frames is tracked; each time the
     view has moved (1 - --overlap) of a frame width, the sharpest of the recent frames becomes
     a keyframe. The first and last frame are always included.
  2. stitching - OpenCV's stitching pipeline (features, bundle adjustment, exposure
     compensation, seam finding, multi-band blending), with each keyframe / photo matched only
     to its neighbours in order (--match-range) so repeated products aren't linked to the wrong
     place. --mode scans (default) for a camera moving sideways or turning moderately,
     --mode panorama for a camera turning on the spot over a wide angle.
  3. border crop - the black border around the stitched area is cut off (--no-crop keeps it).

Videos with a rotation tag and photos with an EXIF orientation are turned upright first.

Outputs, under --output:
  <name>.jpg                  the stitched image: <video name>.jpg, <photo folder name>.jpg
                              (sub-folders of --source preserved) or --name for listed photos
  debug/<name>_inputs.jpg     the keyframes / photos used, side by side and labelled (--debug)
  stitch.json                 per output: inputs, keyframes, the ones used, mode, border crop,
                              output size and the reason when something went wrong
"""

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing.stitch import (MODES, VIDEO_EXTENSIONS, draw_contact_sheet, draw_keyframes,
                                      stitch_images, stitch_video)

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def _natural_key(path):
    """File-name order with numbers compared as numbers: img2 < img10."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', path.name)]


def collect_jobs(sources, name):
    """
    Returns a list of jobs: dict(kind='video'|'images', paths=[...], out=<relative output path>).
    """
    jobs, loose = [], []
    for src in map(Path, sources):
        if src.is_file():
            ext = src.suffix.lower()
            if ext in VIDEO_EXTENSIONS:
                jobs.append(dict(kind='video', paths=[src], out=Path(src.stem + '.jpg')))
            elif ext in IMAGE_EXTENSIONS:
                loose.append(src)
            else:
                print(f"Skipping {src}: not a video or image")
        elif src.is_dir():
            for d in [src] + sorted(p for p in src.rglob('*') if p.is_dir()):
                rel = d.relative_to(src)
                if 'debug' in rel.parts:
                    continue
                files = sorted((p for p in d.iterdir() if p.is_file()), key=_natural_key)
                for v in (p for p in files if p.suffix.lower() in VIDEO_EXTENSIONS):
                    jobs.append(dict(kind='video', paths=[v], out=rel / (v.stem + '.jpg')))
                images = [p for p in files if p.suffix.lower() in IMAGE_EXTENSIONS]
                if len(images) >= 2:
                    out = (rel.parent / (rel.name + '.jpg')) if rel.parts else Path(d.name + '.jpg')
                    jobs.append(dict(kind='images', paths=images, out=out))
        else:
            print(f"Skipping {src}: not found")
    if loose:
        if len(loose) < 2:
            print(f"Skipping {loose[0]}: a photo set needs at least two images")
        else:
            jobs.append(dict(kind='images', paths=loose,
                             out=Path((name or loose[0].stem + '_stitched') + '.jpg')))
    return jobs


def main(args):
    jobs = collect_jobs(args.source, args.name)
    if not jobs:
        raise SystemExit("Nothing to stitch: no videos and no sets of two or more images found")
    out_dir = Path(args.output)

    records, ok = [], 0
    for job in tqdm(jobs, desc="Stitching", unit="set"):
        label = job['out'].stem
        if job['kind'] == 'video':
            res = stitch_video(job['paths'][0], overlap=args.overlap, mode=args.mode,
                               match_range=args.match_range, crop=not args.no_crop,
                               max_frames=args.max_frames, frame_step=args.frame_step,
                               confidence=args.confidence)
            record = {'source': 'video', 'video_path': str(job['paths'][0])}
            unit = 'frames'
        else:
            res = stitch_images(job['paths'], mode=args.mode, match_range=args.match_range,
                                crop=not args.no_crop, confidence=args.confidence)
            record = {'source': 'images'}
            unit = 'images'
        record.update(res.to_dict(), output_path=None)

        if res.image is None:
            tqdm.write(f"{label}: stitching failed - {res.reason}")
        else:
            if res.reason:
                tqdm.write(f"{label}: {res.reason}")
            out_path = out_dir / job['out']
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), res.image, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
            record['output_path'] = str(out_path)
            ok += 1
            tqdm.write(f"{label}: {len(res.used)}/{res.frame_count} {unit} ({res.mode}) "
                       f"-> {res.image.shape[1]}x{res.image.shape[0]}")

        if args.debug and res.keyframes:
            if job['kind'] == 'video':
                sheet = draw_keyframes(job['paths'][0], res.keyframes)
            else:
                imgs = [(k, cv2.imread(str(job['paths'][k]))) for k in res.keyframes]
                sheet = draw_contact_sheet([im for _, im in imgs],
                                           [f"{k}{'' if k in res.used else ' (unused)'}"
                                            for k, _ in imgs])
            if sheet is not None:
                dbg = out_dir / 'debug' / job['out'].with_name(f'{label}_inputs.jpg')
                dbg.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dbg), sheet)
        records.append(record)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'stitch.json', 'w') as f:
        json.dump(records, f, indent=2)
    print(f"Stitched {ok}/{len(jobs)}; images and stitch.json in {out_dir}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Stitch horizontally moving videos or spatially consecutive photos into wide '
                    'still images')
    p.add_argument('--source', required=True, nargs='+',
                   help='Video file(s), image files (one photo set, in the order given) and / or '
                        'folders (each video, and each folder of images, is one output)')
    p.add_argument('--output', required=True, help='Output folder')
    p.add_argument('--name', default=None,
                   help='Output name for image files listed in --source '
                        '(default: <first image>_stitched)')
    p.add_argument('--mode', default='scans', choices=MODES,
                   help='scans: camera moving sideways along the scene (or turning moderately); '
                        'panorama: camera turning on the spot over a wide angle')
    p.add_argument('--match-range', type=int, default=2,
                   help='Each keyframe / photo is matched with this many before and after it')
    p.add_argument('--confidence', type=float, default=0.4,
                   help='Min match confidence for two neighbouring inputs to be joined (pairs further '
                        'apart need 1.0); lower it when '
                        'inputs with little detail in common (plain walls, glass, reflections) '
                        'are left out, raise it if parts are joined in wrong places')
    p.add_argument('--no-crop', action='store_true',
                   help='Keep the black border around the stitched area')
    p.add_argument('--jpeg-quality', type=int, default=95)
    p.add_argument('--debug', action='store_true',
                   help='Also save the keyframes / photos used side by side')

    v = p.add_argument_group('videos')
    v.add_argument('--overlap', type=float, default=0.6,
                   help='Overlap between consecutive keyframes, as a fraction of the frame width')
    v.add_argument('--max-frames', type=int, default=60,
                   help='At most this many keyframes are stitched (evenly thinned)')
    v.add_argument('--frame-step', type=int, default=1,
                   help='Read only every n-th frame (faster on long / high-fps videos)')
    main(p.parse_args())
