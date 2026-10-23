"""Batch-test the pose checker on a folder of photos.

Run:  python check_images.py test_images
      python check_images.py test_images --pose cobra     (force one pose for every image)

For each image it prints the auto-detected pose, its score and the feedback, and
writes an annotated copy (skeleton + angles) to <folder>/out/.  If a filename starts
with a pose short name (tree_1.jpg, warrior2_3.png, ...) the expected pose is shown
too, so you can see at a glance whether auto-detect got it right.
"""
import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageOps

from pose_utils import AUTO, POSES, PoseEstimator, analyze, draw_overlay, resolve_pose

EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def load(path: str) -> np.ndarray:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img.thumbnail((1280, 1280))
    return np.array(img)


def expected_from_name(filename: str) -> str | None:
    stem = os.path.splitext(os.path.basename(filename))[0].lower()
    key = stem.split("_")[0].split("-")[0]
    try:
        return resolve_pose(key)
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default="test_images")
    ap.add_argument("--pose", default="auto", help="force this pose instead of auto-detect")
    args = ap.parse_args()
    forced = resolve_pose(args.pose)

    files = sorted(f for f in os.listdir(args.folder) if f.lower().endswith(EXTS))
    if not files:
        sys.exit(f"No images found in {args.folder}")
    out_dir = os.path.join(args.folder, "out")
    os.makedirs(out_dir, exist_ok=True)

    hits = total_expected = 0
    with PoseEstimator(static=True) as est:
        for f in files:
            path = os.path.join(args.folder, f)
            image = load(path)
            result = analyze(image, forced, est)
            expected = expected_from_name(f)
            print(f"\n=== {f}" + (f"   (expected: {expected})" if expected else ""))

            if result is None:
                print("  no person detected")
                continue
            if result["evaluation"] is None:
                print("  person found, but too few joints visible to judge")
                cv2.imwrite(os.path.join(out_dir, f), cv2.cvtColor(draw_overlay(image, result), cv2.COLOR_RGB2BGR))
                continue

            ev = result["evaluation"]
            mark = ""
            if expected and forced == AUTO:
                total_expected += 1
                ok = result["pose"] == expected
                hits += ok
                mark = "  OK" if ok else "  MISMATCH"
            print(f"  detected: {result['pose']}  score {ev['overall']:.0f}%  "
                  f"({ev['n_visible']}/{ev['n_total']} joints visible){mark}")
            if forced == AUTO and not result["confident"]:
                print("  (low confidence - few joints visible or close runner-up)")
            if forced == AUTO and len(result["ranking"]) > 1:
                print("  runners-up: " + ", ".join(f"{n} {e['overall']:.0f}%" for n, e in result["ranking"][1:4]))
            for line in result["feedback"]:
                print("   -", line)
            if not result["feedback"]:
                print("   - all measured joints within range")

            annotated = cv2.cvtColor(draw_overlay(image, result), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(out_dir, f), annotated)

    print(f"\nAnnotated images written to {out_dir}")
    if total_expected:
        print(f"Auto-detect matched the filename on {hits}/{total_expected} images.")


if __name__ == "__main__":
    main()
