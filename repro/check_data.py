"""Diagnose why a stage found no images.

caption.py and prune.py walk images/<split>/<pid>/image.{png,jpg,jpeg}, the layout
the upstream CLIP_Interrogator_ScienceQA.py assumes and the one the repo README
documents (datasets/ScienceQA/data/scienceqa/images/train/1/image.png). If
ScienceQA's tools/download.sh laid the images out differently - or silently failed,
which its Google Drive fetches do - every stage reports "nothing to do" and the
chain happily builds an empty KG.

Usage: python repro/check_data.py [limit]
"""

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    IMAGES_DIR,
    SQA_DATA,
    SQA_ROOT,
    VALIK_ROOT,
    find_image,
    limited_pids,
    load_problems,
    load_sqa_captions,
    load_splits,
)


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    print(f"VALIK_ROOT = {VALIK_ROOT}")
    print(f"SQA_ROOT   = {SQA_ROOT}   exists={os.path.isdir(SQA_ROOT)}")
    print(f"IMAGES_DIR = {IMAGES_DIR}   exists={os.path.isdir(IMAGES_DIR)}")
    print()

    if not os.path.isdir(SQA_ROOT):
        print("FATAL: SQA_ROOT missing. What is actually under datasets/?")
        for root, dirs, files in os.walk(os.path.join(VALIK_ROOT, "datasets")):
            depth = root[len(VALIK_ROOT):].count(os.sep)
            if depth > 4:
                dirs[:] = []
                continue
            print(f"  {root}  ({len(dirs)} dirs, {len(files)} files)")
        return 1

    print("--- json files ---")
    # captions.json sits in data/, the other two in data/scienceqa/.
    for name, root in (("problems.json", SQA_ROOT),
                       ("pid_splits.json", SQA_ROOT),
                       ("captions.json", SQA_DATA)):
        fp = os.path.join(root, name)
        if os.path.exists(fp):
            print(f"  {name:18} OK       {os.path.getsize(fp) / 1e6:.1f} MB   {fp}")
        else:
            print(f"  {name:18} MISSING  (looked in {root})")
    print()

    problems, pid_splits = load_problems(), load_splits()
    print(f"problems: {len(problems)}   captions.json entries: {len(load_sqa_captions())}")
    print()

    print("--- what is directly under IMAGES_DIR ---")
    if os.path.isdir(IMAGES_DIR):
        entries = sorted(os.listdir(IMAGES_DIR))[:10]
        print(f"  {len(os.listdir(IMAGES_DIR))} entries, first few: {entries}")
        # How deep do the actual image files sit?
        for pattern in ("*.png", "*/*.png", "*/*/*.png", "*/*/*/*.png"):
            hits = glob.glob(os.path.join(IMAGES_DIR, pattern))
            print(f"  {pattern:14} -> {len(hits)} files"
                  + (f"   e.g. {hits[0]}" if hits else ""))
    else:
        print("  IMAGES_DIR does not exist")
    print()

    print("--- per split ---")
    total_expected = total_found = 0
    for split in ("train", "val", "test"):
        pids = limited_pids(pid_splits, split, limit)
        with_img = [p for p in pids if problems.get(p, {}).get("image")]
        found = [p for p in with_img if find_image(p, split)]
        total_expected += len(with_img)
        total_found += len(found)
        print(f"  {split:5} {len(pids):6} pids, {len(with_img):6} say they have an image, "
              f"{len(found):6} found on disk")
        missing = [p for p in with_img if not find_image(p, split)]
        if missing:
            pid = missing[0]
            d = os.path.join(IMAGES_DIR, split, str(pid))
            print(f"        first missing pid={pid}: problems.json says "
                  f"image={problems[pid]['image']!r}")
            print(f"        expected dir {d}  exists={os.path.isdir(d)}")
            if os.path.isdir(d):
                print(f"        dir contains: {os.listdir(d)}")
    print()

    if total_found == 0:
        print("FATAL: no images found. caption.py will report 'nothing to do' and the")
        print("       chain will build an empty KG. Fix the dataset layout first -")
        print("       compare the 'e.g.' path above with common.find_image().")
        return 1
    print(f"OK: {total_found}/{total_expected} images present"
          + (f"  (limit={limit})" if limit else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
