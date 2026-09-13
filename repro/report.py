"""Collect outputs/results/*.json into a Table 3 comparison.

Reference numbers are transcribed from the paper's Table 3 (arXiv:2503.12972v3,
page 7). The reproduction is judged on the ordering and the gaps, not the decimals:
dropping the BLIP-2/LLaVA cascade and the 70B graph model should cost 1-3 points.

Usage: python repro/report.py [results_dir]
"""

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CONTEXTS, GRADES, SUBJECTS  # noqa: E402

ORDER = SUBJECTS + CONTEXTS + GRADES + ["Average"]

PAPER = {
    "nokg": ("Qwen2.5-7B",
             [76.20, 67.83, 77.27, 74.49, 65.79, 79.02, 77.72, 69.35, 74.72]),
    "image_only": ("Qwen2.5-7B (VaLiK Image-only)",
                   [79.14, 71.54, 79.27, 77.16, 69.72, 83.14, 80.65, 73.96, 78.88]),
    "text_image": ("Qwen2.5-7B (VaLiK Text-Image)",
                   [84.15, 75.14, 87.64, 82.99, 73.18, 89.69, 84.40, 80.95, 83.16]),
    "text_only": ("Qwen2.5-7B (VaLiK Text-only)",
                  [84.54, 74.24, 86.91, 82.74, 72.53, 90.03, 84.51, 80.28, 82.98]),
}
QWEN72B_AVG = 78.37  # Table 3, native Qwen2.5-72B - the bar the 7B+VaLiK row clears


def row(label, values, extra=""):
    return f"{label:<34}" + "".join(f"{v:>8.2f}" for v in values) + extra


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs/results"
    found = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        name = os.path.basename(path).rsplit("-", 1)[0]
        found[name] = data

    if not found:
        print(f"no results in {results_dir}/ yet")
        return

    print(f"{'':34}" + "".join(f"{c:>8}" for c in ORDER))
    print("-" * (34 + 8 * len(ORDER)))

    for key in ("nokg", "image_only", "text_only", "text_image"):
        if key not in found:
            continue
        data = found[key]
        mine = [data["accuracy"][c] for c in ORDER]
        label, paper = PAPER[key]
        print(row("paper  " + label, paper))
        print(row("ours   " + label, mine,
                  f"   n={data['n']}  parse={data['parse_rate']:.1f}%"))
        print(row("delta  ", [m - p for m, p in zip(mine, paper)]))
        print()

    # The claims worth checking, in the paper's own terms.
    avgs = {k: found[k]["accuracy"]["Average"] for k in found}
    print("claim checks")
    if "nokg" in avgs and "image_only" in avgs:
        d = avgs["image_only"] - avgs["nokg"]
        print(f"  vision alone helps            {d:+6.2f}  (paper +4.16)  "
              f"{'OK' if d > 0 else 'FAILED'}")
    if "nokg" in avgs and "text_image" in avgs:
        d = avgs["text_image"] - avgs["nokg"]
        print(f"  full VaLiK helps              {d:+6.2f}  (paper +8.44)  "
              f"{'OK' if d > 0 else 'FAILED'}")
    if "image_only" in avgs and "text_image" in avgs:
        d = avgs["text_image"] - avgs["image_only"]
        print(f"  text+image beats image-only   {d:+6.2f}  (paper +4.28)  "
              f"{'OK' if d > 0 else 'FAILED'}")
    if "text_image" in avgs:
        d = avgs["text_image"] - QWEN72B_AVG
        print(f"  7B+VaLiK beats native 72B     {d:+6.2f}  (paper +4.79)  "
              f"{'OK' if d > 0 else 'FAILED'}")

    for key, data in found.items():
        if data["parse_rate"] < 95:
            print(f"  WARNING {key}: parse rate {data['parse_rate']:.1f}% - "
                  f"fix the prompt before trusting this row")


if __name__ == "__main__":
    main()
