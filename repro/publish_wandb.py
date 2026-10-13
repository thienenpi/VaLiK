"""Table 3 on Weights & Biases: the side-by-side paper/ours/delta table and the claim
checks from report.py, as one "table3" run in the pipeline's W&B group.

The stages log themselves while they run (common.wandb_run); this is only the
summary, and eval_scienceqa.py calls it when the last of the three eval rows lands.
Run it by hand to re-publish, e.g. after a --from eval resubmit or a lost connection.
It reads the result files only - seconds on the login node, no GPU.

Usage: python repro/publish_wandb.py [results_dir] [variant]
"""

import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import wandb_run  # noqa: E402
from report import ORDER, PAPER, QWEN72B_AVG  # noqa: E402

# The rows submit_all.sh builds; text_only joins the table when present.
CHAIN_ROWS = ("nokg", "image_only", "text_image")
# (claim, row, baseline row or None for the native 72B, paper's gap)
CLAIMS = [
    ("vision alone helps", "image_only", "nokg", 4.16),
    ("full VaLiK helps", "text_image", "nokg", 8.44),
    ("text+image beats image-only", "text_image", "image_only", 4.28),
    ("7B+VaLiK beats native 72B", "text_image", None, 4.79),
]


def row_summary(cfg, data):
    """Per-column ours/paper/delta for one result file, as W&B summary keys."""
    out = {"n": data["n"], "parse_rate": data["parse_rate"]}
    paper = PAPER[cfg][1] if cfg in PAPER else [None] * len(ORDER)
    for col, p in zip(ORDER, paper):
        out[f"acc/{col}"] = data["accuracy"][col]
        if p is not None:
            out[f"paper/{col}"] = p
            out[f"delta/{col}"] = data["accuracy"][col] - p
    return out


def load(results_dir, variant):
    found = {}
    for path in glob.glob(os.path.join(results_dir, f"*-{variant}.json")):
        cfg = os.path.basename(path)[: -len(f"-{variant}.json")]
        if cfg in PAPER:
            with open(path, encoding="utf-8") as f:
                found[cfg] = json.load(f)
    return found


def publish_table3(results_dir, variant, require=CHAIN_ROWS):
    found = load(results_dir, variant)
    missing = [r for r in require if r not in found]
    if missing:
        print(f"table3: waiting on {', '.join(missing)}", flush=True)
        return
    try:
        import wandb
    except ImportError:
        print("table3: wandb is not installed, skipped", flush=True)
        return
    if os.environ.get("WANDB_MODE") == "disabled":
        return

    table = wandb.Table(columns=["row", "source"] + ORDER + ["n", "parse_rate"])
    for cfg in ("nokg", "image_only", "text_only", "text_image"):
        if cfg not in found:
            continue
        data = found[cfg]
        label, paper = PAPER[cfg]
        mine = [data["accuracy"][c] for c in ORDER]
        table.add_data(label, "paper", *paper, None, None)
        table.add_data(label, "ours", *mine, data["n"], data["parse_rate"])
        table.add_data(label, "delta", *[m - p for m, p in zip(mine, paper)], None, None)

    avgs = {k: d["accuracy"]["Average"] for k, d in found.items()}
    claims = wandb.Table(columns=["claim", "ours", "paper", "holds"])
    for name, a, b, paper_gap in CLAIMS:
        if a in avgs and (b is None or b in avgs):
            gap = avgs[a] - (avgs[b] if b else QWEN72B_AVG)
            claims.add_data(name, round(gap, 2), paper_gap, gap > 0)

    # Fixed id per group: a re-publish overwrites the table instead of adding a run.
    group = os.environ.get("VALIK_WANDB_GROUP", variant)
    limit = int(variant[1:]) if variant.startswith("n") else 0
    run = wandb_run("report", f"table3-{variant}", {"limit": limit, "rows": sorted(found)},
                    run_id="table3-" + re.sub(r"[^A-Za-z0-9_-]", "-", group))
    run.log({"table3": table, "claim_checks": claims})
    run.summary.update({f"avg/{k}": v for k, v in avgs.items()})
    run.finish()
    print(f"table3: published {variant} ({', '.join(sorted(found))})", flush=True)


if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs/results"
    variant = sys.argv[2] if len(sys.argv) > 2 else "full"
    publish_table3(results_dir, variant, require=())
