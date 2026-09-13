"""Stage 1 - CoE-based Visual to Language Modeling (paper Sec 3.1), minimal variant.

The paper cascades BLIP-2 -> LLaVA -> Qwen2-VL (Eq. 2). We run Qwen2-VL-7B alone,
which the authors explicitly sanction in Appendix G: "using a single, strong VLM can
achieve performance comparable to a cascade of smaller, lightweight models". That
cuts the stage from ~16h to ~2.5h wall-clock at an expected cost of ~1 point.

Two departures from src/Image_to_Text/Qwen2VL_ScienceQA.py, both deliberate:
  * max_new_tokens 384, not 32768. The upstream value makes every image pay for a
    32k-token generation budget; the paper's own throughput figure (Appendix F,
    "60 tokens per second, one image every 4 seconds") implies ~240 tokens.
  * sharded by array task with an atomic write, so N array tasks cannot half-write
    the same file. Upstream relies only on "skip if .txt exists", which races.

Usage:
    python repro/caption.py --shard-id 0 --num-shards 4 --splits train test
"""

import argparse
import os
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import iter_split_images  # noqa: E402

PROMPT = """
Please provide a detailed visual description of this image.
Include key objects, their spatial relationships,
notable visual features, and any observable actions or events.
Respond in clear, structured English paragraphs.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--suffix", default=".txt")
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--limit", type=int, default=0,
                    help="first N pids per split (0 = all); smoke runs slice the same prefix everywhere")
    args = ap.parse_args()

    from transformers import AutoProcessor, AutoModelForImageTextToText
    from qwen_vl_utils import process_vision_info

    targets = list(iter_split_images(args.splits, args.limit))
    mine = targets[args.shard_id :: args.num_shards]
    todo = [
        (s, p, img)
        for s, p, img in mine
        if not os.path.exists(os.path.join(os.path.dirname(img), "image" + args.suffix))
    ]
    print(
        f"[shard {args.shard_id}/{args.num_shards}] {len(mine)} assigned, "
        f"{len(todo)} remaining of {len(targets)} total",
        flush=True,
    )
    if not todo:
        print("nothing to do", flush=True)
        return

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda")
    model.eval()

    t0 = time.time()
    for i, (split, pid, img_path) in enumerate(todo):
        out = os.path.join(os.path.dirname(img_path), "image" + args.suffix)
        if os.path.exists(out):  # another shard may have raced us; harmless
            continue
        try:
            image = Image.open(img_path).convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image", "image": image},
                    ],
                }
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")

            with torch.inference_mode():
                generated = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
                )
            trimmed = [
                o[len(j) :] for j, o in zip(inputs.input_ids, generated)
            ]
            desc = processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()

            tmp = f"{out}.tmp{args.shard_id}"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(f"[Description]\n{desc}\n")
            os.replace(tmp, out)
        except Exception as e:  # one bad image must not kill a 2h shard
            print(f"ERR {split}/{pid}: {e}", flush=True)

        if (i + 1) % 50 == 0:
            rate = (i + 1) / (time.time() - t0)
            eta = (len(todo) - i - 1) / rate / 60
            print(
                f"  {i + 1}/{len(todo)}  {rate:.2f} img/s  ETA {eta:.0f} min",
                flush=True,
            )

    print(f"shard {args.shard_id} done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
