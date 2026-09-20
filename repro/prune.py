"""Stage 2 - Cross-Modal Similarity Verification (paper Sec 3.2).

Sentence-level sliding window, CLIP-ViT-L/14 cosine, tau = 0.20 for ScienceQA
(paper Sec 4.1). Sharded so it fits the 1-GPU-per-job budget; CLIP is cheap, the
whole stage is ~15 min.

Two fixes over src/Prune/similarity_verification.py:
  * truncation=True. Upstream calls the processor with padding=True but no
    truncation, and CLIP's text encoder caps at 77 tokens - a single long sentence
    from Qwen2-VL blows up the whole batch. The file's own comment on line 81
    acknowledges the limit without handling it.
  * empty-result guard. When every sentence scores below tau the upstream code
    writes an empty file; for image-only KGs that silently deletes the image from
    the knowledge base. We keep the original text and count it instead, so the
    over-pruning shows up in the log rather than as a missing node.

Table 4 of the paper reports SV *hurting* ScienceQA image-only (NAT -0.92,
LAN -1.28). Reproducing that dip is a signal the implementation is faithful, not
a bug to tune away.

Usage:
    python repro/prune.py --shard-id 0 --num-shards 4 --threshold 0.20
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import iter_split_images, read_caption  # noqa: E402


def chunk_text(text, mode, window_size):
    if mode == "sentence":
        from nltk.tokenize import sent_tokenize

        return sent_tokenize(text)
    words = text.split()
    if mode == "word":
        return words
    return [" ".join(words[i : i + window_size]) for i in range(0, len(words), window_size)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--threshold", type=float, default=0.20)
    ap.add_argument("--mode", choices=["word", "sentence", "window"], default="sentence")
    ap.add_argument("--window-size", type=int, default=5)
    ap.add_argument("--in-suffix", default=".txt")
    ap.add_argument("--out-suffix", default=".pruned.txt")
    ap.add_argument("--batch", type=int, default=64, help="max chunks scored per forward")
    ap.add_argument("--limit", type=int, default=0,
                    help="first N pids per split (0 = all); must match caption.py's --limit")
    args = ap.parse_args()

    if args.mode == "sentence":
        import nltk

        for pkg in ("punkt", "punkt_tab"):
            try:
                nltk.data.find(f"tokenizers/{pkg}")
            except LookupError:
                nltk.download(pkg, quiet=True)

    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    targets = list(iter_split_images(args.splits, args.limit))
    mine = targets[args.shard_id :: args.num_shards]
    todo = [
        (s, p, img)
        for s, p, img in mine
        if not os.path.exists(os.path.join(os.path.dirname(img), "image" + args.out_suffix))
    ]
    print(
        f"[shard {args.shard_id}/{args.num_shards}] {len(mine)} assigned, {len(todo)} remaining",
        flush=True,
    )
    if not targets:
        print(
            "FATAL: no images matched. Run  python repro/check_data.py  to see why.",
            flush=True,
        )
        sys.exit(1)
    if not todo:
        print("nothing to do - every assigned image is already pruned", flush=True)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        # Not fatal - CLIP on CPU is slow but finishes - yet it is never what the
        # job asked for, so say so instead of letting a driver problem look like a
        # successful run. jobs/20_prune.slurm calls require_cuda before us.
        print("WARNING: no usable GPU, scoring on CPU", flush=True)
    model = CLIPModel.from_pretrained(args.model).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.model)

    n_kept = n_total = n_empty = n_done = n_missing = 0
    for split, pid, img_path in todo:
        out = os.path.join(os.path.dirname(img_path), "image" + args.out_suffix)
        text = read_caption(pid, split, args.in_suffix)
        if not text:
            n_missing += 1
            continue
        try:
            chunks = [c.strip() for c in chunk_text(text, args.mode, args.window_size) if c.strip()]
            if not chunks:
                continue
            image = Image.open(img_path).convert("RGB")

            sims = []
            for i in range(0, len(chunks), args.batch):
                batch = chunks[i : i + args.batch]
                inputs = processor(
                    text=batch,
                    images=image,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,  # CLIP text encoder caps at 77 tokens
                ).to(device)
                with torch.inference_mode():
                    o = model(**inputs)
                # HF returns L2-normalised projections, so this dot product is the
                # cosine of paper Eq. 7.
                sims.extend((o.image_embeds @ o.text_embeds.T).squeeze(0).float().cpu().tolist())

            kept = [c for c, s in zip(chunks, sims) if s >= args.threshold]
            n_total += len(chunks)
            n_kept += len(kept)
            if not kept:
                n_empty += 1
                kept = chunks  # never hand an empty description to the KG builder

            tmp = f"{out}.tmp{args.shard_id}"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(" ".join(kept))
            os.replace(tmp, out)
            n_done += 1
        except Exception as e:
            print(f"ERR {split}/{pid}: {e}", flush=True)

        if n_done % 200 == 0 and n_done:
            print(f"  {n_done}/{len(todo)}  kept {100.0 * n_kept / max(n_total, 1):.1f}%", flush=True)

    print(
        f"shard {args.shard_id}: {n_done} files, kept {n_kept}/{n_total} "
        f"({100.0 * n_kept / max(n_total, 1):.1f}%) sentences, "
        f"{n_empty} fully-pruned files restored, "
        f"{n_missing} skipped for a missing image{args.in_suffix}",
        flush=True,
    )

    # A shard that wrote nothing because stage 1 left no captions is a failed shard,
    # not an empty one. Exiting 0 here is what let the chain carry on after both
    # caption shards died on the driver: prune logged "0 files, kept 0/0 (0.0%)",
    # afterok was satisfied, and the KG build went looking for descriptions that were
    # never written.
    if n_done == 0:
        print(
            f"FATAL: shard {args.shard_id} wrote no image{args.out_suffix} at all "
            f"({n_missing} of {len(todo)} assigned images had no "
            f"image{args.in_suffix}).\n"
            "       Stage 1 (jobs/10_caption.slurm) has not run or did not finish; "
            "check its log before re-running this one.",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
