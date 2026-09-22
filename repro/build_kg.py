"""Stage 3 - MMKG construction (paper Sec 3.3, Algorithm 1).

Serves the graph LLM through vLLM rather than Ollama - LightRAG's OpenAI backend
drops straight in, and continuous batching is what makes the stage fit in a day.

Qwen2.5-32B-Instruct-AWQ builds the graph instead of the paper's DeepSeek-R1-70B:
the repo README recommends Qwen2.5, the paper flags R1 as anomalous (Sec 4.2), and
R1's <think> block corrupts LightRAG's delimiter-based entity parsing. <think> is
stripped anyway, so an R1 run stays possible via --llm-model.

Documents are inserted as a list, not one huge string as upstream does: LightRAG
checkpoints per document, so --requeue resumes instead of losing the whole run.

Usage:
    python repro/build_kg.py --mode text_image --working-dir outputs/kg_text_image \
        --base-url http://127.0.0.1:8000/v1
"""

import argparse
import asyncio
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.environ.get("VALIK_ROOT", ".."), "src", "LightRAG"))

from common import (  # noqa: E402
    hf_embed,
    limited_pids,
    load_problems,
    load_splits,
    problem_to_kb_text,
    read_caption,
)

THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def build_documents(mode, split, caption_suffix, limit=0):
    problems, pid_splits = load_problems(), load_splits()
    pids = limited_pids(pid_splits, split, limit)
    docs = []

    if mode in ("text_only", "text_image"):
        for pid in pids:
            p = problems.get(pid)
            if p:
                docs.append(problem_to_kb_text(p))

    if mode in ("image_only", "text_image"):
        n_missing = 0
        for pid in pids:
            p = problems.get(pid)
            if not p or not p.get("image"):
                continue
            cap = read_caption(pid, split, caption_suffix)
            if cap:
                docs.append(cap)
            else:
                n_missing += 1
        if n_missing:
            print(f"WARNING: {n_missing} images have no {caption_suffix} description", flush=True)

    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["text_only", "image_only", "text_image"])
    ap.add_argument("--working-dir", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--llm-model", default="Qwen/Qwen2.5-32B-Instruct-AWQ")
    ap.add_argument("--embed-model", default="nomic-ai/nomic-embed-text-v1.5")
    ap.add_argument("--embed-dim", type=int, default=768)
    ap.add_argument("--max-async", type=int, default=32,
                    help="upstream demo uses 160; vLLM's scheduler preempts badly above ~64")
    ap.add_argument("--max-token-size", type=int, default=32768)
    ap.add_argument("--split", default="train")
    ap.add_argument("--caption-suffix", default=".pruned.txt")
    ap.add_argument("--limit", type=int, default=0,
                    help="first N train pids (0 = all); must match caption.py's --limit")
    args = ap.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    from lightrag import LightRAG
    from lightrag.lightrag import always_get_an_event_loop
    from lightrag.llm.openai import openai_complete_if_cache
    from lightrag.utils import EmbeddingFunc

    async def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        # LightRAG passes response_format="json" as a bare string for keyword
        # extraction; vLLM wants {"type": "json_object"} and 400s on the string.
        if kwargs.pop("keyword_extraction", False):
            kwargs["response_format"] = {"type": "json_object"}
        model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
        out = await openai_complete_if_cache(
            model_name,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            **kwargs,
        )
        return THINK.sub("", out).strip() if isinstance(out, str) else out

    tokenizer = AutoTokenizer.from_pretrained(args.embed_model, trust_remote_code=True)
    embed_model = AutoModel.from_pretrained(args.embed_model, trust_remote_code=True)
    embed_model = embed_model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    os.makedirs(args.working_dir, exist_ok=True)
    rag = LightRAG(
        working_dir=args.working_dir,
        llm_model_func=llm_model_func,
        llm_model_name=args.llm_model,
        llm_model_max_async=args.max_async,
        llm_model_max_token_size=args.max_token_size,
        llm_model_kwargs={"base_url": args.base_url, "api_key": "EMPTY"},
        embedding_func=EmbeddingFunc(
            embedding_dim=args.embed_dim,
            max_token_size=8192,
            func=lambda texts: hf_embed(texts, tokenizer, embed_model),
        ),
    )

    docs = build_documents(args.mode, args.split, args.caption_suffix, args.limit)
    if not docs:
        print(
            f"FATAL: no documents for mode={args.mode}. The caption stage produced "
            f"nothing - run  python repro/check_data.py  first.",
            flush=True,
        )
        sys.exit(1)
    chars = sum(len(d) for d in docs)
    print(
        f"mode={args.mode} split={args.split}: {len(docs)} documents, "
        f"{chars / 1e6:.2f}M chars (~{chars / 4e6:.2f}M tokens)",
        flush=True,
    )

    # Documents already in doc_status are skipped, so a requeued job resumes.
    rag.insert(docs)

    # ainsert logs a failed document and moves on, so without this the stage exits 0
    # having built nothing and .build_done makes the next run skip it.
    counts = always_get_an_event_loop().run_until_complete(
        rag.doc_status.get_status_counts()
    )
    done, failed = counts.get("processed", 0), counts.get("failed", 0)
    print(f"doc_status: {counts}", flush=True)
    if failed:
        print(f"FATAL: {failed} of {done + failed} documents failed to insert; see the "
              "ERROR:lightrag lines above for the first cause.", flush=True)
        sys.exit(1)

    total = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fs in os.walk(args.working_dir)
        for f in fs
    )
    print(f"done. {args.working_dir} = {total / 1e6:.0f} MB "
          f"(paper Sec 4.2 reports 489 MB for the full ScienceQA MMKG)", flush=True)


if __name__ == "__main__":
    main()
