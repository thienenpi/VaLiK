"""Stage 3 - MMKG construction (paper Sec 3.3, Algorithm 1).

Serves the graph-construction LLM through vLLM instead of Ollama. Same models, but
Ollama has no continuous batching, which is why the paper's Appendix F throughput
(196k tokens/hour) is an order of magnitude below what one A100 can actually do.
LightRAG already ships an OpenAI-compatible backend (lightrag/llm/openai.py), so
vLLM drops in with no change to the library.

Deviation from paper Sec 4.1: Qwen2.5-32B-Instruct-AWQ builds the graph, not
DeepSeek-R1-70B. Grounds:
  * the repo README (line 84) recommends Qwen2.5 "for its balance of efficiency
    and effectiveness";
  * the paper itself reports R1 behaving anomalously (Sec 4.2: "its reasoning
    process may introduce complex information that interferes with its judgment") -
    LightRAG parses entities off delimiters, and R1's <think> block corrupts that;
  * Appendix D measures only a 1-5% spread across retrieval model sizes.
We strip <think> blocks anyway so an R1 run stays possible via --llm-model.

Resumability: upstream src/LightRAG/lightrag_ollama_demo.py calls
rag.insert(one_huge_string). LightRAG checkpoints at *document* granularity
(lightrag.py:373-390 filters on doc_status, and _insert_done() runs after each
document), so a single document means a preempted job loses everything. We insert a
list - one document per training problem, one per image description - which makes
--requeue safe and costs nothing.

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
    from lightrag.llm.hf import hf_embed
    from lightrag.llm.openai import openai_complete_if_cache
    from lightrag.utils import EmbeddingFunc

    async def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        # LightRAG's own openai_complete() sets response_format="json" (a bare
        # string) for keyword extraction; the OpenAI schema - and therefore vLLM -
        # wants {"type": "json_object"}, and rejects the string with a 400.
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
    chars = sum(len(d) for d in docs)
    print(
        f"mode={args.mode} split={args.split}: {len(docs)} documents, "
        f"{chars / 1e6:.2f}M chars (~{chars / 4e6:.2f}M tokens)",
        flush=True,
    )

    # Documents already present in doc_status are skipped inside ainsert, so a
    # requeued job picks up where the previous attempt stopped.
    rag.insert(docs)

    total = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fs in os.walk(args.working_dir)
        for f in fs
    )
    print(f"done. {args.working_dir} = {total / 1e6:.0f} MB "
          f"(paper Sec 4.2 reports 489 MB for the full ScienceQA MMKG)", flush=True)


if __name__ == "__main__":
    main()
