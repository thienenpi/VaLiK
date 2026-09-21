"""Evaluation harness for paper Table 3 (ScienceQA).

The upstream repo ships no evaluation code, so the prompt format and answer parsing
here are this reproduction's own reading of Sec 4.1, not the authors'. Judge the
ordering and the gaps, not the absolute numbers.

Which description each row sees:
  --mode nokg + --caption-source sqa    ScienceQA's own captions.json, not ours -
      feeding the baseline our captions would shrink VaLiK's reported gain.
  --mode kg   + --caption-source valik  the MMKG is built from train, so a test
      image reaches the model only through its own pruned description.

Results stream to <out>.jsonl and finished pids are skipped, so this is
--requeue safe.

Usage:
    python repro/eval_scienceqa.py --mode kg --working-dir outputs/kg_text_image \
        --base-url http://127.0.0.1:8000/v1 --out outputs/results/text_image
"""

import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.environ.get("VALIK_ROOT", ".."), "src", "LightRAG"))

from common import (  # noqa: E402
    accuracy_table,
    categorize,
    format_question,
    format_table,
    load_problems,
    load_sqa_captions,
    load_splits,
    parse_choice,
    read_caption,
)

SYSTEM = (
    "You are answering multiple-choice science exam questions. "
    "Reply with the single letter of the correct option and nothing else."
)
RESPONSE_TYPE = "a single letter (A, B, C, D or E) and nothing else"


def caption_for(problem, pid, split, source, suffix, sqa_captions):
    if not problem.get("image") or source == "none":
        return None
    if source == "sqa":
        return sqa_captions.get(pid)
    return read_caption(pid, split, suffix)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["nokg", "kg"])
    ap.add_argument("--working-dir", help="LightRAG working dir (required for --mode kg)")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--llm-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--embed-model", default="nomic-ai/nomic-embed-text-v1.5")
    ap.add_argument("--embed-dim", type=int, default=768)
    ap.add_argument("--query-mode", default="hybrid",
                    help="paper Appendix D: hybrid balances local and global retrieval")
    ap.add_argument("--caption-source", choices=["sqa", "valik", "none"], default=None)
    ap.add_argument("--caption-suffix", default=".pruned.txt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = full split; use e.g. 500 for a smoke test")
    ap.add_argument("--out", required=True, help="output prefix; writes .jsonl and .json")
    args = ap.parse_args()

    if args.caption_source is None:
        args.caption_source = "sqa" if args.mode == "nokg" else "valik"
    if args.mode == "kg" and not args.working_dir:
        ap.error("--working-dir is required for --mode kg")

    problems, pid_splits = load_problems(), load_splits()
    pids = pid_splits[args.split]
    if args.limit:
        pids = pids[: args.limit]

    sqa_captions = load_sqa_captions()
    if args.caption_source == "sqa" and not sqa_captions:
        print("WARNING: captions.json not found; IMG questions get no description", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    jsonl_path = args.out + ".jsonl"
    done = {}
    if os.path.exists(jsonl_path):
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done[r["pid"]] = r
                except json.JSONDecodeError:
                    pass
        print(f"resuming: {len(done)} answers already on disk", flush=True)

    todo = [p for p in pids if p not in done]
    print(f"{args.mode}/{args.caption_source}: {len(todo)} of {len(pids)} to answer", flush=True)

    # ------------------------------------------------------------------ backend
    if args.mode == "nokg":
        from openai import AsyncOpenAI

        client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")

        async def ask(question):
            r = await client.chat.completions.create(
                model=args.llm_model,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": question},
                ],
                temperature=0.0,
                max_tokens=16,
            )
            return r.choices[0].message.content
    else:
        import torch
        from transformers import AutoModel, AutoTokenizer

        from lightrag import LightRAG, QueryParam
        from lightrag.llm.hf import hf_embed
        from lightrag.llm.openai import openai_complete_if_cache
        from lightrag.utils import EmbeddingFunc

        async def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
            # See repro/build_kg.py: bare "json" is rejected by vLLM with a 400.
            if kwargs.pop("keyword_extraction", False):
                kwargs["response_format"] = {"type": "json_object"}
            model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
            return await openai_complete_if_cache(
                model_name,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                **kwargs,
            )

        tok = AutoTokenizer.from_pretrained(args.embed_model, trust_remote_code=True)
        emb = AutoModel.from_pretrained(args.embed_model, trust_remote_code=True)
        emb = emb.to("cuda" if torch.cuda.is_available() else "cpu").eval()

        rag = LightRAG(
            working_dir=args.working_dir,
            llm_model_func=llm_model_func,
            llm_model_name=args.llm_model,
            llm_model_max_async=args.concurrency,
            llm_model_kwargs={"base_url": args.base_url, "api_key": "EMPTY"},
            embedding_func=EmbeddingFunc(
                embedding_dim=args.embed_dim,
                max_token_size=8192,
                func=lambda texts: hf_embed(texts, tok, emb),
            ),
        )
        param = QueryParam(mode=args.query_mode, response_type=RESPONSE_TYPE)

        async def ask(question):
            return await rag.aquery(question, param=param)

    # ---------------------------------------------------------------------- run
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    out_f = open(jsonl_path, "a", encoding="utf-8")
    counter = {"n": 0, "err": 0}
    t0 = time.time()

    async def one(pid):
        p = problems[pid]
        cap = caption_for(p, pid, args.split, args.caption_source,
                          args.caption_suffix, sqa_captions)
        question = format_question(p, cap)
        raw = ""
        async with sem:
            try:
                raw = await ask(question)
            except Exception as e:
                counter["err"] += 1
                if counter["err"] <= 10:
                    print(f"ERR {pid}: {e}", flush=True)
        pred = parse_choice(raw, len(p["choices"]))
        rec = {
            "pid": pid,
            "pred": pred,
            "gold": p["answer"],
            "correct": pred == p["answer"],
            "parsed": pred is not None,
            "raw": (raw or "")[:400],
            **categorize(p),
        }
        async with lock:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            counter["n"] += 1
            if counter["n"] % 200 == 0:
                rate = counter["n"] / (time.time() - t0)
                print(f"  {counter['n']}/{len(todo)}  {rate:.1f} q/s  "
                      f"ETA {(len(todo) - counter['n']) / rate / 60:.0f} min", flush=True)
        return rec

    results = list(done.values())
    if todo:
        results += await asyncio.gather(*(one(p) for p in todo))
    out_f.close()

    # ------------------------------------------------------------------- report
    n_parsed = sum(r["parsed"] for r in results)
    cols = accuracy_table(results)
    title = (f"{args.mode} / KG={args.working_dir or '-'} / "
             f"caption={args.caption_source} / n={len(results)}")
    print()
    print(format_table(cols, title), flush=True)
    print(f"parse rate: {100.0 * n_parsed / max(len(results), 1):.2f}% "
          f"(below ~95% means the prompt is wrong, not the KG)", flush=True)

    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": vars(args),
                "n": len(results),
                "parse_rate": 100.0 * n_parsed / max(len(results), 1),
                "accuracy": cols,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    asyncio.run(main())
