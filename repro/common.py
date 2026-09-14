"""Shared helpers for the VaLiK ScienceQA reproduction.

Everything here is read-only w.r.t. the upstream repo: the original scripts under src/
are left untouched so a diff against Wings-Of-Disaster/VaLiK stays clean.

Category definitions follow the ScienceQA convention used by paper Table 3:
  subject  NAT / SOC / LAN
  context  TXT (hint, no image) / IMG (has image) / NO (neither)
  grade    G1-6 / G7-12
"""

import json
import os
import re

VALIK_ROOT = os.environ.get(
    "VALIK_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
SQA_DATA = os.path.join(VALIK_ROOT, "datasets", "ScienceQA", "data")
SQA_ROOT = os.path.join(SQA_DATA, "scienceqa")
IMAGES_DIR = os.path.join(SQA_ROOT, "images")

LETTERS = "ABCDE"
SUBJECT_MAP = {
    "natural science": "NAT",
    "social science": "SOC",
    "language science": "LAN",
}
SUBJECTS = ["NAT", "SOC", "LAN"]
CONTEXTS = ["TXT", "IMG", "NO"]
GRADES = ["G1-6", "G7-12"]

# Header written by repro/caption.py (and by the upstream Image_to_Text scripts).
_DESC_HEADER = re.compile(r"^\s*\[Description\]\s*", re.IGNORECASE)


# --------------------------------------------------------------------------- data

def load_problems():
    with open(os.path.join(SQA_ROOT, "problems.json"), encoding="utf-8") as f:
        return json.load(f)


def load_splits():
    with open(os.path.join(SQA_ROOT, "pid_splits.json"), encoding="utf-8") as f:
        return json.load(f)


def load_sqa_captions():
    """ScienceQA ships its own generated captions; the paper's no-KG baseline row
    (Table 3, 'Qwen2.5-7B') scores 65.79 on IMG, far above chance, so the text-only
    baseline must be fed *some* description. Using ScienceQA's own captions keeps
    the baseline honest - feeding it our Qwen2-VL captions would inflate it and
    shrink VaLiK's reported gain.

    Upstream ships this at data/captions.json - one level ABOVE data/scienceqa/,
    where problems.json and pid_splits.json live. Both locations are checked.
    """
    for path in (os.path.join(SQA_DATA, "captions.json"),
                 os.path.join(SQA_ROOT, "captions.json")):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return data.get("captions", data)
    return {}


def image_dir(pid, split):
    return os.path.join(IMAGES_DIR, split, str(pid))


def find_image(pid, split):
    d = image_dir(pid, split)
    for name in ("image.png", "image.jpg", "image.jpeg"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def caption_path(pid, split, suffix=".txt"):
    return os.path.join(image_dir(pid, split), "image" + suffix)


def read_caption(pid, split, suffix=".txt"):
    p = caption_path(pid, split, suffix)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        text = f.read()
    return _DESC_HEADER.sub("", text).strip() or None


def limited_pids(pid_splits, split, limit=0):
    """First `limit` pids of a split (0 = all).

    Every stage slices the *same* prefix of pid_splits.json, so a smoke run
    captions, prunes, builds and evaluates one consistent subset instead of four
    unrelated ones.
    """
    pids = pid_splits.get(split, [])
    return pids[:limit] if limit else pids


def iter_split_images(splits, limit=0):
    """Yield (split, pid, image_path) for every problem in `splits` that has an image."""
    problems, pid_splits = load_problems(), load_splits()
    for split in splits:
        for pid in limited_pids(pid_splits, split, limit):
            p = problems.get(pid)
            if not p or not p.get("image"):
                continue
            img = find_image(pid, split)
            if img:
                yield split, pid, img


# ----------------------------------------------------------------------- grouping

def categorize(problem):
    grade_n = int(re.sub(r"\D", "", problem.get("grade", "grade1")) or 1)
    return {
        "subject": SUBJECT_MAP.get(problem.get("subject", ""), "NAT"),
        # IMG takes precedence: Table 3 counts a question with an image as IMG even
        # when it also carries a hint.
        "context": "IMG" if problem.get("image") else ("TXT" if problem.get("hint") else "NO"),
        "grade": "G1-6" if grade_n <= 6 else "G7-12",
    }


# ----------------------------------------------------------------------- prompting

def format_question(problem, caption=None):
    parts = []
    hint = (problem.get("hint") or "").strip()
    if hint:
        parts.append(f"Context: {hint}")
    if caption:
        parts.append(f"Image description: {caption}")
    parts.append(f"Question: {problem['question'].strip()}")
    opts = " ".join(
        f"({LETTERS[i]}) {c}" for i, c in enumerate(problem["choices"])
    )
    parts.append(f"Options: {opts}")
    parts.append(
        "Answer with the single letter of the correct option and nothing else."
    )
    return "\n".join(parts)


def problem_to_kb_text(problem):
    """One knowledge-base document per training problem.

    Mirrors src/Original_Text_Compilation/Get_Text_ScienceQA.py, which concatenates
    question / answer / hint / lecture / solution for the train split only.
    """
    choices = problem["choices"]
    answer = choices[problem["answer"]]
    return (
        f"Question: {problem['question']}\n"
        f"Answer: {answer}\n"
        f"Hint: {problem.get('hint', '')}\n"
        f"Lecture: {problem.get('lecture', '')}\n"
        f"Solution: {problem.get('solution', '')}\n"
    )


# ------------------------------------------------------------------------ parsing

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_ANSWER_KV = re.compile(
    r"(?:answer|option|choice)\s*(?:is)?\s*[:\-]?\s*\(?\s*([A-E])\s*\)?", re.IGNORECASE
)
_LEADING = re.compile(r"^\s*\(?\s*([A-E])\s*[\)\.\:,]?\s*(?:$|\s)")


def parse_choice(text, n_choices):
    """Extract the predicted option index, or None when nothing parses.

    Returning None (instead of guessing 0) keeps the parse-failure rate visible;
    a run with >5% unparsed answers means the prompt is wrong, not the KG.
    """
    if not text:
        return None
    text = _THINK.sub("", text).strip()
    valid = set(LETTERS[:n_choices])

    for pat in (_ANSWER_KV, _LEADING):
        m = pat.search(text)
        if m and m.group(1).upper() in valid:
            return LETTERS.index(m.group(1).upper())

    # Last resort: the final standalone letter mentioned anywhere.
    hits = [c for c in re.findall(r"\b([A-E])\b", text) if c in valid]
    if hits:
        return LETTERS.index(hits[-1])
    return None


# -------------------------------------------------------------------------- table

def accuracy_table(records):
    """records: list of {"correct": bool, "subject":.., "context":.., "grade":..}"""
    def acc(subset):
        return 100.0 * sum(r["correct"] for r in subset) / len(subset) if subset else float("nan")

    cols = {}
    for k in SUBJECTS:
        cols[k] = acc([r for r in records if r["subject"] == k])
    for k in CONTEXTS:
        cols[k] = acc([r for r in records if r["context"] == k])
    for k in GRADES:
        cols[k] = acc([r for r in records if r["grade"] == k])
    cols["Average"] = acc(records)
    return cols


def format_table(cols, title=""):
    order = SUBJECTS + CONTEXTS + GRADES + ["Average"]
    head = " ".join(f"{k:>8}" for k in order)
    body = " ".join(f"{cols[k]:>8.2f}" for k in order)
    return f"{title}\n{head}\n{body}"
