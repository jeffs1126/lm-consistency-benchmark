"""
Paraphrase generation for MMLU questions.

Strategy: use a small, cheap LLM call to produce N semantically equivalent
rewordings of each question while preserving answer choices and correct label.

This is separate from the consistency *evaluation* — we generate paraphrases
once, cache them, then run the benchmark models against the cached set.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from anthropic import Anthropic
from tqdm import tqdm

CACHE_DIR = Path(__file__).parent.parent / "results" / "paraphrases"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PARAPHRASE_SYSTEM = """\
You are a linguistic transformation assistant. Your task is to rewrite a given \
multiple-choice question in N distinct ways that:
1. Preserve the exact same meaning and correct answer.
2. Vary sentence structure, vocabulary, and phrasing — not just synonym swaps.
3. Keep any technical terms accurate (do not simplify domain-specific language).
4. Do NOT change the answer choices (A/B/C/D options stay identical).

Return a JSON array of strings, one per paraphrase. No other text."""


def _paraphrase_prompt(question: str, n: int) -> str:
    return f"Produce {n} paraphrases of this question:\n\n{question}"


def generate_paraphrases(
    df: pd.DataFrame,
    n_paraphrases: int = 3,
    model: str = "claude-haiku-4-5-20251001",
    cache_file: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """
    For each row in df (must have 'question' column), generate n_paraphrases
    alternative wordings. Returns a new DataFrame with columns:
        original_idx, subject, category, question (original),
        paraphrase_idx, paraphrase_text, choices, answer_idx, answer_letter
    """
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    cache_path = CACHE_DIR / (cache_file or "mmlu_paraphrases.jsonl")

    # Load existing cache to skip already-processed rows
    cached: dict[int, list[str]] = {}
    if cache_path.exists():
        with open(cache_path) as f:
            for line in f:
                rec = json.loads(line)
                cached[rec["original_idx"]] = rec["paraphrases"]

    rows_to_process = df.copy().reset_index(drop=True)
    if max_rows:
        rows_to_process = rows_to_process.head(max_rows)

    records = []

    with open(cache_path, "a") as cache_f:
        for idx, row in tqdm(rows_to_process.iterrows(), total=len(rows_to_process), desc="Generating paraphrases"):
            if idx in cached:
                paraphrases = cached[idx]
            else:
                try:
                    response = client.messages.create(
                        model=model,
                        max_tokens=1024,
                        system=PARAPHRASE_SYSTEM,
                        messages=[{"role": "user", "content": _paraphrase_prompt(row["question"], n_paraphrases)}],
                    )
                    raw = response.content[0].text.strip()
                    paraphrases = json.loads(raw)
                    if not isinstance(paraphrases, list):
                        raise ValueError("Expected JSON array")
                    paraphrases = paraphrases[:n_paraphrases]
                except Exception as e:
                    print(f"[warn] row {idx}: {e}")
                    paraphrases = []
                    time.sleep(1)

                cache_f.write(json.dumps({"original_idx": idx, "paraphrases": paraphrases}) + "\n")
                cache_f.flush()
                cached[idx] = paraphrases
                time.sleep(0.05)  # rate limit courtesy

            # Original question as paraphrase_idx=0
            for p_idx, ptext in enumerate([row["question"]] + paraphrases):
                records.append({
                    "original_idx": idx,
                    "subject": row["subject"],
                    "category": row["category"],
                    "question_variant": ptext,
                    "paraphrase_idx": p_idx,  # 0 = original
                    "choices": row["choices"],
                    "answer_idx": row["answer_idx"],
                    "answer_letter": row["answer_letter"],
                })

    return pd.DataFrame(records)
