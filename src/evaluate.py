"""
LLM consistency evaluator.

Two evaluation tracks:
  1. PAWS track  — give LLM both sentences, ask if they are paraphrases.
                   Measure: does the LLM flip its answer between s1→s2 and s2→s1?
  2. MMLU track  — give LLM original question + each paraphrase independently.
                   Measure: does the answer stay consistent? Does accuracy drop?
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
from anthropic import Anthropic
from openai import OpenAI
from tqdm import tqdm

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

PAWS_SYSTEM = """\
You are a semantic similarity judge. Given two sentences, determine whether they \
express the same meaning (i.e., are paraphrases of each other).
Reply with exactly one word: YES or NO."""

PAWS_USER = "Sentence 1: {s1}\nSentence 2: {s2}\n\nAre these paraphrases?"

MMLU_SYSTEM = """\
You are an expert exam taker. Answer the multiple-choice question by selecting \
the single best answer. Reply with exactly one letter: A, B, C, or D."""

MMLU_USER = """\
{question}

A) {A}
B) {B}
C) {C}
D) {D}"""


def _format_mmlu_prompt(question: str, choices: list[str]) -> str:
    return MMLU_USER.format(
        question=question,
        A=choices[0], B=choices[1], C=choices[2], D=choices[3],
    )


# ---------------------------------------------------------------------------
# Model clients
# ---------------------------------------------------------------------------

def _call_anthropic(client: Anthropic, model: str, system: str, user: str) -> str:
    resp = client.messages.create(
        model=model,
        max_tokens=16,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text.strip()


def _call_openai(client: OpenAI, model: str, system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        max_tokens=16,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


def _dispatch(provider: str, client: Any, model: str, system: str, user: str) -> str:
    if provider == "anthropic":
        return _call_anthropic(client, model, system, user)
    elif provider == "openai":
        return _call_openai(client, model, system, user)
    raise ValueError(f"Unknown provider: {provider}")


# ---------------------------------------------------------------------------
# PAWS evaluation
# ---------------------------------------------------------------------------

def evaluate_paws(
    df: pd.DataFrame,
    model: str,
    provider: str = "anthropic",
    max_samples: int = 200,
    results_tag: str = "",
) -> pd.DataFrame:
    """
    For each PAWS pair, query the model twice (s1→s2 and s2→s1) and record
    whether the answer flips. Returns a DataFrame with per-pair results.
    """
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"]) if provider == "anthropic" \
        else OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    sample = df.sample(n=min(max_samples, len(df)), random_state=42).reset_index(drop=True)
    out_path = RESULTS_DIR / f"paws_{model.replace('/', '-')}_{results_tag}.jsonl"

    # Load cached results
    done_ids: set[int] = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                rec = json.loads(line)
                done_ids.add(rec["id"])

    records = []
    with open(out_path, "a") as f:
        for _, row in tqdm(sample.iterrows(), total=len(sample), desc=f"PAWS [{model}]"):
            if row["id"] in done_ids:
                continue
            try:
                ans_fwd = _dispatch(provider, client, model, PAWS_SYSTEM,
                                    PAWS_USER.format(s1=row["sentence1"], s2=row["sentence2"]))
                time.sleep(0.05)
                ans_rev = _dispatch(provider, client, model, PAWS_SYSTEM,
                                    PAWS_USER.format(s1=row["sentence2"], s2=row["sentence1"]))
                time.sleep(0.05)
            except Exception as e:
                print(f"[warn] id {row['id']}: {e}")
                time.sleep(2)
                continue

            rec = {
                "id": int(row["id"]),
                "label": int(row["label"]),
                "ans_fwd": ans_fwd.upper(),
                "ans_rev": ans_rev.upper(),
                "consistent": ans_fwd.upper() == ans_rev.upper(),
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            records.append(rec)

    # Merge with any previously cached rows
    if out_path.exists():
        all_records = []
        with open(out_path) as f:
            for line in f:
                all_records.append(json.loads(line))
        return pd.DataFrame(all_records)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# MMLU evaluation
# ---------------------------------------------------------------------------

def evaluate_mmlu(
    df: pd.DataFrame,
    model: str,
    provider: str = "anthropic",
    results_tag: str = "",
) -> pd.DataFrame:
    """
    For each question variant (original + paraphrases), query the model and
    record the predicted answer. Returns per-variant results.

    df must have columns: original_idx, subject, category, question_variant,
                          paraphrase_idx, choices, answer_idx, answer_letter
    """
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"]) if provider == "anthropic" \
        else OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    out_path = RESULTS_DIR / f"mmlu_{model.replace('/', '-')}_{results_tag}.jsonl"

    done_keys: set[tuple[int, int]] = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                rec = json.loads(line)
                done_keys.add((rec["original_idx"], rec["paraphrase_idx"]))

    records = []
    with open(out_path, "a") as f:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"MMLU [{model}]"):
            key = (int(row["original_idx"]), int(row["paraphrase_idx"]))
            if key in done_keys:
                continue
            try:
                prompt = _format_mmlu_prompt(row["question_variant"], row["choices"])
                raw = _dispatch(provider, client, model, MMLU_SYSTEM, prompt)
                # Extract single letter
                match = re.search(r"[ABCD]", raw.upper())
                pred = match.group(0) if match else "?"
                time.sleep(0.05)
            except Exception as e:
                print(f"[warn] {key}: {e}")
                time.sleep(2)
                continue

            rec = {
                "original_idx": int(row["original_idx"]),
                "subject": row["subject"],
                "category": row["category"],
                "paraphrase_idx": int(row["paraphrase_idx"]),
                "answer_letter": row["answer_letter"],
                "predicted": pred,
                "correct": pred == row["answer_letter"],
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            records.append(rec)

    if out_path.exists():
        all_records = []
        with open(out_path) as f:
            for line in f:
                all_records.append(json.loads(line))
        return pd.DataFrame(all_records)

    return pd.DataFrame(records)
