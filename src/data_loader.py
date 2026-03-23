"""
Data loading utilities for PAWS and MMLU datasets.
Both datasets are streamed from Hugging Face — no manual download needed.
"""

from __future__ import annotations

import random
from typing import Optional
from datasets import load_dataset
import pandas as pd


# ---------------------------------------------------------------------------
# PAWS
# ---------------------------------------------------------------------------

def load_paws(split: str = "test", max_samples: Optional[int] = None) -> pd.DataFrame:
    """
    Load PAWS labeled_final split.

    Each row has:
        id, sentence1, sentence2, label  (1 = paraphrase, 0 = not)

    We keep both classes so our LLM consistency test covers:
        - True paraphrases  → LLM should answer identically
        - Adversarial pairs → LLM should (ideally) detect the semantic shift
    """
    ds = load_dataset("google-research-datasets/paws", "labeled_final", split=split)
    df = ds.to_pandas()

    if max_samples:
        df = df.sample(n=min(max_samples, len(df)), random_state=42).reset_index(drop=True)

    return df[["id", "sentence1", "sentence2", "label"]]


# ---------------------------------------------------------------------------
# MMLU
# ---------------------------------------------------------------------------

MMLU_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

# Broad category groupings for stratified analysis
MMLU_CATEGORY_MAP = {
    "STEM": [
        "abstract_algebra", "anatomy", "astronomy", "college_biology",
        "college_chemistry", "college_computer_science", "college_mathematics",
        "college_physics", "computer_security", "conceptual_physics",
        "electrical_engineering", "elementary_mathematics", "formal_logic",
        "high_school_biology", "high_school_chemistry", "high_school_computer_science",
        "high_school_mathematics", "high_school_physics", "high_school_statistics",
        "machine_learning",
    ],
    "Social Sciences": [
        "econometrics", "high_school_geography", "high_school_government_and_politics",
        "high_school_macroeconomics", "high_school_microeconomics",
        "high_school_psychology", "human_sexuality", "public_relations",
        "sociology", "us_foreign_policy",
    ],
    "Humanities": [
        "high_school_european_history", "high_school_us_history",
        "high_school_world_history", "international_law", "jurisprudence",
        "logical_fallacies", "moral_disputes", "moral_scenarios",
        "philosophy", "prehistory", "professional_law", "world_religions",
    ],
    "Professional / Applied": [
        "business_ethics", "clinical_knowledge", "college_medicine",
        "global_facts", "human_aging", "management", "marketing",
        "medical_genetics", "miscellaneous", "nutrition",
        "professional_accounting", "professional_medicine",
        "professional_psychology", "security_studies", "virology",
    ],
}

def _subject_to_category(subject: str) -> str:
    for cat, subjects in MMLU_CATEGORY_MAP.items():
        if subject in subjects:
            return cat
    return "Other"


def load_mmlu(
    subjects: Optional[list[str]] = None,
    split: str = "test",
    max_per_subject: Optional[int] = 50,
) -> pd.DataFrame:
    """
    Load MMLU questions for the given subjects (defaults to all).

    Each row has:
        subject, category, question, choices (list[str]), answer_idx (0-3), answer_letter
    """
    subjects = subjects or MMLU_SUBJECTS
    records = []

    for subject in subjects:
        try:
            ds = load_dataset("cais/mmlu", subject, split=split)
            df = ds.to_pandas()

            if max_per_subject:
                df = df.sample(n=min(max_per_subject, len(df)), random_state=42)

            df["subject"] = subject
            df["category"] = _subject_to_category(subject)
            records.append(df)
        except Exception as e:
            print(f"[warn] Could not load {subject}: {e}")

    combined = pd.concat(records, ignore_index=True)

    # Normalize column names across dataset versions
    combined = combined.rename(columns={"answer": "answer_idx"})
    combined["answer_letter"] = combined["answer_idx"].apply(lambda i: "ABCD"[int(i)])

    return combined[["subject", "category", "question", "choices", "answer_idx", "answer_letter"]]
