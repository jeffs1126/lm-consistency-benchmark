"""
Consistency and accuracy metrics.

Key metrics computed here:
  - Consistency Rate (CR): % of question groups where the model gives the same
    answer across ALL variants (original + paraphrases)
  - Accuracy (Acc): % of individual answers that are correct
  - Accuracy-Consistency Gap: Acc(original) - Acc(paraphrases)
  - Krippendorff's Alpha: inter-rater reliability across variants (ordinal)
  - Flip Rate (FR): % of PAWS pairs where the model contradicts itself
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import krippendorff
from scipy import stats


# ---------------------------------------------------------------------------
# MMLU metrics
# ---------------------------------------------------------------------------

def consistency_rate(df: pd.DataFrame) -> float:
    """
    % of original questions where the model gives the same answer to ALL
    paraphrase variants (including the original).
    """
    grouped = df.groupby("original_idx")["predicted"].nunique()
    return (grouped == 1).mean()


def accuracy(df: pd.DataFrame) -> float:
    return df["correct"].mean()


def accuracy_by_paraphrase_type(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy split by original (paraphrase_idx=0) vs. paraphrase (>0)."""
    df = df.copy()
    df["type"] = df["paraphrase_idx"].apply(lambda x: "original" if x == 0 else "paraphrase")
    return df.groupby("type")["correct"].agg(["mean", "count"]).rename(columns={"mean": "accuracy"})


def consistency_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Per-category consistency rate."""
    results = []
    for cat, grp in df.groupby("category"):
        cr = consistency_rate(grp)
        acc = accuracy(grp)
        results.append({"category": cat, "consistency_rate": cr, "accuracy": acc, "n_questions": grp["original_idx"].nunique()})
    return pd.DataFrame(results).sort_values("consistency_rate")


def consistency_by_subject(df: pd.DataFrame) -> pd.DataFrame:
    """Per-subject consistency rate (finer granularity)."""
    results = []
    for subj, grp in df.groupby("subject"):
        cr = consistency_rate(grp)
        acc = accuracy(grp)
        results.append({"subject": subj, "consistency_rate": cr, "accuracy": acc, "n_questions": grp["original_idx"].nunique()})
    return pd.DataFrame(results).sort_values("consistency_rate")


def krippendorff_alpha(df: pd.DataFrame) -> float:
    """
    Compute Krippendorff's alpha treating each paraphrase variant as a 'rater'
    and each original question as a 'unit'. Uses nominal level (answers are
    categorical: A/B/C/D).
    """
    # Map letters to ints
    letter_map = {"A": 0, "B": 1, "C": 2, "D": 3, "?": np.nan}
    pivot = df.pivot_table(
        index="original_idx",
        columns="paraphrase_idx",
        values="predicted",
        aggfunc="first",
    ).applymap(lambda x: letter_map.get(x, np.nan))

    data = pivot.values.T  # shape: (n_raters, n_units)
    try:
        return krippendorff.alpha(reliability_data=data, level_of_measurement="nominal")
    except Exception:
        return float("nan")


def accuracy_consistency_correlation(df: pd.DataFrame) -> tuple[float, float]:
    """
    Pearson correlation between per-question accuracy and consistency.
    Returns (r, p_value).
    """
    grp = df.groupby("original_idx").agg(
        acc=("correct", "mean"),
        n_unique=("predicted", "nunique"),
    )
    grp["consistent"] = (grp["n_unique"] == 1).astype(float)
    r, p = stats.pearsonr(grp["acc"], grp["consistent"])
    return float(r), float(p)


# ---------------------------------------------------------------------------
# PAWS metrics
# ---------------------------------------------------------------------------

def paws_flip_rate(df: pd.DataFrame) -> dict[str, float]:
    """
    Overall and class-stratified flip rate for PAWS results.
    flip_rate = % of pairs where ans_fwd != ans_rev
    """
    overall = 1 - df["consistent"].mean()
    by_label = df.groupby("label").apply(lambda g: 1 - g["consistent"].mean()).to_dict()
    return {
        "overall_flip_rate": overall,
        "flip_rate_true_paraphrase": by_label.get(1, float("nan")),
        "flip_rate_non_paraphrase": by_label.get(0, float("nan")),
    }


def paws_accuracy(df: pd.DataFrame) -> dict[str, float]:
    """
    % of forward queries where the LLM's YES/NO matches the ground truth label.
    label=1 → expected YES, label=0 → expected NO
    """
    df = df.copy()
    df["expected"] = df["label"].map({1: "YES", 0: "NO"})
    df["fwd_correct"] = df["ans_fwd"] == df["expected"]
    df["rev_correct"] = df["ans_rev"] == df["expected"]
    return {
        "fwd_accuracy": df["fwd_correct"].mean(),
        "rev_accuracy": df["rev_correct"].mean(),
        "mean_accuracy": ((df["fwd_correct"].astype(float) + df["rev_correct"].astype(float)) / 2).mean(),
    }


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def full_mmlu_report(df: pd.DataFrame, model_name: str) -> dict:
    r, p = accuracy_consistency_correlation(df)
    return {
        "model": model_name,
        "n_questions": df["original_idx"].nunique(),
        "n_variants_per_question": df.groupby("original_idx").size().mean(),
        "overall_consistency_rate": consistency_rate(df),
        "overall_accuracy": accuracy(df),
        "krippendorff_alpha": krippendorff_alpha(df),
        "acc_consistency_pearson_r": r,
        "acc_consistency_pearson_p": p,
        "by_paraphrase_type": accuracy_by_paraphrase_type(df).to_dict(),
        "by_category": consistency_by_category(df).to_dict(orient="records"),
    }
