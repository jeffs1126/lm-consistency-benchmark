"""
Plotting utilities for the consistency benchmark results.
All figures saved to /figures directory.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
COLORS = sns.color_palette("muted")


def _save(name: str) -> None:
    path = FIGURES_DIR / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# MMLU plots
# ---------------------------------------------------------------------------

def plot_consistency_by_category(df_results: pd.DataFrame, model_name: str) -> None:
    """Bar chart: consistency rate per category."""
    fig, ax = plt.subplots(figsize=(8, 5))
    cats = df_results.sort_values("consistency_rate", ascending=True)
    bars = ax.barh(cats["category"], cats["consistency_rate"], color=COLORS[0])
    ax.set_xlabel("Consistency Rate")
    ax.set_title(f"LLM Consistency by Domain — {model_name}")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.axvline(x=cats["consistency_rate"].mean(), color="red", linestyle="--", label="mean")
    ax.legend()
    _save(f"consistency_by_category_{model_name.replace('/', '-')}.png")


def plot_accuracy_vs_consistency(df: pd.DataFrame, model_name: str) -> None:
    """Scatter: per-question accuracy vs consistency (binary)."""
    grp = df.groupby("original_idx").agg(
        acc=("correct", "mean"),
        consistent=("predicted", lambda x: int(x.nunique() == 1)),
        category=("category", "first"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (cat, sub) in enumerate(grp.groupby("category")):
        ax.scatter(
            sub["acc"] + np.random.uniform(-0.015, 0.015, len(sub)),
            sub["consistent"] + np.random.uniform(-0.05, 0.05, len(sub)),
            label=cat, alpha=0.6, s=30, color=COLORS[i % len(COLORS)],
        )
    ax.set_xlabel("Question Accuracy (across variants)")
    ax.set_ylabel("Consistent (1) / Inconsistent (0)")
    ax.set_title(f"Accuracy vs. Consistency — {model_name}")
    ax.legend(fontsize=9, loc="center right")
    _save(f"acc_vs_consistency_{model_name.replace('/', '-')}.png")


def plot_original_vs_paraphrase_accuracy(df: pd.DataFrame, model_name: str) -> None:
    """Grouped bar: accuracy on original vs. paraphrase variants per category."""
    df = df.copy()
    df["type"] = df["paraphrase_idx"].apply(lambda x: "Original" if x == 0 else "Paraphrase")
    pivot = df.groupby(["category", "type"])["correct"].mean().unstack()

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax, color=[COLORS[0], COLORS[1]], width=0.6)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy: Original vs. Paraphrase by Domain — {model_name}")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(title="Variant type")
    _save(f"orig_vs_para_accuracy_{model_name.replace('/', '-')}.png")


def plot_consistency_heatmap(df: pd.DataFrame, model_name: str, top_n: int = 40) -> None:
    """Heatmap of predicted answers per question (rows) x variant (columns)."""
    letter_map = {"A": 0, "B": 1, "C": 2, "D": 3, "?": -1}
    pivot = df.pivot_table(
        index="original_idx",
        columns="paraphrase_idx",
        values="predicted",
        aggfunc="first",
    ).applymap(lambda x: letter_map.get(x, -1))

    pivot = pivot.head(top_n)
    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(
        pivot, ax=ax, cmap="RdYlGn", vmin=-1, vmax=3,
        linewidths=0.3, cbar_kws={"label": "Answer (A=0, B=1, C=2, D=3)"},
    )
    ax.set_title(f"Answer Heatmap (top {top_n} questions) — {model_name}")
    ax.set_xlabel("Paraphrase variant (0 = original)")
    ax.set_ylabel("Question index")
    _save(f"answer_heatmap_{model_name.replace('/', '-')}.png")


# ---------------------------------------------------------------------------
# PAWS plots
# ---------------------------------------------------------------------------

def plot_paws_flip_rate(metrics_by_model: dict[str, dict], tag: str = "") -> None:
    """Bar chart comparing flip rates across models."""
    models = list(metrics_by_model.keys())
    overall = [metrics_by_model[m]["overall_flip_rate"] for m in models]
    true_para = [metrics_by_model[m]["flip_rate_true_paraphrase"] for m in models]
    non_para = [metrics_by_model[m]["flip_rate_non_paraphrase"] for m in models]

    x = np.arange(len(models))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, overall, width, label="Overall", color=COLORS[0])
    ax.bar(x, true_para, width, label="True paraphrases", color=COLORS[1])
    ax.bar(x + width, non_para, width, label="Non-paraphrases", color=COLORS[2])
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Flip Rate")
    ax.set_title("Order-Sensitivity Flip Rate — PAWS Track")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.legend()
    _save(f"paws_flip_rate{('_' + tag) if tag else ''}.png")


def plot_multi_model_consistency(summary_rows: list[dict]) -> None:
    """
    Bar chart comparing overall consistency rate and accuracy across models.
    summary_rows: list of dicts with keys model, overall_consistency_rate, overall_accuracy
    """
    df = pd.DataFrame(summary_rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, col, title in zip(
        axes,
        ["overall_consistency_rate", "overall_accuracy"],
        ["Consistency Rate", "Accuracy"],
    ):
        bars = ax.bar(df["model"], df[col], color=COLORS[:len(df)])
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
        ax.set_xticklabels(df["model"], rotation=20, ha="right")
        for bar, val in zip(bars, df[col]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.1%}", ha="center", va="bottom", fontsize=9)

    plt.suptitle("Multi-Model Benchmark: Consistency vs. Accuracy", y=1.02)
    plt.tight_layout()
    _save("multi_model_comparison.png")
