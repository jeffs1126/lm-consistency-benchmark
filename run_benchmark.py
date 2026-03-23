"""
Main entry point for the LM Consistency Benchmark.

Usage:
    python run_benchmark.py --track mmlu --model claude-haiku-4-5-20251001 --n 100
    python run_benchmark.py --track paws --model claude-haiku-4-5-20251001 --n 200
    python run_benchmark.py --track both --model claude-haiku-4-5-20251001
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

from src.data_loader import load_paws, load_mmlu
from src.paraphrase import generate_paraphrases
from src.evaluate import evaluate_paws, evaluate_mmlu
from src.metrics import full_mmlu_report, paws_flip_rate, paws_accuracy
from src.visualize import (
    plot_consistency_by_category,
    plot_accuracy_vs_consistency,
    plot_original_vs_paraphrase_accuracy,
    plot_consistency_heatmap,
    plot_paws_flip_rate,
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def run_mmlu(model: str, provider: str, n_questions: int, n_paraphrases: int) -> dict:
    print(f"\n=== MMLU Track | model={model} | n={n_questions} | paraphrases={n_paraphrases} ===")

    print("Loading MMLU dataset...")
    mmlu_df = load_mmlu(max_per_subject=max(1, n_questions // 57))

    print("Generating paraphrases (cached after first run)...")
    variants_df = generate_paraphrases(
        mmlu_df,
        n_paraphrases=n_paraphrases,
        max_rows=n_questions,
        cache_file=f"mmlu_paraphrases_n{n_questions}_p{n_paraphrases}.jsonl",
    )

    print("Evaluating model...")
    results_df = evaluate_mmlu(variants_df, model=model, provider=provider, results_tag=f"n{n_questions}")

    report = full_mmlu_report(results_df, model_name=model)
    report_path = RESULTS_DIR / f"mmlu_report_{model.replace('/', '-')}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_path}")

    if "error" in report:
        print(f"  [error] {report['error']}")
        return report

    print(f"  Consistency Rate : {report['overall_consistency_rate']:.1%}")
    print(f"  Accuracy         : {report['overall_accuracy']:.1%}")
    print(f"  Krippendorff α   : {report['krippendorff_alpha']:.3f}")

    print("Generating figures...")
    from src.metrics import consistency_by_category
    cat_df = consistency_by_category(results_df)
    plot_consistency_by_category(cat_df, model)
    plot_accuracy_vs_consistency(results_df, model)
    plot_original_vs_paraphrase_accuracy(results_df, model)
    plot_consistency_heatmap(results_df, model)

    return report


def run_paws(model: str, provider: str, n_samples: int) -> dict:
    print(f"\n=== PAWS Track | model={model} | n={n_samples} ===")

    print("Loading PAWS dataset...")
    paws_df = load_paws(split="test")

    print("Evaluating model...")
    results_df = evaluate_paws(paws_df, model=model, provider=provider, max_samples=n_samples)

    flip_metrics = paws_flip_rate(results_df)
    acc_metrics = paws_accuracy(results_df)
    report = {"model": model, **flip_metrics, **acc_metrics}

    report_path = RESULTS_DIR / f"paws_report_{model.replace('/', '-')}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_path}")
    print(f"  Flip Rate (overall)           : {flip_metrics['overall_flip_rate']:.1%}")
    print(f"  Flip Rate (true paraphrases)  : {flip_metrics['flip_rate_true_paraphrase']:.1%}")
    print(f"  Flip Rate (non-paraphrases)   : {flip_metrics['flip_rate_non_paraphrase']:.1%}")
    print(f"  Forward Accuracy              : {acc_metrics['fwd_accuracy']:.1%}")

    plot_paws_flip_rate({model: flip_metrics}, tag=model.replace("/", "-"))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=["mmlu", "paws", "both"], default="both")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    parser.add_argument("--n", type=int, default=100, help="Number of questions/pairs to evaluate")
    parser.add_argument("--paraphrases", type=int, default=3, help="Number of paraphrases per MMLU question")
    args = parser.parse_args()

    if args.track in ("mmlu", "both"):
        run_mmlu(args.model, args.provider, args.n, args.paraphrases)

    if args.track in ("paws", "both"):
        run_paws(args.model, args.provider, args.n)


if __name__ == "__main__":
    main()
