# LM Consistency Benchmark

> Measure how consistently frontier LLMs answer semantically equivalent questions across paraphrase variants and reordered inputs.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Benchmark](https://img.shields.io/badge/Type-Benchmark-orange)
![Datasets](https://img.shields.io/badge/Datasets-MMLU%20%7C%20PAWS-green)
![Providers](https://img.shields.io/badge/Providers-Anthropic%20%7C%20OpenAI-purple)

**Main finding:** In a pilot run on `claude-haiku-4-5-20251001`, the model answered consistently on only **52.6%** of MMLU questions across paraphrase variants, and exhibited a **49.0% flip rate** on PAWS order-swapped pairs — suggesting that surface-level phrasing has a meaningful, measurable effect on model outputs.

![Consistency by Category](figures/consistency_by_category_claude-haiku-4-5-20251001.png)

---

## Why This Matters

LLM evaluations and production deployments typically assume that semantically equivalent prompts produce equivalent outputs. This benchmark tests that assumption directly.

- **Evaluation reliability:** If a model gives different answers to paraphrased versions of the same question, benchmark scores may be sensitive to prompt wording rather than true capability.
- **Production risk:** In applications where prompt phrasing is not tightly controlled, inconsistent behavior becomes an unpredictable failure mode.
- **Model comparison:** Consistency is a meaningful secondary signal alongside accuracy — two models can achieve the same accuracy with very different consistency profiles.
- **AI safety / alignment:** A model that changes its answer based on surface form rather than semantics may be picking up on spurious statistical patterns rather than grounded reasoning.

---

## What This Benchmark Measures

### Track 1 — PAWS (Order Sensitivity)

| Property | Detail |
|---|---|
| Dataset | [PAWS](https://huggingface.co/datasets/google-research-datasets/paws) `labeled_final` (8K test pairs) |
| Test | Each sentence pair evaluated in both orders: S1→S2 and S2→S1 |
| Stratification | True paraphrases (label=1) vs. adversarial non-paraphrases (label=0) |
| Key output | **Flip rate** — % of pairs where model contradicts itself across orderings |

### Track 2 — MMLU (Paraphrase Consistency)

| Property | Detail |
|---|---|
| Dataset | [MMLU](https://huggingface.co/datasets/cais/mmlu) (57 subjects, ~14K questions) |
| Test | Generate N paraphrases per question via LLM, query target model on each variant independently |
| Key outputs | **Consistency rate**, **Krippendorff's α**, **accuracy drop**, **accuracy–consistency correlation** |

---

## Main Results

All results below are from a pilot run: `claude-haiku-4-5-20251001`, n=100 per track.

### MMLU — Paraphrase Consistency

| Metric | Value | Notes |
|---|---|---|
| **Consistency Rate** | **52.6%** | % of questions answered identically across all 4 variants |
| Accuracy | 36.4% | % correct on original questions |
| Krippendorff's α | 0.509 | Moderate agreement; 0.8+ is the reliability threshold |

**Per-category breakdown (n=100 questions total):**

| Category | Consistency Rate | Accuracy | n |
|---|---|---|---|
| Humanities | 25.0% | 45.8% | 12 |
| Social Sciences | 50.0% | 40.0% | 10 |
| Professional / Applied | 60.0% | 30.0% | 15 |
| STEM | 65.0% | 33.8% | 20 |

> Takeaway: the model changes its answer on nearly **half** of semantically equivalent question variants. Humanities questions show both the lowest consistency (25%) and highest accuracy (45.8%), suggesting the model is confident but brittle in that domain.

![Accuracy vs Consistency](figures/acc_vs_consistency_claude-haiku-4-5-20251001.png)
![Original vs Paraphrase Accuracy](figures/orig_vs_para_accuracy_claude-haiku-4-5-20251001.png)
![Answer Heatmap](figures/answer_heatmap_claude-haiku-4-5-20251001.png)

---

### PAWS — Order Sensitivity

| Metric | Value | Notes |
|---|---|---|
| **Flip Rate (overall)** | **49.0%** | % of pairs where answer changes when S1/S2 order swaps |
| Flip Rate (true paraphrases) | 25.6% | Lower — model partially tracks semantic equivalence |
| Flip Rate (non-paraphrases) | 66.7% | Higher — model is more sensitive to adversarial pairs |
| Forward Accuracy | 58.0% | % correctly labeled in S1→S2 order |

> Takeaway: the 2.6x difference in flip rate between true paraphrases (25.6%) and adversarial non-paraphrases (66.7%) shows the model does capture some semantic signal — but the overall 49% flip rate means order sensitivity is still a dominant driver of output, not just meaning.

![PAWS Flip Rate](figures/paws_flip_rate_claude-haiku-4-5-20251001.png)

---

## Reproduce the Results

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env and add ANTHROPIC_API_KEY and/or OPENAI_API_KEY

# 3. Reproduce the headline MMLU result
python run_benchmark.py --track mmlu --model claude-haiku-4-5-20251001 --n 100 --paraphrases 3

# 4. Reproduce the headline PAWS result
python run_benchmark.py --track paws --model claude-haiku-4-5-20251001 --n 100

# 5. Run both tracks together
python run_benchmark.py --track both --model claude-haiku-4-5-20251001 --n 100

# 6. Run with an OpenAI model
python run_benchmark.py --track mmlu --model gpt-4o-mini --provider openai --n 100
```

Outputs are saved to:
- `results/` — raw JSONL model outputs and summary JSON reports
- `figures/` — all plots (PNG)
- `results/paraphrases/` — cached paraphrase sets (reused across runs)

---

## Project Structure

```
lm-consistency-benchmark/
├── src/
│   ├── data_loader.py    # PAWS + MMLU loading via HuggingFace (no manual download)
│   ├── paraphrase.py     # LLM-based paraphrase generation with disk caching
│   ├── evaluate.py       # Model querying for both tracks (Anthropic + OpenAI)
│   ├── metrics.py        # Consistency rate, Krippendorff's α, flip rate, corr
│   └── visualize.py      # All plots → /figures
├── notebooks/
│   └── analysis.ipynb    # End-to-end results walkthrough
├── results/              # Cached model outputs + summary reports
├── figures/              # Generated plots
├── run_benchmark.py      # CLI entry point
└── requirements.txt
```

**Pipeline:**

```
Dataset (PAWS / MMLU)
        │
        ▼
Paraphrase Generation (cached)
        │
        ▼
Model Query  ──────────────────────────────┐
(Anthropic / OpenAI)                       │
        │                                  │
        ▼                                  ▼
  MMLU metrics                       PAWS metrics
  - consistency rate                 - flip rate
  - Krippendorff's α                 - stratified by label
  - accuracy drop                    - forward accuracy
  - accuracy–consistency corr
        │                                  │
        └──────────────┬───────────────────┘
                       ▼
              Figures + JSON Reports
```

---

## Evaluation Design Choices

**Why PAWS?**
PAWS provides adversarially constructed near-paraphrase pairs — a stronger test than random paraphrases because it controls for word overlap. This separates models that reason about meaning from those relying on surface-level lexical cues.

**Why MMLU?**
MMLU has ground-truth labels across 57 distinct subject areas, enabling domain-stratified analysis. It also has an established baseline accuracy literature, making consistency findings interpretable alongside known capability profiles.

**Why these metrics?**
- *Consistency Rate* is simple and interpretable, but ignores partial agreement.
- *Krippendorff's α* captures inter-variant reliability on a continuous scale and is standard in annotation reliability literature.
- *Flip Rate* on PAWS is a direct behavioral signal, free of ground-truth dependence.
- *Accuracy–Consistency Correlation* tests whether inconsistency is a proxy for difficulty or a separate phenomenon.

**Why LLM-generated paraphrases instead of adversarial prompts?**
Adversarial prompts test robustness to malicious inputs; paraphrases test consistency under natural restatement — a more realistic proxy for production prompt variation. The tradeoff is that generated paraphrases may introduce model-specific artifacts, which we flag as a limitation.

---

## Limitations

- **Sample size:** The reported pilot results use 100 MMLU questions (3 paraphrases each) and 100 PAWS pairs. Results at this scale carry substantial variance and should be treated as indicative, not conclusive. Full-scale runs are on the roadmap.
- **Paraphrase quality:** Paraphrases are LLM-generated and may introduce lexical artifacts, slight meaning shifts, or implicit leakage of the correct answer. Manual auditing of a random sample is a planned quality check.
- **Single model:** All reported results use one model at one temperature setting. Findings should not be generalized until replicated across providers and model sizes.
- **No confidence intervals:** Current reporting uses point estimates. Bootstrap CIs and per-subject variance will be added in a future update.
- **Prompt sensitivity:** The paraphrase generation prompt and model query prompt are not ablated. Different prompt framings may yield different paraphrase distributions.

---

## Roadmap

- [ ] Expand MMLU run to full dataset (n=1,000+)
- [ ] Benchmark 5+ frontier models (GPT-4o, Claude Sonnet, Llama 3, Gemini)
- [ ] Add bootstrap confidence intervals to all metrics
- [ ] Validate paraphrase label preservation via manual audit (10% sample)
- [ ] Add per-subject variance and distribution plots
- [ ] Add experiment config logging for full reproducibility
- [ ] Add unit tests for metrics module
- [ ] Publish results comparison table across models

---

## Metrics Reference

| Metric | Description |
|---|---|
| Consistency Rate | % of questions answered identically across all variants |
| Accuracy | % of answers correct on original questions |
| Krippendorff's α | Inter-rater reliability across variants (nominal scale; 0.8+ = reliable) |
| Flip Rate | % of PAWS pairs where answer changes with input order |
| Accuracy Drop | Accuracy(original) − Accuracy(paraphrases) |

---

## Citation

If you use this benchmark or codebase, please cite:

```bibtex
@misc{lm-consistency-benchmark,
  author       = {jeffs1126},
  title        = {LM Consistency Benchmark: Paraphrase Sensitivity in Frontier Language Models},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/jeffs1126/lm-consistency-benchmark}
}
```

---

## Datasets Used

- [PAWS](https://huggingface.co/datasets/google-research-datasets/paws) — Zhang et al., Google Research
- [MMLU](https://huggingface.co/datasets/cais/mmlu) — Hendrycks et al., CAIS

## License

MIT License. See [LICENSE](LICENSE) for details.
