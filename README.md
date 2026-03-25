# LM Consistency Benchmark

> **Empirical Study of Paraphrase Sensitivity in Frontier Language Models**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Benchmark](https://img.shields.io/badge/Type-Benchmark-orange)
![Datasets](https://img.shields.io/badge/Datasets-MMLU%20%7C%20PAWS-green)
![Providers](https://img.shields.io/badge/Providers-Anthropic%20%7C%20OpenAI-purple)

---

## Introduction

A fundamental assumption underlying LLM evaluation and deployment is that semantically equivalent inputs produce equivalent outputs. If a model truly understands a question, rephrasing it should not change the answer. **This benchmark tests that assumption directly, empirically, and at scale.**

We introduce a two-track evaluation framework for measuring **paraphrase sensitivity** in frontier language models — the degree to which surface-level rephrasing or input reordering changes model behavior:

1. **Paraphrase Consistency (MMLU Track):** For each multiple-choice question, we generate semantically equivalent restatements using an auxiliary LLM and measure whether the target model answers identically across all variants.
2. **Order Sensitivity (PAWS Track):** Using adversarially constructed sentence pairs, we measure whether swapping the presentation order of a pair changes the model's paraphrase judgment.

Results reveal substantial sensitivity: at n=500, tested models change their answers on **~47% of paraphrased MMLU questions** and exhibit a **41% flip rate on PAWS order-swapped pairs**, with meaningful variation across subject domains and pair types.

### Why This Matters

- **Evaluation reliability:** If a model gives different answers to paraphrased versions of the same question, benchmark scores reflect prompt wording sensitivity, not true capability.
- **Production risk:** In applications where prompt phrasing is not tightly controlled, inconsistent behavior is an unpredictable and hard-to-debug failure mode.
- **Model comparison:** Consistency is a meaningful secondary signal — two models can achieve identical accuracy with very different consistency profiles, revealing different underlying reasoning stability.
- **AI safety / alignment:** A model that changes its answer based on surface form rather than semantics may be exploiting spurious statistical patterns rather than grounded understanding.

---

## Scope

### What This Benchmark Covers

| Dimension | Coverage |
|---|---|
| Benchmark datasets | MMLU (57 subjects, ~14K questions), PAWS (8K adversarial pair) |
| Paraphrase types | LLM-generated semantic restatements (MMLU), order-swapped input pairs (PAWS) |
| Models supported | Any Anthropic or OpenAI model via API |
| Metrics | Consistency rate, Krippendorff's α, accuracy drop, flip rate (stratified), accuracy–consistency correlation |
| Analysis | Domain-level breakdown, per-category variance, bootstrap 95% CIs |

### What This Benchmark Does NOT Cover

- **Adversarial robustness** — this study tests natural paraphrase variation, not malicious prompt attacks
- **Chain-of-thought / few-shot prompting** — all evaluations are zero-shot, single-turn only
- **Non-English benchmarks** — English only in the current version
- **Open-source / local models** — API-based models only (no HuggingFace inference)
- **Human-authored paraphrases** — paraphrases are LLM-generated; human validation is a planned extension

### Research Questions

1. **RQ1 — Consistency:** Do frontier LLMs produce the same answer when a question is semantically paraphrased? How does consistency vary across subject domains?
2. **RQ2 — Order sensitivity:** Do models exhibit position bias when sentence pairs are presented in reversed order?
3. **RQ3 — Accuracy–Consistency relationship:** Is inconsistency a proxy for question difficulty, or is it a distinct behavioral phenomenon?
4. **RQ4 — Model comparison:** Do Anthropic and OpenAI model families differ systematically in their paraphrase sensitivity profiles?

---

## Methodology

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

### Evaluation Pipeline

```
Dataset (PAWS / MMLU)
        │
        ▼
Paraphrase Generation (LLM-based, cached to disk)
        │
        ▼
Model Query  ──────────────────────────────┐
(Anthropic / OpenAI API)                   │
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
          Bootstrap CIs + Figures + JSON Reports
```

### Design Choices

**Why PAWS?**
PAWS provides adversarially constructed near-paraphrase pairs — a stronger test than random paraphrases because it controls for word overlap. This separates models that reason about meaning from those relying on surface-level lexical cues.

**Why MMLU?**
MMLU has ground-truth labels across 57 distinct subject areas, enabling domain-stratified analysis. Its established accuracy baseline literature makes consistency findings interpretable alongside known model capability profiles.

**Why these metrics?**
- *Consistency Rate* — simple and interpretable, but ignores partial agreement
- *Krippendorff's α* — captures inter-variant reliability on a continuous scale; standard in annotation reliability literature
- *Flip Rate* — direct behavioral signal on PAWS, free of ground-truth dependence
- *Accuracy–Consistency Correlation* — tests whether inconsistency is a proxy for difficulty or a separate phenomenon

**Why LLM-generated paraphrases?**
Adversarial prompts test robustness to malicious inputs; paraphrases test consistency under natural restatement — a more realistic proxy for production prompt variation.

---

## Results

> Current results: `claude-haiku-4-5-20251001`. Comparative results with `gpt-4o-mini` are being collected and will be added. Bootstrap 95% CIs computed over 500 resamples.

### MMLU — Paraphrase Consistency

**Setting:** `claude-haiku-4-5-20251001` · n=348 questions · 3 paraphrases each

| Metric | Value | 95% CI |
|---|---|---|
| **Consistency Rate** | **47.4%** | 43.6% – 51.4% |
| Accuracy | 30.3% | 27.8% – 32.7% |
| Krippendorff's α | 0.493 | 0.452 – 0.531 |

> **Interpretation:** Krippendorff's α = 0.493 falls well below the acceptable reliability threshold (α ≥ 0.667). The tight 95% CI on consistency rate (43.6–51.4%) confirms this is not a small-sample artifact — the model changes its answer on roughly **half** of semantically equivalent question variants.

**Per-category breakdown:**

| Category | Consistency Rate | Accuracy |
|---|---|---|
| STEM | ~50–65% | ~30–35% |
| Social Sciences | ~45–55% | ~35–42% |
| Humanities | ~25–35% | ~42–48% |
| Professional / Applied | ~55–65% | ~26–32% |

> **Key pattern:** Humanities shows the most striking result — **lowest consistency, highest accuracy** — suggesting the model answers correctly but through a brittle reasoning path that varies with question phrasing.

![Consistency by Category](figures/consistency_by_category_claude-haiku-4-5-20251001.png)
![Accuracy vs Consistency](figures/acc_vs_consistency_claude-haiku-4-5-20251001.png)
![Original vs Paraphrase Accuracy](figures/orig_vs_para_accuracy_claude-haiku-4-5-20251001.png)
![Answer Heatmap](figures/answer_heatmap_claude-haiku-4-5-20251001.png)

---

### PAWS — Order Sensitivity

**Setting:** `claude-haiku-4-5-20251001` · n=500 pairs

| Metric | Value | 95% CI |
|---|---|---|
| **Flip Rate (overall)** | **41.4%** | 37.2% – 45.6% |
| Flip Rate (true paraphrases) | 23.9% | — |
| Flip Rate (non-paraphrases) | 55.4% | — |
| Forward Accuracy | 60.0% | — |

> **Key pattern:** The **2.3× difference** in flip rate between true paraphrases (23.9%) and non-paraphrases (55.4%) confirms the model captures genuine semantic signal. The overall 41.4% flip rate (95% CI: 37.2–45.6%) is substantially above chance (50% would be random), confirming a real but incomplete sensitivity to input ordering. The pilot n=100 estimate of 49% was inflated; n=500 stabilizes near 41%.

![PAWS Flip Rate](figures/paws_flip_rate_claude-haiku-4-5-20251001.png)

---

## Reproduce the Results

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env — add ANTHROPIC_API_KEY and/or OPENAI_API_KEY

# 3. Reproduce the headline MMLU result (n=348, 3 paraphrases)
python run_benchmark.py --track mmlu --model claude-haiku-4-5-20251001 --n 348 --paraphrases 3

# 4. Reproduce the headline PAWS result (n=100)
python run_benchmark.py --track paws --model claude-haiku-4-5-20251001 --n 100

# 5. Run both tracks
python run_benchmark.py --track both --model claude-haiku-4-5-20251001

# 6. Run with an OpenAI model
python run_benchmark.py --track both --model gpt-4o-mini --provider openai --n 500
```

Outputs are saved to:
- `results/` — raw JSONL model outputs and summary JSON reports
- `figures/` — all plots (PNG)
- `results/paraphrases/` — cached paraphrase sets (reused across runs to save cost)

---

## Project Structure

```
lm-consistency-benchmark/
├── src/
│   ├── data_loader.py    # PAWS + MMLU loading via HuggingFace (no manual download)
│   ├── paraphrase.py     # LLM-based paraphrase generation with disk caching
│   ├── evaluate.py       # Model querying for both tracks (Anthropic + OpenAI)
│   ├── metrics.py        # Consistency rate, Krippendorff's α, flip rate, bootstrap CIs
│   └── visualize.py      # All plots → /figures
├── notebooks/
│   └── analysis.ipynb    # End-to-end results walkthrough
├── results/              # Cached model outputs + summary reports
├── figures/              # Generated plots
├── run_benchmark.py      # CLI entry point
└── requirements.txt
```

---

## Limitations

- **Sample size:** Current MMLU results use 348 questions with 3 paraphrases each, and PAWS uses 100 pairs. Bootstrap CIs confirm headline findings are not noise artifacts, but domain-level breakdowns remain underpowered. Full-scale runs (n=1,000+) are in progress.
- **Paraphrase quality:** Paraphrases are LLM-generated and may introduce lexical artifacts, subtle meaning shifts, or implicit answer leakage. Manual quality auditing is a planned validation step.
- **Model coverage:** Current results use one model at default temperature. Cross-model comparative results (`gpt-4o-mini`) are being collected. Findings should not be generalized across providers until replication is complete.
- **Prompt sensitivity:** The paraphrase generation prompt and model query prompt are not ablated. Different framings may shift the measured consistency rate.
- **No chain-of-thought:** All evaluations use zero-shot prompting. CoT prompting may improve or suppress consistency differently across models.

---

## Roadmap

- [ ] Complete `gpt-4o-mini` comparison run and publish side-by-side results table
- [ ] Expand MMLU run to n=1,000+ questions
- [ ] Benchmark additional frontier models (Claude Sonnet, GPT-4o, Llama 3)
- [ ] Validate paraphrase label preservation via manual audit (10% sample)
- [ ] Add per-subject variance and distribution plots
- [ ] Add experiment config logging for full reproducibility
- [ ] Add unit tests for metrics module

---

## Metrics Reference

| Metric | Description |
|---|---|
| Consistency Rate | % of questions answered identically across all variants |
| Accuracy | % of answers correct on original questions |
| Krippendorff's α | Inter-rater reliability across variants (nominal scale; α ≥ 0.8 = reliable) |
| Flip Rate | % of PAWS pairs where answer changes with input order |
| Accuracy Drop | Accuracy(original) − Accuracy(paraphrases) |

---

## Citation

```bibtex
@misc{lm-consistency-benchmark,
  author       = {Junming Song},
  title        = {LM Consistency Benchmark: Empirical Study of Paraphrase Sensitivity in Frontier Language Models},
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
