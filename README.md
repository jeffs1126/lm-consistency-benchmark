# LM Consistency Benchmark

An empirical study of how consistently large language models answer semantically equivalent questions across paraphrase variants.

## Research Question

> Do frontier LLMs produce the same answer when a question is rephrased? Does consistency vary by domain, and does it correlate with accuracy?

## Methodology

### Track 1 — PAWS (Order-Sensitivity)
We query the model on each sentence pair in both orders (S1→S2 and S2→S1) and measure the **flip rate**: the fraction of pairs where the model contradicts itself. We stratify by true paraphrases (label=1) vs. adversarial non-paraphrases (label=0).

**Dataset:** [PAWS](https://huggingface.co/datasets/google-research-datasets/paws) (`labeled_final`, n=8K test pairs)

### Track 2 — MMLU (Paraphrase Consistency)
For each MMLU multiple-choice question, we generate N semantically equivalent paraphrases using a lightweight LLM, then query the target model on each variant independently. We measure:
- **Consistency Rate (CR):** % of questions answered identically across all variants
- **Accuracy Drop:** Accuracy on original vs. paraphrase variants
- **Krippendorff's α:** Inter-variant agreement (nominal)
- **Accuracy–Consistency Correlation:** Do harder questions also show more inconsistency?

**Dataset:** [MMLU](https://huggingface.co/datasets/cais/mmlu) (57 subjects, ~14K questions)

## Key Findings

All results are for `claude-haiku-4-5-20251001`.

### MMLU Track (n=100 questions, 3 paraphrases each)

| Metric | Value |
|---|---|
| Consistency Rate | 52.6% |
| Accuracy | 36.4% |
| Krippendorff's α | 0.509 |

**Consistency by category:**

| Category | Consistency Rate | Accuracy | n |
|---|---|---|---|
| Humanities | 25.0% | 45.8% | 12 |
| Social Sciences | 50.0% | 40.0% | 10 |
| Professional / Applied | 60.0% | 30.0% | 15 |
| STEM | 65.0% | 33.8% | 20 |

![Consistency by Category](figures/consistency_by_category_claude-haiku-4-5-20251001.png)

![Accuracy vs Consistency](figures/acc_vs_consistency_claude-haiku-4-5-20251001.png)

![Original vs Paraphrase Accuracy](figures/orig_vs_para_accuracy_claude-haiku-4-5-20251001.png)

![Answer Heatmap](figures/answer_heatmap_claude-haiku-4-5-20251001.png)

### PAWS Track (n=100 pairs)

| Metric | Value |
|---|---|
| Flip Rate (overall) | 49.0% |
| Flip Rate (true paraphrases) | 25.6% |
| Flip Rate (non-paraphrases) | 66.7% |
| Forward Accuracy | 58.0% |

![PAWS Flip Rate](figures/paws_flip_rate_claude-haiku-4-5-20251001.png)

### Interpretation

The model is consistent on only about half of questions (52.6%), meaning that for nearly half the benchmark, rephrasing the same question changes the model's answer. A Krippendorff's α of 0.509 indicates only moderate inter-variant agreement — well below the 0.8 threshold typically associated with reliable annotation — signaling that surface-level phrasing has a meaningful influence on model outputs. The PAWS results reveal an important asymmetry: the model flips its paraphrase judgment on 25.6% of true paraphrases but on 66.7% of adversarial non-paraphrases, suggesting the model does capture some semantic signal but remains highly sensitive to superficial lexical and syntactic variation. Taken together, these findings raise concerns for high-stakes deployment contexts where prompt wording may not be controlled.

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and fill in API keys
cp .env.example .env

# Run MMLU track (100 questions, 3 paraphrases each)
python run_benchmark.py --track mmlu --model claude-haiku-4-5-20251001 --n 100

# Run PAWS track (200 pairs)
python run_benchmark.py --track paws --model claude-haiku-4-5-20251001 --n 200

# Run both tracks
python run_benchmark.py --track both --model claude-haiku-4-5-20251001

# OpenAI model
python run_benchmark.py --track mmlu --model gpt-4o-mini --provider openai --n 100
```

## Project Structure

```
lm-consistency-benchmark/
├── src/
│   ├── data_loader.py    # PAWS + MMLU loading (HuggingFace, no download needed)
│   ├── paraphrase.py     # LLM-based paraphrase generation with caching
│   ├── evaluate.py       # Model querying for both tracks
│   ├── metrics.py        # Consistency rate, Krippendorff's α, flip rate, etc.
│   └── visualize.py      # All plots saved to /figures
├── results/              # Cached model outputs (gitignored)
├── figures/              # Generated plots
├── notebooks/            # Exploratory analysis
├── run_benchmark.py      # Main CLI entry point
└── requirements.txt
```

## Metrics Reference

| Metric | Description |
|---|---|
| Consistency Rate | % of questions answered the same across all variants |
| Accuracy | % of answers that are correct |
| Krippendorff's α | Inter-rater reliability across variants (nominal scale) |
| Flip Rate | % of PAWS pairs where answer changes with input order |
| Accuracy Drop | Accuracy(original) − Accuracy(paraphrases) |
