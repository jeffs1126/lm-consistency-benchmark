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

*(Populated after running the benchmark)*

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
