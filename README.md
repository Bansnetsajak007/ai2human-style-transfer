# ai2human-style-transfer

> **Can a fine-tuned transformer learn to write like a human?**  
> A comparative study of T5 and Gemma 2B for AI-to-Human text style transfer.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org/)

---

## What This Project Is About

Large language models write differently from humans. Their text is formal, uniform, and structurally predictable in ways that make it statistically detectable. This project asks a simple question: can we fine-tune a transformer to reverse that process, taking AI-generated text and rewriting it so it genuinely reads like something a person wrote?

We build a parallel corpus of 9,975 paired AI-Human passages, identify 11 linguistic markers that separate the two styles, and train both encoder-decoder (T5) and decoder-only (Gemma 2B) models on this corpus. The results show that Gemma 2B, fine-tuned with QLoRA, achieves a BERTScore F1 of 0.868 while shifting all 11 linguistic markers toward human distributions, including an 81% improvement in readability and the introduction of natural contractions from zero to 2.12 per passage.

---

## Key Findings

| Metric | AI Input | Humanized Output | Change |
|--------|----------|-----------------|--------|
| Contractions per passage | 0.01 | 2.12 | +99% |
| Flesch Reading Ease | 33.00 | 59.73 | +81% |
| Sentence Length Variance | 43.93 | 76.38 | +74% |
| Avg Word Length (chars) | 5.51 | 4.63 | -16% |
| Grade Level | 14.15 | 10.53 | -26% |
| BERTScore F1 | | **0.868** | |

---

## Example Output

**AI Input:**
> "Artificial General Intelligence (AGI) refers to a highly advanced form of artificial intelligence that possesses the ability to understand, learn, and apply knowledge across a wide range of tasks at a level comparable to or exceeding that of human intelligence."

**Humanized Output:**
> "What would you think if I told you there was a type of Artificial Intelligence called AGI? AGI is a very powerful type of technology that could potentially help us in many ways. The difference between narrow AI and AGI is that narrow AI has only one purpose while AGI can solve any kind of problem at any time."

---

## Project Structure

```
ai2human-style-transfer/
│
├── data/                          # All dataset files
│   ├── paired_dataset_clean.csv   # Final cleaned parallel corpus (9,975 pairs)
│   ├── paired_dataset_clean_10k.csv
│   ├── paired_dataset_clean_5k.csv
│   ├── human_sentences_10k.txt    # Raw human passages
│   ├── human_sentences_10k_fixed.txt
│   ├── train.csv                  # Train split
│   ├── test_Dataset.txt           # Manual test cases
│   ├── temp_test_10dataset.csv    # Quick test outputs
│   └── progress.txt               # Pipeline checkpoint file
│
├── DecoderInference/              # Gemma 2B inference setup
│   ├── HumanizerModel/            # Fine-tuned Gemma 2B LoRA weights
│   ├── HumanizerTokenizer/        # Saved tokenizer
│   └── inference.py               # Run humanization on any text
│
├── notebooks/                     # All Jupyter notebooks
│   ├── ai_text_generator.ipynb    # Data generation pipeline (NVIDIA API)
│   ├── HumanizerFineTune.ipynb    # Gemma 2B QLoRA fine-tuning
│   └── check.ipynb                # Dataset verification and analysis
│
├── scripts/
│   └── visualize.py               # Linguistic marker visualizations
│
├── docs/
│   └── main.tex                   # Full research paper (LaTeX)
│
├── deep_analysis_5k.png           # Visualization: 5k dataset analysis
├── deep_analysis.png              # Visualization: full dataset analysis
├── .gitignore
└── README.md
```

---

## Linguistic Markers Discovered

Through systematic corpus analysis of 9,975 paired passages, we identify 11 features that reliably separate human from AI writing:

| Marker | Human | AI | Gap |
|--------|-------|----|-----|
| Avg Word Count | 418.4 | 331.3 | -21% |
| Avg Sentence Count | 20.7 | 16.4 | -21% |
| Avg Word Length | 4.53 chars | 5.55 chars | +23% |
| Lexical Diversity | 0.482 | 0.550 | +14% |
| Contractions | 2.64 | 0.01 | -99% |
| Question Marks | 0.80 | 0.11 | -86% |
| Exclamations | 0.30 | 0.01 | -97% |
| Commas | 15.82 | 20.89 | +32% |
| Sentence Variance | 163.93 | 49.75 | -70% |
| Flesch Reading Ease | 63.09 | 29.91 | -53% |
| Grade Level | 9.84 | 14.25 | +45% |

The most discriminative single feature is **contraction usage**, which shows a 99% reduction in AI text and serves as a near-perfect stylistic fingerprint.

---

## Models Trained

### T5-Small (60M parameters)
- Encoder-decoder baseline
- 5k pairs, 5 epochs, Kaggle 2x T4 GPU
- Final validation loss: 5.986
- Result: minor vocabulary simplification only

### T5-Base (220M parameters)
- Encoder-decoder improved baseline
- 10k pairs, 5 epochs, cosine LR schedule
- Final validation loss: 4.520
- Result: better but still limited to surface changes

### Gemma 2B with QLoRA (2.6B parameters)
- Decoder-only, instruction-tuned
- 9,975 pairs chunked to 24,602 training samples
- 3 epochs, 500 steps, 0.79% parameters trained
- BERTScore F1: **0.868**
- Result: full stylistic transformation across all 11 markers

---

## How We Built the Dataset

```
305,797 human passages (Kaggle AI vs Human dataset)
        ↓
Random sample 10,000 passages (seed=42)
        ↓
Generate AI version of each using LLaMA 3.2 3B (NVIDIA NIM API)
        ↓
Deduplicate → 9,975 unique pairs
        ↓
Sentence-aware chunking → 24,602 training chunks
        ↓
Train / Val / Test split (80 / 10 / 10)
```

The generation pipeline includes automatic progress checkpointing every 50 examples, 3 retry attempts per sample, and rate limit handling. Total generation time: approximately 11 hours.

---

## Getting Started

### Installation

```bash
git clone https://github.com/yourusername/ai2human-style-transfer.git
cd ai2human-style-transfer
pip install -r requirements.txt
```

### Run Inference on Your Own Text

```python
from DecoderInference.inference import humanize

ai_text = """
Artificial intelligence systems demonstrate remarkable 
capabilities in natural language understanding and generation,
enabling applications across diverse domains.
"""

humanize(ai_text)
```

### Run the Full Pipeline

```bash
# Step 1: Generate AI versions of human passages
jupyter notebook notebooks/ai_text_generator.ipynb

# Step 2: Analyze and visualize the dataset
python scripts/visualize.py

# Step 3: Fine-tune Gemma 2B
jupyter notebook notebooks/HumanizerFineTune.ipynb
```

---

## Requirements

```
torch>=2.0.0
transformers>=4.38.0
datasets>=2.18.0
peft>=0.10.0
bitsandbytes>=0.43.0
unsloth
trl>=0.8.0
accelerate>=0.27.0
sentencepiece
bert-score
textstat
nltk
pandas
numpy
matplotlib
seaborn
scikit-learn
openai
wordcloud
```

Install everything at once:
```bash
pip install -r requirements.txt
```

---

## Hardware Used

| Task | Hardware | Cost |
|------|----------|------|
| Data generation | NVIDIA NIM API (free tier) | $0 |
| T5 fine-tuning | Kaggle 2x T4 GPU (free tier) | $0 |
| Gemma 2B fine-tuning | Local GPU | $0 |
| Evaluation | Local CPU | $0 |

**Total compute cost: $0.** This project demonstrates that meaningful NLP research is possible without a large budget.

---

## Research Paper

The full research paper is available in `docs/main.tex` and on arXiv.

**Title:** From Machine to Human: A Comparative Study of Seq2Seq and Decoder-Only Transformers for AI-to-Human Text Style Transfer

**Abstract:** We present a systematic comparative study of transformer architectures for AI-to-Human text style transfer. Through empirical analysis of 9,975 parallel pairs, we identify 11 discriminative linguistic markers and demonstrate that fine-tuned Gemma 2B achieves BERTScore F1 of 0.868 while shifting all markers toward human distributions within 500 training steps.

To cite this work:
```bibtex
@article{basnet2025ai2human,
  title={From Machine to Human: A Comparative Study of Seq2Seq 
         and Decoder-Only Transformers for AI-to-Human Text Style Transfer},
  author={Basnet, Sajak},
  journal={arXiv preprint},
  year={2025}
}
```

---

## Results Visualization

The `deep_analysis.png` file contains six plots comparing human and AI text distributions across word count, sentence structure, punctuation usage, readability, burstiness, and word clouds.

The `deep_analysis_5k.png` shows the same analysis on the 5k subset, confirming that the linguistic patterns are stable across dataset sizes.

---

## Limitations

- Dataset is limited to student essays and argumentative writing
- AI corpus generated by a single model (LLaMA 3.2 3B)
- Qualitative evaluation conducted on 15 test passages
- No formal human preference study included
- Outputs not yet tested against live AI detectors

---

## Future Work

- Multi-model corpus generation (GPT-4, Claude, Gemini, LLaMA)
- Genre-diverse datasets including news, technical writing, fiction
- Human preference evaluation study
- Testing against AI detectors (GPTZero, Originality.ai)
- Scaling experiments with Gemma 7B and 27B

---

## Author

**Sajak Basnet**  
Aspiring AI Researcher | NLP enthusiast  
Kathmandu, Nepal  


*Built with zero budget, a lot of patience, and way too many GPU hours on Kaggle free tier.*
