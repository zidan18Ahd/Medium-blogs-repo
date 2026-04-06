# Attention Sink & KeyNorm — GPT-2 Analysis

A focused experiment visualizing and mitigating the attention sink phenomenon in transformer models, using KeyNorm (L2 normalization of Q/K vectors) as a post-hoc intervention on GPT-2.

---

## What This Project Does

This notebook investigates a well-known but under-discussed behavior in large language models called the **attention sink** — where transformer attention heads disproportionately focus on the very first token in a sequence, often regardless of its semantic relevance.

The notebook:
- Loads a pretrained GPT-2 model
- Runs a sample sentence through all 12 attention layers
- Computes the sink score (how much the last query attends to the first token) for both the standard baseline and KeyNorm
- Produces two visualizations comparing the two approaches across all layers

---

## Background: What Is Attention Sink?

In standard transformer attention, weights are computed as:

```
Attention(Q, K, V) = softmax(QKᵀ / √d) · V
```

A consistent finding in LLMs is that the first token — often a BOS token, punctuation, or simply the first word — accumulates a massive share of attention probability mass, even when it carries no semantic information relevant to the query. This is the attention sink.

The reason this happens comes down to how softmax behaves. Since attention weights must sum to 1, the model has to put probability mass somewhere even when it has no strong signal. Over training, it learns to route that uncertainty to a stable, low-information token — the first one in the sequence. It is essentially the model's way of saying it does not know where to look.

This matters because it wastes representational capacity, distorts interpretability, and creates a hard constraint for KV-cache compression: tokens with high attention mass cannot be pruned without degrading output quality.

---

## The Fix: KeyNorm

KeyNorm applies L2 normalization to the query and key vectors before computing attention logits:

```python
q_norm = F.normalize(q, p=2, dim=-1)
k_norm = F.normalize(k, p=2, dim=-1)
logits = torch.matmul(q_norm, k_norm.transpose(-2, -1)) / sqrt(d)
```

By constraining all Q and K vectors to the unit hypersphere, magnitude is removed as a factor. Attention becomes driven purely by directional similarity, which directly prevents any single token from dominating based on geometric coincidence rather than semantic relevance.

---

## Key Results

| Method   | Average Sink Score (across 12 layers) |
|----------|----------------------------------------|
| Baseline | 0.8125 (81.25%)                        |
| KeyNorm  | 0.1152 (11.52%)                        |

KeyNorm reduces the attention sink by approximately 86% — from over 80% of attention mass concentrated on token 0, down to around 11%.

---

## Project Structure

```
.
├── notebook.ipynb         # Main experiment notebook
├── sink_per_layer.png     # Per-layer sink bar chart (Baseline vs KeyNorm)
├── heatmap_layer5.png     # Per-token attention heatmap at Layer 6
└── README.md
```

---

## Getting Started

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU

### Installation

```bash
pip install torch>=2.0.0 transformers>=4.35.0 numpy>=1.24.0 matplotlib>=3.7.0 seaborn>=0.12.0
```

### Run

```bash
jupyter notebook notebook.ipynb
```

The notebook was originally run on a Kaggle T4 GPU, but will work on CPU for this sentence-length input.

---

## Code Walkthrough

### Step 1 — Load Model

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
text = "The Eiffel Tower is located in Paris"
inputs = tokenizer(text, return_tensors="pt").to(device)
```

GPT-2 base (117M parameters, 12 layers, 12 heads, d=768) is loaded in eval mode with no gradient tracking.

### Step 2 — Compute Sink Score Per Layer

For each of the 12 layers, the notebook extracts Q and K from the attention projection, computes attention probabilities for the last query position (the most informative, since causal attention allows it to attend to all prior tokens), averages across all 12 heads, and records how much probability mass lands on token 0.

This is done twice: once with raw Q/K (baseline), once with L2-normalized Q/K (KeyNorm).

### Step 3 — Visualize

Two plots are produced:

- **sink_per_layer.png** — Side-by-side bar chart showing baseline vs KeyNorm sink score at every layer
- **heatmap_layer5.png** — Full per-token attention distribution at Layer 6, showing exactly where attention goes for each word in the sentence under both methods

---

## Interpreting the Results

Across nearly all 12 layers, baseline sink scores are consistently between 0.7 and 0.9. The first token ("The") consumes the vast majority of attention regardless of what the model is trying to predict.

With KeyNorm applied, scores collapse to the 0.05–0.2 range. Attention spreads meaningfully across semantically relevant tokens. "Eiffel," "Tower," and "Paris" all receive weight proportional to their relevance.

The Layer 6 heatmap makes this contrast concrete. Baseline attention is a near-vertical spike at position 0. KeyNorm produces a distributed pattern that actually reflects the semantic structure of the sentence.

---

## Potential Extensions

- Run on larger models: GPT-2 Medium/Large, LLaMA-2, Mistral
- Measure perplexity impact when applying KeyNorm during training from scratch rather than post-hoc
- Compare with other attention regularization variants such as SoftmaxOff, clipped attention, and learned scale parameters
- Analyze per-head heterogeneity to identify which heads are dedicated sink heads vs semantic heads
- Apply to sequences with an explicit BOS token and compare sink behavior
- Combine with StreamingLLM and test whether reduced sink scores enable more aggressive KV-cache pruning

---

## References

1. Xiao, G., Tian, Y., Chen, B., Han, S., & Lewis, M. (2023). Efficient Streaming Language Models with Attention Sinks. *arXiv:2309.17453*. https://arxiv.org/abs/2309.17453

2. Darcet, T., Oquab, M., Mairal, J., & Bourdoukan, R. (2023). Vision Transformers Need Registers. *arXiv:2309.16588*. https://arxiv.org/abs/2309.16588

3. He, B., Zheng, M., Dong, B., & Tao, D. (2023). Simplifying Transformer Blocks. *arXiv:2311.01906*. https://arxiv.org/abs/2311.01906

4. Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding. *arXiv:2104.09864*. https://arxiv.org/abs/2104.09864

5. Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebrón, F., & Sanghai, S. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. *arXiv:2305.13245*. https://arxiv.org/abs/2305.13245

6. Wortsman, M., Lee, J., Gilmer, J., & Kornblith, S. (2023). Replacing softmax with ReLU in Vision Transformers. *arXiv:2309.08586*. https://arxiv.org/abs/2309.08586

7. Jiang, A., et al. (2023). Mistral 7B. *arXiv:2310.06825*. https://arxiv.org/abs/2310.06825

8. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language Models are Unsupervised Multitask Learners. OpenAI Blog. https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

---

## Tech Stack

| Tool                     | Purpose                            |
|--------------------------|------------------------------------|
| PyTorch                  | Model inference and attention math |
| HuggingFace Transformers | GPT-2 weights and tokenizer        |
| Matplotlib               | Visualizations                     |
| NumPy                    | Numerical operations               |
| Kaggle T4 GPU            | Runtime environment                |

---

## License

MIT
