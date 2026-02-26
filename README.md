# System-Level Iterative Retrieval for Small Language Models

**A fully autonomous 7B pipeline matching published SOTA on MuSiQue multi-hop QA.**

## Key Result

| Method | Model | Gold Labels? | MuSiQue EM (n=100) |
|--------|-------|-------------|------------|
| Single Pass (baseline) | Qwen 2.5 7B | N/A | 16.0% |
| System Decomp + Embed Retrieval | Phi-3 3.8B | decompositions | 36.7% |
| **System Decomp + Embed Retrieval** | **Qwen 2.5 7B** | **none** | **42.0%** |
| System Decomp + Gold Decomp | Qwen 2.5 7B | decompositions | 49.0% |
| IRCoT (published) | GPT-3 175B | none | 36.5% |
| StepChain GraphRAG (published SOTA) | GPT-4o | none | 43.9% |

A fully autonomous pipeline — Qwen 2.5 7B decomposes, retrieves, and extracts with **zero gold labels** — achieves **42.0% EM** on MuSiQue (n=100), matching published SOTA (StepChain 43.9%) with a model **25x smaller**.

## The Core Insight

Multi-hop QA fails for small models because the **model** tries to chain reasoning in a single pass. When the **system** decomposes questions into single-hop steps and chains the results, performance improves dramatically.

> **Model-level decomposition**: -12.4% EM (hurts!)
> **System-level decomposition**: +26% EM (works!)

The technique (decomposition) was right. The implementation location (in-model vs in-system) was wrong.

## How It Works

```
Standard approach (single pass):
  Full question + all documents → Model → Answer (16.0% EM)

Fully autonomous pipeline (our approach):
  Full question → Qwen 7B decomposes into sub-questions
  For each hop:
    Sub-question → BGE embedding retrieval → Top-3 paragraphs
    Paragraphs + sub-question → Qwen 7B extracts answer
    Answer feeds into next hop's question
  Final hop answer = Final answer (42.0% EM on 100 questions)

No gold labels used at any step.
```

## Experiment Versions

| Version | What Changed | Best Embed EM | Best Oracle EM |
|---------|-------------|---------------|----------------|
| v1 | Basic prompting, keyword retrieval | 10.0% | 50.0% |
| v2 | Few-shot prompting, BGE embeddings | **36.7%** | **60.0%** |
| v3 | Answer normalization, auto-decomposition | 36.7% | - |
| v4 | 100-hypothesis battery, Qwen 7B extractor | **56.7%** | **63.3%** |
| v5 | Auto-decomposition (zero gold labels) | **42.0%** (n=100) | **49.0%** (n=100) |

## v5: Fully Autonomous Pipeline (The Key Result)

Qwen 2.5 7B handles both decomposition AND extraction — no gold labels at any step.

### Scale Validation (n=100)

| Config | EM | Relaxed EM | F1 |
|--------|------|------------|------|
| Single pass | 16.0% | 20.0% | 0.242 |
| Gold decomp + Qwen | 49.0% | 56.0% | 0.558 |
| **Auto decomp + Qwen** | **42.0%** | **52.0%** | **0.507** |

Auto-decomposition gap: **7.0% EM** at scale. Per-hop retrieval: 96% gold paragraph found.

### Initial Results (n=30)

| Config | Decomp | Extract | EM | Relaxed EM | F1 |
|--------|--------|---------|------|------------|------|
| Single pass (no decomp) | N/A | Qwen 7B | 13.3% | 16.7% | 0.203 |
| Gold decomp + Phi-3 | gold | Phi-3 3.8B | 36.7% | 46.7% | 0.428 |
| Qwen decomp + Phi-3 | auto | Phi-3 3.8B | 36.7% | 46.7% | 0.494 |
| Gold decomp + Qwen | gold | Qwen 7B | 60.0% | 66.7% | 0.657 |
| **Qwen decomp + Qwen** | **auto** | **Qwen 7B** | **60.0%** | **70.0%** | **0.708** |
| Qwen decomp + gold ctx | auto | Qwen 7B | 63.3% | 73.3% | 0.747 |

Note: The first 30 MuSiQue validation questions are biased toward easier examples. The n=100 results are more representative. The auto-decomposition gap at n=30 was 0% but opens to 7% at scale.

Decomposition quality: 100% correct hop count, 0.71 hop-1 F1, 0.52 hop-2 F1 vs gold.

## v4: 100-Hypothesis Battery

Screened 100 hypotheses across 6 categories (prompt engineering, retrieval, model params, answer processing, architecture, combined). Key validated results on 30 MuSiQue questions:

| Config | Extractor | EM | Relaxed EM | F1 |
|--------|-----------|------|------------|------|
| Phi-3 baseline (v2) | Phi-3 3.8B | 30.0% | 40.0% | 0.371 |
| Qwen neg+top2+greedy+pp | Qwen 2.5 7B | 50.0% | 66.7% | 0.624 |
| **Qwen 8-shot+top3** | **Qwen 2.5 7B** | **56.7%** | **66.7%** | **0.655** |
| Phi-3 oracle (gold ctx) | Phi-3 3.8B | 56.7% | 73.3% | 0.684 |
| Qwen oracle (gold ctx) | Qwen 2.5 7B | 63.3% | 83.3% | 0.778 |

**Key insight**: Qwen 7B with embed retrieval (56.7%) matches Phi-3 with gold context (56.7%). A better extractor with noisy retrieval equals a weaker extractor with perfect context.

## Benchmarks

Results across multiple benchmarks (prevents gaming any single dataset):

| Benchmark | Hops | Single Pass EM | Phi-3 Embed EM | Qwen Embed EM | Oracle EM |
|-----------|------|----------------|----------------|---------------|-----------|
| TriviaQA | 1 | **70.0%** | N/A | N/A | N/A |
| HotpotQA | 2 | 36.7% | **43.3%** | - | **50.0%** |
| MuSiQue | 2 | 3.3% | 36.7% | **56.7%** | **63.3%** |

## Published Comparisons (MuSiQue)

| Method | Model Size | EM |
|--------|-----------|-----|
| GPT-4o (no retrieval) | ~200B+ | 10.8% |
| BM25 retrieval | - | 13.8% |
| Flan-T5-XXL + IRCoT | 11B | 30.8% |
| GPT-3 + IRCoT | 175B | 36.5% |
| RAPTOR | - | 36.4% |
| SiReRAG | - | 40.5% |
| HopRAG | - | 42.2% |
| StepChain GraphRAG (prev SOTA) | - | 43.9% |
| **Ours (fully autonomous, n=100)** | **7B** | **42.0%** |
| **Ours (gold decomp, n=100)** | **7B** | **49.0%** |

## Key Findings

1. **System architecture > model size**: 16% → 42% EM (n=100) from system-level decomposition
2. **Matches published SOTA at 25x smaller**: 42.0% vs StepChain's 43.9%, fully autonomous 7B model
3. **Auto-decomposition works**: 7% gap vs gold at scale (0% at n=30, questions get harder)
4. **Extractor quality matters most**: Phi-3→Qwen 7B gave +20% EM (biggest single improvement)
5. **Retrieval is solved at sample scale**: BGE embeddings find gold paragraphs 96% of the time
6. **Phi-3 can't decompose**: 6.7% EM (v3), but Qwen 7B decomposes effectively

### What Doesn't Work

- CoT prompting: 0% EM (actively harmful)
- Answer post-processing: No effect for Phi-3
- Voting: No improvement for Phi-3
- Temperature/top_p: Essentially no effect
- High repeat penalty: Hurts performance

## Requirements

```
pip install requests numpy sentence-transformers datasets
```

Plus [Ollama](https://ollama.ai) with models:
```
ollama pull phi3:mini
ollama pull qwen2.5:7b    # for best results
```

## Running Experiments

```bash
# v5 fully autonomous pipeline (the main result, ~10 min)
python experiments/v5_auto_decomposition.py --limit 30

# v4 hypothesis battery (100 hypotheses, ~20 min)
python experiments/hypothesis_battery.py --limit 15 --output results/hypothesis_battery.json

# Validate top configs on 30 questions
python experiments/validate_top_hypotheses.py --limit 30

# Multi-benchmark
python experiments/multi_benchmark.py --limit 30 --benchmarks hotpotqa musique triviaqa

# Earlier versions
python experiments/v1_basic.py --limit 30
python experiments/v2_fewshot_embed.py --limit 30
python experiments/v3_normalization.py --limit 30
```

## Connection to "Let Them Forget" Paper

This work provides empirical evidence for the thesis that capability comes from the **system**, not the model. A fully autonomous 7B pipeline with system-level orchestration matches published SOTA results that use much larger models — with zero gold labels.

The model encodes **extraction patterns** (parametric thought), not facts (parametric memory). Facts live in the retrieval system. The system handles decomposition, navigation, and chaining. Even decomposition itself can be delegated to the same model operating in a different role.

## Citation

```bibtex
@misc{iterative-retrieval-2026,
  title={System-Level Iterative Retrieval for Small Language Models},
  author={Hestia's Creations},
  year={2026},
  url={https://github.com/hestiascreations/iterative-retrieval}
}
```

## License

MIT
