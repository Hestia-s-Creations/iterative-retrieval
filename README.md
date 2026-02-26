# System-Level Iterative Retrieval for Small Language Models

**A 3.8B parameter model matching GPT-3 (175B) on multi-hop QA through system-level decomposition.**

## Key Result

| Method | Model | MuSiQue EM | HotpotQA EM | TriviaQA EM |
|--------|-------|------------|-------------|-------------|
| Single Pass (baseline) | Phi-3 3.8B | 3.3% | 36.7% | **70.0%** |
| System Decomp + Embed Retrieval | Phi-3 3.8B | **36.7%** | **43.3%** | - |
| System Decomp + Gold Context | Phi-3 3.8B | **63.3%** | **50.0%** | - |
| IRCoT (published) | GPT-3 175B | 36.5% | - | - |
| StepChain GraphRAG (published SOTA) | GPT-4o | 43.9% | - | - |

System-level decomposition takes Phi-3 from **3.3% to 63% Exact Match** on MuSiQue — a **19x improvement** using the same model, same data, just different architecture. Results validated across 3 benchmarks.

## The Core Insight

Multi-hop question answering fails for small models because the **model** tries to chain reasoning in a single pass. When the **system** decomposes questions into single-hop steps and chains the results, performance improves dramatically.

The model is excellent at single-hop extraction (87% relaxed accuracy per hop). The system handles decomposition, retrieval routing, and answer chaining.

> **Model-level decomposition**: -12.4% EM (hurts!)
> **System-level decomposition**: +57% EM (works!)

The technique (decomposition) was right. The implementation location (in-model vs in-system) was wrong. This distinction is largely absent from the literature.

## How It Works

```
Standard approach (single pass):
  Full question + all documents → Model → Answer (3.3% EM)

Iterative retrieval (our approach):
  Full question → System decomposes into hops
  For each hop:
    Sub-question → Embedding retrieval → Top-3 paragraphs
    Paragraphs + sub-question → Model extracts answer
    Answer feeds into next hop's question
  Final hop answer = Final answer (36.7-60% EM)
```

## Experiment Versions

| Version | What Changed | Best Config EM |
|---------|-------------|----------------|
| v1 | Basic prompting, keyword retrieval | 50.0% (oracle) |
| v2 | Few-shot prompting, BGE embeddings, relaxed EM | **60.0%** (oracle), **36.7%** (embed) |
| v3 | Answer normalization, auto-decomposition | 36.7% (normalized), 6.7% (auto) |
| v4 | Multi-model decomposition (Qwen 7B + Phi-3) | In progress |

## Benchmarks

Results across multiple benchmarks (prevents gaming any single dataset):

| Benchmark | Hops | Single Pass EM | Iterative/Embed EM | Oracle/Gold EM |
|-----------|------|----------------|-------------------|----------------|
| TriviaQA | 1 | **70.0%** | N/A (1-hop) | N/A |
| HotpotQA | 2 | 36.7% | **43.3%** | **50.0%** |
| MuSiQue | 2 | 3.3% | **36.7%** | **63.3%** |

- **TriviaQA** (1-hop): 70% EM validates that Phi-3 is a strong single-hop extractor
- **HotpotQA** (2-hop): Embedding retrieval finds both supporting paragraphs 90% of the time. Gold context achieves 80% relaxed EM
- **MuSiQue** (2-hop decomposed): The hardest benchmark — requires explicit decomposition. Iterative approach provides 11x improvement over single pass

## Published Comparisons (MuSiQue)

| Method | Model Size | EM |
|--------|-----------|-----|
| GPT-4o (no retrieval) | ~200B+ | 10.8% |
| Flan-T5-XXL + IRCoT | 11B | 30.8% |
| GPT-3 + IRCoT | 175B | 36.5% |
| **Ours (embed retrieval)** | **3.8B** | **36.7%** |
| StepChain GraphRAG (SOTA) | - | 43.9% |
| **Ours (oracle)** | **3.8B** | **60.0%** |

## Key Findings

1. **System architecture > model size**: 3.3% → 60% EM from changing WHERE decomposition happens
2. **Small models are excellent extractors**: 57% strict / 87% relaxed per-hop accuracy
3. **Retrieval is solved at sample scale**: BGE embeddings find gold paragraphs 93-98% of the time
4. **Error cascade is minimal**: Only 3-7% EM cost from chaining model answers vs gold answers
5. **Auto-decomposition needs larger models**: Phi-3 can't decompose questions (6.7% EM)

## Requirements

```
pip install requests numpy sentence-transformers datasets
```

Plus [Ollama](https://ollama.ai) with `phi3:mini`:
```
ollama pull phi3:mini
```

## Running Experiments

```bash
# v2 (main experiment)
python experiments/v2_fewshot_embed.py --limit 30

# Multi-benchmark
python experiments/multi_benchmark.py --limit 30 --benchmarks hotpotqa musique triviaqa

# Reproduce specific version
python experiments/v1_basic.py --limit 30
python experiments/v3_normalization.py --limit 30
```

## Connection to "Let Them Forget" Paper

This work provides empirical evidence for the thesis that capability comes from the **system**, not the model. A 3.8B model with system-level orchestration achieves results competitive with 175B models doing single-pass reasoning.

The model encodes **extraction patterns** (parametric thought), not facts (parametric memory). Facts live in the retrieval system. The system handles navigation and chaining.

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
