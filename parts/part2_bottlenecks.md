# Part 2: Bottlenecks — Why LLM Inference is So Difficult

## Table of Contents
- [Chapter 4: Core Metrics: Measuring the "Ruler" of Large Model Inference](#chapter-4-core-metrics-measuring-the-ruler-of-large-model-inference)
- [Chapter 5: Starting from Scratch: How Naive LLM Inference Operates](#chapter-5-starting-from-scratch-how-naive-llm-inference-operates)
  - [Section 1: The Unoptimized Inference Workflow](#section-1-the-unoptimized-inference-workflow)
  - [Section 2: Bottleneck Anatomy: Explosion of Computational Complexity](#section-2-bottleneck-anatomy-explosion-of-computational-complexity)
  - [Section 3: Eliciting Optimizations: Can We "Remember" Past Calculations?](#section-3-eliciting-optimizations-can-we-remember-past-calculations)
- [Chapter 6: The Rule-Breaker: KV Cache and the Resulting "Memory Tsunami"](#chapter-6-the-rule-breaker-kv-cache-and-the-resulting-memory-tsunami)
  - [Section 1: Trading Space for Time: Caching K and V](#section-1-trading-space-for-time-caching-k-and-v)
  - [Section 2: Why Only K and V?](#section-2-why-only-k-and-v)
  - [Section 3: The Memory Tsunami: A Multi-Terabyte Math Problem](#section-3-the-memory-tsunami-a-multi-terabyte-math-problem)
- [Chapter 7: Maximizing GPU Utilization: The Evolution of Batching](#chapter-7-maximizing-gpu-utilization-the-evolution-of-batching)
  - [Section 1: Compute-Bound vs. Memory-Bound](#section-1-compute-bound-vs-memory-bound)
  - [Section 2: Batched Matrix Multiplication (BMM)](#section-2-batched-matrix-multiplication-bmm)
  - [Section 3: The Padding Problem: Flaws in Static Batching](#section-3-the-padding-problem-flaws-in-static-batching)
- [Chapter 8: Core Asymmetry: Prefill vs. Decode](#chapter-8-core-asymmetry-prefill-vs-decode)
  - [Section 1: Prefill Phase — The "Blitzkrieg" Consuming Compute (Compute-Bound)](#section-1-prefill-phase--the-blitzkrieg-consuming-compute-compute-bound)
  - [Section 2: Decode Phase — The "War of Attrition" Crushing Bandwidth (Memory-Bound)](#section-2-decode-phase--the-war-of-attrition-crushing-bandwidth-memory-bound)
  - [Section 3: Asymmetry from a Data Perspective](#section-3-asymmetry-from-a-data-perspective)

This part explains the physical and mathematical "brick walls" engineers hit when pushing LLMs into production.

---

## Chapter 4: Core Metrics: Measuring the "Ruler" of Large Model Inference

Before discussing large model inference optimizations, we must establish evaluation criteria. The step-by-step autoregressive trait means we cannot gauge performance merely by using "Response Time" of traditional Web services. This chapter outlines core performance metrics.

*   **TTFT (Time to First Token)**: The latency from when a user dispatches a request to when the model outputs the **first word**. It corresponds to the **Prefill phase**, determining whether the system is responsive.
*   **TBT (Time Between Tokens)**: Generation latency between two adjacent Tokens. It corresponds to the **Decode phase**, dictating streaming output fluency.
*   **TPS (Tokens Per Second)**: Measures how many Tokens the model emits per second. It is the inverse of TBT ($TPS = 1 / TBT$).
*   **Latency**: Total time taken to conclude the full request. Computed as $\text{Latency} = \text{TTFT} + (\text{GeneratedTokens} - 1) \times \text{TBT}$.
*   **Throughput**: Total number of tokens or requests processed by the server per second. It stands as the core indicator for gauging concurrency and Total Cost of Ownership (TCO).

---

## Chapter 5: Starting from Scratch: How Naive LLM Inference Operates

Before introducing advanced optimizations, we must examine how "early ancestors" performed LLM inference to appreciate the severity of inference bottlenecks.

---

### Section 1: The Unoptimized Inference Workflow

Suppose the model generates text following the prompt "Large models are". The unoptimized generation flow proceeds as follows:

1.  **Generating Token 1**:
    *   **Input**: ["Large", "models", "are"]
    *   **Processing**: The entire sentence passes through 80 Transformer layers.
    *   **Output**: Predicts the next word is "the". We get ["Large", "models", "are", "the"].
2.  **Generating Token 2**:
    *   **Input**: ["Large", "models", "are", "the"]
    *   **Processing**: **The 4 words are fed back from layer 1**, traversing the full 80 layers again!
    *   **Output**: Predicts the next word is "future". We get ["Large", "models", "are", "the", "future"].
3.  **Generating Token 3**:
    *   **Input**: ["Large", "models", "are", "the", "future"]
    *   **Processing**: **Fed back yet again from layer 1**, traversing 80 layers.
    *   **Output**: Predicts the next word is ".".

---

### Section 2: Bottleneck Anatomy: Explosion of Computational Complexity

This sequential workflow is academically recognized as **Naive Inference**. Assuming sequence length is $N$, model hidden dimension is $d$, and layer count is $L$, a **single round** of Naive computation yields these dimensions:

1.  **Compute Complexity**: Single-round temporal complexity is $O(L \cdot (N \cdot d^2 + N^2 \cdot d))$
    *   **Linear Layer Compute**: $O(N \cdot d^2)$ per layer across $L$ layers. QKV projections and FFN mappings demand full matrix multiplications.
    *   **Attention Compute**: $O(N^2 \cdot d)$ across $L$ layers. Every word Query computes dot products against preceding Key vectors, generating $N \times N$ matrix multiplications.
2.  **Storage Complexity**: Space complexity is $O(L \cdot d^2) + O(N \cdot d + N^2)$
    *   **Static Weights**: $O(L \cdot d^2)$. Parameters matrices must reside in HBM perpetually, independent of context length.
    *   **Dynamic Activations**: $O(N \cdot d + N^2)$. Memory allocated temporarily for calculations and released instantly upon round completions.
3.  **Parallelization Capabilities**:
    *   **Prefill Phase**: Complete Prompt availability allows **highly parallelized** compute, saturating GPU multi-core compute capacities.
    *   **Decode Phase**: Autoregressively sequential, making steps **strictly serial**. Furthermore, Naive Inference re-computes preceding words in parallel during every serial round, causing extensive compute waste.

**Real-World Projection: Llama 3 (405B)**
Examining the SOTA **Llama 3 (405B)** ($d = 16384$, $L = 126$):
*   **Scenario**: Length $N = 1000$ words (processing a 1000-word prompt).
*   **Single-Round Compute**: Reaching $2 \times L \times (11 \times N \times d^2) \approx \mathbf{744 \text{ Trillion Floating-point Operations (TFLOPs)}}$.
*   **Estimated Time**: Leveraging an **NVIDIA H100** yielding 1 PFLOPS (FP16 theoretical peak), generating **1 single word** takes approximately $744 \div 1000 \approx \mathbf{0.74 \text{ Seconds}}$.
*   Naively processed, the model emits only 1 to 2 words per second, slowing further as contexts lengthen. Quadratic Attention scaling implies that for generating the 1,000,001st token, **generating a single word would consume approximately 2.5 hours**, making production serving wholly unviable.

---

### Section 3: Eliciting Optimizations: Can We "Remember" Past Calculations?

Naive inference effectively kills large model serving. It fails to handle long text sequences or tolerate concurrent workloads.

Engineers wondered: Can we cache the calculated results of preceding words and exclusively compute newly appended tokens? This intuition gave birth to the foundational cornerstone of LLM serving—**KV Cache**.

---

## Chapter 6: The Rule-Breaker: KV Cache and the Resulting "Memory Tsunami"

To break the $O(N^2)$ computational deadlock, KV Cache trades space for time, rewriting large model serving rules.

---

### Section 1: Trading Space for Time: Caching K and V

Returning to the Attention formula: $\text{Attention}(Q, K, V) = \text{softmax}(\frac{Q K^T}{\sqrt{d_k}})V$.
Engineers observed:
*   When generating new tokens, **Queries (Q)** are dynamic intents that must be generated by the newly arrived token.
*   However, **Keys (K)** and **Values (V)** of preceding words **never change** once generated.
*   Caching these "fixed assets" in VRAM means GPUs only need to compute Q, K, and V for **one new Token**. Newly generated K and V vectors append onto existing caches, bounding attention compute to $O(N)$ rather than $O(N^2)$.

---

### Section 2: Why Only K and V?

Why don't we cache Q? Because Query vectors represent "active intents" used up in individual steps. Predicting word 4 requires word 4's Q to search preceding words. Word 4's Q becomes useless during word 5's prediction. **Q is discarded, and only characteristic K and V vectors are cached.**

---

### Section 3: The Memory Tsunami: A Multi-Terabyte Math Problem

We compare **Naive** vs. **KV Cache** complexities against generating the $N\text{-th}$ Token ($L$ layers, $d$ dimension):

| Dimension | Naive Inference | KV Cache Mode |
| :--- | :--- | :--- |
| **Compute Complexity** | `O(L * (N * d^2 + N^2 * d))` | `O(L * (d^2 + N * d))` |
| **Storage Complexity** | `O(L * d^2)` (Weights Only) | `O(L * d^2 + L * N * d)` (Weights + Cache) |

**Core Divergence:**
1.  **Compute Dominance**: KV Cache cuts linear layer computations from $O(N)$ to $O(1)$, and attention computations from $O(N^2)$ to $O(N)$.
2.  **VRAM Space Penalty**: Compute drops at the cost of VRAM footprints scaling linearly against context length $N$ ($O(L \cdot N \cdot d)$).
3.  **Dynamic Activations**: Transient activation memory releases instantly upon forward passes and is omitted in long-term static analyses.

**Calculating the Tsunami via Llama 3 405B** ($L = 126, d = 16384$, unoptimized MHA, 1 Million Tokens):
*   **Footprint per Token**: K and V vectors across 126 layers consume `64 KB * 126 = 8064 KB (~8 MB)` in FP16 formats.
*   **Total Footprint**: `8064 KB * 1,000,000 \approx \mathbf{8.26 \text{ TB}}$`! A single request consumes over **8 Terabytes of VRAM**.

KV Cache shifts LLMs from "Compute-Bound" (stalling on raw arithmetic) to "Memory-Bound" (stalling on VRAM capacities). We resolve this monster using **GQA**, **PagedAttention**, and **RadixAttention**.

---

## Chapter 7: Maximizing GPU Utilization: The Evolution of Batching

Having resolved single-user compute via KV Cache, engineers faced concurrent workload demands.

---

### Section 1: Compute-Bound vs. Memory-Bound

Production LLMs are frequently starved of **Memory Bandwidth** rather than raw compute. Model parameters and KV Caches sit in VRAM, while compute takes place inside SMC cores.
*   **Single User Inference**: For every generated token, GPUs must read monolithic model weights and all accumulated KV Caches from VRAM to the cores. The core spends most of its time **idle, waiting for data transfers**.

---

### Section 2: Batched Matrix Multiplication (BMM)

To prevent core idling, we deploy **Batching**. We stack $N$ user inputs into a 3D tensor. GPUs fetch model weight matrices once and apply them to all $N$ user requests simultaneously, scaling up throughput.

> [!NOTE]
> While users share a single copy of model weights, their KV Caches remain strictly isolated. GPUs must drag $N$ discrete KV Caches from VRAM to cores, scaling VRAM transfers linearly against Batch Size.

---

### Section 3: The Padding Problem: Flaws in Static Batching

Standard **Static Batching** forces all requests in a batch to begin and end simultaneously. Since input and output lengths differ, engines insert massive volumes of **Padding** for shorter prompts. Computing padding wastes resources and chains shorter requests to wooden-barrel limits of the longest request.

---

## Chapter 8: Core Asymmetry: Prefill vs. Decode

---

### Section 1: Prefill Phase — The "Blitzkrieg" Consuming Compute (Compute-Bound)

**1. Mathematical Workflow**
Models absorb **$N$ input tokens** in one shot.
*   **Linear Layer**: $N$ vectors multiply against weight matrices. These are standard **GEMM** operations scaling against $N$ ($O(L \cdot N \cdot d^2)$).
*   **Attention**: Under Causal Mask bounds, each word only attends to preceding words, forming a lower-triangular $N \times N$ attention matrix ($O(L \cdot N^2 \cdot d)$).
*   **Complexity**: Total single-round complexity sums to $O(L \cdot (N \cdot d^2 + N^2 \cdot d))$.

**2. Why Is It Compute-Bound?**
This anchors around **Arithmetic Intensity** (FLOPs computed per Byte fetched).
$$\text{Intensity} = \frac{\text{Linear Compute} + \text{Attention Compute}}{\text{Model Weight Size} + \text{KV Cache Writes}}$$

Evaluating Llama 3 405B at $N = 100,000$:
*   **Total Compute**: $\approx 1.64 \times 10^{17}$ FLOPs.
*   **VRAM Traffic**: 810 GB (Weights) + 51.6 GB (KV Cache) $\approx 861.6$ GB.
*   **Final Intensity**: $1.64 \times 10^{17} / 861.6 \times 10^9 \approx \mathbf{190,000 \text{ FLOPs/Byte}}$.

A modern H100 GPU’s inflection threshold sits at roughly 300 FLOPs/Byte. Exceeding this threshold means GPUs are **Compute-Bound**. The thousands of ALU cores run at peak frequencies, and raw theoretical算力 (TFLOPS) dictates processing speed rather than HBM bandwidth.

---

### Section 2: Decode Phase — The "War of Attrition" Crushing Bandwidth (Memory-Bound)

Once the first word is output, autoregressive Decode iterations begin.

**1. Mathematical Workflow**
Models ingest merely **1 newly generated Token**.
*   **Linear Layer**: Degrades into **GEMV** matrix-vector multiplications.
*   **Attention**: The single new token's Query vector attends to all preceding $N$ Key vectors in the cache.
*   **Complexity**: $O(L \cdot (d^2 + N \cdot d))$.

**2. Why Is It Memory-Bound?**
To generate a **single token**, GPUs execute an absurd action: **they must drag monolithic model weights (hundreds of GBs) plus all accumulated KV Caches from VRAM to the SRAM cores!** Compute volumes are minuscule, leaving cores idle while HBM bandwidth is maxed out.

Evaluating Llama 3 405B at $N = 100,000$:
*   **Total Compute**: $\approx 1.635 \times 10^{12}$ FLOPs.
*   **VRAM Traffic**: 810 GB (Weights) + 51.6 GB (Cache) $\approx 861.6$ GB.
*   **Intensity**: $\approx \mathbf{1.9 \text{ FLOPs/Byte}}$.

Arithmetic intensity (1.9) falls vastly below hardware inflection points (~300). VRAM bandwidth—how fast data feeds cores—dictates TBT latencies, rather than raw TFLOPS capabilities.

---

### Section 3: Asymmetry from a Data Perspective

| Dimension | Prefill Phase | Decode Phase |
| :--- | :--- | :--- |
| **Input Scale** | $N$ Tokens (High) | $1$ Token (Minimal) |
| **Math Kernel** | GEMM | GEMV |
| **Complexity** | $O(L \cdot (N \cdot d^2 + N^2 \cdot d))$ | $O(L \cdot (d^2 + N \cdot d))$ |
| **VRAM Access** | Read Weights + Write Cache | Read Weights + Read & Append Cache |
| **Hardware Bound** | **Compute-Bound** | **Memory-Bound** |
| **GPU Utilization** | Peak | Idle |

**The Engineering Paradox:**
1.  **Throughput vs. Latency**: Elevating Batch Size dilutes weight reading costs to raise throughput. In Decode, larger batches translate to fetching larger discrete KV Caches, stretching out individual TBT latencies.
2.  **Scheduler Dilemmas**: Mixing compute-heavy Prefills alongside memory-heavy Decodes causes the **Straggler Effect**, triggering jitters in ongoing TBTs and breaking output fluency.

We resolve this paradox via **Continuous Batching** and **Chunked Prefill** in Part 3.

---
