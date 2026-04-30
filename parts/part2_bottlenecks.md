# Part 2: Bottlenecks — Why LLM Inference is Hard

## Table of Contents
- [Chapter 4: Performance Metrics: Measuring Inference Speed](#chapter-4-performance-metrics-measuring-inference-speed)
- [Chapter 5: Naive Inference: How Unoptimized Systems Work](#chapter-5-naive-inference-how-unoptimized-systems-work)
  - [Section 1: The Unoptimized Process](#section-1-the-unoptimized-process)
  - [Section 2: Complexity Analysis: The Computational Explosion](#section-2-complexity-analysis-the-computational-explosion)
  - [Section 3: The Solution: Caching Past Computations](#section-3-the-solution-caching-past-computations)
- [Chapter 6: KV Cache: Solving the Compute Bottleneck](#chapter-6-kv-cache-solving-the-compute-bottleneck)
  - [Section 1: The Principle: Caching K and V](#section-1-the-principle-caching-k-and-v)
  - [Section 2: The Scope: Why Only K and V?](#section-2-the-scope-why-only-k-and-v)
  - [Section 3: The Cost: The VRAM Tsunami](#section-3-the-cost-the-vram-tsunami)
- [Chapter 7: Batching: Maximizing GPU Utilization](#chapter-7-batching-maximizing-gpu-utilization)
  - [Section 1: The Bottleneck: Memory Bandwidth](#section-1-the-bottleneck-memory-bandwidth)
  - [Section 2: The Solution: Batched Matrix Multiplication (BMM)](#section-2-the-solution-batched-matrix-multiplication-bmm)
  - [Section 3: The Flaw: Static Batching and Padding](#section-3-the-flaw-static-batching-and-padding)
- [Chapter 8: Core Asymmetry: Prefill vs. Decode](#chapter-8-core-asymmetry-prefill-vs-decode)
  - [Section 1: Prefill Phase: The Compute-Bound Phase](#section-1-prefill-phase-the-compute-bound-phase)
  - [Section 2: Decode Phase: The Memory-Bound Phase](#section-2-decode-phase-the-memory-bound-phase)
  - [Section 3: The Asymmetry: Data Perspective Comparison](#section-3-the-asymmetry-data-perspective-comparison)

This part explains the physical and mathematical limits engineers face when putting LLMs into production.


## Chapter 4: Performance Metrics: Measuring Inference Speed

Traditional web "response time" cannot evaluate autoregressive LLMs that generate text token by token. We need specific metrics:

*   **TTFT (Time to First Token)**: Time from request submission to the first generated token. It corresponds to the **Prefill Phase** and determines perceived responsiveness.
*   **TBT (Time Between Tokens)**: Time between two consecutive tokens. It corresponds to the **Decode Phase** and determines streaming fluency.
*   **TPS (Tokens Per Second)**: Number of tokens generated per second ($TPS = 1 / TBT$).
*   **Latency**: Total time for the request. Formula: $Latency = TTFT + (\text{Tokens} - 1) \times TBT$.
*   **Throughput**: Total tokens or requests processed per second. This determines concurrency and Total Cost of Ownership (TCO).

---

## Chapter 5: Naive Inference: How Unoptimized Systems Work

To understand optimization, we must look at how basic inference works and why it fails at scale.

### Section 1: The Unoptimized Process

Assume the prompt is "Large models are". Generating the first three words without optimization works as follows:

1.  **Generate Word 1**:
    *   **Input**: ["Large", "models", "are"]
    *   **Process**: The prompt passes through all 80 Transformer layers.
    *   **Output**: "the". The sequence becomes ["Large", "models", "are", "the"].
2.  **Generate Word 2**:
    *   **Input**: ["Large", "models", "are", "the"]
    *   **Process**: The system feeds all 4 words back into the first layer and processes them through all 80 layers again.
    *   **Output**: "future". The sequence becomes ["Large", "models", "are", "the", "future"].
3.  **Generate Word 3**:
    *   **Input**: ["Large", "models", "are", "the", "future"]
    *   **Process**: The system feeds all 5 words back into the first layer and processes them through all 80 layers again.
    *   **Output**: ".".

### Section 2: Complexity Analysis: The Computational Explosion

This method is **Naive Inference**. Let $N$ be the current sequence length, $d$ the hidden dimension, and $L$ the number of layers. A single step to generate the next word has the following characteristics:

1.  **Compute Complexity**: $O(L \cdot (N \cdot d^2 + N^2 \cdot d))$ per step.
    *   **Linear Layers**: $O(N \cdot d^2)$ per layer. The system computes matrix multiplications (QKV projections, FFN mappings) for all $N$ words.
    *   **Attention**: $O(N^2 \cdot d)$ per layer. Each word's Query computes a dot product with the Keys of all preceding words, creating an $N \times N$ matrix.

2.  **Storage Complexity**: $O(L \cdot d^2) + O(N \cdot d + N^2)$ per step.
    *   **Static Weights**: $O(L \cdot d^2)$. Model parameters reside permanently in VRAM.
    *   **Dynamic Activations**: $O(N \cdot d + N^2)$. Temporary memory for the current step, released immediately after.

3.  **Parallelization**:
    *   **Prefill Phase**: Highly parallelizable because the prompt is complete. It fully utilizes GPU compute cores.
    *   **Decode Phase**: Strictly serial due to autoregression. The current token depends on the previous one. Naive inference recalculates historical words in parallel during each serial step, wasting massive compute power.

**Case Study: Llama 3 (405B) on NVIDIA H100**

Let's estimate the cost of generating one token in Naive mode:
*   **Model**: $d = 16384$, $L = 126$.
*   **Scenario**: Prompt length $N = 1000$.
*   **Computation**: Linear layers require approx. $2 \times L \times (11 \times N \times d^2) \approx 744$ TFLOPs. (Coefficient 11 accounts for attention and FFN layers; factor of 2 counts multiply-add as 2 operations).
*   **Latency**: An H100 GPU delivers ~1000 TFLOPS (FP16). At 100% utilization, computing this **single token** takes $744 / 1000 \approx 0.74$ seconds.

Generation slows down as context grows due to the $N^2$ Attention complexity. At $N = 1,000,000$, generating the next token would take approx. **2.5 hours**. This is unusable in production.

### Section 3: The Solution: Caching Past Computations

Naive inference cannot support long text or high concurrency. To fix this, engineers cache past computation results and only compute the new token. This technology is **KV Cache**.

---

## Chapter 6: KV Cache: Solving the Compute Bottleneck

KV Cache breaks the $O(N^2)$ loop by trading space for time.

### Section 1: The Principle: Caching K and V

Recall the Attention formula: $\text{Attention}(Q, K, V) = \text{softmax}(\frac{Q K^T}{\sqrt{d_k}})V$.

Key observations:
*   The **Query (Q)** of a new token represents current intent and must be computed fresh.
*   The **Key (K)** and **Value (V)** of past tokens **never change**.

Instead of recomputing, the system stores K and V in VRAM after the first step. For subsequent tokens, the GPU only computes Q, K, and V for the *single* new token, appends them to the cache, and performs attention with all cached K and V. This drops compute complexity from $O(N^2)$ to $O(N)$.

### Section 2: The Scope: Why Only K and V?

Why not cache Q? Because Q represents the "search intent" for the current step.
*   To predict token 4, we use token 4's Q to query tokens 1-3.
*   To predict token 5, we use token 5's Q to query tokens 1-4.
Token 4's Q becomes obsolete after step 4. We only cache K and V because they carry the persistent features of the tokens.

### Section 3: The Cost: The VRAM Tsunami

Let's compare Naive mode and KV Cache mode when generating the $N$-th token:

| Metric | Naive Mode | KV Cache Mode |
| :--- | :--- | :--- |
| **Compute Complexity** | $O(L \cdot (N \cdot d^2 + N^2 \cdot d))$ | $O(L \cdot (d^2 + N \cdot d))$ |
| **Storage Complexity** | $O(L \cdot d^2)$ (Weights only) | $O(L \cdot d^2 + L \cdot N \cdot d)$ (Weights + Cache) |

**Key Differences**:
1. **Compute Reduction**: KV Cache reduces linear layer computation from $O(N)$ to $O(1)$ per token, and attention from $O(N^2)$ to $O(N)$.
2. **VRAM Growth**: VRAM usage now grows linearly with sequence length $N$ ($O(L \cdot N \cdot d)$).
3. **Activations**: The table ignores dynamic activations because they are released immediately and do not accumulate.

Thus, while saving compute, KV Cache triggers a **VRAM Tsunami**. Deep layers and large dimensions require storing K and V for every token, on every layer, for every user.

**The Math of Llama 3 (405B)**:
*   **Layers**: $126$, **Dimension**: $16384$.
*   **Per-Token Size**: $16384 \times 2 \text{ bytes (FP16)} \times 2 \text{ (K and V)} = 64\text{ KB}$ per layer. Total = $64\text{ KB} \times 126 = 8064\text{ KB} \approx 8\text{ MB}$ per token.
*   **At 1M tokens**: $8\text{ MB} \times 1,000,000 \approx 8.26\text{ TB}$ for a **single request**.

This shifts LLM inference from being **compute-bound** to **memory-bound**. Technologies like GQA, PagedAttention, and RadixAttention attempt to solve this storage crisis.

---

## Chapter 7: Batching: Maximizing GPU Utilization

### Section 1: The Bottleneck: Memory Bandwidth

For LLM inference, memory bandwidth is often the primary bottleneck, not compute power.

Model weights and KV Cache reside in VRAM, while computations occur in Streaming Multiprocessors (SMs).
*   **Single-User Case**: To generate *one* token, the GPU moves hundreds of gigabytes of weights and the entire accumulated KV Cache from VRAM to SMs. After computing that token, it discards the data. The next step repeats this massive data transfer.
*   This massive data movement saturates memory bandwidth. Compute cores spend most of their time idling, waiting for data.

### Section 2: The Solution: Batched Matrix Multiplication (BMM)

Batching solves this by processing multiple user requests together. By stacking $N$ user inputs into a 3D tensor, the GPU loads the weight matrix once to compute for all $N$ users, multiplying throughput.

> [!NOTE]
> While users share model weights, their KV Caches are private. The GPU must load each user's KV Cache separately, so KV Cache data movement scales linearly with batch size.

### Section 3: The Flaw: Static Batching and Padding

Traditional **Static Batching** requires all requests in a batch to start and end simultaneously. Since request lengths vary, systems must pad shorter requests with invalid tokens. This wastes compute resources and forces short requests to wait for long ones (the straggler effect).

---

## Chapter 8: Core Asymmetry: Prefill vs. Decode

### Section 1: Prefill Phase: The Compute-Bound Phase

**1. Process and Complexity**
The model processes all $N$ input tokens simultaneously.
*   **Linear Layers**: GEMM (General Matrix-Matrix Multiplication). Complexity is $O(L \cdot N \cdot d^2)$.
*   **Attention**: Due to the Causal Mask, queries only attend to past and current tokens. This generates a lower-triangular $N \times N$ matrix. Complexity is $O(L \cdot N^2 \cdot d)$.
*   **Total Complexity**: $O(L \cdot (N \cdot d^2 + N^2 \cdot d))$. At large $N$, the quadratic attention complexity dominates.

**2. Why It Is Compute-Bound**
**Arithmetic Intensity** measures floating-point operations per byte read from VRAM.

$$\text{Arithmetic Intensity} = \frac{\text{Linear Layer FLOPs} + \text{Attention FLOPs}}{\text{Model Weight Size} + \text{KV Cache Write Volume}}$$

Estimation for Llama 3 405B at $N = 100,000$:
1. **Computation**:
   *   Linear layers: $2 \times N \times P \approx 8.1 \times 10^{16}$ FLOPs.
   *   Attention: $4 \times L \times N^2 \times d \approx 8.26 \times 10^{16}$ FLOPs.
   *   Total: $\approx 1.64 \times 10^{17}$ FLOPs.
2. **Memory Traffic**:
   *   Read weights: $810$ GB.
   *   Write KV Cache: $\approx 51.6$ GB.
   *   Total: $\approx 861.6$ GB.

**Result**: $1.64 \times 10^{17} / 861.6 \times 10^9 \approx 190,000$ FLOPs/Byte.

Since $190,000$ far exceeds the balance point of an H100 GPU ($\approx 300$ FLOPs/Byte, based on 1000 TFLOPS and 3.3 TB/s bandwidth), the system is **Compute-Bound**. The bottleneck is the GPU's peak TFLOPS, not memory bandwidth.

---

### Section 2: Decode Phase: The Memory-Bound Phase

After Prefill, the model enters the autoregressive Decode phase, generating tokens one by one.

**1. Process and Complexity**
Each step takes only the **1 token generated in the previous step** as input.
*   **Linear Layers**: GEMV (General Matrix-Vector Multiplication).
*   **Attention**: The Query of the new token computes dot products with cached Keys of all $N$ past tokens.
*   **Complexity**: $O(L \cdot (d^2 + N \cdot d))$. Attention computation grows with $N$.

**2. Why It Is Memory-Bandwidth-Bound**
To generate **one token**, the GPU must load the entire model weights and the accumulated KV Cache from VRAM to cores.
*   **Low Compute**: Matrix-vector multiplication requires minimal compute, leaving GPU cores idle.
*   **High Traffic**: Memory bandwidth is saturated.

Arithmetic Intensity estimation for Llama 3 405B at $N = 100,000$:
1. **Computation**:
   *   Linear layers: $2 \times 1 \times P = 8.1 \times 10^{11}$ FLOPs.
   *   Attention: $4 \times L \times 1 \times N \times d \approx 8.25 \times 10^{11}$ FLOPs.
   *   Total: $\approx 1.635 \times 10^{12}$ FLOPs.
2. **Memory Traffic**:
   *   Read weights: $810$ GB (loaded every step).
   *   Read KV Cache: $\approx 51.6$ GB.
   *   Total: $\approx 861.6$ GB.

**Result**: $1.635 \times 10^{12} / 861.6 \times 10^9 \approx 1.9$ FLOPs/Byte.

This falls far below the hardware balance point ($\approx 300$). The bottleneck is how fast VRAM feeds data to cores. Upgrading raw compute power yields little benefit without higher memory bandwidth.

---

### Section 3: The Asymmetry: Data Perspective Comparison

| Feature | Prefill Phase | Decode Phase (Single Step) |
| :--- | :--- | :--- |
| **Input Scale** | $N$ Tokens | $1$ Token |
| **Operator** | GEMM | GEMV |
| **Compute Complexity** | $O(L \cdot (N \cdot d^2 + N^2 \cdot d))$ | $O(L \cdot (d^2 + N \cdot d))$ |
| **VRAM Access** | Load Weights + Write KV Cache | Load Weights + Read KV Cache + Append |
| **Bottleneck** | **Compute-Bound** | **Memory-Bandwidth-Bound** |
| **GPU Utilization** | High | Low |

**Engineering Challenges**:
1.  **Throughput vs. Latency**: Large batches improve throughput by amortizing weight loads. But in Decode, larger batches load more private KV Caches, increasing user latency (TBT). In long-context scenarios, where KV Cache size rivals model weights, batching fails to improve throughput.
2.  **Scheduling Conflicts**: Mixing compute-heavy Prefills with memory-heavy Decodes blocks Decode steps. This causes lagging for active users (straggler effect), disrupting streaming output.

Part Three explores how systems like vLLM and SGLang solve these issues via **Continuous Batching** and **Chunked Prefill**.
