## Part 2: Bottlenecks — Why is LLM Inference So Difficult?

This part explains the physical and mathematical "walls" engineers hit when putting LLMs into production.

### Chapter 4: The "Rulers" for Measuring LLM Inference: Core Metrics Analysis

Before diving into various optimization techniques for LLM inference, we must first establish a set of measurement standards. The autoregressive nature of LLMs, which generates text "word by word," means we cannot simply evaluate them using the traditional "response time" metric of Web services. This chapter will introduce the core performance metrics in LLM inference.

*   **TTFT (Time to First Token)**: The time elapsed from when a user sends a request to when the model generates the **first token**. It corresponds to the **Prefill Phase** and determines whether the system feels "responsive."
*   **TBT (Time Between Tokens)**: The generation time between two consecutive tokens. It corresponds to the **Decode Phase** and determines the fluency of the streaming output.
*   **TPS (Tokens Per Second)**: How many tokens the model can generate per second, which is the reciprocal of TBT ($TPS = 1 / TBT$), used to intuitively measure generation speed.
*   **Latency**: The total time taken to complete the entire request. The formula is $Latency = TTFT + (\text{Number of Generated Tokens} - 1) \times TBT$.
*   **Throughput**: The total number of tokens or requests the system can process per second. This is the core metric for measuring server concurrency and Total Cost of Ownership (TCO).

---

### Chapter 5: Starting from Scratch: How Does the Most Naive LLM Inference Work?

Before discussing sophisticated optimization techniques, we must first look at how the "primitives" perform LLM inference. By understanding the most basic inference method, we can truly appreciate how terrifying the bottlenecks of LLM inference are.

#### Section 1: Unoptimized Inference Process

Suppose we want the model to generate subsequent text based on the prompt "Large models are". Without any optimization, the process of generating the 1st to the 3rd word is as follows:

1.  **Generate the 1st word**:
    *   **Input**: ["Large", "models", "are"]
    *   **Processing**: The entire sentence passes through the 80-layer Transformer building.
    *   **Output**: Predicts the next word is "the". Now we have ["Large", "models", "are", "the"].
2.  **Generate the 2nd word**:
    *   **Input**: ["Large", "models", "are", "the"]
    *   **Processing**: **Feed these 4 words back into the 1st floor**, and go through the 80 floors all over again!
    *   **Output**: Predicts the next word is "future". Now we get ["Large", "models", "are", "the", "future"].
3.  **Generate the 3rd word**:
    *   **Input**: ["Large", "models", "are", "the", "future"]
    *   **Processing**: **Once again, feed these 5 words back into the 1st floor**, and go through the 80 floors all over again!
    *   **Output**: Predicts the next word is ".".

#### Section 2: Bottleneck Breakdown: Explosion of Computational Complexity

This step-by-step inference method is academically known as **Naive Inference**. We use a specific step to break down its essence: suppose the current sentence length is $N$, the hidden layer dimension of the model is $d$ (representing the feature dimension of each word vector, indicating the granularity of the model's understanding of words), and the number of layers is $L$. In the Naive mode, for a **single step** of calculation to generate the next word, it behaves completely differently in computation, storage, and parallelization:

1.  **Compute Complexity**: The time complexity per step is $O(L \cdot (N \cdot d^2 + N^2 \cdot d))$
    *   **Linear layer computation**: $O(N \cdot d^2)$ (Note: must be calculated for each layer, total $L$ layers). We need to perform matrix multiplications like QKV projections and FFN mappings for all $N$ words.
    *   **Attention computation**: $O(N^2 \cdot d)$ (Note: must be calculated for each layer, total $L$ layers). For all $N$ words, the Query of each word must calculate the dot product with the Keys of all preceding words. Thus, it is an $N \times N$ matrix multiplication.

2.  **Storage Complexity**: The space complexity per step is $O(L \cdot d^2) + O(N \cdot d + N^2)$
    *   **Static weights**: $O(L \cdot d^2)$. The model's parameter matrices must reside permanently in VRAM, independent of the input length.
    *   **Dynamic activations**: $O(N \cdot d + N^2)$. Temporary memory allocated just to compute this step (released immediately after calculation). Since there is no need to explicitly store the intermediate calculation results of past words (i.e., no KV Cache is needed), this is an extreme practice of "trading time for space."

3.  **Parallelization**:
    *   **Prefill Phase**: Because the input Prompt is complete, the calculation of all words can be executed in a **highly parallelized** manner, fully squeezing the multi-core computing power of the GPU.
    *   **Decode Phase**: Due to the autoregressive nature, the current word strictly depends on the output of the previous word, so the steps are **absolutely serial** and cannot be parallelized across time steps. To make matters worse, Naive Inference must re-calculate historical words in parallel during each serial step, causing a huge waste of computing power.

**Real-world Case Deduction: Using Llama 3 (405B) as an Example**

To give everyone a concrete understanding of the "disastrous" consequences of the Naive mode, we use the current top-tier open-source large model **Llama 3 (405B)** to do an approximate calculation for a real scenario.

*   **Model parameters**: Hidden layer dimension $d = 16384$, number of layers $L = 126$.
*   **Scenario setting**: The current sentence length is $N = 1000$ (e.g., input a prompt of 1000 words), and we are preparing to generate the next word.
*   **Single-step computation**: According to the formula, the linear layer computation for this step is approximately $2 \times L \times (11 \times N \times d^2)$ (Note: the coefficient 11 comes from the conversion of the 4 linear layers of Attention and the 3 linear layers of SwiGLU FFN. Due to different model designs, this coefficient is usually between 10 and 12, here we use 11 as an approximation; multiplying by 2 is because each multiply-add operation is counted as 2 floating-point operations). Plugging in the numbers: $2 \times 126 \times 11 \times 1000 \times (16384^2) \approx 744$ TeraFLOPs (TFLOPs).
*   **Estimated time consumption**: Assuming we use the current mainstream top-tier AI graphics card **NVIDIA H100**, its theoretical peak computing power in half precision (FP16) is about 1,000 TeraFLOPs per second (1 PFLOPS). Under ideal conditions (100% computing power utilization), the pure computation time just to calculate this **1 single word** is as high as: $744 \div 1000 \approx 0.74$ seconds.

This means that in Naive mode, even with the most top-tier graphics card, the model can only spit out one or two words per second, and it gets even slower as the context grows. Because in long-context scenarios, the $N^2$ complexity of Attention will absolutely dominate. This means we absolutely cannot recalculate all preceding words every time a single word is generated. If we push the context length $N$ to the **1 million** tokens that models often support nowadays, the computation to generate the 1,000,001st word will completely explode, and **it will take about 2.5 hours just to output this single word**! This speed is completely unacceptable in a production environment.

#### Section 3: Introducing Optimization: Can We "Remember" Past Calculations?

This inefficient mode of operation is a death sentence for LLM inference—it simply cannot support long text generation, nor can it withstand high concurrency pressure.

System engineers started to think: since the preceding words have already been calculated, can we "cache" their calculation results and only calculate the newly added word each time?
This simple idea directly gave birth to the most important cornerstone technology in the history of LLM inference: **KV Cache**.

---

### Chapter 6: The Game Changer: KV Cache and the Resulting "VRAM Tsunami"

To break the $O(N^2)$ computational infinite loop, KV Cache was born. It trades space for time, completely changing the rules of the game for LLM inference.

#### Section 1: Trading Space for Time: Caching K and V

Back to the Attention formula we discussed in Chapter 1: $\text{Attention}(Q, K, V) = \text{softmax}(\frac{Q K^T}{\sqrt{d_k}})V$.

Engineers surprisingly discovered:
*   When generating a new word, the **Query (Q)** of the new word must be generated by the new word itself, because this represents the current intent.
*   However, the **Key (K)** and **Value (V)** of all preceding words in the context, once generated, **will never change again**!

Since K and V are fixed "assets", why don't we store them in VRAM after they are calculated in the first step?
This is the **KV Cache (Key-Value Cache)**. In subsequent inference, the GPU only needs to calculate Q, K, and V for that **single** new incoming token, append the newly calculated K and V to the cache, and then perform attention calculation with all K and V in the cache.
The computational complexity drops directly from $O(N^2)$ to $O(N)$!

#### Section 2: Why Only K and V?

This is a frequently asked Aha Moment: **Why don't we cache Q?**

Because Q represents the "intent to search," and it is a consumable.
*   When the model predicts the 4th word, we need to use the Q of the 4th word to look at the first 3 words.
*   When the model predicts the 5th word, we need to use the Q of the 5th word to look at the first 4 words.
The Q of the 4th word becomes obsolete right after the 4th step; it provides no help for the 5th step's prediction. Therefore, **Q does not need to be cached; we only need to cache K and V, which carry the features**.

#### Section 3: VRAM Tsunami: A TB-level Math Problem

To see more intuitively the changes brought by KV Cache, let's first compare the differences in computational and storage complexity between the **Unoptimized mode (Naive)** and the **KV Cache mode** when generating the N-th word (assuming the number of model layers is L and the dimension is d):

| Dimension | Unoptimized Mode (Naive) | KV Cache Mode |
| :--- | :--- | :--- |
| **Single-step Compute Complexity** | `O(N * d^2 + N^2 * d)` | `O(d^2 + N * d)` |
| **State Storage Complexity** | `O(L * d^2)` (Model Weights Only) | `O(L * d^2 + L * N * d)` (Weights + Cache) |

**Core Differences Breakdown**:
1. **Dimensionality Reduction Strike on Computation**: KV Cache reduces the linear layer computation per new word from `O(N)` to `O(1)` (only calculating the current single new word), and the attention computation from `O(N^2)` to `O(N)`.
2. **Trading VRAM Space for Time**: There is no free lunch. Although the computation plummets, the cost is that VRAM occupation changes from almost not growing with N, to linearly exploding with N (`O(L * N * d)`).
3. **About Dynamic Activations**: The storage complexity in the table ignores the **dynamic activations** generated during the calculation process. Because activations are ephemeral and will be released after a single forward pass ends, they will not continuously accumulate in VRAM as generation length N increases like KV Cache does, so they are usually ignored when analyzing long-term static storage bottlenecks.

This is why KV Cache, while saving computing power, triggered a terrifying **VRAM Tsunami**.

Because large models are not only deep in layers (e.g., 126 layers), but the K and V vector dimensions of each word are also very large. This means you must store a copy of K and V **for every token of every user, on every floor**.

Let's do the math: Suppose we use the previously mentioned **Llama 3 (405B)** model (126 layers, hidden layer dimension of 16384). Without any optimization (i.e., standard MHA mode), with a context window of 1 million tokens:
*   **Size of each token**: On each layer, the dimension of both K and V vectors is 16384, and each element takes 2 bytes (FP16), which is `16384 * 2 * 2 = 64 KB`. Passing through 126 layers, each token requires a total of `64 KB * 126 = 8064 KB` (approx. 8 MB) of cache.
*   **Total size**: `8064 KB * 1,000,000 \approx 8.26 TB`! The KV Cache for just this **single request** will eat up over **8 TB** of VRAM!

This directly pushes LLM inference from "compute-bound" (not enough computing power) to "memory-bound" (not enough VRAM capacity).

This "primitive" KV Cache mechanism, where VRAM linearly explodes with context, is clearly unsustainable. To slay this VRAM monster, the industry later invented magical techniques like **GQA (Grouped-Query Attention)**, **PagedAttention**, and **RadixAttention**, which we will uncover for you one by one in subsequent chapters.

---

### Chapter 7: Maximizing GPU Utilization: The Evolution of Batching

After solving the compute bottleneck for single-user inference (via KV Cache), the next nightmare engineers faced was: **How to simultaneously serve thousands of users?**

#### Section 1: Compute-Bound vs. Memory-Bound

You might think that what GPUs fear most is massive computation. But for LLM inference, the deadliest bottleneck is actually **memory bandwidth**.

The massive weight matrices of large models (hundreds of GBs) and the KV Cache that grows with the context are all stored in the GPU's Video RAM (VRAM), while computations happen in the Streaming Multiprocessors (SMs).
*   **If handling only one user at a time**: For every single token generated, the GPU not only has to move the hundreds of GBs of weight parameters for the entire building from VRAM into the SM cores, but also move the historically accumulated KV Cache along with them! After calculating this single token, this data is released in the cores. In the next iteration, everything must be moved all over again!
*   In this situation, the volume of data movement is enormous, severely bottlenecking memory bandwidth. The GPU's compute cores spend the vast majority of their time **idling and waiting for data**. Computing power is massively wasted.

#### Section 2: Batched Matrix Multiplication (BMM)

To stop the GPU's computing power from sitting idle, the most direct solution is **Batching**.

Since moving the model weights once is so taxing, let's bring in multiple users' requests at the same time!
Through **Batched Matrix Multiplication (BMM)**, we stack the input vectors of $N$ users into a 3D tensor. The GPU only needs to move the weight matrix once to compute for these $N$ users simultaneously. Throughput is multiplied.

> [!NOTE]
> Although N users share the same set of model weights (only needing to be moved once), each user's KV Cache is completely independent private data. When computing Attention, the GPU must separately move the KV Caches of these N users from VRAM into the compute cores, which causes the KV Cache data movement volume to scale linearly with the Batch Size.

#### Section 3: The Padding Problem: Flaws of "Static Batching"

However, traditional **Static Batching** requires that all requests in a batch must start and end at the same time. Because the input and output lengths of users vary, to align the batch, the system must pad short requests with a large number of invalid tokens (Padding). This not only wastes precious computing resources but also forces short requests to wait for long requests, triggering the wooden barrel effect (weakest link problem).

### Chapter 8: Core Asymmetry: Prefill vs. Decode

#### Section 1: Prefill Phase — The "Blitzkrieg" that Devours Compute (Compute-Bound)

**1. Physical Process and Compute Complexity**

In this phase, the model takes in the user's input of **$N$ tokens** all at once.
*   **Linear layer computation**: The vectors of these $N$ tokens are multiplied by the model weight matrices. Mathematically, this is a standard **General Matrix-Matrix Multiplication (GEMM)**. The computation volume is proportional to the number of tokens $N$, with a complexity of $O(L \cdot N \cdot d^2)$.
*   **Attention computation**: Due to the Causal Mask restriction of the Decoder-Only architecture, **each word can only calculate attention with the words before it**, generating a lower-triangular $N \times N$ attention score matrix. The computation volume is proportional to the square of the number of tokens, with a complexity of $O(L \cdot N^2 \cdot d)$.
*   **Single-step computation complexity**: Adding the two together, the total complexity is $O(L \cdot (N \cdot d^2 + N^2 \cdot d))$. Here, $L$ is the number of model layers, and $d$ is the hidden layer dimension. It can be seen that when the input length $N$ is extremely large, the quadratic complexity of attention computation will rise rapidly.

**2. Why is it "Compute-Bound"?**
This involves a core concept: **Arithmetic Intensity**, which means "how many floating-point operations the GPU can perform for every byte of data read from VRAM".

We can calculate the complete arithmetic intensity for the Prefill phase (including linear layers, Attention, and VRAM writes for KV Cache):

$$\text{Total Arithmetic Intensity} = \frac{\text{Linear Layer FLOPs} + \text{Attention FLOPs}}{\text{Model Weight Size} + \text{KV Cache Write Volume}}$$

Let's estimate using Llama 3 405B with **$N = 100,000$** (100k context) as an example:
1. **Total Computation**:
   *   **Linear layers**: $2 \times N \times P = 2 \times 10^5 \times 405 \times 10^9 = 8.1 \times 10^{16}$ FLOPs.
   *   **Attention**: $4 \times L \times N^2 \times d = 4 \times 126 \times (10^5)^2 \times 16384 \approx 8.26 \times 10^{16}$ FLOPs.
   *   **Total**: Approx. **$1.64 \times 10^{17}$ FLOPs** (At this point, Attention FLOPs are roughly equal to linear layer FLOPs).
2. **Total Memory Traffic**:
   *   **Read weights**: **$810$ GB** (405B model weights in FP16 format).
   *   **Write KV Cache**: Approx. **$51.6$ GB**.
   *   **Total**: Approx. **$861.6$ GB**.

The final **Total Arithmetic Intensity** is:
$$\frac{1.64 \times 10^{17} \text{ FLOPs}}{861.6 \times 10^9 \text{ Bytes}} \approx \mathbf{190,000} \text{ FLOPs/Byte}$$

This means that for every byte read from VRAM, the GPU needs to perform about 190,000 floating-point operations. Meanwhile, the "compute-to-bandwidth" balance point (inflection point) for modern top-tier GPUs (like H100) is usually around a few hundred FLOPs/Byte (for example, H100 SXM in FP16 has about 1000 TFLOPS compute and 3.3 TB/s bandwidth, meaning the inflection point is roughly $300$ FLOPs/Byte).

Since $190,000$ far exceeds the inflection point of $300$, the GPU is inevitably in a **Compute-Bound** state. The loaded weights are fully shared and reused by these 100k tokens, and the thousands of Arithmetic Logic Units (ALUs) on the GPU are completely filled and running at high speed. At this time, the bottleneck limiting inference speed is the GPU's **theoretical peak computing power (TFLOPS)**, not memory bandwidth.

---

#### Section 2: Decode Phase — The "War of Attrition" Crushing Bandwidth (Memory-Bandwidth-Bound)

The moment the Prefill phase spits out the first word, the rules of the game change instantly. The model enters the autoregressive Decode phase, generating text word by word.

**1. Physical Process and Compute Complexity**
At every step in this phase, the model only takes the **1 new token generated in the previous step** as input.
*   **Linear layer computation**: The vector of this 1 token is multiplied by the model weight matrices. Mathematically, this degrades to a standard **General Matrix-Vector Multiplication (GEMV)**.
*   **Attention computation**: The Query vector of this new token must take a dot product with the Key vectors of all $N$ past words cached in the KV Cache.
*   **Single-step computation complexity**: $O(L \cdot (d^2 + N \cdot d))$. As the context length $N$ grows, the proportion of attention computation gradually increases.

**2. Why is it "Memory-Bandwidth-Bound"?**
This is the most painful "memory wall" problem in LLM inference.
To generate this **mere 1 word**, the GPU has to do something utterly absurd: **It must transport the entire hundreds of GBs of model weights residing in VRAM, along with the continuously growing KV Cache from the dialogue, all the way from VRAM into the SRAM compute cores once more!**
*   **Minuscule Computation**: Because the input is only 1 token, the computation volume for matrix-vector multiplication is extremely thin, and the vast majority of the GPU's cores are sitting idle.
*   **Massive Transport**: Memory bandwidth (e.g., H100's HBM3 bandwidth of ~3.3 TB/s) is completely maxed out.
We can calculate the single-step arithmetic intensity formula for **generating a single token** in the Decode phase:

$$\text{Single-Step Arithmetic Intensity} = \frac{\text{Linear Layer FLOPs} + \text{Attention FLOPs}}{\text{Model Weight Size} + \text{KV Cache Read Volume}}$$

We will again estimate using Llama 3 405B with **$N = 100,000$** (having accumulated 100k words of context):
1. **Single-Step Computation**:
   *   **Linear layers**: $2 \times 1 \times P = 8.1 \times 10^{11}$ FLOPs.
   *   **Attention**: $4 \times L \times 1 \times N \times d = 4 \times 126 \times 1 \times 10^5 \times 16384 \approx 8.25 \times 10^{11}$ FLOPs.
   *   **Total**: Approx. **$1.635 \times 10^{12}$ FLOPs**.
2. **Single-Step Memory Traffic**:
   *   **Read weights**: **$810$ GB** (The weights must be completely read once for *every single round* of generation!).
   *   **Read KV Cache**: Because we need to calculate attention against the past 100k words, we must read the KV Cache of these 100k words from VRAM, which is about **$51.6$ GB**.
   *   **Total**: Approx. **$861.6$ GB**.

The final **Single-Step Arithmetic Intensity** is:
$$\frac{1.635 \times 10^{12} \text{ FLOPs}}{861.6 \times 10^9 \text{ Bytes}} \approx \mathbf{1.9} \text{ FLOPs/Byte}$$

At this time, the arithmetic intensity (1.9) is extremely low, falling far below the hardware inflection point (~300). The bottleneck limiting inference speed is no longer how fast the GPU can compute, but **how fast the VRAM can feed data into the cores**. This is why, even if you switch to a graphics card with stronger computing power, the speed of the Decode phase (words spat out per second) might only see limited improvement, unless the new card possesses much higher memory bandwidth.

---

#### Section 3: The "Asymmetry" from a Data Perspective

To give you a more intuitive, quantified understanding of this asymmetry, let's look at a comparison (assuming context length $N$, hidden layer dimension $d$, and layer count $L$):

| Feature | Prefill Phase | Decode Phase (Single Step) |
| :--- | :--- | :--- |
| **Input Scale** | $N$ Tokens (Large) | $1$ Token (Extremely Small) |
| **Math Operator** | Matrix-Matrix Multiplication (GEMM) | Matrix-Vector Multiplication (GEMV) |
| **Single-Step Compute Complexity** | $O(L \cdot (N \cdot d^2 + N^2 \cdot d))$ | $O(L \cdot (d^2 + N \cdot d))$ |
| **VRAM Access** | Load Weights + Write KV Cache | Load Weights + Read KV Cache + Append KV Cache |
| **Hardware Bottleneck** | **Compute-Bound** | **Memory-Bandwidth-Bound** |
| **GPU Utilization** | Extremely High (Ideal for maximizing compute) | Extremely Low (Compute is severely underutilized) |

**The Ultimate Engineering Contradiction**
This asymmetry directly leads to the following engineering challenges:
1.  **The Conflict Between Throughput and Latency**: To increase throughput, we want to increase the Batch Size. For Prefill, this makes the GPU more saturated and efficient; but for Decode, a larger Batch Size means transporting more users' KV Caches, which adds to the already congested memory bandwidth and inflates latency for every user.
2.  **A Scheduler's Nightmare**: When a new user's Prefill request (compute-heavy) arrives, forcefully inserting it into a batch queue currently performing Decode (memory-heavy) will cause severe lagging for the users currently in Decode (the Straggler Effect).

This is exactly what we will see in Part Three of this book: the ultimate contradiction that vLLM's **Continuous Batching** and **Chunked Prefill** attempt to resolve.

In Part Two, we have confronted the "two huge mountains" of LLM inference directly: the **VRAM Tsunami** triggered by KV Cache, and the **Core Asymmetry** caused by the distinct mechanisms of Prefill and Decode. These laws of physics act like a tight headband, strictly constraining the concurrency capabilities and response speed of large models.

Now that the challenges are clear, how do engineers break the deadlock? In the upcoming **Part Three**, we will dive deep into the path to salvation offered by modern high-performance inference engines (such as vLLM and SGLang), examining how they shatter the memory wall and conquer this asymmetry through ingenious algorithmic and architectural designs.
