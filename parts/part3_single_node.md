## Part Three: Single Node — High-Performance Engines Squeezing Every Inch of VRAM

In the second part, we dissected the physical and mathematical bottlenecks of LLM inference: **the VRAM tsunami triggered by KV Cache**, and **the core asymmetry between Prefill and Decode**. These bottlenecks directly paralyze the concurrency capability and response speed of large models in production environments.

To break the deadlock, system engineers and algorithmic scientists have launched a saturated, hardware-software co-designed rescue mission. This part will delve into how modern inference engines (such as vLLM and SGLang) and model architectures themselves solve the aforementioned bottlenecks. We will unfold from two core battlefields:
1. **Smashing the VRAM Wall**: Through GQA (model architecture), PagedAttention (paged management), and RadixAttention (prefix caching), VRAM utilization is increased several times over.
2. **Conquering the Asymmetry**: Through Continuous Batching and Chunked Prefill, padding waste is eliminated, keeping GPU compute and bandwidth saturated at all times.

> [!IMPORTANT]
> It is worth noting that while Continuous Batching and Chunked Prefill greatly improve single-node efficiency, they are "tactical" local optimizations and do not fundamentally resolve the hardware mismatch and resource contention between Prefill and Decode. We will reveal a more thorough "strategic" evolution—**Disaggregated Serving**—in the cluster section of Part Four.

Now, let's first cut into the first battlefield—slimming down VRAM from the model architecture level.



### Chapter 9: VRAM Slimming at the Model Architecture Level: GQA

#### Section 1: The Evolution from MHA to GQA

In the early designs of Transformer, **MHA (Multi-Head Attention)** was adopted.
*   **MHA**: If there are $H$ Query heads, there must be $H$ corresponding Key heads and $H$ Value heads. As we discussed in Section 3 of Chapter 6, the space complexity of KV Cache is $O(L \cdot N \cdot d)$. In MHA, although the hidden dimension $d$ is partitioned into $H$ heads (i.e., $d = H \times d_k$), because the number of KV heads equals the number of Query heads, the total KV Cache size is still determined by the full dimension $d$. For models with hundreds of billions of parameters, to ensure expressive power, $d$ is usually set very large (e.g., Llama 3 405B's $d = 16384$), which directly leads to an extremely massive KV Cache.

To shrink the KV Cache, researchers proposed **MQA (Multi-Query Attention)**:
*   **MQA**: All Query heads **share the same single group** of Key and Value heads. This directly shrinks the KV Cache size to $1/H$ of its original size! The VRAM pressure drops dramatically, but at the cost of a certain decrease in the model's expressive power.

**GQA (Grouped Query Attention)** is a perfect compromise between the two, and is currently widely used in various open-source and closed-source large models (for example, Llama 3 405B also adopts this scheme):
*   **GQA**: The Query heads are grouped (e.g., 8 heads per group), and each group shares one set of Key and Value heads.
*   **Benefits**: It not only significantly cuts down the VRAM footprint of KV Cache, **increasing the system's Throughput (capable of accommodating more concurrency)**, but also indirectly **boosts the TPS of a single request** by reducing the data movement volume during the Decode phase, with almost no loss in model performance.

**Practical Comparison: KV Cache Calculation Based on Llama 3 405B**

To give you a more intuitive understanding of the VRAM slimming effects of these three mechanisms, let's again take the parameter specifications of **Llama 3 405B** as an example (number of layers $L=126$, hidden dimension $d=16384$, number of Query heads $H=128$, dimension of each head $d_k=128$, FP16 format), and calculate the KV Cache size under a **1-million Token** context for different mechanisms:

*   **Standard MHA Mode** (assuming each Query head has independent KV heads):
    *   Size per token per layer: $2 \times 128 \times 128 \times 2 \text{ bytes} = 64 \text{ KB}$
    *   Total size: $64 \text{ KB} \times 126 \text{ layers} \times 1,000,000 \approx \mathbf{8.06 \text{ TB}}$
*   **MQA Mode** (all Query heads share 1 group of KV heads):
    *   Size per token per layer: $2 \times 1 \times 128 \times 2 \text{ bytes} = 0.5 \text{ KB}$
    *   Total size: $0.5 \text{ KB} \times 126 \text{ layers} \times 1,000,000 \approx \mathbf{63 \text{ GB}}$
*   **GQA Mode** (actually adopted by Llama 3, 8 groups of KV heads):
    *   Size per token per layer: $2 \times 8 \times 128 \times 2 \text{ bytes} = 4 \text{ KB}$
    *   Total size: $4 \text{ KB} \times 126 \text{ layers} \times 1,000,000 \approx \mathbf{504 \text{ GB}}$

Dropping from $8 \text{ TB}$ to $504 \text{ GB}$, GQA achieves a 16-fold VRAM compression, directly making it possible to process long contexts on a single card or a small-scale cluster!

#### Section 2: Deep Thinking — Why Does Trimming Only $K$ and $V$ Work?

This is a very profound question: **Why can we just trim $K$ and $V$, leave $Q$ intact, and the model's expressive power is still not significantly affected?**

To understand the meaning behind this, we need to analyze it from three levels: **the role division of $Q$, $K$, $V$**, **information redundancy**, and **the knowledge storage of large models**.

**1. Role Asymmetry: The Questioner ($Q$) vs. The Queried ($K$, $V$)**
In the Attention mechanism, $Q$, $K$, and $V$ play completely different roles:
*   **Query ($Q$)** represents the **"intent" or "question"** (What am I looking for now?). It is dynamic; as the model generates each new word, the intent changes.
*   **Key ($K$)** represents the **"index" or "label"** (What do I have here?).
*   **Value ($V$)** represents the **"content" or "substance"** (What is the actual information I hold?).

**The physical meaning behind this is: The same objective fact ($K$ and $V$) can answer many different questions ($Q$).**

> [!NOTE]
> **Analogy: Researchers in a Library**
> Imagine a scenario: There are **128 researchers** in a library (representing 128 $Q$ heads), and each of them is researching a different topic and asking different questions.
> *   **In ordinary MHA**: The system is extremely extravagant. To serve these 128 researchers, the library not only provides 128 researchers but also photocopies 128 identical sets of encyclopedias for everyone (128 $K$ and $V$ heads). Each researcher only looks at the set on their own desk. This obviously causes a massive waste of space.
> *   **In GQA**: The system is optimized. There are still **128 researchers** in the library, but now only **8 sets of encyclopedias** (8 $K$ and $V$ heads) are purchased. Every 16 researchers share one set of books.
> 
> Although the 16 researchers ask all sorts of weird questions (different $Q$s), the **historical background and objective facts ($K$ and $V$) they want to look up are exactly the same**. One set of books is enough to answer all their questions. This is why the $Q$ heads cannot be reduced (to ensure diversity of thought), while the $K$ and $V$ heads can be reduced (shared knowledge base).

**2. Severe "Information Redundancy" in MHA**
In standard MHA, researchers discovered through visual analysis that **many different Key heads and Value heads learn highly repetitive features.** For instance, there might be 5 heads all focusing on "who is the subject of the sentence," and another 4 heads focusing on "who the pronoun refers to." The essence of GQA is **forced deduplication**: Since several of you heads are paying attention to similar information, why don't you just share the same set of Key and Value! This sharing forces the model to utilize parameters much more efficiently during training, compressing redundant information, thereby slashing the VRAM footprint without compromising expressive power.

**3. The "Hard Knowledge" of Large Models Is Not in KV at All**
This is the most fundamental source of confidence: **The vast majority of a large model's knowledge is actually stored in the FFN (Feed-Forward Network), not in the Attention layer.**
We calculated the bill in previous chapters: in each layer of the Transformer, FFN parameters account for about 82%, while Attention only accounts for less than 20%.
*   **FFN** acts like a massive "knowledge hard drive," storing an immense amount of objective laws and facts.
*   **Attention** acts more like a "scheduler" and "router," responsible for moving information and establishing connections across the context.
Since Attention is only responsible for **moving and correlating** context, compressing its KV cache won't destroy the hundreds of billions of knowledge reserves stored inside the FFN. The model still knows "the capital of France is Paris"; it's just using fewer "memory pointers" (KV) to point to this knowledge during inference.

#### Section 3: A Hundred Flowers Blooming: Other Frontier Progress in Compressing KV

Besides GQA, which is widely adopted in mainstream models, academia and industry have recently seen many exciting new developments to further combat the "VRAM tsunami." While we won't dig into the details here, understanding these directions is crucial for grasping the technological trends of large models:

1.  **MLA (Multi-head Latent Attention)**:
    *   **Principle**: Proposed by the DeepSeek team in DeepSeek-V2. It uses Low-Rank Joint Compression technology to project Key and Value into a low-dimensional Latent Space. During inference, only this extremely small latent vector needs to be cached, achieving a KV Cache compression ratio even more exaggerated than GQA (up to several times higher).
    *   **Reference**: [DeepSeek-V2 Paper](https://arxiv.org/abs/2405.04434)
2.  **Sliding Window Attention**:
    *   **Principle**: When computing attention, the model no longer looks at all historical tokens "from beginning to end", but instead maintains a fixed-size sliding window (e.g., only looking at the most recent 4096 tokens). KV Cache beyond the window is directly discarded, reducing the space complexity of KV Cache from $O(N)$ to $O(1)$.
    *   **Reference**: [Mistral 7B Paper](https://arxiv.org/abs/2310.06825)
3.  **Interleaved Local/Global Attention**:
    *   **Principle**: Combines the advantages of sliding window and global attention. Sliding window attention is used in some layers of the model to save VRAM, while global attention is retained in other layers to capture long-distance dependencies (e.g., some models of Gemma 2 and Mistral adopt similar strategies).
    *   **Reference**: Refer to the official technical reports of the respective models.
4.  **Infini-Attention (Compressive Memory Attention)**:
    *   **Principle**: Proposed by Google, it introduces a "compressive memory" mechanism into standard dot-product attention to store old KV states in a fixed-size memory. It combines masked local attention and long-term linear attention, theoretically allowing the model to process infinitely long contexts without causing a KV Cache explosion.
    *   **Reference**: [Leave No Context Behind Paper](https://arxiv.org/abs/2404.07143)

These advancements reveal a clear trend: **Further reducing the size of K and V through model architecture optimization remains one of the core battlefields for large model optimization today.**

GQA and the architectural optimizations mentioned above are improvements to the **model itself**, requiring the model to adopt this architecture during training. However, besides slashing KV heads from the architectural level, we can also compress the KV Cache from the **precision level** without altering the architecture.

### Chapter 10: Precision Reduction: KV Cache Quantization (FP8/INT8)

If GQA pushes the **spatial structure** to the extreme (reducing data volume), then KV Cache quantization brings the hammer down on **data density**.

#### Section 1: A Cost-Effective Trade of Compute for Bandwidth

Readers might ask: **Isn't this just using extra computation to compress KV Cache storage?**

The answer is: **Absolutely correct! But this is definitely a highly profitable trade-off during the Decode phase.**

In standard inference, both K and V are stored at 16-bit precision (FP16 or BF16), with each element occupying 2 bytes. The core idea of KV Cache quantization is to compress them to 8 bits (FP8 or INT8), where each element only occupies 1 byte.

This does introduce additional computational overhead:
1. **On Write**: When a new Token is generated during the Decode phase, its K and V vectors must first undergo Scaling and Truncation to convert from 16 bits to 8 bits before they can be stored in VRAM.
2. **On Read**: When calculating Attention, the GPU reads these 8-bit K and V from VRAM, and must first "dequantize" them back to 16 bits (or calculate directly on hardware that supports lower precision).

#### Section 2: Why Is This Cost-Effective?

As we discussed in depth in Chapter 8, the Decode phase is a classic **Memory-Bound** operation. The GPU's compute cores are idle most of the time, bitterly waiting for data to be moved over from VRAM.

* **Without Quantization**: The data volume is large, movement is slow, and GPU cores are starved.
* **With Quantization**: Despite the few extra steps of computation for quantization conversion, **the volume of data that needs to be moved is cut right in half**! The pressure on VRAM bandwidth is halved, and the data feeds the GPU cores much faster.

On modern GPUs like the NVIDIA H100, hardware natively supports FP8 tensor calculations, making the compute overhead of this quantization conversion almost negligible. Therefore, trading insignificant computational cost for halved VRAM usage and doubled transfer speeds has become standard in modern high-performance inference engines.

#### Section 3: INT8 vs. FP8: Strikingly Different Paradigms

Two distinct routes exist for 8-bit KV Cache quantization:

*   **INT8 (8-bit Integer)**: This maps floats to integers. It scales and rounds values into 256 discrete grids. Compute requires dequantizing back to FP16 first. Though mathematically lossy, fine-grained quantization (per-token or per-channel) minimizes accuracy drops.
*   **FP8 (8-bit Floating Point)**: This remains a floating-point number with sign, exponent, and mantissa. It just reduces precision and range compared to FP16. On NVIDIA H100, FP8 computes extremely fast without explicit dequantization, dominating server deployments.

#### Section 4: Dynamic vs. Static: Contrasting with Model Weight Quantization

Why not quantize the model weights too?

We do. Weight quantization (e.g., GPTQ, AWQ) is even more common. But they differ in difficulty and impact:

*   **Weight Quantization (Static)**: Weights are fixed. We can analyze them offline and calibrate with data. Even at 4-bit, large models stay smart.
*   **KV Cache Quantization (Dynamic)**: Activations change with every input. Outliers appear dynamically. Compressing to 4-bit destroys the range and collapses output quality. Thus, KV Cache usually stays at 8-bit.

They are often combined:
*   **W4A16 (Weights Only)**: Focuses on capacity, fitting big models into smaller GPUs.
*   **W8A8 / FP8 (Full Quantization)**: Focuses on speed, serving as the top choice for high concurrency.

---

Although the size of the KV Cache is substantially compressed through GQA and quantization techniques, as context grows, it still causes severe fragmentation issues in VRAM. This is why we still need **PagedAttention**.

### Chapter 11: VRAM Management at the Engine Level: PagedAttention

#### Section 1: The Fragmentation Crisis: Wasted by "Booking the Whole Venue"

When large models generate text, since it's impossible to predict the final sequence length generated by the user (it could be just a few tokens, or it could hit the model's maximum context length), traditional VRAM management methods adopted a **static contiguous allocation** strategy. The system had to pre-allocate a sufficiently large contiguous VRAM space for each request based on the model's maximum context length (e.g., 8000 tokens).

This led to severe memory waste:
*   **Internal Fragmentation and Reservation Waste**: The system must pre-allocate an entire contiguous block of VRAM for each request based on the **maximum context length**. During request processing, this VRAM is exclusively occupied. This means that, regardless of whether the final request is long or short, **the unused space (reserved for future tokens) and the space that will never be used due to early termination of the request**, cannot be reused by other requests until the current request is finally released. This "booking the whole venue" mechanism causes severe VRAM idleness.
*   **External Fragmentation**: Even if there is enough total free space left in VRAM, if these spaces are not physically contiguous, the system cannot allocate them to new requests that require a large contiguous block.

According to statistics from the vLLM paper, under the traditional static contiguous allocation method, due to fragmentation issues, the VRAM actually used to store valid KV Cache was often less than 20%, with as much as 80% of memory being wasted for nothing.

---

#### Section 2: Inspiration from OS: Virtual Memory Paging

Faced with this astonishing VRAM black hole, researchers at UC Berkeley (the authors of vLLM) acutely realized: Isn't this exactly the same problem encountered decades ago when early computers ran out of memory?

In computer operating systems, the **virtual memory paging mechanism (Paging)** was invented long ago to solve memory fragmentation. The operating system partitions physical memory into fixed-size "Pages." Programs see logically contiguous memory, but physically it can be scattered anywhere in memory.

The core idea of vLLM is to **transplant the operating system's virtual memory paging mechanism to GPU VRAM management**:
1.  **Block Management**: Instead of allocating massive contiguous VRAM for a single request, it divides the physical VRAM into fixed-size Physical Blocks. For example, each block fixedly stores the K and V matrices for 16 tokens.
2.  **Non-contiguous Physical Allocation**: Logically, a request's Token sequence is contiguous; but in physical VRAM, the blocks corresponding to these tokens can be discretely distributed anywhere in VRAM, without needing physical contiguity.

---

#### Section 3: Block Tables: Near-Zero Memory Waste

Since physical locations are scattered, how can the model find all the preceding words when calculating attention?

To efficiently compute attention in a non-contiguous physical space, PagedAttention introduces **Block Tables**. The Block Table is responsible for maintaining the mapping between logical blocks and physical blocks, similar to a Page Table in an operating system. When calculating $Q \cdot K^T$, the attention mechanism queries the block table to dynamically read the corresponding K and V vectors from discrete physical blocks, completing the calculation.

**Analysis (Intuitive Understanding of the Mapping Mechanism)**:

To help you thoroughly understand how the `BlockTable` maintains the `Request -> Token Block -> Memory Block` mapping, let's look at its tensor dimension design:

`block_table` is fundamentally a 2D tensor with the shape `[max_num_reqs, max_num_blocks_per_req]`.
*   **First Dimension (Row)**: Corresponds to different **Requests**, indexed by `req_idx`.
*   **Second Dimension (Column)**: Corresponds to a specific **Token Block** (logical block) of a request, indexed by `logical_block_idx`.
*   **Stored Value**: This is the corresponding **Memory Block** (physical VRAM block ID).

In other words, the lookup formula is: `physical_block_id = block_table[req_idx][logical_block_idx]`.

**The Magical Benefits Brought by PagedAttention**:

**VRAM Waste Drops to Near Zero**: Because allocation is strictly on-demand (only opening the next one when a 16-seat block is full), VRAM waste is strictly confined to the last unfilled Block (at most wasting positions for 15 tokens). The overall VRAM fragmentation waste rate plummets from 80% to under 4%.

However, PagedAttention mainly solves the issues of **VRAM fragmentation** and **waste within a single request**. By default, **the KV Cache across different Requests remains strictly separate**. If two independent requests arrive at different times, even if they share the exact same prefix (e.g., the same background document), PagedAttention cannot automatically identify and reuse previously computed physical blocks.

This higher-level, cross-request dynamic "cache reuse" is exactly the core of what we will discuss in the next chapter.

---

### Chapter 12: Memory Time Machine: Prefix Caching (RadixAttention)

In Chapter 8, we mentioned the scenario where multiple users share system prompts. However, in practical applications, especially in multi-turn dialogues and RAG (Retrieval-Augmented Generation) scenarios, the situation is much more complex. This chapter will introduce how to use the **Radix Tree** data structure to achieve more advanced cache reuse — **Prefix Caching**.

#### Section 1: The Dilemma of RAG and Multi-Turn Dialogues

In the real-world application of large models, the Prompts we input often contain massive background materials.

**Scenario 1: Multi-Turn Dialogue**
*   Turn 1: You ask "What is an apple?" (The model computes and generates an answer).
*   Turn 2: You follow up with "Is it tasty?". At this point, the actual Prompt fed to the model is: [Your Turn 1 question + AI's Turn 1 answer + Your Turn 2 question].

This involves a very basic but easily overlooked engineering detail: **The HTTP protocol is stateless.**
This means that from the perspective of the large model serving engine (like vLLM), the second-turn dialogue is a **completely independent, brand-new Request**, and will be assigned a **brand-new Request ID**.

In traditional PagedAttention, the Block Table in VRAM is strongly bound to the Request ID. Even though the Turn 2 request's prompt contains Turn 1's content, the engine only recognizes the Request ID. Because the IDs are different, the engine cannot automatically recognize and reuse the already-computed KV Cache from Turn 1.

Therefore, in the era without prefix caching, the large model could only act like an extremely rigid repeater, recalculating the QKV for everything already computed in Turn 1! As the number of dialogue turns increases and the prompt gets longer, the Time To First Token (TTFT) skyrockets linearly. This cross-request VRAM isolation directly created the absolute necessity for introducing **Prefix Caching**.

**Scenario 2: RAG (Knowledge Base Q&A)**
You upload a 100,000-word manual, and then continuously ask questions against it. Every time you ask a question, these 100,000 words serve as the upfront background. Without optimization, every question requires reprocessing these 100,000 words, and the **Time To First Token (TTFT)** will be unbearably high.

**Scenario 3: Fixed System Prompts and Few-Shot Examples**
In enterprise applications or Agents, a lengthy and fixed System Prompt or Few-Shot examples are usually stuffed in before every request. Without optimization, even if 1000 users access the system concurrently, the system would repeatedly calculate the exact same KV Cache 1000 times for these 1000 independent requests, causing extreme waste of compute and VRAM.

**Scenario 4: Parallel Sampling and Beam Search**
In code generation (where the model is asked to output multiple candidate solutions) or when using Beam Search, the system needs to generate multiple different continuations for the same Prompt. Without optimization, the system needs to copy and repeatedly compute the Prompt's KV Cache for each branch. But in a Radix Tree, the shared Prompt naturally becomes a parent node, and each generation branch only needs to bifurcate from that node, avoiding redundant calculations.

---

#### Section 2: Radix Trees: Sharing Physical Memory

To solve this problem, frameworks like SGLang (and later vLLM) introduced **RadixAttention**, which leverages the **Radix Tree** data structure to manage KV Cache.

In a Transformer, due to the nature of positional encoding and self-attention, as long as the preceding token sequence is exactly identical, the computed KV Cache will be absolutely identical.

The Radix Tree operates as follows:
*   **Root Node**: Empty sequence.
*   **Edges**: Represent a continuous sequence of tokens (e.g., a Block of 16 tokens).
*   **Nodes**: Point to the physical blocks corresponding to these tokens.

When a new request comes in, the system starts from the root node and performs a **longest prefix match** of the request's token sequence against the tree's edges.
If a match is successful, it directly reuses the physical blocks pointed to by that node, skipping the matrix multiplication for those tokens entirely! For the unmatched portion that follows (e.g., the user's new question), new physical blocks are allocated, and new leaf nodes sprout on the tree.

Through this method, the historical records of multi-turn dialogues, public documents of RAG, can all be reused instantly like a "time machine," and the speed at which the first token pops out plummets from seconds to tens of milliseconds.

> [!NOTE]
> **Deep Dive Details: How Tokens Are Indexed and Engine Differences**
> 1. **How are tokens indexed?**: The indexing on the Radix Tree is **by no means a text comparison**. Large models have already converted text into **Token IDs** (integers) long before processing. The edges of the tree store these integer sequences. During matching, the system performs highly efficient **integer sequence comparisons**, or calculates a **Hash** on the token sequence (e.g., vLLM relies on hashes to quickly anchor Blocks).
> 2. **Implementation differences between vLLM and SGLang**: Although both utilize radix trees for prefix caching, their granularity differs. **SGLang's** RadixAttention is at the **Token level**, making matching highly flexible (an edge can represent any length of token sequence); whereas **vLLM's** APC (Automatic Prefix Caching) inherits the DNA of PagedAttention and operates at the **Block level** (typically managing and hashing in fixed chunks of 16 tokens).
> 3. **The Relationship between Radix Tree and Block Table**: Your understanding is very accurate. The radix tree does not replace the Block Table; they ultimately both point to the same Physical Blocks, just from different index dimensions:
>    *   **Block Table** is an index based on **`Request -> Logical Block -> Physical Block`**. It serves a single request and is flattened on the GPU side for execution.
>    *   **Radix Tree** is an index based on **`Deduplicated Token Sequence Prefix -> Physical Block`**. It is global, living on the CPU side for cross-request cache sharing and LRU eviction.
>    *   **Collaborative Workflow**: The CPU scheduler uses the radix tree to find reusable physical blocks, adds newly allocated blocks, assembles them into a Block Table for a specific request, and passes it to the GPU. The GPU's lookup logic doesn't need to change at all.
> 
> To help you visualize their relationship more intuitively, we can represent it with the following diagram:
> 
> ```mermaid
> graph TD
>     subgraph CPU ["CPU Management Plane"]
>         RadixTree["Radix Tree<br>Index: Token Sequence ➔ Physical Block ID"]
>     end
> 
>     subgraph GPU_Mem ["GPU VRAM (Data Plane)"]
>         BlockTable["Block Table (Organized by Request)<br>Index: Request ➔ Logical Block ➔ Physical Block ID"]
>         
>         subgraph KV_Cache ["Physical Blocks Pool"]
>             B10["Physical Block 10<br>Cache: 'System Prompt...'"]
>             B11["Physical Block 11<br>Cache: 'User Question...'"]
>         end
>     end
> 
>     RadixTree -->|Maps to| B10
>     
>     RequestA["Request A (Prefix Hit)"] -->|1. Query Tree| RadixTree
>     RequestA -->|2. Assemble| BT_A["Block Table A: [10, 11]"]
>     
>     BT_A -->|3. Pass to GPU| BlockTable
>     
>     BlockTable -->|4. Points to| B10
>     BlockTable -->|4. Points to| B11
>     
>     GPU_Kernel["GPU Attention Kernel"] -->|5. Reads| BlockTable
> ```

---

### Chapter 13: The Train That Never Stops: Continuous Batching and Chunked Prefill

In Chapter 7, we saw the flaws of "static batching": To accommodate long sentences, short sentences are forced to be padded with massive amounts of Padding, wasting compute and VRAM. This chapter will introduce how modern inference engines completely solve this problem through **Continuous Batching** and **Chunked Prefill**.

#### Section 1: Continuous Batching: The Revolving Door Mechanism

To break the "barrel effect" in static batching where "short requests must wait to death for long requests", **Continuous Batching (also called In-flight Batching)**, proposed in the Orca paper and popularized by engines like vLLM, emerged.

**Analogy: The Revolving Door and the High-Speed Train That Never Stops**
Imagine a high-speed train that is always running. At every station (every model forward pass, taking tens of milliseconds), someone gets on and someone gets off.
*   **Dynamic Entry and Exit**: The system no longer waits rigidly for a whole batch of requests to finish generating completely. In the gaps between generating each token, the scheduler checks: Which request hit the end-of-sequence token (EOS)? Kick it out of the Batch immediately (get off); Are there new requests queuing up? Stuff them into the Batch immediately (get on).
*   **Eliminating Padding**: Under the hood, vLLM relies on advanced operators like FlashAttention to flatten the tokens of different requests into a one-dimensional continuous data stream and feed it to the GPU. By passing in the "boundary signposts" (`cu_seqlens` array) for each request, the GPU is able to physically isolate the computations of different requests, **completely eliminating Padding**.

This iteration-level scheduling keeps the GPU compute saturated at all times, boosting **Throughput** several times over compared to static batching, while ensuring relatively stable **Time Between Tokens (TBT)** for each request, avoiding the awkward situation of short requests being blocked by long ones.

#### The Underlying Workflow and Three Major Data Structures of Continuous Batching

To enable the "train that never stops" to run efficiently, and precisely identify tokens from different requests inside the GPU, the inference engine relies on three critical "ledgers" (data structures) at the base level. This explains how the GPU can quickly and accurately locate data without complex pointers and dynamic lookups:

1.  **`cu_seqlens` (Cumulative Sequence Length Array)**:
    *   **Role**: Responsible for **boundary isolation** during the **Prefill** phase.
    *   **Principle**: In continuous batching, the prompts of different requests are flattened into a 1D continuous token stream sent to the GPU. `cu_seqlens` records the start and end boundaries of each request (e.g., `[0, 3, 5]` means the first 3 belong to Request A, and the next 2 belong to Request B). When the Attention Kernel sees this, it knows absolutely not to "cross the line" to read a neighbor request's data when computing self-attention.

2.  **`Block Table`**:
    *   **Role**: Responsible for **historical KV traceability** (not only used for finding history in the **Decode** phase, but the GPU also needs it to look up historical KV when reading the first half of a long prompt in **Chunked Prefill** and when reading shared prefixes in **Prefix Caching**).
    *   **Principle**: This is a flattened 2D array, where rows correspond to the request index in the Batch, and columns correspond to physical block IDs. When the GPU receives a current new token for a certain request in the Batch, it doesn't traverse history via pointers. Instead, it directly uses that request's index in the Batch (e.g., the 3rd one) to query `BlockTable[3]`, directly getting the list of all physical block IDs for that request, and then goes step-by-step into VRAM to read out the historical KV.

3.  **`slot_mapping`**:
    *   **Role**: Responsible for **precise writing** across all phases.
    *   **Principle**: This is a 1D array with a length equal to the total number of tokens in the current Batch. It is pre-calculated by the CPU scheduler and directly tells the GPU into which **absolute slot** (Slot) in physical VRAM every single new token in the current batch should write its KV after calculation. The GPU just needs to execute the blazing-fast `kv_cache[slot_mapping[i]] = new_kv`, entirely avoiding complex physical address calculations inside the GPU.

This design of "CPU prepares the ledgers, GPU simply smashes them directly into VRAM addresses purely via Tensor Indexing" is the ultimate password to achieving high concurrency and low latency.

**Introducing a Special Case: Head-of-Line Blocking and Long Prompts**
The continuous batching mechanism described above is perfect when handling Decode requests (spitting out 1 token at a time, memory-access intensive). However, when a request with a long prompt containing tens of thousands of words is suddenly jammed into the queue, if the system honestly computes all of its Prefill in a single iteration, it will occupy the GPU for a long time, forcing other old requests "on the train" that are currently decoding to "stall" and stop spitting out words. This "head-of-line blocking" awkwardness directly precipitated what we will discuss in the next section—installment-plan style **Chunked Prefill**.

---

#### Section 2: Chunked Prefill: The Perfect Complement

Although continuous batching solved the stuttering in the Decode phase, a new problem arose in the **Prefill phase** (processing the user's input prompt): **Head-of-Line Blocking**.

Suppose a long prompt request containing 1000 tokens arrives. If the system honestly finishes computing all of it in one iteration, it will occupy the GPU for a long time, causing other old requests currently decoding to be forced to "stall" and not spit out words, triggering severe fluctuations in Time To First Token.

**The Breakthrough Solution: Installment Payments**
To solve this problem, the industry introduced **Chunked Prefill**:
*   The system sets a maximum token budget per iteration (e.g., 256).
*   That 1000-token long prompt is sliced into smaller chunks (e.g., `[256, 256, 256, 232]`) and processed in an "installment payment" style across multiple iterations.

**The Ultimate Fusion: The Perfect Carpool of Compute and Bandwidth**
Even more brilliantly, the system can pack **"1 chopped-up Prefill chunk"** and **"dozens of old requests currently decoding"** into the same Batch and send them to the GPU!
*   **Decode requests** only generate 1 token at a time; they are **memory-bound**, leaving GPU Tensor Core compute heavily idle but maxing out VRAM bandwidth.
*   **The Prefill chunk** contains hundreds of tokens; it is **compute-bound**, perfectly filling the GPU compute left idle by the Decode requests!

This "carpool" mode allows GPU compute and bandwidth to hit saturation simultaneously, achieving ultimate resource utilization.

---

### Chapter 14: When VRAM Bursts: Preemption and Scheduling

Even with the fine-grained management of PagedAttention, under extreme high concurrency or bombardment by ultra-long texts, there will still be a day when GPU VRAM is squeezed 100% dry. When VRAM is full but there are still pending requests, how should the scheduler make a choice?

#### Section 1: The Scheduler's Dilemma

Imagine your GPU VRAM is already running at 100% full load, and it can't carve out even one extra Block. At this point, the system faces two different types of memory demands:
1.  **Inactive Data**: Prefix KV Cache generated from previous interactions and cached in the Radix Tree (reference count is 0).
2.  **Active Data**: Requests currently being processed "on the train", which need to allocate new physical blocks when generating new tokens in the next step.

If left unhandled, the system will directly crash with an **OOM (Out of Memory)** error. As the "brain" of the system, the scheduler must trigger different response mechanisms.

---

#### Section 2: Eviction and Tiered Offloading of Inactive Cache

When VRAM bursts, the system first tries to clean up cached data that no one is temporarily using.

The system adopts an **LRU (Least Recently Used)** eviction strategy similar to an operating system:
*   The system maintains the last access timestamp and reference count for each node.
*   Nodes with a reference count greater than 0 (meaning active requests are using them) absolutely cannot be touched.
*   The system scans leaf nodes with a reference count of 0 and finds the ones least recently accessed to reclaim.

**From "Either-Or" to "Multi-Tier Storage": Tiered Offloading**

In traditional cache management, being full means "discarding." But to avoid directly throwing away precious computed results, modern engines (especially advanced systems supporting huge contexts) have introduced the **Tiered KV Cache Offloading** mechanism.

It applies the classic computer Memory Hierarchy to KV Cache management:
*   **Hot Data (GPU HBM)**: Extremely high bandwidth, extremely low latency; stores the currently most active KV Cache.
*   **Warm Data (CPU RAM)**: Connected via PCIe bus, bandwidth is an order of magnitude lower than GPU, but capacity is large and cheap. The least recently used physical blocks are **Offloaded** here.
*   **Cold Data (NVMe SSD)**: Capacity is near infinite, but access speed is the slowest. In scenarios with extreme long context or massive historical dialogues, data can sink further down to the SSD.

When the user asks another question and hits these historical caches, the system asynchronously pulls the data back from SSD/CPU memory to GPU VRAM. This granular management of "trading space for time" endows the system with near-infinite "short-term memory" capacity.

---

#### Section 3: Preemption of Active Requests: Swap vs. Recompute

If VRAM is still insufficient after cleaning the cache, the scheduler has to take draconian measures against **requests currently running**; that is, initiating the **Preemption** mechanism: pause some requests, free up their VRAM, and prioritize ensuring other requests complete smoothly.

**Who is the "Sacrificial Lamb"? — Preemption Selection Strategies for Active Requests**

When deciding to initiate preemption, the first question the scheduler faces is: **Among all the active requests currently "on the train", which one should be picked as the sacrificial lamb?**

Unlike inactive caching which uses LRU (Least Recently Used)—a "looking backward" strategy—for active requests, modern inference engines (like vLLM) generally follow the principle of **"protect old requests, sacrifice new requests"** by adopting a **Reverse FCFS (First-Come-First-Serve)** strategy:
1.  **Minimizing Sunk Costs (Wasted Work)**: Old requests have already completed massive Prefill computations and generated a fair amount of tokens; preempting them would cause a huge waste of compute power. New requests have just started, so the loss is minimal.
2.  **Guaranteeing Eventual Completion (Avoiding Starvation)**: If random preemption is used or old requests are prioritized for preemption, long-text requests might be continually interrupted and never finish.
3.  **Priority Mechanisms**: If the system supports request priorities, low-priority active requests will be picked out first.

After deciding which request to preempt, the next question is: How do we handle its already-computed KV Cache? vLLM provides two classic trade-off strategies:

**Strategy A: Swapping (Swap Out and Swap In)**
*   **Approach**: Copy the paused request's KV Cache from the expensive GPU VRAM over the PCIe bus to the cheaper **CPU memory (Host RAM)**. Once GPU VRAM loosens up, copy it back (Swap In) to resume computation.
*   **Pros and Cons**: It **saves GPU compute** (no recomputation needed), but is extremely **heavy on PCIe bandwidth**. Under high concurrency, frequent transfer of massive data can easily jam the PCIe channel entirely, causing the system throughput to drop off a cliff.

**Strategy B: Discard and Recomputation**
*   **Approach**: Just **completely delete** the paused request's KV Cache in the GPU! When it's its turn to execute again, run its historical input through the Prefill phase from scratch and recalculate the discarded KV Cache.
*   **Pros and Cons**: Sounds stupid, right? But on top-tier cards like A100/H100, the GPU's **compute capacity is severely over-provisioned**, whereas VRAM and bandwidth are the real bottlenecks. The cost of recomputation is, in many cases, **faster than moving dozens of GBs of data over PCIe**!

vLLM defaults to trying Swap first, but in extreme scenarios where VRAM and bandwidth are critically tight, Recomputation is often the ultimate lifeline to keep the system from crashing.

---

#### Section 4: SGLang's Tree-based Management: Integrating Preemption and Eviction

After discussing the traditional Swap and Recompute strategies, let's look at how inference engines natively based on the **Radix Tree**, represented by **SGLang**, handle situations when VRAM is full.

SGLang's core idea is to completely unify **cache sharing, eviction, and preemption of active requests** into the topological structure of a single tree:

1.  **Preemption is Dereferencing**:
    In SGLang, the KV Caches of all requests are branches on the tree. When an active request needs to be preempted (paused) due to insufficient VRAM, the system doesn't need to do any cross-medium data transfer (like Swap), nor does it need to erase data immediately (like Recompute). It simply pauses the request, and the **reference count** of its corresponding node on the Radix Tree **drops to zero**.

2.  **Best-effort Retention**:
    These nodes whose reference counts drop to zero still remain in VRAM, degrading into "inactive cache." If the ensuing VRAM pressure is alleviated and these nodes survive the LRU (Least Recently Used) eviction mechanism, then when the request resumes, the system directly **hits the cache**, achieving a "zero-cost" recovery.

3.  **Ultimate Simplicity**:
    This design avoids explicit Swap state machines and complex cross-device memory scheduling. If VRAM is truly insufficient, the tree's LRU mechanism will naturally prune the least recently used leaf nodes, and the evicted requests will automatically fall back to Recompute upon recovery. This "govern by doing nothing" design performs exceptionally elegantly when handling complex scenarios like multi-turn dialogues and agent branching.

---

### Chapter 15: Trading "Idle Compute" for "Ultimate Latency": Speculative Decoding

On the battlefield of large model inference, we have solved the "land enclosure waste" of VRAM fragmentation through PagedAttention, and eliminated the "Padding bubble" through continuous batching. However, during the auto-regressive Decode phase, we still face a brutal physical reality: **The Memory Bandwidth Bottleneck (Memory-Bound)**.

As we described in Chapter 8, the Decode phase is like "driving a heavy truck to deliver a single screw." To generate just 1 Token, the GPU must move hundreds of GBs of model weights from VRAM to the compute cores completely once. The GPU's mighty Tensor Core compute power is "sleeping soundly" waiting for data the vast majority of the time.

Since the hardware VRAM bandwidth is locked, can the large model **do a bit more work** each time it moves the weights? System engineers and algorithm scientists jointly launched a clever technology—**Speculative Decoding**. Its core idea is: **It does not try to change the physical limitation of VRAM bandwidth, but through a strategy of "trading space (increased computation) for time (reduced latency)", it fully leverages the GPU's idle compute power to increase the Arithmetic Intensity of a single memory access, thereby drastically reducing generation latency at low concurrency and significantly shrinking TBT (boosting single-user TPS). But it's important to note that under extremely high concurrency, it might compete for compute resources, paradoxically having a negative impact on the system's overall Throughput.**

#### Section 1: The Professor and the Assistant — The Core Logic of Speculative Decoding

The working principle of speculative decoding can be understood using the analogy of a **"Professor and Assistant"**:
*   **Target Model**: A highly knowledgeable old professor (the large model). Top-tier knowledge, but writes extremely slowly, yet can tell at a glance if someone else wrote something correctly.
*   **Draft Model**: A young, quick-handed assistant (a super small model or external head). Average knowledge, but extremely fast hands, though occasionally makes mistakes.

If the old professor writes word by word himself, it takes a very long time. The approach of speculative decoding is:
1.  **Drafting**: The assistant relies on feeling and quickly scribbles down $K$ words (e.g., 5 words). Because the assistant model is very small, these 5 words are generated extremely fast.
2.  **Verification**: The large model eats these 5 words all at once. **Note: The large model does not read word by word here, but uses the GPU's parallel capabilities (similar to Prefill) to compute the words it would have said at these 5 positions all in one go.**
3.  **Acceptance and Correction**: If the old professor finds the assistant guessed the first 3 words right, but the 4th word is inappropriate. The professor will correct the 4th word to the right one, and throw away the 5th word (i.e., roll back the KV Cache).
After this round, the old professor only performed one "forward pass", but we actually got 4 high-quality words completely approved by the old professor! This is much faster than the old professor writing 4 times himself.

#### Section 2: The "Reverse Trade" of Arithmetic Intensity

You might keenly observe: The small model computes once, and the large model verifies once again, **doesn't the total computation (FLOPs) increase?**

**Yes, the total computation definitely increases. But this is absolutely a highly profitable trade.**

We calculated in Chapter 8 that the arithmetic intensity of the Decode phase is extremely low (e.g., 1.9 FLOPs/Byte), far below the hardware inflection point. The GPU's compute cores are in a state of severe starvation.
Although speculative decoding increases the computational load, it allows the large model to **process multiple tokens at once**. The weights the large model needs to read to verify 5 words are **exactly the same** as the weights it needs to read to generate 1 word (both are hundreds of GBs). We only moved the building (weights) once, but incidentally processed 5 jobs.
Although this does not physically alter the limitation of VRAM bandwidth, it significantly boosts the arithmetic density of a single memory access, allowing the originally starved GPU cores to feast, "moving the needle," and trading idle compute for shorter elapsed times.

#### Section 3: From Dual Models to External Heads: The Evolution of Architecture

How to elegantly "draft" is the core of technological evolution in recent years. This has gone through an evolution from "dual models" to "single model external heads."

**1. The Classic Dual-Model Scheme**
*   **Principle**: This is the progenitor of speculative decoding. It simultaneously loads two independent models into VRAM: a massive **Target Model** (like Llama-3 70B) and an extremely small **Draft Model** (like Llama-3 8B, or even smaller distilled models).
*   **Workflow**: The small model honestly does autoregressive Decode, generating $K$ words; the large model verifies these $K$ words in parallel.
*   **Pain Points**: The engineering implementation is extremely heavy. You need to serve two models simultaneously and manage two independent sets of KV Cache. If the small model is too independent, the hit rate is hard to guarantee, and the small model itself also occupies precious VRAM.

**2. Medusa: Multi-Head "Blind Guessing"**
To solve the clunkiness of dual models, **Medusa** proposed a very aggressive idea: **No small model!** It directly attaches several parallel, ultra-lightweight external Heads to the final layer's Hidden State of the large model.
*   **Principle**: Head 1 predicts the word at $t+1$; Head 2 **blindly guesses** the word at $t+2$; Head 3 **blindly guesses** the word at $t+3$.
*   **Limitations**: This is a non-autoregressive "blind guess". Head 2 forcefully guesses $t+2$ without knowing what $t+1$ is; as the number of steps increases, the accuracy plummets like a cliff.

**3. Eagle: Feature-Level Autoregression**
To solve the "inelegance" of Medusa, **Eagle** introduced a design more aligned with the chain rule of language.
*   **Principle**: It introduces an extremely small single-layer Transformer at the feature level. It combines the large model's feature $h_t$ with the predicted Token to autoregressively derive $h'_{t+1}$ using the small network, and then uses that to predict $t+2$. This maintains a very high prediction accuracy while avoiding the massive time consumption of the large model's dozens of layers. It's a more elegant way of "drafting".

> [!NOTE]
> **Core Reflections: Who trained these "plugins"? Why don't large models come with them natively? And where are they implemented?**
> 1. **Who trained them?**: They are usually trained by the open-source community or enterprises targeting specific scenarios (e.g., code generation, legal documents). Because the hit rate is extremely dependent on the vocabulary distribution of the business scenario, general heads are often less effective than customized ones.
> 2. **Why not included natively?**: The creators of Base Models (like Meta) focus on building the smartest "brain", while speculative decoding belongs to "system-level acceleration." Decoupling allows the large model to remain pure, letting downstream users customize acceleration plugins according to their needs.
> 3. **Where are they implemented?**: All the scheduling, dual-model communication, and the most difficult **tree-based KV Cache dynamic pruning (rollback)** are all implemented within the **Inference Framework (like vLLM, SGLang)**. The large model itself only does matrix multiplication; the framework is the precise commander.

#### Section 4: Tree Attention and Trade-offs in Production Environments

**1. Tree Attention**
Whether Medusa or Eagle, to improve the hit rate, both adopt a "cast a wide net" strategy: offering Top-K candidate words at every step branching out. In the autoregressive process, this naturally grows a **"Draft Tree"**.
The inference framework (like vLLM) feeds this tree to the large model all at once. The large model's KV Cache will **expand into a tree structure** in the instant of verification, the large model uses Tree Attention to find that single correct path, and then the framework will **prune (i.e., KV Cache rollback)** the cache of other branches, turning it back into a straight line.

It must be pointed out that, although there is the absolute authority of the large model (the old professor) acting as a safety net ensuring final output accuracy is completely unaffected, performing this kind of tree-based pruning and physical rollback of KV Cache on the GPU is not a free lunch. If the draft model's (assistant's) "hit rate" is too low, the system frequently spinning in the invalid loop of "generate-verify-discard-rollback" will not only fail to accelerate, but instead slow down overall inference speed due to the overhead of frequent VRAM pointer operations and management. This further explains why speculative decoding in production environments requires "dynamic trade-offs" based on concurrency and hit rates.

**2. Dynamic Trade-offs in Production Environments**
In production environments, speculative decoding is not turned on blindly:
*   **When concurrency is low / pursuing ultimate latency** (e.g., Batch Size = 1, real-time conversation): Turn on speculative decoding. GPU compute is idle; use the surplus compute to trade for faster generation speeds.
*   **When concurrency is high / pursuing ultimate throughput** (peak times): **Turn off or dial down** speculative decoding. Because at this point, the GPU's compute cores are already stuffed full by massive Batches, doing speculative decoding is pure waste and will actually lead to longer queue times. At this point, it's more profitable to batch a few more requests than to guess a few more words.


In Part Three, we ventured to the absolute forefront of single-node inference optimization, witnessing how "tactical" marvels like GQA, PagedAttention, Continuous Batching, and Speculative Decoding each show their prowess to squeeze the performance of a single graphics card to its limits. These optimizations have successfully brought large models out of the lab, giving them the confidence to serve millions of users.

However, when model parameters scale to hundreds of billions or trillions, and when context windows reach the millions, the physical limits of a single machine will ultimately be shattered. How can we make hundreds or thousands of graphics cards work together? How can we achieve more fundamental "strategic" resource isolation? Please follow me into **Part Four**, where we will break free from the confines of single nodes and overlook the grand chessboard of distributed inference and cluster scheduling.

---
