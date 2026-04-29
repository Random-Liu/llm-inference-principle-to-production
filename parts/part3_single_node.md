## Part Three: Single Node — High-Performance Engines Squeezing Every Inch of VRAM

## Table of Contents
- [Chapter 9: Architecture Optimization: GQA](#chapter-9-architecture-optimization-gqa)
  - [Section 1: Evolution: From MHA to GQA](#section-1-evolution-from-mha-to-gqa)
  - [Section 2: Trimming Rationale: Why Reducing Only K and V Works](#section-2-trimming-rationale-why-reducing-only-k-and-v-works)
  - [Section 3: Frontier Progress: Alternative KV Compression Methods](#section-3-frontier-progress-alternative-kv-compression-methods)
- [Chapter 10: Precision Reduction: KV Cache Quantization (FP8/INT8)](#chapter-10-precision-reduction-kv-cache-quantization-fp8int8)
  - [Section 1: Trade-off: Compute for Bandwidth](#section-1-trade-off-compute-for-bandwidth)
  - [Section 2: Rationale: Why It Works](#section-2-rationale-why-it-works)
  - [Section 3: INT8 vs. FP8: Different Paradigms](#section-3-int8-vs-fp8-different-paradigms)
  - [Section 4: Dynamic vs. Static: Contrast with Weight Quantization](#section-4-dynamic-vs-static-contrast-with-weight-quantization)
- [Chapter 11: VRAM Management at the Engine Level: PagedAttention](#chapter-11-vram-management-at-the-engine-level-pagedattention)
  - [Section 1: Fragmentation Crisis: Waste from Static Contiguous Allocation](#section-1-fragmentation-crisis-waste-from-static-contiguous-allocation)
  - [Section 2: OS Inspiration: Virtual Memory Paging](#section-2-os-inspiration-virtual-memory-paging)
  - [Section 3: Block Tables: Minimizing Memory Waste](#section-3-block-tables-minimizing-memory-waste)
- [Chapter 12: Memory Time Machine: Prefix Caching (RadixAttention)](#chapter-12-memory-time-machine-prefix-caching-radixattention)
  - [Section 1: Dilemma: RAG and Multi-Turn Dialogues](#section-1-dilemma-rag-and-multi-turn-dialogues)
  - [Section 2: Radix Trees: Sharing Physical Memory](#section-2-radix-trees-sharing-physical-memory)
- [Chapter 13: The Train That Never Stops: Continuous Batching and Chunked Prefill](#chapter-13-the-train-that-never-stops-continuous-batching-and-chunked-prefill)
  - [Section 1: Continuous Batching: Revolving Door Mechanism](#section-1-continuous-batching-revolving-door-mechanism)
  - [Section 2: Chunked Prefill: Perfect Complement](#section-2-chunked-prefill-perfect-complement)
- [Chapter 14: When VRAM Bursts: Preemption and Scheduling](#chapter-14-when-vram-bursts-preemption-and-scheduling)
  - [Section 1: Dilemma: Scheduler Decisions](#section-1-dilemma-scheduler-decisions)
  - [Section 2: Cache Management: Eviction and Tiered Offloading](#section-2-cache-management-eviction-and-tiered-offloading)
  - [Section 3: Preemption: Swap vs. Recompute](#section-3-preemption-swap-vs-recompute)
  - [Section 4: SGLang: Unifying Preemption and Eviction under Tree Management](#section-4-sglang-unifying-preemption-and-eviction-under-tree-management)
- [Chapter 15: Trading "Idle Compute" for "Ultimate Latency": Speculative Decoding](#chapter-15-trading-idle-compute-for-ultimate-latency-speculative-decoding)
  - [Section 1: Analogy: The Professor and the Assistant](#section-1-analogy-the-professor-and-the-assistant)
  - [Section 2: Economics: Arithmetic Intensity Trade-offs](#section-2-economics-arithmetic-intensity-trade-offs)
  - [Section 3: Evolution: From Dual Models to External Heads](#section-3-evolution-from-dual-models-to-external-heads)
  - [Section 4: Production: Tree Attention and Dynamic Trade-offs](#section-4-production-tree-attention-and-dynamic-trade-offs)

Part Two dissected the physical and mathematical bottlenecks of LLM inference: the **KV Cache VRAM tsunami** and the **core asymmetry between Prefill and Decode**. These bottlenecks limit concurrency and response speed in production.

To overcome these limits, engineers and scientists designed co-optimized hardware and software. This part explores how modern inference engines (like vLLM and SGLang) and model architectures solve these bottlenecks across two battlefields:
1. **VRAM Optimization**: GQA, PagedAttention, and RadixAttention multiply VRAM utilization.
2. **Asymmetry Management**: Continuous Batching and Chunked Prefill eliminate padding, saturating GPU compute and bandwidth.

> [!IMPORTANT]
> Continuous Batching and Chunked Prefill are "tactical" local optimizations. They do not fundamentally resolve the hardware mismatch and resource contention between Prefill and Decode. We will reveal a more thorough "strategic" evolution—**Disaggregated Serving**—in the cluster section of Part Four.

Now, let's look at the first battlefield: slimming down VRAM at the model architecture level.

### Chapter 9: Architecture Optimization: GQA

#### Section 1: Evolution: From MHA to GQA

Early Transformers used **MHA (Multi-Head Attention)**.
*   **MHA**: $H$ Query heads require $H$ Key and $H$ Value heads. As discussed in [Chapter 6](part2_bottlenecks.md#section-3-the-cost-the-vram-tsunami), KV Cache space complexity is $O(L \cdot N \cdot d)$. Although the hidden dimension $d$ is split into $H$ heads ($d = H \times d_k$), the total KV Cache size depends on the full dimension $d$ because the number of KV heads equals Query heads. For large models like Llama 3 405B ($d = 16384$), large $d$ and long sequences $N$ result in a massive KV Cache.

To reduce KV Cache, researchers proposed **MQA (Multi-Query Attention)**:
*   **MQA**: All Query heads share a single set of Key and Value heads. This shrinks the KV Cache to $1/H$ of its original size, drastically reducing VRAM pressure at the cost of some of the model's expressive power.

**GQA (Grouped Query Attention)** compromises between the two and is widely used in modern models like Llama 3 405B:
*   **GQA**: Query heads are grouped (e.g., 8 heads per group), and each group shares one set of Key and Value heads.
*   **Benefits**: It cuts the VRAM footprint, increasing system throughput. It also boosts single-request TPS by reducing data movement during Decode, with negligible performance loss.

**Comparison: KV Cache Sizes for Llama 3 405B**

To compare these mechanisms, we calculate the KV Cache size for **Llama 3 405B** ($L=126$, $d=16384$, $H=128$, $d_k=128$, FP16) with a **1-million token** context:

*   **Standard MHA Mode** (assuming each Query head has independent KV heads):
    *   Size per token per layer: $2 \times 128 \times 128 \times 2 \text{ bytes} = 64 \text{ KB}$
    *   Total size: $64 \text{ KB} \times 126 \text{ layers} \times 1,000,000 \approx \mathbf{8.06 \text{ TB}}$
*   **MQA Mode** (all Query heads share 1 group of KV heads):
    *   Size per token per layer: $2 \times 1 \times 128 \times 2 \text{ bytes} = 0.5 \text{ KB}$
    *   Total size: $0.5 \text{ KB} \times 126 \text{ layers} \times 1,000,000 \approx \mathbf{63 \text{ GB}}$
*   **GQA Mode** (actually adopted by Llama 3, 8 groups of KV heads):
    *   Size per token per layer: $2 \times 8 \times 128 \times 2 \text{ bytes} = 4 \text{ KB}$
    *   Total size: $4 \text{ KB} \times 126 \text{ layers} \times 1,000,000 \approx \mathbf{504 \text{ GB}}$

Reducing the cache from $8 \text{ TB}$ to $504 \text{ GB}$, GQA achieves a 16-fold compression, enabling long-context processing on a single node or small cluster.

#### Section 2: Trimming Rationale: Why Reducing Only K and V Works

**Why can we trim $K$ and $V$ while leaving $Q$ intact without losing expressive power?**

To understand this, we analyze the roles of $Q$, $K$, and $V$, information redundancy, and knowledge storage.

**1. Role Asymmetry**: $Q$, $K$, and $V$ have different roles:
*   **Query ($Q$)** represents **intent** (What am I looking for?). It is dynamic and changes with every generated token.
*   **Key ($K$)** represents **indices** (What do I have here?).
*   **Value ($V$)** represents **content** (What is the actual information?).

**Conceptually, the same fact ($K$ and $V$) can answer many different questions ($Q$).**

> [!NOTE]
> **Analogy**: Imagine **128 researchers** (Query heads) in a library asking different questions.
> *   **MHA**: The library gives each researcher a dedicated set of encyclopedias (Key/Value heads). This wastes space because the books record heavily overlapping facts.
> *   **GQA**: 16 researchers share one set of books. Only **8 sets** are needed.
> 
> Although researchers ask different questions ($Q$), the facts they look up ($K$ and $V$) are identical. Thus, $Q$ heads ensure diverse thought, while reduced $K/V$ heads save space.

**2. Information Redundancy**: In standard MHA, visual analysis shows that **many Key and Value heads learn repetitive features** (e.g., multiple heads focusing on the subject). GQA enforces deduplication by sharing $K$ and $V$ heads, forcing the model to utilize parameters more efficiently without losing power.

**3. FFN Stores the Knowledge**: **Most knowledge resides in the FFN (Feed-Forward Network), not in Attention.**
FFN parameters account for about 82% of each layer, while Attention accounts for less than 20%.
*   **FFN** stores objective laws and facts.
*   **Attention** routes information and connects context.

Since Attention only **moves and correlates** context, compressing its KV cache does not destroy FFN knowledge. The model still knows facts; it just uses fewer "memory pointers" (KV) during inference.

#### Section 3: Frontier Progress: Alternative KV Compression Methods

Beyond GQA, other developments aim to combat the VRAM tsunami:

1.  **MLA (Multi-head Latent Attention)**:
    *   **Principle**: DeepSeek-V2 projects K and V into a low-dimensional latent space, caching only this vector for higher compression than GQA.
    *   **Reference**: [DeepSeek-V2 Paper](https://arxiv.org/abs/2405.04434)
2.  **Sliding Window Attention**:
    *   **Principle**: Mistral 7B maintains a fixed-size window of recent tokens, discarding older KV cache to reduce complexity to $O(1)$.
    *   **Reference**: [Mistral 7B Paper](https://arxiv.org/abs/2310.06825)
3.  **Interleaved Local/Global Attention**:
    *   **Principle**: Combines sliding window and global attention across layers to balance efficiency and long-distance dependencies.
    *   **Reference**: Refer to the official technical reports of the respective models.
4.  **Infini-Attention (Compressive Memory Attention)**:
    *   **Principle**: Google's method stores old KV states in a fixed-size compressive memory, theoretically supporting infinite context.
    *   **Reference**: [Leave No Context Behind Paper](https://arxiv.org/abs/2404.07143)

These advancements show that **reducing K and V size via architecture remains a core optimization focus.**

These methods require training with the specific architecture. Alternatively, we can compress the KV Cache at the **precision level** without changing the architecture.

### Chapter 10: Precision Reduction: KV Cache Quantization (FP8/INT8)

If GQA pushes the **spatial structure** to the extreme (reducing data volume), then KV Cache quantization brings the hammer down on **data density**.

#### Section 1: Trade-off: Compute for Bandwidth

Standard inference stores K and V at 16-bit precision (FP16 or BF16), using 2 bytes per element. **KV Cache quantization** compresses them to 8 bits (FP8 or INT8), reducing size to 1 byte and halving the VRAM footprint.

This introduces computational overhead:
1. **On Write**: New tokens must be scaled and truncated from 16 to 8 bits before storage.
2. **On Read**: The GPU must dequantize the 8-bit values back to 16 bits for calculation (unless the hardware supports direct low-precision compute).

This essentially uses extra computation to compress storage, which is a highly profitable trade-off during the Decode phase.

#### Section 2: Rationale: Why It Works

As discussed in [Chapter 8](part2_bottlenecks.md#chapter-8-core-asymmetry-prefill-vs-decode), Decode is **Memory-Bound**. GPU cores sit idle, waiting for data from VRAM.

* **Without Quantization**: Data is large, transfer is slow, and cores are starved.
* **With Quantization**: Data volume is halved, doubling transfer speed and feeding cores faster.

On modern GPUs like NVIDIA H100, native FP8 support makes the compute overhead negligible. Trading minor computation for halved VRAM and doubled speed is standard. Compressing to 8-bit typically causes less than 0.5% accuracy loss while doubling concurrent capacity.

#### Section 3: INT8 vs. FP8: Different Paradigms

Two distinct routes exist for 8-bit KV Cache quantization:

* **INT8 (8-bit Integer)**: Maps floats to integers by scaling into 256 discrete grids. It requires dequantizing back to FP16 for compute. Fine-grained quantization minimizes accuracy drops.
* **FP8 (8-bit Floating Point)**: Reduces precision and range while keeping the float structure. NVIDIA H100 computes FP8 extremely fast without explicit dequantization, making it the server standard.

#### Section 4: Dynamic vs. Static: Contrast with Weight Quantization

Weight quantization (e.g., GPTQ, AWQ) is also common, but it differs from KV Cache quantization in difficulty and impact:

* **Weight Quantization (Static)**: Weights are fixed and analyzed offline. Large models retain accuracy even at 4-bit.
* **KV Cache Quantization (Dynamic)**: Activations change with every input, producing dynamic outliers. Compressing to 4-bit destroys range and quality. Thus, KV cache usually stays at 8-bit.

Common combinations:
* **W4A16 (Weights Only)**: Focuses on capacity, fitting large models into smaller GPUs.
* **W8A8 / FP8 (Full Quantization)**: Focuses on speed, ideal for high concurrency.

---

Although the size of the KV Cache is substantially compressed through GQA and quantization techniques, as context grows, it still causes severe fragmentation issues in VRAM. This is why we still need **PagedAttention**.

### Chapter 11: VRAM Management at the Engine Level: PagedAttention

#### Section 1: Fragmentation Crisis: Waste from Static Contiguous Allocation

Because final sequence lengths are unpredictable, traditional VRAM management used **static contiguous allocation**. Systems pre-allocated VRAM based on the maximum context length (e.g., 8000 tokens).

This caused severe waste:
* **Internal Fragmentation**: Systems pre-allocated contiguous VRAM for the maximum length. Unused reserved space could not be used by other requests until the current one finished.
* **External Fragmentation**: Scattered non-contiguous free spaces could not be allocated to new requests requiring contiguous blocks.

The vLLM paper noted that traditional static allocation often utilized less than 20% of VRAM, wasting 80%.

---

#### Section 2: OS Inspiration: Virtual Memory Paging

To solve this, vLLM researchers adopted the **virtual memory paging (Paging)** mechanism from operating systems, which splits physical memory into fixed-size "Pages" to allow logically contiguous memory to be physically scattered.

vLLM applies this to GPU VRAM management:
1. **Block Management**: Physical VRAM is divided into fixed-size Physical Blocks (e.g., storing K and V for 16 tokens).
2. **Non-contiguous Allocation**: Token sequences are logically contiguous but can be stored in non-contiguous physical blocks.

---

#### Section 3: Block Tables: Minimizing Memory Waste

To find scattered tokens, PagedAttention introduces **Block Tables**, similar to OS Page Tables, mapping logical blocks to physical blocks. During $Q \cdot K^T$ calculation, the attention mechanism queries the table to read K and V vectors from discrete blocks.

**Mapping Mechanism**: The `block_table` maintains the `Request -> Token Block -> Memory Block` mapping as a 2D tensor shaped `[max_num_reqs, max_num_blocks_per_req]`:
* **Rows**: Requests (`req_idx`).
* **Columns**: Token Blocks (`logical_block_idx`).
* **Value**: Physical VRAM block ID.
Lookup formula: `physical_block_id = block_table[req_idx][logical_block_idx]`

**Benefits**:
* **Minimal Waste**: On-demand allocation limits waste to the last unfilled block (at most 15 tokens). VRAM waste drops from 80% to under 4%.
* **Limitation**: PagedAttention isolates KV Cache across requests. It cannot automatically reuse identical prefixes across different requests.

The next chapter covers cross-request cache reuse.

---

### Chapter 12: Memory Time Machine: Prefix Caching (RadixAttention)

#### Section 1: Dilemma: RAG and Multi-Turn Dialogues

Prompts often contain massive background materials.

**Scenario 1: Multi-Turn Dialogue**
*   Turn 1: User asks "What is an apple?" (Model computes and answers).
*   Turn 2: User asks "Is it tasty?". The prompt becomes: [Turn 1 Q + Turn 1 A + Turn 2 Q].

Because **HTTP is stateless**, serving engines (like vLLM) treat the second turn as a **brand-new request** with a new ID.

In traditional PagedAttention, Block Tables are bound to Request IDs. Engines cannot automatically reuse Turn 1's KV Cache for Turn 2 because the IDs differ.

Without prefix caching, the model recalculates QKV for Turn 1 content. As turns increase, TTFT grows linearly. This isolation necessitates **Prefix Caching**.

**Scenario 2: RAG (Knowledge Base Q&A)**
Asking questions against a 100,000-word manual requires reprocessing the manual for every question, causing high TTFT.

**Scenario 3: Fixed System Prompts**
Enterprise apps often prepend lengthy System Prompts or few-shot examples. Without optimization, 1000 concurrent users cause the system to recalculate the same KV Cache 1000 times.

**Scenario 4: Parallel Sampling**
In code generation or Beam Search, systems generate multiple continuations for one prompt. Without optimization, the system duplicates and recalculates the prompt's KV cache for each branch.

---

#### Section 2: Radix Trees: Sharing Physical Memory

Frameworks like SGLang and vLLM introduced **RadixAttention**, using a **Radix Tree** to manage KV Cache.

In Transformers, identical preceding token sequences produce identical KV Caches.

The Radix Tree works as follows:
*   **Root Node**: Empty sequence.
*   **Edges**: Continuous token sequences (e.g., 16-token blocks).
*   **Nodes**: Point to corresponding physical blocks.

New requests trigger a **longest prefix match** against the tree's edges. Successful matches reuse physical blocks directly, skipping matrix multiplication. Unmatched portions allocate new blocks and create new leaf nodes.

This enables instant reuse for multi-turn dialogues and RAG, reducing TTFT from seconds to milliseconds.

> [!NOTE]
> **Indexing and Implementation Details**
> 1. **Token Indexing**: The Radix Tree indexes **Token IDs** (integers), not text. Matching involves highly efficient integer sequence comparisons or hashes (e.g., vLLM uses hashes to quickly anchor blocks).
> 2. **vLLM vs. SGLang**: Both use Radix Trees but differ in granularity. SGLang operates at the **Token level**, allowing edges to represent any sequence length for flexible matching. vLLM's Automatic Prefix Caching operates at the **Block level** (typically 16 tokens), inheriting PagedAttention's structure.
> 3. **Radix Tree vs. Block Table**: They do not replace each other but point to the same physical blocks from different dimensions:
>    * **Block Table**: Maps `Request -> Logical Block -> Physical Block`. Maintained by the CPU's Block Manager and passed as a tensor to the GPU for execution.
>    * **Radix Tree**: A global index mapping `Deduplicated Token Sequence Prefix -> Physical Block`. It lives on the CPU for cross-request sharing and LRU eviction.
>    * **Workflow**: The CPU scheduler uses the Radix Tree to find reusable physical blocks, allocates new blocks, assembles them into a Block Table for a request, and passes it to the GPU. GPU execution logic remains unchanged.
> 
> To help you visualize their relationship more intuitively, we can represent it with the following diagram:
> 
> ```mermaid
> graph TD
>     subgraph CPU ["CPU Management Plane"]
>         RadixTree["🌲 Radix Tree<br>Index: Token Sequence ➔ Physical Block ID"]
>         BlockManager["⚙️ Block Manager<br>(Manages mapping & allocation)"]
>     end
> 
>     subgraph GPU_Mem ["GPU VRAM (Data Plane)"]
>         BlockTableTensor["📋 Block Table Tensor<br>(For GPU execution lookup)"]
>         
>         subgraph KV_Cache ["Physical Blocks Pool"]
>             B10["📦 Physical Block 10<br>Cache: 'System Prompt...'"]
>             B11["📦 Physical Block 11<br>Cache: 'User Question...'"]
>         end
>     end
> 
>     RadixTree -->|Maps to| B10
>     
>     RequestA["👤 Request A (Prefix Hit)"] -->|1. Query Tree| RadixTree
>     RadixTree -->|2. Return matched blocks| BlockManager
>     BlockManager -->|3. Assemble| BT_A["📋 Request A's Block Table: [10, 11]"]
>     
>     BT_A -->|4. Pass to GPU| BlockTableTensor
>     
>     BlockTableTensor -->|5. Points to| B10
>     BlockTableTensor -->|5. Points to| B11
>     
>     GPU_Kernel["⚡ GPU Attention Kernel"] -->|6. Reads| BlockTableTensor
> ```

---

### Chapter 13: The Train That Never Stops: Continuous Batching and Chunked Prefill

Chapter 7 discussed static batching flaws: short sentences require massive padding to match long sentences, wasting compute and VRAM. This chapter introduces how engines solve this with **Continuous Batching** and **Chunked Prefill**.

#### Section 1: Continuous Batching: Revolving Door Mechanism

To prevent short requests from waiting for long ones, **Continuous Batching** (or In-flight Batching) emerged, proposed by Orca and popularized by vLLM.

**Analogy**: Imagine a continuous train where passengers get on and off at every station (model forward pass).
* **Dynamic Entry/Exit**: The system does not wait for a full batch to finish. Between token generations, the scheduler removes finished requests (EOS) and inserts new ones from the queue.
* **No Padding**: Engines use operators like FlashAttention to flatten tokens into a 1D stream. Passing boundaries via `cu_seqlens` allows the GPU to isolate request computations, **eliminating padding**.

**Diagram: Matrix vs. Vector**

**1. Static Batching — Padded into a 'Matrix'**

| | Col 1 | Col 2 | Col 3 | Col 4 |
| :--- | :--- | :--- | :--- | :--- |
| **Row 1 (🔵)** | 🔵 T1 | 🔵 T2 | 🔵 T3 | ❌ PAD |
| **Row 2 (🟢)** | 🟢 T1 | 🟢 T2 | ❌ PAD | ❌ PAD |
| **Row 3 (🟡)** | 🟡 T1 | 🟡 T2 | 🟡 T3 | 🟡 T4 |

**2. Continuous Batching — Concatenated into a 'Vector'**

| | Col 1 | Col 2 | Col 3 | Col 4 | Col 5 | Col 6 | Col 7 | Col 8 | Col 9 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Row 1** | 🔵 T1 | 🔵 T2 | 🔵 T3 | 🟢 T1 | 🟢 T2 | 🟡 T1 | 🟡 T2 | 🟡 T3 | 🟡 T4 |

> [!NOTE]
> `cu_seqlens = [0, 3, 5, 9]` (Physically isolates computations via offsets)
This iteration-level scheduling saturates GPU compute, multiplying throughput and maintaining stable **Time Between Tokens (TBT)** without blocking short requests.

#### Core Data Structures

To identify tokens from different requests in the GPU, the engine relies on three ledgers:

1. **`cu_seqlens` (Cumulative Sequence Length Array)**:
   * **Role**: Isolates token boundaries of different requests in the Batch.
   * **Principle**: Flattens input tokens into a 1D stream. `cu_seqlens` records boundaries (e.g., `[0, 3, 4]` means the first 3 belong to Request A, and the 4th belongs to Request B). This is crucial when a batch mixes Prefill (multiple tokens) and Decode (single token) requests, ensuring the Attention Kernel does not cross boundaries.
2. **`Block Table`**:
   * **Role**: Traces historical KV for Decode, Chunked Prefill, and Prefix Caching.
   * **Principle**: A 2D array mapping request index to physical block IDs. The GPU uses the request index to fetch all physical block IDs directly, avoiding pointer traversal.
3. **`slot_mapping`**:
   * **Role**: Ensures precise writing across all phases.
   * **Principle**: A 1D array **pre-calculated by the CPU scheduler**, mapping each token in the batch to an absolute VRAM slot. The GPU executes `kv_cache[slot_mapping[i]] = new_kv` without complex address calculations.

CPU-prepared ledgers combined with GPU Tensor Indexing enable high concurrency and low latency.

**Special Case: Head-of-Line Blocking and Long Prompts**
Continuous batching excels at Decode (1 token at a time). However, long prompts (Prefill) can occupy the GPU for too long, stalling existing Decode requests. This "head-of-line blocking" necessitates **Chunked Prefill**.

---

#### Section 2: Chunked Prefill: Perfect Complement

While Continuous Batching optimizes Decode, long **Prefill** requests can still cause head-of-line blocking, stalling other requests.

**Solution: Chunked Prefill**
* Systems set a maximum token budget per iteration (e.g., 256).
* Long prompts are sliced into chunks (e.g., `[256, 256, 256, 232]`) and processed across multiple iterations.

**Fusion: Compute and Bandwidth Carpooling**
Systems can pack one Prefill chunk and dozens of Decode requests into the same batch:
* **Decode requests** are memory-bound, leaving GPU compute idle.
* **Prefill chunks** are compute-bound, filling the idle compute.

This carpooling saturates both compute and bandwidth.

---

### Chapter 14: When VRAM Bursts: Preemption and Scheduling

Even with PagedAttention, extreme concurrency or ultra-long texts can exhaust GPU VRAM. When VRAM is full and requests are pending, the scheduler must make a choice.

#### Section 1: Dilemma: Scheduler Decisions

When VRAM hits 100% load, the system faces two demands:
1. **Inactive Data**: Cached prefix KV Cache (ref count 0).
2. **Active Data**: Running requests needing new blocks for new tokens.

To avoid **OOM (Out of Memory)** crashes, the scheduler must act.

---

#### Section 2: Cache Management: Eviction and Tiered Offloading

The system first cleans up unused cached data using an **LRU (Least Recently Used)** eviction strategy:
* The system maintains the last access timestamp and reference count for each node.
* Nodes with ref count > 0 are protected.
* Leaf nodes with ref count 0 are scanned, and the least recently accessed are reclaimed.

**Tiered Offloading**
Instead of discarding results, modern engines use **Tiered KV Cache Offloading**, applying memory hierarchy:
* **Hot Data (GPU HBM)**: High bandwidth, low latency; stores active cache.
* **Warm Data (CPU RAM)**: Lower bandwidth (PCIe), large and cheap. Least recently used blocks are **Offloaded** here.
* **Cold Data (NVMe SSD)**: Slowest access, massive capacity for extreme contexts.

When accessed, data is asynchronously pulled back to GPU VRAM, greatly expanding "short-term memory" capacity.

---

#### Section 3: Preemption: Swap vs. Recompute

If VRAM is still insufficient, the scheduler initiates **Preemption**: pausing some requests to free VRAM for others.

**Selection Strategies**
Instead of LRU, engines usually use **Reverse FCFS** ("protect old, sacrifice new"):
1. **Minimize Sunk Cost**: Old requests completed massive compute; new requests lose less if preempted.
2. **Avoid Starvation**: Random preemption might prevent long requests from ever finishing.
3. **Priority**: If supported, low-priority requests are sacrificed first.

**Handling KV Cache: Swap vs. Recompute**
* **Strategy A: Swapping**:
  * **Approach**: Moves the paused request's KV Cache to **CPU memory (Host RAM)** over PCIe, and swaps it back when VRAM is available.
  * **Pros & Cons**: Saves compute but consumes heavy PCIe bandwidth. High concurrency can jam the PCIe channel, dropping throughput.
* **Strategy B: Recomputation**:
  * **Approach**: Deletes the cache and recalculates it by running the historical prompt through Prefill when resumed.
  * **Pros & Cons**: On powerful GPUs (A100/H100), compute capacity is over-provisioned while bandwidth is the bottleneck. Recalculating is sometimes faster than transferring gigabytes over PCIe.

vLLM defaults to Swapping but falls back to Recomputation under extreme pressure.

---

#### Section 4: SGLang: Unifying Preemption and Eviction under Tree Management

SGLang unifies cache sharing, eviction, and preemption within the Radix Tree:
1. **Preemption is Dereferencing**: Paused requests simply drop their node reference count to zero. No data moves or erases.
2. **Best-effort Retention**: Zero-ref nodes stay in VRAM as inactive cache. If they survive LRU eviction, resuming hits the cache directly.
3. **Simplicity**: Avoids state machines. If VRAM is full, LRU prunes leaf nodes, and evicted requests fall back to Recompute.

> [!NOTE]
> **SGLang Evolution: From Recompute to Tiered Offload**
> This describes early SGLang's core logic (eviction implies discard and recompute). Recently, SGLang introduced hierarchical cache offloading (HiCache), offloading inactive Radix Tree nodes to CPU or SSD to combine tree management with multi-tier storage capacity. See [LMSYS Blog: SGLang HiCache](https://www.lmsys.org/blog/2025-09-10-sglang-hicache/).

---

### Chapter 15: Trading "Idle Compute" for "Ultimate Latency": Speculative Decoding

While PagedAttention solves VRAM fragmentation and continuous batching eliminates padding, the auto-regressive Decode phase still faces a **Memory Bandwidth Bottleneck (Memory-Bound)**.

As discussed in Chapter 8, generating one token requires moving hundreds of gigabytes of weights to compute cores. GPU compute power sits idle waiting for data.

**Speculative Decoding** leverages idle compute by trading extra computation for reduced latency. It increases arithmetic intensity per memory access, drastically reducing latency at low concurrency. However, at high concurrency, it may compete for compute resources and reduce overall throughput.

#### Section 1: Analogy: The Professor and the Assistant

**Analogy: Professor and Assistant**
* **Target Model**: The wise but slow professor (large model).
* **Draft Model**: The fast but less knowledgeable assistant (small model).

Process:
1. **Drafting**: The assistant quickly generates $K$ tokens (e.g., 5).
2. **Verification**: The large model processes these 5 tokens in parallel (like Prefill) to check them.
3. **Acceptance and Correction**: If the professor accepts the first 3 but rejects the 4th, they correct the 4th and discard the 5th.

One forward pass yields multiple verified tokens, saving time.

#### Section 2: Economics: Arithmetic Intensity Trade-offs

Total computation (FLOPs) increases, but it is highly profitable.

As discussed in [Chapter 8](part2_bottlenecks.md#chapter-8-core-asymmetry-prefill-vs-decode), Decode arithmetic intensity is extremely low, leaving GPU cores starved.

Speculative decoding lets the model **process multiple tokens at once**. Reading weights to verify 5 tokens takes the same time as reading them for 1 token. We load the weights once to do 5 jobs.

This boosts arithmetic intensity, utilizing idle compute to reduce latency.

#### Section 3: Evolution: From Dual Models to External Heads

**1. Dual-Model Scheme**
* **Principle**: Progenitor method. Loads a massive **Target Model** and a small **Draft Model** into VRAM.
* **Workflow**: The small model generates $K$ tokens autoregressively; the large model verifies them in parallel.
* **Pain Points**: Heavy engineering, managing two sets of KV Cache, and small models occupy VRAM.

**2. Medusa**
* **Principle**: Attaches parallel, lightweight external heads to the large model's final layer instead of using a small model.
* **Mechanism**: Head 1 predicts $t+1$; Head 2 guesses $t+2$ without knowing $t+1$; Head 3 guesses $t+3$ without knowing $t+2$, all in parallel.
* **Limitations**: Non-autoregressive "blind guessing" causes accuracy to drop rapidly with step count.

**3. Eagle**
* **Principle**: Introduces a small single-layer Transformer at the hidden state level.
* **Mechanism**: Combines hidden states with predicted tokens to autoregressively derive the next hidden states.
* **Pros & Cons**:
  * **Vs. Medusa**: Maintains higher accuracy due to its autoregressive nature.
  * **Vs. Target Model**: Significantly faster with minimal overhead, though it may still make incorrect guesses.

> [!NOTE]
> **Indexing and Implementation Details**
> 1. **Who trains them?**: Usually open-source communities or enterprises for specific scenarios (e.g., code generation). Since hit rates depend on scenario-specific vocabulary, customized heads outperform general ones.
> 2. **Why not native?**: Base model creators focus on core capability; speculative decoding is a system-level optimization best left decoupled.
> 3. **Where implemented?**: Scheduling, communication, and **tree-based KV Cache rollback** are implemented in **Inference Frameworks (like vLLM or SGLang)**.

#### Section 4: Production: Tree Attention and Dynamic Trade-offs

**1. Tree Attention**
To improve hit rates, methods offer Top-K candidates, creating a **Draft Tree**. The framework feeds this tree to the large model at once. The large model uses Tree Attention to find the correct path, and the framework **prunes (rolls back)** other branches.

Although the target model guarantees that final output accuracy is unaffected, tree-based pruning and KV Cache rollback on the GPU are not free. If the hit rate is too low, the system frequently loops through "generate-verify-discard-rollback". The overhead of VRAM pointer operations and management can actually slow down inference, requiring systems to dynamically adjust or automatically turn off speculative decoding based on concurrency and hit rates.

**2. Dynamic Trade-offs**
Speculative decoding is used selectively:
* **Low Concurrency / Low Latency**: Turn it on to use idle compute for faster speeds.
* **High Concurrency / High Throughput**: Turn it off to avoid wasting compute that could be used for more batches.

Part Three covered single-node optimizations like GQA, PagedAttention, Continuous Batching, and Speculative Decoding, pushing single GPUs to their limits.

When models scale to trillions of parameters or million-token contexts, single machines fail. **Part Four** covers distributed inference and cluster scheduling.

---
