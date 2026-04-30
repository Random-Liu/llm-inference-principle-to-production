# Part 4: Distributed Chapter — The Concerto Across Single Nodes: Parallel Strategies and High-Speed Interconnects

## Table of Contents
- [Chapter 16: Slicing the Giant: Tensor, Pipeline, and Context Parallelism](#chapter-16-slicing-the-giant-tensor-pipeline-and-context-parallelism)
  - [Section 1: The Necessity of Multiple Machines: The Giant That Doesn't Fit](#section-1-the-necessity-of-multiple-machines-the-giant-that-doesnt-fit)
  - [Section 2: TP and PP: Vertical and Horizontal Slicing](#section-2-tp-and-pp-vertical-and-horizontal-slicing)
  - [Section 3: Automatic Distribution: Distributed Decoupling of Compute and Memory](#section-3-automatic-distribution-distributed-decoupling-of-compute-and-memory)
  - [Section 5: Breaking the Sequence Wall: Context Parallelism](#section-5-breaking-the-sequence-wall-context-parallelism)
  - [Section 6: Hybrid Parallelism: The 3D Concerto of TP, PP, and CP](#section-6-hybrid-parallelism-the-3d-concerto-of-tp-pp-and-cp)
- [Chapter 17: From Brute Force to Precision: Expert Parallelism (EP)](#chapter-17-from-brute-force-to-precision-expert-parallelism-ep)
  - [Section 1: At the Dead End of Deduction: When We Try to Solve MoE with TP and PP](#section-1-at-the-dead-end-of-deduction-when-we-try-to-solve-moe-with-tp-and-pp)
  - [Section 2: The Decision at the Dead End: Cross-host TP vs. Expert Parallelism (EP)](#section-2-the-decision-at-the-dead-end-cross-host-tp-vs-expert-parallelism-ep)
  - [Section 3: The Golden Duo: DP Attention + EP MoE](#section-3-the-golden-duo-dp-attention--ep-moe)
- [Chapter 18: The Perfect Division of Labor: Disaggregated Serving](#chapter-18-the-perfect-division-of-labor-disaggregated-serving)
  - [Section 1: Irreconcilable Contradiction: Hardware Mismatch and Management Dilemma](#section-1-irreconcilable-contradiction-hardware-mismatch-and-management-dilemma)
  - [Section 2: Physical Separation: Decoupling Hardware, Simplifying Management](#section-2-physical-separation-decoupling-hardware-simplifying-management)
  - [Section 3: Typical Workflow of Disaggregated Serving](#section-3-typical-workflow-of-disaggregated-serving)
- [Chapter 19: The Omniscient Traffic Police: Content-Aware Routing](#chapter-19-the-omniscient-traffic-police-content-aware-routing)
  - [Section 1: AI Gateway: The Traffic Police That Knows the Business](#section-1-ai-gateway-the-traffic-police-that-knows-the-business)
  - [Section 2: Cache-Aware Routing and Dynamic Replication](#section-2-cache-aware-routing-and-dynamic-replication)
  - [Section 3: SGLang's System-Level Implementation: Gateway Approximate Tree and Shared L3](#section-3-sglangs-system-level-implementation-gateway-approximate-tree-and-shared-l3)
- [Chapter 20: Opening the Meridians: Network Communication and High-Speed Interconnects in Large Model Inference](#chapter-20-opening-the-meridians-network-communication-and-high-speed-interconnects-in-large-model-inference)
  - [Section 1: The Bloodline within a Single Machine: PCIe, NVLink, and NVSwitch](#section-1-the-bloodline-within-a-single-machine-pcie-nvlink-and-nvswitch)
  - [Section 2: The Cross-Machine Bridge: RDMA and Its Implementations](#section-2-the-cross-machine-bridge-rdma-and-its-implementations)
  - [Section 3: Parallel Modes, Data Volumes, and Metric Impacts](#section-3-parallel-modes-data-volumes-and-metric-impacts)

This part zooms out to cluster-level architecture and how top tech companies serve billions of requests.

## Chapter 16: Slicing the Giant: Tensor, Pipeline, and Context Parallelism

When the model's parameter count soars from 7B (7 billion) to 400B (400 billion) or even larger, the physical limits of a single graphics card and even a single server are completely shattered. We must slice this "giant" into pieces and distribute them across multiple machines for collaborative inference. This chapter will introduce the core technologies of distributed inference.

### Section 1: Multi-Node Necessity: The VRAM Ceiling

We require distributed inference because **large models exceed single-card VRAM capacities**.

Taking a 400B parameter model as an example:
*   **Weight Footprint**: Under half-precision (FP16), model weights alone consume **800 GB** of VRAM.
*   **Hardware Limits**: A classic **NVIDIA H100** typically provides $80$ GB of VRAM. You need at least $10$ H100 GPUs simply to hold the model (without factoring in KV Cache).
*   **Physical Boundaries**: A standard AI server holds a maximum of $8$ GPUs ($640$ GB VRAM). Deploying this model demands spanning at least $2$ physical nodes.

While next-gen architectures like Blackwell (e.g., B200 with 192 GB VRAM) and Rubin reduce required GPU counts, the physical VRAM ceiling persists for three reasons:

1.  **Parameter Explosion**: Model parameter sizes scale exponentially—from GPT-3's 175B in 2020 to Llama 4's projected 2T. Next-gen flagship models target 3T to 5T.
2.  **Long Contexts and KV Cache**: Modern workloads (Agents, RAG) drive context windows from thousands to millions of tokens. The resulting KV Cache can match or surpass the model weights themselves under high concurrency.
3.  **Legacy Hardware ROI**: Production environments heavily rely on existing investments like H100 clusters. Upgrading hardware every year is economically unfeasible; clustering existing nodes via high-speed networks is the standard solution.

Distributed inference is an absolute necessity.

---

### Section 2: TP vs. PP: Vertical and Horizontal Slicing

Engineers rely on two classic slicing strategies to split workloads across GPUs:

**1. Tensor Parallelism (TP) — Vertical Slicing**
*   **Approach**: Splice a single large matrix multiplication (Tensor) vertically or horizontally across different GPUs. For example, GPU 1 calculates the left half, GPU 2 calculates the right half, and they combine results via All-Reduce over high-speed interconnects (e.g., NVLink).
*   **Characteristics**: Operates **within a network layer**. Demands extreme bandwidth and frequent communication, confining it to **intra-node** deployment.

**2. Pipeline Parallelism (PP) — Horizontal Slicing**
*   **Approach**: Split model layers across nodes. If a model has 80 layers, Node A handles layers 1–40, and Node B handles layers 41–80. After Node A finishes computing the hidden states, it sends them across the network to Node B.
*   **Characteristics**: Operates **between layers**. Involves lower communication frequencies, making it ideal for **cross-node** (multi-host) deployments.

We combine TP and PP (e.g., 8-way TP + 2-way PP) to elegantly distribute ultra-large models across 16 or more GPUs.

---

### Section 3: Distributed Decoupling: Splitting Compute and Memory

Slicing models across GPUs naturally decouples **Compute** and **Memory**:

**1. Slicing Compute**
*   **Tensor Parallelism (TP)**: Distributes **layer-internal computation**. A massive matrix multiplication splits into smaller chunks, reducing the computational load on individual GPUs.
*   **Pipeline Parallelism (PP)**: Distributes **inter-layer computation**. Stages act as a sequential relay across different nodes over time.

**2. Slicing Memory**
Distributed environments divide VRAM across these components:
1.  **Model Weights**: TP splits layer weights among node GPUs, while PP divides the layers across different physical nodes.
2.  **KV Cache**: Under TP, sliced attention heads cause each GPU to hold only the K and V vectors for its assigned heads. Under PP, KV Cache binds to specific layers, residing only on the node hosting those layers.
3.  **Activations and Buffers**: Intermediate feature maps scatter across host GPUs. TP frequently synchronizes these activations, whereas PP transmits activations across physical boundaries.

---

### Section 4: Impact: How TP and PP Affect Core Metrics

We evaluate how TP and PP impact core inference metrics (TTFT, TBT, Throughput). We divide time into three distinct phases:
*   **Queueing Time**: Time spent waiting in scheduling queues.
*   **Execution TTFT**: Pure compute time required to generate the first Token.
*   **User-Facing TTFT**: Perceived latency by users (Queueing Time + Execution TTFT + Network Latency).

**1. Tensor Parallelism (TP): Lane Expansion**
*   **Execution TTFT**: **Significantly Reduced**. Amortizing large matrix calculations accelerates Prefill speed (assuming All-Reduce bandwidth isn't a bottleneck).
*   **TBT / TPS**: **Slightly Reduced (TPS Improved)**. Matrix operations accelerate during Decode, but cross-card All-Reduce overhead lowers the speedup efficiency.
*   **Queueing Time and Throughput**: **Dual Optimization with Trade-offs**.
    1.  **Faster Serving**: Requests complete quicker, reducing wait times for subsequent requests.
    2.  **Expanded Concurrency**: Larger aggregated VRAM supports larger Batch Sizes, absorbing queued requests directly.
    *   *Trade-off*: Applying TP to models that natively fit on a single GPU introduces unnecessary communication overhead, hurting total throughput.

**2. Pipeline Parallelism (PP): Relay and Throughput**
*   **Execution TTFT**: **Slightly Increased**. Requests flow through multiple physical nodes, introducing cross-host latency and pipeline warm-up costs.
*   **TBT / TPS**: **Minimal Improvement**. Horizontal slicing does not accelerate the compute of an individual Token.
*   **Queueing Time and Throughput**: **Significant Improvements**.
    *   **Pipelining Reduces Wait**: Node A starts processing Request B's Prefill as soon as Request A moves onto Node B.
    *   **Maximized Throughput**: Parallel execution across distinct stages optimizes GPU utilization, absorbing high traffic volume and cutting average queueing times.

---

### Section 5: Context Parallelism: Shattering the Sequence Ceiling

**1. The Bottleneck**
Long context scenarios shift bottlenecks from model weights to **linearly scaling KV Caches** and **quadratically expanding attention compute loads**.
*   Even with model weights distributed via 8-way TP, single-card VRAM cannot hold KV Caches for millions of tokens.
*   Standard TP only shards the Hidden Dimension, failing to mitigate sequence length expansion.
*   CP shatters this ceiling by splitting context sequences across physical nodes, enabling parallel attention computation to dramatically cut processing times.

**3D Slicing Dimensions in Parallelism (L × N × d):**

![Parallelism Dimensions](../images/parallelism_dimensions.svg)

**2. The Mechanism**
CP shards sequences along the **Sequence Dimension**:
1.  **Sequence Chunking**: A multi-million token sequence splits into $N$ chunks across $N$ GPUs. Each GPU stores and processes only its segment of the KV Cache.
2.  **Ring Attention**: Causal Attention requires tokens to interact with all preceding elements. GPUs arrange into a ring topology, calculating local Attention while asynchronously transmitting KV Cache chunks to the next adjacent node. This resolves VRAM limitations without aggregating data centrally.

> [!NOTE]
> **Deep Dive: Ring Attention Coordination and Load Balancing**
>
> Ring Attention goes beyond data chunking, confronting significant communication and load balancing challenges.
>
> **1. Asynchronous Relay Coordination**
> Suppose three GPUs split a sequence. GPU 1 holds $(Q_1, K_1, V_1)$, GPU 2 holds $(Q_2, K_2, V_2)$, and GPU 3 holds $(Q_3, K_3, V_3)$. In Causal Attention mode, the relay proceeds as follows:
> *   **Step 1**: All GPUs calculate Attention for local data while initiating asynchronous KV chunk transfers (GPU 1 to 2, 2 to 3, and 3 to 1).
> *   **Step 2**: Upon receiving upstream KV chunks, GPUs calculate Attention for the new data. In Causal mode, invalid computations (e.g., GPU 1 examining future data $KV_3$) are masked.
> *   **Step 3**: Subsequent relays ensure every GPU completes compute against all required historical contexts. Online Softmax mechanisms dynamically update Softmax numerators and denominators.
>
> **Ring Attention Flowchart:**
> ```mermaid
> sequenceDiagram
>     participant M1 as "📟 GPU 1 (Holds Q1, KV1)"
>     participant M2 as "📟 GPU 2 (Holds Q2, KV2)"
>     participant M3 as "📟 GPU 3 (Holds Q3, KV3)"
> 
>     Note over M1, M3: Step 1: Local Compute & Transfer
>     par Asynchronous Communication
>         M1->>M2: Send KV1
>     and
>         M2->>M3: Send KV2
>     and
>         M3->>M1: Send KV3
>     end
>     Note over M1: Compute Q1 * KV1
>     Note over M2: Compute Q2 * KV2
>     Note over M3: Compute Q3 * KV3
> 
>     Note over M1, M3: Step 2: Remote Compute & Next Relay
>     par Asynchronous Communication
>         M1->>M2: Forward KV3
>     and
>         M2->>M3: Forward KV1
>     and
>         M3->>M1: Forward KV2
>     end
>     Note over M1: Compute Q1 * KV3 (Masked)
>     Note over M2: Compute Q2 * KV1
>     Note over M3: Compute Q3 * KV2
> 
>     Note over M1, M3: Step 3: Final Compute
>     Note over M1: Compute Q1 * KV2 (Masked)
>     Note over M2: Compute Q2 * KV3 (Masked)
>     Note over M3: Compute Q3 * KV1
> ```
>
> **2. Load Imbalance and Zig-zag Slicing**
> In Causal Attention, workloads accumulate into a lower triangular matrix:
> *   GPU 1 handles 1 unit of compute.
> *   GPU 2 handles 2 units of compute.
> *   GPU 3 handles 3 units of compute.
>
> This creates severe idle times for earlier nodes. To fix this:
> *   **Option A (Brute Force)**: Force all nodes to calculate future blocks and filter them via masking. This wastes ~50% of total compute.
> *   **Option B (Zig-zag / Striping)**: Assign non-contiguous chunks (like dealing cards). Given 6 blocks, GPU 1 gets 1 and 6, GPU 2 gets 2 and 5, and GPU 3 gets 3 and 4. Every node handles exactly 7 units of compute, eliminating idle wait times.
>
> **Zig-zag Load Balancing:**
> ```mermaid
> graph TD
>     subgraph "Contiguous Slicing"
>         N1["📟 GPU 1: Chunks [1, 2]"]
>         N2["📟 GPU 2: Chunks [3, 4]"]
>         N3["📟 GPU 3: Chunks [5, 6]"]
>         N1 -->|"Workload: 3"| NW1["Severely Idle"]
>         N2 -->|"Workload: 7"| NW2["Moderate Load"]
>         N3 -->|"Workload: 11"| NW3["Heavy Load"]
>     end
> 
>     subgraph "Zig-zag Slicing"
>         Z1["📟 GPU 1: Chunks [1, 6]"]
>         Z2["📟 GPU 2: Chunks [2, 5]"]
>         Z3["📟 GPU 3: Chunks [3, 4]"]
>         Z1 -->|"Workload: 7"| ZW1["Perfectly Balanced"]
>         Z2 -->|"Workload: 7"| ZW2["Perfectly Balanced"]
>         Z3 -->|"Workload: 7"| ZW3["Perfectly Balanced"]
>     end
> ```

**3. Core Impact**
1.  **Execution TTFT**: **Dramatically optimizes long-context TTFT**. Parallelizing sequence compute drastically cuts Prefill durations for massive prompts.
2.  **TBT / TPS**: **Minimal impact**. Decoders process tokens sequentially without matrix scaling against past contexts, yielding negligible latency reductions.
3.  **Throughput & Cost**: **Trades network overhead for operational feasibility**. CP introduces high communication costs but acts as the only viable solution for ultra-long text workloads.

> [!IMPORTANT]
> **Cross-Node Context Parallelism**
> When contexts scale to 1M tokens or more, single-node VRAM totalities fail. Context Parallelism must bridge physical racks using InfiniBand or RoCE. This requires advanced compute-communication overlap techniques due to slower cross-node latency.

---

### Section 6: Hybrid Parallelism: 3D Concerto of TP, PP, and CP

Slicing massive 400B parameter models over 1M token contexts demands combining Tensor Parallelism (TP), Pipeline Parallelism (PP), and Context Parallelism (CP).

**1. Physical Architecture**
Frontier setups rely on 3D parallelism featuring clear physical layering:
*   **Intra-Node (NVLink):打满 TP**. Standard servers set $\text{TP}=8$. High-speed NVSwitch interconnections bind these 8 cards into a cohesive physical unit holding weight slices.
*   **Inter-Node (Vertical): PP**. Horizontal stages verticalize model layers across nodes. An 80-layer model splits into two 40-layer stages, minimizing cross-host data transfer.
*   **Inter-Node (Horizontal): CP**. Context parallelism resolves KV Cache overflows across the same PP stage. Million-token windows split into chunks across node groups replicating identical layer weights.

**3D Parallelism Topology (TP + PP + CP):**

![3D Topology](../images/hybrid_parallelism_topology.svg)

Within each **Transformer Block**:
*   **Attention Layers**: Require inter-node CP communication (Ring Attention) to exchange KV Caches.
*   **FFN Layers**: Operate independently. Nodes calculate local tokens without requiring cross-sequence communication.

---

## Chapter 17: From Brute Force to Precision: Expert Parallelism (EP)

Chapter 16 discussed slicing dense models across GPUs. TP, PP, and CP all share a common trait: every token activates all parameters.

As models reach trillions of parameters, this brute-force approach hits physical limits. [Mixture of Experts (MoE)](./part1_principles.md#section-7-mixture-of-experts-sparse-activation) solves this by activating only a fraction of experts per token (e.g., 2 out of 256). This **Sparse Activation** property drives a new parallelism dimension: **Expert Parallelism (EP)**.

### Section 1: At the Dead End of Deduction: When We Try to Solve MoE with TP and PP

MoE decouples model capacity from compute cost. You can build massive models with vast knowledge while consuming minimal compute per inference by activating only a fraction of experts.

Facing MoE models with hundreds of gigabytes or terabytes of total expert weights (e.g., DeepSeek V3 with $671\text{B}$ parameters), our most natural instinct is to use the two weapons we already have— **Tensor Parallelism (TP)** and **Pipeline Parallelism (PP)** —to **shard weights** and address the VRAM capacity bottleneck.

1.  **Step 1: Small Models Use Single-Node TP**
    Small MoE models (e.g., Mixtral 8x7B) fit entirely on 8-GPU servers via **intra-node Tensor Parallelism ($\text{TP}=8$)**. Backed by NVLink's $900\text{GB/s}$ bandwidth, all GPUs compute all experts. While sacrificing theoretical sparsity, it sidesteps cross-node latency.
2.  **Step 2: Larger Models Require Layer-wise PP**
    When parameters grow beyond single-node capacity, we introduce **Pipeline Parallelism (PP)**. Slicing models horizontally across physical nodes limits communication frequency and bandwidth demands.
3.  **Step 3: Dead Ends via Pipeline Bubbles**
    Pipeline Parallelism cannot scale infinitely. Deep PP stages introduce pipeline bubbles that stretch Time Between Tokens (TBT) beyond acceptable user limits. Production environments strict limit PP depth to 4 or 8 stages.

**The Engineering Deadlock:**
With constrained PP depths, the layer groups assigned to a physical node still hold MoE FFN weights that surpass the node's VRAM capacity. Experts within identical layers must shard **across host boundaries**.

---

### Section 2: Dead Ends: Cross-Host TP vs. Expert Parallelism (EP)

Engineers face two divergent architectural paradigms when slicing single layers across host boundaries:

1.  **Option A: Cross-Host Tensor Parallelism (Cross-Host TP)**
    *   **Approach**: Shard layer experts vertically or horizontally across physical nodes.
    *   **Metaphor**: **"Everyone cuts the same tree"**—sharded weights, stationary tokens.
    *   **Trade-offs**:
        *   **Degraded Compute Efficiency**: Slicing matrices too finely eliminates GEMM operations, forcing GPUs into inefficient GEMV calculations.
        *   **Network Paralysis**: Layers trigger massive, cross-node global All-Reduce events, choking 50-100GB/s inter-host (IB/RoCE) networks.
        *   **Synchronous Bottlenecks**: Rigid All-Reduce syncs stall the entire cluster's CUDA cores for any minor network jitter.
2.  **Option B: Expert Parallelism (EP)**
    *   **Approach**: **Stationary weights, moving tokens**. We preserve the integrity of individual experts (e.g., Expert A resides on Machine 1, Expert B on Machine 2). Tokens are routed via All-to-All networks to the host GPU, computed locally, and routed back.
    *   **Metaphor**: **"Divide trees among people"**—stationary weights, moving tokens.
    *   **Revolutionary Benefits**:
        *   **Algorithmic Isomorphism**: Keeps experts intact, letting network routing handle dispatch.
        *   **Maximized Hardware Utilization**: EP bundles tokens bound for the same expert onto a single GPU. This triggers full GEMM operations and saturates Tensor Cores.
        *   **Asynchronous Overlap**: All-to-All communication acts asynchronously. While GPUs compute local tokens, the NIC shunts cross-node tokens in the background, hiding communication latency behind computation time.

**Comparing Cross-Host TP and Expert Parallelism (EP):**

| Dimension | Cross-Host Tensor Parallelism (TP) | Expert Parallelism (EP) |
| :--- | :--- | :--- |
| **Philosophy** | Stationary Tokens, Sharded Weights | Stationary Weights, Routed Tokens |
| **Matrix Compute** | Inefficient GEMV | Highly efficient GEMM |
| **Comm Pattern** | All-Reduce (Synchronous) | All-to-All (Asynchronous) |
| **Comm Frequency** | Scaled by active experts | Fixed at 2 events per layer |
| **Overlap** | Blocking Sync | Background Asynchronous Shunting |

> [!NOTE]
> **Quantifying EP vs. TP Communication Volume**
> Intuitively, EP's All-to-All volume seems immense. However, mathematical analysis proves otherwise. Assuming $M$ nodes and $K$ active experts per token, unoptimized theoretical EP yields merely $1/M$ of TP's cross-node traffic. Even when Operator Fusion scales down TP volume by $K$, EP communication traffic remains just $K/M$ of TP.

---

### Section 3: The Golden Duo: DP Attention + EP MoE

Production serving (e.g., DeepSeek V3/R1) deploys a hybrid topology: **DP (Data Parallelism) for Attention layers, and EP (Expert Parallelism) for FFN (MoE) layers**.

This division leverages architectural **heterogeneity**:
1.  **Attention Layers (Dense & Lightweight)**: Modern mechanisms like GQA or MLA compress weights enough to replicate across all GPUs. Attention layers apply **DP**—nodes process local requests without communication, avoiding All-Reduce overhead.
2.  **FFN Layers (Sparse & Heavy)**: Bulky MoE expert weights shard across nodes via **EP**, routing tokens over the network.

**DP Attention + EP MoE Architecture:**

![DP Attention + EP MoE Topology](../images/dp_attention_ep_moe.svg)

> [!NOTE]
> **Trade-offs between DP and CP for Context Volumes**
> DP and CP both shard **input data** rather than weights, but target different dimensions:
> *   **DP Attention**: Slices the **Batch / Request dimension**, targeting maximum throughput.
> *   **CP Attention**: Slices the **Sequence dimension** for a single request, relying on Ring Attention syncs.
>
> Production deployments profile these trade-offs based on text lengths:
> *   **Short Text Limits**: Naive CP on short sequences shards compute too finely, choking GPUs on frequent cross-node ring syncs.
> *   **Long Text Limits**: Conversely, heavy contexts will OOM individual cards unless CP is engaged.
>
> **Engineering Practices:**
> *   **Standardization**: Deploy subtle CP (e.g., $\text{CP}=2$ or 4). Short context throughput drops by 10–20%, but guarantees universal compatibility.
> *   **Segmentation**: Use gateway routing to isolate workloads. Short-form requests hit a `DP Attention + EP MoE` topology for zero-communication throughput, while long-form contexts target a dedicated `CP Attention + EP MoE` cluster.

---

## Chapter 18: The Perfect Division of Labor: Disaggregated Serving

What is **Disaggregated Serving**? Put simply, it is an architecture that completely strips the **Prefill** phase and **Decode** phase of large model inference and runs them on physical clusters with different hardware configurations.

In Part Three, we introduced **Continuous Batching** and **Chunked Prefill**. They achieved the "perfect carpooling" of Prefill and Decode at the single-machine level, greatly squeezing the performance of a single graphics card. You might ask: Since the single-machine problem has been solved, why go through the trouble of doing disaggregated serving?

The answer is: single-machine optimization is only a **"tactical-level"** limit squeeze. Attempting to perfectly balance Prefill and Decode within a single machine is not only constrained by the physical limits of **hardware mismatch**, but also brings **extremely high complexity in system management and scheduling**. When the service scale reaches an industrial magnitude, the "perfection" within a single machine becomes a macroscopic "burden". This chapter will unveil **Disaggregated Serving** and see how it simultaneously solves hardware mismatch and greatly simplifies resource management.

### Section 1: Irreconcilable Contradiction: Hardware Mismatch and Management Dilemma

We mentioned in [Chapter 8: Core Asymmetry: Prefill vs. Decode](../parts/part2_bottlenecks.md#chapter-8-core-asymmetry-prefill-vs-decode) that Prefill and Decode have completely opposite hardware requirements:
* **Prefill**: Processes massive inputs, requiring extremely high **Compute (FLOPs)**, but relatively small VRAM capacity requirements.
*   **Decode**: Processes generation sequentially, requiring minimal compute but immense **memory bandwidth** and **capacity** to frequently fetch KV Caches.

Running a unified (mixed) deployment architecture leads to dual inefficiencies:
1.  **Mismatched Hardware ROI**: Deploying high-compute GPUs (e.g., B200) to Decode phases wastes Tensor Cores on I/O wait times.
2.  **Micro-Scheduling Complexity**: To force balance on single GPUs, engineers stack complex algorithms like continuous batching or chunked prefilling. Minor imbalances result in immediate spikes in TTFT or TBT, turning resource management into a tightrope walk.

> [!NOTE]
> **Why Chunked Prefill Fails as a Cure-All**
> Sharding Prefill workloads into Decode idle phases still requires purchasing high-bandwidth GPUs. Under high concurrency or long-text workloads, GPUs remain memory-bound. Single-card resource contention inevitably triggers performance jitters.

---

### Section 2: Disaggregated Serving: Decoupling Hardware

Frontier deployments isolate workloads into **Disaggregated Serving** clusters:
1.  **Prefill Cluster**: Holds high-compute GPUs optimized for prompt processing, generating KV Caches at maximum speed.
2.  **Decode Cluster**: Houses memory-optimized GPUs equipped with high HBM bandwidth to store caches and stream sequential tokens.

**Key Benefits:**
1.  **Optimized ROI**: Teams purchase specialized hardware for each cluster, targeting **TTFT on Prefill** and **TBT stability on Decode**.
2.  **Simplified Capacity Planning**: Isolating stages shifts complex micro-scheduling into straightforward capacity scaling:
    *   **RAG / Document Q&A**: Heavy Prefill, light Decode. Engineers expand only the Prefill cluster.
    *   **Agentic / CoT Workloads**: Short prompts followed by massive internal monologues. Teams directionally scale the Decode cluster without paying for unutilized Prefill compute.

---

### Section 3: The Serving Workflow

Disaggregated Serving functions as a point-to-point relay where the **AI Gateway** matches nodes:

1.  **Matchmaking**: Requests hit the AI Gateway. The gateway pairs a Prefill node and a Decode node, assigning a globally unique session ID.
2.  **Parallel Dispatch**: The gateway concurrently shunts connection data to both target nodes.
3.  **Pre-Allocation**:
    *   The Decode node allocates memory in its local KV pool.
    *   It sends target memory addresses to the Prefill node.
4.  **Direct PUSH**:
    *   The Prefill node processes the prompt.
    *   Using **RDMA** protocols (e.g., Mooncake), it directly pushes the KV Cache into the Decode node's VRAM.
5.  **Decoupled Generation**:
    *   The Decode node verifies KV Cache delivery.
    *   It begins sequential generation, streaming tokens back to the user.

```mermaid
sequenceDiagram
    autonumber
    actor User as 🧑 User
    participant Gateway as 🚦 AI Gateway / Scheduler
    participant Prefill as 🚀 Prefill Node<br/>(Compute-Bound)
    participant Decode as 💾 Decode Node<br/>(Memory/Bandwidth-Bound)

    User->>Gateway: 1. Send Prompt Request
    Note over Gateway: Matchmaker role: Select P/D Pair<br/>Assign Session ID
    par Concurrent Dispatch
        Gateway->>Prefill: 2. Shunt Request (Includes Decode VRAM Address)
    and
        Gateway->>Decode: 2. Shunt Request (Includes Session ID)
    end
    
    Note over Decode: 3. Pre-allocate KV VRAM Space
    Decode->>Prefill: 4. Send Memory Address
    
    Note over Prefill: 5. Process Prompt<br/>Generate KV Cache
    
    Note over Prefill, Decode: PUSH KV Cache via RDMA
    Prefill->>Decode: 6. Direct push into Decode VRAM
    
    Prefill->>Gateway: 7. Mark Prefill Complete
    
    Note over Decode: 8. Trigger Sequential Generation
    
    loop Word-by-Word Generation
        Decode->>User: 9. Stream Tokens
    end
    Decode->>Gateway: 10. Release Session Resources
```

Decentralizing data transfers ensures the gateway doesn't become a network bottleneck, turning single-node contention into cluster-level pipeline execution.

---

## Chapter 19: The Omniscient Traffic Police: Content-Aware Routing

In Chapter 18, we split the cluster into a Prefill pool and a Decode pool. So, when massive HTTP requests pour in, who decides which request goes to which machine? This chapter will introduce the "traffic police" in the large model cluster — **Content-Aware Routing**.

### Section 1: AI Gateway: The Traffic Police That Knows the Business

Traditional load balancers (like Nginx or F5) only care about basic physical metrics such as network traffic, concurrent connections, and server CPU/memory utilization. To them, an HTTP request is just a bunch of meaningless bytes.

But in an LLM inference cluster, this "blind" routing leads to disaster. AI Gateways act as business-aware traffic police:
*   **Request Inspection**: Before requests hit GPUs, gateways parse prompt lengths (token counts) and categorize workload types.
*   **Intelligent Shunting**: Gateways combine routing strategies to Prefill and Decode nodes based on the **request profile** (input length vs. expected generation length).

**AI Gateway Routing Policies:**

| Request Profile | Features (Input/Output) | Prefill Routing Policy | Decode Routing Policy |
| :--- | :--- | :--- | :--- |
| **Daily Chat** | Short / Short | **Greedy**: Routes to the shortest queue for rapid TTFT. | **Round-Robin**: Assigns to any low-load node. |
| **RAG** | **Long** / Short | **Compute-First**: Dispatches to underloaded, high-compute nodes. | **Capacity-First**: Mandates nodes with large remaining VRAM for KV Caches. |
| **Long Text Gen** | Short / **Long** | **Fast-Pass**: Dispatches to standard idle nodes. | **Bandwidth-First**: Targets nodes with few active batches and high VRAM bandwidth (secures TBT). |
| **Complex Dialog** | **Long** / **Long** | **Resource Tilt**: Targets top-tier idle nodes; triggers CP when required. | **Strict Double Bound**: Requires large capacity (initial KV) and high bandwidth (long-term streaming). |

> [!NOTE]
> **Preventing HOL (Head-of-Line) Blocking**
> Gateways avoid placing short prompts behind long workloads. If Prefill nodes are saturated, gateways triage short requests via priority queues or fast lanes.

---

### Section 2: Cache-Aware Routing and Dynamic Replication

**Cache-Aware Routing** acts as an AI Gateway's strongest optimization:

**1. The Core Logic**
Similar to **RadixAttention**, if requests share system prompts, RAG documents, or history, nodes locally cache the prefix KV Caches. Blind round-robin routing scatters these requests, forcing redundant prefill calculations and inflating TTFT. Cache-Aware Routing binds identical requests to the node housing the cache.

**2. Operational Flow**
*   **Prefix Matching**: The gateway tracks text prefixes against memory nodes. Incoming requests hit the node yielding the highest prefix match to reuse KV Caches.
*   **Dynamic Replication**: Hotspots (e.g., viral system prompts) can overload single nodes. Upon detecting load imbalances, gateways instruct idle nodes to pull identical prefixes and stand up replicas.

---

### Section 3: System Implementation via SGLang

**SGLang** combines soft gateway routing with hierarchical backend caching:

**1. Tokenizer-Free Approximate Radix Trees**
Gateways maintain a Rust-based Radix Tree mapping **raw strings** instead of Token IDs, bypassing vocabulary tokenization. Under extreme traffic hotspots, gateways automatically toggle from "Match Rate" to "Shortest Queue" to redistribute workloads.

**2. Tiered Shared Storage (HiCache)**
When a new node inherits split traffic for a cached prefix:
*   It fetches KV Caches directly from **L3 Distributed Storage** (e.g., DeepSeek 3FS) instead of recomputing.
*   Once loaded, the node becomes a new replica for the Radix Tree.

> [!NOTE]
> **HiCache vs. Single-Node Offloading**
> Both utilize the "GPU $\rightarrow$ CPU $\rightarrow$ External Storage" pyramid. However, HiCache expands local offloading into **cluster-wide sharing** and structurally couples with prefix trees to actively prefetch data.

---

## Chapter 20: Network Interconnects and Hardware Communication

Distributed inference partitions compute, simultaneously generating communication overheads.

---

### Section 1: Intra-Node Fabrics: PCIe, NVLink, and NVSwitch

Node architectures have undergone significant evolutions:

1.  **PCIe**: The traditional bus (PCIe 5.0 x16 delivers ~64 GB/s). It becomes a severe bottleneck under frequent All-Reduce synchronizations.
2.  **NVLink + NVSwitch**:
    *   **NVLink**: A point-to-point fabric enabling direct GPU-to-GPU VRAM reads/writes (P2P), delivering up to 900 GB/s on H100 GPUs.
    *   **NVSwitch**: Prevents link capacity bottlenecks by routing all 8 GPUs into an internal switch, delivering full-bandwidth full-mesh topology.

---

### Section 2: Cross-Node Fabrics: RDMA and Its Implementations

Standard TCP/IP stacks introduce high latency and CPU overheads when crossing physical hosts.

**1. RDMA: Eliminating Overhead**
RDMA (Remote Direct Memory Access) enables NICs to read/write remote GPU VRAM directly, **bypassing the CPU and kernel**. It drops latency from milliseconds to microseconds, freeing up CPU compute for scheduling and control flow.

**2. InfiniBand vs. RoCE**
*   **InfiniBand (IB)**: A dedicated HPC network. Inherently lossless (credit-based flow control), delivering 400–800Gbps bandwidth at sub-microsecond latencies. Requires specialized NICs and switches.
*   **RoCE**: Runs RDMA over standard Ethernet. Saves on hardware costs but requires complex PFC and ECN configurations to construct a "lossless Ethernet".

**3. NVLink Switch**
NVLink Switch (e.g., NVSwitch in GB200 NVL72) connects GPUs across racks via copper cables, establishing a 72-GPU full-mesh cluster. It alters engineering trade-offs (e.g., expanding TP configurations) and significantly cuts reliance on cross-host Pipeline Parallelism (PP).

---

### Section 3: Parallel Modes, Data Volumes, and Metric Impacts

**Quantitative Distributed Inference Metrics:**

| Mode | Frequency & Scope | Single Transfer | Total Event Volume | Metrics Impacted | Required Network |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tensor Parallelism (TP)** | **Step-Level (High Frequency)**: $2 \times L$ events / step | $O(N \cdot d)$ | $O(L \cdot N \cdot d)$ | **TBT, TTFT** | Intra-Node NVLink |
| **Pipeline Parallelism (PP)** | **Step-Level (Low Frequency)**: $P - 1$ events / step | $O(N \cdot d)$ | $O(P \cdot N \cdot d)$ | **Throughput, TTFT** | Cross-Node IB / RoCE |
| **Context Parallelism (CP)** | **Request-Level (Single Pulse)**: $(M-1) \times L$ events / req | $O(\frac{N}{M} \cdot d)$ | $O(L \cdot N \cdot d)$ | **TTFT** | Intra-Node NVLink / Cross-Node |
| **Expert Parallelism (EP)** | **Step-Level (High Frequency)**: $2 \times L_{\text{MoE}}$ events / step | $O(\frac{K \cdot N}{M} \cdot d)$ | $O(L_{\text{MoE}} \cdot K \cdot N \cdot d)$ | **TBT, Throughput** | Cross-Node IB / RoCE |
| **Disaggregated Serving** | **Request-Level (Single Pulse)**: 1 event / req | $O(L \cdot N \cdot d)$ | $O(L \cdot N \cdot d)$ | **User-Facing TTFT** | Cross-Node IB / RoCE |

> [!NOTE]
> **Parameters**: $L$: Total layers; $L_{\text{MoE}}$: MoE layers; $K$: Active experts per token; $N$: Sequence length; $d$: Hidden dimension; $P$: Pipeline stages ($P \le L$); $M$: Parallel machine count.

**Dimensional Difference (Core Insight):**

*   **Distinguish Time Scale**: TP, PP, and EP are **normalized (step-level)**, bound to forward propagation cycles. Conversely, CP and Disaggregated Serving are **eventized (request-level)**, shielding continuous token generation from network bottlenecks.
*   **Mathematical Reality of Communication Volume**: The $K : 1$ ratio between EP and TP data volumes applies exclusively to "Intra-Node NVLink TP". Slicing via Cross-Host TP amplifies All-Reduce volumes $M$ times, exposing EP's $K / M$ advantage.
*   **Compute-Communication Overlap Capability**: **Tensor Parallelism (TP)** is the only pattern that halts computation for synchronous communication. PP, CP, EP, and Disaggregated Serving hide communication latencies within compute times via pipelining, asynchronous shunting, or event-driven direct pushes.
