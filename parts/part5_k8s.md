# Part Five: Orchestration —— Taming the Supercomputer: Leveraging Kubernetes for AI Compute

## Chapter 20: When "Loose Coupling" Meets "Tight Coupling": The Collision of K8s and LLM Lifecycles

### Section 1: First Principles: Examining Lifecycle Contradictions under Distributed Inference

From first principles, the core essence of Kubernetes (K8s) is: a control plane based on declarative state and eventual consistency, aimed at abstracting heterogeneous infrastructure into a unified resource pool and decoupling computation from state. K8s was originally designed to handle loosely coupled, stateless microservices that can be independently started and stopped.

In contrast, the core essence of large-scale distributed Large Language Model (LLM) inference is: performing large-scale matrix multiplication operations under extremely strict latency and memory (KV Cache) constraints, with high determinism, strong topology dependence, and requiring extremely high-speed inter-process communication (such as NVLink, InfiniBand). In large-scale LLM inference (such as Tensor Parallelism TP, Pipeline Parallelism PP), tasks are essentially tightly coupled, pseudo-stateful (weights and KV cache state), and high-performance computing (HPC) tasks that follow the "All-or-Nothing" (Gang) principle, **which is effectively equivalent to managing a distributed "supercomputer"**.

This fundamental contradiction constitutes the core challenge of orchestrating LLM inference on K8s.

### Section 2: Workload Lifecycle: Core Contradictions Throughout

The workload lifecycle covers the entire process from image pulling, scheduling, execution, auto-scaling to termination. In distributed LLM inference scenarios, the special nature of large models brings unprecedented challenges to this lifecycle chain. These conflicts are also the core propositions to be deeply analyzed in the subsequent chapters of this part:

1.  **Submission & Distribution: Separation of Image and Weights**
    *   **Challenge**: While the inference engine image itself is small, the model weights are extremely huge (tens of GBs to hundreds of GBs). Directly packaging weights causes pull timeouts and violates the principle of decoupling compute and data.
    *   **Evolution Direction**: The industry mainstream has moved towards "separation of image and weights." Weights can be stored as OCI Artifacts but are not pulled as regular container images. OSS Kubernetes introduced features like Image Volumes (OCI Volume) precisely to enable direct mounting of weights in OCI format as volumes. Meanwhile, due to the massive data volume, image and weight distribution need extreme optimization through means like P2P and stream loading (see Chapter 21 for details).

2.  **Scheduling: Topology Awareness and All-or-Nothing**
    *   **Challenge**: The native K8s scheduler is based on scalar counting (e.g., CPU cores, GPU quantity) and cannot understand complex underlying PCIe topology, NUMA architecture, and NVLink interconnect relationships. Meanwhile, distributed inference relies on NCCL communication rings, and missing one card makes the entire group unable to work.
    *   **Evolution Direction**: Scheduling must move towards "topology awareness" to avoid performance avalanches (see Chapter 22 for details), and must support "All-or-Nothing (Gang Scheduling)" batch scheduling to prevent resource deadlocks (see Chapter 23 for details).

3.  **Execution & Scaling: Breathing of the Compute Pool**
    *   **Challenge**: Traditional HPA based on CPU/memory utilization fails completely here (VRAM is often pre-allocated, and compute is bursty). Meanwhile, Pod scaling is limited by the cold start speed of physical machines (Nodes).
    *   **Evolution Direction**: Scaling metrics must shift to engine internal business metrics (such as queue length). Meanwhile, the linkage between Pod scaling and Node scaling needs to be solved, utilizing mechanisms like placeholders (Pause Pods) to hide cold start times (see Chapter 25 for details).

4.  **Lifecycle Management: "All-or-Nothing" Throughout**
    *   **Challenge**: This is not just behavior during failure and termination. In distributed LLM inference, from startup ring creation, health checks, rolling updates to failure recovery, **the entire workload lifecycle requires "All-or-Nothing."** Killing any single Pod reduces the entire group to zombies; updating a single Pod causes version mismatch leading to ring creation deadlocks.
    *   **Evolution Direction**: The traditional paradigm of K8s independently managing Pods must be broken, introducing orchestration primitives that manage "group lifecycles" (such as LeaderWorkerSet) to ensure atomicity throughout the lifecycle (see Chapter 24 for details).

### Section 3: Cluster Lifecycle: Heterogeneous Hardware Bootstrapping and Expensive Graceful Termination

The cluster lifecycle includes infrastructure provisioning, node bootstrapping, component upgrading, and maintenance.

1.  **Provisioning & Bootstrapping**
    *   **Challenge**: The complexity of K8s node initialization rises exponentially. It is extremely dependent on the underlying driver stack of heterogeneous hardware (NVIDIA Driver, CUDA, OFED, etc.), and the compatibility matrix is easily broken. At the network level, SR-IOV or direct mounting of RDMA network cards is required.
    *   **Improvement Direction**: Use IaC (such as NVIDIA GPU Operator) to containerize and automate the installation of drivers and plugins; configure dual-network architecture (Multus CNI), with control flow on standard Ethernet and data flow on high-speed network cards.

2.  **Operations & Upgrading**
    *   **Challenge**: The default graceful termination period of K8s is usually not enough to handle long-context inference tasks, and the cost of evicting inference Pods with long connections and high memory usage is expensive.
    *   **Improvement Direction**: Combine service meshes or smart gateways to stop assigning new requests before upgrading nodes, waiting for existing requests to "drain out"; future forward-looking directions include hot migration of KV Cache state.

---

## Chapter 21: Racing Against Time: Model Distribution and Cold Start Optimization

### Section 1: The Inevitability of Image and Weights Separation

In the Large Language Model (LLM) inference scenario, "packaging model weights into a Docker image" has been widely recognized as an absolute anti-pattern. Because the image pulling mechanism simply cannot bear the concurrent I/O of hundreds of GBs, it will cause K8s nodes to fall into an unavailable state due to pull timeouts or disk explosion.

The mainstream practice in the industry has converged to **"separation of image and weights"**. The container image only contains the inference engine (such as vLLM) and the basic runtime environment, while the model weights are managed as independent static data.

To better support this model, OSS Kubernetes introduced features like **Image Volumes (OCI Volume)**. The original intention of this feature was precisely to better handle model weights stored in the form of OCI images (or Artifacts). By pushing weights to a registry that supports OCI specifications, Pods can directly mount weights as a Volume into the container without pulling the full image, retaining both the version control and distribution capabilities of the image registry and achieving decoupling of compute and data.

### Section 2: Distribution Optimization of Massive Data

When weights are separated from images, the distribution of weights evolves into a classic distributed storage and data orchestration problem. Because model files are extremely huge and trigger a terrifying "thundering herd effect" (a large number of nodes pulling the same model at the exact same second) during elastic scaling, traditional filesystems or Registries are easily crushed.

Mainstream optimization means in the industry include:
1.  **P2P Image and Data Distribution**: Introduce P2P distribution networks like Dragonfly and Kraken to dissolve the pressure on centralized storage into Peer-to-Peer traffic within the local area network, drastically increasing concurrent pull speeds.
2.  **Distributed Caching and Data Orchestration**: Use tools like Fluid with JuiceFS or Alluxio to build distributed caches locally on K8s nodes, achieving data locality and allowing Pods to read weights at speeds close to local disks.

### Section 3: Microscopic Bottlenecks and Optimizations of VRAM Loading

When weights have arrived at the node's local disk through optimization means, the cold start battlefield shifts to the microscopic link of "how to pour hundreds of GBs of data from the hard disk into GPU memory."

Major bottlenecks and optimization directions include:
1.  **Crossing the Physical Limit of PCIe Bus**: Even top-tier PCIe Gen5 takes several seconds to move hundreds of GBs of data. The industry standard is to use the **Safetensors** format combined with the operating system's **mmap (memory mapping)** technology to achieve zero-copy loading, avoiding CPU transit.
2.  **GPUDirect Storage (GDS)**: NVIDIA's black tech that supports data bypassing the CPU and system memory to DMA directly from NVMe solid-state drives to GPU VRAM, completely eating up the PCIe bandwidth.
3.  **Meta Device Initialization**: Using PyTorch's meta device to virtually establish the model topology, avoiding double memory allocation in CPU memory and preventing OOM while shortening initialization time.
4.  **Engine Warmup & Graph Capture (CUDA Graphs)**: Persistently caching the compilation results of CUDA Graphs to skip the lengthy Dummy Profiling stage during cold start.
