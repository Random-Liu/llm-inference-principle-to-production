# TODO List

## English Language Refinement (Concise & Active Style)
Follows the principles in `GEMINI.md`:
- [x] Part 1: Principles ([part1_principles.md](parts/part1_principles.md))
- [x] Part 2: Bottlenecks ([part2_bottlenecks.md](parts/part2_bottlenecks.md))
- [x] Part 3: Single-Node ([part3_single_node.md](parts/part3_single_node.md))
- [ ] Part 4: Distributed ([part4_distributed.md](parts/part4_distributed.md))
- [ ] Part 5: Orchestration ([part5_k8s.md](parts/part5_k8s.md))
- [ ] Part 6: Frontier ([part6_frontier.md](parts/part6_frontier.md))

## Content Creation
- [ ] Complete Part 5: Orchestration ([part5_k8s.md](parts/part5_k8s.md))
- [ ] Complete Part 6: Frontier ([part6_frontier.md](parts/part6_frontier.md))
- [ ] Add evaluation for new challenges: MoE, CoT, Agent


## Research & Learning
- [ ] Explore unseen content:
    - [ ] Agent architecture (e.g., OpenClaw)
    - [ ] Explore new requirements for serving agents
    - [ ] LoRA principles and inference requirements
    - [ ] Expert parallel (EP) and MoE inference requirements
    - [ ] Explore TPU architecture (e.g., topology) and its solutions to Part 5 challenges
    - [ ] Research GB series hardware topology
    - [ ] Understand how `DisaggregatedSet` works in LeaderWorkerSet (for Disaggregated Serving). Ref KEP: https://github.com/kubernetes-sigs/lws/pull/773
    - [ ] Learn the complete CUDA software stack to understand its dependencies
    - [ ] Explore Kubernetes Gateway API Inference Extension
