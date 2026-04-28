# Idea: HA Upgrade for Multi-Host Inference Clusters via Upgrade Domains and Delegated Readiness

## Overview
The goal is to achieve generic, decoupled high availability during Kubernetes node upgrades for multi-host inference workloads (like LWS, Ray, etc.) without introducing custom webhooks or heavy AI-specific schedulers.

## Problem Statement
In traditional Kubernetes cluster upgrades, the control plane (e.g., Node Drain) is typically **workload-agnostic** and operates at the **Pod level**. However, LLM multi-host inference is a **tightly-coupled (Gang)** workload, requiring all Pods in a group to live or die together.

This leads to the following core conflicts:
1. **Native PDB Blind Spot**: Native PodDisruptionBudget controls disruptions only by Pod count. It cannot enforce the group-level semantic where losing a single worker paralyzes the entire group.
2. **Repeated Disruptions and Capacity Avalanches**: Draining nodes sequentially disrupts the same group repeatedly. Random concurrent draining can crush multiple groups simultaneously, leading to service capacity avalanches.
3. **Platform-Application Coupling**: Solutions often resort to custom Webhooks or heavy AI schedulers (like Kueue) that require the platform to understand application-specific labels (like LWS's `group-index`), breaking the platform-application decoupling principle.

## Assumptions
The effectiveness of this idea is based on the following engineering premises and assumptions:
1. **Resource Constraints or Cost Sensitivity**: Cluster hardware resources are fixed, or the team is unwilling to waste resources by spinning up large-scale Blue-Green pools for upgrades.
2. **Draining is Mandatory**: Upgrading cluster nodes and OS requires node draining. Although the industry is exploring "drainless" live upgrade technologies, they are not natively supported and cannot always apply (e.g., when fixing data plane CVE vulnerabilities or updating the kernel, nodes must be rebooted).
3. **Business Must Tolerate Capacity Degradation**: From the first two assumptions, it follows that during upgrades we must **trade time for space**—actively reducing the total service capacity during the upgrade window while maintaining minimal High Availability (HA). Upper-layer applications must accept this temporary capacity degradation.
4. **Missing Native PodGroup PDB**: Kubernetes does not currently support native PodGroup-level eviction and Disruption Budget. Although frontier proposals like **[KEP-4563: Eviction Request API](https://github.com/kubernetes/enhancements/blob/master/keps/sig-node/4563-eviction-request-api/README.md#workload-api-support)** have begun discussing future support for upper-layer Workloads, it is still in early stages. The goal of this idea is to solve the group-level HA upgrade problem **using only existing native Kubernetes mechanisms**.

## Core Concept
This idea leverages native Kubernetes primitives and mechanisms (such as **Topology Labels**, **Pod Readiness Probes**, and **PodDisruptionBudget**) and introduces the common **"Upgrade Domain" (UD)** topology concept. By establishing a contract where a designated Pod (e.g., the Leader) aggregates the health of the entire group, we can use standard PDBs to protect group availability during rolling upgrades.

## The 3-Layer Contract
To onboard onto the platform and enjoy automated, safe upgrades, workloads must adhere to the following contract. **Note: This contract applies only to workloads that care about uptime during upgrades. Workloads that can be completely offline during maintenance windows do not need to follow this contract.**

1.  **Topology Requirement (Spread across UDs)**: Workloads must use `topologySpreadConstraints` or anti-affinity to distribute different inference groups (replicas) across different Upgrade Domains (UDs).
    *   **Example with LWS**:
        *   **Intra-Group Affinity (Pack)**: Configure Worker Pods to have strong `podAffinity` with their corresponding Leader Pod, ensuring all Pods in the same group are packed within the **same Upgrade Domain**.
        *   **Inter-Group Anti-Affinity (Spread)**: Configure Leader Pods of different groups to spread across Upgrade Domains (using `topologySpreadConstraints` or `podAntiAffinity` with the UD label as the `topologyKey`), ensuring groups are distributed across **different Upgrade Domains**.
2.  **Delegated Readiness (Health Aggregation)**: The workload must designate a Pod (e.g., LWS Leader) whose `ReadinessProbe` performs a health check on all members of the group. If any member fails or is evicted, the representative Pod becomes `Unready`.
3.  **PDB for the Representative Pod**: A PDB is defined targeting the representative Pod(s). Since the Pod's readiness reflects the group's health, the PDB effectively becomes a budget for the *number of available groups*.

## Workflow during Upgrade
The platform executes upgrades at the **Upgrade Domain** level. To ensure that PDB interception works effectively, the upgrade control plane must follow this fine-grained logic when draining nodes in a UD:

1.  **Classify Pod Types**: Scan all Pods in the target Upgrade Domain (e.g., `UD 1`) and classify them into "Pods protected by PDB" (e.g., Leaders) and "Pods without PDB" (e.g., Workers).
2.  **Evict Protected Pods First**: **You must first issue eviction requests to all Pods protected by PDB**. If unprotected Pods (Workers) are evicted first, they will destroy the group's integrity and bypass PDB checks. Only by evicting protected Pods first can the Eviction API correctly intercept violations.
3.  **Quickly Evict Remaining Pods**: Only after all protected Pods in the UD are successfully evicted can the upgrade control plane quickly drain the remaining unprotected Pods in that UD.
4.  **Cross-Domain Interception**: If the number of available groups is insufficient, step 2 will be blocked by PDB, and the upgrade control plane will pause until previous groups recover on new nodes in other UDs and become `Ready` again.

---

## Summary and Evaluation
By combining native Kubernetes topology labels, readiness probes, and PDBs, this idea achieves generic high availability upgrades for multi-host inference workloads without breaking platform-application decoupling. Its advantage lies in its generality and cloud-native nature, requiring no custom controllers for specific AI frameworks. However, in practice, the upgrade control plane must account for state propagation delays, and applications must strictly place group health checks in `ReadinessProbe` to prevent accidental kills. Overall, this is a decoupled high availability upgrade solution that adheres to the Kubernetes philosophy and fits resource-constrained environments.
