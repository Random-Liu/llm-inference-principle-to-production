# Part 1: Principles — The Foundational Engine of Transformers and LLMs

## Table of Contents
- [Chapter 1: Demystifying the Transformer: The Magic of Q, K, and V](#chapter-1-demystifying-the-transformer-the-magic-of-q-k-and-v)
  - [Section 1: A Bird's-Eye View: Classic Transformer Architecture](#section-1-a-birds-eye-view-classic-transformer-architecture)
  - [Section 2: Evolution: Decoder-Only Architecture](#section-2-evolution-decoder-only-architecture)
  - [Section 3: The Library Analogy: Intuitive Meaning of QKV](#section-3-the-library-analogy-intuitive-meaning-of-qkv)
  - [Section 4: Mathematical Principles: Matrix Computation of QKV](#section-4-mathematical-principles-matrix-computation-of-qkv)
  - [Section 5: Feed-Forward Network: Knowledge Base](#section-5-feed-forward-network-knowledge-base)
  - [Section 6: Multi-Head Attention: Parallel Perspectives](#section-6-multi-head-attention-parallel-perspectives)
  - [Section 7: Mixture of Experts: Sparse Activation](#section-7-mixture-of-experts-sparse-activation)
- [Chapter 2: Building the Skyscraper: Stacking Layers and Data Flow](#chapter-2-building-the-skyscraper-stacking-layers-and-data-flow)
  - [Section 1: Input Stage: Embeddings and Positional Encoding](#section-1-input-stage-embeddings-and-positional-encoding)
  - [Section 2: Wisdom of Stacking: Why Stacking Multiple Transformers?](#section-2-wisdom-of-stacking-why-stacking-multiple-transformers)
  - [Section 3: The Translator: LM Head](#section-3-the-translator-lm-head)
  - [Section 4: Logits and Softmax: Converting Raw Scores to Probabilities](#section-4-logits-and-softmax-converting-raw-scores-to-probabilities)
  - [Section 5: Pop Science: What Do We Mean by 8B/70B Parameters?](#section-5-pop-science-what-do-we-mean-by-8b70b-parameters)
  - [Section 6: Data Flow: The End-to-End Pipeline](#section-6-data-flow-the-end-to-end-pipeline)
- [Chapter 3: The Art of Operation: Autoregressive Decoding and Text Generation](#chapter-3-the-art-of-operation-autoregressive-decoding-and-text-generation)
  - [Section 1: Prefill Phase: Handling Input Context](#section-1-prefill-phase-handling-input-context)
  - [Section 2: Decode Phase: The Autoregressive Loop](#section-2-decode-phase-the-autoregressive-loop)

This part explains the physical and mathematical foundations of LLMs.

---

## Chapter 1: Demystifying the Transformer: The Magic of Q, K, and V

In Large Language Models (LLMs), all magic stems from the **Transformer** architecture, and its core is the **Self-Attention mechanism**. This chapter breaks down the three most famous letters in self-attention: **Q (Query)**, **K (Key)**, and **V (Value)**. They are the soul of the model's ability to understand context and capture complex relationships between words.

---

### Section 1: A Bird's-Eye View: Classic Transformer Architecture

```mermaid
graph BT
    subgraph Decoder ["Decoder"]
        direction BT
        Out[Generated Text] --> Emb2[Embedding]
        Emb2 --> PE2[Positional Encoding]
        PE2 --> MSA[Masked Self-Attention]
        MSA --> CA[Cross-Attention]
        CA --> FFN2[FFN]
    end
    subgraph Encoder ["Encoder"]
        direction BT
        In[Input Text] --> Emb1[Embedding]
        Emb1 --> PE1[Positional Encoding]
        PE1 --> SA1[Unmasked Self-Attention]
        SA1 --> FFN1[FFN]
    end
    FFN1 -->|Hidden States| CA
    CA ~~~ FFN1
    FFN2 --> Head[LM Head]
    Head --> Prob[Predict Next Token]
```

Before diving into the micro world of QKV, let's grasp the macro workflow of the classic **Encoder-Decoder** Transformer architecture (the original design).

We can compare this process to **Simultaneous Interpretation**:

1.  **Left Side: Encoder — "Listen, Understand, and Take Notes"**
    *   **Input**: For example, the English sentence "The cat is black".
    *   **Process**: Data enters from the bottom and passes through multiple layers.
    *   **Core Mechanism**: Every layer contains a "Self-Attention mechanism". **Please note that self-attention here is "Unmasked"**. This is different from the Masked Self-Attention in models like GPT, which can only look at the preceding context. In the Encoder, all words in the sentence can **observe each other**, understanding the context without any blind spots. This is completely logical in translation tasks because the source sentence is **known and complete**. We don't need to predict it; we only need to fully extract its semantics and maximize context information.
    *   **Output**: The top outputs are **hidden state vectors** rich in contextual information. They represent that the Encoder has completely "understood" the sentence.

2.  **Right Side: Decoder — "Translate and Express"**
    *   **Input**: It receives two pieces of information: first, the "secret secret report understood" passed from the Encoder; second, **the words it translated at previous moments**.
    *   **Process**:
        *   **Masked Self-Attention**: This is the core mechanism of the Decoder. When predicting the next word, it can only observe the words already spoken before it, not "peek" into future words. **Why?** Because during inference, future words have not been generated yet; and during training, if peeking into future words is allowed, the model will cheat by looking up the answers, failing to acquire real prediction ability.
        *   **Multi-Head Cross-Attention**: **This is the most critical step!** The Decoder uses its current intent (Query) to search the secret report from the Encoder for matching clues (Key and Value).
    *   **Output**: Passes through Softmax to predict the probability of the next word (e.g., "est") based on the previously translated "Le chat".

**Summary**: The Encoder is responsible for "understanding the input" (bi-directional observation), and the Decoder is responsible for "generating the output based on understanding" (uni-directional generation).

---

### Section 2: Evolution: Decoder-Only Architecture

```mermaid
graph BT
    subgraph DecoderOnly ["Decoder-Only Block"]
        In[Input Tokens] --> Emb[Embedding]
        Emb --> PE[Positional Encoding]
        PE --> MSA[Masked Self-Attention]
        MSA --> FFN[FFN]
    end
    FFN --> Head[LM Head]
    Head --> Prob[Predict Next Token]
```

Following the classic Transformer, large language models underwent a major architectural evolution. Today's prominent large models such as GPT, Llama, and DeepSeek did not adopt the native Encoder-Decoder dual-tower architecture, but shifted towards the extremely simplified **Decoder-Only** architecture — they discarded the left side Encoder and kept only the right side Decoder.

Seeing this, readers might naturally ask: **Since the Encoder is so skilled at understanding, why did modern models discard it, leaving everyone with a "single tower" Decoder-Only architecture?**

This involves a profound paradigm transition:
1.  **Transformers were originally designed for "Translation"**: In translation tasks, "input" and "output" are inherently segregated entities (e.g., English and Chinese). Therefore, the Encoder first understands the English, and the Decoder translates it into Chinese.
2.  **Modern LLMs play "Text Continuation"**: Scientists discovered that all natural language tasks (Q&A, code generation, reasoning, and translation) can be unified into a continuation game of **"giving the previous text, predicting the next word"**.
3.  **The Decoder Handles Everything**: Since it is simply a continuous stream of text, we no longer need two physically segregated towers. We splice the Prompt and Response together and feed them entirely to the Decoder.

The Decoder-Only architecture is immensely simplified compared to the classic design:
1.  **Encoder Eliminated**: There is no independent Encoder tower.
2.  **Cross-Attention Eliminated**: Since there is no Encoder, cross-tower interactive attentions are no longer needed.
3.  **Unified Input**: Prompt and Response are spliced into a continuous sequence and fed uniformly from the bottom.
4.  **Core Mechanism**: The entire model consists of stacked **Masked Self-Attention** blocks and **Feed-Forward Networks (FFN)** blocks.

Under this architecture, how does the model operate?
*   **Prefill Phase**: Your input Prompt is processed in one go. Although it uses Masked Self-Attention, because the prompt text is known, the model can compute relationships between prompt tokens in parallel (similar to how an Encoder works).
*   **Decode Phase**: The model generates the response word by word. As each new word is generated, it is added to the end of the input sequence to predict the next word. At this point, Masked Self-Attention ensures the generation can only look back at preceding prompts and words, securing causal relationships.

This simplistic design not only optimizes pre-training on massive data but also provides the unified physical foundation for production serving optimizations like **KV Cache**.

---

### Section 3: The Library Analogy: Intuitive Meaning of QKV

Before delving into complex mathematical formulas, let's use a highly intuitive scenario to understand the logical meaning of Q, K, and V.

Imagine walking into a giant **Science Library** searching for material on "noise-canceling Bluetooth headphones". In this scene, Q, K, and V play distinct roles:

1.  **Q (Query)**: Represents **your current intent**. The search words you enter in the computer: "noise-canceling Bluetooth headphones". This represents what information characteristics you are currently seeking.
2.  **K (Key)**: Represents the **indices or tags of book content**. Every book has a name, author, abstract, and classification tag. For instance:
    *   Book A's Key is: "Wired Gaming Headphones Review".
    *   Book B's Key is: "Sony Noise-Canceling Bluetooth Headphones Teardown and Chip Analysis".
3.  **V (Value)**: Represents the **actual knowledge within the books**. If you ultimately decide to read Book B, the detailed words you absorb about chipsets and acoustics are the Value.

**The overall logical flow of the Self-Attention mechanism** involves taking your **Query**, matching it against the **Keys** of all books to calculate a relevance score, and then allocating your reading attention to the **Values** of each book based on those scores.

Mapping this back to LLMs, let’s use "Apple" as a concrete example:

Suppose we have two sentences:
*   Sentence A: "At today’s **new product launch**, **Apple** introduced..."
*   Sentence B: "The box of **apples** I bought at the **supermarket** was very..."

When the model processes the word "Apple":
1.  **It generates its own Query (Q)**: Representing its "searching intent".
    > [!NOTE]
    > In reality, this Query is a complex high-dimensional vector housing hundreds of abstract search dimensions formed during pre-training. We use humanized wording like "I am 'apple', I need 'tech' or 'fruit' clues" simply to facilitate human understanding.
2.  **It matches its Query against the Keys (K) of all preceding words**:
    *   In **Sentence A**, the **Q** of "Apple" scores a high match against the **K** of "**new product launch**" (since launches tie strongly to tech companies).
    *   In **Sentence B**, the **Q** of "apples" matches strongly with the **K** of "**supermarket**" (since supermarkets tie to food and fruit).
3.  **Extracts Values (V) based on weights**:
    *   In **Sentence A**, the model assigns heavy weights to the **V** of "new product launch", causing the final vector for "Apple" to skew toward **"Apple Inc."**.
    *   In **Sentence B**, the model assigns heavy weights to the **V** of "supermarket", skewing the final vector for "apples" toward **"fruit"**.

Through dynamic matching, identical words secure precise definitions across differing contexts.

---

### Section 4: Mathematical Principles: Matrix Computation of QKV

With the logical meaning understood, let’s examine how LLMs dynamically compute Q, K, and V through matrix multiplication.

#### 0. What is Word Embedding?
Before calculating, text must be transformed into computer-friendly numbers. Suppose the input is the word "Apple". The model first looks up a dictionary (Embedding table) and translates "Apple" into a continuous vector $X$ of say 4096 dimensions. This string of numbers represents the initial semantic coordinates of "Apple" in a multi-dimensional space.

#### 1. Linear Mapping (From Embeddings to Q, K, and V)
Suppose we hold the word vector $X$. In Transformer models, there are three core **weight matrices** learned through pre-training: $W_Q$, $W_K$, and $W_V$.

Input word vectors multiply against these three matrices to dynamically generate corresponding Q, K, and V vectors:

$$Q = X W_Q$$
$$K = X W_K$$
$$V = X W_V$$

> [!NOTE]
> **Static Weights vs. Dynamic Data**
> A critical boundary exists: $W_Q, W_K, W_V$ are **static model weights** fixed in VRAM after training, acting as shared "processing rules" for all Tokens. Q, K, and V are **dynamically generated data**, computed in real-time by multiplying the input vector $X$ against the static weight matrices. This is the reason why a single word secures different semantic meanings across various contexts.

#### 2. Computing Similarities and Attention Weights
To determine the attention the current word (Query) allocates to preceding words (Keys), the model computes the **dot product** of the current word's Q and preceding words' K. A higher dot product represents closer semantic proximity.

$$\text{Score} = Q \cdot K^T$$

To prevent numerical explosions and vanishing gradients, the model divides the score by a scaling factor $\sqrt{d_k}$ ($d_k$ represents the vector's dimension). Subsequently, a **Softmax** function maps these scores into a probability distribution where all weights add up to 1:

$$\text{Attention Weights} = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right)$$

#### 3. Extracting Information (Weighted Sum)
Lastly, the model uses the computed attention weights to perform a weighted sum against the **Values** of all words, completing context retrieval:

$$\text{Output} = \sum (\text{Attention Weights} \times V)$$

---

### Section 5: Feed-Forward Network: Knowledge Base

After self-attention completes word-to-word information exchanges, vectors enter the **Feed-Forward Network (FFN)**. While Attention focuses on "finding relationships," FFN handles the "internal reflection" of individual words.

#### 1. The Classic FFN Workflow
Vectors $H$ entering the FFN are not raw attention outputs. They are a **Residual Connection** summing the **Attention output** and the **original input vector $X$**.

This $H$ vector holds a blend of "who I am" (original meaning) and "what I've gone through" (global context). The FFN processes it in three stages:

1.  **Up-Projection**: The input vector $H$ multiplies against a weight matrix $W_1$. In most models, this expands dimensions from 4096 up to 16384, opening a wider space for complex feature extraction.
2.  **Activation**: The up-projected vector passes through a non-linear activation function $\sigma$ (e.g., ReLU, GeLU, or SwiGLU). This introduces non-linear expressivity while filtering and selecting information.
3.  **Down-Projection**: The activated vector multiplies against $W_2$, scaling dimensions from 16384 back to 4096 so it can sum residually with the initial input.

#### 2. Soft KV Memory: The Symmetry of FFN and Attention
While Attention relies on Q, K, and V, FFN only holds $W_1$ and $W_2$. A famous 2020 paper demonstrated profound mathematical symmetry: **FFN is essentially a Key-Value memory retrieval system!**

Let’s contrast formulas:
*   **Attention Formula**: $\text{Output}_{attn} = \text{Softmax}(Q \cdot K^T) \cdot V$
*   **FFN Formula**: $\text{Output}_{ffn} = \sigma(H \cdot W_1) \cdot W_2$

Under this view, FFN computation perfectly mirrors the Q, K, V logic:

1.  **The input vector $H$ is the Query (Q)!** It queries the FFN: "Here is my current status. Are there complementary facts in the memory?"
2.  **The Up-projection matrix $W_1$ acts as the Keys (K)**. We slice $W_1$ column-wise into 16384 vectors. Each represents a specific "pattern" learned by the model. Computing $H \cdot W_1$ yields the similarity between Query $H$ and knowledge Keys $W_1$.
3.  **The Down-projection matrix $W_2$ acts as the Values (V)**. We slice $W_2$ row-wise. Each represents specific knowledge tied to a pattern.

When $H$ enters the FFN, knowledge retrieval proceeds:
If processing **Sentence B** ("apples at the supermarket"):
1.  **Pattern Match**: $H$ (fused as "fruit apple") computes against $W_1$. It matches strongly with "fruit/food" patterns.
2.  **Activation Filter**: $\sigma$ zeroes out unrelated scores (like "tech companies").
3.  **Knowledge Fetch**: Retrieves "crisp, juicy" from $W_2$, blending them into the final output.

Conversely, processing **Sentence A** ("Apple new product launch"):
1.  **Pattern Match**: $H$ matches strongly with "tech, digital, company" patterns.
2.  **Activation Filter**: Filters out the "fruit" pattern.
3.  **Knowledge Fetch**: Retrieves "iPhone, high tech" from $W_2$.

#### 3. Residual Integration
FFN knowledge sums residually with the original vector:
$$x_{new} = H + FFN(H)$$
Tokens carry a **"notepad"**:
*   $H$ states: I am an "apple" situated in an "eating" context.
*   $FFN(H)$ adds: Physical traits include "crisp and juicy".
*   **Summation**: The token understands both the context and the facts, advancing memory down the layers.

---

### Section 6: Multi-Head Attention: Parallel Perspectives

Early simplifications assumed a single set of $W_Q, W_K, W_V$. Forcing complex semantics into a single vector risks oversight. Modern LLMs stack dozens of these matrices in every layer—known as **Multi-Head Attention (MHA)**.

**Why multiple heads?**
Human language holds multi-faceted roles:
*   **Head 1 (Grammar)**: Identifies subject-verb-object relations.
*   **Head 2 (Emotion)**: Catches emotionally charged adjectives.
*   **Head 3 (Reference)**: Identifies what pronouns like "it" or "him" point to.

**The Fourth Matrix: $W_O$**
MHA doesn't just duplicate $W_Q$, $W_K$, and $W_V$. To compress fragmented outputs back into the original dimensions, models introduce an **Output Projection Matrix $W_O$**.

---

### Section 7: Mixture of Experts: Sparse Activation

MoE is the **multi-replica upgrade of FFN**.

#### 1. Dense Model Pain Points
In traditional **Dense** models, each layer holds a single FFN. All tokens pass through the same FFN. Scaling knowledge requires scaling the FFN, driving up compute costs (FLOPs) and inference latency.

#### 2. MoE Solution: Division of Labor
MoE shards a monolithic FFN into multiple smaller FFNs—each called an **Expert**.
*   **Router**: When tokens enter, the Router measures their relevance against experts.
*   **Experts**: Every expert is a standard FFN trained to specialize in particular domains.

#### 3. Sparse Activation
1.  **Router Calculation**: Routes tokens to corresponding specialists.
2.  **Sparse Activation**: Activates Top-K relevant experts (e.g., Experts 3 and 5) and leaves others idle.
3.  **Fusion**: Only activated experts process the tokens; outputs fuse according to weighted relevance.

MoE delivers **"high total parameters (massive knowledge) but low activated parameters (economical compute)"**. All experts must reside in VRAM, demanding strict memory capacities.

---

## Chapter 2: Building the Skyscraper: Stacking Layers and Data Flow

---

### Section 1: Input Stage: Embeddings and Positional Encoding

Before data enters the multi-layer Transformer Blocks, it goes through the entrance lobby for formatting and injection of core insights.
1.  **Word Embedding**: Words map onto high-dimensional continuous coordinates.
2.  **Positional Encoding**: Attention is natively "time-blind". Modern SOTA models apply **RoPE (Rotary Position Embedding)**, rotating Q and K vectors complexly in geometric spaces during dot products. Words close in proximity sustain minimal rotational differences and yield high scores.

---

### Section 2: Wisdom of Stacking: Why Stacking Multiple Transformers?

Armed with positional embeddings, tokens ascend the Transformer Block skyscraper (e.g., 80 layers in Llama 3 70B).
*   **Low Layers**: Extract grammar and local syntactic relationships.
*   **Middle Layers**: Comprehend entities and commonsense facts from FFNs.
*   **High Layers**: Handle abstract concepts and logical reasoning.

**Slanted Information Flow**: Parallel prompt processing means tokens do not move horizontally across a single layer; information moves **slantwise upwards**, letting GPUs compute in parallel while securing deep semantic integration.

```mermaid
graph LR
    subgraph Layer1 ["Layer 1"]
        direction TB
        L1_T1["Token 1"]
        L1_T2["Token 2"]
        L1_T3["Token 3"]
        
        L1_T1 -->|Attention| L1_T2
        L1_T1 -->|Attention| L1_T3
        L1_T2 -->|Attention| L1_T3
    end

    subgraph Layer2 ["Layer 2"]
        direction TB
        L2_T1["Token 1'"]
        L2_T2["Token 2'"]
        L2_T3["Token 3'"]
        
        L2_T1 -->|Attention| L2_T2
        L2_T1 -->|Attention| L2_T3
        L2_T2 -->|Attention| L2_T3
    end

    L1_T1 -->|Inter-layer Transfer| L2_T1
    L1_T2 -->|Inter-layer Transfer| L2_T2
    L1_T3 -->|Inter-layer Transfer| L2_T3
```

---

### Section 3: The Translator: LM Head

When data exits the top layer, each token yields a final hidden state vector ($h_{last}$). The **LM Head** maps these abstract vectors onto a huge vocabulary matrix (shaped `[Vector Dimension, Vocab Size]`), returning raw **Logits**.

> [!NOTE]
> Intermediate layers are kept abstract to prevent logical informational losses. Static models used **Weight Tying** (sharing Embedding and LM Head matrices physically) to save VRAM, whereas SOTA models decouple them.

---

### Section 4: Logits and Softmax: Converting Raw Scores to Probabilities

LM Head outputs raw integer scores (Logits). The **Softmax** function processes them into probability distributions between 0 and 1, ensuring all token probabilities sum to 1.

---

### Section 5: Pop Science: What Do We Mean by 8B/70B Parameters?

Parameters denote all trainable floating-point numbers in weight matrices. Let's analyze **Llama 3 (405B)**:
1.  **Word Embedding Layer**: $128,256 \times 16384 \approx \mathbf{2.1 \text{ Billion Parameters}}$.
2.  **Transformer Layers**: Every layer encompasses Attention and FFN matrices ($16384 \times 53248$), totaling 3.18B parameters per layer. Across 126 layers, parameters total $\approx \mathbf{401.6 \text{ Billion}}$. **FFN consumes ~82% of parameters**, housing the hard knowledge.
3.  **Output Layer (LM Head)**: $\approx \mathbf{2.1 \text{ Billion Parameters}}$.
*   **Final Bill**: $2.1 \text{B} + 401.6 \text{B} + 2.1 \text{B} \approx \mathbf{405.8 \text{ Billion Parameters}}$.

---

### Section 6: Data Flow: The End-to-End Pipeline

1.  **Input Phase**: Prompts map onto word embeddings.
2.  **Stack Transit Phase**: Tokens interact and sum residually with original vectors, ascending the skyscraper until they hit $h_{last}$.
3.  **Output Phase**: $h_{last}$ passes through RMSNorm, is projected by the LM Head into Logits, and converts to probabilities via Softmax.

---

## Chapter 3: The Art of Operation: Autoregressive Decoding and Text Generation

---

### Section 1: Prefill Phase: Handling Input Context

1.  **Tokenization**: The prompt shards into tokens.
2.  **One-Shot Feed**: Fed into the model concurrently.
3.  **Parallel Compute**: Attention weights and vectors are calculated simultaneously on GPUs to extract context comprehension.
4.  **First Probability**: Emits the probability distributions for the subsequent token.

---

### Section 2: Decode Phase: The Autoregressive Loop

1.  **Emit First Token**: E.g., "AI".
2.  **Autoregression Loop**: The token appends to the prompt, fed back into Layer 1. The sequence travels up the skyscraper again to predict the next token. The loop iterates recursively.

---
