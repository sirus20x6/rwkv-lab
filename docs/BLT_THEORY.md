# RWKV-7 with BLT (Byte Latent Token): Mathematics, Complexity, and Theoretical Performance Scaling

This document provides a formal mathematical formulation, complexity analysis, and performance scaling derivation for integrating **BLT (Byte Latent Transformer/Token)** principles with the **RWKV-7** linear recurrent architecture.

---

## 1. Conceptual Alignment & Multilingual Foundations

Traditional language models rely on pre-computed subword vocabularies (e.g., BPE). While computationally efficient, subword tokenization introduces several core limitations:
1. **Multilingual Bias**: Languages with non-Latin scripts suffer from high byte-per-character ratios, leading to shorter effective context windows and higher relative cost.
2. **Robustness Issues**: Character-level typos, capitalization variations, and out-of-vocabulary terms break subword structures.

Operating directly on raw bytes (or characters) solves these issues but inflates the sequence length $T_{\text{byte}}$ by a factor of 3 to 4. For standard Transformers, this causes a $10\times$ to $16\times$ quadratic increase in attention computation.

### The BLT Philosophy: Separating "Meaning" and "Byte Representation"
BLT decouples raw byte ingestion from higher-level semantic reasoning. It uses a **local byte encoder** to dynamically group raw bytes into variable-length **patches** (representing semantic tokens or morphemes), operating the heavy **global core** only at the patch level.

In an RWKV architecture, we can reformulate the **ChannelMix** (Feed-Forward Network) block as a **BLT-style patch processor** (`RWKV7BLTChannelMix`):
- **TimeMix** operates sequentially over the fine-grained byte stream to maintain temporal recurrence, build state, and contextually model character transitions.
- **ChannelMix** represents the semantic mapping or "meaning" extraction. By dynamically pooling the byte representations at learned entropy boundaries, the Feed-Forward network operates on compressed semantic concepts (patches) rather than raw bytes, saving substantial computation.
- Because a well-trained model already possesses prior knowledge of the valid byte representation of words, the patch representation focuses primarily on transferring semantic **meaning**, rendering precise byte reconstruction at every layer less important.

---

## 2. Mathematical Formulation of BLT with RWKV-7

Let $\mathbf{x}_t \in \mathbb{R}^D$ be the continuous hidden state at byte position $t \in \{1, \dots, T_{\text{byte}}\}$ output by the preceding TimeMix layer.

### 2.1. Next-Byte Entropy Prediction & Boundary Thresholding
We project the hidden state to compute next-byte logits:
$$\mathbf{z}_t = \mathbf{W}_{\text{entropy}} \mathbf{x}_t + \mathbf{b}_{\text{entropy}} \in \mathbb{R}^{256}$$

The predicted probability distribution of the next byte is:
$$\mathbf{p}_t = \text{softmax}(\mathbf{z}_t)$$

The Shannon entropy $H_t$ (in nats) of this distribution is:
$$H_t = -\sum_{i=1}^{256} p_{t, i} \log(p_{t, i} + \epsilon)$$

High next-byte entropy $H_t \ge \theta$ signals a **boundary** (e.g., the end of a word or syllable) where the next byte is highly unpredictable. We define the boundary split trigger at step $t$ given current patch length $L_t$ as:
$$\text{split}_t = (L_t \ge L_{\text{max}}) \lor (L_t \ge L_{\text{min}} \land H_t \ge \theta)$$

---

### 2.2. Parallel Training Formulation
In parallel training, the entire sequence of length $T$ is processed. We compute $H_t$ for all $t$ and assign monotonically increasing patch indices $P_t \in \{0, \dots, N-1\}$:
$$P_t = \sum_{\tau=1}^{t} \mathbb{I}(\text{split}_{\tau-1})$$
where $\mathbb{I}$ is the indicator function, and $P_1 = 0$. Let $N$ be the total number of patches.

#### Dynamic Mean Pooling:
The pooled representation for patch $n \in \{0, \dots, N-1\}$ is:
$$\mathbf{h}_n = \frac{1}{|S_n|} \sum_{t \in S_n} \mathbf{x}_t$$
where $S_n = \{t \mid P_t = n\}$.

#### Patch-Level ChannelMix (FFN):
We run ChannelMix over the patch sequence $\{ \mathbf{h}_n \}$:
$$\mathbf{h}_{n-1} = \mathbf{h}_{n-1} \quad (\text{with } \mathbf{h}_{-1} = \mathbf{0})$$
$$\mathbf{m}_n = \mathbf{h}_n + (\mathbf{h}_{n-1} - \mathbf{h}_n) \odot \mathbf{x}_k$$
$$\mathbf{k}_n = \left(\text{ReLU}(\mathbf{m}_n \mathbf{W}_k)\right)^2$$
$$\mathbf{o}_n = \mathbf{k}_n \mathbf{W}_v$$

#### Unpooling (Gathering):
The output at byte position $t$ is mapped back to its corresponding patch:
$$\mathbf{y}_t = \mathbf{o}_{P_t}$$

---

### 2.3. Recurrent Causal Inference Formulation
Autoregressive streaming inference requires computing outputs causally without lookahead. We maintain the recurrent state $\mathcal{S}_t = (\mathbf{s}_t, L_t, \mathbf{h}_{\text{prev}}, \mathbf{y}_{\text{last}})$:
- $\mathbf{s}_t \in \mathbb{R}^D$: running sum of byte states in the current patch.
- $L_t \in \mathbb{R}$: running length of the current patch.
- $\mathbf{h}_{\text{prev}} \in \mathbb{R}^D$: mean-pooled state of the last completed patch.
- $\mathbf{y}_{\text{last}} \in \mathbb{R}^D$: FFN output of the last completed patch.

At step $t$, given input $\mathbf{x}_t$:
1. Accumulate: $\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{x}_t$ and $L_t = L_{t-1} + 1$.
2. Compute entropy: $H_t$ from $\mathbf{x}_t$.
3. Compute current patch mean: $\mathbf{h}_{\text{curr}} = \mathbf{s}_t / L_t$.
4. Check split: $\text{split}_t = (L_t \ge L_{\text{max}}) \lor (L_t \ge L_{\text{min}} \land H_t \ge \theta)$.
5. Apply FFN:
   $$\mathbf{m}_t = \mathbf{h}_{\text{curr}} + (\mathbf{h}_{\text{prev}} - \mathbf{h}_{\text{curr}}) \odot \mathbf{x}_k$$
   $$\mathbf{k}_t = \left(\text{ReLU}(\mathbf{m}_t \mathbf{W}_k)\right)^2$$
   $$\mathbf{y}_{\text{running}} = \mathbf{k}_t \mathbf{W}_v$$
6. State update:
   - If $\text{split}_t$ is **True**:
     - $\mathbf{h}_{\text{prev}} \leftarrow \mathbf{h}_{\text{curr}}$
     - $\mathbf{y}_{\text{last}} \leftarrow \mathbf{y}_{\text{running}}$
     - $\mathbf{s}_{t+1} = \mathbf{0}$, $L_{t+1} = 0$
     - Emit: $\mathbf{y}_t = \mathbf{y}_{\text{running}}$
   - If $\text{split}_t$ is **False**:
     - $\mathbf{s}_{t+1} = \mathbf{s}_t$, $L_{t+1} = L_t$
     - Emit:
       $$\mathbf{y}_t = \begin{cases} \mathbf{y}_{\text{running}} & \text{if } \text{streaming\_mode} = \text{"causal"} \\ \mathbf{y}_{\text{last}} & \text{if } \text{streaming\_mode} = \text{"discrete"} \end{cases}$$

---

### 2.4. Proof of Mathematical Equivalence at Patch Boundaries

**Theorem:** *At any step $t$ where a patch boundary is triggered ($\text{split}_t = \text{True}$), the output $\mathbf{y}_t$ of the recurrent causal inference pass is mathematically identical to the parallel training output at the corresponding position.*

**Proof:**
1. Let the current patch span positions $t_{\text{start}}$ to $t_{\text{end}} = t$.
2. In parallel training, the pooled representation for this patch is:
   $$\mathbf{h}_n = \frac{1}{t_{\text{end}} - t_{\text{start}} + 1} \sum_{\tau = t_{\text{start}}}^{t_{\text{end}}} \mathbf{x}_{\tau}$$
3. In recurrent inference, the accumulator starts accumulating at $t_{\text{start}}$ with $\mathbf{s}_{t_{\text{start}}} = \mathbf{x}_{t_{\text{start}}}$, and at step $t_{\text{end}}$ we have:
   $$\mathbf{s}_{t_{\text{end}}} = \sum_{\tau = t_{\text{start}}}^{t_{\text{end}}} \mathbf{x}_{\tau}, \quad L_{t_{\text{end}}} = t_{\text{end}} - t_{\text{start}} + 1$$
   $$\mathbf{h}_{\text{curr}} = \frac{\mathbf{s}_{t_{\text{end}}}}{L_{t_{\text{end}}}} = \frac{1}{t_{\text{end}} - t_{\text{start}} + 1} \sum_{\tau = t_{\text{start}}}^{t_{\text{end}}} \mathbf{x}_{\tau} = \mathbf{h}_n$$
4. Since $\mathbf{h}_{\text{prev}}$ in recurrent inference holds the pooled state of the previous completed patch $\mathbf{h}_{n-1}$, the inputs to the FFN are identical:
   $$\mathbf{m}_t = \mathbf{h}_{\text{curr}} + (\mathbf{h}_{\text{prev}} - \mathbf{h}_{\text{curr}}) \odot \mathbf{x}_k = \mathbf{h}_n + (\mathbf{h}_{n-1} - \mathbf{h}_n) \odot \mathbf{x}_k = \mathbf{m}_n$$
5. Consequently, the linear projections and activation outputs are identical:
   $$\mathbf{y}_t = \mathbf{y}_{\text{running}} = \left(\text{ReLU}(\mathbf{m}_t \mathbf{W}_k)\right)^2 \mathbf{W}_v = \left(\text{ReLU}(\mathbf{m}_n \mathbf{W}_k)\right)^2 \mathbf{W}_v = \mathbf{y}_{\tau} \quad \forall \tau \in S_n$$
   This completes the proof. $\blacksquare$

---

## 3. Complexity & FLOP Performance Approximations

We compare the theoretical performance of three architectures operating on a sequence of length $T_{\text{byte}}$ bytes:
1. **Standard Byte-Level RWKV-7**: Each byte processed directly.
2. **Standard Subword Token-Level RWKV-7**: Sequences of length $T_{\text{token}} = T_{\text{byte}} / P$, where $P$ is the average token length (e.g., $P \approx 4$).
3. **BLT-RWKV-7**: Recurrent TimeMix at byte-level, but ChannelMix (FFN) at patch-level.

### 3.1. Parameter Definitions
- $L$: Number of layers.
- $D$: Global hidden dimension.
- $F = 4D$: FFN intermediate dimension.
- $P$: Average patch/token size in bytes ($P \approx 4$).

### 3.2. FLOP Calculations per Byte

#### 1. Standard Byte-Level RWKV-7:
Each byte goes through $L$ full layers of TimeMix + ChannelMix.
- TimeMix (receptance, key, value, decay, output projections + state update): $\approx 10 D^2$ FLOPs per layer.
- ChannelMix (FFN projections: $\mathbf{W}_k$ and $\mathbf{W}_v$): $2 \times 2 \times D \times 4D = 16 D^2$ FLOPs per layer.
- **Total FLOPs per Byte**:
  $$\text{FLOPs}_{\text{byte}} \approx L \times (10 D^2 + 16 D^2) = 26 L D^2$$

#### 2. Standard Token-Level RWKV-7:
The sequence length is compressed to $T_{\text{token}} = T_{\text{byte}} / P$.
- **Total FLOPs per Byte**:
  $$\text{FLOPs}_{\text{token}} \approx \frac{26 L D^2}{P}$$
*Note: Token-level models require large vocabulary embedding tables and LM heads ($V \approx 50,000 \dots 100,000$), whereas byte-level models require only $V = 256$.*

#### 3. BLT-RWKV-7 (TimeMix at Byte, ChannelMix at Patch):
TimeMix runs at the byte level ($10 L D^2$ FLOPs), but ChannelMix runs only at the patch level ($16 L D^2 / P$ FLOPs). We add the minor overhead of the entropy head ($2 \times 256 \times D \approx 512 D$ FLOPs per layer, negligible for large $D$).
- **Total FLOPs per Byte**:
  $$\text{FLOPs}_{\text{BLT-RWKV}} \approx L \times \left(10 D^2 + \frac{16 D^2}{P}\right) = \left(10 + \frac{16}{P}\right) L D^2$$

### 3.3. Theoretical Speedup Comparison
The theoretical speedup of **BLT-RWKV-7** over a standard **Byte-Level RWKV-7** is:
$$\mathcal{S} = \frac{26}{10 + 16/P}$$

Let's evaluate this for different average patch sizes $P$:
- For $P = 2$: $\mathcal{S} = \frac{26}{10 + 8} \approx 1.44\times$ speedup.
- For $P = 4$: $\mathcal{S} = \frac{26}{10 + 4} \approx 1.86\times$ speedup.
- For $P = 8$: $\mathcal{S} = \frac{26}{10 + 2} \approx 2.17\times$ speedup.
- Limit as $P \to \infty$: $\mathcal{S} \to 2.60\times$ speedup.

Since ChannelMix typically consumes more than $60\%$ of total model FLOPs, evaluating FFN layers only at patch boundaries offers a substantial speedup while maintaining fine-grained byte recurrence in TimeMix.

### 3.4. VRAM Footprint Savings
In parallel training, we must store activations for backpropagation:
- **Standard Byte-Level**: Stores activations for both TimeMix and ChannelMix for all $T_{\text{byte}}$ positions: $O(T_{\text{byte}} \cdot L \cdot D)$.
- **BLT-RWKV-7**: Stores TimeMix activations for $T_{\text{byte}}$ positions, but ChannelMix activations only for the $N = T_{\text{byte}}/P$ pooled patch positions. This reduces the FFN activation memory footprint by a factor of $P$, leading to significant VRAM savings and enabling training with much longer context lengths.

---

## 4. Conclusion and Future Potential

BLT-RWKV-7 combines the best of both worlds:
1. **Fine-grained temporal modeling** at the byte level via TimeMix, avoiding subword dictionary limitations.
2. **Semantic representation processing** at the patch level via ChannelMix, yielding up to a **$2.6\times$ theoretical speedup** and significantly reduced training VRAM.
3. **Causal, streaming zero-latency inference** with exact parallel-training equivalence at boundaries, making it highly suitable for real-time edge deployment.
