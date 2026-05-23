## Recurrent Architectures for Complex Sequence Modelling

### 1. The Paradigm of Sequential Data and Probabilistic Foundations

In modern predictive modeling, the strategic shift from static to temporal data processing is driven by the necessity to capture **order-significance**. Unlike traditional independent and identically distributed (IID) data, sequential data is defined by its position-dependent information. Transduction tasks—mapping an input sequence to an output sequence—require architectures capable of handling the three pillars of sequential data:

* **Ordered:** Data points possess a specific index; permutations fundamentally alter the signal (e.g., DNA sequence analysis where nucleotide order determines biological function).
* **Variable Length:** Inputs often lack a fixed dimension (e.g., varying sentence lengths in sentiment classification).
* **Dependencies:** Significant temporal or spatial relationships exist (e.g., frame-to-frame dependencies in video activity recognition).

#### Mathematical Foundations of Sequence Models

The core objective is to estimate the joint probability of a sequence $x_1, x_2, \dots, x_T$. Using the chain rule of probability, we formulate Autoregressive models to factor this joint probability:

$$P(x_1, \dots, x_T) = P(x_1) \prod_{t=2}^{T} P(x_t | x_1, \dots, x_{t-1})$$

When applied to text, this represents a Language Model. However, the conditioning term $P(x_t | x_1, \dots, x_{t-1})$ becomes statistically difficult and computationally expensive to model as $t$ grows, necessitating a compact internal representation.

#### The Markov Assumption and Architectural Necessity

To ensure computational tractability, the Markov Assumption simplifies dependencies. A First-Order Markov Model (Bigram) assumes $x_t$ depends only on $x_{t-1}$, losing long-term context. While linear models can extend this to a fixed window $\tau$, they suffer from **Error Accumulation** during multi-step forecasting, where early prediction errors $\epsilon$ propagate through a feedback loop, causing rapid divergence.

Furthermore, standard Multilayer Perceptrons (MLPs) and Convolutional Neural Networks (CNNs) are suboptimal for these tasks. They lack **Parameter Sharing Across Time**, requiring separate weights for every position, and are constrained by **Fixed-Size Vectors**, necessitating information-destroying padding or truncation. Recurrent hidden states are thus required to provide a step-by-step memory mechanism.

---

### 2. Standard Recurrent Neural Networks (RNN): Mechanics and The Gradient Wall

The standard RNN serves as the foundational memory-based architecture. It processes non-independent data points by maintaining a latent hidden state $h_t$ that summarizes the history of inputs.

[Diagram 1: Unrolled Recurrent Neural Network Architecture. Source: ResearchGate](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSPXLrncxmmTHXOv9sxdZ4hdG2qH0aVAf7hGwx56cLJRQ0EziKoN2nCB4g3&s=10)

#### RNN Architecture and Calculated Trace

At each time step, the network interacts with an input vector $x_t$ and the previous hidden state $h_{t-1}$. The fundamental computations are:

$$h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h)$$

$$y_t = W_{hy}h_t + b_y$$

To illustrate the internal mechanics, consider a Next-Character Prediction task where the model predicts 'l' given the input 'e' and the memory of 'h'.

1. **Setup:** Vocabulary `{'h':0, 'e':1, 'l':2, 'o':3}`. Input $x_t$ ('e') = $[0, 1, 0, 0]^T$.
2. **Trace:** Given $h_{t-1} = [0.6, 0.2]^T$ and hypothetical weights, the hidden state update yields $h_t = \tanh([0.46, 0.92]) = [0.43, 0.72]^T$.
3. **Logit Generation:** Applying the output weights $W_{hy}$ results in logit scores for each character: h=0.24, e=0.63, l=2.89, o=-0.61. The highest logit (2.89) correctly predicts 'l'.

#### Backpropagation Through Time (BPTT) and Numerical Instability

Training utilizes BPTT, applying the chain rule to the unrolled computational graph. The gradient $\frac{\partial L}{\partial h_t}$ consists of two components:

* Current Loss: $\frac{\partial L_t}{\partial y_t} \frac{\partial y_t}{\partial h_t}$
* Future Loss: $\frac{\partial L}{\partial h_{t+1}} \frac{\partial h_{t+1}}{\partial h_t}$

The recursive product of $W_{hh}$ over $T$ steps leads to the **Twin Problems**:

* **Gradient Explosion:** Large weights cause gradients to grow exponentially (Infinity/NaN), requiring Gradient Clipping to scale the norm.
* **Gradient Vanishing:** Small weights cause gradients to shrink toward zero, preventing the model from learning long-range dependencies.

> **Practical Compromise:** Truncated BPTT detaches the gradient history after $\tau$ steps while allowing the hidden state to flow forward, maintaining memory but limiting the learnable dependency range.

Explore how vanishing and exploding gradients functionally destroy the learning process in standard RNNs:

---

### 3. Long Short-Term Memory (LSTM): The Gated Memory Solution

LSTMs overcome the vanishing gradient problem through selective information gating and a dual-state architecture.

[Diagram 2: Internal Architecture of an LSTM Cell. Source: Medium](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRIYnxuffWWRm2-wlUdv7CFGVlI5UaoZj8pB6nHGrlehmeT_2n6ovnuGTk&s=10)

#### Dual State Architecture

* **Cell State ($c_t$):** The "conveyor belt" for long-term memory, modified only by linear interactions.
* **Hidden State ($h_t$):** The short-term state and output of the unit.

#### Gating Mechanisms: Strategic Purpose

Each gate uses the concatenation $[h_{t-1}, x_t]$ to determine information flow:

* **Forget Gate ($f_t$):** $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$. Outputs 0 (forget) to 1 (keep) to selectively discard $c_{t-1}$.
* **Input ($i_t$) & Candidate ($\tilde{c}_t$) Gates:** $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$ and $\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$. Together, they filter and prepare new information.
* **Update Equation:** $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$. This additive interaction is the critical differentiator; it allows gradients to flow back without the exponential decay caused by repeated matrix multiplication.
* **Output Gate ($o_t$):** $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$. It filters $c_t$ to produce the final $h_t = o_t \odot \tanh(c_t)$.

---

### 4. Gated Recurrent Units (GRU): Optimized Efficiency

The GRU is a streamlined evolution of the LSTM, consolidating states and reducing parameter complexity without sacrificing dependency-handling performance.

#### Architecture Consolidation

The GRU merges the cell and hidden states into a single vector $h_t$. It replaces the separate memory cell with a Candidate Activation Vector ($\tilde{h}_t$):

$$\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t])$$

#### Primary Gating Mechanisms

* **Reset Gate ($r_t$):** $r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$. It "resets" the past influence when a new, unrelated sequence begins.
* **Update Gate ($z_t$):** $z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$. It controls the linear interpolation: $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$.

#### Architectural Comparison: LSTM vs. GRU

| Feature | LSTM | GRU |
| --- | --- | --- |
| **Number of Gates** | 3 (Forget, Input, Output) | 2 (Reset, Update) |
| **Memory States** | Dual ($c_t$ and $h_t$) | Single ($h_t$) |
| **Computational Efficiency** | Lower (Higher Parameter Count) | Higher (Fewer Parameters) |
| **Training Speed** | Slower Convergence | Faster Convergence |
| **Long-Sequence Perf.** | Superior on highly complex tasks | Comparable on most standard tasks |

---

### 5. Advanced Frameworks: Bidirectional Models and Seq2Seq

Sophisticated NLP and forecasting require handling future context and variable-length mappings.

#### Bidirectional RNN (Bi-RNN)

Unidirectional models only capture past context. In linguistic tasks like "The man who ___ movies is a critic," the word "movies" is essential to predict "reviews." Bi-RNNs solve this by concatenating forward ($\vec{h}_t$) and backward ($\gets_h$) hidden states, providing a comprehensive representation of the full context.

#### Encoder-Decoder (Seq2Seq) Framework

Standard RNNs require matched input/output lengths. Seq2Seq models utilize:

* **Encoder:** Compresses the input into a Fixed-Size Context Vector.
* **Decoder:** Initialized with the context vector (often concatenated with decoder inputs), it generates an output sequence token-by-token.

> **Modern Context:** The "Fixed-Size Context Vector" created a severe bottleneck for long sequences, which ultimately led to the invention of the **Attention Mechanism**. Attention allowed the decoder to look back at *all* encoder states dynamically, paving the way for modern Transformers (like GPT and BERT) which have largely superseded pure RNN architectures.

---

### 6. Quantitative Evaluation and Inference Strategies

#### Metrics and Numerical Examples

* **Perplexity (PPL):** Measures "surprise"; lower values indicate superior language modeling.

$$\text{PPL} = \exp\left(-\frac{1}{T} \sum_{t=1}^T \log P(x_t | x_{<t})\right)$$


* **BLEU Score:** Evaluates n-gram precision with a Brevity Penalty (BP):

$$\text{BLEU} = \exp\left(\min\left(0, 1 - \frac{len_{label}}{len_{pred}}\right)\right) \cdot \left(\prod_{n=1}^{k} p_n\right)^{1/k}$$



*Numerical Example:* Predicting "A B" for target "A B C D E F" gives unigram/bigram precisions of 1.0. However, the brevity penalty $BP = \exp(1 - 6/2) = \exp(-2) \approx 0.135$ correctly reduces the final score from 1.0 to 0.135, penalizing the truncated output.

#### Sequence Search Strategies

* **Greedy Search:** Fast but sub-optimal, choosing the local maximum at each step.
* **Beam Search:** A compromise maintaining $k$ (Beam Size) candidate sequences.
* **Score Normalization:** To prevent bias toward short sequences, the score is normalized by length $L$ with hyperparameter $\alpha$:

$$\text{Score} = \frac{1}{L^\alpha} \log P(y_1, \dots, y_L | c)$$



#### Practitioner's Selection Guide

* **RNN:** Use only for short sequences with extreme computational/hardware constraints.
* **LSTM:** Mandatory for complex linguistics and sequences requiring high-fidelity, very long-term dependencies.
* **GRU:** The default for most professional environments; ideal when balancing performance with limited computational resources or faster iteration requirements.
* **Bi-RNN:** Necessary for classification tasks where the entire context is available at inference.
* **Seq2Seq:** Mandatory for many-to-many tasks with non-matching lengths (e.g., Translation).