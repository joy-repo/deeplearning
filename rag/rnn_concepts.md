# Recurrent Neural Networks (RNN) - A Conceptual Guide

## What is an RNN?
A **Recurrent Neural Network (RNN)** is a type of artificial neural network designed to recognize patterns in sequences of data, such as text, genomes, handwriting, or the spoken word.

Unlike traditional Feedforward Neural Networks (which process inputs independently), RNNs have a "memory" which captures information about what has been calculated so far.

## Key Concept: The Loop
The core idea is that the output of a layer is fed back into the input of the same layer for the next time step.

```mermaid
graph LR
    X[Input X_t] --> H[Hidden State H_t]
    H --> O[Output Y_t]
    H -- H_t-1 --> H
```

At each time step $t$:
1.  **Input**: The network takes the current input $x_t$ and the previous hidden state $h_{t-1}$.
2.  **Process**: It calculates the new hidden state $h_t$.
3.  **Output**: It (optionally) produces an output $y_t$.

## Deep RNNs (Stacked RNNs)
You can stack multiple RNN layers on top of each other to learn more complex patterns.
**Crucial Concept**: The data flows **up** and **forward** simultaneously.

*   Layer 1 processes $x_t$ and passes its hidden state $h^{(1)}_t$ **immediately** to Layer 2.
*   Layer 2 uses $h^{(1)}_t$ as its "input" for that same time step $t$.

It does **NOT** wait for Layer 1 to finish the whole sequence.

```mermaid
graph TD
    subgraph Time Step t
        X[Input x_t] --> L1[Layer 1]
        L1 --> L2[Layer 2]
        L2 --> O[Output y_t]
    end
    subgraph Time Step t+1
        X2[Input x_t+1] --> L1_next[Layer 1]
        L1_next --> L2_next[Layer 2]
    end
    L1 -- h_t --> L1_next
    L2 -- h_t --> L2_next
```

## Trace Example: "I am the Best"
Let's look at exactly what happens at every millisecond for a **2-Layer RNN**.

### Time Step 1: Input "I"
1.  **Layer 1** receives "I" and its empty memory (hidden state $h^{(1)}_0$).
    *   It calculates **$h^{(1)}_1$** (Layer 1's memory of "I").
2.  **Layer 2** **IMMEDIATELY** receives **$h^{(1)}_1$**.
    *   It does *not* receive "I" directly, it receives Layer 1's interpretation of "I".
    *   It calculates **$h^{(2)}_1$**.
3.  **Network Output**: "am" (prediction).

### Time Step 2: Input "am"
1.  **Layer 1** receives "am" AND its previous memory **$h^{(1)}_1$**.
    *   It combines "am" + "I" (from memory) -> New memory **$h^{(1)}_2$**.
2.  **Layer 2** receives **$h^{(1)}_2$** AND its previous memory **$h^{(2)}_1$**.
    *   It calculates **$h^{(2)}_2$**.

### Summary Table
| Time Step | Input | Layer 1 Output (fed to L2) | Layer 2 Input Sources |
| :--- | :--- | :--- | :--- |
| **t=1** | "I" | Memory of "I" | Memory of "I" (from L1) + Empty State |
| **t=2** | "am" | Memory of "I" + "am" | Memory of ("I"+"am") (from L1) + Memory of ("I") (from L2 prev step) |
| **t=3** | "the" | Memory of "I"+"am"+"the" | Memory of ("I"+"am"+"the") (from L1) + Memory of ("I"+"am") (from L2 prev step) |

Notice how Layer 2 always works on the *current* output of Layer 1.


## The Math (Simplified)
$$ h_t = \tanh(W_h h_{t-1} + W_x x_t + b) $$

- $W_h$: Weight matrix for the hidden state (memory).
- $W_x$: Weight matrix for the current input.
- $\tanh$: Activation function to keep values between -1 and 1.

## Why are they useful?
They can handle inputs of varying lengths!
- **One-to-One**: Standard Neural Net (Fixed input -> Fixed output).
- **One-to-Many**: Image Captioning (Image -> Sequence of words).
- **Many-to-One**: Sentiment Analysis (Sequence of words -> Positive/Negative).
- **Many-to-Many**: Machine Translation (English sentence -> French sentence).

## The Problem: Short-term Memory
Basic RNNs suffer from the **Vanishing Gradient Problem**. As the sequence gets longer, it becomes hard for the network to learn dependencies from the beginning of the sequence.
*Solution*: **LSTM (Long Short-Term Memory)** and **GRU (Gated Recurrent Unit)** networks are advanced RNNs designed to remember long-term dependencies.
