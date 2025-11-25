import numpy as np

# 1. Prepare Data
text = "hello"
chars = sorted(list(set(text)))
char_to_int = {c: i for i, c in enumerate(chars)}
int_to_char = {i: c for i, c in enumerate(chars)}

vocab_size = len(chars)
hidden_size = 10
seq_length = len(text) - 1
learning_rate = 0.1

# 2. Model Parameters
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01 # Input to Hidden
Whh = np.random.randn(hidden_size, hidden_size) * 0.01 # Hidden to Hidden
Why = np.random.randn(vocab_size, hidden_size) * 0.01 # Hidden to Output
bh = np.zeros((hidden_size, 1)) # Hidden bias
by = np.zeros((vocab_size, 1)) # Output bias

def lossFun(inputs, targets, hprev):
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0
    
    # Forward pass
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size, 1))
        xs[t][inputs[t]] = 1
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh)
        ys[t] = np.dot(Why, hs[t]) + by
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # Softmax
        loss += -np.log(ps[t][targets[t], 0])
        
    # Backward pass
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])
    
    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(Why.T, dy) + dhnext
        dhraw = (1 - hs[t] * hs[t]) * dh
        dbh += dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t-1].T)
        dhnext = np.dot(Whh.T, dhraw)
        
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)
        
    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

# 3. Training Loop
print("Training...")
hprev = np.zeros((hidden_size, 1))
inputs = [char_to_int[ch] for ch in text[:-1]]
targets = [char_to_int[ch] for ch in text[1:]]

for i in range(1000):
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
    
    # Update weights
    for param, dparam in zip([Wxh, Whh, Why, bh, by], [dWxh, dWhh, dWhy, dbh, dby]):
        param -= learning_rate * dparam
        
    if i % 100 == 0:
        print(f'Iter: {i}, Loss: {loss:.4f}')

# 4. Prediction
print("\nPrediction:")
h = np.zeros((hidden_size, 1))
start_char = "h"
x = np.zeros((vocab_size, 1))
x[char_to_int[start_char]] = 1
predicted_text = start_char

for i in range(4):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    
    predicted_char = int_to_char[ix]
    predicted_text += predicted_char
    
    x = np.zeros((vocab_size, 1))
    x[ix] = 1

print(f"Input: 'h'")
print(f"Output: '{predicted_text}'")
