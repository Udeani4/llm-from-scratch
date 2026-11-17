"""1.2 Self-attention from first principles on a tiny example (NumPy only).
We use T=3 tokens, d_model=4, d_k=d_v=2, single-head.
This script prints intermediate tensors so you can trace the math.

Dimensions summary (single head)
--------------------------------
X:          (B=1, T=3, d_model=4)
Wq/Wk/Wv:   (d_model=4, d_k=2)
Q,K,V:      (1, 3, 2)
Scores:     (1, 3, 3)   = Q @ K^T
Weights:    (1, 3, 3)   = softmax over last dim
Output:     (1, 3, 2)   = Weights @ V
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)

# Toy inputs (batch=1, seq=3, d_model=4)
# Each row is one token's embedding vector (in a real LLM these come from an embedding lookup table).
X = np.array([[[0.1, 0.2, 0.3, 0.4],
               [0.5, 0.4, 0.3, 0.2],
               [0.0, 0.1, 0.0, 0.1]]], dtype=np.float32)

# Weight matrices (learned in real models). We fix numbers for determinism.
# Wq turns each token into a "query" (what am I looking for?).
Wq = np.array([[ 0.2, -0.1],
               [ 0.0,  0.1],
               [ 0.1,  0.2],
               [-0.1,  0.0]], dtype=np.float32)
# Wk turns each token into a "key" (what do I contain / how should others find me?).
Wk = np.array([[ 0.1,  0.1],
               [ 0.0, -0.1],
               [ 0.2,  0.0],
               [ 0.0,  0.2]], dtype=np.float32)
# Wv turns each token into a "value" (the information I hand over if someone attends to me).
Wv = np.array([[ 0.1,  0.0],
               [-0.1,  0.1],
               [ 0.2, -0.1],
               [ 0.0,  0.2]], dtype=np.float32)

# Project to Q, K, V
# The same weight matrix is applied to every token independently (matmul over the last dim).
Q = X @ Wq  # (1,3,2)
K = X @ Wk  # (1,3,2)
V = X @ Wv  # (1,3,2)

print("Q shape:", Q.shape, "\nQ=\n", Q[0])
print("K shape:", K.shape, "\nK=\n", K[0])
print("V shape:", V.shape, "\nV=\n", V[0])

# Scaled dot-products
# Dividing by sqrt(d_k) keeps dot products from growing with dimension; large scores would push
# softmax into a near one-hot regime where gradients vanish.
scale = 1.0 / np.sqrt(Q.shape[-1])
# scores[i, j] = how well query i matches key j. Transpose swaps K's last two axes -> (1,2,3).
attn_scores = (Q @ K.transpose(0,2,1)) * scale  # (1,3,3)

# Causal mask (upper triangle set to -inf so softmax->0)
# True above the diagonal = "future" positions that token i is not allowed to see.
mask = np.triu(np.ones((1,3,3), dtype=bool), k=1)
# Use a huge negative number instead of -inf; exp(-1e9) is effectively 0 after softmax.
attn_scores = np.where(mask, -1e9, attn_scores)

# Softmax over last dim
# Subtracting the row max first is the standard numerically-stable softmax trick (avoids overflow).
weights = np.exp(attn_scores - attn_scores.max(axis=-1, keepdims=True))
# Normalise so each row sums to 1 -> row i is a probability distribution over which tokens to read.
weights = weights / weights.sum(axis=-1, keepdims=True)
print("Weights shape:", weights.shape, "\nAttention Weights (causal)=\n", weights[0])

# Weighted sum of V
# Each output token = weighted average of the value vectors it attended to.
out = weights @ V  # (1,3,2)
print("Output shape:", out.shape, "\nOutput=\n", out[0])