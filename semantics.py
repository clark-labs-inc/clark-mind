"""The math that turns local statistics into SEMANTICS (gradient-free).
-------------------------------------------------------------------------------
The wall: the count substrate captures LOCAL co-occurrence ("X near Y"), not
meaning. The fix is not more data -- it is a different OPERATION on the counts
the brain already has:

    semantics = the low-rank SPECTRAL FACTORIZATION of the co-occurrence matrix.

Build the word x context PPMI matrix from local counts, take its truncated SVD,
and the top singular directions are a SEMANTIC space: two words are close iff
they have similar context distributions -- so synonyms cluster even if they
never co-occur, and analogies (king - man + woman ~ queen) become linear
arithmetic. This is Latent Semantic Analysis, and Levy & Goldberg (2014) proved
word2vec is implicitly this same PMI factorization -- so the gradient-free SVD
yields the embeddings backprop would, no neural net. It is the SAME spectral
math as spectral.py (graph Laplacian), applied to co-occurrence.

This script proves it: RAW co-occurrence neighbours (surface) vs SVD-semantic
neighbours (meaning), on the corpus the brain already reads.

Run:  python3 semantics.py
"""
from __future__ import annotations
import re, numpy as np
np.seterr(over="ignore", invalid="ignore", divide="ignore")


def load_words(path="data/wiki_train.txt", limit=2_000_000):
    txt = open(path, errors="replace").read()[:limit].lower()
    return re.findall(r"[a-z]+", txt)


def cooccur(words, V=4000, win=4):
    from collections import Counter
    freq = Counter(words)
    vocab = [w for w, _ in freq.most_common(V)]
    idx = {w: i for i, w in enumerate(vocab)}
    C = np.zeros((len(vocab), len(vocab)), np.float32)
    ids = [idx.get(w, -1) for w in words]
    n = len(ids)
    for i in range(n):
        wi = ids[i]
        if wi < 0:
            continue
        for j in range(max(0, i - win), min(n, i + win + 1)):
            if j != i and ids[j] >= 0:
                C[wi, ids[j]] += 1.0
    return C, vocab, idx


def ppmi(C):
    """Positive pointwise mutual information -- the right matrix to factorize."""
    tot = C.sum()
    rw = C.sum(1, keepdims=True); cw = C.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        P = (C * tot) / (rw * cw)
        M = np.log(P)
    M[~np.isfinite(M)] = 0.0
    return np.maximum(M, 0.0)


def rsvd(M, k, p=10, q=2, seed=0):
    """Randomized truncated SVD -- gradient-free, fast for the top k dims."""
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((M.shape[1], k + p)).astype(np.float32)
    Y = M @ G
    for _ in range(q):
        Y = M @ (M.T @ Y)
    Q, _ = np.linalg.qr(Y)
    B = Q.T @ M
    Ub, S, _ = np.linalg.svd(B, full_matrices=False)
    return (Q @ Ub[:, :k]), S[:k]


def neighbors(E, idx, vocab, word, n=6):
    if word not in idx:
        return []
    v = E[idx[word]]
    sims = E @ v / (np.linalg.norm(E, axis=1) * (np.linalg.norm(v) + 1e-9) + 1e-9)
    order = np.argsort(-sims)
    return [vocab[i] for i in order if vocab[i] != word][:n]


def raw_neighbors(C, idx, vocab, word, n=6):
    if word not in idx:
        return []
    row = C[idx[word]].copy(); row[idx[word]] = 0
    return [vocab[i] for i in np.argsort(-row)[:n]]


def analogy(E, idx, vocab, a, b, c, n=3):
    if not all(w in idx for w in (a, b, c)):
        return []
    v = E[idx[b]] - E[idx[a]] + E[idx[c]]
    sims = E @ v / (np.linalg.norm(E, axis=1) * (np.linalg.norm(v) + 1e-9) + 1e-9)
    order = np.argsort(-sims)
    seen = {a, b, c}
    return [vocab[i] for i in order if vocab[i] not in seen][:n]


def main():
    print("SEMANTICS FROM LOCAL COUNTS via spectral factorization (no backprop)\n")
    words = load_words()
    print(f"corpus: {len(words):,} words; building co-occurrence + PPMI ...")
    C, vocab, idx = cooccur(words)
    M = ppmi(C)
    U, S = rsvd(M, k=200)
    E = U * np.sqrt(S)                                  # semantic embeddings
    print(f"factorized {M.shape[0]}x{M.shape[1]} PPMI -> {E.shape[1]}-dim "
          f"semantic space (top singular values {S[:3].round(1)})\n")

    print("RAW co-occurrence neighbours (surface) vs SVD-SEMANTIC neighbours:\n")
    for w in ["king", "water", "music", "france", "three", "war", "red", "science"]:
        if w in idx:
            raw = ", ".join(raw_neighbors(C, idx, vocab, w, 5))
            sem = ", ".join(neighbors(E, idx, vocab, w, 5))
            print(f"  {w:9s} raw: {raw:38s}\n            sem: {sem}")
    print("\nanalogies (b - a + c) in the semantic space:")
    for a, b, c in [("man", "woman", "king"), ("france", "paris", "england"),
                    ("one", "two", "three")]:
        r = analogy(E, idx, vocab, a, b, c)
        print(f"  {a}:{b} :: {c}:?  ->  {', '.join(r) if r else '(oov)'}")
    print("\nThe raw rows give whatever sits NEXT TO a word (often surface/"
          "function words);\nthe SVD space groups words by MEANING -- same local "
          "counts, one eigendecomposition.\nThat operation, not more data, is "
          "what crosses 'statistics -> semantics'.")


if __name__ == "__main__":
    main()
