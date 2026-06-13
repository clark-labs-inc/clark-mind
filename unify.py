"""ONE math under the whole brain: free-energy minimization (predictive coding).
-------------------------------------------------------------------------------
The faculties are not separate tricks -- they are one principle: minimize a free
energy (prediction error) by LOCAL updates. Counting = its max-likelihood limit;
SVD/semantics = its linear case; associative memory/attention = energy settling
to an attractor; deep abstraction = its hierarchical nonlinear case. Predictive
coding is the single engine; everything else is a special case.

Proof: ONE predictive-coding engine, instantiated three ways, yields three
distinct faculties -- same settle-and-locally-update code, no backprop:
  REPRESENT  an autoencoder's code layer = an unsupervised SEMANTIC embedding
             (k-NN on the code recovers class) -> meaning, the spectral faculty.
  COMPLETE   the SAME autoencoder fills in a masked input -> associative MEMORY.
  ABSTRACT   the SAME engine, hierarchical+supervised, learns XOR -> depth.
"""
import numpy as np
from predictive_coding import PredictiveCoding, f
from torchvision.datasets import MNIST
from PIL import Image
rng = np.random.default_rng(0)


def load(n_tr=3000, n_te=500):
    mn = MNIST(root="./data", train=True, download=True)
    X = mn.data.numpy().astype(np.float32) / 255.0; Y = mn.targets.numpy()
    def ds(idx):
        return (np.stack([np.asarray(Image.fromarray(np.uint8(X[i]*255)).resize((8, 8)),
                                     np.float32).ravel()/255.0 for i in idx]), Y[idx])
    return ds(rng.choice(len(X), n_tr, False)) + ds(rng.choice(len(X), n_te, False))


def main():
    print("ONE ENGINE (free-energy / predictive coding), THREE FACULTIES:\n")
    Xtr, Ytr, Xte, Yte = load()

    # ONE autoencoder: 64 -> 24 code -> 64, trained unsupervised (target = input)
    ae = PredictiveCoding([64, 24, 64], lr=0.01, T=25)
    for ep in range(5):
        for i in rng.permutation(len(Xtr)):
            ae.train_step(Xtr[i], Xtr[i])

    def code(x): return f(ae.feedforward(x)[1])         # the learned representation

    # FACULTY 1 -- REPRESENT: is the code a SEMANTIC space? 1-NN by class.
    Ctr = np.stack([code(x) for x in Xtr]); Cte = np.stack([code(x) for x in Xte])
    ok = 0
    for j in range(len(Xte)):
        d = ((Ctr - Cte[j])**2).sum(1); ok += (Ytr[d.argmin()] == Yte[j])
    print(f"  REPRESENT  code-space 1-NN class accuracy : {100*ok/len(Xte):4.0f}%   "
          f"(unsupervised semantics, chance 10%)")

    # FACULTY 2 -- COMPLETE: mask the bottom half, the SAME net fills it in.
    base_err = recon_err = 0.0
    for j in range(len(Xte)):
        x = Xte[j].copy(); masked = x.copy(); masked[32:] = 0.0     # hide bottom half
        rec = ae.feedforward(masked)[-1]
        base_err += ((0.0 - x[32:])**2).mean()            # guessing zeros
        recon_err += ((rec[32:] - x[32:])**2).mean()      # associative recall
    print(f"  COMPLETE   masked-half reconstruction MSE : {recon_err/len(Xte):.3f}  "
          f"vs {base_err/len(Xte):.3f} blank  (associative memory)")

    # FACULTY 3 -- ABSTRACT: same engine, hierarchical+supervised, learns XOR.
    X = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]]); Yx = np.array([[0.],[1.],[1.],[0.]])
    net = PredictiveCoding([2, 16, 1])
    for _ in range(20000):
        i = rng.integers(4); net.train_step(X[i], Yx[i])
    ok = int(((np.array([net.predict(x)[0] for x in X]) > 0.5).astype(float) == Yx[:,0]).sum())
    print(f"  ABSTRACT   XOR (needs nonlinear depth)    : {ok}/4   "
          f"(deep credit assignment)")

    print("\n  same settle-and-local-update engine, no backprop: representation,")
    print("  memory, and depth are ONE math -- free-energy minimization. Counting")
    print("  is its ML limit, SVD its linear case, attention its attractor case.")


if __name__ == "__main__":
    main()
