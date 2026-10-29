"""NumPy-only neural network on binned features (entity embeddings + MLP).

Every (feature, bin) pair has a learned embedding vector (bin 0 = missing), so
categorical survey answers, ordinal codes and missingness are all handled the
same way.  The concatenated embeddings feed a 2-hidden-layer ReLU MLP trained
with Adam on the binary cross-entropy (optionally class-weighted).
"""

import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class EmbedMLP:
    def __init__(
        self,
        dim=4,
        hidden=(256, 64),
        lr=2e-3,
        epochs=8,
        batch=1024,
        l2=1e-5,
        dropout=0.1,
        seed=0,
        nb=65,
        l2_emb=0.0,
        ordinal=False,
        feat_drop=0.0,
    ):
        self.__dict__.update(locals())
        del self.__dict__["self"]

    def _init(self, F):
        rng = np.random.default_rng(self.seed)
        self.rng = rng
        D = F * self.dim + (F if self.ordinal else 0)
        sizes = [D, *self.hidden, 1]
        self.params = {
            "E": (rng.standard_normal((F * self.nb, self.dim)) * 0.05).astype(
                np.float32
            )
        }
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            self.params[f"W{i}"] = (
                rng.standard_normal((a, b)) * np.sqrt(2.0 / a)
            ).astype(np.float32)
            self.params[f"b{i}"] = np.zeros(b, np.float32)
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.t = 0
        self.offsets = (np.arange(F) * self.nb).astype(np.int64)

    def _forward(self, Bb, train=False):
        idx = Bb.astype(np.int64) + self.offsets[None, :]  # rows x F
        h = self.params["E"][idx].reshape(len(Bb), -1)
        if (
            self.ordinal
        ):  # bin index scaled to [0,1] keeps the ordering of numeric answers
            h = np.concatenate([h, Bb.astype(np.float32) * self.inv_nbins], 1)
        cache = [(idx, None, h)]
        L = len(self.hidden) + 1
        for i in range(L):
            z = h @ self.params[f"W{i}"] + self.params[f"b{i}"]
            if i < L - 1:
                a = np.maximum(z, 0)
                mask = None
                if train and self.dropout > 0:
                    mask = (self.rng.random(a.shape) > self.dropout).astype(
                        np.float32
                    ) / (1 - self.dropout)
                    a = a * mask
                cache.append((h, z, mask))
                h = a
            else:
                cache.append((h, z, None))
        return z[:, 0], cache

    def _backward(self, dz, cache):
        grads = {}
        L = len(self.hidden) + 1
        d = dz[:, None].astype(np.float32)
        for i in reversed(range(L)):
            h, z, mask = cache[i + 1]
            grads[f"W{i}"] = h.T @ d
            grads[f"b{i}"] = d.sum(0)
            d = d @ self.params[f"W{i}"].T
            if i > 0:
                _, zp, maskp = cache[i]
                if maskp is not None:
                    d = d * maskp
                d = d * (zp > 0)
        idx = cache[0][0]
        if self.ordinal:
            d = d[:, : idx.shape[1] * self.dim]
        flat = idx.ravel()
        dd = np.ascontiguousarray(d).reshape(-1, self.dim)
        size = len(self.params["E"])
        gE = np.stack(
            [
                np.bincount(flat, weights=dd[:, k], minlength=size)
                for k in range(self.dim)
            ],
            1,
        ).astype(np.float32)
        grads["E"] = gE
        return grads

    def _adam(self, grads):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k, g in grads.items():
            g = g + (self.l2_emb if k == "E" else self.l2) * self.params[k]
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh = self.m[k] / (1 - b1**self.t)
            vh = self.v[k] / (1 - b2**self.t)
            self.params[k] -= (self.lr * mh / (np.sqrt(vh) + eps)).astype(np.float32)

    def fit(self, B, y, pos_weight=1.0, callback=None):
        """B: rows x F uint8 bins."""
        self.inv_nbins = (1.0 / np.maximum(B.max(0), 1)).astype(np.float32)
        self._init(B.shape[1])
        n = len(y)
        y = y.astype(np.float32)
        steps = self.epochs * ((n + self.batch - 1) // self.batch)
        step = 0
        base_lr = self.lr
        for ep in range(self.epochs):
            perm = self.rng.permutation(n)
            for s in range(0, n, self.batch):
                ib = perm[s : s + self.batch]
                Bb = B[ib]
                if (
                    self.feat_drop > 0
                ):  # input augmentation: randomly hide answers (-> 'missing' bin)
                    Bb = np.where(
                        self.rng.random(Bb.shape) < self.feat_drop, 0, Bb
                    ).astype(np.uint8)
                z, cache = self._forward(Bb, train=True)
                p = _sigmoid(z)
                w = np.where(y[ib] == 1, pos_weight, 1.0)
                dz = (p - y[ib]) * w / len(ib)
                self.lr = base_lr * 0.5 * (1 + np.cos(np.pi * step / steps))
                step += 1  # cosine decay
                self._adam(self._backward(dz, cache))
            if callback:
                callback(self, ep)
        self.lr = base_lr
        return self

    def decision_function(self, B, chunk=20000):
        return np.concatenate(
            [self._forward(B[i : i + chunk])[0] for i in range(0, len(B), chunk)]
        )


def nn_to_arrays(m, prefix):
    out = {prefix + k: v for k, v in m.params.items()}
    out[prefix + "inv_nbins"] = m.inv_nbins
    out[prefix + "cfg"] = np.array(
        [m.dim, len(m.hidden), *m.hidden, m.nb, int(m.ordinal)]
    )
    return out


def nn_from_arrays(d, prefix):
    c = d[prefix + "cfg"]
    nh = int(c[1])
    m = EmbedMLP(
        dim=int(c[0]),
        hidden=tuple(int(v) for v in c[2 : 2 + nh]),
        nb=int(c[2 + nh]),
        ordinal=bool(c[3 + nh]),
    )
    m.params = {
        k[len(prefix) :]: d[k]
        for k in d.files
        if k.startswith(prefix)
        and k[len(prefix) :] in ["E"] + [f"{a}{i}" for a in "Wb" for i in range(nh + 1)]
    }
    m.inv_nbins = d[prefix + "inv_nbins"]
    m.offsets = (np.arange(len(m.inv_nbins)) * m.nb).astype(np.int64)
    return m
