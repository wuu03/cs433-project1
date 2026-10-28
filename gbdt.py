"""NumPy-only histogram gradient-boosted decision trees (leaf-wise growth).

Design (LightGBM-style, written from scratch with NumPy only):
  * every feature is pre-binned into <= 64 uint8 bins; bin 0 is reserved for
    missing values so the tree can learn where missing values should go;
  * trees grow leaf-wise (always split the leaf with the largest loss reduction)
    up to `num_leaves`; histograms of the larger child are obtained by
    subtracting the smaller child's histogram from the parent's;
  * second-order (Newton) leaf values with L2 regularisation `lam`;
  * row bagging and per-tree feature subsampling for variance reduction.
"""
import numpy as np


class Binner:
    """Maps raw float features to small integer bins (0 = missing)."""

    def __init__(self, max_bins=63):
        self.max_bins = max_bins

    def fit(self, x):
        self.thresholds = []
        for j in range(x.shape[1]):
            v = x[:, j]
            v = v[np.isfinite(v)]
            u = np.unique(v)
            if len(u) <= 1:
                t = np.array([], dtype=np.float64)
            elif len(u) <= self.max_bins:
                t = (u[:-1] + u[1:]) / 2.0  # one bin per distinct value
            else:
                q = np.quantile(v, np.linspace(0, 1, self.max_bins + 1)[1:-1])
                t = np.unique(q)
            self.thresholds.append(t.astype(np.float64))
        return self

    def transform(self, x):
        """Returns the transposed (features x rows) uint8 bin matrix."""
        out = np.zeros((x.shape[1], x.shape[0]), dtype=np.uint8)
        for j, t in enumerate(self.thresholds):
            col = x[:, j]
            b = 1 + np.searchsorted(t, col, side="right")
            b[~np.isfinite(col)] = 0
            out[j] = b
        return out


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class GBDT:
    def __init__(self, n_trees=500, lr=0.03, num_leaves=31, min_child=100, lam=5.0,
                 subsample=0.8, colsample=0.5, min_gain=0.0, pos_weight=1.0, seed=0,
                 max_bins=64, cat_features=(), goss=None, min_hess=None):
        self.__dict__.update(locals())
        del self.__dict__["self"]

    # ---------- histogram + split search -------------------------------------------------
    def _hist(self, Bsub, idx, g, h):
        nb = self.max_bins + 1
        F = Bsub.shape[0]
        G = np.empty((F, nb)); H = np.empty((F, nb)); C = np.empty((F, nb))
        gi = g[idx]; hi = h[idx]
        sub = Bsub[:, idx]
        for f in range(F):
            r = sub[f].astype(np.intp)  # cast once; bincount would cast for every call
            G[f] = np.bincount(r, weights=gi, minlength=nb)
            H[f] = np.bincount(r, weights=hi, minlength=nb)
            if self.min_hess is None:
                C[f] = np.bincount(r, minlength=nb)
        if self.min_hess is not None:
            C = H  # child-size constraint is then expressed in hessian units
        return G, H, C

    def _best_split(self, G, H, C, catmask):
        """Vectorised search over all (feature, threshold, missing-direction) candidates.

        Ordinal features: left = {bins 1..b} (+ missing bin optionally).
        Categorical features: bins sorted by gradient ratio first (Fisher's trick).
        Returns (gain, feature, order-or-None, cut, missing_left).
        """
        lam = self.lam
        mc = self.min_child if self.min_hess is None else self.min_hess
        Gt = G.sum(1, keepdims=True); Ht = H.sum(1, keepdims=True); Ct = C.sum(1, keepdims=True)
        parent = Gt[:, 0] ** 2 / (Ht[:, 0] + lam)
        G1, H1, C1 = G[:, 1:], H[:, 1:], C[:, 1:]
        order = None
        if catmask.any():
            # sort categorical bins by g/h so a prefix split = best subset split
            ratio = np.where(C1 > 0, G1 / (H1 + lam), np.inf)
            order = np.tile(np.arange(G1.shape[1]), (G1.shape[0], 1))
            order[catmask] = np.argsort(ratio[catmask], axis=1, kind="stable")
            G1 = np.take_along_axis(G1, order, 1); H1 = np.take_along_axis(H1, order, 1); C1 = np.take_along_axis(C1, order, 1)
        gl = np.cumsum(G1, 1)[:, :-1]; hl = np.cumsum(H1, 1)[:, :-1]; cl = np.cumsum(C1, 1)[:, :-1]
        best = (-np.inf, -1, None, -1, False)
        for miss_left in (False, True):
            GL = gl + (G[:, :1] if miss_left else 0); HL = hl + (H[:, :1] if miss_left else 0); CL = cl + (C[:, :1] if miss_left else 0)
            GR = Gt - GL; HR = Ht - HL; CR = Ct - CL
            gain = GL ** 2 / (HL + lam) + GR ** 2 / (HR + lam) - parent[:, None]
            gain[(CL < mc) | (CR < mc) | (HL < 1e-3) | (HR < 1e-3)] = -np.inf
            k = np.argmax(gain)
            f, b = divmod(int(k), gain.shape[1])
            if gain[f, b] > best[0]:
                best = (float(gain[f, b]), f, None if order is None else order[f].copy(), b, miss_left)
        return best

    # ---------- training ----------------------------------------------------------------
    def fit(self, B, y, B_valid=None, init=None, init_valid=None, sample_weight=None, verbose=0):
        """B: (features x rows) uint8 bins from Binner.transform."""
        rng = np.random.default_rng(self.seed)
        F, n = B.shape
        y = y.astype(np.float64)
        w = np.ones(n) if sample_weight is None else sample_weight.astype(np.float64)
        w = w * np.where(y == 1, self.pos_weight, 1.0)
        p0 = np.sum(w * y) / np.sum(w)
        self.base = float(np.log(p0 / (1 - p0))) if init is None else 0.0
        z = np.full(n, self.base) if init is None else init.astype(np.float64).copy()
        self.trees = []
        cat_all = np.zeros(F, dtype=bool); cat_all[list(self.cat_features)] = True
        zv = None
        if B_valid is not None:
            zv = np.full(B_valid.shape[1], self.base) if init_valid is None else init_valid.astype(np.float64).copy()
            self.valid_trace = []
        for t in range(self.n_trees):
            p = _sigmoid(z)
            g = (p - y) * w; h = np.maximum(p * (1 - p), 1e-6) * w
            feats = np.sort(rng.choice(F, max(1, int(round(self.colsample * F))), replace=False))
            if self.goss is not None:
                # gradient-based one-side sampling: keep the largest |g|, subsample the rest
                a, b = self.goss
                ag = np.abs(g); cut = np.quantile(ag, 1 - a)
                big = ag >= cut
                small = (~big) & (rng.random(n) < b / (1 - a))
                rows = np.flatnonzero(big | small)
                amp = np.where(small, (1 - a) / b, 1.0)
                g = g * amp; h = h * amp
            else:
                rows = np.flatnonzero(rng.random(n) < self.subsample) if self.subsample < 1 else np.arange(n)
            Bsub = B[feats]
            tree = self._grow(Bsub, feats, cat_all[feats], rows, g, h)
            self.trees.append(tree)
            z += self.lr * self._predict_tree(tree, B)
            if zv is not None:
                zv += self.lr * self._predict_tree(tree, B_valid)
                if verbose and (t + 1) % verbose == 0:
                    self.valid_trace.append((t + 1, zv.copy()))
        self.train_margin = z
        return self

    def _grow(self, Bsub, feats, catmask, rows, g, h):
        lam = self.lam
        # node storage
        feat, left, right, value, cut_bins = [], [], [], [], []
        def new_node(G, H):
            feat.append(-1); left.append(-1); right.append(-1)
            value.append(-G / (H + lam)); cut_bins.append(None)
            return len(feat) - 1
        G, H, C = self._hist(Bsub, rows, g, h)
        root = new_node(G[0].sum(), H[0].sum())
        leaves = {root: (rows, (G, H, C), self._best_split(G, H, C, catmask))}
        n_leaves = 1
        while n_leaves < self.num_leaves:
            cand = [(v[2][0], k) for k, v in leaves.items() if v[2][1] >= 0 and v[2][0] > self.min_gain]
            if not cand:
                break
            _, node = max(cand)
            idx, (Gp, Hp, Cp), (gain, f, order, b, miss_left) = leaves.pop(node)
            nb1 = Gp.shape[1] - 1
            left_bins = np.zeros(Gp.shape[1], dtype=bool)
            sel = (order[: b + 1] if order is not None else np.arange(b + 1)) + 1
            left_bins[sel] = True
            left_bins[0] = miss_left
            go_left = left_bins[Bsub[f, idx]]
            li, ri = idx[go_left], idx[~go_left]
            small, large = (li, ri) if len(li) <= len(ri) else (ri, li)
            Gs, Hs, Cs = self._hist(Bsub, small, g, h)
            Gl_, Hl_, Cl_ = Gp - Gs, Hp - Hs, Cp - Cs
            hs_small = (Gs, Hs, Cs); hs_large = (Gl_, Hl_, Cl_)
            hl_, hr_ = (hs_small, hs_large) if len(li) <= len(ri) else (hs_large, hs_small)
            feat[node] = int(feats[f]); cut_bins[node] = left_bins
            ln = new_node(hl_[0][0].sum(), hl_[1][0].sum()); rn = new_node(hr_[0][0].sum(), hr_[1][0].sum())
            left[node] = ln; right[node] = rn
            for nn, ii, hh in ((ln, li, hl_), (rn, ri, hr_)):
                leaves[nn] = (ii, hh, self._best_split(*hh, catmask))
            n_leaves += 1
        return dict(feature=np.array(feat), left=np.array(left), right=np.array(right),
                    value=np.array(value), masks=np.array([m if m is not None else np.zeros(self.max_bins + 1, bool) for m in cut_bins]))

    @staticmethod
    def _predict_tree(tree, B):
        n = B.shape[1]
        node = np.zeros(n, dtype=np.int32)
        feat, left, right, masks = tree["feature"], tree["left"], tree["right"], tree["masks"]
        active = np.arange(n)
        while len(active):
            nd = node[active]
            f = feat[nd]
            internal = f >= 0
            active, nd, f = active[internal], nd[internal], f[internal]
            if not len(active):
                break
            go = masks[nd, B[f, active]]
            node[active] = np.where(go, left[nd], right[nd])
        return tree["value"][node]

    def decision_function(self, B, n_trees=None):
        z = np.full(B.shape[1], self.base)
        for tree in self.trees[: n_trees or len(self.trees)]:
            z += self.lr * self._predict_tree(tree, B)
        return z


# ---------------- serialisation (plain .npz, no pickle) -----------------------------
def binner_to_arrays(b):
    lens = np.array([len(t) for t in b.thresholds])
    return {"bin_thr": np.concatenate(b.thresholds) if lens.sum() else np.zeros(0), "bin_len": lens,
            "bin_max": np.array(b.max_bins)}


def binner_from_arrays(d):
    b = Binner(int(d["bin_max"]))
    b.thresholds = np.split(d["bin_thr"], np.cumsum(d["bin_len"])[:-1])
    return b


def gbdt_to_arrays(m, prefix):
    sizes = np.array([len(t["feature"]) for t in m.trees])
    cat = lambda k: np.concatenate([t[k] for t in m.trees])
    return {prefix + "sizes": sizes, prefix + "feature": cat("feature"), prefix + "left": cat("left"),
            prefix + "right": cat("right"), prefix + "value": cat("value"),
            prefix + "masks": np.packbits(cat("masks"), axis=1), prefix + "meta": np.array([m.base, m.lr, m.max_bins])}


def gbdt_from_arrays(d, prefix):
    base, lr, max_bins = d[prefix + "meta"]
    m = GBDT(lr=float(lr), max_bins=int(max_bins)); m.base = float(base)
    nb = int(max_bins) + 1
    masks = np.unpackbits(d[prefix + "masks"], axis=1)[:, :nb].astype(bool)
    cuts = np.cumsum(d[prefix + "sizes"])[:-1]
    parts = {k: np.split(d[prefix + k], cuts) for k in ("feature", "left", "right", "value")}
    mparts = np.split(masks, cuts)
    m.trees = [dict(feature=parts["feature"][i], left=parts["left"][i], right=parts["right"][i],
                    value=parts["value"][i], masks=mparts[i]) for i in range(len(mparts))]
    return m
