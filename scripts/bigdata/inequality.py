"""Step D6 helpers: weighted Gini and Theil T (with its between/within-group split) on per-cell access results.

Every cell is weighted by its children of the level's ages (w); x is the cell's value (places per child A, or
distance to the nearest school in km).

Gini at 140 M cells. Sorting 140 M values on one machine is avoided: Spark groups the cells into geometric value
bins (each bin spans a factor of 1.0001, so values inside a bin differ by at most 0.01 %; x = 0 has its own bin)
and returns, per bin, the sum of w and of w*x. The Gini is then computed on the sorted bins with the Lorenz
trapezoid rule. Ties are exact under this rule, so the only error comes from the order of values inside a bin
(checked against an exact sort on Kerala in notebook 27).

Theil T = sum_i (w_i/W)(x_i/mu) ln(x_i/mu), mu = X/W, X = sum_i w_i x_i (0 ln 0 = 0). With groups g:
T = sum_g s_g T_g (within) + sum_g s_g ln(mu_g/mu) (between), s_g = X_g / X. It needs only the additive sums
W, X and L = sum_i w_i x_i ln x_i, so groups can be summed at sub-district level and rolled up exactly.
"""

import numpy as np
import pandas as pd

BIN_RATIO = 1.0001
ZERO_BIN = -(2 ** 31)  # bin of x <= 0


def bin_sql(x):
    """Spark SQL: geometric bin of x (int)."""
    return f"CASE WHEN {x} > 0 THEN CAST(floor(ln({x}) / ln({BIN_RATIO!r})) AS INT) ELSE {ZERO_BIN} END"


def gini(b):
    """Weighted Gini from rows with columns bin (or the value itself), W, X; rows with the same bin are merged."""
    g = b.groupby("bin", sort=True)[["W", "X"]].sum()
    W, X = g["W"].sum(), g["X"].sum()
    if W <= 0 or X <= 0:
        return np.nan
    s = g["X"].cumsum().to_numpy()
    prev = np.concatenate([[0.0], s[:-1]])
    return float(1 - ((g["W"].to_numpy() / W) * (prev + s) / X).sum())


def gini_exact(x, w):
    """Weighted Gini on raw values (for the check): every value is its own bin."""
    return gini(pd.DataFrame({"bin": x, "W": w, "X": w * x}))


def theil(df, group):
    """Theil T split by `group` from additive sums W, X, L per row. Returns (total, between, within, per_group)."""
    g = df.groupby(group)[["W", "X", "L"]].sum()
    g = g[g["W"] > 0]
    W, X, L = g["W"].sum(), g["X"].sum(), g["L"].sum()
    if X <= 0:
        return np.nan, np.nan, np.nan, g.assign(T=np.nan)
    mu = X / W
    total = L / X - np.log(mu)
    pos = g["X"] > 0
    mu_g = g["X"] / g["W"]
    T_g = pd.Series(np.nan, index=g.index)
    T_g[pos] = g.loc[pos, "L"] / g.loc[pos, "X"] - np.log(mu_g[pos])
    s = g["X"] / X
    within = float((s[pos] * T_g[pos]).sum())
    between = float((s[pos] * np.log(mu_g[pos] / mu)).sum())
    return float(total), between, within, g.assign(T=T_g)


if __name__ == "__main__":  # self-check on made-up data
    rng = np.random.default_rng(1)
    x = np.concatenate([np.zeros(50), rng.lognormal(0, 1, 2000)])
    w = rng.uniform(0.1, 3, len(x))
    # Gini: exact formula vs the textbook pairwise definition sum_ij w_i w_j |x_i - x_j| / (2 W^2 mu)
    pair = (w[:, None] * w[None, :] * np.abs(x[:, None] - x[None, :])).sum() / (2 * w.sum() ** 2 * (w * x).sum() / w.sum())
    assert abs(gini_exact(x, w) - pair) < 1e-9, (gini_exact(x, w), pair)
    # binned Gini (as Spark does it) is within 1e-4 of the exact one
    bins = np.where(x > 0, np.floor(np.log(np.where(x > 0, x, 1)) / np.log(BIN_RATIO)), ZERO_BIN)
    assert abs(gini(pd.DataFrame({"bin": bins, "W": w, "X": w * x})) - gini_exact(x, w)) < 1e-4
    assert abs(gini_exact(np.ones(10), np.ones(10))) < 1e-12  # all equal -> 0
    # Theil: between + within = total, total = direct formula, equal groups -> between 0
    grp = rng.integers(0, 7, len(x))
    xl = np.where(x > 0, x * np.log(np.where(x > 0, x, 1)), 0)
    t, b, wi, _ = theil(pd.DataFrame({"g": grp, "W": w, "X": w * x, "L": w * xl}), "g")
    mu = (w * x).sum() / w.sum()
    r = x / mu
    direct = (w / w.sum() * np.where(r > 0, r * np.log(np.where(r > 0, r, 1)), 0)).sum()
    assert abs(t - direct) < 1e-9 and abs(b + wi - t) < 1e-9
    t2, b2, _, _ = theil(pd.DataFrame({"g": [0, 0, 1, 1], "W": [1, 1, 1, 1], "X": [1, 3, 1, 3], "L": [0, 3 * np.log(3)] * 2}), "g")
    assert abs(b2) < 1e-12
    print("ok")
