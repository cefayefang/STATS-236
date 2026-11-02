"""Statistical primitives used by the STAT 236 LastFM notebook.

No data, fitted partitions, or results are loaded at import time. Network fitting
is separate from music-based evaluation so tuning cannot inspect the outcome.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.stats import beta
from sklearn.cluster import KMeans


def leading_eigs(matrix, k, seed=236):
    """Deterministic eigenpairs ordered by magnitude, with a positive Perron vector.

    Positive power iteration refines the denominator to relative entrywise
    tolerance. A small normwise eigensolver residual alone cannot guarantee
    relative accuracy in tiny coordinates, which matters for eigenvector ratios.
    Input must be a connected, symmetric, nonnegative adjacency/Laplacian matrix.
    """
    matrix = sp.csr_matrix(matrix, dtype=np.float64)
    if not 1 <= k < matrix.shape[0]:
        raise ValueError("Require 1 <= k < number of vertices")
    vals, vecs = eigsh(matrix, k=k, which="LM", v0=np.random.default_rng(seed).normal(size=matrix.shape[0]), tol=1e-11)
    order = np.argsort(-np.abs(vals), kind="stable")
    vals, vecs = vals[order], vecs[:, order]
    v = np.full(matrix.shape[0], 1 / np.sqrt(matrix.shape[0]))
    # Shifting removes bipartite sign oscillation without changing eigenvectors.
    shift = abs(vals[0])
    for _ in range(20000):
        w = matrix @ v + shift * v
        w /= np.linalg.norm(w)
        relative = np.max(np.abs(w - v) / np.maximum(w, np.finfo(float).tiny))
        v = w
        if relative < 1e-10:
            break
    else:
        raise RuntimeError("Positive eigenvector refinement did not converge")
    perron = int(np.argmax(vals))
    vecs[:, perron] = v
    vals[perron] = float(v @ (matrix @ v))
    # The largest positive eigenvalue is the denominator even in a bipartite tie.
    order = [perron] + [j for j in range(k) if j != perron]
    return vals[order], vecs[:, order]


def reg_laplacian(adjacency, tau):
    if tau <= 0:
        raise ValueError("tau must be positive")
    d = np.asarray(adjacency.sum(axis=1)).ravel()
    scale = sp.diags(1 / np.sqrt(d + tau))
    return (scale @ adjacency @ scale).tocsr()


def relabel_by_size(labels):
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    rank = np.empty_like(counts)
    rank[np.argsort(-counts, kind="stable")] = np.arange(len(counts))
    return rank[inverse]


def kmeans(embedding, k, seed=236, n_init=20):
    if not np.isfinite(embedding).all():
        raise ValueError("Embedding contains nonfinite ratios")
    return relabel_by_size(KMeans(k, n_init=n_init, random_state=seed, algorithm="lloyd").fit_predict(embedding))


def score_ratios(vecs, k, truncate=True):
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = vecs[:, 1:k] / vecs[:, [0]]
    if not np.isfinite(ratios).all():
        raise ValueError("Nonfinite eigenvector ratios; check the Perron vector")
    return np.clip(ratios, -np.log(len(vecs)), np.log(len(vecs))) if truncate else ratios


def scoreplus_M(vals, k, t=0.1):
    # Paper section 3.1 uses the signed ratio after magnitude ordering. The
    # graph-only K selection is a separate magnitude-eigengap heuristic.
    return k + 1 if 1 - vals[k] / vals[k - 1] <= t else k


def scoreplus_labels(vals, vecs, k, t=0.1, seed=236, truncate=True, n_init=20):
    m = scoreplus_M(vals, k, t)
    eta = vecs[:, :m] * vals[:m]
    return kmeans(score_ratios(eta, m, truncate=truncate), k, seed, n_init)


def jaccard_matrix(features):
    """Float64 Jaccard matrix; zero similarity for empty unions, zero diagonal."""
    x = sp.csr_matrix(features, dtype=np.float64)
    x.eliminate_zeros()
    if np.any(x.data != 1):
        raise ValueError("Jaccard requires a binary incidence matrix")
    similarity = (x @ x.T).toarray()
    counts = np.asarray(x.sum(axis=1)).ravel()
    # Row blocks avoid simultaneously allocating another n x n float64 array.
    for lo in range(0, len(counts), 512):
        block = similarity[lo:lo + 512]
        union = counts[lo:lo + 512, None] + counts[None, :] - block
        np.divide(block, union, out=block, where=union > 0)
        block[union == 0] = 0
    np.fill_diagonal(similarity, 0)
    return similarity


def onehot(labels):
    labels = np.asarray(labels, dtype=int)
    if labels.min() != 0 or not np.array_equal(np.unique(labels), np.arange(labels.max() + 1)):
        raise ValueError("Community labels must be contiguous from zero")
    return np.eye(labels.max() + 1, dtype=np.float64)[labels]


def block_statistics(w, sizes):
    sizes = np.asarray(sizes, dtype=float)
    valid = sizes >= 2
    if len(sizes) < 2 or not valid.any():
        raise ValueError("Need multiple communities and at least one nonsingleton")
    within = np.divide(np.diag(w), sizes * (sizes - 1), out=np.full(len(sizes), np.nan), where=valid)
    between = w / np.outer(sizes, sizes)
    same = within[valid].mean()
    diff = between[np.triu_indices(len(sizes), 1)].mean()
    same_count = (sizes * (sizes - 1)).sum()
    n = sizes.sum()
    same_pair = np.trace(w) / same_count
    diff_pair = (w.sum() - np.trace(w)) / (n * (n - 1) - same_count)
    return dict(T=float(same - diff), S_same=float(same), S_diff=float(diff), ratio=float(same / diff),
                T_pair=float(same_pair - diff_pair), pair_S_same=float(same_pair), pair_S_diff=float(diff_pair),
                pair_ratio=float(same_pair / diff_pair), within=within, between=between, sizes=sizes)


def T_stat(similarity, labels, full=False):
    z = onehot(labels)
    result = block_statistics(z.T @ (similarity @ z), z.sum(axis=0))
    return result if full else result["T"]


def T_pairs(similarity, labels, full=False):
    result = T_stat(similarity, labels, full=True)
    return dict(T=result["T_pair"], S_same=result["pair_S_same"], S_diff=result["pair_S_diff"], ratio=result["pair_ratio"]) if full else result["T_pair"]


def make_strata(*variables, q=5):
    """Quantile bins without arbitrarily splitting tied covariate values by ID."""
    bins = []
    for v in variables:
        v = np.asarray(v)
        if not np.isfinite(v).all():
            raise ValueError("Stratification covariates must be finite")
        if len(np.unique(v)) == 1:
            bins.append(np.zeros(len(v), dtype=int))
        else:
            bins.append(np.asarray(pd.qcut(v, q, labels=False, duplicates="drop"), dtype=int))
    if not bins or len({len(b) for b in bins}) != 1:
        raise ValueError("Stratification variables must have matching lengths")
    return np.unique(np.column_stack(bins), axis=0, return_inverse=True)[1]


def combine_strata(*codes):
    return np.unique(np.column_stack(codes), axis=0, return_inverse=True)[1]


def strata_diagnostics(strata, labels, **covariates):
    rows = []
    for s in np.unique(strata):
        members = np.flatnonzero(strata == s)
        counts = np.bincount(np.asarray(labels)[members])
        size = len(members)
        # Probability a uniformly shuffled label differs from the original.
        row = dict(stratum=int(s), size=size, distinct_communities=int((counts > 0).sum()),
                   expected_changed_fraction=float(1 - np.square(counts / size).sum()))
        for name, values in covariates.items():
            v = np.asarray(values)[members]
            row.update({name + "_min":float(v.min()), name + "_max":float(v.max()), name + "_mean":float(v.mean())})
        rows.append(row)
    return pd.DataFrame(rows)


def binomial_interval(successes, trials, alpha=0.05):
    """Clopper-Pearson interval, used only for Monte Carlo proportions."""
    lo = 0.0 if successes == 0 else beta.ppf(alpha / 2, successes, trials - successes + 1)
    hi = 1.0 if successes == trials else beta.ppf(1 - alpha / 2, successes + 1, trials - successes)
    return float(lo), float(hi)


def summarize_null(observed, null):
    null = np.asarray(null, dtype=float)
    if len(null) < 2:
        raise ValueError("At least two Monte Carlo draws are required")
    # Count numerical ties as exceedances, never as evidence against the null.
    tolerance = 100 * np.finfo(float).eps * max(abs(observed), 1.0)
    exceedances = int(np.count_nonzero(null >= observed - tolerance))
    lo, hi = binomial_interval(exceedances, len(null))
    mean, sd = float(null.mean()), float(null.std(ddof=1))
    quantiles = np.quantile(null, [0.025, 0.975])
    return dict(T=float(observed), null=null, B=len(null), exceedances=exceedances,
                p=(1 + exceedances) / (len(null) + 1), mc_tail_ci_low=lo, mc_tail_ci_high=hi,
                null_mean=mean, null_sd=sd, null_mean_mcse=sd / np.sqrt(len(null)),
                null_q025=float(quantiles[0]), null_q975=float(quantiles[1]),
                excess_T=float(observed - mean), z=float((observed - mean) / sd) if sd > 0 else np.nan)


def perm_test(similarity, labels, strata, B, seed=236, stat=T_stat):
    """Uniform independent within-stratum label permutations, conditional on sizes.

    This is valid under the *joint invariance* null for record assignments within
    strata, not merely a null about equality of means. Coarse bins approximate
    adjustment; observational exchangeability remains a substantive assumption.
    The pair-weighted sensitivity uses the same permutations at no extra product.
    """
    labels = np.asarray(labels, dtype=int)
    if len(labels) != len(strata) or similarity.shape != (len(labels), len(labels)):
        raise ValueError("Similarity, labels, and strata must describe the same users")
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    z = onehot(labels)
    sizes = z.sum(axis=0)
    observed = block_statistics(z.T @ (similarity @ z), sizes)
    null, pair_null = np.empty(B), np.empty(B)
    rows = np.arange(len(labels))
    for b in range(B):
        permuted = labels.copy()
        for group in groups:
            permuted[group] = labels[rng.permutation(group)]
        z.fill(0)
        z[rows, permuted] = 1
        result = block_statistics(z.T @ (similarity @ z), sizes)
        null[b], pair_null[b] = result["T"], result["T_pair"]
    equal = summarize_null(observed["T"], null)
    pairs = summarize_null(observed["T_pair"], pair_null)
    if stat is T_pairs:
        return pairs
    if stat is not T_stat:
        raise ValueError("Unsupported statistic")
    equal["pair_result"] = pairs
    return equal


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues, kind="stable")
    adjusted = np.minimum(1, np.maximum.accumulate(pvalues[order] * np.arange(len(pvalues), 0, -1)))
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result
