"""
Tools for performing clustering, relying mostly on scipy and scikits-learn.

Clustering strategies

- unfold epoch -> cluster euclidean
- reduction -> distance -> cluster
- reduction -> manifold embedding -> cluster
- ica
- 

"""

import multiprocessing

import numpy as np

from scipy.spatial.distance import pdist

from sklearn.manifold import SpectralEmbedding
from sklearn.cluster import DBSCAN

from . import util


def xrange(*args):
    return list(range(*args))


def _mp_pairwise_norm(u, r=xrange, n_jobs=1):
    triu = ((i, j, u[i], u[j]) for i in r(len(u)) for j in r(len(u)) if j > i)
    with util.mpool(n_jobs) as p:
        for _ in p.imap(_mp_pairwise_norm_single, triu, 1000):
            yield _


def _mp_pairwise_norm_single(xxx_todo_changeme):
    (i, j, ui, uj) = xxx_todo_changeme
    return i, j, np.linalg.norm(ui.T.dot(uj))


def subspace_similarity(epochs, n_components=3, n_jobs=1):
    """
    Compute similarity between event subspaces, based on the norm of the
    produce of first `n_components` left singular vectors.

    """

    us = [np.linalg.svd(e)[0][:, :n_components] for e in epochs]

    dist = np.zeros((len(epochs), len(epochs)))

    for i, j, dij in _mp_pairwise_norm(us, n_jobs=n_jobs):
        dist[i, j] = dist[j, i] = dij

    return us, dist


def _mp_pairwise_epoch_corr(u, r=xrange, n_jobs=1):
    triu = ((i, j, u[i], u[j]) for i in r(len(u)) for j in r(len(u)) if j > i)
    pool = multiprocessing.Pool(n_jobs)
    with util.mpool(n_jobs) as p:
        for _ in p.imap(_mp_pairwise_epoch_corr_single, triu, 1000):
            yield _


def _mp_pairwise_epoch_corr_single(xxx_todo_changeme1):
    (i, j, ui, uj) = xxx_todo_changeme1
    return i, j, abs((ui * uj).mean(axis=0)).mean()


def temporal_similarity(epochs, n_components=3, n_jobs=1):
    """
    Compute similarity between event subspaces, based on the norm of the
    produce of first `n_components` left singular vectors.

    """

    dist = np.zeros((len(epochs), len(epochs)))

    for i, j, dij in _mp_pairwise_epoch_corr(epochs, n_jobs=n_jobs):
        dist[i, j] = dist[j, i] = dij

    return dist


def spectral_dbscan(aff, n_dim=2, eps=0.2, min_samples=50):
    """
    Perform spectral embedding of a set of observations with affinity matrix
    `aff` into `n_dim` dimensions, and subsequently cluster the resulting data
    with the DBSCAN algorithm, with an neighborhood of `eps`th percentile
    distance and minimum number of samples `min_samples`.

    """

    xi = SpectralEmbedding(n_dim, affinity="precomputed").fit_transform(aff)
    pd = pdist(xi)
    """
    import pylab as pl
    pl.hist(pd, 100)
    """
    eps = np.percentile(pd, 100 * eps)
    print(eps)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(xi)
    return xi.T, db.labels_
