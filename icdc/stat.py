"""
iidc.stat
=========

Statistics utilities

"""

import numpy as np

from scipy import stats, signal, optimize

from . import util


def find(condition):  # from matplotlib.mlab
    res, = np.nonzero(np.ravel(condition))
    return res


def fdr(p, alpha=0.05):
    r"""
    Implements the false discover rate correction of Benjamini
    and Hochberg, following Wasserman's All of Statistics, pg 167.

    """

    p_sorted = np.sort(p.flat[:])
    count_p = np.r_[: p.size] + 1
    limit = count_p * alpha / np.sum(1.0 / count_p) / p.size
    under = find(p_sorted < limit)
    p_thresh = p_sorted[under[-1]] if under.size > 0 else 0.0
    h0_rejected = p < p_thresh
    return p_thresh, h0_rejected


def lfdr(x, nbins=50, dc=1, doplot=True):
    """
    Implements the local false discovery rate correction, due to Efron 2005.

    Where FDR performs a correction based on the CDF, lFDR performs a
    correction based on the PDF, which works by computing density statistics on
    `x`, where a "null" hypothesis distribution H0, here a Gaussian
    distribution, is fit to the center of the data, and this provides an
    estimation of how likely each bin comes from the Gaussian distribution, a
    measure called the local false discovery rate.

    The use of a Gaussian in this implementation is not required, in fact, the
    distribution to be fit does not even have to have an analytic PDF, rather
    it is simply necessary to be able to compute a numerical density for the H0.

    Decimating the data by specifying `dc` as an integer greater than 1 
    can accelerate the density estimation and safer on large datasets 
    where said estimation takes more time.

    Parameters
    ----------
    x : array
        Data to analyze
    dc : int
        Decimate data, accelerating analysis
    nbins : int or None, optional
        Number of points at which to evaluate the density

    Returns
    -------
    xb : array
        Points at which the densities are evaluated
    f : array
        Density of `x`
    cf : array
        Estimated "center density" of H0
    fdr : array
        Estimated false discovery rate 
    llx : array
        Log lfdr for each datapoint in `x`

    """

    k = stats.gaussian_kde(x[::dc])
    xb = np.r_[x.min() : x.max() : 1j * nbins]
    f = k(xb)
    f /= f.sum()

    # TODO use argsort to avoid evaluating f on linspace twice
    dxb = np.interp(
        np.r_[0.0 : 1.0 : 1j * nbins], np.cumsum(f), xb + (xb[1] - xb[0]) / 2.0
    )
    xb = np.unique(np.r_[xb, dxb])
    xb.sort()
    f = k(xb)
    f /= f.sum()

    # estimate center sub-density
    def err(par):
        mu, sigma, alo, ahi = par
        sl = slice(np.argmin(np.abs(xb - alo)), np.argmin(np.abs(xb - ahi)))
        f0 = stats.norm.pdf(xb[sl], loc=mu, scale=sigma)
        f1 = f[sl]
        f0 = f0 / f0.max() * f1.max()
        return np.sum((f0 - f1) ** 2) / np.sum(f1 ** 2) - f1.sum()

    mu0, sig0 = x.mean(), x.std()
    mu, sig, _, _ = optimize.fmin(
        err, (mu0, sig0, mu0 - sig0, mu0 + sig0), disp=0, maxiter=1000
    )
    cf = stats.norm.pdf(xb, loc=mu, scale=sig)
    cf = cf / cf.max() * f.max()

    # compute lfdr & transform data to log-lfdr
    fdr = np.clip(cf / f, 0.0, 1.0)
    llx = np.interp(x, xb, np.log(fdr))

    if doplot:
        import pylab as pl

        pl.semilogy(xb, f, "k")
        pl.semilogy(xb, cf, "k--")
        pl.semilogy(xb, fdr, "k.")
        pl.grid(True)
        pl.ylim([1e-5, 1.0])
        plevels = [0.2, 0.1, 0.05, 0.01, 0.001]
        pl.yticks(plevels, list(map(str, plevels)))

    return xb, f, cf, fdr, llx


def threshold_duration(sig, length):
    """
    Threshold significance row-wise in `sig` for `length` elements.

    """

    sig = sig.copy()
    for j, sigj in enumerate(sig):
        bounds = np.where(sigj[1:] != sigj[:-1])[0] + 1
        segments = np.split(sigj, bounds)
        bounds = np.r_[0, bounds, len(sigj)]
        for i, seg in enumerate(segments):
            if seg.sum() < length:
                sigj[bounds[i] : bounds[i + 1]] = False
    return sig


def per_channel_events_pvalues(fs, data, events):
    """
    Compute ttest per channel's events.

    """

    P = []
    for e in events:
        if isinstance(e, np.ndarray):
            _, ep = util.extract_windows(fs, data, e)
            _, p = stats.ttest_1samp(ep, 0.0, axis=0)
            P.append(p)
    return np.array(P)
