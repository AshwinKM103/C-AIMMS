"""fluxmem/fusion.py -- BMM-gated memory fusion (FluxMem Eqs. 17-26 + Alg. 1; COLM Eq. 7).

Highest logic risk in this package: a two-component Beta Mixture Model fit
by EM in log-space, plus the merge-vs-new decision built on its posterior.
Implements the equations verbatim -- see the plan's equation reference card
for the `F Eq. NN` citations in each function's docstring.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp
from scipy.stats import beta as beta_dist

from fluxmem.config import FusionConfig
from fluxmem.interfaces import EpisodicUnit

_N_COMPONENTS = 2


def normalize_scores(scores: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """F Eq. 17: min-max normalize to `(0,1)` with an `eps` inset.

    `x_i = eps + (1-2*eps)*(s_i-s_min)/(s_max-s_min)`; when `s_max == s_min`
    (all scores identical, including the single-score case), every `x_i` is
    `0.5` rather than a division by zero.
    """
    scores = np.asarray(scores, dtype=np.float64)
    s_min, s_max = scores.min(), scores.max()
    if s_max == s_min:
        return np.full_like(scores, 0.5)
    return eps + (1 - 2 * eps) * (scores - s_min) / (s_max - s_min)


@dataclass
class BetaMixtureParams:
    """Fitted two-component Beta mixture: `p(x) = pi[0]*Beta(a0,b0) + pi[1]*Beta(a1,b1)`."""

    pi: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    high_compat_index: int


class BetaMixtureModel:
    """Two-component Beta Mixture Model over normalized scores `x in (0,1)`.

    Quantile init (`mu0 <- q_0.3`, `mu1 <- q_0.7`, F Eqs. 18ff.); log-space
    E-step (F Eq. 20) via `scipy.stats.beta.logpdf` + `logsumexp` -- the
    paper calls out log-space explicitly for numerical stability near the
    boundaries `x -> 0+/1-`, where linear-space Beta densities can under-
    or overflow. M-step: `pi_k = N_k/n` (Eq. 21); responsibility-weighted
    `mu_k, sigma^2_k` (Eq. 22); `kappa = mu*(1-mu)/sigma^2 - 1`, `alpha =
    mu*kappa`, `beta = (1-mu)*kappa` (Eq. 23).

    Both `sigma^2 -> 0` (a component collapsing onto a single point) and
    `kappa <= 0` (an invalid Beta parameterization that falls out of the
    moment-matching algebra when `sigma^2` is large relative to `mu*(1-mu)`)
    are reachable degenerate cases -- both are clamped to a floor rather
    than left to produce non-finite `(alpha, beta)`.
    """

    def __init__(
        self,
        n_iter: int = 50,
        init_variance: float = 0.01,
        kappa_floor: float = 1e-3,
        variance_floor: float = 1e-6,
        responsibility_floor: float = 1e-6,
        mu_floor: float = 1e-6,
    ) -> None:
        self._n_iter = n_iter
        self._init_variance = init_variance
        self._kappa_floor = kappa_floor
        self._variance_floor = variance_floor
        self._responsibility_floor = responsibility_floor
        self._mu_floor = mu_floor
        self._params: BetaMixtureParams | None = None

    @property
    def is_fitted(self) -> bool:
        return self._params is not None

    @property
    def params(self) -> BetaMixtureParams:
        if self._params is None:
            raise RuntimeError("BetaMixtureModel has not been fit yet")
        return self._params

    def fit(self, x: np.ndarray) -> BetaMixtureParams:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 1 or x.size == 0:
            raise ValueError("x must be a non-empty 1-D array of normalized scores in (0,1)")

        mu0, mu1 = np.quantile(x, [0.3, 0.7])
        if mu1 <= mu0:
            # Degenerate quantiles (n=1, or all-identical x) -- separate the
            # two initial components so EM has somewhere to go, rather than
            # starting both components on the same point.
            center = (mu0 + mu1) / 2.0
            mu0, mu1 = max(1e-3, center - 0.05), min(1 - 1e-3, center + 0.05)

        pi = np.array([0.5, 0.5])
        alpha, beta_params = self._moment_match(
            np.array([mu0, mu1]), np.full(_N_COMPONENTS, self._init_variance)
        )

        for _ in range(self._n_iter):
            responsibilities = self._e_step(x, pi, alpha, beta_params)
            pi, alpha, beta_params = self._m_step(x, responsibilities)

        high_idx = int(np.argmax(alpha / (alpha + beta_params)))
        self._params = BetaMixtureParams(
            pi=pi, alpha=alpha, beta=beta_params, high_compat_index=high_idx
        )
        return self._params

    def _moment_match(self, mu: np.ndarray, var: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """F Eq. 23: `kappa = mu*(1-mu)/sigma^2 - 1`, `alpha = mu*kappa`, `beta = (1-mu)*kappa`.

        `mu` is clamped away from exactly 0 or 1 -- reachable when a
        component's responsibility fully collapses (all responsibilities
        underflow to exact 0.0 in float64 for a badly-separated point), and
        otherwise produces `alpha=0` or `beta=0`, an invalid Beta
        parameterization the variance/kappa floors alone do not catch.
        """
        mu = np.clip(mu, self._mu_floor, 1 - self._mu_floor)
        var = np.maximum(var, self._variance_floor)
        kappa = mu * (1 - mu) / var - 1
        kappa = np.maximum(kappa, self._kappa_floor)
        return mu * kappa, (1 - mu) * kappa

    def _e_step(
        self, x: np.ndarray, pi: np.ndarray, alpha: np.ndarray, beta_params: np.ndarray
    ) -> np.ndarray:
        """F Eq. 20, log-space: returns responsibilities, shape `(n_components, n)`."""
        log_pdf = np.stack(
            [
                np.log(pi[k]) + beta_dist.logpdf(x, alpha[k], beta_params[k])
                for k in range(_N_COMPONENTS)
            ]
        )
        log_norm = logsumexp(log_pdf, axis=0)
        return np.exp(log_pdf - log_norm)

    def _m_step(
        self, x: np.ndarray, responsibilities: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """F Eqs. 21-23."""
        n = x.shape[0]
        n_k = responsibilities.sum(axis=1)
        n_k_safe = np.maximum(n_k, self._responsibility_floor)
        pi = n_k / n
        mu = (responsibilities * x[None, :]).sum(axis=1) / n_k_safe
        var = (responsibilities * (x[None, :] - mu[:, None]) ** 2).sum(axis=1) / n_k_safe
        alpha, beta_params = self._moment_match(mu, var)
        return pi, alpha, beta_params

    def posterior(self, x: np.ndarray) -> np.ndarray:
        """F Eq. 25: `g(x) = pi[k*]*Beta(x;a[k*],b[k*]) / sum_k pi[k]*Beta(x;a[k],b[k])`."""
        params = self.params
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        log_pdf = np.stack(
            [
                np.log(params.pi[k]) + beta_dist.logpdf(x, params.alpha[k], params.beta[k])
                for k in range(_N_COMPONENTS)
            ]
        )
        log_norm = logsumexp(log_pdf, axis=0)
        return np.exp(log_pdf[params.high_compat_index] - log_norm)


@dataclass
class MergeDecision:
    """Result of `select_merge_target`: either merge into `target_id`, or `NEW`."""

    action: str  # "MERGE" or "NEW"
    target_id: str | None
    normalized_scores: dict[str, float] = field(default_factory=dict)
    posterior_scores: dict[str, float] = field(default_factory=dict)


def select_merge_target(
    incoming: EpisodicUnit,
    candidates: list[EpisodicUnit],
    scorer: Callable[[EpisodicUnit, EpisodicUnit], float],
    cfg: FusionConfig,
) -> MergeDecision:
    """F Alg. 1: threshold on the BMM posterior, with a min-keep top-k fallback.

    `I = {i : g(x_i) >= tau}`; if `|I| < m_min`, `I <- TopK({x_i}, m_min)`
    (F Eq. 26, by raw normalized score `x_i`, not posterior); if `I` is
    still empty, `NEW`; else merge into `argmax_{i in I} x_i`.

    With `m_min >= 1`, the "`I` empty after the fallback" branch is reachable
    only when `candidates` is empty in the first place -- an Alg. 1
    subtlety that is easy to implement as dead code by accident, so it is
    handled explicitly (the early return below) rather than relied upon to
    fall out of the general path.
    """
    if not candidates:
        return MergeDecision(action="NEW", target_id=None)

    raw_scores = np.array([scorer(incoming, c) for c in candidates])
    x = normalize_scores(raw_scores, cfg.eps)

    bmm = BetaMixtureModel(n_iter=cfg.em_iters)
    bmm.fit(x)
    posteriors = bmm.posterior(x)

    ids = [c.episode_id for c in candidates]
    normalized_scores = dict(zip(ids, x.tolist(), strict=True))
    posterior_scores = dict(zip(ids, posteriors.tolist(), strict=True))

    eligible = [i for i, g in enumerate(posteriors) if g >= cfg.tau]
    if len(eligible) < cfg.m_min:
        eligible = list(np.argsort(-x)[: cfg.m_min])

    if not eligible:
        return MergeDecision(
            action="NEW",
            target_id=None,
            normalized_scores=normalized_scores,
            posterior_scores=posterior_scores,
        )

    best_idx = max(eligible, key=lambda i: x[i])
    return MergeDecision(
        action="MERGE",
        target_id=ids[best_idx],
        normalized_scores=normalized_scores,
        posterior_scores=posterior_scores,
    )
