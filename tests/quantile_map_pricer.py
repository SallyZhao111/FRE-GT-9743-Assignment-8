from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

try:
    from scipy.stats import qmc
except Exception:  # pragma: no cover - fallback for environments without qmc
    qmc = None

from fixedincomelib.analytics.sabr import SABRAnalytics, SabrMetrics


@dataclass(frozen=True)
class SabrMarginalSpec:
    forward: float
    sigma_atm_normal: float
    beta: float
    rho: float
    nu: float
    shift: float = 0.0


def _default_grid_bounds(spec: SabrMarginalSpec, time_to_expiry: float) -> Tuple[float, float]:
    approx_std = max(spec.sigma_atm_normal * np.sqrt(time_to_expiry), 1e-5)
    x_min = spec.forward - 8.0 * approx_std
    x_max = spec.forward + 8.0 * approx_std
    lower_physical_bound = -spec.shift + 1e-10
    x_min = max(x_min, lower_physical_bound)
    x_max = max(x_max, x_min + 20.0 * approx_std)
    return x_min, x_max


def _sobol_uniform(num_paths: int, dim: int, seed: int = 42, skip: int = 1024) -> np.ndarray:
    """
    (i) Generate quasi-random vectors s ~ U[0,1]^d via Sobol.
    Falls back to pseudo-random uniform if Sobol is unavailable.
    """
    if qmc is None:
        rng = np.random.default_rng(seed)
        return rng.random((num_paths, dim))

    engine = qmc.Sobol(d=dim, scramble=True, seed=seed)
    if skip > 0:
        engine.fast_forward(skip)
    m = int(np.ceil(np.log2(num_paths)))
    s = engine.random_base2(m=m)
    return s[:num_paths]


def _build_quantile_map(
    spec: SabrMarginalSpec,
    time_to_expiry: float,
    num_grid: int = 801,
    grid_bounds: Tuple[float, float] | None = None,
) -> Dict[str, np.ndarray | float]:
    """
    Build 1D quantile map I such that x ~= I(y), where y ~ N(0,1).
    """
    if grid_bounds is None:
        x_min, x_max = _default_grid_bounds(spec, time_to_expiry)
    else:
        x_min, x_max = grid_bounds

    x_grid = np.linspace(x_min, x_max, num_grid)

    alpha = SABRAnalytics.alpha_from_atm_normal_sigma(
        forward=spec.forward,
        time_to_expiry=time_to_expiry,
        sigma_atm_normal=spec.sigma_atm_normal,
        beta=spec.beta,
        rho=spec.rho,
        nu=spec.nu,
        shift=spec.shift,
        calc_risk=False,
    )[SabrMetrics.ALPHA]

    cdf_res = SABRAnalytics.pdf_and_cdf(
        forward=spec.forward,
        time_to_expiry=time_to_expiry,
        alpha=alpha,
        beta=spec.beta,
        rho=spec.rho,
        nu=spec.nu,
        grids=x_grid,
        shift=spec.shift,
    )

    cdf = np.asarray(cdf_res["cdf"], dtype=float)
    eps = 1e-10
    cdf = np.clip(cdf, eps, 1.0 - eps)
    cdf = np.maximum.accumulate(cdf)

    # Keep strictly increasing points for inversion/interpolation stability.
    keep = np.r_[True, np.diff(cdf) > 1e-12]
    u = cdf[keep]
    x = x_grid[keep]

    # Ensure enough interpolation points.
    if u.shape[0] < 2:
        u = np.array([eps, 1.0 - eps], dtype=float)
        x = np.array([x_grid[0], x_grid[-1]], dtype=float)
    else:
        if u[0] > eps:
            u = np.r_[eps, u]
            x = np.r_[x[0], x]
        if u[-1] < 1.0 - eps:
            u = np.r_[u, 1.0 - eps]
            x = np.r_[x, x[-1]]

    y = norm.ppf(u)

    return {
        "alpha": float(alpha),
        "x_grid": x_grid,
        "cdf_grid": cdf,
        "u_nodes": u,
        "y_nodes": y,
        "x_nodes": x,
    }


def _apply_quantile_map(y: np.ndarray, y_nodes: np.ndarray, x_nodes: np.ndarray) -> np.ndarray:
    return np.interp(y, y_nodes, x_nodes, left=x_nodes[0], right=x_nodes[-1])


def price_spread_option_quantile_map(
    spec_1: SabrMarginalSpec,
    spec_2: SabrMarginalSpec,
    time_to_expiry: float,
    corr_12: float,
    strike: float,
    num_paths: int,
    num_grid: int = 801,
    seed: int = 42,
) -> Dict[str, np.ndarray | float | int]:
    """
    Implements (i) to (v) in the assignment:
      (i)  Sobol uniforms s ~ U[0,1]^d
      (ii) Independent normals z = Phi^{-1}(s)
      (iii) Correlate y = B z, where B B^T = rho
      (iv)  Component-wise quantile maps x_i = I_i(y_i)
      (v)   Monte Carlo estimate of E[(S1(T)-S2(T)-K)^+]
    """
    if abs(corr_12) >= 1.0:
        raise ValueError("corr_12 must be in (-1, 1)")

    map_1 = _build_quantile_map(spec_1, time_to_expiry=time_to_expiry, num_grid=num_grid)
    map_2 = _build_quantile_map(spec_2, time_to_expiry=time_to_expiry, num_grid=num_grid)

    # (i) + (ii)
    s = _sobol_uniform(num_paths=num_paths, dim=2, seed=seed)
    eps = 1e-12
    s = np.clip(s, eps, 1.0 - eps)
    z = norm.ppf(s)

    # (iii)
    corr = np.array([[1.0, corr_12], [corr_12, 1.0]], dtype=float)
    b = np.linalg.cholesky(corr)
    y = z @ b.T

    # (iv)
    x1 = _apply_quantile_map(y[:, 0], map_1["y_nodes"], map_1["x_nodes"])
    x2 = _apply_quantile_map(y[:, 1], map_2["y_nodes"], map_2["x_nodes"])

    # (v)
    payoff = np.maximum(x1 - x2 - strike, 0.0)
    price = float(np.mean(payoff))
    std_err = float(np.std(payoff, ddof=1) / np.sqrt(num_paths))

    return {
        "price": price,
        "std_error": std_err,
        "num_paths": num_paths,
        "payoff_mean": price,
        "payoff_std": float(np.std(payoff, ddof=1)),
        "x1_samples": x1,
        "x2_samples": x2,
        "payoff_samples": payoff,
        "map_1": map_1,
        "map_2": map_2,
    }


def run_convergence(
    spec_1: SabrMarginalSpec,
    spec_2: SabrMarginalSpec,
    time_to_expiry: float,
    corr_12: float,
    strike: float,
    path_list: List[int],
    num_grid: int = 801,
    seed: int = 42,
) -> pd.DataFrame:
    rows = []
    for i, n in enumerate(path_list):
        res = price_spread_option_quantile_map(
            spec_1=spec_1,
            spec_2=spec_2,
            time_to_expiry=time_to_expiry,
            corr_12=corr_12,
            strike=strike,
            num_paths=n,
            num_grid=num_grid,
            seed=seed + i,
        )
        rows.append(
            {
                "num_paths": n,
                "price": res["price"],
                "std_error": res["std_error"],
                "payoff_std": res["payoff_std"],
            }
        )
    return pd.DataFrame(rows).sort_values("num_paths")


if __name__ == "__main__":
    # Assignment Q4 parameters
    tte = 10.0
    strike = 0.001
    corr_12 = 0.9

    spec_1 = SabrMarginalSpec(
        forward=0.03,
        sigma_atm_normal=0.01,
        beta=0.5,
        nu=0.2,
        rho=0.25,
        shift=0.0,
    )
    spec_2 = SabrMarginalSpec(
        forward=0.05,
        sigma_atm_normal=0.012,
        beta=0.5,
        nu=0.15,
        rho=0.45,
        shift=0.0,
    )

    path_list = [2_048, 8_192, 32_768, 131_072]
    df = run_convergence(
        spec_1=spec_1,
        spec_2=spec_2,
        time_to_expiry=tte,
        corr_12=corr_12,
        strike=strike,
        path_list=path_list,
        num_grid=801,
        seed=42,
    )
    print(df.to_string(index=False))
