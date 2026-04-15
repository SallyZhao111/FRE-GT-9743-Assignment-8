import pandas as pd
from fixedincomelib.analytics import *

# european option


def qfEuropeanOptionLogNormal(
    forward: float,
    strike: float,
    time_to_expiry: float,
    log_normal_sigma: float,
    option_type: Optional[str] = "call",
    calc_risk: Optional[bool] = False,
):

    res = EuropeanOptionAnalytics.european_option_log_normal(
        forward,
        strike,
        time_to_expiry,
        log_normal_sigma,
        CallOrPut.from_string(option_type),
        calc_risk,
    )

    # remove tte risk
    if SimpleMetrics.TTE_RISK in res:
        res.pop(SimpleMetrics.TTE_RISK)

    return res


def qfEuropeanOptionImpliedLogNormalVol(
    pv: float,
    forward: float,
    strike: float,
    time_to_expiry: float,
    option_type: Optional[str] = "call",
    calc_risk: Optional[bool] = False,
    tol: Optional[float] = 1e-8,
):

    res = EuropeanOptionAnalytics.implied_lognormal_vol_sensitivities(
        pv, forward, strike, time_to_expiry, CallOrPut.from_string(option_type), calc_risk, tol
    )

    return res


def qfEuropeanOptionNormal(
    forward: float,
    strike: float,
    time_to_expiry: float,
    log_normal_sigma: float,
    option_type: Optional[str] = "call",
    calc_risk: Optional[bool] = False,
):

    res = EuropeanOptionAnalytics.european_option_normal(
        forward,
        strike,
        time_to_expiry,
        log_normal_sigma,
        CallOrPut.from_string(option_type),
        calc_risk,
    )

    # remove tte risk
    if SimpleMetrics.TTE_RISK in res:
        res.pop(SimpleMetrics.TTE_RISK)

    return res


def qfEuropeanOptionImpliedNormalVol(
    pv: float,
    forward: float,
    strike: float,
    time_to_expiry: float,
    option_type: Optional[str] = "call",
    calc_risk: Optional[bool] = False,
    tol: Optional[float] = 1e-8,
):

    res = EuropeanOptionAnalytics.implied_normal_vol_sensitivities(
        pv, forward, strike, time_to_expiry, CallOrPut.from_string(option_type), calc_risk, tol
    )

    return res


def qfEuropeanOptionNormalVolFromLogNormalVol(
    forward: float,
    strike: float,
    time_to_expiry: float,
    log_normal_sigma: float,
    shift: Optional[float] = 0.0,
    calc_risk: Optional[bool] = False,
    tol: Optional[float] = 1e-8,
):

    res = EuropeanOptionAnalytics.lognormal_vol_to_normal_vol(
        forward, strike, time_to_expiry, log_normal_sigma, calc_risk, shift, tol
    )

    return res


def qfEuropeanOptionLogNormalVolFromNormalVol(
    forward: float,
    strike: float,
    time_to_expiry: float,
    normal_sigma: float,
    shift: Optional[float] = 0.0,
    calc_risk: Optional[bool] = False,
    tol: Optional[float] = 1e-8,
):

    res = EuropeanOptionAnalytics.normal_vol_to_lognormal_vol(
        forward, strike, time_to_expiry, normal_sigma, calc_risk, shift, tol
    )

    return res


### sabr


def qfEuropeanOptionSABRLogNormalSigma(
    forward: float,
    strike: float,
    time_to_expiry: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    shift: Optional[float] = 0.0,
    calc_risk: Optional[bool] = False,
):

    res = SABRAnalytics.lognormal_vol_from_alpha(
        forward, strike, time_to_expiry, alpha, beta, rho, nu, shift, calc_risk
    )

    return res


def qfEuropeanOptionSABRAlphaFromATMLogNormalSigma(
    forward: float,
    time_to_expiry: float,
    sigma_atm_log_normal: float,
    beta: float,
    rho: float,
    nu: float,
    shift: Optional[float] = 0.0,
    calc_risk: Optional[bool] = False,
    max_iter: Optional[int] = 50,
    tol: Optional[float] = 1e-8,
):

    res = SABRAnalytics.alpha_from_atm_lognormal_sigma(
        forward,
        time_to_expiry,
        sigma_atm_log_normal,
        beta,
        rho,
        nu,
        shift,
        calc_risk,
        max_iter,
        tol,
    )

    return res


def qfEuropeanOptionSABRAlphaFromATMNormalSigma(
    forward: float,
    time_to_expiry: float,
    sigma_atm_normal: float,
    beta: float,
    rho: float,
    nu: float,
    shift: Optional[float] = 0.0,
    calc_risk: Optional[bool] = False,
    max_iter: Optional[int] = 50,
    tol: Optional[float] = 1e-8,
):

    res = SABRAnalytics.alpha_from_atm_normal_sigma(
        forward, time_to_expiry, sigma_atm_normal, beta, rho, nu, shift, calc_risk, max_iter, tol
    )

    return res


def qfEuropeanOptionSABRATMNormalSigmaFromAlpha(
    forward: float,
    time_to_expiry: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    shift: Optional[float] = 0.0,
    calc_risk: Optional[bool] = False,
    tol: Optional[float] = 1e-8,
):

    res = SABRAnalytics.atm_normal_sigma_from_alpha(
        forward, time_to_expiry, alpha, beta, rho, nu, shift, calc_risk, tol
    )

    return res


def qfEuropeanOptionSABR(
    forward: float,
    strike: float,
    time_to_expiry: float,
    option_type: str,
    alpha_or_atm_ln_sigma: float,
    beta: float,
    rho: float,
    nu: float,
    shift: Optional[float] = 0.0,
    calc_risk: Optional[bool] = False,
    is_alpha_parameterized: Optional[bool] = True,
):

    res = None
    if is_alpha_parameterized:
        res = SABRAnalytics.european_option_alpha(
            forward,
            strike,
            time_to_expiry,
            CallOrPut.from_string(option_type),
            alpha_or_atm_ln_sigma,
            beta,
            rho,
            nu,
            shift,
            calc_risk,
        )
    else:
        res = SABRAnalytics.european_option_ln_sigma(
            forward,
            strike,
            time_to_expiry,
            CallOrPut.from_string(option_type),
            alpha_or_atm_ln_sigma,
            beta,
            rho,
            nu,
            shift,
            calc_risk,
        )

    return res


def qfEuropeanOptionSABRNormal(
    forward: float,
    strike: float,
    time_to_expiry: float,
    option_type: str,
    atm_normal_sigma: float,
    beta: float,
    rho: float,
    nu: float,
    shift: Optional[float] = 0.0,
    calc_risk: Optional[bool] = False,
):

    res = SABRAnalytics.european_option_normal_sigma(
        forward,
        strike,
        time_to_expiry,
        CallOrPut.from_string(option_type),
        atm_normal_sigma,
        beta,
        rho,
        nu,
        shift,
        calc_risk,
    )

    return res


def qfEuropeanOptionSABRPdfAndCdf(
    forward: float,
    time_to_expiry: float,
    alpha_or_atm_sigma: float,
    beta: float,
    rho: float,
    nu: float,
    grid_min: float,
    grid_max: float,
    num_pts: int,
    shift: Optional[float] = 0,
    is_alpha_parameterized: Optional[bool] = True,
    is_ln_sigma: Optional[bool] = True,
):

    ### generate grid
    ln_x_min, ln_x_max = np.log(grid_min + shift), np.log(grid_max + shift)
    # num_pts_ = num_pts
    num_pts_ = num_pts if num_pts % 2 == 1 else num_pts + 1
    spacing = (ln_x_max - ln_x_min) / (num_pts_ - 1)
    ln_xs = np.arange(ln_x_min, ln_x_max + spacing / 1e2, spacing)
    xs = [np.exp(ln_x) - shift for ln_x in ln_xs]

    ### conversion to alpha
    alpha = alpha_or_atm_sigma
    if not is_alpha_parameterized:
        if is_ln_sigma:
            alpha = SABRAnalytics.alpha_from_atm_lognormal_sigma(
                forward, time_to_expiry, alpha_or_atm_sigma, beta, rho, nu, shift
            )[SabrMetrics.ALPHA]
        else:
            alpha = SABRAnalytics.alpha_from_atm_normal_sigma(
                forward, time_to_expiry, alpha_or_atm_sigma, beta, rho, nu, shift
            )[SabrMetrics.ALPHA]

    ### sample pdf/cdf and pack up
    dist = SABRAnalytics.pdf_and_cdf(forward, time_to_expiry, alpha, beta, rho, nu, xs, shift)
    df = pd.DataFrame(columns=["Forward", "ShiftedForward", "Cdf", "Pdf"])
    df.Forward = dist["grids"]
    df.ShiftedForward = dist["grids"] + shift
    df.Cdf = dist["cdf"]
    df.Pdf = dist["pdf"]

    return df


def qfSABRQuantileMapSpreadOptionFromNormalATM(
    forward_1: float,
    sigma_atm_normal_1: float,
    beta_1: float,
    rho_1: float,
    nu_1: float,
    forward_2: float,
    sigma_atm_normal_2: float,
    beta_2: float,
    rho_2: float,
    nu_2: float,
    corr_12: float,
    strike: float,
    time_to_expiry: float,
    num_paths: int,
    shift_1: Optional[float] = 0.0,
    shift_2: Optional[float] = 0.0,
    num_grid: Optional[int] = 801,
    seed: Optional[int] = 42,
    use_sobol: Optional[bool] = True,
    return_paths: Optional[bool] = False,
):
    return SABRAnalytics.quantile_map_spread_option_price_from_normal_atm(
        forward_1=forward_1,
        sigma_atm_normal_1=sigma_atm_normal_1,
        beta_1=beta_1,
        rho_1=rho_1,
        nu_1=nu_1,
        forward_2=forward_2,
        sigma_atm_normal_2=sigma_atm_normal_2,
        beta_2=beta_2,
        rho_2=rho_2,
        nu_2=nu_2,
        corr_12=corr_12,
        strike=strike,
        time_to_expiry=time_to_expiry,
        num_paths=num_paths,
        shift_1=shift_1,
        shift_2=shift_2,
        num_grid=num_grid,
        seed=seed,
        use_sobol=use_sobol,
        return_paths=return_paths,
    )
