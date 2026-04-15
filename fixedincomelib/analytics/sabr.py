from enum import Enum
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from scipy.stats import norm
try:
    from scipy.stats import qmc
except Exception:  # pragma: no cover
    qmc = None
from fixedincomelib.analytics.european_options import (
    CallOrPut,
    SimpleMetrics,
    EuropeanOptionAnalytics,
)


class SabrMetrics(Enum):

    # parameters
    ALPHA = "alpha"
    BETA = "beta"
    NU = "nu"
    RHO = "rho"

    # risk
    DALPHA = "dalpha"
    DLNSIGMA = "dlnsigma"
    DNORMALSIGMA = "dnormalsigma"
    DBETA = "dbeta"
    DRHO = "drho"
    DNU = "dnu"
    DFORWARD = "dforward"
    DSTRIKE = "dstrike"
    DTTE = "dtte"
    DSTRIKESTRIKE = "dstrikestrike"

    # (alpha, beta, nu, rho, forward, strike, tte) => \sigma_k
    D_LN_SIGMA_D_FORWARD = "d_ln_sigma_d_forward"
    D_LN_SIGMA_D_STRIKE = "d_ln_sigma_d_strike"
    D_LN_SIGMA_D_TTE = "d_ln_sigma_d_tte"
    D_LN_SIGMA_D_ALPHA = "d_ln_sigma_d_alpha"
    D_LN_SIGMA_D_BETA = "d_ln_sigma_d_beta"
    D_LN_SIGMA_D_NU = "d_ln_sigma_d_nu"
    D_LN_SIGMA_D_RHO = "d_ln_sigma_d_rho"
    D_LN_SIGMA_D_STRIKESTRIKE = "d_ln_sigma_d_strike_strike"

    # (\sigma_ln_atm, f, tte, beta, nu, rho) => alpha
    D_ALPHA_D_LN_SIGMA_ATM = "d_alpha_d_ln_sigma_atm"
    D_ALPHA_D_FORWARD = "d_alpha_d_forward"
    D_ALPHA_D_TTE = "d_alpha_d_tte"
    D_ALPHA_D_BETA = "d_alpha_d_beta"
    D_ALPHA_D_NU = "d_alpha_d_nu"
    D_ALPHA_D_RHO = "d_alpha_d_rho"

    # (alpha, beta, nu, rho, f, tte) => \sigma_n_atm
    D_NORMAL_SIGMA_D_ALPHA = "d_normal_sigma_d_alpha"
    D_NORMAL_SIGMA_D_BETA = "d_normal_sigma_d_beta"
    D_NORMAL_SIGMA_D_NU = "d_normal_sigma_d_nu"
    D_NORMAL_SIGMA_D_RHO = "d_normal_sigma_d_rho"
    D_NORMAL_SIGMA_D_FORWARD = "d_normal_sigma_d_forward"
    D_NORMAL_SIGMA_D_TTE = "d_normal_sigma_d_tte"
    D_ALPHA_D_NORMAL_SIGMA_ATM = "d_alpha_d_normal_sigma_atm"

    @classmethod
    def from_string(cls, value: str) -> "SabrMetrics":
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.lower())
        except ValueError as e:
            raise ValueError(f"Invalid token: {value}") from e

    def to_string(self) -> str:
        return self.value


class SABRAnalytics:

    EPSILON = 1e-6

    ### parameters conversion

    # solver to back out lognormal vol from alpha and sensitivities
    # please implement the _vol_and_risk function to make this work
    @staticmethod
    def lognormal_vol_from_alpha(
        forward: float,
        strike: float,
        time_to_expiry: float,
        alpha: float,
        beta: float,
        rho: float,
        nu: float,
        shift: Optional[float] = 0.0,
        calc_risk: Optional[bool] = False,
    ) -> Dict[SabrMetrics | SimpleMetrics, float]:

        res: Dict[Any, float] = {}

        ln_sigma, risks = SABRAnalytics._vol_and_risk(
            forward + shift, strike + shift, time_to_expiry, alpha, beta, rho, nu, calc_risk
        )
        res[SimpleMetrics.IMPLIED_LOG_NORMAL_VOL] = ln_sigma

        if len(risks) == 0:
            return res

        res.update(risks)
        return res

    @staticmethod
    def alpha_from_atm_lognormal_sigma(
        forward: float,
        time_to_expiry: float,
        sigma_atm_lognormal: float,
        beta: float,
        rho: float,
        nu: float,
        shift: Optional[float] = 0.0,
        calc_risk: Optional[bool] = False,
        max_iter: Optional[int] = 50,
        tol: Optional[float] = 1e-12,
    ) -> Dict[SabrMetrics, float]:

        if forward + shift <= 0.0:
            raise ValueError("forward must be > 0")
        if time_to_expiry < 0.0:
            raise ValueError("time_to_expiry must be >= 0")
        if sigma_atm_lognormal <= 0.0:
            raise ValueError("sigma_atm_lognormal must be > 0")
        if abs(rho) >= 1.0:
            raise ValueError("rho must be in (-1,1)")
        if nu < 0.0:
            raise ValueError("nu must be >= 0")
        if not (0.0 <= beta <= 1.0):
            raise ValueError("beta should be in [0,1] for standard SABR usage")

        # newton + bisec fallback
        # root finding
        # f = F(alpha, theta) - ln_sigma = 0
        # where F is lognormal_vol_from_alpha
        # alpha^* = alpha(ln_sigma, theta)

        
        this_res = None
        alpha = sigma_atm_lognormal * (forward + shift) ** (1.0 - beta)

        for _ in range(max_iter):
            this_res = SABRAnalytics.lognormal_vol_from_alpha(
                forward=forward,
                strike=forward,
                time_to_expiry=time_to_expiry,
                alpha=alpha,
                beta=beta,
                rho=rho,
                nu=nu,
                shift=shift,
                calc_risk=True,
            )

            sigma_model = this_res[SimpleMetrics.IMPLIED_LOG_NORMAL_VOL]
            f_val = sigma_model - sigma_atm_lognormal

            if abs(f_val) < tol:
                break

            dfdalpha = this_res[SabrMetrics.D_LN_SIGMA_D_ALPHA]
            if abs(dfdalpha) < 1e-14:
                raise RuntimeError(
                    "alpha_from_atm_lognormal_sigma: derivative wrt alpha too small"
                )

            alpha_new = alpha - f_val / dfdalpha

            if alpha_new <= 0.0:
                alpha_new = 0.5 * alpha

            if abs(alpha_new - alpha) < tol:
                alpha = alpha_new
                break

            alpha = alpha_new

        else:
            raise RuntimeError("alpha_from_atm_lognormal_sigma: Newton did not converge")

        res: Dict[SabrMetrics, float] = {SabrMetrics.ALPHA: alpha}

        if calc_risk:
            dfdalpha = this_res[SabrMetrics.D_LN_SIGMA_D_ALPHA]

            res[SabrMetrics.D_ALPHA_D_LN_SIGMA_ATM] = 1.0 / dfdalpha
            res[SabrMetrics.D_ALPHA_D_FORWARD] = (
                -this_res[SabrMetrics.D_LN_SIGMA_D_FORWARD] / dfdalpha
            )
            res[SabrMetrics.D_ALPHA_D_TTE] = (
                -this_res[SabrMetrics.D_LN_SIGMA_D_TTE] / dfdalpha
            )
            res[SabrMetrics.D_ALPHA_D_BETA] = (
                -this_res[SabrMetrics.D_LN_SIGMA_D_BETA] / dfdalpha
            )
            res[SabrMetrics.D_ALPHA_D_NU] = (
                -this_res[SabrMetrics.D_LN_SIGMA_D_NU] / dfdalpha
            )
            res[SabrMetrics.D_ALPHA_D_RHO] = (
                -this_res[SabrMetrics.D_LN_SIGMA_D_RHO] / dfdalpha
            )

        return res


    # conversion to alpha from normal atm vol
    @staticmethod
    def alpha_from_atm_normal_sigma(
        forward: float,
        time_to_expiry: float,
        sigma_atm_normal: float,
        beta: float,
        rho: float,
        nu: float,
        shift: Optional[float] = 0.0,
        calc_risk: bool = False,
        max_iter: int = 50,
        tol: float = 1e-8,
    ) -> Dict[SabrMetrics, float]:

        # at atm, from nv vol to ln vol
        # please check the functions in 'EuropeanOptionAnalytics.py'

        # compute implied log normal vol

        # risk aggregation
        # Step 1: convert ATM normal vol -> ATM lognormal vol
        ln_res = EuropeanOptionAnalytics.normal_vol_to_lognormal_vol(
            forward + shift,
            forward + shift,
            time_to_expiry,
            sigma_atm_normal,
            calc_risk=calc_risk,
            shift=0.0,
            tol=tol,
        )
        sigma_atm_ln = ln_res[SimpleMetrics.IMPLIED_LOG_NORMAL_VOL]

        # Step 2: ATM lognormal vol -> alpha
        alpha_res = SABRAnalytics.alpha_from_atm_lognormal_sigma(
            forward=forward,
            time_to_expiry=time_to_expiry,
            sigma_atm_lognormal=sigma_atm_ln,
            beta=beta,
            rho=rho,
            nu=nu,
            shift=shift,
            calc_risk=calc_risk,
            max_iter=max_iter,
            tol=tol,
        )

        final_res = {
            SabrMetrics.ALPHA: alpha_res[SabrMetrics.ALPHA]
        }

        if calc_risk:
            d_alpha_d_ln = alpha_res[SabrMetrics.D_ALPHA_D_LN_SIGMA_ATM]
            d_ln_d_n = ln_res[SimpleMetrics.D_LN_VOL_D_N_VOL]

            # d alpha / d sigma_N_atm
            final_res[SabrMetrics.D_ALPHA_D_NORMAL_SIGMA_ATM] = d_alpha_d_ln * d_ln_d_n

            # chain rule for beta, nu, rho:
            # alpha = alpha( sigma_ln(sigma_n, F, T), beta, nu, rho, F, T )
            final_res[SabrMetrics.D_ALPHA_D_BETA] = alpha_res[SabrMetrics.D_ALPHA_D_BETA]
            final_res[SabrMetrics.D_ALPHA_D_NU] = alpha_res[SabrMetrics.D_ALPHA_D_NU]
            final_res[SabrMetrics.D_ALPHA_D_RHO] = alpha_res[SabrMetrics.D_ALPHA_D_RHO]

            # forward and tte need chain rule because ln vol conversion also depends on F, T
            d_ln_d_f = ln_res.get(SimpleMetrics.D_LN_VOL_D_FORWARD, 0.0)
            d_ln_d_t = ln_res.get(SimpleMetrics.D_LN_VOL_D_TTE, 0.0)

            final_res[SabrMetrics.D_ALPHA_D_FORWARD] = (
                alpha_res[SabrMetrics.D_ALPHA_D_FORWARD]
                + d_alpha_d_ln * d_ln_d_f
            )

            final_res[SabrMetrics.D_ALPHA_D_TTE] = (
                alpha_res[SabrMetrics.D_ALPHA_D_TTE]
                + d_alpha_d_ln * d_ln_d_t
            )

        return final_res
    ### option pricing

    @staticmethod
    def european_option_alpha(
        forward: float,
        strike: float,
        time_to_expiry: float,
        opt_type: CallOrPut,
        alpha: float,
        beta: float,
        rho: float,
        nu: float,
        shift: Optional[float] = 0.0,
        calc_risk: Optional[bool] = False,
    ):

        ### pv
        ln_sigma_and_sensitivities = SABRAnalytics.lognormal_vol_from_alpha(
            forward, strike, time_to_expiry, alpha, beta, rho, nu, shift, calc_risk
        )
        ln_iv = ln_sigma_and_sensitivities[SimpleMetrics.IMPLIED_LOG_NORMAL_VOL]
        value_and_sensitivities = EuropeanOptionAnalytics.european_option_log_normal(
            forward + shift, strike + shift, time_to_expiry, ln_iv, opt_type, calc_risk
        )

        ### risk(analytic)
        if calc_risk:
            ## first order risks
            dvdsigma = value_and_sensitivities[SimpleMetrics.VEGA]
            value_and_sensitivities.pop(SimpleMetrics.VEGA)
            # delta
            value_and_sensitivities[SimpleMetrics.DELTA] += (
                dvdsigma * ln_sigma_and_sensitivities[SabrMetrics.D_LN_SIGMA_D_FORWARD]
            )
            # theta
            value_and_sensitivities[SimpleMetrics.THETA] -= (
                dvdsigma * ln_sigma_and_sensitivities[SabrMetrics.D_LN_SIGMA_D_TTE]
            )
            # sabr alpha/beta/nu/rho
            for key, risk in [
                (SabrMetrics.DALPHA, SabrMetrics.D_LN_SIGMA_D_ALPHA),
                (SabrMetrics.DBETA, SabrMetrics.D_LN_SIGMA_D_BETA),
                (SabrMetrics.DRHO, SabrMetrics.D_LN_SIGMA_D_RHO),
                (SabrMetrics.DNU, SabrMetrics.D_LN_SIGMA_D_NU),
            ]:
                value_and_sensitivities[key] = dvdsigma * ln_sigma_and_sensitivities[risk]
            # strike
            value_and_sensitivities[SimpleMetrics.STRIKE_RISK] += (
                dvdsigma * ln_sigma_and_sensitivities[SabrMetrics.D_LN_SIGMA_D_STRIKE]
            )

            ## second order risk (bump reval)
            v_base = value_and_sensitivities[SimpleMetrics.PV]
            # strike
            res_up = SABRAnalytics.lognormal_vol_from_alpha(
                forward, strike + SABRAnalytics.EPSILON, time_to_expiry, alpha, beta, rho, nu, shift
            )
            vol_up = res_up[SimpleMetrics.IMPLIED_LOG_NORMAL_VOL]
            v_up = EuropeanOptionAnalytics.european_option_log_normal(
                forward + shift,
                strike + shift + SABRAnalytics.EPSILON,
                time_to_expiry,
                vol_up,
                opt_type,
            )[SimpleMetrics.PV]

            res_dn = SABRAnalytics.lognormal_vol_from_alpha(
                forward, strike - SABRAnalytics.EPSILON, time_to_expiry, alpha, beta, rho, nu, shift
            )
            vol_dn = res_dn[SimpleMetrics.IMPLIED_LOG_NORMAL_VOL]
            v_dn = EuropeanOptionAnalytics.european_option_log_normal(
                forward + shift,
                strike + shift - SABRAnalytics.EPSILON,
                time_to_expiry,
                vol_dn,
                opt_type,
            )[SimpleMetrics.PV]
            value_and_sensitivities[SimpleMetrics.STRIKE_RISK_2] = (v_up - 2 * v_base + v_dn) / (
                SABRAnalytics.EPSILON**2
            )

            # gamma
            res_up = SABRAnalytics.lognormal_vol_from_alpha(
                forward + SABRAnalytics.EPSILON, strike, time_to_expiry, alpha, beta, rho, nu, shift
            )
            vol_up = res_up[SimpleMetrics.IMPLIED_LOG_NORMAL_VOL]
            v_up = EuropeanOptionAnalytics.european_option_log_normal(
                forward + shift + SABRAnalytics.EPSILON,
                strike + shift,
                time_to_expiry,
                vol_up,
                opt_type,
            )[SimpleMetrics.PV]
            res_dn = SABRAnalytics.lognormal_vol_from_alpha(
                forward - SABRAnalytics.EPSILON, strike, time_to_expiry, alpha, beta, rho, nu, shift
            )
            vol_dn = res_dn[SimpleMetrics.IMPLIED_LOG_NORMAL_VOL]
            v_dn = EuropeanOptionAnalytics.european_option_log_normal(
                forward + shift - SABRAnalytics.EPSILON,
                strike + shift,
                time_to_expiry,
                vol_dn,
                opt_type,
            )[SimpleMetrics.PV]
            value_and_sensitivities[SimpleMetrics.GAMMA] = (v_up - 2 * v_base + v_dn) / (
                SABRAnalytics.EPSILON**2
            )

        return value_and_sensitivities

    # Given function
    @staticmethod
    def european_option_ln_sigma(
        forward: float,
        strike: float,
        time_to_expiry: float,
        opt_type: CallOrPut,
        ln_sigma_atm: float,
        beta: float,
        rho: float,
        nu: float,
        shift: Optional[float] = 0.0,
        calc_risk: Optional[bool] = False,
    ):

        ### pv
        alpha_and_sensitivities = SABRAnalytics.alpha_from_atm_lognormal_sigma(
            forward, time_to_expiry, ln_sigma_atm, beta, rho, nu, shift, calc_risk
        )
        alpha = alpha_and_sensitivities[SabrMetrics.ALPHA]
        value_and_sensitivities = SABRAnalytics.european_option_alpha(
            forward, strike, time_to_expiry, opt_type, alpha, beta, rho, nu, shift, calc_risk
        )

        ### risk
        if calc_risk:
            ## first order risks
            dvdalpha = value_and_sensitivities[SabrMetrics.DALPHA]
            value_and_sensitivities.pop(SabrMetrics.DALPHA)

            # delta
            value_and_sensitivities[SimpleMetrics.DELTA] += (
                dvdalpha * alpha_and_sensitivities[SabrMetrics.D_ALPHA_D_FORWARD]
            )
            # theta
            value_and_sensitivities[SimpleMetrics.THETA] -= (
                dvdalpha * alpha_and_sensitivities[SabrMetrics.D_ALPHA_D_TTE]
            )
            # ln_sigma
            value_and_sensitivities[SabrMetrics.DLNSIGMA] = (
                dvdalpha * alpha_and_sensitivities[SabrMetrics.D_ALPHA_D_LN_SIGMA_ATM]
            )
            # sabr beta/rho/nu
            for key, risk in [
                (SabrMetrics.DBETA, SabrMetrics.D_ALPHA_D_BETA),
                (SabrMetrics.DRHO, SabrMetrics.D_ALPHA_D_RHO),
                (SabrMetrics.DNU, SabrMetrics.D_ALPHA_D_NU),
            ]:
                value_and_sensitivities[key] += dvdalpha * alpha_and_sensitivities[risk]

            ## second order risk (bump reval)
            v_base = value_and_sensitivities[SimpleMetrics.PV]

            # gamma
            res_up = SABRAnalytics.alpha_from_atm_lognormal_sigma(
                forward + SABRAnalytics.EPSILON, time_to_expiry, ln_sigma_atm, beta, rho, nu, shift
            )
            alpha_up = res_up[SabrMetrics.ALPHA]
            v_up = SABRAnalytics.european_option_alpha(
                forward + SABRAnalytics.EPSILON,
                strike,
                time_to_expiry,
                opt_type,
                alpha_up,
                beta,
                rho,
                nu,
                shift,
            )[SimpleMetrics.PV]
            res_dn = SABRAnalytics.alpha_from_atm_lognormal_sigma(
                forward - SABRAnalytics.EPSILON, time_to_expiry, ln_sigma_atm, beta, rho, nu, shift
            )
            alpha_dn = res_dn[SabrMetrics.ALPHA]
            v_dn = SABRAnalytics.european_option_alpha(
                forward - SABRAnalytics.EPSILON,
                strike,
                time_to_expiry,
                opt_type,
                alpha_dn,
                beta,
                rho,
                nu,
                shift,
            )[SimpleMetrics.PV]
            value_and_sensitivities[SimpleMetrics.GAMMA] = (v_up - 2 * v_base + v_dn) / (
                SABRAnalytics.EPSILON**2
            )

        return value_and_sensitivities

    # European call/put SABR risk with normal vol input, please implement this function with european_option_alpha api
    @staticmethod
    def european_option_normal_sigma(
        forward: float,
        strike: float,
        time_to_expiry: float,
        opt_type: CallOrPut,
        normal_sigma_atm: float,
        beta: float,
        rho: float,
        nu: float,
        shift: Optional[float] = 0.0,
        calc_risk: Optional[bool] = False,
    ):
        """
        Please implement this function with european_option_alpha api

        """
        value_and_sensitivities = {}
        # Step 1: sigma_N_atm -> alpha
        alpha_and_sensitivities = SABRAnalytics.alpha_from_atm_normal_sigma(
            forward,
            time_to_expiry,
            normal_sigma_atm,
            beta,
            rho,
            nu,
            shift,
            calc_risk,
        )
        alpha = alpha_and_sensitivities[SabrMetrics.ALPHA]
        ### pv
        # Step 2: price under alpha-parameterization
        value_and_sensitivities = SABRAnalytics.european_option_alpha(
            forward,
            strike,
            time_to_expiry,
            opt_type,
            alpha,
            beta,
            rho,
            nu,
            shift,
            calc_risk,
        )


        # ### risk
        # if calc_risk:
        #     ## first order risks

        #     # sabr beta/rho/nu

        #     # second order risk (bump reval)

        #     # gamma
        #     pass

        # return value_and_sensitivities
        ### risk
        if calc_risk:
            ## first order risks
            dvdalpha = value_and_sensitivities[SabrMetrics.DALPHA]
            value_and_sensitivities.pop(SabrMetrics.DALPHA)

            # delta
            value_and_sensitivities[SimpleMetrics.DELTA] += (
                dvdalpha * alpha_and_sensitivities[SabrMetrics.D_ALPHA_D_FORWARD]
            )

            # theta
            value_and_sensitivities[SimpleMetrics.THETA] -= (
                dvdalpha * alpha_and_sensitivities[SabrMetrics.D_ALPHA_D_TTE]
            )

            # normal atm vol risk
            value_and_sensitivities[SabrMetrics.DNORMALSIGMA] = (
                dvdalpha * alpha_and_sensitivities[SabrMetrics.D_ALPHA_D_NORMAL_SIGMA_ATM]
            )

            # sabr beta/rho/nu
            value_and_sensitivities[SabrMetrics.DBETA] += (
                dvdalpha * alpha_and_sensitivities[SabrMetrics.D_ALPHA_D_BETA]
            )
            value_and_sensitivities[SabrMetrics.DRHO] += (
                dvdalpha * alpha_and_sensitivities[SabrMetrics.D_ALPHA_D_RHO]
            )
            value_and_sensitivities[SabrMetrics.DNU] += (
                dvdalpha * alpha_and_sensitivities[SabrMetrics.D_ALPHA_D_NU]
            )

            ## second order risk (bump reval)
            v_base = value_and_sensitivities[SimpleMetrics.PV]

            # gamma
            res_up = SABRAnalytics.alpha_from_atm_normal_sigma(
                forward + SABRAnalytics.EPSILON,
                time_to_expiry,
                normal_sigma_atm,
                beta,
                rho,
                nu,
                shift,
                False,
            )
            alpha_up = res_up[SabrMetrics.ALPHA]
            v_up = SABRAnalytics.european_option_alpha(
                forward + SABRAnalytics.EPSILON,
                strike,
                time_to_expiry,
                opt_type,
                alpha_up,
                beta,
                rho,
                nu,
                shift,
                False,
            )[SimpleMetrics.PV]

            res_dn = SABRAnalytics.alpha_from_atm_normal_sigma(
                forward - SABRAnalytics.EPSILON,
                time_to_expiry,
                normal_sigma_atm,
                beta,
                rho,
                nu,
                shift,
                False,
            )
            alpha_dn = res_dn[SabrMetrics.ALPHA]
            v_dn = SABRAnalytics.european_option_alpha(
                forward - SABRAnalytics.EPSILON,
                strike,
                time_to_expiry,
                opt_type,
                alpha_dn,
                beta,
                rho,
                nu,
                shift,
                False,
            )[SimpleMetrics.PV]

            value_and_sensitivities[SimpleMetrics.GAMMA] = (
                v_up - 2.0 * v_base + v_dn
            ) / (SABRAnalytics.EPSILON**2)

        return value_and_sensitivities

   

    ### helpers

    @staticmethod
    def w2_risk(F, K, T, a, b, r, n) -> Dict:

        risk = {}

        risk[SabrMetrics.DALPHA] = (1 - b) ** 2 / 12 * a / (F * K) ** (1 - b) + b * r * n / (
            4 * (F * K) ** ((1 - b) / 2)
        )
        risk[SabrMetrics.DBETA] = (
            1 / 12 * (b - 1) * a**2 * (F * K) ** (b - 1)
            + 1 / 24 * (b - 1) ** 2 * a**2 * (F * K) ** (b - 1) * np.log(F * K)
            + 1 / 4 * a * r * n * (F * K) ** ((b - 1) / 2)
            + 1 / 8 * a * b * r * n * (F * K) ** ((b - 1) / 2) * np.log(F * K)
        )
        risk[SabrMetrics.DRHO] = 1 / 4 * a * b * n * (F * K) ** ((b - 1) / 2) - 1 / 4 * n**2 * r
        risk[SabrMetrics.DNU] = (
            1 / 4 * a * b * r * (F * K) ** ((b - 1) / 2) + 1 / 6 * n - 1 / 4 * r**2 * n
        )
        risk[SabrMetrics.DFORWARD] = (b - 1) ** 3 / 24 * a**2 * (F * K) ** (
            b - 2
        ) * K + a * r * n * b * (b - 1) / 8 * K ** ((b - 1) / 2) * F ** ((b - 3) / 2)

        risk[SabrMetrics.DSTRIKE] = (b - 1) ** 3 / 24 * a**2 * F ** (b - 1) * K ** (
            b - 2
        ) + a * b * r * n * (b - 1) / 8 * F ** ((b - 1) / 2) * K ** ((b - 3) / 2)

        risk[SabrMetrics.DSTRIKESTRIKE] = (b - 1) ** 3 / 24 * a**2 * (b - 2) * F ** (
            b - 1
        ) * K ** (b - 3) + a * b * r * n / 16 * (b - 1) * (b - 3) * F ** ((b - 1) / 2) * K ** (
            (b - 5) / 2
        )

        return risk

    @staticmethod
    def w1_risk(F, K, T, a, b, r, n) -> Dict:

        log_FK = np.log(F / K)

        risk = {}
        risk[SabrMetrics.DALPHA] = 0.0
        risk[SabrMetrics.DBETA] = (b - 1) / 12.0 * log_FK**2 + (b - 1) ** 3 / 480 * log_FK**4
        risk[SabrMetrics.DRHO] = 0.0
        risk[SabrMetrics.DNU] = 0.0
        risk[SabrMetrics.DFORWARD] = (b - 1) ** 2 / 12 * log_FK / F + (
            b - 1
        ) ** 4 / 480 / F * log_FK**3
        risk[SabrMetrics.DSTRIKE] = (
            -((b - 1) ** 2) / 12 * log_FK / K - (b - 1) ** 4 / 480 / K * log_FK**3
        )
        risk[SabrMetrics.DSTRIKESTRIKE] = (
            (b - 1) ** 2 / 12 / K**2
            + (b - 1) ** 2 / 12 * log_FK / K**2
            + (b - 1) ** 4 / 160 * log_FK**2 / K**2
            + (b - 1) ** 4 / 480 * log_FK**3 / K**2
        )

        return risk

    @staticmethod
    def z_risk(F, K, T, a, b, r, n) -> Dict:

        log_FK = np.log(F / K)
        fk = (F * K) ** ((1 - b) / 2)
        # z = n / a * log_FK * fk

        risk = {}
        risk[SabrMetrics.DALPHA] = -n / a * log_FK * fk / a
        risk[SabrMetrics.DBETA] = -1.0 / 2 * n / a * log_FK * fk * np.log(F * K)
        risk[SabrMetrics.DRHO] = 0.0
        risk[SabrMetrics.DNU] = 1.0 / a * log_FK * fk
        risk[SabrMetrics.DFORWARD] = (
            n * (1 - b) * K / 2 / a * (F * K) ** ((-b - 1) / 2) * log_FK + n / a * fk / F
        )
        risk[SabrMetrics.DSTRIKE] = (
            n * F * (1 - b) / 2 / a * log_FK * (F * K) ** ((-b - 1) / 2) - n / a * fk / K
        )
        risk[SabrMetrics.DSTRIKESTRIKE] = (
            n / a * F ** ((1 - b) / 2) * K ** ((-b - 3) / 2) * (log_FK * (b**2 - 1) / 4 + b)
        )

        return risk

    @staticmethod
    def x_risk(F, K, T, a, b, r, n) -> Dict:

        logFK = np.log(F / K)
        fk = (F * K) ** ((1 - b) / 2)
        z = n / a * fk * logFK
        dx_dz = 1 / np.sqrt(1 - 2 * r * z + z**2)

        risk = {}
        risk_z = SABRAnalytics.z_risk(F, K, T, a, b, r, n)

        risk[SabrMetrics.DALPHA] = dx_dz * risk_z[SabrMetrics.DALPHA]
        risk[SabrMetrics.DBETA] = dx_dz * risk_z[SabrMetrics.DBETA]
        risk[SabrMetrics.DRHO] = 1 / (1 - r) + (-z * dx_dz - 1) / (1 / dx_dz + z - r)
        risk[SabrMetrics.DNU] = dx_dz * risk_z[SabrMetrics.DNU]
        risk[SabrMetrics.DFORWARD] = dx_dz * risk_z[SabrMetrics.DFORWARD]
        risk[SabrMetrics.DSTRIKE] = dx_dz * risk_z[SabrMetrics.DSTRIKE]

        risk[SabrMetrics.DSTRIKESTRIKE] = (r - z) * dx_dz**3 * (
            risk_z[SabrMetrics.DSTRIKE] ** 2
        ) + dx_dz * risk_z[SabrMetrics.DSTRIKESTRIKE]

        return risk

    @staticmethod
    def C_risk(F, K, T, a, b, r, n) -> Dict:

        log_FK = np.log(F / K)
        fk = (F * K) ** ((1 - b) / 2)

        z = n / a * log_FK * fk
        risk = {}

        C0 = 1.0
        C1 = -r / 2.0
        C2 = -(r**2) / 4.0 + 1.0 / 6.0
        C3 = -(1.0 / 4.0 * r**2 - 5.0 / 24.0) * r
        C4 = -5.0 / 16.0 * r**4 + 1.0 / 3.0 * r**2 - 17.0 / 360.0
        C5 = -(7.0 / 16.0 * r**4 - 55.0 / 96.0 * r**2 + 37.0 / 240.0) * r

        dC_dz = C1 + 2 * C2 * z + 3 * C3 * z**2 + 4 * C4 * z**3 + 5 * C5 * z**4
        dC2_dz2 = 2 * C2 + 6 * C3 * z + 12 * C4 * z**2 + 20 * C5 * z**3

        risk[SabrMetrics.DRHO] = (
            -1.0 / 2 * z
            + 5.0 / 24 * z**3
            - 37.0 / 240 * z**5
            - 1.0 / 2 * z**2 * r
            + 2.0 / 3 * z**4 * r
            - 3.0 / 4 * z**3 * r**2
            + 55.0 / 32 * z**5 * r**2
            - 5.0 / 4 * z**4 * r**3
            - 35.0 / 16 * z**5 * r**4
        )
        risk_z = SABRAnalytics.z_risk(F, K, T, a, b, r, n)

        risk[SabrMetrics.DALPHA] = dC_dz * risk_z[SabrMetrics.DALPHA]
        risk[SabrMetrics.DBETA] = dC_dz * risk_z[SabrMetrics.DBETA]
        risk[SabrMetrics.DNU] = dC_dz * risk_z[SabrMetrics.DNU]
        risk[SabrMetrics.DFORWARD] = dC_dz * risk_z[SabrMetrics.DFORWARD]
        risk[SabrMetrics.DSTRIKE] = dC_dz * risk_z[SabrMetrics.DSTRIKE]
        risk[SabrMetrics.DSTRIKESTRIKE] = (
            dC_dz * risk_z[SabrMetrics.DSTRIKESTRIKE] + dC2_dz2 * risk_z[SabrMetrics.DSTRIKE] ** 2
        )
        return risk

    @staticmethod
    def _vol_and_risk(
        F, K, T, a, b, r, n, calc_risk=False, z_cut=1e-2
    ) -> Tuple[float, Dict[SabrMetrics, float]]:
        """
        Get analytical solution Lognormal Vol and Greeks
        """

        log_FK = np.log(F / K)
        fk = (F * K) ** ((1 - b) / 2)
        greeks: Dict[SabrMetrics, float] = {}

        z = n / a * log_FK * fk
        w1 = ((1 - b) ** 2 / 24.0) * log_FK**2 + ((1 - b) ** 4 / 1920.0) * log_FK**4
        w2 = (
            ((1 - b) ** 2 / 24.0) * a**2 / ((F * K) ** (1 - b))
            + 0.25 * r * b * n * a / fk
            + (2.0 - 3.0 * r**2) * n**2 / 24.0
        )

        pref = a / fk
        A = 1.0 + w2 * T
        B = 1.0 + w1
        
        pref_risk = {
            SabrMetrics.DALPHA: 1.0 / fk,
            SabrMetrics.DBETA: 0.5 * np.log(F * K) * pref,
            SabrMetrics.DRHO: 0.0,
            SabrMetrics.DNU: 0.0,
            SabrMetrics.DFORWARD: -((1.0 - b) / (2.0 * F)) * pref,
            SabrMetrics.DSTRIKE: -((1.0 - b) / (2.0 * K)) * pref,
            SabrMetrics.DSTRIKESTRIKE: ((1.0 - b) * (3.0 - b) / (4.0 * K**2)) * pref,
            SabrMetrics.DTTE: 0.0,
        }

        w1_r = SABRAnalytics.w1_risk(F, K, T, a, b, r, n)
        w2_r = SABRAnalytics.w2_risk(F, K, T, a, b, r, n)


        if abs(z) < z_cut:
            # expansion when z is small
            # calculate vol and risk, you can use the helper functions above w2_risk, w1_risk, z_risk, x_risk, C_risk
            # to get the risk for each component and then aggregate them to get the risk for vol
            # expansion when z is small: z / x(z) ~ C(z)
            C0 = 1.0
            C1 = -r / 2.0
            C2 = -(r**2) / 4.0 + 1.0 / 6.0
            C3 = -(1.0 / 4.0 * r**2 - 5.0 / 24.0) * r
            C4 = -5.0 / 16.0 * r**4 + 1.0 / 3.0 * r**2 - 17.0 / 360.0
            C5 = -(7.0 / 16.0 * r**4 - 55.0 / 96.0 * r**2 + 37.0 / 240.0) * r

            C = C0 + C1 * z + C2 * z**2 + C3 * z**3 + C4 * z**4 + C5 * z**5
            sigma = pref * C * A / B
            if calc_risk:
                C_r = SABRAnalytics.C_risk(F, K, T, a, b, r, n)

                for key in [
                    SabrMetrics.DALPHA,
                    SabrMetrics.DBETA,
                    SabrMetrics.DRHO,
                    SabrMetrics.DNU,
                    SabrMetrics.DFORWARD,
                    SabrMetrics.DSTRIKE,
                ]:
                    greeks_map = (
                        pref_risk[key] / pref
                        + C_r[key] / C
                        + T * w2_r[key] / A
                        - w1_r[key] / B
                    )
                    greeks[
                        {
                            SabrMetrics.DALPHA: SabrMetrics.D_LN_SIGMA_D_ALPHA,
                            SabrMetrics.DBETA: SabrMetrics.D_LN_SIGMA_D_BETA,
                            SabrMetrics.DRHO: SabrMetrics.D_LN_SIGMA_D_RHO,
                            SabrMetrics.DNU: SabrMetrics.D_LN_SIGMA_D_NU,
                            SabrMetrics.DFORWARD: SabrMetrics.D_LN_SIGMA_D_FORWARD,
                            SabrMetrics.DSTRIKE: SabrMetrics.D_LN_SIGMA_D_STRIKE,
                        }[key]
                    ] = sigma * greeks_map

                greeks[SabrMetrics.D_LN_SIGMA_D_TTE] = sigma * (w2 / A)

                # second derivative wrt strike
                dlog_pref_dK = pref_risk[SabrMetrics.DSTRIKE] / pref
                d2log_pref_dK2 = (1.0 - b) / (2.0 * K**2)

                dlog_A_dK = T * w2_r[SabrMetrics.DSTRIKE] / A
                d2log_A_dK2 = (
                    T * w2_r[SabrMetrics.DSTRIKESTRIKE] / A
                    - (T * w2_r[SabrMetrics.DSTRIKE] / A) ** 2
                )

                dlog_B_dK = w1_r[SabrMetrics.DSTRIKE] / B
                d2log_B_dK2 = (
                    w1_r[SabrMetrics.DSTRIKESTRIKE] / B
                    - (w1_r[SabrMetrics.DSTRIKE] / B) ** 2
                )

                dlog_C_dK = C_r[SabrMetrics.DSTRIKE] / C
                d2log_C_dK2 = (
                    C_r[SabrMetrics.DSTRIKESTRIKE] / C
                    - (C_r[SabrMetrics.DSTRIKE] / C) ** 2
                )

                h1 = dlog_pref_dK + dlog_A_dK - dlog_B_dK + dlog_C_dK
                h2 = d2log_pref_dK2 + d2log_A_dK2 - d2log_B_dK2 + d2log_C_dK2

                greeks[SabrMetrics.D_LN_SIGMA_D_STRIKESTRIKE] = sigma * (h1**2 + h2)

            return sigma, greeks

        # raw SABR
        x = np.log((np.sqrt(1.0 - 2.0 * r * z + z**2) + z - r) / (1.0 - r))
        C = z / x
        sigma = pref * C * A / B

        if calc_risk:
            z_r = SABRAnalytics.z_risk(F, K, T, a, b, r, n)
            x_r = SABRAnalytics.x_risk(F, K, T, a, b, r, n)

            # first derivatives of C = z / x
            C_r = {}
            for key in [
                SabrMetrics.DALPHA,
                SabrMetrics.DBETA,
                SabrMetrics.DRHO,
                SabrMetrics.DNU,
                SabrMetrics.DFORWARD,
                SabrMetrics.DSTRIKE,
            ]:
                C_r[key] = z_r[key] / x - z * x_r[key] / (x**2)

            # second derivative wrt strike
            C_r[SabrMetrics.DSTRIKESTRIKE] = (
                z_r[SabrMetrics.DSTRIKESTRIKE] / x
                - z * x_r[SabrMetrics.DSTRIKESTRIKE] / (x**2)
                - 2.0 * z_r[SabrMetrics.DSTRIKE] * x_r[SabrMetrics.DSTRIKE] / (x**2)
                + 2.0 * z * (x_r[SabrMetrics.DSTRIKE] ** 2) / (x**3)
            )

            for key in [
                SabrMetrics.DALPHA,
                SabrMetrics.DBETA,
                SabrMetrics.DRHO,
                SabrMetrics.DNU,
                SabrMetrics.DFORWARD,
                SabrMetrics.DSTRIKE,
            ]:
                greeks_map = (
                    pref_risk[key] / pref
                    + C_r[key] / C
                    + T * w2_r[key] / A
                    - w1_r[key] / B
                )
                greeks[
                    {
                        SabrMetrics.DALPHA: SabrMetrics.D_LN_SIGMA_D_ALPHA,
                        SabrMetrics.DBETA: SabrMetrics.D_LN_SIGMA_D_BETA,
                        SabrMetrics.DRHO: SabrMetrics.D_LN_SIGMA_D_RHO,
                        SabrMetrics.DNU: SabrMetrics.D_LN_SIGMA_D_NU,
                        SabrMetrics.DFORWARD: SabrMetrics.D_LN_SIGMA_D_FORWARD,
                        SabrMetrics.DSTRIKE: SabrMetrics.D_LN_SIGMA_D_STRIKE,
                    }[key]
                ] = sigma * greeks_map

            greeks[SabrMetrics.D_LN_SIGMA_D_TTE] = sigma * (w2 / A)

            # second derivative wrt strike
            dlog_pref_dK = pref_risk[SabrMetrics.DSTRIKE] / pref
            d2log_pref_dK2 = (1.0 - b) / (2.0 * K**2)

            dlog_A_dK = T * w2_r[SabrMetrics.DSTRIKE] / A
            d2log_A_dK2 = (
                T * w2_r[SabrMetrics.DSTRIKESTRIKE] / A
                - (T * w2_r[SabrMetrics.DSTRIKE] / A) ** 2
            )

            dlog_B_dK = w1_r[SabrMetrics.DSTRIKE] / B
            d2log_B_dK2 = (
                w1_r[SabrMetrics.DSTRIKESTRIKE] / B
                - (w1_r[SabrMetrics.DSTRIKE] / B) ** 2
            )

            dlog_C_dK = C_r[SabrMetrics.DSTRIKE] / C
            d2log_C_dK2 = (
                C_r[SabrMetrics.DSTRIKESTRIKE] / C
                - (C_r[SabrMetrics.DSTRIKE] / C) ** 2
            )

            h1 = dlog_pref_dK + dlog_A_dK - dlog_B_dK + dlog_C_dK
            h2 = d2log_pref_dK2 + d2log_A_dK2 - d2log_B_dK2 + d2log_C_dK2

            greeks[SabrMetrics.D_LN_SIGMA_D_STRIKESTRIKE] = sigma * (h1**2 + h2)

        return sigma, greeks

 ### optional: if you want to calculate analytical pdf and cdf of the SABR model

    @staticmethod
    def simulate_terminal_distribution(
        forward: float,
        time_to_expiry: float,
        alpha: float,
        beta: float,
        rho: float,
        nu: float,
        step_size: float,
        num_paths: int,
        shift: Optional[float] = 0.0,
        seed: Optional[int] = None,
    ) -> Dict[str, np.ndarray | float | int]:
        """
        Simulate shifted SABR terminal distribution with an Euler scheme.

        On shifted forward X_t = F_t + shift:
            dX_t = sigma_t * X_t^beta dW1_t
            d sigma_t = nu * sigma_t dW2_t
            corr(dW1_t, dW2_t) = rho
        """

        if forward + shift <= 0.0:
            raise ValueError("forward + shift must be positive")
        if time_to_expiry <= 0.0:
            raise ValueError("time_to_expiry must be positive")
        if alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if nu < 0.0:
            raise ValueError("nu must be non-negative")
        if abs(rho) >= 1.0:
            raise ValueError("rho must be in (-1, 1)")
        if not (0.0 <= beta <= 1.0):
            raise ValueError("beta must be in [0, 1]")
        if step_size <= 0.0:
            raise ValueError("step_size must be positive")
        if num_paths <= 0:
            raise ValueError("num_paths must be positive")

        n_steps = max(1, int(np.ceil(time_to_expiry / step_size)))
        dt = time_to_expiry / n_steps
        sqrt_dt = np.sqrt(dt)
        sqrt_one_minus_rho2 = np.sqrt(1.0 - rho**2)

        rng = np.random.default_rng(seed)
        floor = 1e-12

        x = np.full(num_paths, forward + shift, dtype=float)
        sigma = np.full(num_paths, alpha, dtype=float)

        for _ in range(n_steps):
            z1 = rng.standard_normal(num_paths)
            z2 = rng.standard_normal(num_paths)

            dW1 = sqrt_dt * z1
            dW2 = sqrt_dt * (rho * z1 + sqrt_one_minus_rho2 * z2)

            x = x + sigma * np.power(np.maximum(x, floor), beta) * dW1
            x = np.maximum(x, floor)

            if nu > 0.0:
                # exact lognormal update for SABR volatility process
                sigma = sigma * np.exp(-0.5 * nu * nu * dt + nu * dW2)

        terminal_forward = x - shift

        return {
            "terminal_forward": terminal_forward,
            "terminal_shifted_forward": x,
            "terminal_vol": sigma,
            "num_steps": n_steps,
            "dt": dt,
            "num_paths": num_paths,
        }

    @staticmethod
    def _build_quantile_nodes(
        forward: float,
        time_to_expiry: float,
        alpha: float,
        beta: float,
        rho: float,
        nu: float,
        shift: float,
        sigma_atm_normal: Optional[float] = None,
        num_grid: int = 801,
    ) -> Dict[str, np.ndarray]:
        """
        Build monotone quantile map nodes (y, x) where x ~= I(y), y ~ N(0, 1).
        """
        if num_grid < 11:
            raise ValueError("num_grid must be >= 11")

        if sigma_atm_normal is None:
            sigma_atm_normal = SABRAnalytics.atm_normal_sigma_from_alpha(
                forward=forward,
                time_to_expiry=time_to_expiry,
                alpha=alpha,
                beta=beta,
                rho=rho,
                nu=nu,
                shift=shift,
                calc_risk=False,
            )[SimpleMetrics.IMPLIED_NORMAL_VOL]

        approx_std = max(float(sigma_atm_normal) * np.sqrt(time_to_expiry), 1e-5)
        lower_bound = max(-shift + 1e-10, forward - 8.0 * approx_std)
        upper_bound = max(lower_bound + 20.0 * approx_std, forward + 8.0 * approx_std)
        x_grid = np.linspace(lower_bound, upper_bound, num_grid)

        dist = SABRAnalytics.pdf_and_cdf(
            forward=forward,
            time_to_expiry=time_to_expiry,
            alpha=alpha,
            beta=beta,
            rho=rho,
            nu=nu,
            grids=x_grid,
            shift=shift,
        )
        cdf = np.asarray(dist["cdf"], dtype=float)
        eps = 1e-10
        cdf = np.clip(cdf, eps, 1.0 - eps)
        cdf = np.maximum.accumulate(cdf)

        keep = np.r_[True, np.diff(cdf) > 1e-12]
        u_nodes = cdf[keep]
        x_nodes = x_grid[keep]

        if u_nodes.shape[0] < 2:
            u_nodes = np.array([eps, 1.0 - eps], dtype=float)
            x_nodes = np.array([x_grid[0], x_grid[-1]], dtype=float)
        else:
            if u_nodes[0] > eps:
                u_nodes = np.r_[eps, u_nodes]
                x_nodes = np.r_[x_nodes[0], x_nodes]
            if u_nodes[-1] < 1.0 - eps:
                u_nodes = np.r_[u_nodes, 1.0 - eps]
                x_nodes = np.r_[x_nodes, x_nodes[-1]]

        y_nodes = norm.ppf(u_nodes)
        return {"x_nodes": x_nodes, "u_nodes": u_nodes, "y_nodes": y_nodes}

    @staticmethod
    def quantile_map_spread_option_price_from_normal_atm(
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
    ) -> Dict[str, Any]:
        """
        Assignment Q4 quantile-map spread option pricing.
        """
        if abs(corr_12) >= 1.0:
            raise ValueError("corr_12 must be in (-1,1)")
        if num_paths <= 0:
            raise ValueError("num_paths must be positive")
        if time_to_expiry <= 0.0:
            raise ValueError("time_to_expiry must be positive")

        alpha_1 = SABRAnalytics.alpha_from_atm_normal_sigma(
            forward=forward_1,
            time_to_expiry=time_to_expiry,
            sigma_atm_normal=sigma_atm_normal_1,
            beta=beta_1,
            rho=rho_1,
            nu=nu_1,
            shift=shift_1,
            calc_risk=False,
        )[SabrMetrics.ALPHA]
        alpha_2 = SABRAnalytics.alpha_from_atm_normal_sigma(
            forward=forward_2,
            time_to_expiry=time_to_expiry,
            sigma_atm_normal=sigma_atm_normal_2,
            beta=beta_2,
            rho=rho_2,
            nu=nu_2,
            shift=shift_2,
            calc_risk=False,
        )[SabrMetrics.ALPHA]

        map_1 = SABRAnalytics._build_quantile_nodes(
            forward=forward_1,
            time_to_expiry=time_to_expiry,
            alpha=alpha_1,
            beta=beta_1,
            rho=rho_1,
            nu=nu_1,
            shift=shift_1,
            sigma_atm_normal=sigma_atm_normal_1,
            num_grid=num_grid,
        )
        map_2 = SABRAnalytics._build_quantile_nodes(
            forward=forward_2,
            time_to_expiry=time_to_expiry,
            alpha=alpha_2,
            beta=beta_2,
            rho=rho_2,
            nu=nu_2,
            shift=shift_2,
            sigma_atm_normal=sigma_atm_normal_2,
            num_grid=num_grid,
        )

        if use_sobol and qmc is not None:
            sobol = qmc.Sobol(d=2, scramble=True, seed=seed)
            if num_paths > 0 and (num_paths & (num_paths - 1) == 0):
                # Use balanced Sobol blocks when path count is a power of two.
                m = int(np.log2(num_paths))
                s = sobol.random_base2(m=m)
            else:
                # Fallback for arbitrary path counts (still low-discrepancy).
                s = sobol.random(num_paths)
        else:
            rng = np.random.default_rng(seed)
            s = rng.random((num_paths, 2))

        eps = 1e-12
        s = np.clip(s, eps, 1.0 - eps)
        z = norm.ppf(s)

        corr = np.array([[1.0, corr_12], [corr_12, 1.0]], dtype=float)
        b = np.linalg.cholesky(corr)
        y = z @ b.T

        x1 = np.interp(y[:, 0], map_1["y_nodes"], map_1["x_nodes"])
        x2 = np.interp(y[:, 1], map_2["y_nodes"], map_2["x_nodes"])
        payoff = np.maximum(x1 - x2 - strike, 0.0)

        price = float(np.mean(payoff))
        std_error = float(np.std(payoff, ddof=1) / np.sqrt(num_paths))
        out = {
            "price": price,
            "std_error": std_error,
            "num_paths": num_paths,
            "alpha_1": float(alpha_1),
            "alpha_2": float(alpha_2),
        }
        if return_paths:
            out["x1_samples"] = x1
            out["x2_samples"] = x2
            out["payoff_samples"] = payoff
        return out

    # @staticmethod
    # def pdf_and_cdf(
    #     forward: float,
    #     time_to_expiry: float,
    #     alpha: float,
    #     beta: float,
    #     rho: float,
    #     nu: float,
    #     grids: List | np.ndarray,
    #     shift: Optional[float] = 0,
    # ):
    #     pass
    @staticmethod
    def pdf_and_cdf(
        forward: float,
        time_to_expiry: float,
        alpha: float,
        beta: float,
        rho: float,
        nu: float,
        grids: List | np.ndarray,
        shift: Optional[float] = 0,
    ):
        """
        Compute SABR-implied pdf and cdf on a strike grid using
        Breeden-Litzenberger relations under the forward measure.

        For undiscounted call prices C(K):
            pdf(K) = d^2 C / dK^2
            cdf(K) = P(S_T <= K) = 1 + dC/dK
        """

        grids = np.asarray(grids, dtype=float)
        if grids.ndim != 1:
            raise ValueError("grids must be a 1D array-like")
        if len(grids) < 3:
            raise ValueError("grids must contain at least 3 points")
        if np.any(grids + shift <= 0.0):
            raise ValueError("all grid strikes plus shift must be positive")
        if np.any(np.diff(grids) <= 0.0):
            raise ValueError("grids must be strictly increasing")

        # call prices on the strike grid
        call_prices = np.array(
            [
                SABRAnalytics.european_option_alpha(
                    forward=forward,
                    strike=k,
                    time_to_expiry=time_to_expiry,
                    opt_type=CallOrPut.CALL,
                    alpha=alpha,
                    beta=beta,
                    rho=rho,
                    nu=nu,
                    shift=shift,
                    calc_risk=False,
                )[SimpleMetrics.PV]
                for k in grids
            ],
            dtype=float,
        )

        n = len(grids)
        pdf = np.zeros(n, dtype=float)
        dC_dK = np.zeros(n, dtype=float)

        # interior points: non-uniform central differences
        for i in range(1, n - 1):
            h_minus = grids[i] - grids[i - 1]
            h_plus = grids[i + 1] - grids[i]

            dC_dK[i] = (
                -h_plus / (h_minus * (h_minus + h_plus)) * call_prices[i - 1]
                + (h_plus - h_minus) / (h_minus * h_plus) * call_prices[i]
                + h_minus / (h_plus * (h_minus + h_plus)) * call_prices[i + 1]
            )

            pdf[i] = 2.0 * (
                call_prices[i - 1] / (h_minus * (h_minus + h_plus))
                - call_prices[i] / (h_minus * h_plus)
                + call_prices[i + 1] / (h_plus * (h_minus + h_plus))
            )

        # left boundary: simple one-sided approximations
        h0 = grids[1] - grids[0]
        dC_dK[0] = (-3.0 * call_prices[0] + 4.0 * call_prices[1] - call_prices[2]) / (2.0 * h0)
        pdf[0] = (call_prices[0] - 2.0 * call_prices[1] + call_prices[2]) / (h0**2)

        # right boundary
        hm0 = grids[-1] - grids[-2]
        dC_dK[-1] = (3.0 * call_prices[-1] - 4.0 * call_prices[-2] + call_prices[-3]) / (
            2.0 * hm0
        )
        pdf[-1] = (call_prices[-3] - 2.0 * call_prices[-2] + call_prices[-1]) / (hm0**2)

        # under forward measure: dC/dK = -(1 - CDF)
        cdf = 1.0 + dC_dK

        # keep raw pdf for negative-density detection
        # only clip cdf lightly for numerical noise
        cdf = np.clip(cdf, -1e-10, 1.0 + 1e-10)

        return {
            "grids": grids,
            "call_prices": call_prices,
            "pdf": pdf,
            "cdf": cdf,
        }
