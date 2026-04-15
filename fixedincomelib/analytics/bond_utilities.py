import numpy as np
from typing import Optional, List

from fixedincomelib.market.bond_specs import BondSpecs

from ..product.linear_products import ProductBond, ProductFixedAccrued, ProductBulletCashflow
from fixedincomelib.date import *


class BondUtils:

    @staticmethod
    def clean_price(bond: ProductBond) -> float:
        return bond.traded_price

    @staticmethod
    def accrued_interest(bond: ProductBond) -> float:
        full_period_length = bond.ai_t + bond.period_lengths[0]  # ai_t + remaining
        # ai_t = accrued time/year
        freq = frequency_from_period(
            bond.conv.coupon_accrual_period
        )  # accrued return without considering coupon frequency
        ## TODO: check if divide freq or not
        return bond.current_coupon_rate * bond.ai_t / full_period_length * freq

    @staticmethod
    def accrued_interest_amount(bond: ProductBond) -> float:
        # Dollar accrued interest
        return BondUtils.accrued_interest(bond) * bond.face_value

    @staticmethod
    def yield_to_price(bond: ProductBond, ytm: float, clean: bool = True) -> float:

        coupon_rates = np.array(bond.coupon_rates)  # c_i
        period_lengths = np.array(bond.period_lengths)  # tau_i
        coupon_amounts = coupon_rates * period_lengths * bond.face_value

        t_vec_year = np.cumsum(period_lengths)
        discount = np.exp(-ytm * t_vec_year)  # exp(-y*T_i)
        coupon_pv = np.dot(coupon_amounts, discount)  # sum_i c_i * tau_i * exp(-y*T_i)

        t_principal_years = t_vec_year[-1]
        principal_pv = (
            bond.bond_specs.__getitem__(BondSpecs.REDEMPTION_PERCENTAGE)
            * bond.face_value
            * np.exp(-ytm * t_principal_years)
        )

        dirty_price = coupon_pv + principal_pv
        if clean:
            return dirty_price - BondUtils.accrued_interest_amount(bond)
        return dirty_price

    @classmethod
    def _zcb_yield_guess(cls, bond: ProductBond, dirty_price: float) -> float:
        # take bond as zero coupon bond, solve for yield
        # y = - ln(price/N/R)/T
        T = np.sum(bond.period_lengths)
        redemption = bond.bond_specs.__getitem__(BondSpecs.REDEMPTION_PERCENTAGE)
        return -np.log(dirty_price / (redemption * bond.face_value)) / T

    @classmethod
    def _price_derivatives(cls, bond: ProductBond, ytm: float) -> List[float]:

        coupon_rates = np.array(bond.coupon_rates)
        period_lengths = np.array(bond.period_lengths)
        coupon_amounts = coupon_rates * period_lengths * bond.face_value

        t_vec_year = np.cumsum(period_lengths)
        discount = np.exp(-ytm * t_vec_year)

        coupon_pv = np.dot(coupon_amounts, discount)
        coupon_dpdy = -np.dot(coupon_amounts * t_vec_year, discount)
        coupon_d2pdy2 = np.dot(coupon_amounts * t_vec_year**2, discount)

        redemption = bond.bond_specs.__getitem__(BondSpecs.REDEMPTION_PERCENTAGE)
        t_principal_years = t_vec_year[-1]
        principal_pv = redemption * bond.face_value * np.exp(-ytm * t_principal_years)
        principal_dpdy = (
            -redemption * bond.face_value * t_principal_years * np.exp(-ytm * t_principal_years)
        )
        principal_d2pdy2 = (
            redemption * bond.face_value * t_principal_years**2 * np.exp(-ytm * t_principal_years)
        )

        P = coupon_pv + principal_pv
        dPdy = coupon_dpdy + principal_dpdy
        d2Pdy2 = coupon_d2pdy2 + principal_d2pdy2

        return [P, dPdy, d2Pdy2]

    @staticmethod
    def price_to_yield(
        bond: ProductBond,
        price: float,
        clean: bool = True,
        max_iter: Optional[int] = 100,
        tol: Optional[float] = 1e-8,
    ) -> float:
        dirty_price = price + BondUtils.accrued_interest_amount(bond) if clean else price

        ytm = BondUtils._zcb_yield_guess(bond, dirty_price)
        # solve for yield using root finding
        for _ in range(max_iter):
            P, dPdy, d2Pdy2 = BondUtils._price_derivatives(bond, ytm)
            residual = P - dirty_price
            if abs(residual) < tol:
                break
            ytm -= residual / dPdy
        else:
            raise Exception("Failed to converge to a solution for yield.")

        P, dPdy, d2Pdy2 = BondUtils._price_derivatives(bond, ytm)

        return ytm, dPdy, d2Pdy2
