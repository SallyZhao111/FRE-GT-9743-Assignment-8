from typing import Union, Dict, Any, Tuple
from datetime import datetime

from sympy import prod
from fixedincomelib.market import *
from fixedincomelib.date import TermOrTerminationDate, Date, add_period
from fixedincomelib.product.product_interfaces import ProductBuilderRegistry
from fixedincomelib.product.linear_products import (
    ProductOvernightIndexBasisSwap,
    ProductRFRFuture,
    ProductRFRSwap,
    ProductZeroSpread,
    BondSpecs,
    ProductBond,
)
from fixedincomelib.product.utilities import LongOrShort, PayOrReceive

from fixedincomelib.product.linear_products import (
    ProductFxForward,
    ProductOvernightIndexBasisSwap,
    ProductRFRFuture,
    ProductRFRSwap,
    ProductZeroSpread,
)
from fixedincomelib.product.utilities import LongOrShort, PayOrReceive
# from fixedincomelib.analytics import BondUtils

class ProductFactory:

    @classmethod
    def create_product_from_data_convention(
        cls,
        value_date: Date,
        axis1: str,
        data_convention: DataConvention,
        values: float,
        **kwargs: Any,
    ):

        convention_obj: DataConvention = data_convention
        prod_type = convention_obj.type()
        func = ProductBuilderRegistry().get(prod_type)
        return func(value_date, axis1, convention_obj, values, **kwargs)

    @classmethod
    def create_rfr_future(
        cls,
        value_date: Date,
        axis1: str,
        data_convention: DataConventionRFRFuture,
        values: float,
        **kwargs: Any,
    ) -> ProductRFRFuture:

        term_or_effective_date, term_or_termnation_date = ProductFactory._tokenize_axis1(axis1)
        if term_or_effective_date.is_term():
            raise Exception("Effective date is not valid.")
        if term_or_termnation_date is None:
            raise Exception("Term or Termination date is missing.")
        long_or_short = LongOrShort.from_string(kwargs.get("long_or_short", "long"))
        amount = kwargs.get("amount", 1.0)
        return ProductRFRFuture(
            effective_date=term_or_effective_date.get_date(),
            term_or_termination_date=term_or_termnation_date,
            future_conv=data_convention.name,
            long_or_short=long_or_short,
            amount=amount,
            strike=values,
        )

    @classmethod
    def create_rfr_swap(
        cls,
        value_date: Date,
        axis1: str,
        data_convention: DataConventionRFRSwap,
        values: float,
        **kwargs: Any,
    ) -> ProductRFRSwap:

        term_or_effective_date, term_or_termination_date = ProductFactory._tokenize_axis1(axis1)

        pay_offset = data_convention.payment_offset
        on_index_str = data_convention.index_str
        on_index = data_convention.index
        accrual_period = data_convention.acc_period
        accrual_basis = data_convention.acc_basis
        pay_buinsess_day_convention = data_convention.business_day_convention
        pay_hoiday_convention = data_convention.holiday_convention
        pay_or_rec = kwargs.get("pay_or_rec", "receive")
        spread = kwargs.get("spread", 0.0)
        compounding_method = CompoundingMethod.from_string(
            kwargs.get("compound_method", "compound")
        )

        effective_date = value_date
        if term_or_termination_date is None:
            # spot starting
            effective_date = on_index.fixingDate(value_date)
            term_or_termination_date = term_or_effective_date
        else:
            # forwad starting
            if term_or_effective_date.is_term():
                this_date = on_index.fixingDate(value_date)
                effective_date = add_period(
                    this_date,
                    term_or_effective_date.get_term(),
                    on_index.businessDayConvention(),
                    pay_hoiday_convention,
                )  # fix holiday convention
            else:
                effective_date = term_or_effective_date.get_date()

        return ProductRFRSwap(
            effective_date=effective_date,
            term_or_termination_date=term_or_termination_date,
            payment_off_set=pay_offset,
            on_index=on_index_str,
            fixed_rate=values,
            pay_or_rec=PayOrReceive.from_string(pay_or_rec),
            notional=kwargs.get("notinoal", 1e4),
            accrual_period=accrual_period,
            accrual_basis=accrual_basis,
            floating_leg_accrual_period=accrual_period,
            pay_business_day_convention=pay_buinsess_day_convention,
            pay_holiday_convention=pay_hoiday_convention,
            spread=spread,
            compounding_method=compounding_method,
        )

    @classmethod
    def create_overnight_index_basis_swap(
        cls,
        value_date: Date,
        axis1: str,
        data_convention: DataConventionOvernightIndexBasisSwap,
        values: float,
        **kwargs: Any,
    ) -> ProductOvernightIndexBasisSwap:

        term_or_effective_date, term_or_termination_date = ProductFactory._tokenize_axis1(axis1)

        pay_offset = data_convention.payment_offset
        on_index_1_str = data_convention.index_1_str
        on_index_2_str = data_convention.index_2_str
        on_index_1 = data_convention.index_1
        on_index_2 = data_convention.index_2
        accrual_period_1 = data_convention.acc_period_1
        accrual_period_2 = data_convention.acc_period_2
        accrual_basis = data_convention.acc_basis
        pay_buinsess_day_convention = data_convention.business_day_convention
        pay_hoiday_convention = data_convention.holiday_convention
        pay_or_rec = kwargs.get("pay_or_rec", "receive")
        compounding_method = CompoundingMethod.from_string(
            kwargs.get("compound_method", "compound")
        )

        effective_date = value_date
        if term_or_termination_date is None:
            # spot starting
            effective_date_1 = on_index_1.fixingDate(value_date)
            effective_date_2 = on_index_2.fixingDate(value_date)
            assert effective_date_1 == effective_date_2
            effective_date = effective_date_1
            term_or_termination_date = term_or_effective_date
        else:
            # forwad starting
            if term_or_effective_date.is_term():
                this_date = on_index_1.fixingDate(value_date)
                effective_date = add_period(
                    this_date,
                    term_or_effective_date.get_term(),
                    on_index_1.businessDayConvention(),
                    pay_hoiday_convention,
                )
            else:
                effective_date = term_or_effective_date.get_date()

        return ProductOvernightIndexBasisSwap(
            effective_date=effective_date,
            term_or_termination_date=term_or_termination_date,
            payment_off_set=pay_offset,
            on_index_1=on_index_1_str,
            on_index_2=on_index_2_str,
            spread_over_leg_1=values,
            pay_or_rec_leg_1=PayOrReceive.from_string(pay_or_rec),
            notional=kwargs.get("notinoal", 1e4),
            accrual_period_1=accrual_period_1,
            accrual_basis=accrual_basis,
            accrual_period_2=accrual_period_2,
            pay_business_day_convention=pay_buinsess_day_convention,
            pay_holiday_convention=pay_hoiday_convention,
            compounding_method=compounding_method,
        )

    @classmethod
    def create_zero_spread_product(
        cls,
        value_date: Date,
        axis1: str,
        data_convention: DataConventionZeroSpread,
        values: float,
        **kwargs: Any,
    ) -> ProductZeroSpread:

        index: ql.InterestRateIndex = data_convention.index
        tenor, _ = ProductFactory._tokenize_axis1(axis1)
        term_date = None
        if tenor.is_term():
            calendar: ql.QuantLib.Calendar = index.fixingCalendar()
            term_date = Date(
                calendar.advance(
                    value_date, tenor.get_term(), data_convention.business_day_convention.value
                )
            )
        else:
            term_date = tenor.get_date()
        long_or_short = LongOrShort.from_string(kwargs.get("long_or_short", "long"))
        amount = kwargs.get("amount", 1e4)
        return ProductZeroSpread(
            term_date, data_convention.index_str, values, amount, long_or_short
        )

    @classmethod
    def create_bond(
        cls,
        value_date: Date,
        axis1: str,  # bond name
        data_convention: DataConventionBondFixed,
        values: float = None,
        **kwargs: Any,
    ) -> ProductBond:

        bond_specs: BondSpecs = BondSpecsRegistry().get(axis1)
        trade_date = value_date
        buy_sell = LongOrShort.from_string(kwargs.get("buy_sell", "long"))
        
        values_ = values
        
        if values_ <= 1.:
            # product 
            prod = ProductBond(
                name=axis1,
                bond_specs=bond_specs,
                trade_date=trade_date,
                buy_sell=buy_sell.to_string())
            values_ = BondUtils.yield_to_price(prod, values)

        return ProductBond(
            name=axis1,
            bond_specs=bond_specs,
            trade_date=trade_date,
            buy_sell=buy_sell.to_string(),
            traded_price=values_,
        )

    @classmethod
    def create_fx_forward(
        cls,
        value_date: Date,
        axis1: str,
        data_convention: DataConventionFxPair,
        values: float,
        **kwargs: Any,
    ) -> ProductFxForward:

        index: FXIndex = data_convention.index
        tenor, _ = ProductFactory._tokenize_axis1(axis1)
        term_date = None
        if tenor.is_term():
            # TODO: @wanling, can we allow multiple holiday centers for add_period ???
            # .      as you can see, we do need it here
            term_date = add_period(
                value_date, tenor.get_term(), index.base_business_day_conv, index.base_holidays
            )
        else:
            term_date = tenor.get_date()

        return ProductFxForward(
            termination_date=term_date,
            fx_pair=index.name(),
            pay_or_receive=PayOrReceive.PAY,
            settlement_currency=index.quoted_ccy,
            foreign_notional=1e4,
            strike=values,
        )

    ### utilities
    @staticmethod
    def _tokenize_axis1(axis1: str):

        axis1 = axis1.strip()
        if "x" in axis1.lower():
            tokens = axis1.replace("X", "x").split("x")
            return TermOrTerminationDate(tokens[0]), TermOrTerminationDate(tokens[1])
        else:
            return TermOrTerminationDate(axis1), None


### support product factory
ProductBuilderRegistry().register(DataConventionRFRFuture.type(), ProductFactory.create_rfr_future)
ProductBuilderRegistry().register(DataConventionRFRSwap.type(), ProductFactory.create_rfr_swap)
ProductBuilderRegistry().register(
    DataConventionOvernightIndexBasisSwap.type(), ProductFactory.create_overnight_index_basis_swap
)
ProductBuilderRegistry().register(
    DataConventionZeroSpread.type(), ProductFactory.create_zero_spread_product
)
ProductBuilderRegistry().register(DataConventionBondFixed._type, ProductFactory.create_bond)

ProductBuilderRegistry().register(DataConventionFxPair.type(), ProductFactory.create_fx_forward)
