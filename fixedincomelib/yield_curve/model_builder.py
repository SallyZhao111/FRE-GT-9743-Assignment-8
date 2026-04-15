from ast import Dict
from unittest.util import sorted_list_difference
import numpy as np
from typing import List

from sympy import product
from fixedincomelib.date import *
from fixedincomelib.data import *
from fixedincomelib.market import *
from fixedincomelib.model import *
from fixedincomelib.product import *
from fixedincomelib.utilities import *
from fixedincomelib.valuation import *
from fixedincomelib.valuation.valuation_engine import ValuationRequest
from fixedincomelib.yield_curve.calibration_utils import YieldCurveCalibration
from fixedincomelib.yield_curve.build_method import (
    YieldCurveFXBuildMethod,
    YieldCurveIndexBuildMethod,
    YieldCurveBuildMethodCommon,
    YieldCurveFundingBuildMethod,
)
from fixedincomelib.yield_curve.yield_curve_model import YieldCurve, YieldCurveModelComponent
# from fixedincomelib.analytics.bond_utilities import BondUtils


class YieldCurveBuilder:

    @staticmethod
    def create_model_yield_curve(
        value_date: Date,
        data_collection: DataCollection,
        build_method_collection: BuildMethodCollection,
    ):

        # create the skelton
        model_yield_curve = YieldCurve(value_date, data_collection, build_method_collection)

        # parse build method collection
        build_methods_pack, residual_build_methods = (
            YieldCurveBuilder._sort_out_build_method_and_funding(
                build_method_collection, data_collection
            )
        )

        # components require calibration
        components_to_be_calibrated = []
        for _, bm_pack in build_methods_pack.items():

            # funding goes first
            build_methods_funding = bm_pack.get(YieldCurveFundingBuildMethod._build_method_type, [])
            for bm in build_methods_funding:
                this_bm: YieldCurveFundingBuildMethod = bm
                mkt_data_list = []
                for data_type in this_bm.calibration_instruments():
                    data_conv = this_bm[data_type]
                    if data_conv != "":
                        mkt_data = data_collection.get_data_from_data_collection(
                            data_type, data_conv
                        )
                        mkt_data_list.append(mkt_data)
                component = YieldCurveBuilder.prepare_calibrate_instruments(
                    value_date, mkt_data_list, this_bm, bm_pack[DataGeneric._data_shape]
                )
                model_yield_curve.set_model_component(this_bm.target, component)
                components_to_be_calibrated.append(
                    [component, bm_pack[YieldCurveBuildMethodCommon._build_method_type]]
                )

            # then index
            build_methods = bm_pack.get(YieldCurveIndexBuildMethod._build_method_type, [])
            for bm in build_methods:
                this_bm: YieldCurveIndexBuildMethod = bm
                mkt_data_list = []
                for data_type in this_bm.calibration_instruments():
                    data_conv = this_bm[data_type]
                    if data_conv != "":
                        mkt_data = data_collection.get_data_from_data_collection(
                            data_type, data_conv
                        )
                        mkt_data_list.append(mkt_data)
                component = YieldCurveBuilder.prepare_calibrate_instruments(
                    value_date, mkt_data_list, this_bm, bm_pack[DataGeneric._data_shape]
                )
                model_yield_curve.set_model_component(this_bm.target_index.name(), component)
                components_to_be_calibrated.append(
                    [component, bm_pack[YieldCurveBuildMethodCommon._build_method_type]]
                )

            # then spot fx
            build_methods = bm_pack.get(YieldCurveFXBuildMethod._build_method_type, [])
            for bm in build_methods:
                this_bm: YieldCurveFXBuildMethod = bm
                mkt_data_list = []
                for data_type in this_bm.calibration_instruments():
                    data_conv = this_bm[data_type]
                    if data_conv != "":
                        mkt_data = data_collection.get_data_from_data_collection(
                            data_type, data_conv
                        )
                        mkt_data_list.append(mkt_data)
                component = YieldCurveBuilder.prepare_calibrate_instruments(
                    value_date, mkt_data_list, this_bm, bm_pack[DataGeneric._data_shape]
                )
                model_yield_curve.set_model_component(this_bm.target_index.name(), component)
                components_to_be_calibrated.append(
                    [component, bm_pack[YieldCurveBuildMethodCommon._build_method_type]]
                )

            # then currency-based (xccy)
            # TODO: @wanling

        # components does not require calibration
        for this_bm in residual_build_methods:
            data_conv_ifr = this_bm.instantaneous_forward_rate
            if data_conv_ifr is not None:
                state_data = data_collection.get_data_from_data_collection(
                    "INSTANTANEOUS FORWARD RATE", data_conv_ifr.name
                )
                component = YieldCurveBuilder.calibrate_single_component_from_state_data(
                    value_date, data_conv_ifr, state_data, this_bm
                )
                model_yield_curve.set_model_component(this_bm.target_index.name(), component)

        # calibration
        for component, solver_info in components_to_be_calibrated:
            YieldCurveBuilder.calibrate_single_component_from_mkt_data(
                model_yield_curve, component, solver_info
            )

        return model_yield_curve

    @staticmethod
    def calibrate_single_component_from_state_data(
        value_date: Date,
        data_conv: DataConventionIFR,
        state_data: Data1D,
        build_method: YieldCurveIndexBuildMethod | YieldCurveFundingBuildMethod,
    ):

        time_to_anchored_dates = []
        values = []
        market_data = []
        for i in range(len(state_data.axis1)):
            this_x = state_data.axis1[i]
            if TermOrTerminationDate(this_x).is_term():
                # if it is term
                moved_date = add_period(
                    value_date,
                    Period(this_x),
                    data_conv.business_day_convention,
                    data_conv.holiday_convention,
                )
                time = accrued(value_date, moved_date)
            else:
                # if it is date
                time = accrued(value_date, Date(this_x))
            time_to_anchored_dates.append(time)
            values.append(state_data.values[i])
            market_data.append(
                [
                    "INSTANTANEOUS FORWARD RATE",
                    data_conv.conv_name,
                    this_x,
                    "",
                    state_data.values[i],
                    state_data.data_identifier.unit(),
                ]
            )

        # check if time instances are sorted
        assert np.all(np.diff(time_to_anchored_dates) >= 0)
        combined_data = np.asarray([time_to_anchored_dates, values])

        return YieldCurveModelComponent(
            value_date,
            build_method.target_index,
            combined_data,
            build_method,
            market_data=market_data,
        )

    @staticmethod
    def calibrate_single_component_from_mkt_data(
        model: Model, model_component: ModelComponent, solver_info: YieldCurveBuildMethodCommon
    ):

        calib_prod = model_component.calibration_product
        calib_funding = model_component.calibration_funding
        assert len(calib_prod) == len(calib_funding)
        for id, (prod, funding) in enumerate(zip(calib_prod, calib_funding)):
            fi_vp = FundingIndexParameter({"Funding Index": funding})
            vpc = ValuationParametersCollection([fi_vp])
            engine = ValuationEngineProductRegistry.new_valuation_engine(
                model, prod, vpc, ValuationRequest.PV_DETAILED
            )
            ### TODO: this is really bad, just to make things work'
            ###       ideally, we would like to get the initial guess when preparing the calibration instruments
            initial_guess = 0.0
            if isinstance(prod, ProductFxForward):
                fx_prod: ProductFxForward = prod
                initial_guess = fx_prod.strike

            YieldCurveCalibration.calibrate_state___var(
                engine, model_component.component_identifier, id, solver_info, initial_guess
            )

    @staticmethod
    def prepare_calibrate_instruments(
        value_date: Date, mkt_data_list: List[Data1D], build_method: BuildMethod, fpt: pd.DataFrame
    ):

        dt, dconv = "DATA TYPE", "DATA CONVENTION"
        ### create calibration instrument tuples, i.e., (time_to_anchored_time, product)
        valid_count = set()
        calib_instrument_quadruple = []
        for data in mkt_data_list:
            data_convention = data.data_convention
            v = fpt[
                (fpt[dt].str.upper() == data.data_type.upper())
                & (fpt[dconv].str.upper() == data.data_convention.name.upper())
            ]
            funding_identifier = (
                v["FUNDING IDENTIFIER"].values[0] if len(v) != 0 else build_method.target
            )

            for axis1, value in zip(data.axis1, data.values):
                
                
                prod: Product = ProductFactory.create_product_from_data_convention(
                    value_date, axis1, data_convention, value, quote_style='yield'
                )

                tmp_acc = accrued(value_date, prod.last_date)
                if tmp_acc == 0.0:
                    tmp_acc = 1e-4  # not nice solution, but just for the sake of it
                valid_count.add(tmp_acc)
                calib_instrument_quadruple.append(
                    (
                        tmp_acc,
                        prod,
                        funding_identifier,
                        [
                            data.data_type,
                            data.data_convention.conv_name,
                            axis1,
                            "",
                            value,
                            data.data_identifier.unit(),
                        ],
                    )
                )
        assert len(valid_count) == len(calib_instrument_quadruple)
        sorted_calib_instruments = sorted(
            calib_instrument_quadruple,
            key=lambda calib_instrument_tuples: calib_instrument_tuples[0],
        )

        ### initialize state data and calibration product
        calib_instruments = []
        funding_identifiers = []
        time_to_anchored_dates = []
        sorted_mkt_data = []
        for each in sorted_calib_instruments:
            time_to_anchored_dates.append(each[0])
            calib_instruments.append(each[1])
            funding_identifiers.append(each[2])
            sorted_mkt_data.append(each[3])
        state_data = np.asarray([time_to_anchored_dates, [0.0] * len(time_to_anchored_dates)])

        ### initialize model component
        return YieldCurveModelComponent(
            value_date,
            build_method.target_index,
            state_data,
            build_method,
            calib_instruments,
            funding_identifiers,
            sorted_mkt_data,
        )

    ### utils
    @staticmethod
    def _sort_out_build_method_and_funding(
        build_method_collection: BuildMethodCollection, data_collection: DataCollection
    ) -> Tuple[dict, List[BuildMethod]]:

        key_common = YieldCurveBuildMethodCommon._build_method_type
        key_others = YieldCurveIndexBuildMethod._build_method_type
        key_others_fi = YieldCurveFundingBuildMethod._build_method_type
        key_others_fx = YieldCurveFXBuildMethod._build_method_type
        key_funding = DataGeneric._data_shape
        ordered_bm_list = (
            {}
        )  # {CURRENCY : {'COMMON' : bm , 'YIELD CURVE' : [bms], 'DATAGENERIC' : df  }}

        ### identify common build methods
        other_build_methods = []
        for _, bm in build_method_collection.items:
            this_bm: BuildMethod = bm
            if Currency(this_bm.target).is_valid:
                funding_table: DataGeneric = data_collection.get_data_from_data_collection(
                    "DATA GENERIC", this_bm["FUNDING PARAMETERS"]
                )
                df_funding_table = funding_table.display()
                ordered_bm_list[this_bm.target] = {
                    key_common: this_bm,
                    key_funding: df_funding_table,
                }
            else:
                other_build_methods.append(this_bm)

        ### for given a currency, sort out relevant build methods
        residual_build_methods = []
        for bm in other_build_methods:

            ccy_str = bm.target_index.currency().code()
            if ccy_str in ordered_bm_list:
                if type(bm) == YieldCurveFundingBuildMethod:
                    if key_others_fi not in ordered_bm_list[ccy_str]:
                        ordered_bm_list[ccy_str][key_others_fi] = [bm]
                    else:
                        ordered_bm_list[ccy_str][key_others_fi].append(bm)
                elif type(bm) == YieldCurveFXBuildMethod:
                    if key_others_fx not in ordered_bm_list[ccy_str]:
                        ordered_bm_list[ccy_str][key_others_fx] = [bm]
                    else:
                        ordered_bm_list[ccy_str][key_others_fx].append(bm)
                else:
                    if key_others not in ordered_bm_list[ccy_str]:
                        ordered_bm_list[ccy_str][key_others] = [bm]
                    else:
                        ordered_bm_list[ccy_str][key_others].append(bm)
            else:
                residual_build_methods.append(bm)

        # sort our dependency
        for ccy_str, content in ordered_bm_list.items():
            # sort non-funding build method
            sorted_bms = YieldCurveBuilder._sort_out_bm_dependency(content[key_others])
            content[key_others] = sorted_bms
            # TODO: sort funding build method

        # ### no currency method ?
        # YieldCurveBuilder._sort_out_bm_dependency(residual_build_methods)

        return ordered_bm_list, residual_build_methods

    @staticmethod
    def _sort_out_bm_dependency(build_methods: List[BuildMethod]):
        bms_sorted = []
        not_stand_alone = []
        for bm in build_methods:
            if bm.reference_index is not None:
                not_stand_alone.append(bm)
            else:
                bms_sorted.append(bm)
        bms_sorted += not_stand_alone
        return bms_sorted


### registry
ModelBuilderRegistry().register(
    YieldCurve._model_type.to_string(), YieldCurveBuilder.create_model_yield_curve
)
