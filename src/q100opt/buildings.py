# -*- coding: utf-8 -*-

"""Module for optimising the energy system of single buildings.

Please use this module with care. It is work in progress and not tested yet!

Contact: Johannes Röder <johannes.roeder@uni-bremen.de>

SPDX-License-Identifier: MIT

"""
import math
import os

import numpy as np
import pandas as pd

try:
    from oemof.thermal.compression_heatpumps_and_chillers import calc_cops
    from oemof.thermal.stratified_thermal_storage import calculate_capacities
    from oemof.thermal.stratified_thermal_storage import calculate_losses

except ImportError:
    print("Need to install oemof.thermal to use the buildings module.")

from q100opt.setup_model import load_csv_data

dir_name = os.path.dirname(__file__)

# DEFAULT_WEATHER = pd.read_csv(
#     os.path.join(
#         os.path.dirname(os.path.abspath(__file__)),
#         "default_data/weather/TRY2015_523938130651_Jahr.csv"
#     ), skiprows=36, delimiter=r"\s+",
# )

DEFAULT_PV_SYSTEM = {
    "module": "Module A",
    "inverter": "Inverter C",
}

DEFAULT_TABLE_COLLECTION_1 = load_csv_data(
    os.path.join(dir_name, "default_data/building_one_temp_level")
)

KWARGS_GIS_ATTR = pd.read_csv(
    os.path.join(dir_name, "building_attributes.csv")
)


class Building:
    """Building base class.

    The building base class contains basic parameters
    of the buildings energy system that are relevant for doing investment
    optimisations and operation optimisation.

    Examples
    --------
    Basic usage examples of the Building with a random selection of
    attributes:
    >>> from q100opt.buildings import Building
    >>> my_building = Building(
    ...     id="My_House",
    ...     electricity_demand=pd.Series([2, 4, 5, 1, 3])
    ...     )
    """

    def __init__(self,
                 commodity_data=None,
                 tech_data=None,
                 weather=None,
                 timesteps=8760,
                 system_configuration="one_temp_level",
                 electricity_demand=None,
                 space_heating_demand=None,
                 hot_water_demand=None,
                 pv_1_profile=None,
                 pv_2_profile=None,
                 pv_3_profile=None,
                 **kwargs_gis
                 ):
        """
        Parameters
        ----------
        commodity_data : dict
            Dictionary with commodity data for the im- and export of
            energy carriers to the external energy system,
            e.g. electricity import and export, gas import, biomass import.
        tech_data : pandas.DataFrame
            Table with cost and efficiency parameters of energy converter and
            storage units.
        weather : pandas.DataFrame
            Table with weather data. The columns must be named as follows:
                - "temperature" : Outside temperature in [°C]
                - "ghi" : Global horizontal solar irradiation in [W/m²]
                - "dhi" : ...
                - "dni" : ...
        timesteps : int
            Number of timesteps of input data.
        system_configuration : str
            Setting of the buildings' heating system. This impacts the
            oemof.solph model configuration. Options:
                `one_temp_level`:
                    Space heating and domestic hot water are
                    assumed to have the same temperature level.
        electricity_demand : pandas.Series
            Sequence / Series with electricity demand values.
        space_heating_demand : pandas.Series
            Sequence / Series with electricity demand values.
        hot_water_demand : pandas.Series
            Sequence / Series with electricity demand values.
        pv_1_profile : pandas.Series
            Sequence with normed PV profile of roof 1.
        pv_2_profile : pandas.Series
            Sequence with normed PV profile of roof 2.
        pv_3_profile : pandas.Series
            Sequence with normed PV profile of roof 3.

        kwargs_gis :
            Additional scalar or str parameters settings. Please see
            `q100opt.buildings_attributes.csv` for further explanation.
        """

        for key in kwargs_gis.keys():
            if key not in list(KWARGS_GIS_ATTR['Attribute'].values):
                ae = "Attribute `{}` is not allowed!".format(key)
                raise AttributeError(ae)

        if commodity_data is not None:
            self.commodities = {k: v for k, v in commodity_data.items()}
        else:
            self.commodities = None

        self.techdata = tech_data,
        self.weather_data = weather,

        self.num_timesteps = timesteps

        self.id = kwargs_gis.get("id")

        # some general buildings attributes
        self.type = kwargs_gis.get("building_group")
        self.year = kwargs_gis.get("year")
        self.levels = kwargs_gis.get("levels")
        self.apartments = kwargs_gis.get("apartments")
        self.ground_area = kwargs_gis.get("ground_area")
        self.gross_floor_area = kwargs_gis.get("gross_floor_area")
        self.net_floor_area = kwargs_gis.get("net_floor_area")

        self.heat_load_space_heating = \
            kwargs_gis.get("heat_load_space_heating")
        self.heat_load_dhw = kwargs_gis.get("heat_load_dhw")
        self.heat_load_total = kwargs_gis.get("heat_load_total")

        self.heating_system = {
            "system": system_configuration,
            "temp_heating_limit": kwargs_gis.get("temp_heating_limit", 15),
            "temp_forward_limit": kwargs_gis.get("temp_forward_limit", 60),
            "temp_forward_winter": kwargs_gis.get("temp_forward_winter", 80),
            "temp_return": kwargs_gis.get("temp_return", 15),
        }

        if self.weather_data[0] is not None:
            self.heating_system.update(
                {"temp_forward": self.calc_temp_forward()}
            )
        else:
            self.heating_system.update({"temp_forward": None})

        self.demand = {
            "electricity": electricity_demand,
            "heating": space_heating_demand,
            "hotwater": hot_water_demand,
        }

        for k, v in self.demand.items():
            if v is None:
                self.demand[k] = pd.Series(np.zeros(8760))

        if self.heating_system["system"] == "one_temp_level":
            table_collection_template = DEFAULT_TABLE_COLLECTION_1
        else:
            raise ValueError(
                "There is no other table collection than"
                "`one_temp_level` template implemented yet."
            )

        energy_converter = {}
        for trafo in table_collection_template["Transformer"]["label"]:
            energy_converter[trafo] = {
                'maximum': kwargs_gis.get(trafo + ".maximum", 10),
                'installed': kwargs_gis.get(trafo + ".installed", 0),
            }

        self.energy_converter = pd.DataFrame(energy_converter).T

        energy_storages = {}
        for storage in table_collection_template["Storages"]["label"]:
            energy_storages[storage] = {
                'maximum': kwargs_gis.get(storage + ".maximum", 100),
                'installed': kwargs_gis.get(storage + ".installed", 0),
            }

        self.energy_storages = pd.DataFrame(energy_storages).T

        self.roof_data = None

        self.pv = dict()
        self.set_pv_attributes(
            pv_1_profile, pv_2_profile, pv_3_profile,
            **kwargs_gis
        )

        self.table_collection = table_collection_template
        self.pareto_front = None
        self.results = dict()

    def set_pv_attributes(self, pv_1_profile=None, pv_2_profile=None,
                          pv_3_profile=None, **kwargs):
        """Method for directly adding PV systems to the building.

        Parameters
        ----------
        pv_1_profile : pandas.Series
            Sequence with normed PV profile of roof 1.
        pv_2_profile : pandas.Series
            Sequence with normed PV profile of roof 2.
        pv_3_profile : pandas.Series
            Sequence with normed PV profile of roof 3.

        kwargs :
            pv_1_installed : float
                Installed capacity in [kW_peak] of roof 1.
            pv_2_installed : float
                Installed capacity in [kW_peak] of roof 2.
            pv_3_installed : float
                Installed capacity in [kW_peak] of roof 3.
            pv_1_max : float
                Maximum capacity in [kW_peak] of roof 1.
            pv_2_max : float
                Maximum capacity in [kW_peak] of roof 2.
            pv_3_max : float
                Maximum capacity in [kW_peak] of roof 3.

        Examples
        --------
        >>> from q100opt.buildings import Building
        >>> my_building = Building()
        >>> my_building.set_pv_attributes(
        ...     pv_1_max=15,
        ...     pv_1_profile=pd.Series([0, 6, 34, 5, 0, 1])
        ...     )
        >>> assert(my_building.pv["potentials"]["pv_1"]["maximum"] == 15)
        >>> assert(sum(my_building.pv["potentials"]["pv_1"]["profile"]) == 46)
        """
        if pv_1_profile is None:
            pv_1_profile = pd.Series(np.zeros(self.num_timesteps))
        if pv_2_profile is None:
            pv_2_profile = pd.Series(np.zeros(self.num_timesteps))
        if pv_3_profile is None:
            pv_3_profile = pd.Series(np.zeros(self.num_timesteps))

        self.pv.update({
            'potentials': {
                "pv_1": {"profile": pv_1_profile,
                         "installed": kwargs.get("pv_1_installed", 0),
                         "maximum": kwargs.get("pv_1_max", 0)
                         },
                "pv_2": {"profile": pv_2_profile,
                         "installed": kwargs.get("pv_2_installed", 0),
                         "maximum": kwargs.get("pv_2_max", 0)},
                "pv_3": {"profile": pv_3_profile,
                         "installed": kwargs.get("pv_3_installed", 0),
                         "maximum": kwargs.get("pv_3_max", 0)},
            }
        })

        pv_system = DEFAULT_PV_SYSTEM.copy()
        # pv_system.update(kwargs)

        self.pv.update({
            "pv_system": pv_system
        })

    def calc_pv_profiles(self):
        """Pre-calculation of roof specific pv profiles for each roof."""
        if self.roof_data is None:
            e1 = "Please provide roof data for a pre-calulation of the " \
                 "PV profiles."
            raise ValueError(e1)
        else:
            # TODO : Write function for PV Profile calculation with PVlib.
            pass

    def calc_temp_forward(self):
        """Calculates the forward temperature series via linear interpolation.

        For outside temperature values equal or lower -12°C, the
        design temperature "temp_heat_forward_winter" is assumed.

        For outside temperature values between the heating limit (default 15°C)
        and -12 °C the values are linear interpolated.
        """
        return np.interp(
            self.weather_data[0]["weather.temperature"],
            [-12, self.heating_system["temp_heating_limit"]],
            [self.heating_system["temp_forward_winter"],
             self.heating_system["temp_forward_limit"]]
        )

    def calc_heat_load_profile(self, method="VDI-xy"):
        """Generates/Calculates the space heating profile.

        Parameters
        ----------
        method : str
            Calculation method.
        buildings_type : str
            Type of buildng (EFH, MFH, GHD, ...)
        year : int
            Year of construction

        TODO : Complete method using external libraries/methods.
        """
        return pd.Series()

    def calc_heat_dhw_profile(self, method="VDI-xy"):
        """Generates/Calculates the domestic hot water heat profile.

        Parameters
        ----------
        method : str
            Calculation method.
        buildings_type : str
            Type of buildng (EFH, MFH, GHD, ...)
        year : int
            Year of construction
        aparments : int
            Number of household living in that buildng

        TODO : Complete method using external libraries/methods.
        """
        return pd.Series()

    def calc_electricity_profile(self, method="VDI-xy"):
        """Generates/Calculates the electricity profile of the building.

        Parameters
        ----------
        method : str
            Calculation method.

        TODO : Complete method using external libraries/methods.
        """
        return pd.Series()

    def create_table_collection(self):
        pass

    def prepare_table_collection(self):
        """This method is an adapter for the setting up the table collection.

        Here, all steps are performed that are equal for both, the investment
        optimisation and the operation optimisation.
        """
        def _add_commodity_tables():
            """
            The tables for the external commodity sources and sinks are added
            from the input data.
            """
            for k, v in self.commodities.items():
                if k not in tables.keys():
                    tables.update({k: v})
                else:
                    # this is the case for the `Timeseries`
                    tables[k] = pd.concat([tables[k], v], axis=1)

        def _add_pv_param():
            """
            Fills the table `Source_fix` with the PV parameters from
            the Kataster data, and the `tech data`.
            """
            PVs = ["pv_1", "pv_2", "pv_3"]
            for pv in PVs:
                index = \
                    tables['Source_fix'].loc[
                        tables['Source_fix']['label'] == pv].index[0]

                tables['Source_fix'].loc[index, 'invest.maximum'] = \
                    self.pv["potentials"][pv]['maximum']

                tables['Source_fix'].loc[index, 'flow.nominal_value'] = \
                    self.pv["potentials"][pv]['installed']

                tables['Source_fix'].loc[index, 'invest.minimum'] = \
                    self.techdata[0].loc["pv"]["minimum"]

                tables['Source_fix'].loc[index, 'invest.ep_costs'] = \
                    self.techdata[0].loc["pv"]["ep_costs"]

                tables['Source_fix'].loc[index, 'invest.offset'] = \
                    self.techdata[0].loc["pv"]["offset"]

                tables['Timeseries'][pv + '.fix'] = \
                    self.pv["potentials"][pv]['profile'].values

        def _add_demands():
            """Adds the demand timeseries to the table collection."""
            for r, c in tables["Sink_fix"].iterrows():
                tables['Timeseries'][c['label'] + '.fix'] = \
                    self.demand[c['label']].values

        def _add_transformer():
            """Creates the Transformer table."""

            trafos = tables["Transformer"]
            trafos.set_index("label", inplace=True)

            trafos["eff_out_1"] = trafos["eff_out_1"].astype(object)
            trafos["flow.max"] = trafos["flow.max"].astype(object)

            for r, c in trafos.iterrows():

                trafos.loc[r, "flow.nominal_value"] = \
                    self.energy_converter.at[r, "installed"]

                trafos.loc[r, "invest.ep_costs"] = \
                    self.techdata[0].loc[r]["ep_costs"]

                trafos.loc[r, "invest.offset"] = \
                    self.techdata[0].loc[r]["offset"]

                trafos.loc[r, "invest.minimum"] = \
                    self.techdata[0].loc[r]["minimum"]

                trafos.loc[r, "invest.maximum"] = \
                    self.energy_converter.at[r, "maximum"]

                if self.techdata[0].loc[r]["type"] == "boiler":

                    trafos.loc[r, "eff_out_1"] = \
                        self.techdata[0].loc[r]["efficiency"]

                elif self.techdata[0].loc[r]["type"] == "chp":

                    trafos.loc[r, "eff_out_1"] = \
                        self.techdata[0].loc[r]["efficiency"]

                    trafos.loc[r, "eff_out_2"] = \
                        self.techdata[0].loc[r]["efficiency_2"]

                elif r == "heatpump-air":

                    hp_data = self.techdata[0].loc[r]

                    cop_series = calc_cops(
                        mode='heat_pump',
                        temp_high=pd.Series(
                            self.heating_system['temp_forward']),
                        temp_low=self.weather_data[0]['weather.temperature'],
                        quality_grade=hp_data['carnot_quality'],
                        factor_icing=hp_data['factor_icing'],
                        temp_threshold_icing=hp_data['temp_icing'],
                    )

                    # nominal COP is needed for the calculation of the
                    # maximum heat output
                    cop_nom = calc_cops(
                        mode='heat_pump',
                        temp_high=[hp_data['temp_cop_nominal_sink']],
                        temp_low=[hp_data['temp_cop_nominal_source']],
                        quality_grade=hp_data['carnot_quality'],
                        factor_icing=hp_data['factor_icing'],
                        temp_threshold_icing=hp_data['temp_icing'],
                    )

                    max_series = pd.Series(calc_Q_max(
                        cop_series, cop_nom[0], maximum_one=False,
                        # correction_factor=1.2
                    ))

                    trafos.loc[r, "eff_out_1"] = "series"
                    tables['Timeseries'][r + '.eff_out_1'] = cop_series

                    trafos.loc[r, "flow.max"] = "series"
                    tables['Timeseries'][r + '.max'] = max_series

                elif r == "heatpump-geo":

                    hp_data = self.techdata[0].loc[r]

                    cop_nom = calc_cops(
                        mode='heat_pump',
                        temp_high=[hp_data['temp_cop_nominal_sink']],
                        temp_low=[hp_data['temp_cop_nominal_source']],
                        quality_grade=hp_data['carnot_quality'],
                    )

                    cop_series = calc_cops(
                        mode='heat_pump',
                        temp_high=pd.Series(
                            self.heating_system['temp_forward']),
                        temp_low=[hp_data['temp_source']] * 8760,
                        quality_grade=hp_data['carnot_quality']
                    )

                    max_series = pd.Series(calc_Q_max(
                        cop_series, cop_nom[0], maximum_one=False,
                        # correction_factor=1.2
                    ))

                    trafos.loc[r, "eff_out_1"] = "series"
                    tables['Timeseries'][r + '.eff_out_1'] = cop_series

                    trafos.loc[r, "flow.max"] = "series"
                    tables['Timeseries'][r + '.max'] = max_series

                    trafos.loc[r, "flow.summed_max"] = \
                        hp_data['max_full_load_hours']

                else:
                    raise ValueError(
                        "Transformer type {} is not know.".format(r)
                    )

            trafos.reset_index(inplace=True)

            tables.update({"Transformer": trafos})

        def _add_storages():
            storages = tables["Storages"]
            storages.set_index("label", inplace=True)

            def _add_battery_storage():
                """This method performs the pre-calculation for the battery."""
                lab = "battery_storage"
                if lab in storages.index:
                    tech_data = self.techdata[0].loc[lab]

                    # investment parameters

                    storages.loc[lab, "invest.maximum"] = \
                        self.energy_storages.at[lab, "maximum"]

                    storages.loc[lab, "storage.nominal_storage_capacity"] = \
                        self.energy_storages.at[lab, "installed"]

                    storages.loc[lab, "invest.minimum"] = tech_data["minimum"]

                    storages.loc[lab, "invest.ep_costs"] = \
                        tech_data["ep_costs"]

                    storages.loc[lab, "invest.offset"] = tech_data["offset"]

                    # technical parameters

                    storages.loc[
                        lab, "storage.invest_relation_output_capacity"] = \
                        tech_data["c-rate"]

                    storages.loc[
                        lab, "storage.invest_relation_input_capacity"] = \
                        tech_data["c-rate"]

                    storages.loc[lab, "storage.loss_rate"] = \
                        tech_data["loss-rate-1/h"]

                    in_out_flow_conversion_factor = math.sqrt(
                        tech_data["eta-storage"])

                    storages.loc[lab, "storage.inflow_conversion_factor"] = \
                        in_out_flow_conversion_factor

                    storages.loc[lab, "storage.outflow_conversion_factor"] = \
                        in_out_flow_conversion_factor

            def _add_thermal_storage():
                """
                This method performs the pre-calculation of the thermal
                storage.
                """
                lab = "thermal_storage"
                tech_data = self.techdata[0].loc[lab]

                # default_values ############
                # this values are the base for the calculation of the loss
                # factors, and the maximum storage capacity
                diameter_loss_basis = 1
                temp_delta_default = 25
                # ###########################

                u_value = \
                    tech_data['insulation-lambda-W/mK'] / \
                    (0.001 * tech_data['insulation-thickness-mm'])

                temp_h = self.heating_system["temp_forward"]
                temp_c = self.heating_system["temp_return"]
                temp_env = tech_data['temp_env']

                temp_delta = temp_h - temp_c

                max_storage_content = temp_delta / temp_delta_default

                losses = calculate_losses(
                    u_value, diameter=diameter_loss_basis, temp_h=temp_h,
                    temp_c=temp_c, temp_env=temp_env,
                )

                loss_rate = losses[0] * max_storage_content
                fix_relativ_losses = losses[1] * max_storage_content

                storages.loc[lab, "invest.maximum"] = \
                    self.energy_storages.at[lab, "maximum"]

                storages.loc[lab, "storage.nominal_storage_capacity"] = \
                    self.energy_storages.at[lab, "installed"]

                storages.loc[lab, "invest.minimum"] = \
                    tech_data["minimum"]

                storages.loc[lab, "invest.ep_costs"] = \
                    tech_data["ep_costs"]

                storages.loc[lab, "invest.offset"] = \
                    tech_data["offset"]

                # parameter as series
                storages["storage.max_storage_level"] = \
                    storages["storage.max_storage_level"].astype(object)
                storages["storage.loss_rate"] = \
                    storages["storage.loss_rate"].astype(object)
                storages["storage.fixed_losses_relative"] = \
                    storages["storage.fixed_losses_relative"].astype(object)

                storages.loc[lab, "storage.max_storage_level"] = "series"
                tables['Timeseries'][
                    lab + '.max_storage_level'] = max_storage_content

                storages.loc[lab, "storage.loss_rate"] = "series"
                tables['Timeseries'][lab + '.loss_rate'] = loss_rate

                storages.loc[lab, "storage.fixed_losses_relative"] = "series"
                tables['Timeseries'][lab + '.fixed_losses_relative'] = \
                    fix_relativ_losses

            _add_battery_storage()

            _add_thermal_storage()

            storages.reset_index(inplace=True)

            tables.update({"Storages": storages})

        tables = self.table_collection

        _add_commodity_tables()
        _add_pv_param()
        _add_demands()
        _add_transformer()
        _add_storages()

        self.table_collection = tables


class BuildingInvestModel(Building):
    """Investment optimisation model for the energy converters and storages.

    Parameters
    ----------
    See :class:`Building`
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_table_collection(self):
        """Creates the table collection for the energy system model."""
        self.prepare_table_collection()

        # for the investment model, all investment columns are set to 1
        for k, v in self.table_collection.items():
            if 'investment' in v.columns:
                v['investment'] = 1

        return self.table_collection


class BuildingOperationModel(Building):
    """Operation optimisation model for the energy converters and storages.

    Given the energy converter and storage units of a buildings,
    the demand (e.g. heat and electricity), this model
    optimises the energy supply of the building regarding costs and emissions.

    Parameters
    ----------
    See :class:`Building`

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_table_collection(self):
        """Creates the table collection for the energy system model."""
        self.prepare_table_collection()

        # for the operation model, all investment columns are set to 0
        for k, v in self.table_collection.items():
            if 'investment' in v.columns:
                v['investment'] = 0

        # TODO : For the battery: Add Storage in- and outlfow nominal
        #  value based on c-rate:
        #
        # storages.at[lab, "inflow.nominal_value"] = \
        #     tech_data["c-rate"] * self.energy_storages.at[
        #         lab, "installed"]
        #
        # storages.at[lab, "outflow.nominal_value"] = \
        #     tech_data["c-rate"] * self.energy_storages.at[
        #         lab, "installed"]


class District:
    """District class with many buildings."""
    pass


def calc_Q_max(cop_series, cop_nominal, maximum_one=False,
               correction_factor=1):
    """
    Calculates the maximal heating capacity (relative value) of a
    heat pump.

    In the end, the maximum heating power is proportional to:
     COP(T) / COP_nominal

    Parameters
    ----------
    cop_series : list
        Series with COPs of Heatpump.
    cop_nominal : scalar
        Nominal COP of heatpump.
    maximum_one : bool
        Limit the maximum heating power to the nominal power. So, the maximum
        relative heating power is 1.
    correction_factor : scalar
        Factor for correcting the maximum heating power:

            (cop_actual / cop_nominal) ** correction_factor

        This leads to further downscaling the maximum heating output at low
        temperatures.

    Returns
    -------
    list : Maximum heating power of heatpump as factor of nominal power:

        Q_heat_relative = cop_actual / cop_nominal

    # TODO : Suggest and implement in oemof.thermal
    """
    max_Q_hot = [(actual_cop / cop_nominal) ** correction_factor
                 for actual_cop in cop_series]

    if maximum_one:
        max_Q_hot = [1 if x > 1 else x for x in max_Q_hot]

    return max_Q_hot
