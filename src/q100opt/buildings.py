# -*- coding: utf-8 -*-

"""Module for optimising the energy system of single buildings.

Please use this module with care. It is work in progress and not tested yet!

Contact: Johannes Röder <johannes.roeder@uni-bremen.de>

SPDX-License-Identifier: MIT

"""
import os
import pandas as pd
import numpy as np

from q100opt.setup_model import  load_csv_data
from oemof.thermal.compression_heatpumps_and_chillers import calc_cops

PV_SYSTEM = {
    "module": "Module A",
    "inverter": "Inverter C",
}

DEFAULT_TABLE_COLLECTION_1 = load_csv_data(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "default_data/building_one_temp_level"
    )
)


class Building:
    """Building base class.

    Long description: The building base class contains basic parameters
    of the buildings energy system that are relevant for doing investment
    optimisations and operation optimisation.

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
    name : str
        ID or name of building.
    building_group : str
        Building group: "EFH" (single-family house), "MFH" (multi-family house)
        or "NWG" (not residential building)
    year : int
        Year of construction, e.g. 1977.
    levels: float
        Number of floor levels.
    apartments : int
        Number of apartments.
    ground_area : float
        Ground area (German: Grundfläche des Gebäudes (GF)) in m².
    gross_floor_area : float
        Total floor area (German: Brutto Grundfläche (BGF)) in m².
    net_floor_area: float
        Net floor area (German: NGF) in m².

    electricity_demand : pandas.Series
        Sequence / Series with electricity demand values.
    space_heating_demand : pandas.Series
        Sequence / Series with electricity demand values.
    hot_water_demand : pandas.Series
        Sequence / Series with electricity demand values.


    Examples
    --------
    Basic usage examples of the Building with a random selection of
    attributes:
    >>> from q100opt.buildings import Building
    >>> my_building = Building(
    ...     name="My_House",
    ...     electricity_demand=pd.Series([2, 4, 5, 1, 3])
    ...     )
    """

    def __init__(self, commodity_data=None, tech_data=None,
                 weather=None, name=None, timesteps=8760,
                 # heating system
                 system_configuration="one_temp_level",
                 temp_heating_limit=15,
                 temp_heat_forward_limit=65,
                 temp_heat_forward_winter=75,
                 temp_heat_return=50,
                 **kwargs):
        if commodity_data is not None:
            self.commodities = {k: v for k, v in commodity_data.items()}
        else:
            self.commodities = None
        self.techdata = tech_data,
        self.weather_data = weather,
        self.id = name

        # optimisation settings
        self.num_timesteps = timesteps

        # some general buildings attributes
        self.type = kwargs.get("building_group")
        self.year = kwargs.get("year")
        self.levels = kwargs.get("levels")
        self.apartments = kwargs.get("apartments")
        self.ground_area = kwargs.get("ground_area")
        self.gross_floor_area = kwargs.get("gross_floor_area")
        self.net_floor_area = kwargs.get("net_floor_area")

        self.heat_load_space_heating = kwargs.get("heat_load_space_heating")
        self.heat_load_dhw = kwargs.get("heat_load_dhw")
        self.heat_load_total = kwargs.get("heat_load_total")

        self.heating_system = {
            "system": system_configuration,
            "temp_heating_limit": temp_heating_limit,
            "temp_heat_forward_limit": temp_heat_forward_limit,
            "temp_heat_forward_winter": temp_heat_forward_winter,
            "temp_heat_return": temp_heat_return,
        }

        self.heating_system.update(
            {"temp_forward": self.calc_temp_forward()}
        )

        self.demand = {
            "electricity": kwargs.get("electricity_demand",
                                      pd.Series(np.zeros(8760))),
            "heating": kwargs.get("space_heating_demand",
                                  pd.Series(np.zeros(8760))),
            "hotwater": kwargs.get("hot_water_demand",
                                   pd.Series(np.zeros(8760))),
        }

        energy_converter = {}
        for trafo in DEFAULT_TABLE_COLLECTION_1["Transformer"]["label"]:
            energy_converter[trafo] = {
                'maximum': kwargs.get(trafo + ".maximum", 0),
                'installed': kwargs.get(trafo + ".installed", 0),
            }

        self.energy_converter = pd.DataFrame(energy_converter).T

        energy_storages = {}
        for storage in DEFAULT_TABLE_COLLECTION_1["Storages"]["label"]:
            energy_converter[storage] = {
                'maximum': kwargs.get(storage + ".maximum", 0),
                'installed': kwargs.get(storage + ".installed", 0),
            }

        self.energy_storages = pd.DataFrame(energy_storages).T

        self.roof_data = None

        self.pv = dict()
        self.set_pv_attributes(**kwargs)

        if self.heating_system["system"] == "one_temp_level":
            self.table_collection = DEFAULT_TABLE_COLLECTION_1
        else:
            self.table_collection = {
                "Bus": pd.DataFrame(columns=["label", "excess", "shortage"]),
                "Source": pd.DataFrame(),
                "Sink": pd.DataFrame(),
                "Timeseries": pd.DataFrame(),
                "Source_fix": pd.DataFrame(),
                "Sink_fix": pd.DataFrame(),
                "Transformer": pd.DataFrame(),
                "Storages": pd.DataFrame(),
            }

        self.pareto_front = None
        self.results = dict()

    def set_pv_attributes(self, **kwargs):
        """Set up the PV attributes of the building.

        Examples
        --------
        Manually set PV System:
        >>> from q100opt.buildings import Building
        >>> my_building = Building()
        >>> my_building.set_pv_attributes(
        ...     pv_1_max=15,
        ...     pv_1_profile=[0, 6, 34, 5, 0, 1]
        ...     )
        >>> assert(my_building.pv["potentials"]["pv_1"]["maximum"] == 15)
        """
        self.pv.update({
            'potentials': {
                "pv_1": {"profile": kwargs.get(
                    "pv_1_profile",
                    pd.Series(np.zeros(8760))),
                    "installed": kwargs.get("pv_1_installed", 0),
                    "maximum": kwargs.get("pv_1_max", 0)},
                "pv_2": {"profile": kwargs.get(
                    "pv_2_profile", pd.Series(np.zeros(8760))),
                    "installed": kwargs.get("pv_2_installed", 0),
                    "maximum": kwargs.get("pv_2_max", 0)},
                "pv_3": {"profile": kwargs.get(
                    "pv_3_profile", pd.Series(np.zeros(8760))),
                    "installed": kwargs.get("pv_3_installed", 0),
                    "maximum": kwargs.get("pv_3_max", 0)},
            }
        })

        pv_system = PV_SYSTEM.copy()
        # pv_system.update(kwargs)

        self.pv.update({
            "pv_system": pv_system
        })

    def precalc_pv_profiles(self):
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
            [self.heating_system["temp_heat_forward_winter"],
             self.heating_system["temp_heat_forward_limit"]]
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


class BuildingInvestModel(Building):
    """Investment optimisation model for the energy converters and storages.

    Parameters
    ----------



    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_table_collection(self):
        """..."""
        tables = self.table_collection

        # 2) commodity sources and sinks
        for k, v in self.commodities.items():
            if k not in tables.keys():
                tables.update({k: v})
            else:
                tables[k] = pd.concat([tables[k], v], axis=1)

        # 3) Investment Sources (PV)
        PVs = ["pv_1", "pv_2", "pv_3"]
        for pv in PVs:
            index = \
                tables['Source_fix'].loc[
                    tables['Source_fix']['label'] == pv].index[0]

            tables['Source_fix'].at[index, 'invest.maximum'] = \
                self.pv["potentials"][pv]['maximum']

            tables['Source_fix'].at[index, 'invest.minimum'] = \
                self.techdata[0].loc["pv"]["minimum"]

            tables['Source_fix'].at[index, 'invest.ep_costs'] = \
                self.techdata[0].loc["pv"]["ep_costs"]

            tables['Source_fix'].at[index, 'invest.offset'] = \
                self.techdata[0].loc["pv"]["offset"]

            tables['Timeseries'][pv + '.fix'] = \
                self.pv["potentials"][pv]['profile'].values

        # 4) Sink_fix (demand) tables (and update of timeseries)
        for r, c in tables["Sink_fix"].iterrows():
            tables['Timeseries'][c['label'] + '.fix'] = \
                self.demand[c['label']].values

        # 5) Transformer table
        trafos = tables["Transformer"]
        trafos.set_index("label", inplace=True)
        trafos["investment"] = 1

        trafos["eff_out_1"] = trafos["eff_out_1"].astype(object)

        for r, c in trafos.iterrows():

            trafos.at[r, "invest.ep_costs"] = \
                self.techdata[0].loc[r]["ep_costs"]

            trafos.at[r, "invest.offset"] = \
                self.techdata[0].loc[r]["offset"]

            trafos.at[r, "invest.minimum"] = \
                self.techdata[0].loc[r]["minimum"]

            trafos.at[r, "invest.maximum"] = \
                self.energy_converter.at[r, "maximum"]

            if self.techdata[0].loc[r]["type"] == "boiler":

                trafos.at[r, "eff_out_1"] = \
                    self.techdata[0].loc[r]["efficiency"]

            elif self.techdata[0].loc[r]["type"] == "chp":

                trafos.at[r, "eff_out_1"] = \
                    self.techdata[0].loc[r]["efficiency"]

                trafos.at[r, "eff_out_2"] = \
                    self.techdata[0].loc[r]["efficiency_2"]

            elif r == "heatpump-air":

                hp_data = self.techdata[0].loc[r]

                cop_nom = calc_cops(
                    mode='heat_pump',
                    temp_high=[hp_data['temp_cop_nominal_sink']],
                    temp_low=[hp_data['temp_cop_nominal_source']],
                    quality_grade=hp_data['carnot_quality'],
                    factor_icing=hp_data['factor_icing'],
                    temp_threshold_icing=hp_data['temp_icing'],
                )

                cop_series = calc_cops(
                    mode='heat_pump',
                    temp_high=pd.Series(self.heating_system['temp_forward']),
                    temp_low=self.weather_data[0]['weather.temperature'],
                    quality_grade=hp_data['carnot_quality'],
                    factor_icing=hp_data['factor_icing'],
                    temp_threshold_icing=hp_data['temp_icing'],
                )

                max_series = pd.Series(calc_Q_max(
                    cop_series, cop_nom[0], maximum_one=False,
                    # correction_factor=1.2
                ))

                trafos.at[r, "eff_out_1"] = "series"

                tables['Timeseries'][r + '.eff_out_1'] = cop_series

                tables['Timeseries'][r + '.max'] = max_series

            elif r == "heatpump-geothermal-probe":

                hp_data = self.techdata[0].loc[r]

                cop_nom = calc_cops(
                    mode='heat_pump',
                    temp_high=[hp_data['temp_cop_nominal_sink']],
                    temp_low=[hp_data['temp_cop_nominal_source']],
                    quality_grade=hp_data['carnot_quality'],
                )

                cop_series = calc_cops(
                    mode='heat_pump',
                    temp_high=pd.Series(self.heating_system['temp_forward']),
                    temp_low=[hp_data['temp_source']] * 8760,
                    quality_grade=hp_data['carnot_quality']
                )

                max_series = pd.Series(calc_Q_max(
                    cop_series, cop_nom[0], maximum_one=False,
                    # correction_factor=1.2
                ))

                trafos.at[r, "eff_out_1"] = "series"

                trafos.at[r, "flow.summed_max"] = \
                    hp_data['max_full_load_hours']

                tables['Timeseries'][r + '.eff_out_1'] = cop_series

                tables['Timeseries'][r + '.max'] = max_series

            else:
                raise ValueError(
                    "Transformer type {} is not know.".format(r)
                )

        trafos.reset_index(inplace=True)

        tables.update({"Transformer": trafos})

        # 6) Add storage options
        # TODO
        tables.update({"Storages": pd.DataFrame(columns=['label'])})

        self.table_collection = tables

        return tables


class BuildingOperationModel(Building):
    """Operation optimisation model for the energy converters and storages.

    Given the energy converter and storage units of a buildings,
    the demand (e.g. heat and electricity), this model
    optimises the energy supply of the building regarding costs and emissions.

    Parameters
    ----------
    heat_supply : str
        Heat supply options are:
            - "gas-boiler"
            - "heatpump-air"
            - "heatpump-geothermal-probe"
            - "pellet-boiler"
            - "wood-chips-boiler"
    pv_1 : float
        Installed PV capacity in [kW_peak] on roof 1.
    pv_2 : float
        Installed PV capacity in [kW_peak] on roof 2.
    pv_3 : float
        Installed PV capacity in [kW_peak] on roof 3.
    solarthermal : float
        Number of m² installed solar thermal capacity.
    solarthermal_type : str
        Technology of solarthermal. Options are:
            - "flat-collector"
            - "vaccum-tube-collector"
    solarthermal_roof: int
        Roof, on that solarthermal collectors are installed: 1, 2 or 3
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class District:
    """District class with many buildings."""
    pass


def calc_Q_max(cop_series, cop_nominal, maximum_one=False,
               correction_factor=1):
    """
    Calculates the maximal heating capacity (relative value) of a
    heat pump.

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
