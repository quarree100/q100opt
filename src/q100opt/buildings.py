# -*- coding: utf-8 -*-

"""Module for optimising the energy system of single buildings.

Please use this module with care. It is work in progress and not tested yet!

Contact: Johannes Röder <johannes.roeder@uni-bremen.de>

SPDX-License-Identifier: MIT

"""
import copy
import logging
import math
import os
import pickle

import numpy as np
import pandas as pd

try:
    from oemof.thermal.compression_heatpumps_and_chillers import calc_cops
    from oemof.thermal.solar_thermal_collector import flat_plate_precalc
    from oemof.thermal.stratified_thermal_storage import calculate_losses

except ImportError:
    raise ImportError(
        "Need to install oemof.thermal: \n"
        "`pip install https://github.com/user/repository/archive/branch.zip`"
    )

from q100opt.setup_model import load_csv_data

dir_name = os.path.dirname(__file__)

DEFAULT_WEATHER = pd.read_csv(
    os.path.join(dir_name, "default_data/weather/weather.csv")
)

DEFAULT_PV_SYSTEM = {
    "space_demand": 5,  # [m²/kWp]
    "module": "Module A",
    "inverter": "Inverter C",
}

DEFAULT_TABLE_COLLECTION_1 = load_csv_data(
    os.path.join(dir_name, "default_data/building_one_temp_level")
)

DEFAULT_TECH_DATA = pd.read_csv(
    os.path.join(dir_name, "default_data/techdata/techdata.csv"),
    index_col=0, skiprows=1,
)

DEFAULT_COMMODITY_DATA = load_csv_data(
    os.path.join(dir_name, "default_data/commodities")
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
                 name=None,
                 commodity_data=None,
                 tech_data=None,
                 weather=None,
                 timesteps=8760,
                 start_date="2018-01-01 00:00",
                 freq="H",
                 timezone='CET',
                 location=(52.516254, 13.377535),
                 system_configuration="one_temp_level",
                 electricity_demand=None,
                 space_heating_demand=None,
                 hot_water_demand=None,
                 pv_1_profile=None,
                 pv_2_profile=None,
                 pv_3_profile=None,
                 solar_thermal_collector=None,
                 pv_system=DEFAULT_PV_SYSTEM,
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
        solar_thermal_collector : q100opt.buildings.SolarThermalCollector
            Specific solar thermal collector type.

        kwargs_gis :
            Additional scalar or str parameters settings. Please see
            `q100opt.buildings_attributes.csv` for further explanation.
        """

        # check for allowed attributes
        for key in kwargs_gis.keys():
            if key not in list(KWARGS_GIS_ATTR['Attribute'].values):
                ae = "Attribute `{}` is not allowed!".format(key)
                raise AttributeError(ae)

        if commodity_data is not None:
            self.commodities = {k: v for k, v in commodity_data.items()}
        else:
            self.commodities = DEFAULT_COMMODITY_DATA

        if tech_data is not None:
            self.techdata = tech_data
        else:
            self.techdata = DEFAULT_TECH_DATA

        if weather is not None:
            self.weather_data = weather
        else:
            self.weather_data = DEFAULT_WEATHER

        self.num_timesteps = timesteps

        if start_date is not None:
            self.datetimeindex = pd.date_range(
                start=start_date, periods=timesteps, freq=freq, tz=timezone
            )
            self.weather_data.index = self.datetimeindex

        self.location = location

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
            "temp_return": kwargs_gis.get("temp_return", 40),
        }

        if self.weather_data is not None:
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
            table_collection_template = \
                copy.deepcopy(DEFAULT_TABLE_COLLECTION_1)
        else:
            raise ValueError(
                "There is no other table collection than"
                "`one_temp_level` template implemented yet."
            )

        energy_converter = {}
        for trafo in table_collection_template[
                "Transformer"]["label"].copy().tolist():
            energy_converter[trafo] = {
                'maximum': kwargs_gis.get(trafo + ".maximum", 10),
                'installed': kwargs_gis.get(trafo + ".installed", 0),
            }

        self.energy_converter = pd.DataFrame(energy_converter).T

        energy_storages = {}
        for storage in table_collection_template[
                "Storages"]["label"].copy().tolist():
            energy_storages[storage] = {
                'maximum': kwargs_gis.get(storage + ".maximum", 100),
                'installed': kwargs_gis.get(storage + ".installed", 0),
            }

        self.energy_storages = pd.DataFrame(energy_storages).T

        self.roof_data = [
            RoofArea(
                area=kwargs_gis.get("roof_1_area_usable", 0),
                pitch=kwargs_gis.get("roof_1_pitch", 0),
                azimuth=kwargs_gis.get("roof_1_azimuth", 0),
            ),
            RoofArea(
                area=kwargs_gis.get("roof_2_area_usable", 0),
                pitch=kwargs_gis.get("roof_2_pitch", 0),
                azimuth=kwargs_gis.get("roof_2_azimuth", 0),
            ),
            RoofArea(
                area=kwargs_gis.get("roof_3_area_usable", 0),
                pitch=kwargs_gis.get("roof_3_pitch", 0),
                azimuth=kwargs_gis.get("roof_3_azimuth", 0),
            ),
        ]

        self.pv = dict()
        self.set_pv_attributes(
            pv_1_profile, pv_2_profile, pv_3_profile,
            **kwargs_gis
        )
        self.pv_system = pv_system

        self.solar_thermal_collector = solar_thermal_collector

        self.solarthermal = []

        if self.solar_thermal_collector is not None:
            self.set_st_attributes(
                maximum_values=[
                    kwargs_gis.get("st_1_max", 0),
                    kwargs_gis.get("st_2_max", 0),
                    kwargs_gis.get("st_3_max", 0),
                ],
                installed_values=[
                    kwargs_gis.get("st_1_installed", 0),
                    kwargs_gis.get("st_2_installed", 0),
                    kwargs_gis.get("st_3_installed", 0),
                ]
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

    def set_st_attributes(self, maximum_values, installed_values):
        """
        Calculates the normed collector heat [W/m²] for each of the roofs
        and appends the results as pandas.DataFrame to the list (attribute
        `solarthermal`).
        """
        for i in range(len(self.roof_data)):

            roof = self.roof_data[i]

            delta_temp_mean = 0.5 * (self.heating_system["temp_forward"] - \
                              self.heating_system["temp_return"])

            precalc_data = flat_plate_precalc(
                lat=self.location[0],
                long=self.location[1],
                collector_tilt=roof.pitch,
                collector_azimuth=roof.azimuth,
                eta_0=self.solar_thermal_collector.eta_0,
                a_1=self.solar_thermal_collector.a_1,
                a_2=self.solar_thermal_collector.a_2,
                temp_collector_inlet=self.heating_system["temp_return"],
                delta_temp_n=delta_temp_mean,
                irradiance_global=self.weather_data["global_horizontal_W_m2"],
                irradiance_diffuse=self.weather_data["diffuse_horizontal_W_m2"],
                temp_amb=self.weather_data["weather.temperature"],
            )

            st_sysytem = {
                "precalc_data": precalc_data,
                "profile": precalc_data["collectors_heat"].values,
                "maximum": maximum_values[i] * roof.area,
                "installed": installed_values[i],
            }

            logging.info("Solarthermal profile %i of %i calculated.",
                         i+1, len(self.roof_data))

            self.solarthermal.append(st_sysytem)

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
            self.weather_data["weather.temperature"],
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
                if k != "Timeseries":
                    tables.update({k: v})
                else:
                    # this is the case for the `Timeseries` if given
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
                    self.techdata.loc["pv"]["minimum"]

                tables['Source_fix'].loc[index, 'invest.ep_costs'] = \
                    self.techdata.loc["pv"]["ep_costs"]

                tables['Source_fix'].loc[index, 'invest.offset'] = \
                    self.techdata.loc["pv"]["offset"]

                pv_num = pv.split('_')[1]
                tables['Source_fix'].loc[
                    index, 'invest.roof_area_' + pv_num
                ] = \
                    self.pv_system["space_demand"]

                tables['Timeseries'][pv + '.fix'] = \
                    self.pv["potentials"][pv]['profile'].values

        def _add_st_param():
            """
            Fills the table `Source_fix` with the Solarthermal parameters.
            """

            prefix = "solarthermal_"
            bus_label = "b_heat_st"

            if len(self.solarthermal) > 0:

                # Add extra Bus for solar thermal heat with excess

                st_bus = pd.Series(
                    index=tables["Bus"].columns
                )

                st_bus["label"] = "b_heat_st"
                st_bus["active"] = 1
                st_bus["excess"] = 1
                st_bus["shortage"] = 0
                st_bus["excess_costs"] = 0

                tables["Bus"] = \
                    tables["Bus"].append(st_bus, ignore_index=True)

                # Add Transformer with peripherial losses and extra electricity
                peri_losses = self.solar_thermal_collector.peripherial_loss
                elec_help = self.solar_thermal_collector.auxilary_electricity

                st_trafo = pd.Series(index=tables["Transformer"].columns)

                st_trafo["label"] = "solar-thermal-peri"
                st_trafo["active"] = 1
                st_trafo["investment"] = 0
                st_trafo["in_1"] = "b_heat_st"
                st_trafo["in_2"] = "b_elec"
                st_trafo["out_1"] = "b_heat"
                st_trafo["out_2"] = 0

                st_trafo["eff_in_1"] = 1
                st_trafo["eff_in_2"] = elec_help * (1 - peri_losses)
                st_trafo["eff_out_1"] = 1 - peri_losses

                tables["Transformer"] = \
                    tables["Transformer"].append(st_trafo, ignore_index=True)

            for i in range(len(self.solarthermal)):

                st_row = pd.Series(
                    index=tables["Source_fix"].columns
                )

                st_label = prefix + str(i+1)

                st_row["label"] = st_label

                st_row["to"] = bus_label

                st_row["invest.maximum"] = self.solarthermal[i]["maximum"]

                st_row["flow.nominal_value"] = \
                    self.solarthermal[i]["installed"]

                st_row["invest.minimum"] = \
                    self.techdata.loc["solarthermal"]["minimum"]

                st_row["invest.ep_costs"] = \
                    self.techdata.loc["solarthermal"]["ep_costs"]

                st_row["invest.offset"] = \
                    self.techdata.loc["solarthermal"]["offset"]

                st_row["invest.roof_area_" + str(i+1)] = 1

                # convert W/m² >> kW/m² !!
                tables['Timeseries'][st_label + '.fix'] = \
                    self.solarthermal[i]['profile'] * 0.001

                tables["Source_fix"] = \
                    tables["Source_fix"].append(st_row, ignore_index=True)

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

                trafos.loc[r, "invest.ep_costs"] = \
                    self.techdata.loc[r]["ep_costs"]

                trafos.loc[r, "invest.offset"] = \
                    self.techdata.loc[r]["offset"]

                trafos.loc[r, "invest.minimum"] = \
                    self.techdata.loc[r]["minimum"]

                if r == "solar-thermal-peri":
                    continue

                trafos.loc[r, "flow.nominal_value"] = \
                    self.energy_converter.at[r, "installed"]

                trafos.loc[r, "invest.maximum"] = \
                    self.energy_converter.at[r, "maximum"]

                if (self.techdata.loc[r]["type"] == "boiler") or (
                        self.techdata.loc[r]["type"] == "substation"):

                    trafos.loc[r, "eff_out_1"] = \
                        self.techdata.loc[r]["efficiency"]

                elif self.techdata.loc[r]["type"] == "chp":

                    trafos.loc[r, "eff_out_1"] = \
                        self.techdata.loc[r]["efficiency"]

                    trafos.loc[r, "eff_out_2"] = \
                        self.techdata.loc[r]["efficiency_2"]

                elif r == "heatpump-air":

                    hp_data = self.techdata.loc[r]

                    cop_series = calc_cops(
                        mode='heat_pump',
                        temp_high=pd.Series(
                            self.heating_system['temp_forward']),
                        temp_low=self.weather_data['weather.temperature'],
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

                    hp_data = self.techdata.loc[r]

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

                elif r == "substation":

                    trafos.loc[r, "eff_out_1"] = 1

                else:
                    raise ValueError(
                        "Transformer type {} is not know.".format(r)
                    )

            trafos.reset_index(inplace=True)

            tables.update({"Transformer": trafos})

        def _add_storages():
            """Pre-calculates and adds the parameter of the storages."""
            storages = tables["Storages"]
            storages.set_index("label", inplace=True)

            # add battery storage
            tech_data = self.techdata.loc["battery-storage"]
            storages = _add_battery_storage(
                storages, tech_data,
                self.energy_storages.at["battery-storage", "maximum"],
                self.energy_storages.at["battery-storage", "installed"],
            )

            # add thermal storage
            lab = "thermal-storage"
            tech_data_ts = self.techdata.loc[lab]
            storages, timeseries = _add_thermal_storage(
                storages, tech_data_ts,
                maximum=self.energy_storages.at[lab, "maximum"],
                installed=self.energy_storages.at[lab, "installed"],
                timeseries=tables['Timeseries'],
                temp_forward=self.heating_system["temp_forward"],
                temp_return=self.heating_system["temp_return"],
                lab=lab
            )

            tables['Timeseries'] = timeseries

            storages.reset_index(inplace=True)

            tables.update({"Storages": storages})

        tables = self.table_collection

        _add_commodity_tables()
        _add_pv_param()
        _add_st_param()
        _add_demands()
        _add_transformer()
        _add_storages()

        self.table_collection = tables

    def dump(self, path=None, filename=None):
        """Dumps the Buidlings Instance.

        The oemof.solph.EnergySystems and oemof.solph.Models of the
        q100opt.DistrictScenarios are removed before dumping, only the results
        are dumped.
        """
        # delete all oemof.solph.EnergySystems and oemof.solph.Models
        for _, v in self.pareto_front.__dict__.items():
            if hasattr(v, 'es') or hasattr(v, 'model'):
                setattr(v, 'es', None)
                setattr(v, 'model', None)

        for _, des in self.pareto_front.district_scenarios.items():
            setattr(des, 'es', None)
            setattr(des, 'model', None)

        pickle.dump(
            self.__dict__, open(os.path.join(path, filename), "wb")
        )

        logging.info(
            "Building dumped to {} as {}".format(path, filename)
        )


class BuildingInvestModel(Building):
    """Investment optimisation model for the energy converters and storages.

    Parameters
    ----------
    See :class:`Building`
    """
    def __init__(self, exclusive_roof_constraint=True, **kwargs):
        super().__init__(**kwargs)

        self.exclusive_roof_constraint = exclusive_roof_constraint

    def create_table_collection(self):
        """Creates the table collection for the energy system model."""
        self.prepare_table_collection()

        # for the investment model, all investment columns are set to 1
        for k, v in self.table_collection.items():
            if 'investment' in v.columns:
                v['investment'] = 1

        for x in ['Storages', 'Transformer']:
            for r, c in self.table_collection[x].iterrows():
                if c['invest.maximum'] == 0:
                    self.table_collection[x].loc[r, 'active'] = 0

        if self.exclusive_roof_constraint:
            self.add_constraint_table()

        return self.table_collection

    def add_constraint_table(self):
        """
        Creates the `Additional_constraints` table of the table collection.
        """
        add_con_table = pd.DataFrame(
            columns=["type", "keyword", "limit"],
            index=range(len(self.roof_data))
        )

        add_con_table["type"] = "additional_resource"
        add_con_table["keyword"] = [
            "roof_area_" + str(x+1) for x in range(len(self.roof_data))
        ]
        add_con_table["limit"] = [x.area for x in self.roof_data]

        self.table_collection["Additional_constraints"] = add_con_table


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


class SolarThermalCollector:
    """
    The SolarThermalCollector class represents a specific type of a
    Solarthermal Collector and contains the parameters given by the
    manufacturer.

    All parameter needed for calculating of the thermal efficiency dependent on
    the operation and are not part of this class.

    Examples
    --------
    Basic usage examples of the SolarThermalCollector:
    >>> from q100opt.buildings import SolarThermalCollector
    >>> my_collector = SolarThermalCollector(
    ...     eta_0=0.825,
    ...     a_1=3.41,
    ...     a_2=0.002,
    ...     )
    """
    def __init__(self,
                 eta_0=0.825,
                 a_1=3.41,
                 a_2=0.0161,
                 peripherial_loss=0.03,
                 auxilary_electricity=0.01,
                 ):
        """
        Default values of collector databaesd on model "SOL 27 premium W"
        of STIEBEL ELTRON, see:

        https://www.stiebel-eltron.de/content/dam/ste/de/de/products/downloads
        /Planungsunterlagen/Planungshandbuch/Planungshandbuch_EE_Solar.pdf

        Parameters
        ----------
        eta_0 : float
            Optical efficiency of the collector
        a_1 : float
            Thermal loss parameter 1
        a_2 : float
            Thermal loss parameter 2
        peripherial_loss : float
            Thermal loss of periperhie as share of collector heat generation
            (Value between 0 and 1)
        auxilary_electricity : float
            Electricity consumption of solar thermal system relative to the
            net thermal output of the system.
            (Value between 0 and 1)
        """
        # Collector data
        self.eta_0 = eta_0
        self.a_1 = a_1
        self.a_2 = a_2

        # System characteristics
        self.peripherial_loss = peripherial_loss
        self.auxilary_electricity = auxilary_electricity


class RoofArea:
    """
    The RoofArea class represents a specific roof area for the installation
    of PV or solar thermal.

    A RoofArea is characterized by an usable area, a slope angle, and the
    orientation (azimuth).

    Examples
    --------
    Basic usage examples of the RoofArea:
    >>> from q100opt.buildings import RoofArea
    >>> my_roof_1 = RoofArea(
    ...     area=20,
    ...     pitch=30,
    ...     azimuth=0,
    ...     )
    """
    def __init__(self,
                 area=None,
                 pitch=None,
                 azimuth=None,
                 ):
        """

        Parameters
        ----------
        area : float
            Usable area for solar applicataion in [m²]
        pitch : float
            Pitch angle of the area [°]
        azimuth : float
            Orientation of area analog to PVlib [°]
        """
        self.area = area
        self.pitch = pitch
        self.azimuth = azimuth


def _add_battery_storage(storages, tech_data, maximum, installed,
                         lab="battery-storage"):
    """
    This function forms the pre-calculation for the battery
    and adds the attributes of the battery storage
    to the `Storage` sheet of the table collection.

    Notes
    -----
    - The charging and de-charging power are assumed to equal, and given
      by the c-rate.
    - The storage efficiency is split to a storage inflow efficiency and
      a storage outflow efficiency. Both, are: sqrt(storage efficiency).
    """
    # investment and capacity parameters
    storages.loc[lab, "invest.maximum"] = maximum

    storages.loc[lab, "storage.nominal_storage_capacity"] = installed

    storages.loc[lab, "invest.minimum"] = tech_data["minimum"]

    storages.loc[lab, "invest.ep_costs"] = tech_data["ep_costs"]

    storages.loc[lab, "invest.offset"] = tech_data["offset"]

    # technical parameters

    storages.loc[
        lab, "storage.invest_relation_output_capacity"] = tech_data["c-rate"]

    storages.loc[
        lab, "storage.invest_relation_input_capacity"] = tech_data["c-rate"]

    storages.loc[lab, "storage.loss_rate"] = tech_data["loss-rate-1/h"]

    in_out_flow_conversion_factor = math.sqrt(tech_data["eta-storage"])

    storages.loc[lab, "storage.inflow_conversion_factor"] = \
        in_out_flow_conversion_factor

    storages.loc[lab, "storage.outflow_conversion_factor"] = \
        in_out_flow_conversion_factor

    return storages


def _add_thermal_storage(storages, tech_data, maximum, installed, timeseries,
                         temp_forward, temp_return, lab="thermal-storage"):
    """
    This function pre-calculates the parameter of the thermal storages
    and adds it to the sheet "Storages" of the table collection.

    Parameters
    ----------
    storages : pandas.DataFrame
        Sheet `Storages` of the table collection.
    tech_data : pandas.Series
        Economical and technical parameters of the thermal storage.
    maximum : float
        Maximum thermal capacity of the storage in [kWh].
    installed : float
        Installed thermal capacity of the storage in [kWh].
    timeseries : pandas.DataFrame
        `Timeseries` sheet of the table-collection. Time-dependent parameters
        will be added there.
    temp_forward : float or array
        Forward temperature of the heating system.
    temp_return : float
        Return temperature of the heating system.
    lab : str
        Label of the storage.

    Returns
    -------
    storages : pandas.DataFrame
        Updated `Storages` sheet of the table collection
    timeseries : pandas.DataFrame
        Updated `Timeseries` sheet of the table collection
    """
    # default_values ############
    # this values are the base for the calculation of the loss
    # factors, and the maximum storage capacity
    diameter_loss_basis = tech_data['diameter-m']
    temp_delta_default = tech_data['delta_T_default-K']
    # ###########################

    u_value = \
        tech_data['insulation-lambda-W/mK'] / \
        (0.001 * tech_data['insulation-thickness-mm'])

    temp_h = temp_forward
    temp_c = temp_return
    temp_env = tech_data['temp_env']

    temp_delta = temp_h - temp_c

    # the resulting storage capacity is the temperature delta at
    # each timestep (of the heating system), divided by the
    # default temperature delta, the costs are related to:
    max_storage_content = temp_delta / temp_delta_default

    # the loss factors are calculated via oemof.thermal
    # here, for a cylindrical storage, a diameter must be given.
    # then, the loss factors linearly depend on the height of the
    # storage.
    losses = calculate_losses(
        u_value, diameter=diameter_loss_basis, temp_h=temp_h,
        temp_c=temp_c, temp_env=temp_env,
    )

    # since the delta T of the storage changes in each timestep,
    # the relative loss factor also changes over time.
    # (if a constant delta T is assumed, the loss factor would be
    # constant, independent of the storage size (die Höhe des
    # Speichers kürz sich raus - siehe oemof.thermal Doku)
    loss_rate = losses[0] * max_storage_content

    # on the other side, the fixed part of the losses (caused by
    # temperature difference of the de-charged storage (return
    # temperature) and the surrounding)), becomes a constant factor
    # again after the multiplication with the maximum storage
    # content.
    fix_relativ_losses = losses[1] * max_storage_content

    storages.loc[lab, "invest.maximum"] = maximum

    storages.loc[lab, "storage.nominal_storage_capacity"] = installed

    storages.loc[lab, "invest.minimum"] = tech_data["minimum"]

    storages.loc[lab, "invest.ep_costs"] = tech_data["ep_costs"]

    storages.loc[lab, "invest.offset"] = tech_data["offset"]

    # parameter as series # TODO : really needed?
    storages["storage.max_storage_level"] = \
        storages["storage.max_storage_level"].astype(object)
    storages["storage.loss_rate"] = \
        storages["storage.loss_rate"].astype(object)
    storages["storage.fixed_losses_relative"] = \
        storages["storage.fixed_losses_relative"].astype(object)

    storages.loc[lab, "storage.max_storage_level"] = "series"
    timeseries[
        lab + '.max_storage_level'] = max_storage_content

    storages.loc[lab, "storage.loss_rate"] = "series"
    timeseries[lab + '.loss_rate'] = loss_rate

    storages.loc[lab, "storage.fixed_losses_relative"] = "series"
    timeseries[lab + '.fixed_losses_relative'] = \
        fix_relativ_losses

    return storages, timeseries


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


def restore_invest_building(path, filename):
    """Restores a building invest instance from dump."""
    building = BuildingInvestModel()

    building.__dict__ = \
        pickle.load(open(os.path.join(path, filename), "rb"))

    return building
