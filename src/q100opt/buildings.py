# -*- coding: utf-8 -*-

"""Module for optimising the energy system of single buildings.

Please use this module with care. It is work in progress and not tested yet!

Contact: Johannes Röder <johannes.roeder@uni-bremen.de>

SPDX-License-Identifier: MIT

"""
import pandas as pd
import q100opt
from oemof.thermal.compression_heatpumps_and_chillers import calc_cops

PV_SYSTEM = {
    "module": "Module A",
    "inverter": "Inverter C",
}


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
    >>> from q100opt import buildings
    >>> my_building = q100opt.Building(
    ...     name="My_House"
    ...     electricity_demand=pd.Series([2, 4, 5, 1, 3])
    ...     )
    """

    def __init__(self, commodity_data, tech_data, weather=None, name=None,
                 **kwargs):
        self.commodities = {k: v for k, v in commodity_data.items()},
        self.techdata = tech_data,
        self.weather = weather,
        self.id = name

        # some general buildings attributes
        self.type = kwargs.get("building_group")
        self.year = kwargs.get("year")
        self.levels = kwargs.get("levels")
        self.apartments = kwargs.get("apartments")
        self.ground_area = kwargs.get("ground_area")
        self.gross_floor_area = kwargs.get("gross_floor_area")
        self.net_floor_area = kwargs.get("net_floor_area")

        self.demand = {
            "electricity": kwargs.get("electricity_demand"),
            "heating": kwargs.get("space_heating_demand"),
            "hotwater": kwargs.get("hot_water_demand"),
        }

        self.pv = None
        self.set_pv_attributes(**kwargs)

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
        """Set ups the PV attributes of the buildings."""
        self.pv.update({
            'potentials': {
                "pv_1": {"profile": kwargs.get("pv_1_profile"),
                         "maximum": kwargs.get("pv_1_max")},
                "pv_2": {"profile": kwargs.get("pv_2_profile"),
                         "maximum": kwargs.get("pv_2_max")},
                "pv_3": {"profile": kwargs.get("pv_3_profile"),
                         "maximum": kwargs.get("pv_3_max")},
            }
        })

        pv_system = PV_SYSTEM.copy()
        pv_system.update(kwargs)

        self.pv.update({
            "pv_system": pv_system
        })


class BuildingInvestModel(Building):
    """Investment optimisation model for the energy converters and storages.

    Parameters
    ----------



    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_table_collection(self):
        """..."""
        tables = {}

        # TODO : create functions for each table

        # 1) Define bus table TODO : e.g. dependent on demands
        buses = pd.DataFrame(
            ["b_gas", "b_elec", "b_heat"],
            columns=['label']
        )
        buses['excess'] = 0
        buses['shortage'] = 0

        tables.update({"Bus": buses})

        # 2) commodity sources and sinks
        com = self.commodities[0]['commodities']
        tables.update({
            "Source": com.loc[com["type"] == "Source"],
            "Sink": com.loc[com["type"] == "Sink"],
            "Timeseries": self.commodities[0]['timeseries'],
        })

        # 3) Investment Sources (PV)
        PVs = []
        for k, v in self.pv_potential.items():
            if v['maximum'] > 0:
                PVs.append(k)
        source_fix_table = pd.DataFrame(PVs, columns=["label"])
        source_fix_table["investment"] = 1
        source_fix_table["to"] = "b_elec"
        source_fix_table["invest.ep_costs"] = \
            self.techdata[0].loc["pv"]["ep_costs"]
        source_fix_table["invest.offset"] = \
            self.techdata[0].loc["pv"]["offset"]

        source_fix_table = source_fix_table.set_index("label")
        for pv in PVs:
            source_fix_table.at[pv, 'invest.maximum'] = \
                self.pv_potential[pv]['maximum']
            tables['Timeseries'][pv + '.fix'] = \
                self.pv_potential[pv]['profile'].values

        source_fix_table.reset_index(inplace=True)

        tables.update({"Source_fix": source_fix_table})

        # 4) Sink_fix (demand) tables (and update of timeseries)
        demand_table = pd.DataFrame(
            columns=['label', 'from', 'nominal_value']
        )

        # TODO : not static, but dependent on given demands
        demand_table.loc[0] = ["heating", "b_heat", 1]
        demand_table.loc[1] = ["hotwater", "b_heat", 1]
        demand_table.loc[2] = ["electricity", "b_elec", 1]

        tables.update({"Sink_fix": demand_table})

        for r, c in demand_table.iterrows():
            tables['Timeseries'][c['label'] + '.fix'] = \
                self.demand[c['label']].values

        # 5) Transformer table
        # TODO: this labels must be given somehow
        trafo_labels = ["gas_boiler", "heatpump_air"]

        trafos = pd.DataFrame(index=trafo_labels)
        trafos.index.name = 'label'
        trafos["investment"] = 1

        for r, c in trafos.iterrows():
            trafos.at[r, "invest.ep_costs"] = \
                self.techdata[0].loc[r]["ep_costs"]
            trafos.at[r, "invest.offset"] = \
                self.techdata[0].loc[r]["offset"]

        # TODO : das müsste auch noch anders gemacht werden:
        trafos.at["gas_boiler", "in_1"] = "b_gas"
        trafos.at["gas_boiler", "in_2"] = 0
        trafos.at["gas_boiler", "out_1"] = "b_heat"
        trafos.at["gas_boiler", "out_2"] = 0
        trafos.at["gas_boiler", "eff_out_1"] = \
            self.techdata[0].loc["gas_boiler"]["efficiency"]

        trafos.at["heatpump_air", "in_1"] = "b_elec"
        trafos.at["heatpump_air", "in_2"] = 0
        trafos.at["heatpump_air", "out_1"] = "b_heat"
        trafos.at["heatpump_air", "out_2"] = 0
        trafos.at["heatpump_air", "eff_out_1"] = 2.9

        trafos["eff_in_1"] = 1
        trafos["invest.maximum"] = 100

        # TODO : Add precalculation with oemof.thermal
        # cop_series = calc_cops(
        #     mode='heat_pump',
        #     temp_high=self.kataster_data[0]['temp_space_heating'],
        #     temp_low=self.weather[0]['weather.temperature'],
        #     quality_grade=self.techdata[0].loc["heatpump_air"]['carnot_quality'],
        #     factor_icing=self.techdata[0].loc["heatpump_air"]['factor_icing'],
        #     temp_threshold_icing=self.techdata[0].loc["heatpump_air"][
        #         'temp_icing'],
        # )

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
            - "heat-pump-air"
            - "heat-pump-geothermal-probe"
            - "pellets"
            - "wood-chips"
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
