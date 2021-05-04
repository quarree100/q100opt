# -*- coding: utf-8 -*-

"""Module for optimising the energy system of single buildings.

Please use this module with care. It is work in progress and not tested yet!

Contact: Johannes Röder <johannes.roeder@uni-bremen.de>

SPDX-License-Identifier: MIT

"""
import pandas as pd
from oemof.thermal.compression_heatpumps_and_chillers import calc_cops


class Building:
    """Building class"""

    def __init__(self,
                 electricity_demand=None,
                 space_heating_demand=None,
                 hot_warm_demand=None,
                 pv_profile_1=None,
                 pv_profile_2=None,
                 pv_profile_3=None,
                 commodity_data=None,
                 tech_data=None,
                 weather=None,
                 kataster_data=None
                 ):
        self.demand = {
            "electricity": electricity_demand,
            "heating": space_heating_demand,
            "hotwater": hot_warm_demand
        }
        self.pv_potential = {
            "pv_1": {"profile": pv_profile_1,
                     "maximum": kataster_data["PV_1_maximum"]},
            "pv_2": {"profile": pv_profile_2,
                     "maximum": kataster_data["PV_2_maximum"]},
            "pv_3": {"profile": pv_profile_3,
                     "maximum": kataster_data["PV_3_maximum"]},
        }
        self.kataster_data = kataster_data,
        self.weather = weather,
        self.commodities = commodity_data,
        self.techdata = tech_data,
        self.table_collection = None
        self.pareto_front = None
        self.results = dict()

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


class District:
    """District class with many buildings."""
    pass
