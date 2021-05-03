# -*- coding: utf-8 -*-

"""Module for optimising the energy system of single buildings.

Please use this module with care. It is work in progress and not tested yet!

Contact: Johannes Röder <johannes.roeder@uni-bremen.de>

SPDX-License-Identifier: MIT

"""


class Building:
    """Building class"""
    def __init__(self,
                 electricity_demand=None,
                 space_heating_demand=None,
                 hot_warm_demand=None,
                 pv_profile_1=None,
                 pv_profile_2=None,
                 pv_profile_3=None,
                 buildings_type=None, **kwargs):
        self.demand = {
            "electricity": electricity_demand,
            "heating": space_heating_demand,
            "hotwater": hot_warm_demand
        }
        self.pv_potential = {
            "pv_1": pv_profile_1,
            "pv_2": pv_profile_2,
            "pv_3": pv_profile_3,
        }
        self.building_type = buildings_type
        self.table_collection = None
        self.pareto_front = None
        self.results = dict()
