import pandas as pd
import pytest

from q100opt.buildings import DEFAULT_TABLE_COLLECTION_1
from q100opt.buildings import Building
from q100opt.buildings import _add_battery_storage


def test_building_init():
    Building()


def test_no_roof_data_error():
    house = Building()
    with pytest.raises(ValueError,
                       match="Please provide roof data for a pre-calulation"):
        house.calc_pv_profiles()


def test_add_battery():
    storages = DEFAULT_TABLE_COLLECTION_1['Storages']
    storages.set_index("label", inplace=True)
    tech_data = pd.Series(
        [1, 3, 4, 0.5, 0.81, 0.0001],
        index=[
            "ep_costs", "offset", "minimum",
            "c-rate", "eta-storage", "loss-rate-1/h"
        ],
    )
    storages = _add_battery_storage(storages, tech_data, 10, 10)
    bat_sto = storages.loc["battery-storage"].dropna()
    assert bat_sto.equals(pd.Series(
        [1, 0, "b_elec", 10, 0.0001, 0.9, 0.9, 0.5, 0.5, 1, 10, 4, 3],
        index=["active", "investment", "bus",
               "storage.nominal_storage_capacity", "storage.loss_rate",
               "storage.inflow_conversion_factor",
               "storage.outflow_conversion_factor",
               "storage.invest_relation_input_capacity",
               "storage.invest_relation_output_capacity", "invest.ep_costs",
               "invest.maximum", "invest.minimum", "invest.offset"]
    ))
