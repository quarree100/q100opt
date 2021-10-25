import pandas as pd
from matplotlib import pyplot as plt
from oemof.tools import logger
import logging

import q100opt.plots as plots
from q100opt.buildings import BuildingInvestModel, SolarThermalCollector
from q100opt.scenario_tools import ParetoFront
from q100opt.setup_model import load_csv_data

logger.define_logging(screen_level=logging.INFO)

# read data
timeseries = pd.read_csv("data/test-building-timeseries.csv")

weather = pd.read_csv("data/weather.csv")

tech_data = pd.read_csv("data/techdata.csv", index_col=0, skiprows=1)

commodity_data = load_csv_data("data/commodities")

# define data, that could/should be in the Kataster
kataster = {
    'heat_load_space_heating': 10,         # heat load space heating [kW]
    'heat_load_dhw': 4,          # heat load hot water [kW]
    'heat_load_total': 12,            # total heat load [kW]
    'pv_1_max': 5,      # maximum kWp of PV area 1
    'pv_2_max': 3,      # maximum kWp of PV area 2
    'pv_3_max': 0,      # maximum kWp of PV area 2

    # roof areas
    # roof 1, e.g. west orientation
    'roof_1_azimuth': 90,          # Ausrichtung [°]
    'roof_1_pitch': 40,             # Dachneigung in [°]
    'roof_1_area_usable': 20,       # [m²]
    # roof 1, e.g. south orientation
    'roof_2_azimuth': 180,           # Ausrichtung [°]
    'roof_2_pitch': 40,             # Dachneigung in [°]
    'roof_2_area_usable': 20,       # [m²]
    # roof 1, e.g. east orientation
    'roof_3_azimuth': 270,          # Ausrichtung [°]
    'roof_3_pitch': 40,             # Dachneigung in [°]
    'roof_3_area_usable': 20,       # [m²]

    # solar thermal options
    # maximum share of roof area considered for solar thermal
    'st_1_max': 0.8,
    'st_2_max': 0.8,
    'st_3_max': 0.8,

    # maximum values of units (for investment model)
    "gas-boiler.maximum": 100,
    "pellet-boiler.maximum": 0,
    "wood-boiler.maximum": 0,
    "heatpump-geo.maximum": 10,
    "heatpump-air.maximum": 10,
    "thermal-storage.maximum": 100,
    "battery-storage.maximum": 100,
    "substation.maximum": 100,

    # installed capacities for operation model
    "gas-boiler.installed": 10,
    "pellet-boiler.installed": 0,
    "wood-boiler.installed": 0,
    "heatpump-geo.installed": 0,
    "heatpump-air.installed": 10,
    "thermal-storage.installed": 0,
    "battery-storage.installed": 0,
}

my_collector = SolarThermalCollector(
    eta_0=0.825,
    a_1=3.41,
    a_2=0.0161,
)

house = BuildingInvestModel(
    space_heating_demand=timeseries["E_th_RH"],
    electricity_demand=timeseries["E_el"],
    hot_water_demand=timeseries["E_th_TWE"],
    pv_1_profile=timeseries["E_el_PV_1"],
    pv_2_profile=timeseries["E_el_PV_2"],
    commodity_data=commodity_data,
    tech_data=tech_data,
    weather=weather,
    timesteps=8760,
    start_date="2015-01-01 01:00",
    location=(52.516254, 13.377535),
    solar_thermal_collector=my_collector,
    exclusive_roof_constraint=True,     # for each roof a constraint with limited area is created
    pv_system={"space_demand": 5},      # [m²/kWp],
    **kataster,
)

table_collection = house.create_table_collection()

# ab hier wäre es aufbauend auf den bestehenden Funktionen von q100opt

house.pareto_front = ParetoFront(
    table_collection=house.table_collection,
    number_of_points=5,
    number_of_time_steps=700,
)

house.pareto_front.calc_pareto_front(solver='gurobi', tee=True)

# some plots

house.pareto_front.results["pareto_front"].plot(
    x='emissions', y='costs', kind='scatter'
)
plt.xlabel('emissions')
plt.ylabel('costs')
plt.show()

for emission_limit, scenario in house.pareto_front.district_scenarios.items():
    plots.plot_investments(
        scenario.results['main'], title="Emissionscenario: " + emission_limit
    )
