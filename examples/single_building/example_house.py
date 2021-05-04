import pandas as pd
from matplotlib import pyplot as plt

from q100opt.buildings import Building
from q100opt.scenario_tools import ParetoFront
import q100opt.plots as plots

# read data
timeseries = pd.read_csv("data/test-building-timeseries.csv")
weather = pd.read_csv("data/weather.csv")
tech_data = pd.read_csv("data/techdata.csv", index_col=0, skiprows=1)
commodity_data = {
    'commodities': pd.read_csv("data/commodity.csv"),
    'timeseries': pd.read_csv("data/commodity-timeseries.csv")
}

# define data, that could be in the Kataster
kataster = {
    'heat_load_sh': 10,         # heat load space heating [kW]
    'heat_load_hw': 4,          # heat load hot water [kW]
    'heat_load': 12,            # total heat load [kW]
    'temp_space_heating': 70,   # forward temperature space heating [°C]
    'variable_forward_temperature': [   # [temp_AT], [temp_forward]
        [-12, -2, 5, 12],
        [75, 70, 65, 55]
    ],
    'hot_water_generation': 'electric-boiler',   # type of hot water generation
    'PV_1_maximum': 5,      # maximum kWp of PV area 1
    'PV_2_maximum': 3,      # maximum kWp of PV area 2
    'PV_3_maximum': 0,      # maximum kWp of PV area 2
    'Battery_maximum': 20,  # maximum capacity of LiIon Battery
}

house = Building(
    space_heating_demand=timeseries["E_th_RH"],
    electricity_demand=timeseries["E_el"],
    hot_warm_demand=timeseries["E_th_TWE"],
    pv_profile_1=timeseries["E_el_PV_1"],
    pv_profile_2=timeseries["E_el_PV_2"],
    commodity_data=commodity_data,
    tech_data=tech_data,
    weather=weather,
    kataster_data=kataster,
)

house.create_table_collection()

# ab hier wäre es aufbauend auf den bestehenden Funktionen von q100opt

house.pareto_front = ParetoFront(
    table_collection=house.table_collection,
    number_of_points=5,
    number_of_time_steps=8760,
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
