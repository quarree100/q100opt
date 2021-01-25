from q100opt.setup_model import load_csv_data
from q100opt.setup_model import DistrictScenario
from q100opt import postprocessing as pp


table_collection = load_csv_data('data')

ds = DistrictScenario(
    name='my_scenario',
    table_collection=table_collection,
    number_of_time_steps=8760,
    year=2018,
    emission_limit=505000,
)

ds.solve(solver='gurobi')

# POSTPROCESSING #######################

results = ds.es.results['main']

pp.plot_invest_flows(results)
pp.plot_invest_storages(results)
