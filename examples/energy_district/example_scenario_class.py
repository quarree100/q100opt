from q100opt.setup_model import load_csv_data
from q100opt.setup_model import DistrictScenario
from q100opt import postprocessing as pp


table_collection = load_csv_data('data')

ds = DistrictScenario(
    name='my_scenario',
    table_collection=table_collection,
    number_of_time_steps=500,
    year=2018,
    emission_limit=50500,
)

ds.solve(solver='cbc')

# POSTPROCESSING #######################

results = ds.es.results['main']

# plots invests
pp.plot_invest_flows(results)
pp.plot_invest_storages(results)

# # plots time series
# pp.plot_storages_soc(results)
# pp.plot_buses(res=results, es=ds.es)

# # export table collection
# ds.tables_to_csv()
# ds.tables_to_excel()

print('dump district energy system')

import os

ds.es.dump(dpath=os.path.dirname(os.path.abspath(__file__)),
           filename='dump_district_es.dump')

import oemof.solph as solph

es_restore = solph.EnergySystem()

es_restore.restore(
    dpath=os.path.dirname(os.path.abspath(__file__)),
    filename='dump_district_es.dump'
)

tc = es_restore.results['Table collection']


# ds.dump()
# ds_restore = DistrictScenario()

# logging.info("Restore the energy system and the results.")
# energysystem = solph.EnergySystem()
# energysystem.restore(dpath=None, filename=None)

