import oemof.solph as solph
import pandas as pd
from matplotlib import pyplot as plt

from q100opt.setup_model import add_buses
from q100opt.setup_model import add_sinks
from q100opt.setup_model import add_sinks_fix
from q100opt.setup_model import add_sources
from q100opt.setup_model import add_sources_fix
from q100opt.setup_model import add_storages
from q100opt.setup_model import add_transformer
from q100opt.setup_model import check_active
from q100opt.setup_model import load_csv_data

# load data
nd = load_csv_data('data')

# check active
nd = check_active(nd)

# create nodes
nodes, busd = add_buses(nd['Bus'])
sources = add_sources(nd['Source'], busd, nd['Timeseries'])
sources_fix = add_sources_fix(nd['Source_fix'], busd, nd['Timeseries'])
sinks = add_sinks(nd['Sink'], busd, nd['Timeseries'])
sinks_fix = add_sinks_fix(nd['Sink_fix'], busd, nd['Timeseries'])
storages = add_storages(nd['Storages'], busd)
transformer = add_transformer(nd['Transformer'], busd, nd['Timeseries'])

nodes = nodes + sources + sources_fix + sinks_fix + sinks + storages + \
        transformer

# initialise oemof-solph EnergySystem
date_time_index = pd.date_range('1/1/2018', periods=8760, freq='H')
es = solph.EnergySystem(timeindex=date_time_index)
es.add(*nodes)

# initialise the operational model
om = solph.Model(es)

# Global CONSTRAINTS: emission limit
solph.constraints.generic_integral_limit(
    om, keyword='emission_factor', limit=505000)

om.solve(solver='gurobi', solve_kwargs={'tee': True})

# POSTPROCESSING #######################

es.results['main'] = solph.processing.results(om)
results = es.results['main']

# plot investment results transformer
flows_invest = [x for x in results.keys() if x[1] is not None
                if hasattr(results[x]['scalars'], 'invest')
                if isinstance(x[0], solph.Transformer)]

invest_val = [results[x]['scalars']['invest'] for x in flows_invest]
invest_lab = [x[0].label for x in flows_invest]

plt.bar(invest_lab, invest_val)
plt.ylabel('Installed Transformer Capacity [kW]')
plt.show()

# plot investment results storages
store_invest = [x for x in results.keys() if x[1] is None
                if hasattr(results[x]['scalars'], 'invest')
                if isinstance(x[0], solph.GenericStorage)]

invest_val_s = [results[x]['scalars']['invest'] for x in store_invest]
invest_lab_s = [x[0].label for x in store_invest]

plt.bar(invest_lab_s, invest_val_s)
plt.ylabel('Installed Storage Capacity [kWh]')
plt.show()
