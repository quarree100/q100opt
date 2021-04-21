from q100opt.setup_model import load_csv_data
from q100opt.scenario_tools import DistrictScenario
from q100opt import plots as plots
from oemof.network.graph import create_nx_graph
import logging

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

results = ds.results['main']

# plots invests
plots.plot_invest_flows(results)
plots.plot_invest_storages(results)

# # plots time series
plots.plot_storages_soc(results)
plots.plot_buses(res=results, es=ds.es)

# # export table collection
# ds.tables_to_csv()
# ds.tables_to_excel()

# print('dump district energy system')
# ds.dump()

# plot esys graph I (Luis)
try:
    import networkx as nx

    grph = create_nx_graph(ds.es)
    pos = nx.drawing.nx_agraph.graphviz_layout(grph, prog='neato')
    plots.plot_graph(pos, grph)
    logging.info('Energy system Graph OK.')

    # plot esys graph II (oemof examples)
    graph = create_nx_graph(ds.es)
    plots.draw_graph(
        grph=graph,
        plot=True,
        layout="neato",
        node_size=1000,
        node_color={"b_heat_gen": "#cd3333", "b_el_ez": "#cd3333"},
    )

except ImportError:
    print("Need to install networkx to create energy system graph.")
