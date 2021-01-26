from matplotlib import pyplot as plt
import oemof.solph as solph
import networkx as nx
import logging

try:
    import pygraphviz
except ImportError:
    logging.info('Module pygraphviz not found: Graph was not plotted.')


def plot_invest_flows(results):
    """Plots all investment flows in a bar plot."""

    flows_invest = [x for x in results.keys() if x[1] is not None
                    if hasattr(results[x]['scalars'], 'invest')
                    if isinstance(x[0], solph.Transformer)]

    invest_val = [results[x]['scalars']['invest'] for x in flows_invest]
    invest_lab = [x[0].label for x in flows_invest]

    plt.bar(invest_lab, invest_val)
    plt.ylabel('Installed Flow Capacity [kW]')
    plt.xlabel('"From node" label (x[0].label of Flow)')
    plt.show()


def plot_invest_storages(results):
    """Plots investment storages as bar plot.s"""

    store_invest = [x for x in results.keys() if x[1] is None
                    if hasattr(results[x]['scalars'], 'invest')
                    if isinstance(x[0], solph.GenericStorage)]

    invest_val_s = [results[x]['scalars']['invest'] for x in store_invest]
    invest_lab_s = [x[0].label for x in store_invest]

    plt.bar(invest_lab_s, invest_val_s)
    plt.ylabel('Installed Storage Capacity [kWh]')
    plt.xlabel('Label of storage')
    plt.show()


def plot_buses(res=None, es=None):

    l_buses = []

    for n in es.nodes:
        type_name =\
            str(type(n)).replace("<class 'oemof.solph.", "").replace("'>", "")
        if type_name == "network.Bus":
            l_buses.append(n.label)

    for n in l_buses:
        bus_sequences = solph.views.node(res, n)["sequences"]
        bus_sequences.plot(kind='line', drawstyle="steps-mid", subplots=False,
                           sharey=True)
        plt.show()


def plot_storages_soc(res=None):

    nodes = [x for x in res.keys() if x[1] is None]
    node_storage_invest_label = [x[0].label for x in nodes if isinstance(
        x[0], solph.components.GenericStorage)
                           if hasattr(res[x]['scalars'], 'invest')]

    for n in node_storage_invest_label:
        soc_sequences = solph.views.node(res, n)["sequences"]
        soc_sequences = soc_sequences.drop(soc_sequences.columns[[0, 2]], 1)
        soc_sequences.plot(kind='line', drawstyle="steps-mid", subplots=False,
                           sharey=True)
        plt.show()


def draw_graph(
    grph,
    edge_labels=True,
    node_color="#AFAFAF",
    edge_color="#CFCFCF",
    plot=True,
    node_size=2000,
    with_labels=True,
    arrows=True,
    layout="neato",
):
    """
    Source: https://github.com/oemof/oemof-examples/blob/master/oemof_examples/
    oemof.solph/v0.4.x/excel_reader/dispatch.py

    Parameters
    ----------
    grph : networkxGraph
        A graph to draw.
    edge_labels : boolean
        Use nominal values of flow as edge label
    node_color : dict or string
        Hex color code oder matplotlib color for each node. If string, all
        colors are the same.
    edge_color : string
        Hex color code oder matplotlib color for edge color.
    plot : boolean
        Show matplotlib plot.
    node_size : integer
        Size of nodes.
    with_labels : boolean
        Draw node labels.
    arrows : boolean
        Draw arrows on directed edges. Works only if an optimization_model has
        been passed.
    layout : string
        networkx graph layout, one of: neato, dot, twopi, circo, fdp, sfdp.
    """
    if type(node_color) is dict:
        node_color = [node_color.get(g, "#AFAFAF") for g in grph.nodes()]

    # set drawing options
    options = {
        "with_labels": with_labels,
        "node_color": node_color,
        "edge_color": edge_color,
        "node_size": node_size,
        "arrows": arrows,
    }

    pos = nx.drawing.nx_agraph.graphviz_layout(grph, prog=layout)

    # draw graph
    nx.draw(grph, pos=pos, **options)

    # add edge labels for all edges
    if edge_labels is True and plt:
        labels = nx.get_edge_attributes(grph, "weight")
        nx.draw_networkx_edge_labels(grph, pos=pos, edge_labels=labels)

    # show output
    if plot is True:
        plt.show()


def plot_graph(pos, grph, plot=True):
    """Plots an EnergySystem graph."""

    pos_keys = list()

    for i in pos.keys():
        pos_keys.append(i)

    bus_gas_keys = list()
    bus_el_keys = list()
    bus_heat_keys = list()
    trans_keys = list()
    nets_keys = list()
    store_keys = list()
    others_keys = list()

    for i in pos_keys:
        x = i[0:4]
        y = i[0:2]
        if x == 'b_ga':
            bus_gas_keys.append(i)
        elif x == 'b_el':
            bus_el_keys.append(i)
        elif x == 'b_he':
            bus_heat_keys.append(i)
        elif y == 'st':
            store_keys.append(i)
        elif y == 't_':
            trans_keys.append(i)
        elif y == 'n_':
            nets_keys.append(i)
        else:
            others_keys.append(i)

    bus_gas_nodes = bus_gas_keys
    bus_el_nodes = bus_el_keys
    bus_heat_nodes = bus_heat_keys
    trans_nodes = trans_keys
    nets_nodes = nets_keys
    store_nodes = store_keys
    others_nodes = others_keys

    buses_el = grph.subgraph(bus_el_nodes)
    pos_buses_el = {x: pos[x] for x in bus_el_keys}

    buses_gas = grph.subgraph(bus_gas_nodes)
    pos_buses_gas = {x: pos[x] for x in bus_gas_keys}

    buses_heat = grph.subgraph(bus_heat_nodes)
    pos_buses_heat = {x: pos[x] for x in bus_heat_keys}

    trans = grph.subgraph(trans_nodes)
    pos_trans = {x: pos[x] for x in trans_keys}

    sources = grph.subgraph(nets_nodes)
    pos_sources = {x: pos[x] for x in nets_keys}

    store = grph.subgraph(store_nodes)
    pos_store = {x: pos[x] for x in store_keys}

    others = grph.subgraph(others_nodes)
    pos_others = {x: pos[x] for x in others_keys}

    sizenodes = 800

    nx.draw(grph, pos=pos, node_shape='1', with_labels=True,
            node_color='#ffffff', edge_color='#CFCFCF', node_size=sizenodes,
            arrows=True)
    nx.draw(buses_el, pos=pos_buses_el, node_shape='p', node_color='#0049db',
            node_size=sizenodes)
    nx.draw(buses_gas, pos=pos_buses_gas, node_shape='p', node_color='#f2e60e',
            node_size=sizenodes)
    nx.draw(buses_heat, pos=pos_buses_heat, node_shape='p',
            node_color='#f95c8b', node_size=sizenodes)
    nx.draw(trans, pos=pos_trans, node_shape='s', node_color='#85a8c2',
            node_size=sizenodes)
    nx.draw(sources, pos=pos_sources, node_shape='P', node_color='#7FFFD4',
            node_size=sizenodes)
    nx.draw(store, pos=pos_store, node_shape='o', node_color='#ac88ff',
            node_size=sizenodes)
    nx.draw(others, pos=pos_others, node_shape='v', node_color='#71f442',
            node_size=sizenodes)

    # show output
    if plot is True:
        plt.show()

    return
