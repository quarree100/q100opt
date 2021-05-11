# -*- coding: utf-8 -*-

"""

This module holds a diverse mixture of plotting functions.

Work in progress!

Johannes Röder <johannes.roeder@uni-bremen.de>

SPDX-License-Identifier: MIT

"""
import os

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from oemof import solph as solph

from .postprocessing import get_invest_converter
from .postprocessing import get_invest_storages


def plot_invest_flows(results):
    """Plots all investment flows in a bar plot."""
    flows_invest = get_invest_converter(results)

    invest_val = [results[x]['scalars']['invest'] for x in flows_invest]
    invest_lab = [x[0].label for x in flows_invest]

    plt.bar(invest_lab, invest_val)
    plt.ylabel('Installed Flow Capacity [kW]')
    plt.xlabel('"From node" label (x[0].label of Flow)')
    plt.show()


def plot_invest_storages(results):
    """Plots investment storages as bar plot.s"""
    store_invest = get_invest_storages(results)

    invest_val_s = [results[x]['scalars']['invest'] for x in store_invest]
    invest_lab_s = [x[0].label for x in store_invest]

    plt.bar(invest_lab_s, invest_val_s)
    plt.ylabel('Installed Storage Capacity [kWh]')
    plt.xlabel('Label of storage')
    plt.show()


def plot_investments(results, show=True, title=None):
    """..."""
    store_invest = get_invest_storages(results)

    invest_val_s = [results[x]['scalars']['invest'] for x in store_invest]
    invest_lab_s = [x[0].label for x in store_invest]

    flows_invest = get_invest_converter(results)

    invest_val = [results[x]['scalars']['invest'] for x in flows_invest]
    invest_lab = [x[0].label for x in flows_invest]

    # create the figure
    fig, ax = plt.subplots(1, 2, figsize=[6.4 * 1.5, 4.8])

    ax[0].bar(x=invest_lab, height=invest_val)
    ax[0].set_ylabel('Installed capacity [kW]')
    ax[0].set_xlabel('Energy converter units')

    ax[1].bar(x=invest_lab_s, height=invest_val_s)
    ax[1].set_ylabel('Installed capacity [kWh]')
    ax[1].set_xlabel('Energy storages')

    fig.autofmt_xdate(rotation=45)

    fig.suptitle(title)

    fig.tight_layout()

    if show:
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


def draw_graph(grph, edge_labels=True, node_color="#AFAFAF",
               edge_color="#CFCFCF", plot=True, node_size=2000,
               with_labels=True, arrows=True, layout="neato"):
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
    bus_H2_keys = list()
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
        elif x == 'b_H2':
            bus_H2_keys.append(i)
        elif y == 's_':
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
    bus_H2_nodes = bus_H2_keys
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

    buses_H2 = grph.subgraph(bus_H2_nodes)
    pos_buses_H2 = {x: pos[x] for x in bus_H2_keys}

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
    nx.draw(buses_H2, pos=pos_buses_H2, node_shape='p',
            node_color='#fe6ace', node_size=sizenodes)
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


def plot_bus_stack(results, label_bus='b_heat', label_demand='t_pump_heatgrid',
                   interval=None, title=None):
    """Performs a stacked plot of the flows in and out of a given bus.

    Parameters
    ----------
    results : dict
        Results of the solph.EnergySystem (results['main'])
    label_bus : str
        Label of the bus that is viewed.
    label_demand : str
        Label of the oemof Node, to which the energy is delivered.
    interval : list
        List with two elements with a startdate and an end date.
        If `interval=None`, the start and end of the whole results
        index is used.
    title : str
        Title of plot.

    Returns
    -------
    Nothing, shows a plot.
    """
    timeindex = next(iter(results.values()))['sequences'].index

    if interval is None:
        interval = [timeindex[0], timeindex[-1]]

    r = results

    flows = [x for x in r.keys() if x[1] is not None]
    flows_to_bus = [x for x in flows if x[1].label == label_bus]
    flows_from_bus = [x for x in flows if x[0].label == label_bus]
    flows_from_without_demand = [x for x in flows_from_bus
                                 if x[1].label != label_demand]

    # build df
    df_in = pd.DataFrame(index=timeindex)
    for flow in flows_to_bus:
        df_in[flow[0].label] = r[flow]["sequences"]["flow"]

    df_out = pd.DataFrame(index=timeindex)
    for flow in flows_from_without_demand:
        df_out[flow[1].label] = r[flow]["sequences"]["flow"]
    ######################
    l_data_in = []
    for col in df_in.columns:
        l_data_in.append(df_in[col].values)

    l_data_out = []
    for col in df_out.columns:
        l_data_out.append(-df_out[col].values)

    demand = solph.views.node(r, label_bus)["sequences"][
        ((label_bus, label_demand), 'flow')]
    #########
    fig, ax1 = plt.subplots()

    ax1.set_ylabel('Wärmeleistung [kW]')
    ax1.plot(demand, linewidth=1.5, color='000000', label='demand',
             drawstyle='steps-mid')
    ax1.stackplot(df_in.index, l_data_in,
                  labels=list(df_in.columns))
    ax1.stackplot(df_out.index, l_data_out,
                  labels=list(df_out.columns))
    ax1.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
               fontsize='x-small',
               mode="expand", borderaxespad=0, ncol=2)
    ax1.tick_params(axis='y')

    plt.xlim(interval[0], interval[1])
    plt.legend()
    plt.title(title)
    fig.tight_layout()
    plt.show()


def stacked_bar_plot(df, show=True, ylabel=None, title=None):
    """Creates a stacked bar plot."""
    plt.figure()

    names = df.columns
    width = 0.35
    r = [n for n in range(len(df.columns))]

    plt.bar(r, df.iloc[0].values, width=width, label=df.iloc[0].name)
    for i in [x + 1 for x in range(len(df.index) - 1)]:
        bottom = np.zeros(len(df.iloc[0].values))
        for j in range(i):
            bottom = bottom + df.iloc[j].values
        plt.bar(r, df.iloc[i].values, bottom=bottom, width=width,
                label=df.iloc[i].name)

    plt.xticks(r, names, rotation='vertical')
    plt.xlabel("Scenario")
    plt.ylabel(ylabel=ylabel)
    plt.legend()
    plt.tight_layout()
    plt.title(title)

    if show:
        plt.show()


def grouped_bar_plot(df, show=True, ylabel=None, xlabel="Label", title=None):
    """Creates a figure with a grouped bar plot of a given DataFrame."""
    df.plot(kind='bar')
    plt.xlabel(xlabel=xlabel)
    plt.ylabel(ylabel=ylabel)
    plt.legend()
    plt.tight_layout()
    plt.title(title)
    if show:
        plt.show()


def plot_invest_values(pf, title=None, show=True, path=None,
                       filename="investment.png"):
    """
    Creates plot for the investment decisions (converter and storges)
    of the pareto front class.

    Parameters
    ----------
    pf : q100opt.scenario_tools.ParetoFront
        Pareto front class with processed results.
    title : str
        Title of figure.
    show : bool
        Show the plot.
    """
    # get the data
    idx = pd.IndexSlice

    df_invest_conv = pf.results['costs'].loc[
        :, idx["capex", "converter", :, "invest_value"]
    ]
    df_invest_conv.columns = df_invest_conv.columns.get_level_values(2)

    df_invest_store = pf.results['costs'].loc[
        :, idx["capex", "storage", :, "invest_value"]
    ]
    df_invest_store.columns = df_invest_store.columns.get_level_values(2)

    # create the figure
    fig, ax = plt.subplots(1, 2, figsize=[6.4*1.5, 4.8])

    df_invest_conv.plot(ax=ax[0], kind='bar')
    ax[0].set_ylabel('Installed capacity [kW]')
    ax[0].set_xlabel('Energy converter units')
    ax[0].legend()
    ax[0].grid(axis='y')

    df_invest_store.plot(ax=ax[1], kind='bar')
    ax[1].set_ylabel('Installed capacity [kWh]')
    ax[1].set_xlabel('Energy storages')
    ax[1].legend()
    ax[1].grid(axis='y')

    plt.legend()

    # fig.title(title)
    fig.suptitle(title)
    fig.tight_layout()

    if show:
        plt.show()

    if path:
        fig.savefig(os.path.join(path, filename))
