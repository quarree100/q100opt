from matplotlib import pyplot as plt
import oemof.solph as solph
import networkx as nx
import logging
import pandas as pd
import numpy as np

try:
    import pygraphviz
except ImportError:
    logging.info('Module pygraphviz not found: Graph was not plotted.')


def analyse_emissions(results):
    """Performs analysis of emissions.

    Parameters
    ----------
    results : dict
        Results of oemof.solph Energysystem.

    Returns
    -------
    dict :  Table with detailed emission analysis,
            containing 2 keys: 'summary' and 'sequences'.
    """
    return analyse_flow_attribute(results, keyword='emission_factor')


def analyse_costs(results):
    """Performs a cost analysis.

    Parameters
    ----------
    results : dict
        Results of oemof.solph Energysystem.

    Returns
    -------
    dict :  Table with detailed cost summary,
            containing 3 keys: 'capex', 'opex' and 'all'.
    """
    costs = {
        'capex': analyse_capex(results),
        'opex': analyse_flow_attribute(results, keyword='variable_costs'),
    }

    capex = pd.concat({'capex': costs['capex']}, names=['cost_type'])
    opex = pd.concat({'opex': costs['opex']['summary']}, names=['cost_type'])
    all = pd.concat([capex, opex])

    costs.update({'all': all})

    return costs


def analyse_capex(results):
    """Analysis and Summary of the investment costs of the EnergySystem.

    Parameters
    ----------
    results : q100opt.DistrictScenario.results
        The results Dictionary of the District Scenario class
        (a dictionary containing the processed oemof.solph.results
        with the key 'main' and the oemof.solph parameters
        with the key 'param'.)

    Returns
    -------
    pd.DataFrame :
        The table contains both the parameter and result value
        of the Investment objects.
         - Columns: 'ep_costs', 'offset', 'invest_value' and 'costs'
         - Index:
            - First level: 'converter' or 'storage
              (Converter are all flows comming from a solph.Transformer or
              a solph.Source)
            - Second level: Label of the corresponding oemof.solph component:
              in case of 'converter', the label from which the flow is comming.
              in case of 'storage', the label of the GenericStorage.
    """
    # energy converter units
    df_converter = get_invest_converter_table(results)
    df_converter['category'] = 'converter'

    # energy storages units
    df_storages = get_invest_storage_table(results)
    df_storages['category'] = 'storage'

    df_result = pd.concat([df_converter, df_storages])

    df_result.index = pd.MultiIndex.from_frame(
        df_result[['category', 'label']])
    df_result.drop(df_result[['category', 'label']], axis=1, inplace=True)

    return df_result


def get_invest_converter(results):
    """
    Gets the keys of investment converter units of the energy system.
    Only the flows from a solph.Transformer or a solph.Source are considered.
    """
    return [
        x for x in results.keys()
        if hasattr(results[x]['scalars'], 'invest')
        if isinstance(x[0], solph.Transformer) or
           isinstance(x[0], solph.Source)
    ]


def get_invest_storages(results):
    """
    Gets the investment storages of the energy system.
    Only the investment of the solph.components.GenericStorage is considered,
    and not a investment in the in- or outflow.
    """
    return [
        x for x in results.keys()
        if x[1] is None
        if hasattr(results[x]['scalars'], 'invest')
        if isinstance(x[0], solph.components.GenericStorage)
    ]


def get_invest_converter_table(results):
    """
    Returns a table with a summary of investment flows of energy converter
    units. These are oemof.solph.Flows comming from a solph.Transformer or
    a solph.Source.

    Parameters
    ----------
    results : q100opt.DistrictScenario.results
        The results Dictionary of the District Scenario class
        (a dictionary containing the processed oemof.solph.results
        with the key 'main' and the oemof.solph parameters
        with the key 'param'.)

    Returns
    -------
    pd.DataFrame :
        The table contains both the parameter and result value
        of the Investment objects.
         - Columns: 'label', 'ep_costs', 'offset', 'invest_value' and 'costs'
           The 'label' column is the label of the corresponding
           oemof.solph.Transformer or Source, from that the flow is coming.
    """
    converter_units = get_invest_converter(results['main'])
    return get_invest_table(results, converter_units)


def get_invest_storage_table(results):
    """
    Returns a table with a summary of investment flows of all
    oemof.solph.components.GeneicStorage units.

    results : q100opt.DistrictScenario.results
        The results Dictionary of the District Scenario class
        (a dictionary containing the processed oemof.solph.results
        with the key 'main' and the oemof.solph parameters
        with the key 'param'.)

    Returns
    -------
    pd.DataFrame :
        The table contains both the parameter and result value
        of the Investment objects.
         - Columns: 'label', 'ep_costs', 'offset', 'invest_value' and 'costs'
           The 'label' column is the label of the corresponding oemof.solph
           label, which is the label from which the flow is coming.
    """
    storages = get_invest_storages(results['main'])
    return get_invest_table(results, storages)


def get_invest_table(results, keys):
    """
    Returns the investment data for a list of "results keys".

    Parameters
    ----------
    results : dict
        oemof.solph results dictionary (results['main])
    keys : list
        Keys of flows and nodes

    Returns
    -------
    pd.DataFrame :
        The table contains both the parameter and result value
        of the Investment objects.
         - Columns: 'label', 'ep_costs', 'offset', 'invest_value' and 'costs'
           The 'label' column is the label of the corresponding oemof.solph
           label, which is the label from which the flow is coming.
    """
    invest_lab = [x[0].label for x in keys]

    df = pd.DataFrame(data=invest_lab, columns=['label'])
    df['ep_costs'] = [results['param'][x]['scalars']['investment_ep_costs']
                      for x in keys]
    df['offset'] = [results['param'][x]['scalars']['investment_offset']
                    for x in keys]
    df['invest_value'] = [results['main'][x]['scalars']['invest']
                          for x in keys]
    df['costs'] = df['invest_value'] * df['ep_costs'] + \
                  df['offset'] * np.sign(df['invest_value'])
    return df


def analyse_flow_attribute(des_results, keyword='variable_costs'):
    """Analysis and Summary of flow attribute keyword of the EnergySystem.

    Parameters
    ----------
    des_results : q100opt.DistrictScenario.results
        The results Dictionary of the District Scenario class
        (a dictionary containing the processed oemof.solph.results
        with the key 'main' and the oemof.solph parameters
        with the key 'param'.)
    keyword : str
        Keyword for that values are analyzed,
        e.g. variable_costs or emission_factor.

    Returns
    -------
    dict :  All relevant data with variable_costs.
            Keys of dictionary: 'summary' and 'sequences'.
    """
    param = des_results['param']
    results = des_results['main']

    var_cost_flows = get_attr_flows(des_results, key=keyword)
    df = pd.DataFrame(index=next(iter(results.values()))['sequences'].index)
    len_index = len(df)

    # define columns of result dataframe
    if keyword == 'variable_costs':
        key_product = 'costs'
    elif keyword == 'emission_factor':
        key_product = 'emissions'
    else:
        key_product = 'product'

    for flow in var_cost_flows:

        if isinstance(flow[0], solph.Source):
            category = 'source'
            label = flow[0].label
        elif isinstance(flow[0], solph.Transformer):
            category = 'converter'
            label = flow[0].label
        elif isinstance(flow[1], solph.Sink):
            category = 'sink'
            label = flow[1].label
        else:
            label = flow[0].label + '-' + flow[1].label
            category = 'unknown'
            logging.warning(
                "Flow/Node category of {} not specified!".format(label)
            )

        if keyword in param[flow]['scalars'].keys():
            df[(category, label, keyword)] = param[flow]['scalars'][keyword]
        else:
            df[(category, label, keyword)] = \
                param[flow]['sequences'][keyword].values[:len_index]

        # 2) get flow results
        df[(category, label, 'flow')] = results[flow]["sequences"].values

        # 3) calc a * b
        df[(category, label, key_product)] = \
            df[(category, label, keyword)] * df[(category, label, 'flow')]

        df.columns = pd.MultiIndex.from_tuples(
            list(df.columns), names=('category', 'label', 'value')
        )

    df.sort_index(axis=1, inplace=True)

    df_sum = df.iloc[:, df.columns.isin(['flow', key_product], level=2)].sum()

    df_summary = df_sum.unstack(level=2)

    df_summary['var_' + key_product + '_av_flow'] = \
        df_summary[key_product] / df_summary['flow']

    df_mean = \
        df.iloc[:, df.columns.get_level_values(2) == keyword]\
            .mean().unstack(level=2).rename(
            columns={keyword: 'var_' + key_product + '_av_param'})

    df_summary = df_summary.join(df_mean)

    return {'summary': df_summary,
            'sequences': df}


def get_attr_flows(results, key='variable_costs'):
    """
    Return all flows of an EnergySystem for a given attribute,
    which is not zero.

    Parameters
    ----------
    results : dict
        Results dicionary of the oemof.solph optimisation including the
        Parameters with key 'param'.
    key : str

    Returns
    -------
    list : List of flows, where a non zero attribute value is given either
           at the 'scalars' or 'sequences'.
    """
    param = results['param']

    list_keys = list(param.keys())

    var_scalars = [
        x for x in list_keys
        if key in param[x]['scalars'].keys()
        if abs(param[x]['scalars'][key]) > 0
    ]

    var_sequences = [
        x for x in list_keys
        if key in param[x]['sequences'].keys()
        if abs(param[x]['sequences'][key].sum()) > 0
    ]

    var_cost_flows = var_scalars + var_sequences

    return var_cost_flows


def get_attr_flow_results(des_results, key='variable_costs'):
    """
    Return the parameter and flow results for all flows of an EnergySystem
    for a given attribute, which is not zero.

    Parameters
    ----------
    des_results : dict
        Results of district energy system. Must have the keys: 'main', 'param'.
    key : str
        Flow attribute.

    Returns
    -------
    pd.DataFrame : Multiindex DataFrame.
        - Index : Timeindex of oemof.solph.EnergySystem.
        - First column index level: <from>-<to>, where from an to are the
          labels of the Nodes.
        - Second column index level:
            - attribute parameter
            - resulting flow value
            - product of parameter and flow column
    """
    attr_flows = get_attr_flows(des_results, key=key)

    param = des_results['Param']
    results = des_results['Main']

    df = pd.DataFrame(index=next(iter(results.values()))['sequences'].index)

    len_index = len(df)

    for flow in attr_flows:

        label = flow[0].label + '-' + flow[1].label

        # 1) get parameters
        if key in param[flow]['scalars'].keys():
            df[(label, key)] = param[flow]['scalars'][key]
        else:
            df[(label, key)] = param[flow]['sequences'][key].values[:len_index]

        # 2) get flow results
        df[(label, 'flow')] = results[flow]["sequences"].values

        # 3) calc a * b
        if key == 'variable_costs':
            key_product = 'costs'
        elif key == 'emission_factor':
            key_product = 'emissions'
        else:
            key_product = 'product'

        df[(label, key_product)] = df[(label, key)] * df[(label, 'flow')]

        df.columns = pd.MultiIndex.from_tuples(
            list(df.columns), names=('from-to', 'value')
        )

    return df


def get_boundary_flows(results):
    """Gets the results of flows of the sinks and sources.

    Parameters
    ----------
    results : dict
        Results of the oemof.solph.Energysystem (results['main'])

    Returns
    -------
    dict :  Dictionary with two keys:
            - 'sequences': pandas.DataFrame
              with the flow values at each timestep. The columns are a tuple:
              ('sink', <label_of_sink>) for all solph.Sinks
              ('source', <label_of_source>) for all solph.Sources
            - 'summary': pandas.Series (sum of 'sequences')
    """
    label_sources = get_label_sources(results)
    label_sinks = get_label_sinks(results)

    # sources
    data_sources = \
        [solph.views.node(results, lab)['sequences'] for lab in label_sources]
    column_sources = \
        [('source', lab) for lab in label_sources]

    df_sources = pd.concat(data_sources, axis=1, join='inner')
    df_sources.columns = column_sources

    # sinks
    data_sinks = \
        [solph.views.node(results, lab)['sequences'] for lab in label_sinks]
    column_sinks = \
        [('sink', lab) for lab in label_sinks]

    df_sinks = pd.concat(data_sinks, axis=1, join='inner')
    df_sinks.columns = column_sinks

    df_seq = pd.concat([df_sources, df_sinks], axis=1)

    df_sum = df_seq.sum()

    return {'summary': df_sum,
            'sequences': df_seq}


def get_sum_flow(results, label):
    """Return the sum of a flow."""
    return solph.views.node(results, label)["sequences"].sum()[0]


def get_label_sources(results):
    """Return a list of sources of the results of an solph.Energysystem."""
    return [x[0].label for x in results.keys()
            if isinstance(x[0], solph.Source)]


def get_label_sinks(results):
    """Return a list of sinks of the results of an solph.Energysystem."""
    return [x[1].label for x in results.keys()
            if isinstance(x[1], solph.Sink)]


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
