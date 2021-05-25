# -*- coding: utf-8 -*-

"""

This module holds functions processing the results
of an oemof.solph optimisation model, that are used by methods of the classes
`q100opt.scenario_tools.DistrictScenario` and
`q100opt.scenario_tools.ParetoFront`.

Please use this module with care. It is work in progress!

Contact: Johannes Röder <johannes.roeder@uni-bremen.de>

SPDX-License-Identifier: MIT

"""
import logging

import numpy as np
import oemof.solph as solph
import pandas as pd
from oemof.solph import views


def analyse_emissions(results):
    """
    Performs analysis of emissions.

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
    """
    Performs a cost analysis.

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
    opex = pd.concat({'opex': costs['opex']['sum']}, names=['cost_type'])
    all = pd.concat([capex, opex])

    costs.update({'all': all})

    return costs


def analyse_capex(results):
    """
    Analysis and Summary of the investment costs of the EnergySystem.

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
        if isinstance(x[0], solph.Transformer) or isinstance(
            x[0], solph.Source)
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
    df['costs'] = df['invest_value'] * df['ep_costs'] + df[
        'offset'] * np.sign(df['invest_value'])

    return df


def analyse_flow_attribute(des_results, keyword='variable_costs'):
    """
    Analysis and Summary of flow attribute keyword of the EnergySystem.

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
        df.iloc[:, df.columns.get_level_values(2) == keyword].mean().unstack(
            level=2).rename(columns={
                keyword: 'var_' + key_product + '_av_param'})

    df_summary = df_summary.join(df_mean)

    return {'sum': df_summary,
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


def get_all_sequences(results):
    """..."""
    d_node_types = {
        'sink': solph.Sink,
        'source': solph.Source,
        'transformer': solph.Transformer,
        'storage_flow': solph.GenericStorage,
    }

    l_df = []

    for typ, solph_class in d_node_types.items():
        group = {
            k: v["sequences"]
            for k, v in results.items()
            if k[1] is not None
            if isinstance(k[0], solph_class) or isinstance(k[1], solph_class)
        }

        if bool(group):
            df = views.convert_to_multiindex(group)
            df_mi = df.columns.to_frame()
            df_mi.reset_index(drop=True, inplace=True)
            df_mi['from'] = [x.label for x in df_mi['from']]
            df_mi['to'] = [x.label for x in df_mi['to']]
            df_mi['type'] = typ
            df.columns = pd.MultiIndex.from_frame(
                df_mi[['type', 'from', 'to']]
            )

            l_df.append(df)

    df_results = pd.concat(l_df, axis=1)

    # add storage content with extra type=storage_content
    group = {
        k: v["sequences"]
        for k, v in results.items()
        if k[1] is None
        if isinstance(k[0], solph.GenericStorage) or isinstance(
            k[1], solph.GenericStorage)
    }

    if bool(group):
        df = views.convert_to_multiindex(group)
        df_mi = df.columns.to_frame()
        df_mi.reset_index(drop=True, inplace=True)
        df_mi['from'] = [x.label for x in df_mi['from']]
        df.columns = pd.MultiIndex.from_frame(df_mi[['type', 'from', 'to']])

        df_results = pd.concat([df_results, df], axis=1)

    return df_results


def get_boundary_flows(results):
    """
    Gets the results of flows of the sinks and sources.

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

    return {'sum': df_sum,
            'sequences': df_seq}


def get_trafo_flow(results, label_bus):
    """
    Returns the flows from a solph.Transformer for a given solph.Bus.

    Parameters
    ----------
    results : dict
        Results of the oemof.solph.Energysystem (results['main'])
    label_bus : str
        Label of bus.

    Returns
    -------
    dict :  Dictionary with two keys:
            - 'sequences': pandas.DataFrame
              with the flow values at each timestep. The columns are a tuple:
              ('sink', <label_of_sink>) for all solph.Sinks
              ('source', <label_of_source>) for all solph.Sources
            - 'summary': pandas.Series (sum of 'sequences')
    """
    flows = [
        x for x in results.keys()
        if x[1] is not None
        if isinstance(x[0], solph.Transformer)
        if x[1].label == label_bus
    ]

    l_table = [results[x]['sequences']['flow'] for x in flows]
    l_labels = [x[0].label for x in flows]

    df_seq = pd.concat(l_table, axis=1, join='inner')
    df_seq.columns = [('converter', lab) for lab in l_labels]

    df_sum = df_seq.sum()

    return {'sum': df_sum,
            'sequences': df_seq}


def analyse_bus(results, bus_label):
    """..."""
    df_seq = solph.views.node(results, bus_label)["sequences"]
    df_seq.columns = pd.MultiIndex.from_tuples(df_seq.columns)

    idx = pd.IndexSlice
    df_seq = df_seq.loc[:, idx[:, "flow"]]
    df_seq.columns = df_seq.columns.get_level_values(0)

    df_sum = df_seq.sum()

    return {'sum': df_sum,
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
