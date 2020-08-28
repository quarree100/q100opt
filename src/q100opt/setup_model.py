# -*- coding: utf-8 -*-

"""Function for reading data and setting up an oemof-solph EnergySystem.

SPDX-License-Identifier: MIT

"""
import os

import oemof.solph as solph
import pandas as pd


def load_csv_data(path):
    """Loading csv data.

    Loading all csv files of the given path as pandas DataFrames into a
    dictionary.
    The keys of the dictionary are the names of the csv files
    (without .csv).

    Parameters
    ----------
    path : str

    Returns
    -------
    dict
    """
    dct = {}

    for name in os.listdir(path):

        key = name.split('.csv')[0]
        val = pd.read_csv(os.path.join(path, name))
        dct.update([(key, val)])

    return dct


def check_active(dct):
    """
    Checks for active components.

    Delete not "active" rows, and the column
    'active' of all components dataframes.

    Parameters
    ----------
    dct : dict
        Holding the Dataframes of solph components

    Returns
    -------
    dict
    """
    for k, v in dct.items():
        if 'active' in v.columns:
            v_new = v[v['active'] == 1].copy()
            v_new.drop('active', axis=1, inplace=True)
            dct[k] = v_new

    return dct


def add_buses(table):
    """Instantiates the oemof-solph.Buses based on tabular data.

    Retruns the Buses in a Dictionary and in a List.
    If excess and shortage is given, additional sinks and sources are created.

    Parameters
    ----------
    table : pandas.DataFrame
        Dateframe with all Buses.

    Returns
    -------
    nodes : list
        A list with all oemof-solph Buses of the Dataframe table.
    busd : dict
        Dictionary with all oemof Bus object. Keys are equal to the label of
        the bus.

    Examples
    --------
    >>> import pandas as pd
    >>> from q100opt.setup_model import add_buses
    >>> data_bus = pd.DataFrame([['label_1', 0, 0, 0, 0],
    ... ['label_2', 0, 0, 0, 0]],
    ... columns=['label', 'excess', 'shortage', 'shortage_costs',
    ... 'excess_costs'])
    >>> nodes, buses = add_buses(data_bus)
    """
    busd = {}
    nodes = []

    for i, b in table.iterrows():

        bus = solph.Bus(label=b['label'])
        nodes.append(bus)

        busd[b['label']] = bus
        if b['excess']:
            nodes.append(
                solph.Sink(label=b['label'] + '_excess',
                           inputs={busd[b['label']]: solph.Flow(
                               variable_costs=b['excess_costs'])})
            )
        if b['shortage']:
            nodes.append(
                solph.Source(label=b['label'] + '_shortage',
                             outputs={busd[b['label']]: solph.Flow(
                                 variable_costs=b['shortage_costs'])})
            )

    return nodes, busd


def get_invest_obj(row):
    """
    Filters all attributes for the investment attributes with
    the prefix`invest.`, if attribute 'investment' occurs, and if attribute
    `investment` is set to 1.

    Parameters
    ----------
    row : pd.Series
        Parameters for single oemof object.

    Returns
    -------
    invest_objects : dict

    """

    index = list(row.index)

    if 'investment' in index:
        if row['investment']:
            invest_attr = {}
            ia_list = [x.split('.')[1] for x in index
                       if x.split('.')[0] == 'invest']
            for ia in ia_list:
                invest_attr[ia] = row['invest.' + ia]
            invest_object = solph.Investment(**invest_attr)

        else:
            invest_object = None
    else:
        invest_object = None

    return invest_object


def get_flow_att(row, ts):
    """

    Parameters
    ----------
    row : pd.Series
        Series with all attributes given by the parameter table (equal 1 row)
    ts : pd.DataFrame
        DataFrame with all input time series for the oemof-solph model.

    Returns
    -------
    flow_attr : dict
        Dictionary with all Flow specific attribues.
    """

    att = list(row.index)
    fa_list = [x.split('.')[1] for x in att if x.split('.')[0] == 'flow']

    flow_attr = {}

    for fa in fa_list:
        if row['flow.' + fa] == 'series':
            flow_attr[fa] = ts[row['label'] + '.' + fa].values
        else:
            flow_attr[fa] = float(row['flow.' + fa])

    return flow_attr


def add_sources(tab, busd, timeseries=None):
    """

    Parameters
    ----------
    tab : pd.DataFrame
        Table with parameters of Sources.
    busd : dict
        Dictionary with Buses.
    timeseries : pd.DataFrame
        (Optional) Table with all timeseries parameters.

    Returns
    -------
    sources : list
        List with oemof Source (non fix sources) objects.
    """
    sources = []

    for i, cs in tab.iterrows():

        flow_attr = get_flow_att(cs, timeseries)

        io = get_invest_obj(cs)

        if io is not None:
            flow_attr['nominal_value'] = None

        sources.append(
            solph.Source(
                label=cs['label'],
                outputs={busd[cs['to']]: solph.Flow(
                    investment=io, **flow_attr)})
        )

    return sources


def add_sources_fix(tab, busd, timeseries):
    """

    Parameters
    ----------
    tab : pd.DataFrame
        Table with parameters of Sources.
    busd : dict
        Dictionary with Buses.
    timeseries : pd.DataFrame
        Table with all timeseries parameters.

    Returns
    -------
    sources : list
        List with oemof Source (only fix source) objects.

    Note
    ----
    At the moment, there are no additional flow attributes allowed, and
    `nominal_value` must be given in the table.
    """
    sources_fix = []

    for k, l in tab.iterrows():

        flow_attr = {}

        io = get_invest_obj(l)

        if io is not None:
            flow_attr['nominal_value'] = None
        else:
            flow_attr['nominal_value'] = l['flow.nominal_value']

        flow_attr['fix'] = timeseries[l['label'] + '.fix'].values

        sources_fix.append(
            solph.Source(
                label=l['label'],
                outputs={busd[l['to']]: solph.Flow(
                    **flow_attr, investment=io)})
        )

    return sources_fix


def add_sinks(tab, busd, timeseries=None):
    """

    Parameters
    ----------
    tab : pd.DataFrame
        Table with parameters of Sinks.
    busd : dict
        Dictionary with Buses.
    timeseries : pd.DataFrame
        (Optional) Table with all timeseries parameters.

    Returns
    -------
    sources : list
        List with oemof Source (non fix sources) objects.
    """
    sinks = []

    for i, cs in tab.iterrows():

        flow_attr = get_flow_att(cs, timeseries)

        sinks.append(
            solph.Sink(
                label=cs['label'],
                inputs={busd[cs['from']]: solph.Flow(**flow_attr)})
        )

    return sinks
