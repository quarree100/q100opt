# -*- coding: utf-8 -*-

"""Function for reading data and setting up an oemof-solph EnergySystem.

Please use this module with care. It is work in progress!

Contact: Johannes Röder <johannes.roeder@uni-bremen.de>

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


def load_xlsx_data(filename):
    """Reads all sheets of xlsx file into dictionary."""
    return pd.read_excel(filename, sheet_name=None)


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


def check_nonconvex_invest_type(dct):
    """
    Checks if flow attribute 'invest.nonconvex' is type bool, if the attribute
    is present.

    Parameters
    ----------
    dct : dict
        Dictionary with all paramerters for the oemof-solph components.

    Returns
    -------
    dict : Updated Dictionary is returned.
    """

    for k, v in dct.items():
        if 'invest.nonconvex' in v.columns:
            v['invest.nonconvex'] = v['invest.nonconvex'].astype('bool')
        dct[k] = v

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
    tuple : a tuple containing:
        - nodes ([list]): A list with all oemof-solph Buses of the
            Dataframe table.
        - busd ([dict]): Dictionary with all oemof Bus object.
            Keys are equal to the label of the bus.

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

    for _, b in table.iterrows():

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

    If the invest attribute "offset" is given and if it is > 0, the invest
    attribute "nonconvex=True" is added.

    Parameters
    ----------
    row : pd.Series
        Parameters for single oemof object.

    Returns
    -------
    dict

    """

    index = list(row.index)

    if 'investment' in index:
        if row['investment']:
            invest_attr = {}
            ia_list = [x.split('.')[1] for x in index
                       if x.split('.')[0] == 'invest']
            for ia in ia_list:
                invest_attr[ia] = row['invest.' + ia]

            if 'offset' in ia_list and invest_attr['offset'] > 0:
                invest_attr['nonconvex'] = True

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
    dict : All Flow specific attribues.
    """
    row = row.copy()
    row.dropna(inplace=True)
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
    list : Oemof Sources (non fix sources) objects.
    """
    sources = []

    for _, cs in tab.iterrows():

        cs = cs.copy()
        cs.dropna(inplace=True)

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
    list : List with oemof Source (only fix source) objects.

    Note
    ----
    At the moment, there are no additional flow attributes allowed, and
    `nominal_value` must be given in the table.
    """
    sources_fix = []

    for _, sf in tab.iterrows():

        sf = sf.copy()
        sf.dropna(inplace=True)

        flow_attr = {}

        io = get_invest_obj(sf)

        if io is not None:
            flow_attr['nominal_value'] = None
        else:
            flow_attr['nominal_value'] = sf['flow.nominal_value']

        flow_attr['fix'] = timeseries[sf['label'] + '.fix'].values

        sources_fix.append(
            solph.Source(
                label=sf['label'],
                outputs={busd[sf['to']]: solph.Flow(
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
    list : oemof Sinks (non fix sources) objects.

    Note
    ----
    No investment possible.
    """
    sinks = []

    for _, cs in tab.iterrows():

        cs = cs.copy()
        cs.dropna(inplace=True)

        flow_attr = get_flow_att(cs, timeseries)

        sinks.append(
            solph.Sink(
                label=cs['label'],
                inputs={busd[cs['from']]: solph.Flow(**flow_attr)})
        )

    return sinks


def add_sinks_fix(tab, busd, timeseries):
    """
    Add fix sinks, e.g. for energy demands.

    Parameters
    ----------
    tab : pd.DataFrame
        Table with parameters of Sinks.
    busd : dict
        Dictionary with Buses.
    timeseries : pd.DataFrame
        (Required) Table with all timeseries parameters.

    Returns
    -------
    list : oemof Sinks (non fix sources) objects.

    Note
    ----
    No investment possible.
    """
    sinks_fix = []

    for _, cs in tab.iterrows():

        cs = cs.copy()
        cs.dropna(inplace=True)

        sinks_fix.append(
            solph.Sink(
                label=cs['label'],
                inputs={busd[cs['from']]: solph.Flow(
                    nominal_value=cs['nominal_value'],
                    fix=timeseries[cs['label'] + '.fix'].values
                )})
        )

    return sinks_fix


def add_storages(tab, busd, timeseries=None):
    """

    Parameters
    ----------
    tab : pd.DataFrame
        Table with parameters of Storages.
    busd : dict
        Dictionary with Buses.

    Returns
    -------
    list : oemof GenericStorage components.
    """
    storages = []

    for _, s in tab.iterrows():

        s = s.copy()
        s.dropna(inplace=True)

        att = list(s.index)

        att_storage = [
            x.split('.')[1] for x in att if x.split('.')[0] == 'storage']

        att_inflow = [
            x.split('.')[1] for x in att if x.split('.')[0] == 'inflow']

        att_outflow = [
            x.split('.')[1] for x in att if x.split('.')[0] == 'outflow']

        sto_attr = {}
        for fa in att_storage:
            if s['storage.' + fa] == "series":
                sto_attr[fa] = timeseries[s['label'] + '.' + fa].values
            else:
                sto_attr[fa] = s['storage.' + fa]

        in_attr = {}
        for fa in att_inflow:
            in_attr[fa] = s['inflow.' + fa]

        out_attr = {}
        for fa in att_outflow:
            out_attr[fa] = s['outflow.' + fa]

        io = get_invest_obj(s)

        if io is not None:
            sto_attr['nominal_storage_capacity'] = None
            # makes sure that not both constraint `nominal_value` and
            # `invest_relation_input_output` are set in investment case:
            # in_attr = {}
            # out_attr = {}

        storages.append(
            solph.components.GenericStorage(
                label=s['label'],
                inputs={busd[s['bus']]: solph.Flow(**in_attr)},
                outputs={busd[s['bus']]: solph.Flow(**out_attr)},
                investment=io,
                **sto_attr,
            )
        )

    return storages


def add_transformer(tab, busd, timeseries=None):
    """

    Parameters
    ----------
    tab : pandas.DataFrame
        Table with all Transformer parameter
    busd : dict
        Dictionary with all oemof-solph Bus objects.
    timeseries : pandas.DataFrame
        Table with all Timeseries for Transformer.

    Returns
    -------
    list : oemof-solph Transformer objects.

    """
    transformer = []

    for _, t in tab.iterrows():

        row = t.copy()
        row.dropna(inplace=True)
        att = list(row.index)

        flow_out1_attr = get_flow_att(row, timeseries)

        att_inflow1 = [
            x.split('.')[1] for x in att if x.split('.')[0] == 'inflow1']

        in_attr1 = {}
        for fa in att_inflow1:
            in_attr1[fa] = row['inflow1.' + fa]

        io = get_invest_obj(row)

        if io is not None:
            flow_out1_attr['nominal_value'] = None

        d_in = {busd[row['in_1']]: solph.Flow(**in_attr1)}

        if 'nonconvex_flow' in row.index:
            if row['nonconvex_flow'] == 1:
                nc = solph.NonConvex()
            else:
                nc = None
        else:
            nc = None

        d_out = {busd[row['out_1']]: solph.Flow(
            investment=io,
            nonconvex=nc,
            **flow_out1_attr
        )}

        # check if timeseries in conversion factors and convert to float
        att = list(row.index)
        eff_list = [x for x in att if x.split('_')[0] == 'eff']
        d_eff = {}
        for eff in eff_list:
            if row[eff] == 'series':
                d_eff[eff] = timeseries[row['label'] + '.' + eff]
            else:
                d_eff[eff] = float(row[eff])

        cv = {busd[row['in_1']]: d_eff['eff_in_1'],
              busd[row['out_1']]: d_eff['eff_out_1']}

        # update inflows and conversion factors, if a second inflow bus label
        # is given
        if not (row['in_2'] == '0' or row['in_2'] == 0):
            d_in.update({busd[row['in_2']]: solph.Flow()})
            cv.update({busd[row['in_2']]: d_eff['eff_in_2']})

        # update outflows and conversion factors, if a second outflow bus label
        # is given
        if not (row['out_2'] == '0' or row['out_2'] == 0):
            d_out.update({busd[row['out_2']]: solph.Flow()})
            cv.update({busd[row['out_2']]: d_eff['eff_out_2']})

        transformer.append(
            solph.Transformer(
                label=row['label'],
                inputs=d_in,
                outputs=d_out,
                conversion_factors=cv
            )
        )

    return transformer


def add_links(tab, busd):
    """..."""
    links = []

    for _, t in tab.iterrows():
        links.append(
            solph.custom.Link(
                label=t['label'],
                inputs={busd[t['b0']]: solph.Flow(),
                        busd[t['b1']]: solph.Flow()},
                outputs={busd[t['b0']]: solph.Flow(),
                         busd[t['b1']]: solph.Flow()},
                conversion_factors={
                    (busd[t['b0']], busd[t['b1']]): t['cv_0>1'],
                    (busd[t['b1']], busd[t['b0']]): t['cv_1>0'],
                }
            )
        )

    return links
