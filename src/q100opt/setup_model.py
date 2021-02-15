# -*- coding: utf-8 -*-

"""Function for reading data and setting up an oemof-solph EnergySystem.

SPDX-License-Identifier: MIT

"""
import datetime
import logging
import os
from copy import deepcopy

import oemof.solph as solph
import pandas as pd
import pickle

from q100opt.external import Scenario
from q100opt.postprocessing import analyse_costs


class DistrictScenario(Scenario):

    """Scenario class for urban energy systems"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_path = kwargs.get("input_path", None)
        self.emission_limit = kwargs.get("emission_limit", None)
        self.location = kwargs.get("location", None)
        self.number_of_time_steps = \
            kwargs.get("number_of_time_steps", 10)

    def load_csv(self, path=None):
        if path is not None:
            self.location = path
        self.table_collection = load_csv_data(self.location)
        return self

    def check_input(self):
        self.table_collection = check_active(self.table_collection)
        self.table_collection = check_nonconvex_invest_type(
            self.table_collection)
        return self

    def initialise_energy_system(self):
        """Initialises the oemof.solph Energysystem."""
        date_time_index = pd.date_range(
            "1/1/{0}".format(self.year),
            periods=self.number_of_time_steps,
            freq="H"
        )
        self.es = solph.EnergySystem(timeindex=date_time_index)

    def create_nodes(self):
        nd = self.table_collection
        nod, busd = add_buses(nd['Bus'])
        nod.extend(
            add_sources(nd['Source'], busd, nd['Timeseries']) +
            add_sources_fix(nd['Source_fix'], busd, nd['Timeseries']) +
            add_sinks(nd['Sink'], busd, nd['Timeseries']) +
            add_sinks_fix(nd['Sink_fix'], busd, nd['Timeseries']) +
            add_storages(nd['Storages'], busd) +
            add_transformer(nd['Transformer'], busd, nd['Timeseries'])
        )
        return nod

    def table2es(self):
        if self.es is None:
            self.initialise_energy_system()
        self.check_input()
        nodes = self.create_nodes()
        self.es.add(*nodes)

    def add_emission_constr(self):
        if self.emission_limit is not None:
            if self.model is not None:
                solph.constraints.generic_integral_limit(
                    self.model, keyword='emission_factor',
                    limit=self.emission_limit)
            else:
                ValueError("The model must be created first.")
        return self

    def solve(self, with_duals=False, tee=True, logfile=None, solver=None):

        if self.es is None:
            self.table2es()

        self.create_model()
        self.add_emission_constr()

        logging.info("Optimising using {0}.".format(solver))

        if with_duals:
            self.model.receive_duals()

        if self.debug:
            filename = os.path.join(
                solph.helpers.extend_basic_path("lp_files"), "q100opt.lp"
            )
            logging.info("Store lp-file in {0}.".format(filename))
            self.model.write(
                filename, io_options={"symbolic_solver_labels": True}
            )

        self.model.solve(
            solver=solver, solve_kwargs={"tee": tee, "logfile": logfile}
        )
        self.es.results["main"] = solph.processing.results(self.model)
        self.es.results["meta"] = solph.processing.meta_results(self.model)
        self.es.results["param"] = solph.processing.parameter_as_dict(self.es)
        self.es.results["meta"]["scenario"] = self.scenario_info(solver)
        if self.location is not None:
            self.es.results["meta"]["in_location"] = self.location
        self.es.results['meta']["datetime"] = datetime.datetime.now()
        self.es.results["meta"]["solph_version"] = solph.__version__
        self.es.results['meta']['emission_limit'] = self.emission_limit
        self.es.results['costs'] = self.model.objective()
        self.es.results['table_collection'] = self.table_collection
        if hasattr(self.model, 'integral_limit_emission_factor'):
            self.es.results['emissions'] = \
                self.model.integral_limit_emission_factor()
        self.results = self.es.results
        self.results['timeindex'] = self.es.timeindex

    def plot(self):
        pass

    def tables_to_csv(self, path=None):
        """Dump scenario into a csv-collection."""
        if path is None:
            bpath = os.path.join(os.path.expanduser("~"), ".q100opt")
            if not os.path.isdir(bpath):
                os.mkdir(bpath)
            dpath = os.path.join(bpath, "dumps")
            if not os.path.isdir(dpath):
                os.mkdir(dpath)
            path = os.path.join(dpath, "csv_export")
            if not os.path.isdir(path):
                os.mkdir(path)

        for name, df in self.table_collection.items():
            name = name.replace(" ", "_") + ".csv"
            filename = os.path.join(path, name)
            df.to_csv(filename)
        logging.info("Scenario saved as csv-collection to {0}".format(path))

    def tables_to_excel(self, dpath=None, filename=None):
        """Dump scenario into an excel-file."""
        if dpath is None:
            bpath = os.path.join(os.path.expanduser("~"), ".q100opt")
            if not os.path.isdir(bpath):
                os.mkdir(bpath)
            dpath = os.path.join(bpath, "dumps")
            if not os.path.isdir(dpath):
                os.mkdir(dpath)

        if filename is None:
            filename = "ds_dump.xlsx"

        writer = pd.ExcelWriter(os.path.join(dpath, filename))
        for name, df in sorted(self.table_collection.items()):
            df.to_excel(writer, name)
        writer.save()
        logging.info("Scenario saved as excel file to {0}".format(filename))

    def dump(self, path=None, filename=None):
        """Dump energysystem of District scenario."""
        if path is None:
            bpath = os.path.join(os.path.expanduser("~"), ".q100opt")
            if not os.path.isdir(bpath):
                os.mkdir(bpath)
            dpath = os.path.join(bpath, "dumps")
            if not os.path.isdir(dpath):
                os.mkdir(dpath)
            path = os.path.join(dpath, "energysystem")
            if not os.path.isdir(path):
                os.mkdir(path)

        if filename is None:
            filename = "ds_dump.oemof"

        if not os.path.isdir(path):
            os.makedirs(path)

        if self.model is not None:
            setattr(self, 'model', None)
        if self.es is not None:
            setattr(self, 'es', None)

        pickle.dump(self, open(os.path.join(path, filename), "wb"))

        logging.info("DistrictScenario dumped"
                     " to {} as {}".format(path, filename))

    def restore_des(self, path=None, filename=None):
        pass
    #     """Restores DistrictScenario from dumped EnergySystem."""
    #     if path is None:
    #         path = os.path.join(
    #             os.path.expanduser("~"), ".q100opt", "dumps", "energysystem"
    #         )
    #
    #     if filename is None:
    #         filename = "ds_dump.oemof"
    #
    #     es_restore = solph.EnergySystem()
    #     es_restore.restore(dpath=path, filename=filename)
    #     logging.info(
    #         "Restoring EnergySystem will overwrite existing attributes."
    #     )

    def analyse_costs(self):
        """Performs a cost analysis."""
        self.results['cost_analysis'] = analyse_costs(
            results=self.results
        )


def load_district_scenario(path, filename):
    """Load a TableBuilder class."""
    return pickle.load(open(os.path.join(path, filename), "rb"))


class ParetoFront(DistrictScenario):
    """Class for calculation pareto fronts with costs and emission."""
    def __init__(self, emission_limits=None, number_of_points=2,
                 dist_type='linear',
                 off_set=1,
                 **kwargs):
        super().__init__(**kwargs)
        self.number = number_of_points
        self.dist_type = dist_type
        self.off_set = off_set
        self.solve_attr = {
            'solver': kwargs.get("solver", 'gurobi'),
            'tee': kwargs.get("tee", True)
        }
        if self.table_collection is not {}:
            self.table_collection_co2opt = co2_optimisation(
                self.table_collection
            )
        else:
            ValueError('Provide a table_collection!')
        self.ds_min_co2 = self._get_min_emission()
        self.ds_max_co2 = self._get_max_emssion()
        self.e_min = self.ds_min_co2.es.results['meta']['objective']
        self.e_max = self.ds_max_co2.es.results['emissions']
        if emission_limits is not None:
            self.emission_limits = emission_limits
        else:
            self.emission_limits = self._calc_emission_limits()
        self.district_scenarios = {}
        self.pareto_front = None

    def _get_min_emission(self):
        """Calculates the pareto point with minimum emission."""
        sc_co2opt = DistrictScenario(
            emission_limit=1000000000,
            table_collection=self.table_collection_co2opt,
            number_of_time_steps=self.number_of_time_steps,
            year=self.year,
        )
        sc_co2opt.solve(**self.solve_attr)
        return sc_co2opt

    def _get_max_emssion(self):
        sc_costopt = DistrictScenario(
            emission_limit=1000000000,
            table_collection=self.table_collection,
            number_of_time_steps=self.number_of_time_steps,
            year=self.year,
        )
        sc_costopt.solve(**self.solve_attr)
        return sc_costopt

    def _calc_emission_limits(self):
        """Calculates the emission limits of the pareto front."""
        if self.dist_type == 'linear':
            limits = []
            e_start = self.e_min + self.off_set
            interval = (self.e_max - e_start) / (self.number - 1)
            for i in range(self.number):
                limits.append(e_start + i * interval)
        else:
            raise ValueError(
                'No other method than "linear" for calculation the emission'
                ' limits implemented yet.'
            )
        return limits

    def _get_pareto_results(self):
        """Gets all cost an emission values of pareto front."""
        index = list(self.district_scenarios.keys())
        columns = ['costs', 'emissions']
        df_pareto = pd.DataFrame(index=index, columns=columns)
        for r, c in df_pareto.iterrows():
            df_pareto.at[r, 'costs'] = \
                self.district_scenarios[r].es.results['Costs']
            df_pareto.at[r, 'emissions'] = \
                self.district_scenarios[r].es.results['Emissions']
        return df_pareto

    def calc_pareto_front(self, dump_esys=False):
        """Calculates the Pareto front for all emission limits."""
        for e in self.emission_limits:
            e_str = str(int(round(e)))
            ds_name = self.name + '_' + e_str
            ds = DistrictScenario(
                name=ds_name,
                emission_limit=e,
                table_collection=self.table_collection,
                number_of_time_steps=self.number_of_time_steps,
                year=self.year,
            )
            ds.solve(**self.solve_attr)

            self.district_scenarios.update(
                {e_str: ds}
            )

            if dump_esys:

                esys_path = os.path.join(self.results_fn, self.name,
                                         "energy_system")
                if not os.path.isdir(esys_path):
                    os.mkdir(esys_path)

                ds.dump(path=esys_path, filename=e_str)

        self.pareto_front = self._get_pareto_results()

    def store_results(self, path=None, esys=False):
        """Store all results of pareto front."""
        if path is None:
            bpath = os.path.join(os.path.expanduser("~"), ".q100opt")
            if not os.path.isdir(bpath):
                os.mkdir(bpath)
            dpath = os.path.join(bpath, "dumps")
            if not os.path.isdir(dpath):
                os.mkdir(dpath)
            path = os.path.join(dpath, "pareto")
            if not os.path.isdir(path):
                os.mkdir(path)

        # dump energy systems
        esys_path = os.path.join(path, "energy_system")
        if not os.path.isdir(esys_path):
            os.mkdir(esys_path)

        if esys:
            for name, scenario in self.district_scenarios.items():
                scenario.es.dump(
                    dpath=esys_path,
                    filename=name,
                )
            logging.info("EnerySystems dumped to {0}".format(esys_path))

        # store table_collection
        tables_path = os.path.join(path, "input_tables")
        if not os.path.isdir(tables_path):
            os.mkdir(tables_path)

        for name, df in self.table_collection.items():
            name = name.replace(" ", "_") + ".csv"
            filename = os.path.join(tables_path, name)
            df.to_csv(filename)
        logging.info(
            "Scenario saved as csv-collection to {0}".format(tables_path))

        # store pareto results
        path_pareto = os.path.join(path, 'pareto_results.xlsx')
        self.pareto_front.to_excel(path_pareto)
        logging.info(
            "Pareto front table saved as xlsx to {0}".format(path_pareto))

    def restore_from_results(self, path):
        """Restores a Pareto front class from a result folder."""
        pass


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

    for _, l in tab.iterrows():

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
    list : oemof Sinks (non fix sources) objects.

    Note
    ----
    No investment possible.
    """
    sinks = []

    for _, cs in tab.iterrows():

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

        sinks_fix.append(
            solph.Sink(
                label=cs['label'],
                inputs={busd[cs['from']]: solph.Flow(
                    nominal_value=cs['nominal_value'],
                    fix=timeseries[cs['label'] + '.fix'].values
                )})
        )

    return sinks_fix


def add_storages(tab, busd):
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

        att = list(s.index)
        fa_list = [
            x.split('.')[1] for x in att if x.split('.')[0] == 'storage']

        sto_attr = {}

        for fa in fa_list:
            sto_attr[fa] = s['storage.' + fa]

        io = get_invest_obj(s)

        if io is not None:
            sto_attr['nominal_storage_capacity'] = None
            sto_attr['invest_relation_input_capacity'] = \
                s['invest_relation_input_capacity']
            sto_attr['invest_relation_output_capacity'] = \
                s['invest_relation_output_capacity']

        storages.append(
            solph.components.GenericStorage(
                label=s['label'],
                inputs={busd[s['bus']]: solph.Flow()},
                outputs={busd[s['bus']]: solph.Flow()},
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

        flow_out1_attr = get_flow_att(t, timeseries)

        io = get_invest_obj(t)

        if io is not None:
            flow_out1_attr['nominal_value'] = None

        d_in = {busd[t['in_1']]: solph.Flow()}

        d_out = {busd[t['out_1']]: solph.Flow(
            investment=io, **flow_out1_attr
        )}

        # check if timeseries in conversion factors and convert to float
        att = list(t.index)
        eff_list = [x for x in att if x.split('_')[0] == 'eff']
        d_eff = {}
        for eff in eff_list:
            if t[eff] == 'series':
                d_eff[eff] = timeseries[t['label'] + '.' + eff]
            else:
                d_eff[eff] = float(t[eff])

        cv = {busd[t['in_1']]: d_eff['eff_in_1'],
              busd[t['out_1']]: d_eff['eff_out_1']}

        # update inflows and conversion factors, if a second inflow bus label
        # is given
        if not (t['in_2'] == '0' or t['in_2'] == 0):
            d_in.update({busd[t['in_2']]: solph.Flow()})
            cv.update({busd[t['in_2']]: d_eff['eff_in_2']})

        # update outflows and conversion factors, if a second outflow bus label
        # is given
        if not (t['out_2'] == '0' or t['out_2'] == 0):
            d_out.update({busd[t['out_2']]: solph.Flow()})
            cv.update({busd[t['out_2']]: d_eff['eff_out_2']})

        transformer.append(
            solph.Transformer(
                label=t['label'],
                inputs=d_in,
                outputs=d_out,
                conversion_factors=cv
            )
        )

    return transformer


def co2_optimisation(d_data_origin):

    d_data = deepcopy(d_data_origin)

    # 1. variable_costs <--> emission_factor
    for _, tab in d_data.items():
        if ('flow.variable_costs' or 'flow.emission_factor') in tab.columns:
            var_costs = tab['flow.variable_costs'].copy()
            co2 = tab['flow.emission_factor'].copy()

            tab['flow.variable_costs'] = co2
            tab['flow.emission_factor'] = var_costs

        # check if there is excess, shortage is active
        for key in ['excess', 'shortage']:
            if key in tab.columns:
                if tab[key].sum() > 0:
                    ValueError('There is active excess/shortage. Please check'
                               'if there are no costs involved.')

    # 1b : Timeseries table
    for col_name in list(d_data['Timeseries'].columns):
        if 'variable_costs' in col_name:
            prefix = col_name.split('.')[0]
            cost_series = d_data['Timeseries'][col_name].copy()
            co2_series = \
                d_data['Timeseries'][prefix + '.emission_factor'].copy()
            d_data['Timeseries'][col_name] = co2_series
            d_data['Timeseries'][prefix + '.emission_factor'] = cost_series

    # 2. investment costs to zero
    for _, tab in d_data.items():
        if 'invest.ep_costs' in tab.columns:
            tab['invest.ep_costs'] = 0.0000001
        if 'invest.offet' in tab.columns:
            tab['invest.offset'] = 0

    return d_data


def calc_pareto_front(inputpath=None, scenario_name=None, outputpath=None,
                      number=2, dist_type='linear', off_set=1):

    # some base attributes
    es_attr = {
        'name': scenario_name,
        'year': 2018,
        'debug': False,
    }

    solve_attr = {
        'solver': 'gurobi',
        'with_duals': False,
        'tee': True,
    }

    table_collection_costopt = load_csv_data(path=inputpath + scenario_name)
    table_collection_co2opt = co2_optimisation(table_collection_costopt)

    # 1. get min emission value

    sc_co2opt = DistrictScenario(
        emission_limit=1000000000,
        table_collection=table_collection_co2opt,
        location=inputpath,
        **es_attr
    )
    sc_co2opt.solve(**solve_attr)
    sc_co2opt.dump_es(filename=outputpath + scenario_name + '/co2opt')
    e_min = sc_co2opt.es.results['meta']['objective']

    # 2. get max emission value

    sc_costopt = DistrictScenario(
        emission_limit=1000000000,
        table_collection=table_collection_costopt,
        location=inputpath,
        **es_attr
    )
    sc_costopt.solve(**solve_attr)
    sc_costopt.dump_es(filename=outputpath + scenario_name + '/costopt')
    e_max = sc_costopt.es.results['emissions']

    # 3. calc emission limits

    e_limits = []
    if dist_type == 'linear':
        e_start = e_min + off_set
        interval = (e_max - e_start) / (number - 1)
        for i in range(number):
            e_limits.append(e_start + i*interval)

    # 4. calc pareto front

    df_pareto = pd.DataFrame()
    for e in e_limits:
        sc_costopt.emission_limit = e
        sc_costopt.solve(**solve_attr)
        e_name = str(int(round(e)))
        sc_costopt.dump_es(
            filename=outputpath + scenario_name + '/' + e_name
        )
        costs = sc_costopt.es.results['costs']
        df_pareto = df_pareto.append(
            pd.DataFrame(
                [[e_name, e, costs]],
                columns=['limit_name', 'emissions', 'costs']
            ),
            ignore_index=True,
        )

    # 5. save results of pareto front to output path as excel

    df_pareto.to_excel(outputpath + scenario_name + '/pareto.xlsx')

    return df_pareto
