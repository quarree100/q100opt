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

from .external import Scenario
from .postprocessing import analyse_costs, analyse_emissions, \
    get_boundary_flows, get_trafo_flow, \
    analyse_bus
from .setup_model import load_csv_data, check_active,\
    check_nonconvex_invest_type, add_transformer, add_storages, add_sinks, \
    add_sources, add_sources_fix, add_buses, add_sinks_fix


class DistrictScenario(Scenario):

    """Scenario class for urban energy systems"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_path = kwargs.get("input_path", None)
        self.emission_limit = kwargs.get("emission_limit", None)
        self.location = kwargs.get("location", None)
        self.number_of_time_steps = \
            kwargs.get("number_of_time_steps", 10)
        self.results = dict()

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

        # store directly at district energy system
        self.results["main"] = solph.processing.results(self.model)
        self.results["meta"] = solph.processing.meta_results(self.model)
        self.results["param"] = solph.processing.parameter_as_dict(self.es)
        self.results["meta"]["scenario"] = self.scenario_info(solver)
        if self.location is not None:
            self.results["meta"]["in_location"] = self.location
        self.results['meta']["datetime"] = datetime.datetime.now()
        self.results["meta"]["solph_version"] = solph.__version__
        self.results['meta']['emission_limit'] = self.emission_limit
        self.results['meta']['solver']['solver'] = solver
        self.results['costs'] = self.model.objective()
        self.results['table_collection'] = self.table_collection
        if hasattr(self.model, 'integral_limit_emission_factor'):
            self.results['emissions'] = \
                self.model.integral_limit_emission_factor()
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

        dump_des = deepcopy(self)

        if dump_des.model is not None:
            setattr(dump_des, 'model', None)
        if dump_des.es is not None:
            setattr(dump_des, 'es', None)

        pickle.dump(
            dump_des.__dict__, open(os.path.join(path, filename), "wb")
        )

        logging.info("DistrictScenario dumped"
                     " to {} as {}".format(path, filename))

    def restore(self, path=None, filename=None):
        """Restores a district energy system from dump."""
        self.__dict__ = load_district_scenario(path, filename).__dict__
        logging.info("DistrictEnergySystem restored.")

    def analyse_results(self, heat_bus_label='b_heat_gen',
                        elec_bus_label='b_el_ez'):
        """Calls all analysis methods."""
        self.analyse_costs()
        self.analyse_emissions()
        self.analyse_kpi()
        self.analyse_boundary_flows()
        self.analyse_heat_generation_flows(heat_bus_label=heat_bus_label)
        self.analyse_heat_bus(heat_bus_label=heat_bus_label)
        self.analyse_electricity_bus(elec_bus_label=elec_bus_label)

    def analyse_costs(self):
        """Performs a cost analysis."""
        if 'cost_analysis' not in self.results.keys():
            self.results['cost_analysis'] = analyse_costs(
                results=self.results
            )

            logging.info("Economic analysis completed.")

        # check if objective and recalculation match
        total_costs = self.results['cost_analysis']['all']['costs'].sum()
        objective_value = self.results['meta']['objective']

        if abs(total_costs - objective_value) > 0.01:
            raise ValueError(
                "The objective value and the re-calculated costs do not match!"
            )
        else:
            logging.info(
                "Check passed: Objective value and recalculated costs match."
            )

        return self.results['cost_analysis']

    def analyse_emissions(self):
        """Performs a summary of emissions of the energy system."""
        if 'emission_analysis' not in self.results.keys():
            self.results['emission_analysis'] = analyse_emissions(
                results=self.results
            )

            logging.info("Emission analysis completed.")

        # check if constraint and recalculation match
        total_em = self.results[
            'emission_analysis']['summary']['emissions'].sum()
        emission_value = self.results['emissions']

        if abs(total_em - emission_value) > 0.01:
            raise ValueError(
                "The constraint emission value and the re-calculated emissions"
                " do not match!"
            )
        else:
            logging.info(
                "Check passed: Constraint emission value and recalculated"
                " emission match."
            )

        return self.results['emission_analysis']

    def analyse_kpi(self, label_end_energy=None):
        """Description."""
        if label_end_energy is None:
            label_end_energy = ['demand_heat']

        costs = self.results['meta']['objective']

        emissions = self.results['emissions']

        end_energy = sum([
            solph.views.node(
                self.results['main'], x)["sequences"].values.sum()
            for x in label_end_energy])

        kpi_dct = {
            'absolute costs [€/a]': costs,
            'absolute emission [kg/a]': emissions,
            'end_energy [kWh/a]': end_energy,
            'specific costs [€/kWh]': costs/end_energy,
            'specific emission [kg/kWh]': emissions/end_energy,
        }

        df_kpi = pd.DataFrame.from_dict(kpi_dct, orient='index')

        self.results['kpi'] = df_kpi

        return df_kpi

    def analyse_boundary_flows(self):
        """Returns the sequences and sums of all sinks and sources.

        See postprocessing.get_boundary_flows!
        """
        if 'boundary_flows' not in self.results.keys():
            self.results['boundary_flows'] = \
                get_boundary_flows(self.results['main'])

            logging.info("Boundary flows analysis completed.")

        return self.results['boundary_flows']

    def analyse_heat_generation_flows(self, heat_bus_label='b_heat'):
        """Gets all heat generation flows."""
        if 'heat_generation' not in self.results.keys():
            self.results['heat_generation'] = \
                get_trafo_flow(self.results['main'], label_bus=heat_bus_label)

            logging.info("Heat generation flow analysis completed.")

        return self.results['heat_generation']

    def analyse_heat_bus(self, heat_bus_label='b_heat'):
        """..."""
        if 'heat_bus' not in self.results.keys():
            self.results['heat_bus'] = \
                analyse_bus(self.results['main'], bus_label=heat_bus_label)

            logging.info("Heat bus analysed.")

        return self.results['heat_bus']

    def analyse_electricity_bus(self, elec_bus_label='b_elec'):
        """..."""
        if 'electricity_bus' not in self.results.keys():
            self.results['electricity_bus'] = \
                analyse_bus(self.results['main'], bus_label=elec_bus_label)

            logging.info("Electricity bus analysed.")

        return self.results['electricity_bus']


def load_district_scenario(path, filename):
    """Load a DistrictScenario class."""
    des_restore = DistrictScenario()

    des_restore.__dict__ = \
        pickle.load(open(os.path.join(path, filename), "rb"))

    return des_restore


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
        self.table_collection_co2opt = None
        self.ds_min_co2 = None
        self.ds_max_co2 = None
        self.e_min = None
        self.e_max = None
        self.emission_limits = emission_limits
        self.district_scenarios = dict()
        self.pareto_front = None

    def _get_min_emission(self, **kwargs):
        """Calculates the pareto point with minimum emission."""
        sc_co2opt = DistrictScenario(
            emission_limit=1000000000,
            table_collection=self.table_collection_co2opt,
            number_of_time_steps=self.number_of_time_steps,
            year=self.year,
        )
        sc_co2opt.solve(**kwargs)
        return sc_co2opt

    def _get_max_emssion(self, **kwargs):
        sc_costopt = DistrictScenario(
            emission_limit=1000000000,
            table_collection=self.table_collection,
            number_of_time_steps=self.number_of_time_steps,
            year=self.year,
        )
        sc_costopt.solve(**kwargs)
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
                self.district_scenarios[r].results['costs']
            df_pareto.at[r, 'emissions'] = \
                self.district_scenarios[r].results['emissions']
        return df_pareto

    def calc_pareto_front(self, dump_esys=False, **kwargs):
        """Calculates the Pareto front for all emission limits."""
        if self.table_collection is not None:
            self.table_collection_co2opt = \
                co2_optimisation(self.table_collection)
        else:
            ValueError('Provide a table_collection!')

        self.ds_min_co2 = self._get_min_emission(**kwargs)
        self.ds_max_co2 = self._get_max_emssion(**kwargs)
        self.e_min = self.ds_min_co2.results['meta']['objective']
        self.e_max = self.ds_max_co2.results['emissions']

        if self.emission_limits is None:
            self.emission_limits = self._calc_emission_limits()

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
            ds.solve(**kwargs)

            self.district_scenarios.update(
                {e_str: ds}
            )

            if dump_esys:
                esys_path = os.path.join(self.results_fn, self.name,
                                         "energy_system")
                if not os.path.isdir(esys_path):
                    os.mkdir(esys_path)

                ds.dump(path=esys_path, filename=e_str + '_dump.des')

        self.results['pareto_front'] = self._get_pareto_results()

    def store_results(self, path=None):
        """
        Store main results and input table of pareto front in a not python
        readable way (.xlsx / .csv).
        """
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
        self.results['pareto_front'].to_excel(path_pareto)
        logging.info(
            "Pareto front table saved as xlsx to {0}".format(path_pareto))

    def dump(self, path=None, filename=None):
        """Dumps the pareto front instance."""

        # delete all oemof.solph.EnergySystems and oemof.solph.Models
        for k, v in self.__dict__.items():
            if hasattr(v, 'es') or hasattr(v, 'model'):
                setattr(v, 'es', None)
                setattr(v, 'model', None)

        for _, des in self.district_scenarios.items():
            setattr(des, 'es', None)
            setattr(des, 'model', None)

        pickle.dump(
            self.__dict__, open(os.path.join(path, filename), "wb")
        )

        logging.info(
            "ParetoFront dumped to {} as {}".format(path, filename)
        )

    def restore(self, path=None, filename=None):
        """Restores a district energy system from dump."""
        self.__dict__ = load_pareto_front(path, filename).__dict__
        logging.info("DistrictEnergySystem restored.")

    def analyse_kpi(self, label_end_energy=None):
        """Performs some postprocessing methods for all DistrictEnergySystems.
        """
        if label_end_energy is None:
            label_end_energy = ['demand_heat']

        d_kpi = {}
        for e_key, des in self.district_scenarios.items():
            d_kpi.update(
                {e_key: des.analyse_kpi(label_end_energy=label_end_energy)}
            )

        df_kpi = pd.concat(d_kpi, axis=1)

        return df_kpi


def load_pareto_front(path, filename):
    """Load a ParetoFront class."""
    pf_restore = ParetoFront()

    pf_restore.__dict__ = \
        pickle.load(open(os.path.join(path, filename), "rb"))

    return pf_restore


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
