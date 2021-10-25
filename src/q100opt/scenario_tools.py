# -*- coding: utf-8 -*-

"""

This module holds Classes and Functions for solving linear optimisation
problems based on tabular data.

Please use this module with care. It is work in progress and properly
tested yet!

Contact: Johannes Röder <johannes.roeder@uni-bremen.de>

SPDX-License-Identifier: MIT

"""
import datetime
import logging
import os
import pickle
import warnings
from copy import deepcopy

import oemof.solph as solph
import pandas as pd

from .external import Scenario
from .plots import plot_es_graph
from .postprocessing import analyse_bus
from .postprocessing import analyse_costs
from .postprocessing import analyse_emissions
from .postprocessing import get_all_sequences
from .postprocessing import get_boundary_flows
from .postprocessing import get_trafo_flow
from .setup_model import add_buses
from .setup_model import add_links
from .setup_model import add_sinks
from .setup_model import add_sinks_fix
from .setup_model import add_sources
from .setup_model import add_sources_fix
from .setup_model import add_storages
from .setup_model import add_transformer
from .setup_model import check_active
from .setup_model import check_nonconvex_invest_type
from .setup_model import load_csv_data


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
            add_storages(nd['Storages'], busd, nd['Timeseries']) +
            add_transformer(nd['Transformer'], busd, nd['Timeseries'])
        )

        if 'Link' in nd.keys():
            nod.extend(add_links(nd['Link'], busd))

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

    def add_couple_invest_contr(self, couple_invest_flow):
        """
        Adds a solph.contraint for coupling investment flows.

        syntax of couple_invest_flow:

            couple_invest_flow={
                'flow1': ("label_from", "label_to"),
                'flow2': ("label_from", "label_to"),
            }

        Make sure, that these flows are InvestmentFlows.

        TODO : Include this methdo in additional constraints sheet.
        """
        flow1_from = self.es.groups[couple_invest_flow['flow1'][0]]
        flow1_to = self.es.groups[couple_invest_flow['flow1'][1]]
        investflow_1 = \
            self.model.InvestmentFlow.invest[flow1_from, flow1_to]

        flow2_from = self.es.groups[couple_invest_flow['flow2'][0]]
        flow2_to = self.es.groups[couple_invest_flow['flow2'][1]]
        investflow_2 = \
            self.model.InvestmentFlow.invest[flow2_from, flow2_to]

        solph.constraints.equate_variables(
            self.model,
            investflow_1,
            investflow_2,
            factor1=1,
            name="couple_investment_flows"
        )

    def add_addtional_constraints(self):
        """This method adds the additional constraints from the sheet
        `Additional_constraints` from the table_collection.

        At the moment, only the "Generic Investment Limit" is implemented.
        """
        for _, c in self.table_collection["Additional_constraints"].iterrows():

            if c["type"] == "additional_resource":

                self.model = \
                    solph.constraints.additional_investment_flow_limit(
                        self.model, c["keyword"], limit=c["limit"]
                    )
            else:
                raise NotImplementedError(
                    "This type of constraint %s does not exists, or is not "
                    "implemented yet.", c["type"]
                )

    def solve(self, with_duals=False, tee=True, logfile=None, solver=None,
              couple_invest_flow=None, **kwargs):

        if self.es is None:
            self.table2es()

        self.create_model()
        self.add_emission_constr()

        if "Additional_constraints" in self.table_collection.keys():
            self.add_addtional_constraints()

        if couple_invest_flow is not None:
            self.add_couple_invest_contr(couple_invest_flow)

        logging.info("Optimising using {0} ...".format(solver))

        if with_duals:
            self.model.receive_duals()

        if self.debug:
            filename = os.path.join(
                solph.helpers.extend_basic_path("lp_files"), "q100opt.lp"
            )
            self.model.write(
                filename, io_options={"symbolic_solver_labels": True}
            )
            logging.info("Store lp-file in {0}.".format(filename))

        solver_kwargs = {
            "cmdline_options": kwargs.get(
                "solver_cmdline_options", {})}

        self.model.solve(
            solver=solver, solve_kwargs={"tee": tee, "logfile": logfile},
            **solver_kwargs
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

    def plot(self, show=True):
        """Plots the energy system graph."""
        if self.es is None:
            self.table2es()

        plot_es_graph(self.es, show=show)

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
        """Dump results of District scenario."""
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

    def analyse_results(self, heat_bus_label='b_heat',
                        elec_bus_label='b_elec', label_end_energy=None):
        """Calls all analysis methods."""
        for label in [heat_bus_label, elec_bus_label]:
            check_label(self.results['main'], label)

        if label_end_energy is None:
            label_end_energy = ['demand_heat']

        for label in label_end_energy:
            check_label(self.results['main'], label)

        self.analyse_costs()
        self.analyse_emissions()
        self.analyse_kpi(label_end_energy=label_end_energy)
        self.analyse_sequences()
        self.results['sum'] = self.results['sequences'].sum()
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
            'emission_analysis']['sum']['emissions'].sum()
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

        if 'kpi' not in self.results.keys():
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

            kpi = pd.Series(kpi_dct)

            self.results['kpi'] = kpi

        else:
            kpi = self.results['kpi']

        return kpi

    def analyse_sequences(self):
        """..."""
        if 'sequences' not in self.results.keys():
            self.results['sequences'] = \
                get_all_sequences(self.results['main'])

            ind_length = len(self.results['timeindex'])
            df_param = self.results['table_collection']['Timeseries'].copy()
            df_param = df_param.iloc[:ind_length]

            if 'Unnamed: 0' in df_param.columns:
                df_param.drop(['Unnamed: 0'], axis=1, inplace=True)

            list_of_tuples = [
                ('parameter', x.split('.')[0], x.split('.')[1])
                for x in df_param.columns
            ]

            df_param.columns = pd.MultiIndex.from_tuples(list_of_tuples)
            df_param.index = self.results['timeindex']

            self.results['sequences'] = pd.concat(
                [self.results['sequences'], df_param], axis=1
            )

            logging.info("All sequences processed into one DataFrame.")

        return self.results['sequences']

    def analyse_boundary_flows(self):
        """
        Returns the sequences and sums of all sinks and sources.

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


def check_label(results, label):
    """..."""
    pass


class ParetoFront(DistrictScenario):
    """Class for calculation pareto fronts with costs and emission."""
    def __init__(self, emission_limits=None, number_of_points=2,
                 dist_type='linear', emission_limits_relative=None,
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

        self.emission_limits_relative = emission_limits_relative
        self.emission_limits = emission_limits

        self.district_scenarios = dict()
        self.pareto_front = None

        # ToDo: sort results District Scenarios
        # self.ordered_scenarios = [
        #     str(x) for x in sorted([int(x) for x in self.des.keys()],
        #                            reverse=True)
        # ]

    def _get_min_emission(self, **kwargs):
        """Calculates the pareto point with minimum emission."""
        logging.info("Calculate minimum emission limit ...")

        sc_co2opt = DistrictScenario(
            emission_limit=1000000000,
            table_collection=self.table_collection_co2opt,
            number_of_time_steps=self.number_of_time_steps,
            year=self.year,
        )
        sc_co2opt.solve(**kwargs)
        return sc_co2opt

    def _get_max_emssion(self, **kwargs):
        logging.info("Calculate maximum emission limit ...")

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
        elif self.dist_type == 'logarithmic':
            limits = []
            e_start = self.e_min + self.off_set
            lim_last = self.e_max
            limits.append(lim_last)
            for i in range(self.number-2):
                lim_last = lim_last - (lim_last - e_start) * 0.5
                limits.append(lim_last)
            limits.append(e_start)

        else:
            raise ValueError(
                'No other method than "linear" for calculation the emission'
                ' limits implemented yet.'
            )
        return limits

    def _get_abs_emission_limits(self, emission_limits):
        """
        Calculates the absolute emission limits from the relative
        limits in str format.
        """
        e_limits = [float(x) for x in emission_limits
                    if x != "zero"]

        e_limits = [x * (self.e_max - self.e_min) + self.e_min
                    for x in e_limits]

        if ("zero" in emission_limits) and (self.e_min * self.e_max < 0):
            e_null = [0.0]
        else:
            e_null = []

        if self.emission_limits is not None:
            self.emission_limits = self.emission_limits + e_limits + e_null
        else:
            self.emission_limits = e_limits + e_null

        self.emission_limits = list(set(self.emission_limits))

        return self.emission_limits

    def _get_pareto_results(self):
        """Gets all cost an emission values of pareto front."""
        index = list(self.district_scenarios.keys())
        columns = ['costs', 'emissions']
        df_pareto = pd.DataFrame(index=index, columns=columns)
        for r, _ in df_pareto.iterrows():
            df_pareto.at[r, 'costs'] = \
                self.district_scenarios[r].results['costs']
            df_pareto.at[r, 'emissions'] = \
                self.district_scenarios[r].results['emissions']
        return df_pareto

    def update_pareto_front(self, emission_limits, dump_esys=False, **kwargs):
        """Calculates additional points to the `ParetoFront`.

        This method updates the attribute `district_scenarios` and the
        `results` of the pareto front.

        Parameters
        ----------
        emission_limits : list
            Additional emission limits as list of string or floats,
            e.g. ["0.75"], that are fractions of the maximum emission limit.
        dump_esys : bool
            See :func:`~scenario_tools.ParetoFront.calc_pareto_front`.
        kwargs : dict
            See :func:`~scenario_tools.ParetoFront.calc_pareto_front`.
        """
        self.emission_limits = self._get_abs_emission_limits(emission_limits)

        self.calc_pareto_front(dump_esys=dump_esys, **kwargs)

    def calc_pareto_front(self, dump_esys=False, **kwargs):
        """
        Calculates the Pareto front for a given number of points, or
        for given emission limits.

        First, the cost-optimal and emission optimal solutions are calculated.
        Therefore, two optimisation runs are performed.
        For the emission optimisation, the table_collection is prepared by
        exchanging the `variable_cost` values and the `emission_factor` values.
        """
        if (self.e_min is None) or (self.e_max is None):
            if self.table_collection is not None:
                self.table_collection_co2opt = \
                    co2_optimisation(self.table_collection)
            else:
                ValueError('Provide a table_collection!')

            self.ds_min_co2 = self._get_min_emission(**kwargs)
            self.ds_max_co2 = self._get_max_emssion(**kwargs)
            self.e_min = self.ds_min_co2.results['meta']['objective']
            self.e_max = self.ds_max_co2.results['emissions']

            # set max emission scenario as point "1.00"
            self.district_scenarios["1.00"] = self.ds_max_co2
            self.district_scenarios["1.00"].emission_limit = self.e_max
            self.district_scenarios["1.00"].name = self.name + '_' + "1.00"

        if self.emission_limits is None:
            if self.emission_limits_relative is not None:
                self.emission_limits = self._get_abs_emission_limits(
                    emission_limits=self.emission_limits_relative
                )
            else:
                self.emission_limits = self._calc_emission_limits()

        for e in self.emission_limits:

            num = self.emission_limits.index(e) + 1
            logging.info(
                "Calculate emission limit {} of {} ...".format(
                    num, len(self.emission_limits)
                )
            )

            # Scenario name relative to emission range
            e_rel = (e - self.e_min) / (self.e_max - self.e_min)
            e_str = "{:.2f}".format(e_rel)
            # e_str = str(int(round(e)))

            if e_str not in self.district_scenarios.keys():
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

        # sort dictionary
        self.district_scenarios = {
            k: self.district_scenarios[k]
            for k in sorted(self.district_scenarios.keys())
        }

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
        """
        Dumps the results of the pareto front instance.

        The oemof.solph.EnergySystems and oemof.solph.Models of the
        q100opt.DistrictScenarios are removed before dumping, only the results
        are dumped.
        """
        # delete all oemof.solph.EnergySystems and oemof.solph.Models
        for _, v in self.__dict__.items():
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

    def analyse_results(self, heat_bus_label='b_heat',
                        elec_bus_label='b_elec',
                        label_end_energy=None):
        """Performs several analysis methods of the ParetoFront class.

        The processed results are stored within the attribute "results".

        Parameters
        ----------
        heat_bus_label : str
            Label of the (central) heat bus, the heat generation units are
            feeding.
        elec_bus_label : str
            Label of the electricity bus.
        label_end_energy : list
            List with labels of end energy flows. For the KPI calculation
            the absolute cost and emission values are related to the
            delivered end-energy by specific cost and emission values.
            As default, 'demand_heat' is used as end-energy flow.

        """
        for _, des in self.district_scenarios.items():
            des.analyse_results(
                heat_bus_label=heat_bus_label, elec_bus_label=elec_bus_label,
                label_end_energy=label_end_energy
            )

        self.analyse_costs(label_end_energy=label_end_energy)

        self.results['kpi'] = self.analyse_kpi(
            label_end_energy=label_end_energy
        )
        self.results['heat_generation'] = self.analyse_heat_generation_flows(
            heat_bus_label=heat_bus_label
        )
        self.results['sequences'] = self.analyse_sequences()
        self.results['sum'] = self.results['sequences'].sum().unstack(level=0)
        self.results['costs'] = self.get_all_costs()
        self.results['emissions'] = self.get_all_emissions()
        self.results['scalars'] = self.get_all_scalars()

    def analyse_kpi(self, label_end_energy=None):
        """
        Performs some postprocessing methods for all
        DistrictEnergySystems in the `ParetoFront`.

        Parameters
        ----------
        label_end_energy : list
            List with labels of end energy flows. For the KPI calculation
            the absolute cost and emission values are related to the
            delivered end-energy by specific cost and emission values.
            As default, 'demand_heat' is used as end-energy flow.

        Returns
        -------
        pandas.DataFrame
            Table with absolute and specific cost and emission values.

        """
        if label_end_energy is None:
            label_end_energy = ['demand_heat']

        d_kpi = {}
        for e_key, des in self.district_scenarios.items():
            d_kpi.update(
                {e_key: des.analyse_kpi(label_end_energy=label_end_energy)}
            )

        df_kpi = pd.concat(d_kpi, axis=1)

        self.results["kpi"] = df_kpi

        return df_kpi

    def analyse_costs(self, label_end_energy=None):
        """
        Gives a cost summary for all
        DistrictEnergySystems in the `ParetoFront`.

        Parameters
        ----------
        label_end_energy : list
            List with labels of end energy flows. For the KPI calculation
            the absolute cost and emission values are related to the
            delivered end-energy by specific cost and emission values.
            As default, 'demand_heat' is used as end-energy flow.

        Returns
        -------
        pandas.DataFrame
            Table with absolute and specific cost and emission values.

        """
        d_costs = {}
        for e_key, des in self.district_scenarios.items():
            d_costs.update(
                {e_key: des.analyse_costs()}
            )

        self.results['cost_analysis'] = d_costs

        return d_costs

    def get_all_costs(self):
        """
        Puts all cost analysis of the individual DistrictScenarios into
        one Multi-index DataFrame.
        """
        d_costs = {}
        for e_key, des in self.district_scenarios.items():
            d_costs.update(
                {e_key: des.results["cost_analysis"]["all"].stack()}
            )

        df_costs = pd.concat(d_costs, names=['emission_limit'])

        return df_costs.unstack(level=0).T

    def get_all_emissions(self):
        """
        Puts all emissions analyses of the individual DistrictScenarios into
        one Multi-index DataFrame.
        """
        d_emissions = {}
        for e_key, des in self.district_scenarios.items():
            d_emissions.update(
                {e_key: pd.concat(
                    {'emission': des.results["emission_analysis"]["sum"]}
                ).stack()}
            )

        df_emissions = pd.concat(d_emissions, names=['emission_limit'])

        df_emissions = df_emissions.unstack(level=0).T

        return df_emissions

    def get_all_scalars(self):
        """Puts all scalar results of each scenario in one DataFrame."""
        df_kpi = self.results['kpi'].T

        df_mi = pd.DataFrame(df_kpi.columns, columns=['value'])
        df_mi['category'] = None
        df_mi['label'] = None
        df_mi['type'] = 'kpi'

        mi = pd.MultiIndex.from_frame(
            df_mi[['type', 'category', 'label', 'value']]
        )

        df_kpi.columns = mi
        df_sum = pd.concat([self.results['sum']],
                           keys=['sum'], names=[None]).T

        df_scalars = pd.concat([
            df_kpi,
            self.results['costs'],
            self.results['emissions'],
            df_sum,
        ], axis=1)

        return df_scalars

    def analyse_heat_generation_flows(self, heat_bus_label='b_heat'):
        """..."""
        # def _get_keys():
        #     l_keys = []
        #     for res_key, res in v.results.items():
        #         if isinstance(res, dict):
        #             if 'sum' in res.keys():
        #                 l_keys.append(res_key)
        #     return l_keys
        #
        # d_flow_results = {}
        # for k, v in self.district_scenarios.items():
        #     keys = _get_keys()
        #     for re_k in keys:
        #         d_flow_results.update(
        #             {re_k: {k: v.results[re_k]['sum']}}
        #         )
        d_hg = {}
        for k, v in self.district_scenarios.items():
            d_hg.update({
                k: v.results['heat_generation']['sum']
            })
        df_heat_gen = pd.concat(d_hg, axis=1)
        return df_heat_gen

    def analyse_sequences(self):
        """..."""
        d_seq = {}
        for e_key, des in self.district_scenarios.items():
            d_seq.update(
                {e_key: des.results["sequences"]}
            )

        df_seq = pd.concat(d_seq.values(), axis=1, keys=d_seq.keys())

        return df_seq

    def plot(self, show=True):
        """Plots the energy system graph."""
        des = next(iter(self.district_scenarios.values()))
        des.plot(show=show)

    def export_results_to_xlsx(self, filename, override=True,
                               include_sequences=True):
        """Exports all results of paretor front to a excel file."""
        def _export_results():
            with pd.ExcelWriter(filename) as writer:
                r = self.results
                r["kpi"].to_excel(writer, sheet_name="kpi")
                r["costs"].T.to_excel(writer, sheet_name="costs")
                r["emissions"].T.to_excel(writer, sheet_name="emissions")
                r["sum"].to_excel(writer, sheet_name="sum")
                if include_sequences:
                    r["sequences"].to_excel(writer, sheet_name="sequences")
                logging.info(
                    "Results exported to : %s" % filename
                )

        filename = filename + ".xlsx"

        if os.path.isfile(filename):
            if override:
                logging.info(
                    "Results already exist and are overwritten"
                    " : %s" % filename
                )
                _export_results()
            else:
                logging.info(
                    "Results already exist and are not overwritten"
                    " : %s" % filename
                )
        else:
            _export_results()


def load_pareto_front(path, filename):
    """Load a ParetoFront class."""
    pf_restore = ParetoFront()

    pf_restore.__dict__ = \
        pickle.load(open(os.path.join(path, filename), "rb"))

    return pf_restore


def calc_pareto_front(inputpath=None, scenario_name=None, outputpath=None,
                      number=2, dist_type='linear', off_set=1):
    """
    TODO: Maybe remove function, it is not used at the moment.
    """

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
    """
    Takes a table collection and exchanges the flow values for
    `variable_costs` and `emission_factor`.

    Parameters
    ----------
    d_data_origin : dict
        original table_collection for cost optimisation.

    Returns
    -------
    dict : table_collection for emission optimisation.
    """

    d_data = deepcopy(d_data_origin)

    # 1. variable_costs <--> emission_factor
    for key, tab in d_data.items():

        var_cost_col = [
            x for x in tab.columns
            if 'variable_costs' in x
        ]

        var_emission_col = [
            x for x in tab.columns
            if 'emission_factor' in x
        ]

        prefix = set([x.split('.')[0] for x in var_cost_col] +
                     [x.split('.')[0] for x in var_emission_col])

        for pre in prefix:
            tab.rename(
                columns={pre + '.emission_factor': pre + '.variable_costs',
                         pre + '.variable_costs': pre + '.emission_factor',
                         }, inplace=True)

        # check if there is excess, shortage is active
        for key in ['excess', 'shortage']:
            if key in tab.columns:
                if tab[key].sum() > 0:
                    warnings.warn(
                        'Convert to CO2opt tables:'
                        ' There is active excess/shortage. Please check'
                        ' if there are no costs involved.'
                    )

    # 2. investment costs to zero
    for _, tab in d_data.items():
        # nonconvex investment with offset and minimum > then, the investment
        #  can be also 0, and it is no "hard" minimum.
        if 'invest.minimum' in tab.columns:
            if 'invest.offset' in tab.columns:
                for r, c in tab.iterrows():
                    if c['invest.offset'] > 0:
                        tab.at[r, 'invest.minimum'] = 0

        if 'invest.ep_costs' in tab.columns:
            tab['invest.ep_costs'] = 0.0000001
        if 'invest.offset' in tab.columns:
            tab['invest.offset'] = 0

    return d_data
