# -*- coding: utf-8 -*-

"""Scenario class from deflex.

See:
https://github.com/reegis/deflex/blob/master/deflex/scenario_tools.py

SPDX-FileCopyrightText: 2016-2019 Uwe Krien <krien@uni-bremen.de>

SPDX-License-Identifier: MIT
"""
__copyright__ = "Uwe Krien <krien@uni-bremen.de>"
__license__ = "MIT"

# Python libraries
import os
import calendar
import datetime
import shutil
import dill as pickle
import logging

# External libraries
import pandas as pd

# oemof libraries
from oemof import solph


class Scenario:
    """Scenario class."""

    def __init__(self, **kwargs):
        self.name = kwargs.get("name", "unnamed_scenario")
        self.table_collection = kwargs.get("table_collection", {})
        self.year = kwargs.get("year", None)
        self.ignore_errors = kwargs.get("ignore_errors", False)
        self.round_values = kwargs.get("round_values", 0)
        self.model = kwargs.get("model", None)
        self.es = kwargs.get("es", None)
        self.results = kwargs.get("results", None)
        self.results_fn = kwargs.get("results_fn", None)
        self.debug = kwargs.get("debug", None)
        self.location = None
        self.map = None
        self.meta = kwargs.get("meta", None)

    def initialise_energy_system(self):
        """
        Returns
        -------
        """
        if self.debug is True:
            number_of_time_steps = 3
        else:
            try:
                if calendar.isleap(self.year):
                    number_of_time_steps = 8784
                else:
                    number_of_time_steps = 8760
            except TypeError:
                msg = (
                    "You cannot create an EnergySystem with self.year={0}, "
                    "of type {1}."
                )
                raise TypeError(msg.format(self.year, type(self.year)))

        date_time_index = pd.date_range(
            "1/1/{0}".format(self.year), periods=number_of_time_steps, freq="H"
        )
        return solph.EnergySystem(timeindex=date_time_index)

    def load_excel(self, filename=None):
        pass

    def load_csv(self, path=None):
        pass

    def to_excel(self, filename):
        """Dump scenario into an excel-file."""
        # create path if it does not exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        writer = pd.ExcelWriter(filename)
        for name, df in sorted(self.table_collection.items()):
            df.to_excel(writer, name)
        writer.save()
        logging.info("Scenario saved as excel file to {0}".format(filename))

    def to_csv(self, path):
        """Dump scenario into a csv-collection."""
        if os.path.isdir(path):
            shutil.rmtree(os.path.join(path))
        os.makedirs(path)

        for name, df in self.table_collection.items():
            name = name.replace(" ", "_") + ".csv"
            filename = os.path.join(path, name)
            df.to_csv(filename)
        logging.info("Scenario saved as csv-collection to {0}".format(path))

    def check_table(self, table_name):
        """
        Parameters
        ----------
        table_name
        Returns
        -------
        """
        if self.table_collection[table_name].isnull().values.any():
            c = []
            for column in self.table_collection[table_name].columns:
                if self.table_collection[table_name][column].isnull().any():
                    c.append(column)
            msg = "Nan Values in the {0} table (columns: {1})."
            raise ValueError(msg.format(table_name, c))
        return self

    def create_nodes(self):
        """
        Returns
        -------
        dict
        """
        pass

    def initialise_es(self, year=None):
        """
        Parameters
        ----------
        year
        Returns
        -------
        """
        if year is not None:
            self.year = year
        self.es = self.initialise_energy_system()
        return self

    def add_nodes(self, nodes):
        """
        Parameters
        ----------
        nodes : dict
            Dictionary with a unique key and values of type oemof.network.Node.
        Returns
        -------
        self
        """
        if self.es is None:
            self.initialise_es()
        self.es.add(*nodes.values())
        return self

    def table2es(self):
        """
        Returns
        -------
        """
        if self.es is None:
            self.es = self.initialise_energy_system()
        nodes = self.create_nodes()
        self.es.add(*nodes.values())
        return self

    def create_model(self):
        """
        Returns
        -------
        """
        self.model = solph.Model(self.es)
        return self

    def dump_es(self, filename):
        """
        Parameters
        ----------
        filename
        Returns
        -------
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        f = open(filename, "wb")
        if self.meta is None:
            if self.es.results is not None and "Meta" in self.es.results:
                self.meta = self.es.results["meta"]
        pickle.dump(self.meta, f)
        pickle.dump(self.es.__dict__, f)
        f.close()
        logging.info("Results dumped to {0}.".format(filename))

    def restore_es(self, filename=None):
        """
        Parameters
        ----------
        filename
        Returns
        -------
        """
        if filename is None:
            filename = self.results_fn
        else:
            self.results_fn = filename
        if self.es is None:
            self.es = solph.EnergySystem()
        f = open(filename, "rb")
        self.meta = pickle.load(f)
        self.es.__dict__ = pickle.load(f)
        f.close()
        self.results = self.es.results["main"]
        logging.info("Results restored from {0}.".format(filename))

    def scenario_info(self, solver_name):
        pass

    def solve(self, with_duals=False, tee=True, logfile=None, solver=None):
        """
        Parameters
        ----------
        with_duals
        tee
        logfile
        solver
        Returns
        -------
        """
        logging.info("Optimising using {0}.".format(solver))

        if with_duals:
            self.model.receive_duals()

        if self.debug:
            filename = os.path.join(
                solph.helpers.extend_basic_path("lp_files"), "reegis.lp"
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
        self.es.results["meta"]["in_location"] = self.location
        self.es.results["meta"]["file_date"] = datetime.datetime.fromtimestamp(
            os.path.getmtime(self.location)
        )
        self.es.results["meta"]["solph_version"] = solph.__version__
        self.results = self.es.results["main"]

    def plot_nodes(self, show=None, filename=None, **kwargs):
        pass
