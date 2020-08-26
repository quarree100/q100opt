# -*- coding: utf-8 -*-

"""
Function for reading predefined data and setting up an oemof-solph EnergySystem.

SPDX-License-Identifier: MIT

"""
import os

import pandas as pd


def load_csv_data(path):

    dict = {}

    for name in os.listdir(path):

        key = name.split('.csv')[0]
        val = pd.read_csv(os.path.join(path, name))
        dict.update([(key, val)])

    return dict
