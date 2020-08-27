
import os

import oemof.solph as solph
import pandas as pd

from q100opt.cli import main
from q100opt.setup_model import add_buses
from q100opt.setup_model import check_active
from q100opt.setup_model import load_csv_data

basedir = os.path.dirname(__file__)


def test_main():
    assert main([]) == 0


def test_import_csv():
    dir_import = os.path.join(basedir, '_files/import_csv')
    dic = load_csv_data(dir_import)
    assert dic['Source']['label'][0] == 'heat'


def test_check_active():
    d = {'Something': pd.DataFrame([[4, 1], [3, 0]], columns=['A', 'active'])}
    d = check_active(d)
    assert ((len(d['Something'].index) == 1) and
            ('active' not in d['Something'].columns))


def test_no_active():
    d = {'Something': pd.DataFrame([[4, 1], [3, 0]], columns=['A', 'B'])}
    d_new = check_active(d)
    assert (d == d_new)


def test_add_bus():
    data_bus = pd.DataFrame(
        [['label_1', 0, 0, 0, 0], ['label_2', 0, 0, 0, 0]],
        columns=['label', 'excess', 'shortage', 'shortage_costs',
                 'excess_costs'])
    n, b = add_buses(data_bus)
    assert ((len(n) == 2) and (isinstance(b['label_1'], solph.Bus)))


def test_add_bus_ex_short():
    data_bus = pd.DataFrame(
        [['label_1', 1, 1, 0, 0], ['label_2', 0, 0, 0, 0]],
        columns=['label', 'excess', 'shortage', 'shortage_costs',
                 'excess_costs'])
    n, b = add_buses(data_bus)
    assert ((len(n) == 4) and
            ([x.label for x in n if isinstance(x, solph.Sink)][0] ==
             'label_1_excess') and
            (isinstance(b['label_1'], solph.Bus)))
