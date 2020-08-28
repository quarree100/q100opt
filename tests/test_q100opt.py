
import os

import oemof.solph as solph
import pandas as pd

from q100opt.cli import main
from q100opt.setup_model import add_buses
from q100opt.setup_model import add_sources
from q100opt.setup_model import check_active
from q100opt.setup_model import get_invest_obj
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


def test_get_invest_1():
    series = pd.Series([1, 1, 1], index=['A', 'B', 'C'])
    assert get_invest_obj(series) is None


def test_get_invest_2():
    series = pd.Series([0, 1, 1], index=['investment', 'B', 'C'])
    assert get_invest_obj(series) is None


def test_get_invest_3():
    series = pd.Series([1, 5, 1], index=['investment', 'invest.ep_costs', 'C'])
    io = get_invest_obj(series)
    assert io.ep_costs == 5


def test_add_source_1():
    tab = pd.DataFrame(
        [['label_1', 'b_1', 1], ['label_2', 'b_2', 56]],
        columns=['label', 'to', 'flow.variable_costs'])
    b1 = solph.Bus(label='b_1')
    b2 = solph.Bus(label='b_2')
    d = {'b_1': b1, 'b_2': b2}
    sources = add_sources(tab, d)
    assert sources[1].label == 'label_2'


def test_add_source_ts():
    tab = pd.DataFrame(
        [['label_1', 'b_1', 'series'], ['label_2', 'b_2', 56]],
        columns=['label', 'to', 'flow.variable_costs'])
    b1 = solph.Bus(label='b_1')
    b2 = solph.Bus(label='b_2')
    d = {'b_1': b1, 'b_2': b2}
    ts = pd.DataFrame([6, 8, 7], columns=['label_1.variable_costs'])
    sources = add_sources(tab, d, ts)
    assert sources[0].outputs[b1].variable_costs.sum() == 21


def test_add_source_invest():
    tab = pd.DataFrame(
        [['label_1', 'b_1', 1, 1], ['label_2', 'b_2', 56, 0]],
        columns=['label', 'to', 'flow.variable_costs', 'investment'])
    b1 = solph.Bus(label='b_1')
    b2 = solph.Bus(label='b_2')
    d = {'b_1': b1, 'b_2': b2}
    sources = add_sources(tab, d)
    assert hasattr(sources[0].outputs[b1], 'investment')
