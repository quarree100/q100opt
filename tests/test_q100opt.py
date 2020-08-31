
import os

import oemof.solph as solph
import pandas as pd

from q100opt.cli import main
from q100opt.setup_model import add_buses
from q100opt.setup_model import add_sinks
from q100opt.setup_model import add_sinks_fix
from q100opt.setup_model import add_sources
from q100opt.setup_model import add_sources_fix
from q100opt.setup_model import add_storages
from q100opt.setup_model import add_transformer
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


def test_add_source_fix():
    tab = pd.DataFrame(
        [['label_1', 'b_1', 12, 0, 66, 23],
         ['label_2', 'b_1', 56, 1, 15, 34]],
        columns=['label', 'to', 'flow.variable_costs', 'investment',
                 'invest.ep_costs', 'flow.nominal_value'])
    b1 = solph.Bus(label='b_1')
    d = {'b_1': b1}
    ts = pd.DataFrame([[6, 8], [3, 4]], columns=['label_1.fix', 'label_2.fix'])
    sources = add_sources_fix(tab, d, ts)
    assert sources[0].outputs[b1].nominal_value == 23
    assert sources[0].outputs[b1].fix.sum() == 9
    assert sources[1].outputs[b1].fix.sum() == 12
    assert hasattr(sources[1].outputs[b1], 'investment')


def test_add_sinks():
    tab = pd.DataFrame(
        [['label_1', 'b_1', 1], ['label_2', 'b_1', 56]],
        columns=['label', 'from', 'flow.variable_costs'])
    b1 = solph.Bus(label='b_1')
    d = {'b_1': b1}
    sinks = add_sinks(tab, d)
    inflow = sinks[0].inputs[b1]
    assert (len(sinks) == 2) and (hasattr(inflow, 'variable_costs'))


def test_add_sinks_fix():
    tab = pd.DataFrame(
        [['elec', 'b_1', 1], ['heat', 'b_1', 56]],
        columns=['label', 'from', 'nominal_value'])
    ts = pd.DataFrame([[9, 2], [9, 2]], columns=['heat.fix', 'elec.fix'])
    b1 = solph.Bus(label='b_1')
    d = {'b_1': b1}
    sinks_fix = add_sinks_fix(tab, d, ts)
    assert (sinks_fix[0].inputs[b1].fix.sum() == 4) and \
           (sinks_fix[1].inputs[b1].fix.sum() == 18) and \
           (isinstance(sinks_fix[0], solph.Sink))


def test_add_storages():
    tab = pd.DataFrame(
        [['S1', 1, 'b1', 45, 3, 0.2, 0.3],
         ['S2', 0, 'b1', 150, 2, 0.3, 0.2]],
        columns=['label', 'investment', 'bus',
                 'storage.nominal_storage_capacity', 'invest.ep_costs',
                 'invest_relation_input_capacity',
                 'invest_relation_output_capacity']
    )
    b1 = solph.Bus(label='b1')
    storages = add_storages(tab, {'b1': b1})
    assert (storages[0].nominal_storage_capacity is None) and \
           (hasattr(storages[1], 'investment'))


def test_add_trafo_no_invest_series():
    tab = pd.DataFrame(
        [['trafo_1', 'b0', '0', 'b1', '0', 1.2, 'series', 230]],
        columns=['label', 'in_1', 'in_2', 'out_1', 'out_2', 'eff_in_1',
                 'eff_out_1', 'flow.nominal_value']
    )
    timeseries = pd.DataFrame(
        [0.3, 1.2, 0.7], columns=['trafo_1.eff_out_1']
    )
    b0 = solph.Bus(label='b0')
    b1 = solph.Bus(label='b1')
    transformer = add_transformer(tab, {'b0': b0, 'b1': b1}, timeseries)
    assert (transformer[0].outputs[b1].nominal_value == 230.0)


def test_add_trafo_2in_2out():
    tab = pd.DataFrame(
        [['trafo_1', 'b0', 'b2', 'b1', 'b3', 1.2, 0.1, 'series', 2, 230]],
        columns=['label', 'in_1', 'in_2', 'out_1', 'out_2', 'eff_in_1',
                 'eff_in_2', 'eff_out_1', 'eff_out_2', 'flow.nominal_value']
    )
    timeseries = pd.DataFrame(
        [0.3, 1.2, 0.7], columns=['trafo_1.eff_out_1']
    )
    b0 = solph.Bus(label='b0')
    b1 = solph.Bus(label='b1')
    b2 = solph.Bus(label='b2')
    b3 = solph.Bus(label='b3')
    transformer = add_transformer(
        tab, {'b0': b0, 'b1': b1, 'b2': b2, 'b3': b3}, timeseries)
    assert len(transformer[0].outputs) == 2 and \
           len(transformer[0].inputs) == 2


def test_add_trafo_invest():
    tab = pd.DataFrame(
        [['trafo_1', 'b0', '0', 'b1', '0', 1.2, 'series', 230, 1]],
        columns=['label', 'in_1', 'in_2', 'out_1', 'out_2', 'eff_in_1',
                 'eff_out_1', 'flow.nominal_value', 'investment']
    )
    timeseries = pd.DataFrame(
        [0.3, 1.2, 0.7], columns=['trafo_1.eff_out_1']
    )
    b0 = solph.Bus(label='b0')
    b1 = solph.Bus(label='b1')
    transformer = add_transformer(tab, {'b0': b0, 'b1': b1}, timeseries)
    assert (transformer[0].outputs[b1].nominal_value is None and
            hasattr(transformer[0].outputs[b1], 'investment'))
