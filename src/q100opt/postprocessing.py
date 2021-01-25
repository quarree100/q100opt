from matplotlib import pyplot as plt
import oemof.solph as solph


def plot_invest_flows(results):
    """Plots all investment flows in a bar plot."""

    flows_invest = [x for x in results.keys() if x[1] is not None
                    if hasattr(results[x]['scalars'], 'invest')
                    if isinstance(x[0], solph.Transformer)]

    invest_val = [results[x]['scalars']['invest'] for x in flows_invest]
    invest_lab = [x[0].label for x in flows_invest]

    plt.bar(invest_lab, invest_val)
    plt.ylabel('Installed Flow Capacity [kW]')
    plt.xlabel('"From node" label (x[0].label of Flow)')
    plt.show()


def plot_invest_storages(results):
    """Plots investment storages as bar plot.s"""

    store_invest = [x for x in results.keys() if x[1] is None
                    if hasattr(results[x]['scalars'], 'invest')
                    if isinstance(x[0], solph.GenericStorage)]

    invest_val_s = [results[x]['scalars']['invest'] for x in store_invest]
    invest_lab_s = [x[0].label for x in store_invest]

    plt.bar(invest_lab_s, invest_val_s)
    plt.ylabel('Installed Storage Capacity [kWh]')
    plt.xlabel('Label of storage')
    plt.show()
