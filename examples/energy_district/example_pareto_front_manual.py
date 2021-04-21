from matplotlib import pyplot as plt

from q100opt.scenario_tools import ParetoFront
from q100opt.setup_model import load_csv_data

table_collection = load_csv_data('data')

pf = ParetoFront(
    table_collection=table_collection,
    emission_limits=[200, 300, 500, 1000, 2000]
)

pf.calc_pareto_front(solver='cbc', tee=True)

pf.results["pareto_front"].plot(x='emissions', y='costs', kind='scatter')
plt.xlabel('emissions')
plt.ylabel('costs')
plt.show()
