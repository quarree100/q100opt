from q100opt.scenario_tools import ParetoFront
from matplotlib import pyplot as plt
from q100opt.setup_model import load_csv_data

table_collection = load_csv_data('data')

pf = ParetoFront(
    table_collection=table_collection,
    number_of_points=5,
    solver='cbc',
)

pf.calc_pareto_front()

pf.pareto_front.plot(x='emissions', y='costs', kind='scatter')
plt.xlabel('emissions')
plt.ylabel('costs')
plt.show()

# pf.store_results()
