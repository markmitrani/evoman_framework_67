import pandas as pd
from es import CellularES, ES

statistics = {
        'enemy': [],
        'gen': [],
        'metric_max': [],
        'metric_avg': [],
        'method': []

}
winners = {
}
for algorithm in [CellularES, ES]:
    for enemy_num in [3, 5, 7]:
        best_so_far = None
        for i in range(1):
            ea = algorithm(enemy=enemy_num)
            for record in ea.run_n():
                statistics['enemy'].append(enemy_num)
                statistics['gen'].append(ea.get_gen())
                statistics['metric_max'].append(record['max'])
                statistics['metric_avg'].append(record['avg'])
                statistics['method'].append(str(ea))
            try:
                best_so_far.fitness
            except: 
                best_so_far = ea.get_best()
            else:
                best_so_far = best_so_far if best_so_far.fitness > ea.get_best().fitness else ea.get_best()
        winners[enemy_num] = best_so_far
        print(statistics)

        df = pd.DataFrame.from_dict(statistics)
        df.to_csv('statistics.csv')
        pd.DataFrame.from_dict(winners).to_csv('winners.csv')