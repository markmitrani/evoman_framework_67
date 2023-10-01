import pandas as pd
import time
import datetime
import os
import csv
from es import CellularES, ES

statistics = {
        'enemy': [],
        'gen': [],
        'metric_max': [],
        'metric_avg': [],
        'method': []

}
winners = {
    'method': [],
    'enemy': [],
    'vector': []
}
for algorithm in [CellularES, ES]:
    for enemy_num in [3, 5, 7]:
        for i in range(10):
            start_time = time.time()
            ea = algorithm(enemy=enemy_num)
            for record in ea.run_n():
                statistics['enemy'].append(enemy_num)
                statistics['gen'].append(ea.get_gen())
                statistics['metric_max'].append(record['max'])
                statistics['metric_avg'].append(record['avg'])
                statistics['method'].append(str(ea))

            if not os.path.exists('results'):
                os.mkdir('results')
            if not os.path.exists(f'results/{enemy_num}'):
                os.mkdir(f'results/{enemy_num}')
            if not os.path.exists(f'results/{enemy_num}/{str(ea)}'):
                os.mkdir(f'results/{enemy_num}/{str(ea)}')
            if not os.path.exists(f'results/{enemy_num}/{str(ea)}/{str(datetime.date.today())}'):
                os.mkdir(f'results/{enemy_num}/{str(ea)}/{str(datetime.date.today())}')

            best_so_far = ea.get_best()
            with open(f'results/{enemy_num}/{str(ea)}/{str(datetime.date.today())}/winner_{i}.csv', 'w') as f:
                f.writelines(';'.join([str(i) for i in list(best_so_far)]))
            print(f'{str(ea)} on enemy {enemy_num} it {i} done!')
            #df = pd.DataFrame.from_dict(statistics)
            #df.to_csv('statistics.csv')
            end_time = time.time()
            #print(f'Execute time: {end_time-start_time}')