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
    #########################################
    # GRID SEARCH
    for cxpb in grid_search_params["CXPB"]:
        for mutpb in grid_search_params["MUTPB"]:
            for pop_size in grid_search_params["POPULATION_SIZE"]:
                # Create an instance of your algorithm with the current hyperparameters
                ea = algorithm(enemy=enemy_num)

                # Set the algorithm's hyperparameters
                algorithm.CXPB = cxpb
                algorithm.MUTPB = mutpb
                algorithm.POPULATION_SIZE = pop_size
                # Set other hyperparameters in your algorithm as needed

                for enemy_num in [3, 5, 7]:
                    for i in range(10):
                        start_time = time.time()

                        for record in ea.run_n():
                            statistics['enemy'].append(enemy_num)
                            statistics['gen'].append(ea.get_gen())
                            statistics['metric_max'].append(record['max'])
                            statistics['metric_avg'].append(record['avg'])
                            statistics['method'].append(str(ea))
                            grid_search_results[(cxpb, mutpb, pop_size)] = record["max"]

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
                        
    # Find the best hyperparameters from Grid Search
    best_grid_search_params = max(grid_search_results, key=grid_search_results.get)
    best_grid_search_value = grid_search_results[best_grid_search_params]

    # Print the best hyperparameters and value from Grid Search
    print("Best Hyperparameters (Grid Search):", best_grid_search_params)
    print("Best Value (Grid Search):", best_grid_search_value)
