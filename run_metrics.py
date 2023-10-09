import pandas as pd
import time
import datetime
import os
import csv
import random
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

# Define the hyperparameter search spaces for Random Search
random_search_params = {
    "CXPB": (0.0, 1.0),  # Specify the search range
    "MUTPB": (0.0, 1.0),  # Specify the search range
    "POPULATION_SIZE": (50, 500),  # Specify the search range
    # Define other hyperparameters and their search ranges as needed
}

# Perform Random Search
random_search_results = {}
num_random_trials = 100  # Adjust the number of random trials as needed

for algorithm in [CellularES, ES]:
    #########################################
    # RANDOM SEARCH

    for _ in range(num_random_trials):
        # Generate random hyperparameters within the defined search spaces
        cxpb = random.uniform(*random_search_params["CXPB"])
        mutpb = random.uniform(*random_search_params["MUTPB"])
        pop_size = random.randint(*random_search_params["POPULATION_SIZE"])
        # Generate other random hyperparameters as needed

        # Create an instance of your algorithm with the random hyperparameters
        ea = algorithm(enemy=enemy_num)

        # Set the algorithm's hyperparameters
        algorithm.CXPB = cxpb
        algorithm.MUTPB = mutpb
        algorithm.POPULATION_SIZE = pop_size
        # Set other random hyperparameters in your algorithm as needed

        for enemy_num in [3, 5, 7]:
            for i in range(10):
                start_time = time.time()

                for record in ea.run_n():
                    statistics['enemy'].append(enemy_num)
                    statistics['gen'].append(ea.get_gen())
                    statistics['metric_max'].append(record['max'])
                    statistics['metric_avg'].append(record['avg'])
                    statistics['method'].append(str(ea))
                    random_search_results[(cxpb, mutpb, pop_size)] = record["max"]

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

    # Find the best hyperparameters from Random Search
    best_random_search_params = max(random_search_results, key=random_search_results.get)
    best_random_search_value = random_search_results[best_random_search_params]

    # Print the best hyperparameters and value from Random Search
    print("Best Hyperparameters (Random Search):", best_random_search_params)
    print("Best Value (Random Search):", best_random_search_value)
