from pso import PSO, MSPSO
import time
import numpy as np
import pandas as pd
from ea import EA
enemy_groups = [
    [4, 6, 8],
    [1, 3, 5]
]

POPULATION_SIZE = 100
SWARM_NUM = 20
NGEN = 100
R = 25

data_dict = {
    'algorithm': [],
    'run': [],
    'gen': [],
    'mean_fitness': [],
    'max_fitness': [],
    'enemy_group': [],
}

for algorithm in [PSO, MSPSO]: #[ PSO, MSPSO]:
    for enemy_gr in enemy_groups:
        for i in range(10):
            a: EA = algorithm(enemy=enemy_gr, multimode=True,
                            POPULATION_SIZE=POPULATION_SIZE,
                            SWARM_NUM=SWARM_NUM,
                            NGEN=NGEN,
                            R=R)
            start = time.time()
            for r in a.run_n(print_each_gen=True):
                data_dict['algorithm'].append(str(a))
                data_dict['run'].append(str(i))
                data_dict['gen'].append(a.get_gen())
                data_dict['mean_fitness'].append(r['avg'])
                data_dict['max_fitness'].append(r['max'])
                data_dict['enemy_group'].append(str(enemy_gr))
            c_best = a.get_best()
            np.savetxt(f'good_weights_test/{str(a)}_{enemy_gr}_{c_best.fitness.values[0]}_{c_best.gain}_{c_best.defeated}.txt', c_best)
            print(f'Exec time run {time.time() - start}')
        print(f'Finished run {i} of group {enemy_gr} on algo {str(a)}')    

df = pd.DataFrame.from_dict(data_dict)
df.to_csv('data_report2.csv')
