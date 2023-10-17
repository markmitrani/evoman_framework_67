from pso import PSO, MSPSO
import time
import numpy as np
import pandas as pd
from ea import EA
enemy_groups = [
    [4, 6, 8],
    [1, 3, 5]
]

POPULATION_SIZE = 150
SWARM_NUM = 15
NGEN = 100

data_dict = {
    'algorithm': [],
    'run': [],
    'gen': [],
    'mean_fitness': [],
    'max_fitness': [],
    'enemy_group': [],
}

for algorithm in [ PSO, MSPSO]:
    for enemy_gr in enemy_groups:
        for i in range(10):
            a: EA = algorithm(enemy=enemy_gr, multimode=True,
                           POPULATION_SIZE=POPULATION_SIZE,
                           SWARM_NUM=SWARM_NUM,
                           NGEN=NGEN)
            start = time.time()
            for r in a.run_n():
                data_dict['algorithm'].append(str(a))
                data_dict['run'].append(str(i))
                data_dict['gen'].append(a.get_gen())
                data_dict['mean_fitness'].append(r['avg'])
                data_dict['max_fitness'].append(r['max'])
                data_dict['enemy_group'].append(str(enemy_gr))
            c_best = a.get_best()
            f, p, e, t, d = a.env.play(pcont=c_best)
            np.savetxt(f'best_weights_report/{str(a)}_{enemy_gr}_{f}_{p-e}_{d}.txt', c_best)
            print(f'Exec time run {time.time() - start}')
        print(f'Finished run {i} of group {enemy_gr} on algo {str(a)}')    

df = pd.DataFrame.from_dict(data_dict)
df.to_csv('data_report2.csv')
