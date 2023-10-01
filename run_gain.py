from es import CellularES, ES
import pandas as pd
from nn_controller import player_controller
from evoman.environment import Environment
import os


headless = True

def evaluate(individual, env):
    #c = player_controller(H_NODES_LAYERS)
    #env.player_controller.set(individual, 20)
    f,p,e,t = env.play(pcont=individual)
    # Added the +10 because selection algorithms don't work on negative numbers / 0
    return (f,p-e )

data_map = {
    'enemy': [],
    'algorithm': [],
    'gain': [],
    'fitness': [],

}
for algorithm in ['Cellular-ES', 'ES']:
    for enemy_num in [3, 5, 7]:
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        experiment_name='cellular__es'
        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name)
        env = Environment(experiment_name=experiment_name,
                    enemies=[enemy_num],
                    playermode="ai",
                    player_controller=player_controller([10]),
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)
        for i in range(10):
            path_csv = f'results/{enemy_num}/{algorithm}/2023-09-26/winner_{i}.csv'
            with open(path_csv, 'r') as f:
                l = f.readline() 
                values = [float(v) for v in l.split(';')]
                for j in range(5):
                    f, g = evaluate(values, env)
                    data_map['enemy'].append(enemy_num)
                    data_map['algorithm'].append(algorithm)
                    data_map['gain'].append(g)
                    data_map['fitness'].append(f)

df = pd.DataFrame.from_dict(data_map)
df.to_csv('gain_statistics.csv')
