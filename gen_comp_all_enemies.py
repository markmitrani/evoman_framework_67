import os
import pandas as pd
import numpy as np
from demo_controller import player_controller
from evoman.environment import Environment


best_weights_dict = {
    'algorithm': [],
    'enemy_group': [],
    'gain': [],
    'defeated': [],
    'fitness': [],
    'player_life': [],
    'enemy_life': [],
    'time': []
}

headless = True
for filename in os.listdir('best_weights_report'):
    X = np.loadtxt(os.path.join('best_weights_report', filename))
    parts = filename.split('_')
    algo_name = parts[0]
    enemy_group = parts[1]
    #enemy_group_int = [int(num) for num in enemy_group.replace('[', '').replace(']', '').split(',')]


    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    experiment_name='cellular__es'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)
    env = Environment(experiment_name=experiment_name,
                enemies=[1,2,3,4,5,6,7,8],
                multiplemode='yes',
                playermode="ai",
                player_controller=player_controller(10),
                enemymode="static",
                level=2,
                speed="fastest",
                visuals=False)

    f, p, e, t, d = env.play(pcont=X)

    for i in range(5):
        best_weights_dict['player_life'].append(p)
        best_weights_dict['enemy_life'].append(e)
        best_weights_dict['fitness'].append(f)
        best_weights_dict['gain'].append(p-e)
        best_weights_dict['time'].append(t)
        best_weights_dict['defeated'].append(d)
        best_weights_dict['algorithm'].append(algo_name)
        best_weights_dict['enemy_group'].append(enemy_group)


df = pd.DataFrame.from_dict(best_weights_dict)
df.to_csv('data_gain.csv')