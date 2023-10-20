import os
import numpy as np
from demo_controller import player_controller
from evoman.environment import Environment


best_weights_dict = {
    'MSPSO': {
        '[1, 3, 5]': (-1, []),
        '[4, 6, 8]': (-1, [])
    },
    'PSO': {
        '[1, 3, 5]': [-1, []],
        '[4, 6, 8]': [-1, []]
    }
}

headless = True
for filename in os.listdir('best_weights_report'):
    X = np.loadtxt(os.path.join('best_weights_report', filename))
    parts = filename.split('_')
    algo_name = parts[0]
    enemy_group = parts[1]
    enemy_group_int = [int(num) for num in enemy_group.replace('[', '').replace(']', '').split(',')]


    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    experiment_name='cellular__es'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)
    env = Environment(experiment_name=experiment_name,
                enemies=enemy_group_int,
                multiplemode='yes',
                playermode="ai",
                player_controller=player_controller(10),
                enemymode="static",
                level=2,
                speed="fastest",
                visuals=False)
    f, p, e, t, d = env.play(pcont=X)

    if f > best_weights_dict[algo_name][enemy_group][0]:
        best_weights_dict[algo_name][enemy_group] = (f, X)


for k, v in best_weights_dict.items():
    for kk, vv in best_weights_dict[k].items():
        print(f'{k}_{kk}.txt', vv[0])
        np.savetxt(f'winner_weights_report/{k}_{kk}.txt', vv[1])
