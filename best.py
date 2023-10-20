import os
import numpy as np
from demo_controller import player_controller
from evoman.environment import Environment

headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
experiment_name='cellular__es'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

enemies = [i for i in range(1, 9)]
env = Environment(experiment_name=experiment_name,
            enemies=enemies,
            multiplemode='yes',
            playermode="ai",
            player_controller=player_controller(10),
            enemymode="static",
            level=2,
            speed="fastest",
            visuals=False)

def evaluate(env, individual):
    #c = player_controller(H_NODES_LAYERS)
    #env.player_controller.set(individual, 20)
    f,p,e,t,d = env.play(pcont=individual)
    return (p-e, p-e, d)

best_score = (-1, [])
for filename in os.listdir('good_weights'):
    f_path = os.path.join('good_weights', filename)
    X = np.loadtxt(f_path)

    g = evaluate(env, X)[0]

    if g > best_score[0]:
        best_score = (g, X)

print(best_score)
np.savetxt('best_weight.txt', best_score[1])

    




