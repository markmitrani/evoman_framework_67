import math
import time
import os
import numpy as np
from evoman.environment import Environment
from cmaes import CMA
import optuna


from demo_controller import player_controller

H_NODES_LAYERS=[10]
def evaluate(env, ind):
    #c = player_controller(H_NODES_LAYERS)
    #env.player_controller.set(individual, 20)
    f,p,e,t,d = env.play(pcont=ind)
    return (f, )

def func(trial):
    NGEN = trial.suggest_int('NGEN', 200, 300)
    POPULATION_SIZE = trial.suggest_int('POPULATION_SIZE', 10, 150)
    MIN_VALUE = -1#trial.suggest_float('MIN_VALUE', -1, 0)
    MAX_VALUE = 1#trial.suggest_float('MAX_VALUE', 0, 1)

    headless=True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    experiment_name='cellular__es'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    enemy = [1, 2, 3, 4, 5, 6, 7]
    if type(enemy) == int:
        enemy = [enemy]
    env = Environment(experiment_name=experiment_name,
                enemies=enemy,
                multiplemode='yes',
                playermode="ai",
                player_controller=player_controller(H_NODES_LAYERS[0]),
                enemymode="static",
                level=2,
                speed="fastest",
                visuals=False)
    #toolbox.register('evaluate', evaluate, env=env)

    bounds = np.array([[MIN_VALUE, MAX_VALUE] for _ in range(265)])
    lower_bounds, upper_bounds = np.array([MIN_VALUE] * 265), np.array([MAX_VALUE] * 265)

    mean = lower_bounds + (np.random.rand(265) * (upper_bounds - lower_bounds))
    sigma = MAX_VALUE * 2 / 5  # 1/5 of the domain width
    optimizer = CMA(mean=mean, sigma=sigma, bounds=bounds, seed=0)

    n_restarts = 0  # A small restart doesn't count in the n_restarts
    small_n_eval, large_n_eval = 0, 0
    popsize0 = optimizer.population_size
    inc_popsize = 2

    # Initial run is with "normal" population size; it is
    # the large population before first doubling, but its
    # budget accounting is the same as in case of small
    # population.
    poptype = "small"

    for generation in range(200):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = -evaluate(env=env, ind=x)[0]
            solutions.append((x, value))
            if value < -78:
                np.savetxt(f'good_weights/bipopcmaes_1234567_{generation}_{value}_{time.perf_counter()}.txt', x)
            print(f"#{generation} {value} ")
        optimizer.tell(solutions)

        if optimizer.should_stop():
            n_eval = optimizer.population_size * optimizer.generation
            if poptype == "small":
                small_n_eval += n_eval
            else:  # poptype == "large"
                large_n_eval += n_eval

            if small_n_eval < large_n_eval:
                poptype = "small"
                popsize_multiplier = inc_popsize ** n_restarts
                popsize = math.floor(
                    popsize0 * popsize_multiplier ** (np.random.uniform() ** 2)
                )
            else:
                poptype = "large"
                n_restarts += 1
                popsize = popsize0 * (inc_popsize ** n_restarts)

            mean = lower_bounds + (np.random.rand(265) * (upper_bounds - lower_bounds))
            optimizer = CMA(
                mean=mean,
                sigma=sigma,
                bounds=bounds,
                population_size=popsize,
            )
            print("Restart CMA-ES with popsize={} ({})".format(popsize, poptype))
    env.enemies = [i for i in range(1, 9)]
    #es.env.update_parameter("speed", "normal")
    #es.env.update_parameter("visuals", True)
    #es.env.visuals = True
    #es.env.speed = "normal"
    f, p, e, t, defeated = env.play(pcont=es.get_best())
    return len(defeated)
    #print(f'f, p, e, t, defeated: {es.env.play(pcont=es.get_best())}')

if __name__ == "__main__":
    study = optuna.create_study(
        direction='maximize',
        storage="sqlite:///cmaes_all_enemies",  # Specify the storage URL here.
        study_name="cmaes_all_enemies",
        load_if_exists=True

    )
    study.optimize(func, n_trials=100)
    print(study.best_params)
