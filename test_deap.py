import random
import math
import operator
import os
import scipy
import numpy as np
import array
from typing import Dict, List
from itertools import chain
from deap import base, creator, tools, algorithms, cma

from evoman.environment import Environment
from deap_constants import *
from es import ES, CMA
from deap_algorithms import *
#from nn_controller import player_controller
from demo_controller import player_controller
from graph import GridGraph, manhattan_distance, Node
import cma as e_cma
import optuna


def run_es_opt(trial):
    NGEN = trial.suggest_int('NGEN', 10, 150)
    POPULATION_SIZE = trial.suggest_int('POPULATION_SIZE', 10, 150)
    MIN_VALUE = trial.suggest_float('MIN_VALUE', -100, -0.1)
    MAX_VALUE = trial.suggest_float('MAX_VALUE', 0.1, 100)
    MIN_STRATEGY = trial.suggest_float('MIN_VALUE', -100, -0.1)
    MAX_STRATEGY = trial.suggest_float('MAX_STRATEGY', 0.1, 100)
    TOUR_SIZE = trial.suggest_int('TOUR_SIZE', 2, 20)

    # DEAP Params
    ALPHA = trial.suggest_float('ALPHA', 0.1, 1.0)
    C = trial.suggest_float('C', 0.1, 1.0)
    INDPB = trial.suggest_float('INDPB', 0.1, 1.0)
    CXPB = trial.suggest_float('CXPB', 0.1, 1.0)
    MUTPB = trial.suggest_float('MUTPB', 0.1, 1.0)

    es = ES([1, 2, 3, 4, 5, 6, 7, 8], True, NGEN=NGEN, POPULATION_SIZE=POPULATION_SIZE, \
            MIN_VALUES=MIN_VALUE, MAX_VALUE=MAX_VALUE, MIN_STRATEGY=MIN_STRATEGY, MAX_STRATEGY=MAX_STRATEGY,
            TOUR_SIZE=TOUR_SIZE, ALPHA=ALPHA, C=C, INDPB=INDPB, CXPB=CXPB, MUTPB=MUTPB)
    for record in  es.run_n():
        pass
    es.env.enemies = [i for i in range(1, 9)]
    #es.env.update_parameter("speed", "normal")
    #es.env.update_parameter("visuals", True)
    #es.env.visuals = True
    #es.env.speed = "normal"
    f, p, e, t, defeated = es.env.play(pcont=es.get_best())
    return len(defeated)
    #print(f'f, p, e, t, defeated: {es.env.play(pcont=es.get_best())}')



def main_es():
    study = optuna.create_study(
        direction='maximize',
        storage="sqlite:///es_all_enemies",  # Specify the storage URL here.
        study_name="es_all_enemies",
        load_if_exists=True

    )
    study.optimize(run_es_opt, n_trials=100)
    print(study.best_params)

def main():
    es = ES([1, 4, 6, 7], multimode=True)
    cma_es = e_cma.CMAEvolutionStrategy(np.random.uniform(-10, 10, size=IND_SIZE), 5, 
                                      inopts={'popsize': 50})

    while not cma_es.stop():
        X = cma_es.ask()
        fit = [-1 * (es.env.play(pcont=x)[0]) for x in X]
        cma_es.tell(X, fit)
        cma_es.disp()
    print(cma_es.result_pretty())
    res = cma_es.result[0]
    #es.env.update_parameter("speed", "normal")
    #es.env.update_parameter("visuals", True)
    #es.env.visuals = True
    #es.env.speed = "normal"
    print(es.env.play(res))
    from  datetime import datetime
    with open(f'solutions_demo/{datetime.now()}.txt') as f:
        for val in res:
            f.write(f'{val}\n')

def evaluate(env, individual):
    #c = player_controller(H_NODES_LAYERS)
    #env.player_controller.set(individual, 20)
    f,p,e,t,d = env.play(pcont=individual)
    return (f, p-e, d)

def generateES(icls, scls, size, imin, imax, smin, smax):
    ind = icls(random.uniform(imin, imax) for _ in range(size))
    ind.strategy = scls(random.uniform(smin, smax) for _ in range(size))
    return ind

def main_NSGAii():
    enemies = [1, 2, 3, 4, 7]

    toolbox = base.Toolbox()
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    creator.create('Individual', array.array, fitness=creator.FitnessMax, strategy=None, typecode='d')
    creator.create('Strategy', array.array, typecode='d')

    toolbox.register('attr_float', random.random)
    toolbox.register("individual", generateES, creator.Individual, creator.Strategy,
        IND_SIZE, MIN_VALUE, MAX_VALUE, MIN_STRATEGY, MAX_STRATEGY)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxESBlend, alpha=ALPHA)
    #toolbox.register("mate", tools.cxESTwoPoint)
    toolbox.register('mutate', tools.mutESLogNormal, c=C, indpb=INDPB)
    #toolbox.register("mutate", tools.mutESLogNormal, c=0.2, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)
    #toolbox.register("select", tools.selNSGA3)



    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    experiment_name='cellular__es'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)
    env = Environment(experiment_name=experiment_name,
                enemies=enemies,
                multiplemode='yes',
                playermode="ai",
                player_controller=player_controller(H_NODES_LAYERS[0]),
                enemymode="static",
                level=2,
                speed="fastest",
                visuals=False)
    toolbox.register('evaluate', evaluate, env=env)
    pop = toolbox.population(POPULATION_SIZE)

    # Constants for Mutation / Crossover
    fitnesses = [toolbox.evaluate(individual=ind) for ind in pop]
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    winner = None
    for g in range(NGEN):
        parents = toolbox.clone(pop) 
        #parents = toolbox.select(pop, k=len(pop)*2)
        # Crossover and Mutation
        offspring = algorithms.varAnd(parents, toolbox, CXPB, MUTPB)

        # Evaluate fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = [toolbox.evaluate(individual=ind) for ind in invalid_ind]
        #fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replacement
        pop[:] = toolbox.select(pop + offspring, k=len(pop))
        #self.pop = self.toolbox.select(self.pop + offspring, len(self.pop))

        #TODO How to ?
        # TODO Update mutation parameters here

        #env.update_solutions(pop)
        record = stats.compile(pop)
        candidate_winner = pop[np.argmax([i.fitness for i in pop])]
        if type(winner) is not creator.Individual:
            winner = candidate_winner
        winner = winner if winner.fitness > candidate_winner.fitness else candidate_winner
        print(g, record)
    env.multiplemode='yes'
    env.enemies = [i for i in range(1, 9)]
    #env.update_parameter("speed", "normal")
    #env.update_parameter("visuals", True)
    #env.visuals = True
    #env.speed = "normal"
    f, p, e, t, deafeated = env.play(pcont=winner)
    gain = p-e
    print(f'f, gain, defeated: {f}, {gain}, {deafeated}')

def main_CMA():
    cma_es = CMA(enemy=[1, 4 ,6, 7], multimode=True)
    for it in cma_es.run_n():
        pass

    cma_es.env.enemies = [i for i in range(1, 9)]
    #env.update_parameter("speed", "normal")
    #env.update_parameter("visuals", True)
    #env.visuals = True
    #env.speed = "normal"
    f, p, e, t, deafeated = cma_es.env.play(pcont=cma_es.get_best())
    gain = p-e
    print(f'f, gain, defeated: {f}, {gain}, {deafeated}')

def generate(size, pmin, pmax, smin, smax, gain, defeated):
    part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size)) 
    part.speed = [random.uniform(smin, smax) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    part.gain=gain
    part.defeated = defeated
    return part

def updateParticle(part, best, w, phi1, phi2):
    u1 = np.random.uniform(0, phi1, size=len(part))
    u2 = np.random.uniform(0, phi2, size=len(part))
    v_u1 = u1 * (np.array(part.best) - np.array(part))
    #v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
    v_u2 = u2 * (np.array(best) - np.array(part))
    #v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
    part.speed = (w * np.array(part.speed) + v_u1 + v_u2).tolist()
    #part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))
    for i, speed in enumerate(part.speed):
        if abs(speed) < part.smin:
            part.speed[i] = math.copysign(part.smin, speed)
        elif abs(speed) > part.smax:
            part.speed[i] = math.copysign(part.smax, speed)
    part[:] = (np.array(part)  + np.array(part.speed)).tolist()

def run_pso_opt(trial):
    ngen = 300
    population_size = trial.suggest_int('POPULATION_SIZE', 10, 375)
    min_value = trial.suggest_float('MIN_VALUE', -1, 0)
    max_value = trial.suggest_float('MAX_VALUE', 0, 1)
    smin_value = trial.suggest_float('SMIN_VALUE', -1, 0)
    smax_value = trial.suggest_float('SMAX_VALUE', 0, 1)

    # DEAP Params
    phi_1 = trial.suggest_float('PHI_1', 0.0, 5)
    phi_2 = trial.suggest_float('PHI_2', 0.0, 5)

    print('\n')
    print(min_value, max_value, smin_value, smax_value, phi_1, phi_2)

    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Particle", list, fitness=creator.FitnessMax, speed=list, 
        smin=None, smax=None, best=None, gain=None, defeated=0)
    toolbox = base.Toolbox()
    toolbox.register("particle", generate, size=265, pmin=min_value, \
                    pmax=max_value, smin=smin_value, smax=smax_value, gain=None, defeated=0)
    toolbox.register("population", tools.initRepeat, list, toolbox.particle)
    toolbox.register("update", updateParticle, phi1=phi_1, phi2=phi_2)
    enemies = [1, 2, 3, 4, 5, 6, 7, 8]
    env = Environment(experiment_name='cellular__es',
                enemies=enemies,
                multiplemode='yes',
                playermode="ai",
                player_controller=player_controller(H_NODES_LAYERS[0]),
                enemymode="static",
                level=2,
                speed="fastest",
                visuals=False)
    toolbox.register('evaluate', evaluate, env=env)
    pop = toolbox.population(n=population_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    best = None

    w = 1.0
    w_dec = 0.0035 
    for g in range(ngen):
        for part in pop:
            fitness, gain, defeated = toolbox.evaluate(individual=part)
            part.gain = gain
            part.defeated = defeated
            part.fitness.values = (fitness, )
            if not part.best or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.defeated = part.defeated
                part.best.gain = part.gain
                part.best.fitness.values = part.fitness.values
            if not best or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.gain = part.gain
                best.defeated = part.defeated
                best.fitness.values = part.fitness.values
        for part in pop:
            toolbox.update(part, best, w)
        w -= w_dec
        if len(best.defeated) >= 6:
            np.savetxt(f'good_weights/pso_opt_12345678_{best.gain}_{best.defeated}.txt')
        print(g, best.fitness, best.gain, best.defeated)

        # Gather all the fitnesses in one list and print the stats
        #jap_1yRPEXhCQWprtAjMjSF2ut2skjJTPcE127884

    env.enemies = [i for i in range(1, 9)]
    #es.env.update_parameter("speed", "normal")
    #es.env.update_parameter("visuals", True)
    #es.env.visuals = True
    #es.env.speed = "normal"
    f, p, e, t, defeated = env.play(pcont=best)
    return len(defeated)
    #print(f'f, p, e, t, defeated: {es.env.play(pcont=es.get_best())}')

def main_PSO():
    study = optuna.create_study(
        direction='maximize',
        storage="sqlite:///pso_all_enemies_no_time",  # Specify the storage URL here.
        study_name="pso_all_enemies_no_time",
        load_if_exists=True
    )
    study.optimize(run_pso_opt, n_trials=100)
    print(study.best_params)

#main_NSGAii()
#main_CMA()
#main()
#main_es()
main_PSO()
