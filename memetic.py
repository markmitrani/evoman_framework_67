import random
import copy
from cProfile import Profile
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

start_pop = []
for filename in os.listdir('good_weights'):
    x = np.loadtxt('good_weights/' + filename)
    start_pop.append(x)

def run_es_opt(trial):
    NGEN = 200
    MIN_VALUE = -1
    MAX_VALUE = 1
    MIN_STRATEGY = -1
    MAX_STRATEGY = 1
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
    copy_pop = copy.deepcopy(start_pop)
    new_pop = []
    for vec in copy_pop:
        ind = creator.Individual(vec)
        ind.strategy = np.random.uniform(MIN_STRATEGY, MAX_STRATEGY, size=265)
        ind.fitness.values = (es.evaluate(ind)[1], )
        new_pop.append(ind)
    es.pop = new_pop

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
        storage="sqlite:///es_all_enemies_memetic",  # Specify the storage URL here.
        study_name="es_all_enemies_memetic",
        load_if_exists=True

    )
    study.optimize(run_es_opt, n_trials=100)
    print(study.best_params)
main_es()