import random
import os
import numpy as np
from typing import Dict, List

from deap import base, creator, tools, algorithms

from evoman.environment import Environment
from deap_constants import CXPB, MUTPB, calculate_ind_size
from deap_algorithms import *
from nn_controller import player_controller


def evaluate(env, individual):
    #c = player_controller(H_NODES_LAYERS)
    #env.player_controller.set(individual, 20)
    f,p,e,t = env.play(pcont=individual)
    # Added the +10 because selection algorithms don't work on negative numbers / 0
    return (f+10, )


NGEN = 50
H_NODES_LAYERS = [20, 10]
IND_SIZE = calculate_ind_size(H_NODES_LAYERS)
MIN_VALUE = -30
MAX_VALUE = 30
MIN_STRATEGY = -1
MAX_STRATEGY = 1

headless = False
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name='test_deap'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

env = Environment(experiment_name=experiment_name,
                  enemies=[2],
                  playermode="ai",
                  player_controller=player_controller(H_NODES_LAYERS),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)


def self_adaptive_correlated_mutation(ind):
    for i in range(len(ind.strategy)):
        ind.strategy[i] *= np.exp(random.gauss(0, 1))

    for i in range(len(ind.strategy)):
        ind.strategy[i] += random.gauss(0, ind.strategy[i])

    return ind,


def generateES(icls, scls, size, imin, imax, smin, smax):
    ind = icls(random.uniform(imin, imax) for _ in range(size))
    ind.strategy = scls(random.uniform(smin, smax) for _ in range(size))
    return ind

def checkStrategy(minstrategy):
    def decorator(func):
        def wrappper(*args, **kargs):
            children = func(*args, **kargs)
            for child in children:
                for i, s in enumerate(child.strategy):
                    if s < minstrategy:
                        child.strategy[i] = minstrategy
            return children
        return wrappper
    return decorator

# DEAP Configurations

creator.create('FitnessMax', base.Fitness, weights=(1.0,))
creator.create('Individual', np.ndarray, fitness=creator.FitnessMax, strategy=None)
creator.create('Strategy', np.ndarray, typecode="d")

toolbox = base.Toolbox()
toolbox.register('attr_float', random.random)
#toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attr_float, n=IND_SIZE)
toolbox.register("individual", generateES, creator.Individual, creator.Strategy,
    IND_SIZE, MIN_VALUE, MAX_VALUE, MIN_STRATEGY, MAX_STRATEGY)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
toolbox.register('evaluate', evaluate, env)
#toolbox.register("select", tools.selTournament, tournsize=3)
#toolbox.register("mate", tools.cxESBlend, alpha=0.1)
toolbox.register("mate", tools.cxUniform, indpb=0.2)
toolbox.register("mutate", self_adaptive_correlated_mutation)

#toolbox.decorate("mate", checkStrategy(MIN_STRATEGY))
toolbox.decorate("mutate", checkStrategy(MIN_STRATEGY))
#toolbox.register("mutate", self_adaptive_correlated_mutation, sigma=[initial_sigma] * IND_SIZE)
toolbox.register("select", tools.selTournament, tournsize=4)

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

hof = tools.HallOfFame(maxsize=1)
winner = None

def main():
    pop = toolbox.population(n=150)
    # Constants for Mutation / Crossover
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in range(NGEN):
        # Crossover and Mutation
        offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)

        # Evaluate fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replacement
        #pop[:] = offspring
        pop = toolbox.select(pop + offspring, len(pop))

        #TODO How to ?
        # TODO Update mutation parameters here

        #env.update_solutions(pop)
        record = stats.compile(pop)
        candidate_winner = pop[np.argmax([i.fitness for i in pop])]
        global winner
        if type(winner) is not creator.Individual:
            winner = candidate_winner
        winner = winner if winner.fitness > candidate_winner.fitness else candidate_winner
        print(g, record)

main()
env.update_parameter("speed", "normal")
env.update_parameter("visuals", True)
env.visuals = True
env.speed = "normal"
evaluate(env, winner)
