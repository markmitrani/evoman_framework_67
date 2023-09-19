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

headless = False 
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

env = Environment(experiment_name='test_deap',
                  enemies=[2],
                  playermode="ai",
                  player_controller=player_controller(H_NODES_LAYERS),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)



# DEAP Configurations

creator.create('FitnessMax', base.Fitness, weights=(1.0,))
creator.create('Individual', np.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register('attr_float', random.random)
toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attr_float, n=IND_SIZE)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
toolbox.register('evaluate', evaluate, env)
#toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxUniform, indpb=0.2)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
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