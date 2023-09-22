import random
import os
import numpy as np
import array
from typing import Dict, List
from itertools import chain

from deap import base, creator, tools, algorithms

from evoman.environment import Environment
from deap_constants import CXPB, MUTPB, calculate_ind_size
from deap_algorithms import *
from nn_controller import player_controller
from graph import GridGraph, manhattan_distance


def evaluate(env, individual):
    #c = player_controller(H_NODES_LAYERS)
    #env.player_controller.set(individual, 20)
    f,p,e,t = env.play(pcont=individual)
    # Added the +10 because selection algorithms don't work on negative numbers / 0
    return (f, )


NGEN = 50
POPULATION_SIZE = 15
GRID_N_SIZE = 5
GRID_M_SIZE = 5
H_NODES_LAYERS = [1]
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
        ind[i] += random.gauss(0, ind.strategy[i])

    return ind


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
creator.create('Individual', array.array, fitness=creator.FitnessMax, strategy=None, typecode='d')
creator.create('Strategy', array.array, typecode='d')

toolbox = base.Toolbox()
toolbox.register('attr_float', random.random)
toolbox.register("individual", generateES, creator.Individual, creator.Strategy,
    IND_SIZE, MIN_VALUE, MAX_VALUE, MIN_STRATEGY, MAX_STRATEGY)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

toolbox.register('evaluate', evaluate, env)
toolbox.register("mate", tools.cxESTwoPoint)
toolbox.register("mutate", self_adaptive_correlated_mutation)
toolbox.register("select", tools.selTournament, tournsize=2)

#toolbox.decorate("mutate", checkStrategy(MIN_STRATEGY))

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

hof = tools.HallOfFame(maxsize=1)
winner = None

def main():
    # Initialize grid graph with populations
    gg = GridGraph(GRID_N_SIZE, GRID_M_SIZE, pop_size=POPULATION_SIZE, toolbox=toolbox)



    gg.update_fitnesses(toolbox)
    for g in range(NGEN):
        # Iterate over each deme
        new_gg: GridGraph = gg.deepcopy()
        for node in gg: 
            
            node: GridGraph.Node = node
            clone_node: GridGraph.Node = new_gg[node.get_coords()[0]][node.get_coords()[1]]
            node_pop = node.get_pop()

            # Get the population of all the neighbouring nodes
            neighbour_pop = [n_i.get_pop() for n_i in gg.get_knn(node.get_coords()[0], node.get_coords()[1], manhattan_distance, max_dist=1)]
            neighbour_pop = list(chain.from_iterable(neighbour_pop))

            for ind_idx, ind in enumerate(node_pop):
                offspring = toolbox.clone(ind)
                change_ocurred = False

                # Crossover
                if random.random() < CXPB:
                    change_ocurred = True
                    partner = random.choice(neighbour_pop)
                    while offspring == partner:
                        partner = random.choice(neighbour_pop)

                    offspring_1, offspring_2 = toolbox.mate(offspring, partner)

                    offspring = random.choice([offspring_1, offspring_2])

                    del offspring.fitness.values

                # Mutation
                if random.random() < MUTPB:
                    change_ocurred = True
                    offspring = toolbox.mutate(offspring)
                    del offspring.fitness.values

                if not change_ocurred:
                    continue
                
                offspring.fitness.values = toolbox.evaluate(offspring)

                # Replacement
                if offspring.fitness.values[0] >= ind.fitness.values[0]:
                    clone_node[ind_idx] = offspring
                else:
                    assert(offspring != ind)
                    clone_node[ind_idx] = ind
            

            #record = stats.compile(clone_node.get_popu())
            #print(f'gen: {g}, node: {node}, record: {record}')



        gg = new_gg
        allpop = gg.allpop()
        record = stats.compile(allpop)
        candidate_winner = allpop[np.argmax([i.fitness.values[0] for i in allpop])]
        global winner
        if type(winner) is not creator.Individual:
            winner = candidate_winner
        winner = winner if winner.fitness > candidate_winner.fitness else candidate_winner
        print(candidate_winner.fitness)
        print(f'gen: {g}, record: {record}')

        '''
        #env.update_solutions(pop)

        candidate_winner = pop[np.argmax([i.fitness for i in pop])]
        global winner
        if type(winner) is not creator.Individual:
            winner = candidate_winner
        winner = winner if winner.fitness > candidate_winner.fitness else candidate_winner
        print(g, record)
        '''

main()
env.update_parameter("speed", "normal")
env.update_parameter("visuals", True)
env.visuals = True
env.speed = "normal"
evaluate(env, winner)
