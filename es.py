import random
import os
import numpy as np
import pandas as pd
import array
from typing import Any, Dict, List
from itertools import chain

from deap import base, creator, tools, algorithms

from evoman.environment import Environment
from deap_constants import CXPB, MUTPB, calculate_ind_size
from deap_algorithms import *
from nn_controller import player_controller
from graph import GridGraph, manhattan_distance, Node

headless = True

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

NGEN = 50
POPULATION_SIZE_GRID = 15
POPULATION_SIZE = 375
GRID_N_SIZE = 5
GRID_M_SIZE = 5
H_NODES_LAYERS = [10]
IND_SIZE = calculate_ind_size(H_NODES_LAYERS)
MIN_VALUE = -30
MAX_VALUE = 30
MIN_STRATEGY = -1
MAX_STRATEGY = 1

class CellularES:
    def __init__(self, enemy):
        self.enemy=enemy
        self.env = None
        self.toolbox = base.Toolbox()
        self.generation = 0
        self.winner = None
        self.init_DEAP()
        self.init_EVOMAN()

    def __repr__(self) -> str:
        return 'Cellular-ES'
    def __str__(self):
        return 'Cellular-ES'

    def get_gen(self):
        return self.generation

    def get_best(self):
        return self.winner

    def init_EVOMAN(self):
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        experiment_name='cellular__es'
        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name)
        self.env = Environment(experiment_name=experiment_name,
                    enemies=[self.enemy],
                    playermode="ai",
                    player_controller=player_controller(H_NODES_LAYERS),
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)
        self.toolbox.register('evaluate', self.evaluate)
        # Initialize grid graph with populations
        self.gg = GridGraph(GRID_N_SIZE, GRID_M_SIZE, pop_size=POPULATION_SIZE_GRID, toolbox=self.toolbox)

        self.gg.update_fitnesses(self.toolbox)

    def init_DEAP(self):
        creator.create('FitnessMax', base.Fitness, weights=(1.0,))
        creator.create('Individual', array.array, fitness=creator.FitnessMax, strategy=None, typecode='d')
        creator.create('Strategy', array.array, typecode='d')

        self.toolbox.register('attr_float', random.random)
        self.toolbox.register("individual", generateES, creator.Individual, creator.Strategy,
            IND_SIZE, MIN_VALUE, MAX_VALUE, MIN_STRATEGY, MAX_STRATEGY)
        self.toolbox.register('population', tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("mate", tools.cxESTwoPoint)
        self.toolbox.register('mutate', self_adaptive_correlated_mutation)
        #toolbox.register("mutate", tools.mutESLogNormal, c=0.2, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=4)


        self.stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)

    def evaluate(self, individual):
        #c = player_controller(H_NODES_LAYERS)
        #env.player_controller.set(individual, 20)
        f,p,e,t = self.env.play(pcont=individual)
        # Added the +10 because selection algorithms don't work on negative numbers / 0
        return (f, )
        
    def run_n(self):
        for g in range(NGEN):
            record = self.run_cycle()
            if g == 400:
                print(f'gen: {g}, record: {record}')
            self.generation = g
            yield record

    def run_cycle(self):
        new_gg: GridGraph = self.gg.deepcopy()
        for node in self.gg: 
            
            node: Node = node
            n_x, n_y = node.get_coords()
            clone_node: Node = new_gg[n_x][n_y]
            node_pop = node.get_pop()

            # Get the population of all the neighbouring nodes
            neighbour_pop = self.gg.get_knn_pop(n_x, n_y, manhattan_distance, max_dist=2)
            #print(neighbour_pop)
            neighbour_pop = self.toolbox.select(neighbour_pop, len(neighbour_pop))

            new_node_pop = []
            for ind_idx, ind in enumerate(node_pop):
                offspring = self.toolbox.clone(ind)
                change_ocurred = False

                # Crossover
                if random.random() < CXPB:
                    change_ocurred = True
                    partner = random.choice(neighbour_pop)
                    while offspring == partner:
                        partner = random.choice(neighbour_pop)

                    offspring_1, offspring_2 = self.toolbox.mate(offspring, partner)

                    offspring = tools.selBest([offspring_1, offspring_2], k=2)[0]

                    del offspring.fitness.values

                # Mutation
                if random.random() < MUTPB:
                    change_ocurred = True
                    offspring = self.toolbox.mutate(offspring)[0]
                    del offspring.fitness.values

                if not change_ocurred:
                    new_node_pop.append(offspring)
                    continue

                offspring.fitness.values = self.toolbox.evaluate(offspring)

                # Replacement
                if offspring.fitness.values[0] >= ind.fitness.values[0]:
                    new_node_pop.append(offspring)
                    #clone_node[ind_idx] = offspring
                else:
                    new_node_pop.append(self.toolbox.clone(ind))
                    #clone_node[ind_idx] = ind
            clone_node.set_pop(new_node_pop)

            #record = stats.compile(clone_node.get_popu())
            #print(f'gen: {g}, node: {node}, record: {record}')



        self.gg = new_gg
        allpop = self.gg.allpop()
        record = self.stats.compile(allpop)
        candidate_winner = allpop[np.argmax([i.fitness.values[0] for i in allpop])]

        if type(self.winner) is not creator.Individual:
            self.winner = candidate_winner

        self.winner = self.winner if self.winner.fitness > candidate_winner.fitness else candidate_winner
        return record

class ES:
    def __init__(self, enemy):
        self.enemy=enemy
        self.env = None
        self.toolbox = base.Toolbox()
        self.generation = 0
        self.winner = None
        self.init_DEAP()
        self.init_EVOMAN()

    def __repr__(self) -> str:
        return 'ES'

    def __str__(self) -> str:
        return 'ES'

    def get_gen(self):
        return self.generation

    def get_best(self):
        return self.winner

    def init_EVOMAN(self):
        headless = False
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        experiment_name='cellular__es'
        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name)
        self.env = Environment(experiment_name=experiment_name,
                    enemies=[self.enemy],
                    playermode="ai",
                    player_controller=player_controller(H_NODES_LAYERS),
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)
        self.toolbox.register('evaluate', self.evaluate)
        self.pop = self.toolbox.population(POPULATION_SIZE)

        # Constants for Mutation / Crossover
        fitnesses = map(self.toolbox.evaluate, self.pop)
        for ind, fit in zip(self.pop, fitnesses):
            ind.fitness.values = fit


    def init_DEAP(self):
        creator.create('FitnessMax', base.Fitness, weights=(1.0,))
        creator.create('Individual', array.array, fitness=creator.FitnessMax, strategy=None, typecode='d')
        creator.create('Strategy', array.array, typecode='d')

        self.toolbox.register('attr_float', random.random)
        self.toolbox.register("individual", generateES, creator.Individual, creator.Strategy,
            IND_SIZE, MIN_VALUE, MAX_VALUE, MIN_STRATEGY, MAX_STRATEGY)
        self.toolbox.register('population', tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("mate", tools.cxESTwoPoint)
        self.toolbox.register('mutate', self_adaptive_correlated_mutation)
        #toolbox.register("mutate", tools.mutESLogNormal, c=0.2, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=4)


        self.stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)

    def evaluate(self, individual):
        #c = player_controller(H_NODES_LAYERS)
        #env.player_controller.set(individual, 20)
        f,p,e,t = self.env.play(pcont=individual)
        # Added the +10 because selection algorithms don't work on negative numbers / 0
        return (f, )
        
    def run_n(self):
        for g in range(NGEN):
            record = self.run_cycle()
            if g == 400:
                print(f'gen: {g}, record: {record}')
            self.generation = g
            yield record

    def run_cycle(self):
        # Crossover and Mutation
        offspring = algorithms.varAnd(self.pop, self.toolbox, CXPB, MUTPB)

        # Evaluate fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replacement
        #pop[:] = offspring
        self.pop = self.toolbox.select(self.pop + offspring, len(self.pop))

        #TODO How to ?
        # TODO Update mutation parameters here

        #env.update_solutions(pop)
        record = self.stats.compile(self.pop)
        candidate_winner = self.pop[np.argmax([i.fitness for i in self.pop])]
        if type(self.winner) is not creator.Individual:
            self.winner = candidate_winner
        self.winner = self.winner if self.winner.fitness > candidate_winner.fitness else candidate_winner
        return record





