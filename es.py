import random
import os
import numpy as np
import pandas as pd
import array
from typing import Any, Dict, List
from itertools import chain

from deap import base, creator, tools, algorithms, cma

from evoman.environment import Environment
from ea import EA
from deap_constants import *
from deap_algorithms import *
#from nn_controller import player_controller
from demo_controller import player_controller
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


class CellularES(EA):
    def __init__(self, enemy, multimode=False, **kwargs):
        super().__init__(enemy, multimode, kwargs)
        GRID_N_SIZE = self.kwargs.get('GRID_N_SIZE', GRID_N_SIZE)
        GRID_M_SIZE = self.kwargs.get('GRID_M_SIZE', GRID_M_SIZE)
        POPULATION_SIZE_GRID = self.kwargs.get('POPULATION_SIZE_GRID', POPULATION_SIZE_GRID)
        self.gg = GridGraph(GRID_N_SIZE, GRID_M_SIZE, pop_size=POPULATION_SIZE_GRID, toolbox=self.toolbox)
        self.gg.update_fitnesses(self.toolbox)

    def __repr__(self) -> str:
        return 'Cellular-ES'
    def __str__(self):
        return 'Cellular-ES'

    def init_DEAP(self):
        creator.create('FitnessMax', base.Fitness, weights=(1.0,))
        creator.create('Individual', array.array, fitness=creator.FitnessMax, strategy=None, typecode='d')
        creator.create('Strategy', array.array, typecode='d')

        MIN_VALUE = self.kwargs.get('MIN_VALUE', MIN_VALUE)
        MAX_VALUE = self.kwargs.get('MAX_VALUE', MAX_VALUE)
        MIN_STRATEGY = self.kwargs.get('MIN_STRATEGY', MIN_STRATEGY)
        MAX_STRATEGY = self.kwargs.get('MAX_STRATEGY', MAX_STRATEGY)
        TOUR_SIZE = self.kwargs.get('TOUR_SIZE', TOUR_SIZE)
        self.toolbox.register('attr_float', random.random)
        self.toolbox.register("individual", generateES, creator.Individual, creator.Strategy,
            IND_SIZE, MIN_VALUE, MAX_VALUE, MIN_STRATEGY, MAX_STRATEGY)
        self.toolbox.register('population', tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("mate", tools.cxESTwoPoint)
        self.toolbox.register('mutate', self_adaptive_correlated_mutation)
        #toolbox.register("mutate", tools.mutESLogNormal, c=0.2, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=TOUR_SIZE)

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

                offspring.fitness.values = (self.toolbox.evaluate(offspring)[1], )

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
        self.hof.update(allpop)

        return record

class ES(EA):
    def __init__(self, enemy, multimode=False, **kwargs):
        super().__init__(enemy, multimode, **kwargs)
        self.population_size = self.kwargs.get('POPULATION_SIZE', POPULATION_SIZE)
        self.pop = self.toolbox.population(self.population_size)

        # Constants for Mutation / Crossover
        fitnesses = map(self.toolbox.evaluate, self.pop)
        for ind, fit in zip(self.pop, fitnesses):
            ind.fitness.values = (fit[1], )

    def __repr__(self) -> str:
        return 'ES'

    def __str__(self) -> str:
        return 'ES'

    def init_DEAP(self):
        creator.create('FitnessMax', base.Fitness, weights=(1.0,))
        creator.create('Individual', array.array, fitness=creator.FitnessMax, strategy=None, typecode='d')
        creator.create('Strategy', array.array, typecode='d')

        min_value = self.kwargs.get('MIN_VALUE', MIN_VALUE)
        max_value = self.kwargs.get('MAX_VALUE', MAX_VALUE)
        min_strategy = self.kwargs.get('MIN_STRATEGY', MIN_STRATEGY)
        max_strategy = self.kwargs.get('MAX_STRATEGY', MAX_STRATEGY)
        tour_size = self.kwargs.get('TOUR_SIZE', TOUR_SIZE)

        self.toolbox.register('attr_float', random.random)
        self.toolbox.register("individual", generateES, creator.Individual, creator.Strategy,
            IND_SIZE, min_value, max_value, min_strategy, max_strategy)
        self.toolbox.register('population', tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("mate", tools.cxESTwoPoint)
        #self.toolbox.register('mutate', tools.mutESLogNormal)
        self.toolbox.register("mutate", tools.mutESLogNormal, c=C, indpb=INDPB)
        self.toolbox.register("select", tools.selNSGA2)
        #self.toolbox.register("select", tools.selTournament, tournsize=tour_size)

    def run_cycle(self):
        parents = self.toolbox.select(self.pop, 3*self.population_size)
        cxpb = self.kwargs.get('CXPB', CXPB)
        mutpb = self.kwargs.get('MUTPB', MUTPB)

        # Crossover and Mutation
        offspring = algorithms.varAnd(parents, self.toolbox, cxpb, mutpb)

        # Evaluate fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit[1], )

        # Replacement
        #self.pop[:] = offspring
        possible_pop = self.pop+offspring

        self.pop[:] = sorted(possible_pop, key=lambda x: x.fitness, reverse=True)[:self.population_size]

        #self.pop = self.toolbox.select(self.pop + offspring, self.population_size)
        #self.pop = self.toolbox.select(self.pop + offspring, len(self.pop))

        self.hof.update(self.pop)
        record = self.stats.compile(self.pop)

        return record

class CMA(EA):
    def __init__(self, enemy, multimode=False):
        super().__init__(enemy, multimode)
        

    def __repr__(self) -> str:
        return 'CMA-ES'

    def __str__(self) -> str:
        return 'CMA-ES'

    def init_DEAP(self):
        creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
        creator.create('Individual', array.array, fitness=creator.FitnessMin, typecode='d')

        MIN_VALUE = self.kwargs.get('MIN_VALUE', MIN_VALUE)
        MAX_VALUE = self.kwargs.get('MAX_VALUE', MAX_VALUE)
        MIN_STRATEGY = self.kwargs.get('MIN_STRATEGY', MIN_STRATEGY)
        MAX_STRATEGY = self.kwargs.get('MAX_STRATEGY', MAX_STRATEGY)
        TOUR_SIZE = self.kwargs.get('TOUR_SIZE', TOUR_SIZE)

        strategy = cma.Strategy(centroid=np.random.uniform(MIN_VALUE, MAX_VALUE, IND_SIZE), \
                                sigma=np.random.uniform(MIN_STRATEGY, MAX_STRATEGY),\
                                lambda_=2*IND_SIZE)
        self.toolbox.register("generate", strategy.generate, creator.Individual)
        self.toolbox.register("update", strategy.update)

    def evaluate(self, individual):
        f,p,e,t,d = self.env.play(pcont=individual)
        return (-f, )

    def run_cycle(self):
        pop = self.toolbox.generate()
        fitnesses = [self.toolbox.evaluate(individual=ind) for ind in pop]
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = (fit[1], )
        self.hof.update(pop)
        record = self.stats.compile(pop)

        self.toolbox.update(pop)

        return record