import random
import os
import numpy as np
from typing import Dict, List

from deap import base, creator, tools, algorithms

from evoman.environment import Environment
from evoman.controller import Controller
from deap_constants import ActivationFunctions, map_to_action, norm, CXPB, MUTPB, calculate_ind_size
from deap_algorithms import *


def evaluate(env, individual):
    #c = player_controller(H_NODES_LAYERS)
    #env.player_controller.set(individual, 20)
    f,p,e,t = env.play(pcont=individual)
    # Added the +10 because selection algorithms don't work on negative numbers / 0
    return (f+10, )

def activation_function_choose():
    return ActivationFunctions.sigmoid_activation

class player_controller(Controller):
    def __init__(self, n_hidden: List):
        # Set of hidden layer, each item in list
        # is a hidden node
        self.n_hidden = n_hidden
        self.weights = list()
        self.bias = list()

    def set(self, controller, n_inputs):
        last_layer_num = 10
        last_slice = 0
        self.weights = []
        self.bias = []
        for layer_n in self.n_hidden:
            # Get slice representing layer_n biases for each node output
            self.bias.append(controller[last_slice:layer_n].reshape(1, layer_n))
            # Now calcuate the amount of weights from previous layer to next layer for
            # fully connected topology
            weights_slice = last_slice + last_layer_num * layer_n + layer_n
            # Add weights
            self.weights.append(controller[layer_n+last_slice:weights_slice].reshape(last_layer_num, layer_n))

            # Update variables
            last_slice = weights_slice
            last_layer_num = layer_n


        # Add weights for output layer from last hidden node layer
        self.bias.append(controller[last_slice:last_slice+5].reshape(1, 5)) 
        self.weights.append(controller[last_slice+5:].reshape(last_layer_num, 5))

    def remove_unimportant_bulles(self, inputs):
        bullets = []

        for i in range(8):
            x = 4 + i * 2
            y = 4 + i * 2 + 1
            if inputs[x] == 0.0 and inputs[y] == 0.0:
                continue
            bullets.append((inputs[x], inputs[y]))
        closest_3 = sorted(bullets, key = lambda x: np.sqrt(np.power(x[0], 2) + np.power(x[1], 2)))[:3]

        while len(closest_3) != 3:
            closest_3.append((0., 0.))

        ans = []
        for t in closest_3:
            ans.append(t[0])
            ans.append(t[1])
        return np.array(ans)

        
    def control(self, inputs, controller):
        # Normalises the input using min-max scaling ??? # TODO fix
        inputs = np.concatenate((inputs[:4], self.remove_unimportant_bulles(inputs)), axis=0)
        inputs = np.array(inputs)
        inputs = (inputs-min(inputs)) / float((max(inputs) - min(inputs)))

        #
        if not self.n_hidden:
            return [0]*5

        output = inputs
        for w, b in zip(self.weights, self.bias):
            # Choose activation function #TODO Make it variable ?
            activation_func = activation_function_choose()
            # A = B X D + C
            output = activation_func(output.dot(w) + b)
        actions = output[0]
        return list(map_to_action(actions))

NGEN = 50
H_NODES_LAYERS = [20]
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
toolbox.register("select", tools.selRoulette)

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