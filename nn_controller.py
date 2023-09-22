
import numpy as np
from typing import List
from evoman.controller import Controller
from deap_constants import map_to_action, ActivationFunctions

def activation_function_choose():
    return ActivationFunctions.sigmoid_activation

class player_controller(Controller):
    def __init__(self, n_hidden: List):
        # Set of hidden layer, each item in list
        # is a hidden node
        self.n_hidden = n_hidden
        self.weights = list()
        self.bias = list()

    def set(self, cont, n_inputs):
        last_layer_num = 10
        last_slice = 0
        self.weights = []
        self.bias = []
        controller = np.array(cont)
        for layer_n in self.n_hidden:
            # Get slice representing layer_n biases for each node output
            self.bias.append(controller[last_slice:last_slice+layer_n].reshape(1, layer_n))
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
            ans.append(self.normalize_distance(t[0]))
            ans.append(self.normalize_distance(t[1]))
        return np.array(ans)

    def normalize_distance(self, d):
        if d == 0.:
            return d
        # 2^-(d/150)^2
        g_d = np.power(2, -np.power(d/150, 2))

        if d < 0:
            return -g_d
        return g_d


        
    def control(self, inputs, controller):
        # Normalises the input using min-max scaling ??? # TODO fix
        inputs = np.concatenate((inputs[:4], self.remove_unimportant_bulles(inputs)), axis=0)

        inputs[0] = self.normalize_distance(inputs[0])
        inputs[1] = self.normalize_distance(inputs[0])

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
