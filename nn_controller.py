
import numpy as np
from typing import List
from evoman.controller import Controller
import tensorflow.compat.v1 as tf
#from tensorflow import keras
#from tensorflow.keras import layers
from deap_constants import map_to_action, ActivationFunctions

tf.disable_v2_behavior()
def activation_function_choose():
    return ActivationFunctions.sigmoid_activation


class keras_player_controller(Controller):
    input_layer = tf.keras.layers.Input(10)
    dense_output = tf.keras.layers.Dense(20, activation='sigmoid')(input_layer)
    output_layer = tf.keras.layers.Dense(5)(dense_output)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
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

        inputs = inputs.reshape(1, 10)
        model_weights = []
        #print(f'weights: {model.get_weights()[1]}')
        for i, w in enumerate(self.weights):
            #print(f' Weights: {np.array(w)}')
            #print(f' Bias: {np.array(self.bias[i])[0]}')
            model_weights.append(np.array(w))
            model_weights.append(np.array(self.bias[i])[0])
                
        self.model.set_weights(model_weights)


        '''
        output = inputs
        for w, b in zip(self.weights, self.bias):
            # Choose activation function #TODO Make it variable ?
            activation_func = activation_function_choose()
            # A = B X D + C
            output = activation_func(output.dot(w) + b)
        actions = output[0]
        '''
        actions = self.model.predict(x=inputs, verbose=0)
        return list(map_to_action(actions[0]))
    def optimize_(X, env):
        last_layer_num = 10
        last_slice = 0
        weights = []
        bias = []
        for layer_n in 20:
            # Get slice representing layer_n biases for each node output
            bias.append(X[last_slice:last_slice+layer_n].reshape(1, layer_n))
            # Now calcuate the amount of weights from previous layer to next layer for
            # fully connected topology
            weights_slice = last_slice + last_layer_num * layer_n + layer_n
            # Add weights
            weights.append(X[layer_n+last_slice:weights_slice].reshape(last_layer_num, layer_n))

            # Update variables
            last_slice = weights_slice
            last_layer_num = layer_n


        # Add weights for output layer from last hidden node layer
        bias.append(X[last_slice:last_slice+5].reshape(1, 5)) 
        weights.append(X[last_slice+5:].reshape(last_layer_num, 5))

        inputs = inputs.reshape(1, 10)
        model_weights = []
        #print(f'weights: {model.get_weights()[1]}')
        for i, w in enumerate(weights):
            #print(f' Weights: {np.array(w)}')
            #print(f' Bias: {np.array(self.bias[i])[0]}')
            model_weights.append(np.array(w))
            model_weights.append(np.array(bias[i])[0])

        opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        input_layer = tf.keras.layers.Input(10)
        dense_output = tf.keras.layers.Dense(20, activation='sigmoid')(input_layer)
        output_layer = tf.keras.layers.Dense(5)(dense_output)
        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        model.set_weights(model_weights)

        with tf.GradientTape() as tape:
            logits = model(X)
            print(logits)
            #loss_value = loss_fn(env, logits)

        #gradients = tape.gradient(loss_value, model.trainable_weights)
        #opt.apply_gradients(zip(gradients, model.trainable_weights))
        

        return model.get_weights().flatten()



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
            #ans.append(t[0])
            #ans.append(t[1])
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
