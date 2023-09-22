import numpy as np
from typing import List

CXPB = 0.5
MUTPB = 0.2
INPUT_SIZE = 10

class ActivationFunctions:
    def sigmoid_activation(x):
        return 1./(1.+np.exp(-x))

    def sigmoid_prime(z, *args):
        sig = ActivationFunctions.sigmoid_activation(z)
        return sig * (1 - sig)

def map_to_action(output: List):
		# takes decisions about sprite actions
        map_o = {
              'left': 0,
              'right': 0,
              'jump' : 0,
              'shoot': 0,
              'release': 0
        }
        for i, k in enumerate(map_o):
            if output[i] > 0.5:
                map_o[k] = 1
        return map_o.values()

def norm(x, pfit_pop):
    if ( max(pfit_pop) - min(pfit_pop) ) > 0:
        x_norm = ( x - min(pfit_pop) )/( max(pfit_pop) - min(pfit_pop) )
    else:
        x_norm = 0

    if x_norm <= 0:
        x_norm = 0.0000000001
    return x_norm

def calculate_ind_size(layer_list: List):
    last_size = INPUT_SIZE # Input nodes
    ind_size = 0

    for layer_nodes in layer_list:
        ind_size +=  (layer_nodes * last_size + layer_nodes)
        last_size = layer_nodes

    # Output nodes taken into account
    return ind_size + (5 * last_size) + 5