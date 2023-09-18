import numpy as np
from typing import List

class ActivationFunctions:
    def sigmoid_activation(x):
        return 1./(1.+np.exp(-x))

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