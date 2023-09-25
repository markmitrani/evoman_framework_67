
import numpy as np
import random

def self_adaptive_correlated_mutation(ind):
    for i in range(len(ind.strategy)):
        ind.strategy[i] *= np.exp(random.gauss(0, 1))

    for i in range(len(ind.strategy)):
        ind[i] += random.gauss(0, ind.strategy[i])

    return ind,