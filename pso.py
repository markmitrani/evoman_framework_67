from sklearn import preprocessing
from ea import EA
import random
import operator
from itertools import chain
import numpy as np
import math
from deap import creator, base, tools
from deap_constants import *

def generate(size, pmin, pmax, smin, smax, gain, defeated, b_in_swarm):
    part = creator.Particle(np.random.uniform(pmin, pmax) for _ in range(size)) 
    part.speed = [np.random.uniform(smin, smax) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    part.gain = gain
    part.defeated = defeated
    part.b_in_swarm = b_in_swarm
    return part

def updateParticle(part, best, w, phi1, phi2):
    u1 = (random.uniform(0, phi1) for _ in range(len(part)))
    u2 = (random.uniform(0, phi2) for _ in range(len(part)))
    v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
    v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
    part.speed = list(map(operator.mul, map(operator.add, part.speed, map(operator.add, v_u1, v_u2)), [w] * len(part)))
    for i, speed in enumerate(part.speed):
        if abs(speed) < part.smin:
            part.speed[i] = math.copysign(part.smin, speed)
        elif abs(speed) > part.smax:
            part.speed[i] = math.copysign(part.smax, speed)

    new_x = list(map(operator.add, part, part.speed))
    X_np = np.array(new_x)
    X_std = np.divide(X_np - np.min(X_np),  np.max(X_np) - np.min(X_np))
    X_scaled = 2 * X_std - 1
    part[:] = X_scaled.tolist()

class PSO(EA):
    def __repr__(self) -> str:
        return 'PSO'

    def __str__(self) -> str:
        return 'PSO'

    def get_best(self):
        return self.best

    def __init__(self,enemy, multimode=False, **kwargs):
        self.phi_1 = kwargs.get('PHI_1', PHI_1)
        self.phi_2 = kwargs.get('PHI_2', PHI_2)

        self.max_value = kwargs.get('MAX_VALUE', MAX_VALUE)
        self.min_value = kwargs.get('MIN_VALUE', MIN_VALUE)

        self.smax_value = kwargs.get('SMAX_VALUE', SMAX_VALUE)
        self.smin_value = kwargs.get('SMIN_VALUE', SMIN_VALUE)

        self.w = kwargs.get('W', W)
        #self.w_dec = kwargs.get('W_DEC', W_DEC)
        self.w_dec = 1 / kwargs.get('NGEN', NGEN) 

        self.population_size = kwargs.get('POPULATION_SIZE', POPULATION_SIZE)
        super().__init__(enemy, multimode, **kwargs)

        self.pop = self.toolbox.population(n=self.population_size)
        self.best = None


    def init_DEAP(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Particle", np.ndarray, fitness=creator.FitnessMax, speed=list, 
        smin=None, smax=None, best=None, gain=None, defeated=0, b_in_swarm=False)

        self.toolbox = base.Toolbox()
        self.toolbox.register("particle", generate, size=265, pmin=self.min_value, \
                            pmax=self.max_value, smin=self.smin_value, smax=self.smax_value, gain=None, defeated=0, b_in_swarm=False)

        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.particle)
        self.toolbox.register("update", updateParticle, phi1=self.phi_1, phi2=self.phi_2)

    def run_cycle(self):
        for part in self.pop:
            fitness, gain, defeated = self.toolbox.evaluate(individual=part)
            part.gain = gain
            part.defeated = defeated
            part.fitness.values = (fitness, )

            if  part.best is None or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.defeated = part.defeated
                part.best.gain = part.gain
                part.best.fitness.values = part.fitness.values

            if self.best is None or self.best.fitness < part.fitness:
                self.best = creator.Particle(part)
                self.best.gain = part.gain
                self.best.defeated = part.defeated
                self.best.fitness.values = part.fitness.values

        for part in self.pop:
            self.toolbox.update(part, self.best, self.w)

        self.w -= self.w_dec
        record = self.stats.compile(self.pop)
        return record

class MSPSO(EA):
    def __repr__(self) -> str:
        return 'MSPSO'

    def __str__(self) -> str:
        return 'MSPSO'

    def __init__(self, enemy, multimode=False, **kwargs):
        self.phi_1 = kwargs.get('PHI_1', PHI_1)
        self.phi_2 = kwargs.get('PHI_2', PHI_2)

        self.max_value = kwargs.get('MAX_VALUE', MAX_VALUE)
        self.min_value = kwargs.get('MIN_VALUE', MIN_VALUE)

        self.smax_value = kwargs.get('SMAX_VALUE', SMAX_VALUE)
        self.smin_value = kwargs.get('SMIN_VALUE', SMIN_VALUE)
        self.swarm_num = kwargs.get('SWARM_NUM', SWARM_NUM)

        self.w = kwargs.get('W', W)
        self.w_dec = 1 / kwargs.get('NGEN', NGEN) 
        #self.w_dec = kwargs.get('W_DEC', W_DEC)

        self.population_size = kwargs.get('POPULATION_SIZE', POPULATION_SIZE)
        super().__init__(enemy, multimode, **kwargs)

        self.swarms = []
        for i in range(self.swarm_num):
            swarm = self.toolbox.population(n=self.population_size // self.swarm_num)
            self.swarms.append(swarm)
        self.swarm_best = [None] * self.swarm_num
        self.best = None

    def init_DEAP(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Particle", np.ndarray, fitness=creator.FitnessMax, speed=list, 
        smin=None, smax=None, best=None, gain=None, defeated=0, b_in_swarm=False)

        self.toolbox = base.Toolbox()
        self.toolbox.register("particle", generate, size=265, pmin=self.min_value, \
                            pmax=self.max_value, smin=self.smin_value, smax=self.smax_value, gain=None, defeated=0, b_in_swarm=False)

        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.particle)
        self.toolbox.register("update", updateParticle, phi1=self.phi_1, phi2=self.phi_2)

    def get_best(self):
        return self.best

    def run_cycle(self):
        for nro, swarm in enumerate(self.swarms):
            for part in swarm:
                fitness, gain, defeated = self.toolbox.evaluate(individual=part)
                part.gain = gain
                part.defeated = defeated
                part.fitness.values = (fitness, )

                if part.best is None or part.best.fitness < part.fitness:
                    part.best = creator.Particle(part)
                    part.best.defeated = part.defeated
                    part.best.gain = part.gain
                    part.best.fitness.values = part.fitness.values

                if  self.swarm_best[nro] is None or self.swarm_best[nro].fitness < part.fitness:
                    self.swarm_best[nro] = creator.Particle(part)
                    self.swarm_best[nro].gain = part.gain
                    self.swarm_best[nro].defeated = part.defeated
                    self.swarm_best[nro].fitness.values = part.fitness.values

                if self.best is None or self.best.fitness < part.fitness:
                    self.best = creator.Particle(part)
                    self.best.gain = part.gain
                    self.best.defeated = part.defeated
                    self.best.fitness.values = part.fitness.values

            for part in swarm:
                self.toolbox.update(part, self.swarm_best[nro], self.w)

        self.w -= self.w_dec



        record = self.stats.compile(list(chain.from_iterable(self.swarms)))

        if self.generation % self.kwargs.get('R', R) == 0:
            total_pop = list(chain.from_iterable(self.swarms))
            np.random.shuffle(total_pop)
            self.swarms = []
            swarm_size = len(total_pop) // self.swarm_num
            last_pos = 0
            for i in range(self.swarm_num):
                slice_pop = total_pop[last_pos:last_pos+swarm_size]
                last_pos = last_pos+swarm_size
                self.swarms.append(slice_pop)
            self.swarm_best = [None] * self.swarm_num
        return record