from demo_controller import player_controller
from evoman.environment import Environment
import os
import numpy as np
from deap import base, tools
from deap_constants import *


class EA:
    def __init__(self, enemy, multimode=False, **kwargs):
        self.enemy=enemy
        self.env = None
        self.toolbox = base.Toolbox()
        self.generation = 0
        self.kwargs=kwargs
        self.init_DEAP()
        self.init_logging()
        multimode_str = 'yes' if multimode else 'no'
        self.init_EVOMAN(multimode=multimode_str)

    def init_logging(self):
        self.hof = tools.HallOfFame(1)
        self.stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)

    def evaluate(self, individual):
        X_np = np.array(individual)
        X_std = np.divide(X_np - np.min(X_np),  np.max(X_np) - np.min(X_np))
        X_scaled = 2 * X_std - 1
        f,p,e,t,d = self.env.play(pcont=X_scaled.tolist())
        return (f, p-e,d )


    def get_gen(self):
        return self.generation

    def init_DEAP():
        pass

    def feasible(self, individual):
        f, p, e, t,d = self.env.play(pcont=individual) 
        return e == 0

    def distance(self, individual):
        f, p, e, t, d = self.env.play(pcont=individual)
        return e

    def init_EVOMAN(self, multimode):
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        experiment_name='cellular__es'
        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name)
        if type(self.enemy) == int:
            self.enemy = [self.enemy]
        self.env = Environment(experiment_name=experiment_name,
                    enemies=self.enemy,
                    multiplemode=multimode,
                    playermode="ai",
                    player_controller=player_controller(H_NODES_LAYERS[0]),
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)
        self.toolbox.register('evaluate', self.evaluate)
        #self.toolbox.decorate("evaluate", tools.DeltaPenality(self.feasible, 5, self.distance))


    def run_n(self, print_each_gen=False):
        for g in range(self.kwargs.get('NGEN', NGEN)):
            record = self.run_cycle()
            if print_each_gen:
                print(f'gen: {g}, record: {record}')
            self.generation = g
            yield record

    def get_best(self):
        return self.hof[0]

    def run_cycle(self):
        pass
    def __repr__(self) -> str:
        return 'EA'
    def __str__(self):
        return 'EA'

    def extinction(self, percent=0.5):
        self.pop = sorted(self.pop, reverse=True)[:len(self.pop)*percent]

