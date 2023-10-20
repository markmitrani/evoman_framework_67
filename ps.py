import os
from cProfile import Profile
from pstats import SortKey, Stats
import operator
import math
import numpy as np
from evoman.environment import Environment
from demo_controller import player_controller
from itertools import chain
from deap import base, creator, tools, algorithms, cma

ngen = 219
population_size = 5000
min_value = -1
max_value = 1 
smin_value = -0.5
smax_value = 0.5

# DEAP Params
phi_1 = 1.9275832140007672
phi_2 = 1.3673660004244517

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
    u1 = np.random.uniform(0, phi1, size=265)
    u2 = np.random.uniform(0, phi2, size=265)
    v_u1 = u1 * (np.array(part.best) - np.array(part))
    #v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
    v_u2 = u2 * (np.array(best) - np.array(part))
    #v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
    part.speed = (w * np.array(part.speed) + v_u1 + v_u2).tolist()
    #part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))
    for i, speed in enumerate(part.speed):
        if abs(speed) < part.smin:
            part.speed[i] = math.copysign(part.smin, speed)
        elif abs(speed) > part.smax:
            part.speed[i] = math.copysign(part.smax, speed)
    part[:] = (np.array(part) + np.array(part.speed)).tolist()

def evaluate(env, individual):
    #c = player_controller(H_NODES_LAYERS)
    #env.player_controller.set(individual, 20)
    f,p,e,t,d = env.play(pcont=individual)
    return (f, p-e, d)

headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Particle", list, fitness=creator.FitnessMax, speed=list, 
smin=None, smax=None, best=None, gain=None, defeated=0, b_in_swarm=False)

toolbox = base.Toolbox()
toolbox.register("particle", generate, size=265, pmin=min_value, \
                    pmax=max_value, smin=smin_value, smax=smax_value, gain=None, defeated=0, b_in_swarm=False)

toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle, phi1=phi_1, phi2=phi_2)

enemies = [1, 2, 3, 4, 5, 6, 7, 8]
env = Environment(experiment_name='cellular__es',
            enemies=enemies,
            multiplemode='yes',
            playermode="ai",
            player_controller=player_controller(10),
            enemymode="static",
            level=2,
            speed="fastest",
            visuals=False)
toolbox.register('evaluate', evaluate, env=env)
pop = toolbox.population(n=population_size)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

best = None

w = 1.0
w_dec = 0.0035 
for g in range(ngen):
    with Profile() as profile:
        for part in pop:

            fitness, gain, defeated = toolbox.evaluate(individual=part)
            part.gain = gain
            part.defeated = defeated
            part.fitness.values = (fitness, )

            if not part.best or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.defeated = part.defeated
                part.best.gain = part.gain
                part.best.fitness.values = part.fitness.values

            if not best or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.gain = part.gain
                best.defeated = part.defeated
                best.fitness.values = part.fitness.values

        for part in pop:
            toolbox.update(part, best, w)

        w -= w_dec
        
        if len(best.defeated) >= 7:
            np.savetxt(f'good_weights/pso_opt_12345678_{best.gain}_{best.defeated}.txt')
        print(g, best.fitness, best.gain, best.defeated)
        print(
         Stats(profile)
         .strip_dirs()
         .sort_stats(SortKey.CALLS)
         .print_stats()
        )

        # Gather all the fitnesses in one list and print the stats
        #jap_1yRPEXhCQWprtAjMjSF2ut2skjJTPcE127884

env.enemies = [i for i in range(1, 9)]
#es.env.update_parameter("speed", "normal")
#es.env.update_parameter("visuals", True)
#es.env.visuals = True
#es.env.speed = "normal"
f, p, e, t, defeated = env.play(pcont=best)
print(f'f, gain, defeated: {f, p-e, defeated}')