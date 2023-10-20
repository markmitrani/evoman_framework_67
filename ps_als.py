import numpy as np
from cProfile import Profile
import copy
import os
import pandas as pd
from deap import base, creator, tools, algorithms, cma
from evoman.environment import Environment
from demo_controller import player_controller
import matplotlib.pyplot as plt

NGEN = 200
MIN_VALUE = -1
MAX_VALUE = 1 
POPULATION_SIZE = 250

C_1 = 1.9275832140007672
C_2 = 1.3673660004244517

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Particle", list, fitness=creator.FitnessMax, density=np.float64(),
                dist=np.float64(), best=None, gain=None, defeated=0)

enemies = [i for i in range(1, 9)]
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
experiment_name='cellular__es'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)
env = Environment(experiment_name=experiment_name,
            enemies=enemies,
            multiplemode='yes',
            playermode="ai",
            player_controller=player_controller(10),
            enemymode="static",
            level=2,
            speed="fastest",
            visuals=False)

def evaluate(env, individual):
    f,p,e,t,d = env.play(pcont=individual)
    return (p-e, p-e, d)

def manhattan_distance(ind_1, ind_2):
    dist = np.float64(0.0)     
    for x, y in zip(np.array(ind_1), np.array(ind_2)):
        dist += np.abs(x-y)
    return dist


def calculate_local_density(pop, d_c):
    for ind_1 in pop:
        density = np.float64(0.0)
        for ind_2 in pop:
            if ind_1 == ind_2:
                continue
            d_i_j = manhattan_distance(ind_1, ind_2)
            i_step = np.power(np.divide(d_i_j, d_c), 2)
            i_step = np.exp(-i_step)
            density += i_step
        ind_1.density = density

def calculate_dist(pop):
    min_dist = np.inf
    for i, ind in  enumerate(pop):
        if i == 0:
            max_dist = -np.inf
            for ind_2 in pop[1:]:
                dist = manhattan_distance(ind, ind_2)
                max_dist = max(max_dist, dist)
            ind.dist = max_dist
            continue
        min_dist = np.inf
        for ind_2 in pop[:i-1]:
            dist = manhattan_distance(ind, ind_2)
            min_dist = np.minimum(min_dist, dist)
        ind.dist = min_dist 
        

    for ind_2 in pop:
        if ind_2.density <= ind.density:
            continue
        dist = manhattan_distance(ind, ind_2)
        min_dist = min(min_dist, dist)

def ord_part_update(x, w, c_1, c_2, local_best):
    rand_1 = np.random.uniform(0, 1, size=len(x))
    rand_2 = np.random.uniform(0, 1, size=len(x))

    vu_1 = c_1 * rand_1 * (np.array(x.best) - np.array(x))
    vu_2 = c_2 * rand_2 * (np.array(local_best) - np.array(x))

    x[:] = (w * np.array(x) + vu_1 + vu_2).tolist()


def best_part_update(x, w, c_1, c_2, best_avg):
    rand_1 = np.random.uniform(0, 1, size=len(x))
    rand_2 = np.random.uniform(0, 1, size=len(x))

    vu_1 = c_1 * rand_1 * (np.array(x.best) - np.array(x))
    vu_2 = c_2 * rand_2 * (np.array(best_avg) - np.array(x))

    x[:] = (w * np.array(x) + vu_1 + vu_2).tolist()



    # 1 / C SUM (cg_best)

def generate(size, pmin, pmax,  gain, defeated, swarm, density, dist, best ,best_swarm):
    part = creator.Particle(np.random.uniform(pmin, pmax) for _ in range(size)) 
    part.gain = gain
    part.defeated = defeated
    part.swarm = swarm
    part.density = density
    part.dist = dist
    part.best = best
    part.best_swarm = best_swarm
    return part

toolbox = base.Toolbox()
toolbox.register("particle", generate, size=265, pmin=MIN_VALUE, \
                    pmax=MAX_VALUE, swarm=0, density=0, gain=None, defeated=0, dist=0,
                    best=None, best_swarm=False)

toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update_ord", ord_part_update, c_1 = C_1, c_2 = C_2)
toolbox.register("update_cb", best_part_update, c_1 = C_1, c_2 = C_2)
toolbox.register("update_cb", best_part_update, c_1 = C_1, c_2 = C_2)
toolbox.register('evaluate', evaluate, env=env)

pop = toolbox.population(n=POPULATION_SIZE)

w = 1.0
w_dec = 0.0035 
d_c = 100
n_subswarms = 5
best = None
for g in range(NGEN):
    calculate_local_density(pop, d_c)

    pop[:] = sorted(pop, key=lambda x: x.density, reverse=True)
    calculate_dist(pop)
    pop[:] = sorted(pop, key=lambda x: (x.dist, x.density), reverse=True)

    swarm_center_coords = []
    swarm_cnt = 0
    while swarm_cnt < n_subswarms:
        swarm_center = pop.pop(0)
        swarm_center.swarm = swarm_cnt
        swarm_center.best_swarm = True
        pop.append(swarm_center)
        swarm_center_coords.append((swarm_cnt, list(swarm_center)))

        swarm_cnt += 1



    #df = pd.DataFrame.from_dict({'density': [ind.density for ind in pop], 'distance': [ind.dist for i, ind in enumerate(pop)]})
    #df.plot()
    #plt.show()



    for part in pop:
        fitness, gain, defeated = toolbox.evaluate(individual=part)
        part.gain = gain
        part.defeated = defeated
        part.fitness.values = (fitness, )

        swarm_num_distance = [(swarm_num, manhattan_distance(center_coords, part)) for swarm_num, center_coords in swarm_center_coords]
        part.swarm = min(swarm_num_distance, key=lambda x: x[1])[0]

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

    cgBest = {}
    for i in range(n_subswarms):
        cgBest[i] = (-10000, list())

    for part in pop:
        if part.fitness.values[0] > cgBest[part.swarm][0]:
            cgBest[part.swarm] = (part.fitness.values[0], list(part))


    for part in pop:
        if part.best_swarm:
            mean_weights = np.sum([np.array(t[1]) for t in cgBest.values()])
            toolbox.update_cb(x=part, w=w, best_avg=1.0/n_subswarms * mean_weights)
        else:
            past_part = copy.deepcopy(part)
            toolbox.update_ord(x=part, w=w, local_best=cgBest[part.swarm][1])
    w -= w_dec

    if len(best.defeated) >= 7:
        np.savetxt(f'good_weights/pso_opt_12345678_{best.gain}_{best.defeated}.txt')
    print(g, best.fitness, best.gain, best.defeated)