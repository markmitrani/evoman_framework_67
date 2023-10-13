import os
import copy
from cProfile import Profile
from pstats import SortKey, Stats
import operator
import math
import numpy as np
from evoman.environment import Environment
from demo_controller import player_controller
from itertools import chain
from deap import base, creator, tools, algorithms, cma

ngen = 5000
population_size = 550
min_value = -1
max_value = 1 
smin_value = -0.5
smax_value = 0.5

# DEAP Params
phi_1 = 0.7828674502864883
phi_2 = 1.7686860023244022

start_pop = []
for filename in os.listdir('good_weights'):
    x = np.loadtxt('good_weights/' + filename)
    start_pop.append(x)

# Define the number of swarms and particles per swarm
num_swarms = 5
particles_per_swarm = population_size // num_swarms

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
    v_u2 = u2 * (np.array(best) - np.array(part))
    part.speed = (w * np.array(part.speed) + v_u1 + v_u2).tolist()
    
    # Update particle position
    for i, speed in enumerate(part.speed):
        if np.abs(speed) < part.smin:
            part.speed[i] = math.copysign(part.smin, speed)
        elif np.abs(speed) > part.smax:
            part.speed[i] = math.copysign(part.smax, speed)
    
    # Update particle position based on the updated speed
    part[:] = (np.array(part) + np.array(part.speed)).tolist()

def evaluate(env, individual):
    #c = player_controller(H_NODES_LAYERS)
    #env.player_controller.set(individual, 20)
    f,p,e,t,d = env.play(pcont=individual)
    return (p-e, p-e, d)

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

enemies = [2, 3, 4, 5, 6, 7, 8]
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



# Create a list to hold all the swarms
swarms = []

for _ in range(num_swarms):
    # Create a swarm and initialize its particles
    if _ == 0:
        swarm = []
        for vec in start_pop:
            part = creator.Particle(vec)
            part.speed = np.random.uniform(-1, 1,size=265)
            part.smin = -1 
            part.smax = 1 
            part.gain = 0
            part.defeated = []
            part.b_in_swarm = False
            swarm.append(part)
    else:
        swarm = toolbox.population(n=particles_per_swarm)
    swarms.append(swarm)

# Define the global best particle (across all swarms)
global_best = None

# Set up parameters for Multi-Swarm PSO
w = 1.0
w_dec = 0.0035

for g in range(ngen):
    # Iterate through each swarm
    for swarm in swarms:
        for part in swarm:
            fitness, gain, defeated = toolbox.evaluate(individual=part)
            part.gain = gain
            part.defeated = defeated
            part.fitness.values = (fitness,)

            if not part.best or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.defeated = part.defeated
                part.best.gain = part.gain
                part.best.fitness.values = part.fitness.values

            if not global_best or global_best.fitness < part.fitness:
                global_best = creator.Particle(part)
                global_best.gain = part.gain
                global_best.defeated = part.defeated
                global_best.fitness.values = part.fitness.values

        for part in swarm:
            toolbox.update(part, global_best, w)

        w -= w_dec

        if len(global_best.defeated) >= 7:
            filename = f'good_weights/multi_pso_12345678_{global_best.gain}_{global_best.defeated}.txt'
            np.savetxt(filename)
            print(f'Saved weights to {filename}')

        print(f'Swarm {swarms.index(swarm)}, Generation {g}: Best Fitness: {global_best.fitness}, Gain: {global_best.gain}, Defeated: {global_best.defeated}')

        # Add communication between swarms by allowing some particles to migrate
        if len(swarms) > 1:
            for i, swarm in enumerate(swarms):
                left_swarm = swarms[(i - 1) % len(swarms)]
                right_swarm = swarms[(i + 1) % len(swarms)]

                for part in swarm:
                    if np.random.rand() < 0.1:
                        target_swarm = left_swarm if np.random.choice([True, False]) else right_swarm
                        target_part = target_swarm[np.random.choice([i for i in range(len(target_swarm))])]
                        part[:] = target_part[:]
    if g % 50 == 0:
        for swarm in swarms:
            for part in swarm:
                if np.random.rand() < 0.3:
                    part[:] = generate(265, min_value, max_value, smin_value, smax_value, None, 0, False)

    
# Play the best controller found
env.enemies = [i for i in range(1, 9)]
f, p, e, t, defeated = env.play(pcont=global_best)
print(f'f, gain, defeated: {f, p-e, defeated}')
