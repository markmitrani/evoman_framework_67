import os
import optuna
import copy
from pstats import SortKey, Stats
import operator
import math
import numpy as np
from evoman.environment import Environment
from demo_controller import player_controller
from itertools import chain
from deap import base, creator, tools, algorithms, cma


pswarm_population = 250
pswarm_ngen = 20
ga_population = 250
ga_ngen = 200

min_value = -1
max_value = 1 
smin_value = -0.5
smax_value = 0.5



# GA params

def generate(size, pmin, pmax, smin, smax, gain, defeated, b_in_swarm, x=[]):
    if not x:
        part = creator.Particle(np.random.uniform(pmin, pmax) for _ in range(size))
    else:
        part = creator.Particle(x)
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
    return (f, p-e, d)

headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Particle", list, fitness=creator.FitnessMax, speed=list, 
smin=None, smax=None, best=None, gain=None, defeated=0, b_in_swarm=False)
creator.create("Individual", list, fitness=creator.FitnessMax, gain=None, deafeated=0)

toolbox = base.Toolbox()
toolbox.register("particle", generate, size=265, pmin=min_value, \
                    pmax=max_value, smin=smin_value, smax=smax_value, gain=None, defeated=0, b_in_swarm=False)

toolbox.register("attr_float", np.random.uniform, min_value, max_value)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=265)


toolbox.register("population_part", tools.initRepeat, list, toolbox.particle)
toolbox.register("population_ind", tools.initRepeat, list, toolbox.individual)

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
ga_pop = toolbox.population_ind(n=ga_population)


def pso_pop(pop, w, w_dec, num_swarms):
    # Define the global best particle (across all swarms)
    global_best = None
    particles_per_swarm = pswarm_population // num_swarms

    last_best = 0.0
    last_best_times = 0.0
    cnt = 0
    swarms = []
    for _ in range(num_swarms):
        # Create a swarm and initialize its particles
        swarm = []
        for i in range(particles_per_swarm):
            swarm.append(generate(265, min_value, max_value, smin_value, smax_value, 0, [], False, copy.deepcopy(list(pop[cnt]))))
            cnt += 1
        swarms.append(swarm)

     
    for g in range(pswarm_ngen):
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

        print(g, global_best.fitness, global_best.gain, global_best.defeated)
        if len(global_best.defeated) >= 7:
            filename = f'good_weights/mpsoga_12345678_{global_best.gain}_{global_best.defeated}.txt'
            np.savetxt(filename, X=list(global_best))

        if global_best.fitness.values[0] == last_best:
            last_best_times += 1
        if last_best_times == 5:
            last_best_times = 0
            for swarm in swarms:
                for part in swarm:
                    if np.random.rand() < 0.3:
                        part[:] = generate(265, min_value, max_value, smin_value, smax_value, None, 0, False)



def run_exp(trial):
    # PSwarm Params
    phi_1 = trial.suggest_float('PHI_1', 0, 3) 
    phi_2 = trial.suggest_float('PHI_2', 0, 3) 

    MUTPB = trial.suggest_float('MUTPB', 0, 1)
    CXPB = trial.suggest_float('CXPB', 0, 1)
    ALPHA = trial.suggest_float('ALPHA', 0, 1)
    #K = trial.suggest_int('K', 2, 40)
    W = trial.suggest_float('W', 0, 3) #1.0
    W_DEC = trial.suggest_float('W_DEC', 0.001, 0.01)
    NUM_SWARMS = trial.suggest_int('NUM_SWARSM', 1, 50)

    toolbox.register("mate", tools.cxBlend, alpha=ALPHA)
    toolbox.register("mutate", tools.cxTwoPoint)
    toolbox.register('select', tools.selNSGA2)
    toolbox.register("update", updateParticle, phi1=phi_1, phi2=phi_2)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    for ga_g in range(ga_ngen):
        # Select the next generation individuals
        offspring = toolbox.select(ga_pop, len(ga_pop))

        # Apply PSO to offspring
        offspring = pso_pop(offspring, W, W_DEC, NUM_SWARMS)

        # Clone the selected individuals
        offspring = map(toolbox.clone, offspring)

        # Apply crossover on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Apply mutation on the offspring
        for mutant in offspring:
            if np.random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            fitness, gain, defeated = fit
            ind.fitness.values = fitness
            ind.gain = gain
            ind.defeated = defeated



        # The population is entirely replaced by the offspring
        ga_pop[:] = toolbox.select(offspring+ga_pop)
        hof.update(ga_pop)
        if len(hof[0].defeated) >= 7:
            filename = f'good_weights/mpsoga_12345678_{hof[0].gain}_{hof[0].defeated}.txt'
            np.savetxt(filename, X=list(hof[0]))
        stats.compile(ga_pop)
        print(ga_g, stats)

    # Play the best controller found
    env.enemies = [i for i in range(1, 9)]
    f, p, e, t, defeated = env.play(pcont=hof[0])
    print(f'f, gain, defeated: {f, p-e, defeated}')
    return len(p-e)

def main():
    study = optuna.create_study(
        direction='maximize',
        storage="sqlite:///mpsoga_all_enemies_no_time",  # Specify the storage URL here.
        study_name="mpsoga_all_enemies_no_time",
        load_if_exists=True
    )
    study.optimize(run_exp, n_trials=100)
    print(study.best_params)
main()