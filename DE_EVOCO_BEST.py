'''DE/best/2/bin'''
import numpy as np
import os 
import random
from evoman.environment import Environment
from demo_controller import player_controller
from deap import base, creator, tools, algorithms

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def evaluate(env, individual):
    #c = player_controller(H_NODES_LAYERS)
    #env.player_controller.set(individual, 20)
    f,p,e,t,d = env.play(pcont=individual)
    return (f, )

dim = 265

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


# Create a Toolbox
toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, -1, 1)  # Define your parameter bounds
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=dim)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate, env)
toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Crossover operator

# Custom mutation function for 'DE/best/2/bin'
def mutation_best_2_bin(individual, indpb):
    mutant1 = toolbox.clone(individual)
    mutant2 = toolbox.clone(individual)
    
    best_individual = tools.selBest(population, k=1)[0]  # Select the best individual
    for i in range(len(individual)):
        if random.random() < indpb:
            mutant1[i] = best_individual[i] + random.uniform(-1, 1) * (individual[i] - best_individual[i])
            mutant2[i] = best_individual[i] + random.uniform(-1, 1) * (individual[i] - best_individual[i])
    
    return mutant1, mutant2

toolbox.register("mutate", mutation_best_2_bin, indpb=0.2)  # Mutation operator

toolbox.register("select", tools.selBest)  # Selection operator

pop_size = 100

# Initialize population
population = toolbox.population(n=pop_size)


stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)


# Run the DE algorithm
algorithms.eaMuPlusLambda(population, toolbox, mu=pop_size, lambda_=pop_size*2, cxpb=0.7, mutpb=0.2, ngen=100, stats=stats, halloffame=None, verbose=True)

# The evolved solutions are in the population
best_solution = tools.selBest(population, k=1)[0]

print(f'f, p, e, t, defeated: {env.play(pcont=best_solution)}')