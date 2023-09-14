import os

import neat
import math
import visualize
from typing import Dict, List
from evoman.environment import Environment
from evoman.controller import Controller

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

class player_controller(Controller):
    def __init__(self, net=None):
        self.net = net
    def set(self,controller, n_inputs):
        self.net = controller.net
        pass
    def control(self, inputs, controller):
		# Normalises the input using min-max scaling
        if not self.net:
            return [0]*5
        inputs = (inputs-min(inputs))/float((max(inputs)-min(inputs)))
        output = self.net.activate(inputs)
        output_actions = list(map_to_action(output))
        return output_actions





# number of weights for multilayer with 10 hidden neurons


def simulate(env, cont):
    f,p,e,t = env.play(pcont=cont)
    return f

def eval_genome(genome):
    return simulate(env,genome)

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 100
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        c = player_controller(net)
        f = eval_genome(c)
        genome.fitness = f
        env.update_solutions(genome.connections.values())

def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 100)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    env.update_parameter("speed", "normal")
    env.update_parameter("visuals", True)
    env.visuals = True
    env.speed = "normal"

    #os.unsetenv("SDL_VIDEODRIVER")

    c = player_controller(winner_net)
    f = eval_genome(c)
    print('Run final')

    #visualize.plot_stats(stats, ylog=False, view=True)
    #visualize.plot_species(stats, view=True)

    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-49')
    #p.run(eval_genomes, 10)
    #env.save_state()
    return winner.fitness



if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    best_fit = None
    headless = False
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    env = Environment(experiment_name='test',
                    enemies=[1],
                    playermode="ai",
                    player_controller=player_controller(),
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)

    def fitness_search(enemy_h, p_h, time):
        return  0.8*(100-enemy_h) + p_h*0.2 - math.log(time)
        #return 100.01 + p_h - enemy_h
    env.state_to_log() # checks environment state
    run_mode = 'train' # train or test
    env.fitness_func = fitness_search
    fit = run(config_path)
        