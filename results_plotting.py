import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_fitness(generation_stats):
    std_avg_vec = np.array(np.std(generation_stats['mean']))
    std_max_vec = np.array(np.std(generation_stats['max']))

    avg_vec = np.array(generation_stats['mean'])
    max_vec = np.array(generation_stats['max'])
    print(std_avg_vec)

    lb = avg_vec - std_avg_vec
    up = avg_vec + std_avg_vec
    plt.fill_between([i+1 for i in range(50)], lb, up, alpha=0.3)

    lb = max_vec - std_max_vec
    up = max_vec + std_max_vec

    plt.title('Generation vs Fitness ')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.fill_between([i+1 for i in range(50)], lb, up, alpha=0.3)


    sns.lineplot(x=[i+1 for i in range(50)], y =generation_stats['mean'])
    sns.lineplot(x=[i+1 for i in range(50)], y =generation_stats['max'])
    plt.legend(title='Metrics', labels=['Mean', 'Max'])
    plt.show()

def plot_boxplots(gain_stats):
    avgs = np.array([])
    maxs = np.array([])
    enemies = []

    for k, v in gain_stats:
        avgs = np.concatenate([avgs, v['mean']])
        maxs = np.concatenate([maxs], v['max'])
        enemies += [str(k)] * len(v['mean'])

    df = pd.DataFrame.from_dict({'Mean': avgs, 'Max': maxs, 'Enemy': enemies})

    df = df.melt(id_vars=['Enemy'], var_name='metric', value_name='values')

    sns.boxplot(data=df, x='Enemy', y='values', hue='metric')
    plt.title('Enemy vs Gain ')
    plt.xlabel('Enemy')
    plt.ylabel('Gain')


    plt.legend()
    plt.show()