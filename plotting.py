import numpy as np
import scipy
from scipy.stats import ttest_ind
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_fitness():
    df = pd.read_csv('statistics.csv')
    for enemy in df['enemy'].unique():
        plt.title('Generation vs Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')

        df_enemy = df[df.enemy == enemy]

        std_avg_vec = np.array(np.std(df_enemy['metric_avg']))
        std_max_vec = np.array(np.std(df_enemy['metric_max']))

        avg_vec = np.array(df_enemy['metric_avg'])
        max_vec = np.array(df_enemy['metric_max'])
        print(std_avg_vec)

        lb = avg_vec - std_avg_vec
        up = avg_vec + std_avg_vec
        plt.fill_between([i+1 for i in range(50)], lb, up, alpha=0.3)

        lb = max_vec - std_max_vec
        up = max_vec + std_max_vec


        plt.fill_between([i+1 for i in range(50)], lb, up, alpha=0.3)


        sns.lineplot(x=[i+1 for i in range(50)], y =df['metric_avg'])
        sns.lineplot(x=[i+1 for i in range(50)], y =df['metric_max'])
        plt.legend(title='Metrics', labels=['Mean', 'Max'])
        plt.show()

def plot_line_from_csv():
    stats = pd.read_csv('statistics.csv')

    # Separate the stats dataframe into multiple dfs, grouping by enemy.

    for enemy_num in stats['enemy'].unique():
        df = stats[stats['enemy'] == enemy_num]
        # Sort each df by generation, ascending
        df.sort_values(by='gen', ascending=True, inplace=True)



        #f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        plt.title(f"Enemy {enemy_num}")
        df = df.rename(columns={'metric_max': 'Max', 'metric_avg': 'Avg'})
        cellular_es = df[df['method'] == 'Cellular-ES']
        standard_es = df[df['method'] == 'ES']

        max_val = max(max(df['Avg']), max(df['Max']))
        df_melt = pd.melt(df, id_vars=['gen', 'method'], value_vars=['Max', 'Avg'])
        df_melt = df_melt.rename(columns={'method': 'Algo', 'variable': 'Metric', 'gen': 'Gen', 'value': 'Fitness'})
        ax = sns.lineplot(data=df_melt, x='Gen', y='Fitness', hue='Algo', 
                        style='Metric', alpha=0.5, palette='hls', errorbar='sd',
                        linewidth=4)

        ax.set_ylim([0, 100])
        #ax.axhline(y = max_val, color = 'green', linestyle = '-.', alpha=0.8, label='Max Fitness')
        ax.set_yticks(range(0, 100+5, 5))
        ax.set_xticks(range(0, 50+10, 10))


        ax.set_xlabel('X')
        plt.legend()
        #sns.lineplot(data=df_melt, x='gen', y='value', hue='method',  style='variable')

        plt.xlabel("Generation")
        #plt.ylabel("Fitness")
        plt.savefig(f'fitness_img_{enemy_num}.png')
        plt.clf()
        #plt.show()


        '''
        # Group by 'generations' and calculate the desired metrics
        cellular_es_grouped = cellular_es.groupby('gen').agg(
            metric_max=pd.NamedAgg(column='metric_max', aggfunc='mean'),
            metric_max_std=pd.NamedAgg(column='metric_max', aggfunc='std'),
            metric_avg=pd.NamedAgg(column='metric_avg', aggfunc='mean'),
            metric_avg_std=pd.NamedAgg(column='metric_avg', aggfunc='std')
        ).reset_index()

        standard_es_grouped = standard_es.groupby('gen').agg(
            metric_max=pd.NamedAgg(column='metric_max', aggfunc='mean'),
            metric_max_std=pd.NamedAgg(column='metric_max', aggfunc='std'),
            metric_avg=pd.NamedAgg(column='metric_avg', aggfunc='mean'),
            metric_avg_std=pd.NamedAgg(column='metric_avg', aggfunc='std')
        ).reset_index()
        df_grouped = df.groupby(['gen', 'method']).agg(
            metric_max=pd.NamedAgg(column='metric_max', aggfunc='mean'),
            metric_max_std=pd.NamedAgg(column='metric_max', aggfunc='std'),
            metric_avg=pd.NamedAgg(column='metric_avg', aggfunc='mean'),
            metric_avg_std=pd.NamedAgg(column='metric_avg', aggfunc='std')
        ).reset_index()






        #plt.plot('gen', 'metric_max', color='orange', label='Cellular ES max', data=cellular_es_grouped)
        #plt.plot('gen', 'metric_max', color='red', label='ES max', data=standard_es_grouped)

        # Create the filled regions for the standard deviation using fill_between
        ax.fill_between(cellular_es_grouped['gen'], cellular_es_grouped['metric_max'] - cellular_es_grouped['metric_max_std'], \
                        cellular_es_grouped['metric_max'] + cellular_es_grouped['metric_max_std'], color='orange',
                        alpha=0.1)
        # Create the filled regions using fill_between
        ax.fill_between(standard_es_grouped['gen'],
                        standard_es_grouped['metric_max'] - standard_es_grouped['metric_max_std'], \
                        standard_es_grouped['metric_max'] + standard_es_grouped['metric_max_std'], color='red',
                        alpha=0.1)

        plt.plot('gen', 'metric_avg', color='blue', label='Cellular ES avg', data=cellular_es_grouped)
        plt.plot('gen', 'metric_avg', color='green', label='ES avg', data=standard_es_grouped)

        # Create the filled regions for the standard deviation using fill_between
        ax.fill_between(cellular_es_grouped['gen'], cellular_es_grouped['metric_avg'] - cellular_es_grouped['metric_avg_std'], \
                        cellular_es_grouped['metric_avg'] + cellular_es_grouped['metric_avg_std'], color='blue',
                        alpha=0.1)
        # Create the filled regions using fill_between
        ax.fill_between(standard_es_grouped['gen'],
                        standard_es_grouped['metric_avg'] - standard_es_grouped['metric_avg_std'], \
                        standard_es_grouped['metric_avg'] + standard_es_grouped['metric_avg_std'], color='green',
                        alpha=0.1)


    #    ax.fill_between(gen, standard_es_avg - standard_es_avg_std, standard_es_avg + standard_es_avg_std,
    #                    color='green', alpha=0.3)


        """
        plt.errorbar(cellular_es_grouped['gen'], cellular_es_grouped['metric_avg'],
                     yerr=cellular_es_grouped['metric_avg_std'], fmt='o', color='blue', capsize=5,
                     label='Cellular ES avg with Error')
        plt.errorbar(standard_es_grouped['gen'], standard_es_grouped['metric_avg'],
                     yerr=standard_es_grouped['metric_avg_std'], fmt='o', color='green', capsize=5,
                     label='ES avg with Error')
        """
        plt.legend()
        # plt.savefig(f"enemy{enemy_num}.png")
        plt.show()
        '''


def split_by_enemy_num(stats):
    # Get unique enemy numbers
    unique_enemies = stats['enemy'].unique()

    # Split the dataframe by each unique enemy number
    list_of_dfs = [stats[ stats['enemy'] == enemy ] for enemy in unique_enemies]

    return list_of_dfs


def plot_boxplots():
    df = pd.read_csv('gain_statistics.csv')

    #df = df.melt(id_vars=['enemy'], var_name='algorithm', value_name='gain')

    enemy_p_value = []  
    for enemy in df['enemy'].unique():
        alg_1 = df['gain'][df['enemy'] == enemy][df['algorithm'] == 'ES']
        alg_2 = df['gain'][df['enemy'] == enemy][df['algorithm'] == 'Cellular-ES']
        print(f'ES Mean enemy {enemy}: {np.mean(alg_1)}')
        print(f'ES STD enemy {enemy}: {np.std(alg_1)}')

        print(f'cES Mean enemy {enemy}: {np.mean(alg_2)}')
        print(f'cES STD enemy {enemy}: {np.std(alg_2)}')

        t = ttest_ind(alg_1, alg_2)
        print(t)
        enemy_p_value.append(t[1])

    b = sns.boxplot(data=df, x='enemy', y='gain', width=0.4, hue='algorithm', color='skyblue', linewidth=2, showfliers=True)
    #b = sns.stripplot(data=df, x='enemy', y='gain', color='crimson', alpha=0.1, linewidth=1)
    sns.despine(offset = 5, trim = True)
    #b.set_title(f'E3 = {enemy_p_value[0]:.3f}, E5={enemy_p_value[1]:.3f}, E7={enemy_p_value[2]:.3f}', fontsize=16)
    b.set_xlabel('Enemy', fontsize=14)
    b.set_ylabel('Gain', fontsize=14)
    b.text(0.81, 0.95, '*', fontsize=25, fontweight='bold', color='black', transform=b.transAxes)
    print(enemy_p_value)



    plt.legend(loc='lower right')
    plt.savefig('gain_img.png')
    #plt.show()
plot_line_from_csv()
#plot_fitness()
#plot_boxplots()