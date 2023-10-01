import matplotlib.pyplot as plt
import pandas as pd


def plot_line_from_csv(filename):
    stats = pd.read_csv(filename)

    # Separate the stats dataframe into multiple dfs, grouping by enemy.
    split_stats = split_by_enemy_num(stats)

    for df in split_stats:
        enemy_num = stats['enemy'].iloc[0]

        # Sort each df by generation, ascending
        df.sort_values(by='gen', ascending=True, inplace=True)


        cellular_es = df[df['method'] == 'Cellular-ES']
        standard_es = df[df['method'] == 'ES']


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

        fig, ax = plt.subplots()

        plt.title(f"Task I: Enemy {enemy_num}")

        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.plot('gen', 'metric_max', color='orange', label='Cellular ES max', data=cellular_es_grouped)
        plt.plot('gen', 'metric_max', color='red', label='ES max', data=standard_es_grouped)

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


def split_by_enemy_num(stats):
    # Get unique enemy numbers
    unique_enemies = stats['enemy'].unique()

    # Split the dataframe by each unique enemy number
    list_of_dfs = [stats[ stats['enemy'] == enemy ] for enemy in unique_enemies]

    return list_of_dfs

def main():
    plot_line_from_csv('statistics.csv')

if __name__ == main():
    main()