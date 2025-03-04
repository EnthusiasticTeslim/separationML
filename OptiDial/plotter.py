import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def visualize_sensitivity_data(
            SR_list: np.array, 
            EC_list: np.array,
            var_list: list,
            colors: list,
              var_name=r'$\rm Cell \ Voltage \ (V)$',
              figsize=(10, 5)):
    """ plot the sensitivity analysis results 
    Parameters
    ----------
      SR_list: np.array, list of separation efficiency
      EC_list: np.array, list of energy consumption
      var_list: list, list of variables
      colors: list, list of colors
      var_name: str, name of the variable
      figsize: tuple, size of the figure
    Returns
    -------
      fig: figure
    """

    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].bar(np.arange(len(var_list)), SR_list, color=colors)
    # ax[0].set_ylim([0, 1])
    ax[0].set_xticks(np.arange(len(var_list)))
    ax[0].set_xticklabels(var_list)
    ax[0].set_xlabel(var_name)
    ax[0].set_ylabel(r'$\rm SR$')

    ax[1].bar(np.arange(len(var_list)), EC_list, color=colors)
    ax[1].set_xticks(np.arange(len(var_list)))
    ax[1].set_xticklabels(var_list)
    ax[1].set_xlabel(var_name)
    ax[1].set_ylabel(r'$\rm EC \ (KWh/m^3)$')

    # add legend from the list: var_list
    for i, var in enumerate(var_list):
        ax[1].bar(0, 0, color=colors[i], label=f'{var}')

    ax[1].legend(fontsize=10)

    plt.tight_layout()

    return fig

def rolling_average(data, window_size):
    """
    Compute the rolling average of a dataset
    Parameters
    ----------
      data: list, list of values to smooth
      window_size: int, window size of the moving average
    Returns
    -------
      list: smoothed values
    """
    weight = np.ones(window_size) / window_size
    rolled_data = np.convolve(data, weight, mode='valid')
    return rolled_data


def box_plot(ppo, a2c, sac, reward_id=1, write_path='images/rl/reward-one'):
    """ Function to plot the boxplot of the results
    Parameters
    ----------
    ppo : pd.DataFrame
        Dataframe containing the results of PPO
    a2c : pd.DataFrame
        Dataframe containing the results of A2C
    sac : pd.DataFrame
        Dataframe containing the results of SAC
    """
    # copy the dataframes
    df_ppo = ppo.copy()
    df_a2c = a2c.copy()
    df_sac = sac.copy()

    # Combine datasets for 'SR' and 'EC' plotting
    df_ppo['Algorithm'] = 'PPO'
    df_a2c['Algorithm'] = 'A2C'
    df_sac['Algorithm'] = 'SAC'

    combined_df = pd.concat([df_ppo, df_a2c, df_sac])

    # Plot the boxplot of the results
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    # Define custom color palette
    custom_palette = {"PPO": "blue", "SAC": "green", "A2C": "red"}

    # Boxplot for 'SR' in ax[0]
    sns.boxplot(data=combined_df, x='Algorithm', y='SR', ax=ax[0], hue='Algorithm', dodge=False, palette=custom_palette)
    ax[0].set_ylabel(r'$\rm SR$', fontsize=14)
    ax[0].set_xlabel(r'$\rm RL\ algorithm$', fontsize=14)
    ax[0].set_ylim(ymax=1.0)

    # Boxplot for 'EC' in ax[1]
    sns.boxplot(data=combined_df, x='Algorithm', y='EC', ax=ax[1], hue='Algorithm', dodge=False, palette=custom_palette)
    ax[1].set_ylabel(r'$\rm EC \ (kWh/m^3)$', fontsize=14)
    ax[1].set_xlabel(r'$\rm RL\ algorithm$', fontsize=14)

    plt.tight_layout()
    plt.show()


    # save the figure
    fig.savefig(os.path.join(write_path, f'boxplot_reward_{reward_id}.png'), dpi=300, bbox_inches='tight')