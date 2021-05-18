import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_utils import neighbor_smooth


def ignore_hidden_and_png(items):
    """Sometimes we get random hidden folders..."""
    return [item for item in items if item[0] != '.' and not item.endswith('png')]


def plot_all_runs(env_dir, plot_each):

    for algo_folder in ignore_hidden_and_png(os.listdir(env_dir)):

        run_folders = ignore_hidden_and_png(os.listdir(os.path.join(env_dir, algo_folder)))

        ep_rets_s = []

        for run_folder in run_folders:
            csv_path = os.path.join(env_dir, algo_folder, run_folder, 'progress.csv')

            df = pd.read_csv(csv_path)

            steps = df['timestep'].to_numpy()
            ep_rets = df['test_ep_ret'].to_numpy()
            ep_rets = neighbor_smooth(list(ep_rets), 11)

            if plot_each == 'True':
                plt.plot(steps, ep_rets, alpha=0.4, linestyle='--')

            ep_rets_s.append(ep_rets)

        ep_rets_s = np.array(ep_rets_s)
        mean_ep_ret = ep_rets_s.mean(axis=0)  # average across all seeds
        std_ep_ret = ep_rets_s.std(axis=0)

        plt.plot(steps, mean_ep_ret, label=f'{algo_folder} ({len(run_folders)} runs)')
        plt.fill_between(steps, mean_ep_ret-std_ep_ret, mean_ep_ret+std_ep_ret, alpha=0.2)
        plt.ylim(0, 7000)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--plot_each', type=str, required=False, default='False')
    args = parser.parse_args()

    env_dir = args.env

    plot_all_runs(args.env, args.plot_each)

    plt.title(args.env)
    plt.xlabel('Epoch')
    plt.ylabel('Test-time Return')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(f'{env_dir}/avg_return.png', dpi=200)
