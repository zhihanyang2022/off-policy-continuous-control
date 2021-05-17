import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_utils import eleven_neighbor_smooth


def ignore_hidden_and_png(items):
    """Sometimes we get random hidden folders..."""
    return [item for item in items if item[0] != '.' and not item.endswith('png')]


def plot_all_runs(env_dir, xs=None):

    for algo_folder in ignore_hidden_and_png(os.listdir(env_dir)):

        run_folders = ignore_hidden_and_png(os.listdir(os.path.join(env_dir, algo_folder)))

        ep_rets_s = []

        for run_folder in run_folders:
            csv_path = os.path.join(env_dir, algo_folder, run_folder, 'progress.csv')

            df = pd.read_csv(csv_path)

            ep_rets = df['test_ep_ret'].to_numpy()

            ep_rets_s.append(eleven_neighbor_smooth(list(ep_rets), 5))

        ep_rets_s = np.array(ep_rets_s)
        mean_ep_ret = ep_rets_s.mean(axis=0)  # average across all seeds
        max_ep_ret = ep_rets_s.max(axis=0)
        min_ep_ret = ep_rets_s.min(axis=0)

        if xs is None:
            xs = np.arange(1, len(mean_ep_ret) + 1)

        plt.plot(xs, mean_ep_ret, label=f'{algo_folder} ({len(run_folders)} runs)')
        plt.fill_between(xs, min_ep_ret, max_ep_ret, alpha=0.2)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=True)
    args = parser.parse_args()

    env_dir = args.env

    plot_all_runs(args.env)

    plt.title(args.env)
    plt.xlabel('Epoch')
    plt.ylabel('Test-time Return')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(f'{env_dir}/avg_return.png', dpi=200)
