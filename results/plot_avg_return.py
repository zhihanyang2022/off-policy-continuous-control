import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, required=True)
args = parser.parse_args()

env_dir = args.env


def ignore_hidden_and_png(items):
    """Sometimes we get random hidden folders..."""
    return [item for item in items if item[0] != '.' and not item.endswith('png')]


def smooth(scalars, retain_prop=0.90):
    current = scalars[0]
    smoothed_scalars = [current]
    for scalar in scalars[1:]:
        current = retain_prop * current + (1 - retain_prop) * scalar
        smoothed_scalars.append(current)
    return smoothed_scalars


for algo_folder in ignore_hidden_and_png(os.listdir(env_dir)):

    run_folders = ignore_hidden_and_png(os.listdir(os.path.join(env_dir, algo_folder)))

    ep_rets_s = []

    for run_folder in run_folders:
        csv_path = os.path.join(env_dir, algo_folder, run_folder, 'progress.csv')

        df = pd.read_csv(csv_path)

        ep_rets = df['test_ep_ret'].to_numpy()

        ep_rets_s.append(smooth(ep_rets))

    ep_rets_s = np.array(ep_rets_s)
    mean_ep_ret = ep_rets_s.mean(axis=0)  # average across all seeds
    sd_ep_ret = ep_rets_s.std(axis=0)

    plt.plot(np.arange(1, len(mean_ep_ret) + 1), mean_ep_ret, label=f'{algo_folder} ({len(run_folders)} runs)')
    plt.fill_between(np.arange(1, len(mean_ep_ret) + 1), mean_ep_ret - sd_ep_ret, mean_ep_ret + sd_ep_ret, alpha=0.2)

plt.title(args.env)
plt.xlabel('Epoch')
plt.ylabel('Test-time Return')
plt.legend(loc='lower right')
plt.grid()
plt.savefig(f'{env_dir}/avg_return.png', dpi=200)
