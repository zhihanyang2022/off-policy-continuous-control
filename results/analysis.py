import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, required=True)
args = parser.parse_args()

env_dir = args.env

for algo_folder in os.listdir(env_dir):

    ep_rets_s = []

    for seed_folder in os.listdir(os.path.join(env_dir, algo_folder)):

        csv_path = os.path.join(env_dir, algo_folder, seed_folder, 'progress.csv')

        df = pd.read_csv(csv_path)

        ep_rets = df['test_mean_ep_ret'].to_numpy()

        ep_rets_s.append(ep_rets)

    ep_rets_s = np.array(ep_rets_s)
    mean_ep_ret = ep_rets_s.mean(axis=0)  # average across all seeds

    plt.plot(np.arange(1, len(mean_ep_ret)+1), mean_ep_ret, label=algo_folder)

plt.title(args.env)
plt.xlabel('Epoch')
plt.ylabel('Return')
plt.legend()
plt.grid()
plt.show()


