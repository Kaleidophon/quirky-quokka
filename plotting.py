"""
Define plotting functions here.
"""

# EXT
import numpy as np
import matplotlib.pyplot as plt


def smooth(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def plot_exp_performance(exps, path):
    for exp, env_name in exps:
        plt.figure()

        for res, exp_name in exp:
            episode_duration, cum_reward = res
            plt.plot(smooth(episode_duration, 10), label=exp_name +  "episode duration")
            #plt.plot(smooth(cum_reward, 10), label=exp_name + "cumulative reward")
        plt.title('Episode durations per episode in ' + env_name )
        plt.legend()
        plt.savefig(f"{path}/{env_name}.png")
