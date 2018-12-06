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

        for episode_duration, exp_name in exp:
            plt.plot(smooth(episode_duration, 10), label=exp_name)
        plt.title('Episode durations per episode in ' + env_name)
        plt.legend()
        plt.savefig(f"{path}/{env_name}.png")


def plot_exps_with_intervals(q_data: np.array, dq_data: np.array, title, file_name, true_q: float=None,
                             true_dq: float=None, smooth_curves=False):
    """
    Plot scores with intervals. Expects a K x D matrix with K trials with D data points each.
    """
    def get_curves(data):
        median = np.median(data, axis=0)

        # Average of the two most extreme values to get upper / lower bounds
        upper = np.mean(np.sort(data, axis=0)[-2:, :], axis=0)
        lower = np.mean(np.sort(data, axis=0)[:2, :], axis=0)

        if smooth_curves:
            return smooth(lower, 10), smooth(median, 10), smooth(upper, 10)
        else:
            return lower, median, upper

    # Get the median, lower and upper curves
    q_lower, q_median, q_upper = get_curves(q_data)
    dq_lower, dq_median, dq_upper = get_curves(dq_data)

    # Plot everything
    x = np.arange(0, q_lower.shape[0])
    plt.plot(q_median, label="DQN", color="firebrick")
    plt.fill_between(x, q_upper, q_median, facecolor="lightcoral", alpha=0.6)
    plt.fill_between(x, q_median, q_lower, facecolor="lightcoral", alpha=0.6)

    plt.plot(dq_median, label="Double DQN", color="navy")
    plt.fill_between(x, dq_upper, dq_median, facecolor="lightsteelblue", alpha=0.6)
    plt.fill_between(x, dq_median, dq_lower, facecolor="lightsteelblue", alpha=0.6)

    # If the true values are given, plot them as a straight line
    if true_q is not None:
        plt.axhline(true_q, label="True DQN value", color="firebrick", linestyle='dashed')
    if true_dq is not None:
        plt.axhline(true_dq, label="True Double DQN value", color="lightsteelblue", linestyle='dashed')

    plt.title(title)
    plt.legend(fontsize=8)

    plt.savefig(file_name)
