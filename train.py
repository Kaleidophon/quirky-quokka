"""
Defining training functions and experiments for the project.
"""

# STD
import random

# EXT
import numpy as np
import torch
import torch.optim as optim
import gym
import torch.nn.functional as F

# PROJECT
from models import ReplayMemory, QNetwork
from plotting import plot_exp_performance, plot_exps_with_intervals
from analyze import test_difference, get_actual_returns, test_gaussian
from hyperparameters import HYPERPARAMETERS

# CONSTANTS
EPS = float(np.finfo(np.float32).eps)
ENVIRONMENTS = ['CartPole-v1', 'Acrobot-v1']


def get_epsilon(it):
    return 0.05 if it >= 1000 else - 0.00095 * it + 1


def select_action(model, state, epsilon):
    with torch.no_grad():
        action = model(torch.Tensor(state))
        return torch.argmax(action).item() if random.random() > epsilon else random.choice([0,1])


def compute_q_val(model, state, action):
    action_index = torch.stack(action.chunk(state.size(0)))
    return model(state).gather(1, action_index)


def compute_target(model, reward, next_state, done, discount_factor):
    targets = reward + (model(next_state).max(1)[0] * discount_factor) * (1-done.float())
    return targets.unsqueeze(1)


def train(model, memory, optimizer, batch_size, discount_factor, model_2):
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)

    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)

    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)
    done = torch.tensor(done, dtype=torch.uint8)  # Boolean

    # compute the q value
    q_val = compute_q_val(model, state, action)

    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        model_ = model_2 if model_2 is not None else model
        target = compute_target(model_, reward, next_state, done, discount_factor)

    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())


def run_episodes(train, model, memory, env, num_episodes, batch_size, discount_factor, learn_rate, model_2=None, update_target_q=10):
    optimizer = optim.Adam(model.parameters(), learn_rate)
    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []
    q_vals = []
    episode_rewards = []

    for i in range(num_episodes):
        steps = 0
        state = env.reset()
        episode_q_vals = []
        cum_reward = 0
        done = False
        while not done:
            steps += 1

            if model_2 is not None and steps % update_target_q == 0:
                model_2.load_state_dict(model.state_dict())

            eps = get_epsilon(global_steps)
            action = select_action(model, state, eps)

            q_val = compute_q_val(model, torch.tensor([state], dtype=torch.float), torch.tensor([action], dtype=torch.int64))
            episode_q_vals.append(q_val.detach().numpy().squeeze().tolist())

            next_state, reward, done, _ = env.step(action)
            cum_reward += reward

            memory.push((state, action, reward, next_state, done))
            state = next_state
            loss = train(model, memory, optimizer, batch_size, discount_factor, model_2)
        q_vals.append(np.mean(episode_q_vals))
        episode_durations.append(steps)
        global_steps += steps
        episode_rewards.append(cum_reward)

    return episode_durations, q_vals, episode_rewards


def run_single_dqn(env, memory_size, num_hidden, batch_size, discount_factor, learn_rate, **hyper):
    memory = ReplayMemory(memory_size)
    n_out = env.action_space.n
    n_in = len(env.observation_space.low)
    model = QNetwork(n_in, n_out, num_hidden)
    episode_durations, q_vals, cum_reward = run_episodes(train, model, memory, env, num_episodes, batch_size, discount_factor, learn_rate)

    return model, episode_durations, q_vals, cum_reward


def run_double_dqn(env, memory_size, num_hidden, batch_size, discount_factor, learn_rate, update_target_q):
    memory = ReplayMemory(memory_size)
    n_out = env.action_space.n
    n_in = len(env.observation_space.low)
    model = QNetwork(n_in, n_out, num_hidden)
    model_2 = QNetwork(n_in, n_out, num_hidden)

    episode_durations, q_vals, cum_reward = run_episodes(train, model, memory, env, num_episodes, batch_size, discount_factor, learn_rate, model_2)
    return model, episode_durations, q_vals, cum_reward


if __name__ == "__main__":
    # Let's run it!
    num_episodes = 100

    # init envs
    envs = {name: gym.envs.make(name) for name in ENVIRONMENTS}
    # collect experiments
    exps = [('Single DQN', run_single_dqn), ('Double DQN', run_double_dqn)]

    # train
    #exp_results = [([(exp(env), exp_name) for exp_name, exp in exps], env_name) for env_name, env in envs.items()]
    #plot_exp_performance(exp_results, path="./img")

    k = 10  # Number of models being trained
    env = envs["CartPole-v1"]
    hyperparams = HYPERPARAMETERS["CartPole-v1"]
    q_models, dq_models = [], []
    q_scores, q_durations, q_rewards = np.zeros((k, num_episodes)), np.zeros((k, num_episodes)), np.zeros((k, num_episodes))
    dq_scores, dq_durations, dq_rewards = np.zeros((k, num_episodes)), np.zeros((k, num_episodes)), np.zeros((k, num_episodes))

    for run in range(k):
        print(f"\rRun #{run+1}...", end="", flush=True)
        q_model, q_durations[run, :], q_scores[run, :], q_rewards[run, :] = run_single_dqn(env, **hyperparams)
        dq_model, dq_durations[run, :], dq_scores[run, :], dq_rewards[run, :] = run_single_dqn(env, **hyperparams)

        q_models.append(q_model)
        dq_models.append(dq_model)

    # Get true average q function values
    true_q = get_actual_returns(env, q_models, hyperparams["discount_factor"])
    true_dq = get_actual_returns(env, dq_models, hyperparams["discount_factor"])

    #for data in ["q_scores", "q_durations", "q_rewards", "dq_scores", "dq_durations", "dq_rewards"]:
    #    print(data)
    #    test_gaussian(eval(data))

    # Do significance-testing
    print("Q-values")
    _, significant_scores = test_difference(q_scores, dq_scores)
    print("Rewards")
    _, significant_rewards = test_difference(q_rewards, dq_rewards)
    print("Durations")
    _, significant_durations = test_difference(q_durations, dq_durations)

    plot_exps_with_intervals(
        q_scores, dq_scores, title="CartPole Q-Values", file_name="./img/qvalues.png",
        smooth_curves=False, true_q=true_q, true_dq=true_dq, significant_values=significant_scores
    )

    plot_exps_with_intervals(
        q_rewards, dq_rewards, title="CartPole Rewards", file_name="./img/rewards.png",
        smooth_curves=False, significant_values=significant_rewards
    )

    plot_exps_with_intervals(
        q_durations, dq_durations, title="CartPole Durations", file_name="./img/durations.png",
        smooth_curves=False, significant_values=significant_durations
    )



