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
from gym.spaces import Box

# PROJECT
from analyze import get_actual_returns, discrete_to_continuous, SPLITS, load_data
from models import ReplayMemory, QNetwork
from hyperparameters import HYPERPARAMETERS
from plotting import plot_data_for_env

# CONSTANTS
EPS = float(np.finfo(np.float32).eps)
ENVIRONMENTS = ["MountainCar-v0", "CartPole-v1", "Pendulum-v0", "Acrobot-v1"]


def get_epsilon(it):
    return 0.05 if it >= 1000 else - 0.00095 * it + 1


def select_action(model, state, epsilon):
    with torch.no_grad():
        action = model(torch.Tensor(state))
        return torch.argmax(action).item() if random.random() > epsilon else random.choice([0,1])


def compute_q_val(model, state, action):
    q_val = model(state)
    q_val = q_val.gather(1, action.unsqueeze(1).view(-1, 1))
    return q_val


def compute_target(model, reward, next_state, done, discount_factor, target_net, double_dqn=False):

    if not double_dqn:
        targets = reward + (target_net(next_state).max(1)[0] * discount_factor) * (1-done.float())
    else:
        greedy_actions = model(next_state).argmax(1)
        target_q = target_net(next_state).gather(1, greedy_actions.view(-1, 1)).squeeze(1)
        targets = reward + target_q * discount_factor * (1-done.float())

    return targets.unsqueeze(1)


def train(model, memory, optimizer, batch_size, discount_factor, target_net, double_dqn):
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
    action = action.squeeze()

    # compute the q value
    q_val = compute_q_val(model, state, action)

    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_target(model, reward, next_state, done, discount_factor, target_net, double_dqn)

    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())


def run_episodes(train, model, memory, env, num_episodes, batch_size, discount_factor, learn_rate, target_net,
                 update_target_q=10, max_steps=1000, double_dqn=False):
    optimizer1 = optim.Adam(model.parameters(), learn_rate)
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

            eps = get_epsilon(global_steps)

            action = select_action(model, state, eps)

            if steps % update_target_q == 0:
                target_net.load_state_dict(model.state_dict())

            train(model, memory, optimizer1, batch_size, discount_factor, target_net, double_dqn)

            q_val = compute_q_val(model, torch.tensor([state], dtype=torch.float),
                                  torch.tensor([action], dtype=torch.int64))
            episode_q_vals.append(q_val.detach().numpy().squeeze().tolist())

            # only convert to continuous action when actually performing an action in the envs
            action_env = action

            if isinstance(env.action_space, Box):
                action_env = [discrete_to_continuous(action, env)]

            next_state, reward, done, _ = env.step(action_env)

            cum_reward += reward

            memory.push((state, action, reward, next_state, done))
            state = next_state

            if steps >= max_steps:
                done = True

        q_vals.append(np.mean(episode_q_vals))
        episode_durations.append(steps)
        global_steps += steps
        episode_rewards.append(cum_reward)

    return episode_durations, q_vals, episode_rewards


def run_dqn(env, num_episodes, memory_size, num_hidden, batch_size, discount_factor, learn_rate, update_target_q,
            max_steps, double_dqn=False):
    memory = ReplayMemory(memory_size)

    # continuous action space
    if isinstance(env.action_space, Box):
        dims = env.action_space.shape[0]
        n_out = SPLITS ** dims
    # discrete action space
    else:
        n_out = env.action_space.n

    n_in = len(env.observation_space.low)
    model = QNetwork(n_in, n_out, num_hidden)
    target_net = QNetwork(n_in, n_out, num_hidden)

    episode_durations, q_vals, cum_reward = run_episodes(
        train=train, model=model, memory=memory, env=env, num_episodes=num_episodes, batch_size=batch_size,
        discount_factor=discount_factor, learn_rate=learn_rate, target_net=target_net,
        update_target_q=update_target_q, max_steps=max_steps, double_dqn=double_dqn
    )
    return model, episode_durations, q_vals, cum_reward


def create_data_for_env(env_name, env, hyperparams, dqn_experiment, ddqn_experiment, num_episodes=100, k=10,
                        model_path=None, data_path=None):

    print(f"Running {k} experiments for {env_name}...")
    q_models, dq_models = [], []
    q_values, q_durations, q_rewards = np.zeros((k, num_episodes)), np.zeros((k, num_episodes)), np.zeros(
        (k, num_episodes))
    dq_values, dq_durations, dq_rewards = np.zeros((k, num_episodes)), np.zeros((k, num_episodes)), np.zeros(
        (k, num_episodes))

    for run in range(k):
        print(f"\rRun #{run+1}...", end="", flush=True)
        q_model, q_durations[run, :], q_values[run, :], q_rewards[run, :] = dqn_experiment(env, num_episodes, double_dqn=False, **hyperparams)
        dq_model, dq_durations[run, :], dq_values[run, :], dq_rewards[run, :] = ddqn_experiment(env, num_episodes, double_dqn=True, **hyperparams)

        q_models.append(q_model)
        dq_models.append(dq_model)

    # Save models
    for model_type, models in zip(["dqn", "ddqn"], [q_models, dq_models]):
        for i, model in enumerate(models):
            torch.save(model, f"{model_path}{env_name}_{model_type}{i}.pt")

    # Get true average q function values
    true_q = get_actual_returns(env, q_models, hyperparams["discount_factor"])
    true_dq = get_actual_returns(env, dq_models, hyperparams["discount_factor"])

    # Accumulate the data into single objects
    q_data, dq_data = {}, {}
    q_data["values"], q_data["rewards"], q_data["durations"], q_data["true"] = q_values, q_rewards, q_durations, true_q
    dq_data["values"], dq_data["rewards"], dq_data["durations"], dq_data["true"] = dq_values, dq_rewards, dq_durations, true_dq

    # Save data if path is given
    np.save(f"{data_path}{env_name}_q_data", q_data)
    np.save(f"{data_path}{env_name}_dq_data", dq_data)

    return q_data, dq_data


if __name__ == "__main__":

    # Init envs
    envs = {name: gym.envs.make(name) for name in ENVIRONMENTS}

    # Although hyperparameter search wasn't done with seeding (to avoid overfitting to a specific seed), it makes
    # sense to seed here to guarantee reproducability of the models and plots
    seed = 42  # This is not randomly chosen
    random.seed(seed)
    torch.manual_seed(seed)
    for env in envs.values():
        env.seed(seed)

    for env_name, env in envs.items():
        q_data, dq_data = create_data_for_env(
            env_name, env, HYPERPARAMETERS[env_name], k=2,  model_path="./models/", data_path="./data/",
            dqn_experiment=run_dqn, ddqn_experiment=run_dqn, num_episodes=100,
        )
        plot_data_for_env(env, q_data, dq_data, "./img/")

    # Example on how to load data
    # q_data, dq_data = load_data("./data/CartPole-v1_q_data.npy"), load_data("./data/CartPole-v1_dq_data.npy")
