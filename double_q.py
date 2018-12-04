import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm as _tqdm
import random
import copy
import gym


def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer

EPS = float(np.finfo(np.float32).eps)



class QNetwork(nn.Module):

    def __init__(self,n_in, n_out, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(n_in, num_hidden)
        self.l2 = nn.Linear(num_hidden, n_out)

    def forward(self, x):
        out = self.l1(x)
        out = F.relu(out)
        out = self.l2(out)
        return out



class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        if self.capacity == len(self.memory):
            self.memory.pop(0)
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def get_epsilon(it):
    return 0.05 if it >= 1000 else - 0.00095 * it + 1


def select_action(model, state, epsilon):
    with torch.no_grad():
        action = model(torch.Tensor(state))
        return torch.argmax(action).item() if random.random() > epsilon else random.choice([0,1])


def compute_q_val(model, state, action):
    action_index = torch.stack(action.chunk(state.size(0)))
    return model(state).gather(1 ,action_index)

def compute_target(model, reward, next_state, done, discount_factor):
    targets = reward + ( model(next_state).max(1)[0] * discount_factor )  * (1-done.float())
    return targets.unsqueeze(1)

def train(model, memory, optimizer, batch_size, discount_factor, model_2):
    # DO NOT MODIFY THIS FUNCTION

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
    global_steps = 0 # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = [] #
    for i in range(num_episodes):
        steps = 0
        state = env.reset()
        done = False
        while not done:
            steps += 1
            if model_2 is not None and steps % update_target_q == 0:
                model_2 = copy.deepcopy(model)
            eps = get_epsilon(global_steps)
            action = select_action(model, state, eps)
            next_state, reward, done, _ = env.step(action)
            memory.push((state, action, reward, next_state, done))
            state = next_state
            loss = train(model, memory, optimizer, batch_size, discount_factor, model_2)

        episode_durations.append(steps)
        global_steps += steps
    return episode_durations


# And see the results
def smooth(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def plot_exp_performance(exps):
    for exp, env_name in exps:
        plt.figure()
        for episode_duration, exp_name in exp:
            plt.plot(smooth(episode_duration, 10), label=exp_name)
        plt.title('Episode durations per episode in ' + env_name )
        plt.legend()
        plt.savefig(env_name +'.png')



def run_single_dqn(env):
    memory = ReplayMemory(memory_size)
    n_out = env.action_space.n
    n_in = len(env.observation_space.low)
    model = QNetwork(n_in, n_out, num_hidden)
    episode_durations = run_episodes(train, model, memory, env, num_episodes, batch_size, discount_factor, learn_rate)
    return episode_durations

def run_double_dqn(env):
    memory = ReplayMemory(memory_size)
    n_out = env.action_space.n
    n_in  = len(env.observation_space.low)
    model = QNetwork(n_in, n_out, num_hidden)
    model_2 = QNetwork(n_in, n_out, num_hidden)

    episode_durations = run_episodes(train, model, memory, env, num_episodes, batch_size, discount_factor, learn_rate, model_2)
    return episode_durations


# Let's run it!
num_episodes = 100
batch_size = 64
discount_factor = 0.8
learn_rate = 1e-3
num_hidden = 128
seed = 42  # This is not randomly chosen
memory_size = 10000
# We will seed the algorithm (before initializing QNetwork!) for reproducability
random.seed(seed)
torch.manual_seed(seed)

# env that will be used
env_names = ['CartPole-v1','Acrobot-v1']
# init envs
envs = [gym.envs.make(name) for name in env_names]
# collect experiments
exps = [('Single DQN', run_single_dqn), ('Double DQN', run_double_dqn)]

# seed envs
[env.seed(seed) for env in envs]
# train
exp_results =  [([(exp(env), exp_name) for exp_name, exp in exps], env_name) for env, env_name in zip(envs, env_names) ]
plot_exp_performance(exp_results)
