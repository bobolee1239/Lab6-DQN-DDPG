'''DLP DQN Lab'''
__author__    = 'brian-th.lee'
__copyright__ = 'Copyright 2020, NCTU CGI Lab'
# -----------------------------------------------
import gym
import time
import torch
import random
import logging
import argparse
import itertools

import numpy    as np
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from collections import deque
from datetime    import datetime as dt

import pdb

logging.basicConfig(level=logging.DEBUG)
# -----------------------------------------------

class ReplayMemory:
    __slots__ = ['buffer']

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, *transition):
        # (state, action, reward, next_state, not_done)
        self.buffer.append(tuple(map(tuple, transition)))

    def sample(self, batch_size, device):
        '''sample a batch of transition tensors'''
        transitions = random.sample(self.buffer, batch_size)
        return (torch.tensor(x, dtype=torch.float, device=device)
                for x in zip(*transitions))


class Net(nn.Module):
    def __init__(self, state_dim=8, action_dim=4, hidden_dim=32):
        super(Net, self).__init__()

        self.act = nn.ReLU()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.adv = nn.Linear(hidden_dim, action_dim)
        self.var = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        '''
            - x : tensor in shape of (N, state_dim)
        '''
        h1 = self.fc1(x)
        h1 = self.act(h1)

        h2 = self.fc2(h1)
        h2 = self.act(h2)

        advantage = self.adv(h2)
        value     = self.var(h2)
        avg_adv   = torch.mean(advantage, dim=-1, keepdim=True)
        Q_value   = value + advantage - avg_adv

        return Q_value


class DQN:
    def __init__(self, args):
        self._behavior_net = Net().to(args.device)
        self._target_net   = Net().to(args.device)
        # -------------------------------------------
        # initialize target network
        # -------------------------------------------
        self._target_net.load_state_dict(self._behavior_net.state_dict())

        self._optimizer = torch.optim.RMSprop(
                            self._behavior_net.parameters(), 
                            lr=args.lr
                          )
        self._criteria  = nn.MSELoss()
        # memory
        self._memory    = ReplayMemory(capacity=args.capacity)

        ## config ##
        self.device      = args.device
        self.batch_size  = args.batch_size
        self.gamma       = args.gamma
        self.freq        = args.freq
        self.target_freq = args.target_freq

    def select_best_action(self, state):
        '''
            - state: (state_dim, )
        '''
        state  = torch.tensor(state).to(self.device)
        state  = DQN.reshape_input_state(state)
        qvars  = self._behavior_net(state)      # (1, act_dim)
        action = torch.argmax(qvars, dim=-1)    # (1, )

        return action.item()

    def select_action(self, state, epsilon, action_space):
        '''
        epsilon-greedy based on behavior network

            -state = (state_dim, )
        '''
        if random.random() < epsilon:
            return action_space.sample()
        else:
            return self.select_best_action(state)

    def append(self, state, action, reward, next_state, done):
        self._memory.append(
            state, 
            [action], 
            [reward / 10], 
            next_state,
            [int(not done)]
        )

    def update(self, total_steps):
        if total_steps % self.freq == 0:
            self._update_behavior_network(self.gamma)
        if total_steps % self.target_freq == 0:
            self._update_target_network()

    def _update_behavior_network(self, gamma):
        # sample a minibatch of transitions
        ret = self._memory.sample(self.batch_size, self.device)
        state, action, reward, next_state, tocont = ret 

        q_values      = self._behavior_net(state)                     # (N, act_dim)
        q_value, act  = torch.max(q_values, dim=-1, keepdim=True)     # (N, 1)

        with torch.no_grad():
           qs_next     = self._target_net(next_state)               # (N, act_dim)
           q_next, act = torch.max(qs_next, dim=-1, keepdim=True)   # (N, 1)
           q_target    = q_next*tocont + reward

        loss = self._criteria(q_value, q_target)

        # optimize
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._behavior_net.parameters(), 5)
        self._optimizer.step()

    def _update_target_network(self):
        '''
        update target network by copying from behavior network
        '''
        self._target_net.load_state_dict(self._behavior_net.state_dict())

    def save(self, model_path, checkpoint=False):
        if checkpoint:
            torch.save(
                {
                    'behavior_net': self._behavior_net.state_dict(),
                    'target_net': self._target_net.state_dict(),
                    'optimizer': self._optimizer.state_dict(),
                }, model_path)
        else:
            torch.save({
                'behavior_net': self._behavior_net.state_dict(),
            }, model_path)

    def load(self, model_path, checkpoint=False):
        model = torch.load(model_path)
        self._behavior_net.load_state_dict(model['behavior_net'])
        if checkpoint:
            self._target_net.load_state_dict(model['target_net'])
            self._optimizer.load_state_dict(model['optimizer'])
    
    @staticmethod
    def reshape_input_state(state):
        state_shape = len(state.shape)
        if state_shape == 1:
            state = state.unsqueeze(0)
        elif state_shape != 2:
            raise ValueError(f'Wrong state shape: {state_shape}')

        return state

# -------------------------------------------------------

def train(args, env_name, agent, writer):
    logging.info('* Start Training')

    env          = gym.make(env_name)
    action_space = env.action_space

    total_steps, epsilon, ewma_reward = 0, 1., 0.

    for episode in range(args.episode):
        total_reward = 0
        state        = env.reset()

        for t in itertools.count(start=1):
            if args.render and episode > 700:
                env.render()
                time.sleep(0.0082)
            # select action
            if total_steps < args.warmup:
                action = action_space.sample()
            else:
                action = agent.select_action(state, epsilon, action_space)
                epsilon = max(epsilon * args.eps_decay, args.eps_min)
            # execute action
            next_state, reward, done, _ = env.step(action)
            # store transition
            agent.append(state, action, reward, next_state, done)
            if total_steps >= args.warmup:
                agent.update(total_steps)

            state         = next_state
            total_reward += reward
            total_steps  += 1

            if done:
                ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward
                writer.add_scalar('Train/Episode Reward', total_reward,
                                  total_steps)
                writer.add_scalar('Train/Ewma Reward', ewma_reward,
                                  total_steps)
                logging.info(
                    '  - Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}\tEpsilon: {:.3f}'
                    .format(total_steps, episode, t, total_reward, ewma_reward,
                            epsilon))
                break
    env.close()


def test(args, env_name, agent, writer):
    logging.info('* Start Testing')
    env = gym.make(env_name)

    action_space = env.action_space
    epsilon      = args.test_epsilon
    seeds        = (args.seed + i for i in range(10))
    rewards      = []

    for n_episode, seed in enumerate(seeds):
        total_reward = 0
        env.seed(seed)
        state = env.reset()

        for t in itertools.count(start=1):
            env.render()
            time.sleep(0.03)
            #action = agent.select_action(state, epsilon, action_space)
            action = agent.select_best_action(state)
            # execute action
            next_state, reward, done, _ = env.step(action)

            state         = next_state
            total_reward += reward

            if done:
                writer.add_scalar('Test/Episode Reward', total_reward, n_episode)
                break
    logging.info('  - Average Reward', np.mean(rewards))
    env.close()


def main():
    ## arguments ##
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-m', '--model', default='dqn.pth')
    parser.add_argument('--logdir', default='log/dqn')
    # train
    parser.add_argument('--warmup', default=10000, type=int)
    parser.add_argument('--episode', default=1200, type=int)
    parser.add_argument('--capacity', default=10000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=.0005, type=float)
    parser.add_argument('--eps_decay', default=.99982, type=float)
    parser.add_argument('--eps_min', default=.01, type=float)
    parser.add_argument('--gamma', default=.99, type=float)
    parser.add_argument('--freq', default=4, type=int)
    parser.add_argument('--target_freq', default=10000, type=int)
    # test
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--seed', default=2021111, type=int)
    parser.add_argument('--test_epsilon', default=.001, type=float)
    args = parser.parse_args()

    ## main ##
    env_name = 'LunarLander-v2'
    agent = DQN(args)
    writer = SummaryWriter(f'log/DQN-{time.time()}')
    if not args.test_only:
        train(args, env_name, agent, writer)
        agent.save(args.model)
    agent.load(args.model)
    test(args, env_name, agent, writer)


if __name__ == '__main__':
    main()
