'''DLP DDPG Lab'''
__author__ = 'brian-th.lee'
__copyright__ = 'Copyright 2020, NCTU CGI Lab'
# --------------------------------------------------
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

# --------------------------------------------------------

class GaussianNoise:
    def __init__(self, dim, mu=None, std=None):
        self.mu = mu if mu else np.zeros(dim)
        self.std = std if std else np.ones(dim) * .1

    def sample(self):
        return np.random.normal(self.mu, self.std)


class ReplayMemory:
    __slots__ = ['buffer']

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, *transition):
        # (state, action, reward, next_state, done)
        self.buffer.append(tuple(map(tuple, transition)))

    def sample(self, batch_size, device):
        '''sample a batch of transition tensors'''
        transitions = random.sample(self.buffer, batch_size)
        return (torch.tensor(x, dtype=torch.float, device=device)
                for x in zip(*transitions))


class ActorNet(nn.Module):
    def __init__(self, state_dim=8, action_dim=2, hidden_dim=(400, 300)):
        super(ActorNet, self).__init__()
        h1, h2 = hidden_dim

        self.relu = nn.ReLU()
        self.fc1  = nn.Linear(state_dim, h1)
        self.fc2  = nn.Linear(h1       , h2)

        self.output = nn.Sequential(
                        nn.Linear(h2, action_dim),
                        nn.Tanh()
                      )

    def forward(self, x):
        '''
            - x (N, state_dim)
            - out (N, 2)
        '''
        hidden1 = self.fc1(x)
        hidden1 = self.relu(hidden1)

        hidden2 = self.fc2(hidden1)
        hidden2 = self.relu(hidden2)

        out = self.output(hidden2)
        
        return out


class CriticNet(nn.Module):
    def __init__(self, state_dim=8, action_dim=2, hidden_dim=(400, 300)):
        super().__init__()
        h1, h2 = hidden_dim
        self.critic_head = nn.Sequential(
            nn.Linear(state_dim, h1),
            nn.ReLU(),
        )
        self.critic = nn.Sequential(
            nn.Linear(h1 + action_dim, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def forward(self, x, action):
        x = self.critic_head(x)
        return self.critic(torch.cat([x, action], dim=1))


class DDPG:
    def __init__(self, args):
        # behavior network
        self._actor_net  = ActorNet().to(args.device)
        self._critic_net = CriticNet().to(args.device)
        # target network
        self._target_actor_net  = ActorNet().to(args.device)
        self._target_critic_net = CriticNet().to(args.device)
        # initialize target network
        self._target_actor_net.load_state_dict(self._actor_net.state_dict())
        self._target_critic_net.load_state_dict(self._critic_net.state_dict())
        ## OPTIMIZER ##
        self._actor_opt  = torch.optim.RMSprop(self._actor_net.parameters() , lr=args.lra)
        self._critic_opt = torch.optim.RMSprop(self._critic_net.parameters(), lr=args.lrc)
        self.mse_loss    = nn.MSELoss()
        # action noise
        self._action_noise  = GaussianNoise(dim=2, mu=0.0, std=1.0)
        self.nse_energy     = 1.0
        self.min_nse_energy = 0.1
        # memory
        self._memory = ReplayMemory(capacity=args.capacity)

        ## config ##
        self.device     = args.device
        self.batch_size = args.batch_size
        self.tau        = args.tau
        self.gamma      = args.gamma

    def eval(self):
        self._actor_net.eval()
        self._critic_net.eval()

    def train(self):
        self._actor_net.train()
        self._critic_net.train()

    def select_action(self, state, noise=True):
        '''based on the behavior (actor) network and exploration noise'''
        during_train = self._actor_net.training
        if during_train:
            self.eval()
        state  = torch.Tensor(state).to(self.device)
        state  = DDPG.reshape_input_state(state)

        with torch.no_grad():
            action_est = self._actor_net(state)
            nse        = self._action_noise.sample() * self.nse_energy
            action     = action_est.cpu().numpy() + nse

        self.nse_energy = min(self.min_nse_energy, self.nse_energy - 0.0001)

        if during_train:
            self.train()

        return action


    def append(self, state, action, reward, next_state, done):
        self._memory.append(state, action, [reward / 100], next_state,
                            [int(not done)])

    def update(self):
        # update the behavior networks
        self._update_behavior_network(self.gamma)
        # update the target networks
        self._update_target_network(self._target_actor_net, self._actor_net,
                                    self.tau)
        self._update_target_network(self._target_critic_net, self._critic_net,
                                    self.tau)

    def _update_behavior_network(self, gamma):
        actor_net, critic_net, target_actor_net, target_critic_net = self._actor_net, self._critic_net, self._target_actor_net, self._target_critic_net
        actor_opt, critic_opt = self._actor_opt, self._critic_opt

        # sample a minibatch of transitions
        state, action, reward, next_state, tocont = self._memory.sample(
            self.batch_size, self.device)

        ## update critic ##
        # critic loss
        q_value = critic_net(state, action)                 # (N, 1)

        with torch.no_grad():
           a_next = target_actor_net(next_state)            # (N, act_dim)
           q_next = target_critic_net(next_state, a_next)   # (N, 1)
           q_target = q_next*gamma*tocont + reward

        criterion   = self.mse_loss
        critic_loss = criterion(q_value, q_target)

        # optimize critic
        actor_net.zero_grad()
        critic_net.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        ## update actor ##
        # actor loss
        action     = actor_net(state)
        actor_loss = -critic_net(state, action).mean()
        # optimize actor
        actor_net.zero_grad()
        critic_net.zero_grad()
        actor_loss.backward()
        actor_opt.step()

    def save(self, model_path, checkpoint=False):
        if checkpoint:
            torch.save(
                {
                    'actor': self._actor_net.state_dict(),
                    'critic': self._critic_net.state_dict(),
                    'target_actor': self._target_actor_net.state_dict(),
                    'target_critic': self._target_critic_net.state_dict(),
                    'actor_opt': self._actor_opt.state_dict(),
                    'critic_opt': self._critic_opt.state_dict(),
                }, model_path)
        else:
            torch.save(
                {
                    'actor': self._actor_net.state_dict(),
                    'critic': self._critic_net.state_dict(),
                }, model_path)

    def load(self, model_path, checkpoint=False):
        model = torch.load(model_path)
        self._actor_net.load_state_dict(model['actor'])
        self._critic_net.load_state_dict(model['critic'])
        if checkpoint:
            self._target_actor_net.load_state_dict(model['target_actor'])
            self._target_critic_net.load_state_dict(model['target_critic'])
            self._actor_opt.load_state_dict(model['actor_opt'])
            self._critic_opt.load_state_dict(model['critic_opt'])

    @staticmethod
    def _update_target_network(target_net, net, tau):
        '''update target network by _soft_ copying from behavior network'''
        for target, behavior in zip(target_net.parameters(), net.parameters()):
            target.data.copy_(
                target.data*(1-tau) + behavior.data*tau
            )

    @staticmethod
    def reshape_input_state(state):
        state_shape = len(state.shape)
        if state_shape == 1:
            state = state.unsqueeze(0)
        elif state_shape != 2:
            raise ValueError(f'Wrong state shape: {state_shape}')

        return state


def train(args, env_name, agent, writer):
    print('Start Training')
    env = gym.make(env_name)
    total_steps, ewma_reward = 0, 0.
    for episode in range(args.episode):
        total_reward = 0
        state = env.reset()
        for t in itertools.count(start=1):
            if args.render and episode > 300:
                env.render()
                time.sleep(0.0082)
            # select action
            if total_steps < args.warmup:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, noise=True)
                action = action.ravel()
            # execute action
            next_state, reward, done, _ = env.step(action)
            # store transition
            agent.append(state, action, reward, next_state, done)
            if total_steps >= args.warmup:
                agent.update()

            state = next_state
            total_reward += reward
            total_steps += 1
            if done:
                ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward
                writer.add_scalar('Train/Episode Reward', total_reward,
                                  total_steps)
                writer.add_scalar('Train/Ewma Reward', ewma_reward,
                                  total_steps)
                print(
                    'Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}'
                    .format(total_steps, episode, t, total_reward,
                            ewma_reward))
                break
    env.close()


def test(args, env_name, agent, writer):
    print('Start Testing')
    env = gym.make(env_name)
    seeds = (args.seed + i for i in range(10))
    rewards = []
    for n_episode, seed in enumerate(seeds):
        total_reward = 0
        env.seed(seed)
        state = env.reset()

        for t in itertools.count(start=1):
            env.render()
            time.sleep(0.03)
            # select action
            action = agent.select_action(state, noise=False)
            # execute action
            next_state, reward, done, _ = env.step(action)
            # store transition
            state = next_state
            total_reward += reward
            total_steps += 1
            if done:
                rewards.append(total_reward)
                writer.add_scalar('Test/Episode Reward', total_reward, n_episode)
                break
    print('Average Reward', np.mean(rewards))
    env.close()


def main():
    ## arguments ##
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-m', '--model', default='ddpg.pth')
    parser.add_argument('--logdir', default='log/ddpg')
    # train
    parser.add_argument('--warmup', default=10000, type=int)
    parser.add_argument('--episode', default=1200, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--capacity', default=50000, type=int)
    parser.add_argument('--lra', default=1e-3, type=float)
    parser.add_argument('--lrc', default=1e-3, type=float)
    parser.add_argument('--gamma', default=.99, type=float)
    parser.add_argument('--tau', default=.005, type=float)
    # test
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--seed', default=20201111, type=int)
    args = parser.parse_args()

    ## main ##
    env_name = 'LunarLanderContinuous-v2'
    agent = DDPG(args)
    writer = SummaryWriter(f'log/DDPG-{time.time()}')
    if not args.test_only:
        train(args, env_name, agent, writer)
        agent.save(args.model)
    agent.load(args.model)
    test(args, env_name, agent, writer)


if __name__ == '__main__':
    main()
