import torch.nn as nn
import torch
import gym
import torch.nn.functional as F
import random
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', '-e', type=int, default=1000)
parser.add_argument('--gamma', '-b' , default=0.9)
parser.add_argument('--lr', default=1e-3)
parser.add_argument('--replay_size', default=1024)
parser.add_argument('--batch_size', default=32)
parser.add_argument('--clip_param', default=0.2)
parser.add_argument('--update_times', default=20)
args = parser.parse_args()

args.device = "cuda" if torch.cuda.is_available() else "cpu"

class Actor(nn.Module):
    def __init__(self, num_action, num_state):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 32)
        self.fc2 = nn.Linear(32, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        a_prob = F.softmax(x, dim=1)
        return a_prob

class Critic(nn.Module):
    def __init__(self, num_state):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class PPO():
    def __init__(self, args, env):
        self.clip_param = args.clip_param
        self.device = args.device
        self.actor = Actor(env.action_space.n, env.observation_space.shape[0])
        self.critic = Critic(env.observation_space.shape[0])
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.buffer = ExperienceReplayMemory(args.replay_size)
        self.num_feats = env.observation_space.shape[0]
        self.update_times = args.update_times
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr)

    def select_action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s), 0).to(self.device)
        a_prob = self.actor(s)
        dist = Categorical(a_prob)
        a = dist.sample()
        return a.item(), a_prob[:, a.item()].item()

    def append_to_replay(self, s, a, a_prob, r, s_):
        self.buffer.push((s, a, a_prob, r, s_))

    def preprocess(self):
        transitions = self.buffer.memory

        batch_state, batch_action, batch_action_prob, batch_reward, batch_next_state = zip(*transitions)
        shape = (-1, self.num_feats)
        batch_state = torch.tensor(batch_state, device=self.device, dtype=torch.float).view(shape)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).squeeze().view(-1, 1)
        batch_action_prob = torch.tensor(batch_action_prob, device=self.device, dtype=float).view(-1, 1)
        # batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).squeeze().view(-1, 1)
        batch_next_state = torch.tensor(batch_next_state, device=self.device, dtype=torch.float).view(shape)
        return batch_state, batch_action, batch_action_prob, batch_reward, batch_next_state
    def update(self):
        state, action, old_action_prob, reward, next_state = self.preprocess()
        Gt = []
        R = 0
        for r in reward[::-1]:
            R = r + R * self.gamma
            Gt.insert(0, R)
        Gt = torch.tensor(Gt)
        for i in range(self.update_times):
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer.__len__())), self.batch_size, False):
                Gt_index = Gt[index].view(-1, 1).to(self.device)
                V = self.critic(state[index])
                At = Gt_index - V
                At = At.detach()
                action_prob = self.actor(state[index]).gather(1, action[index])

                rt = action_prob / old_action_prob[index]

                surr1 = rt * At
                surr2 = torch.clamp(rt, 1 - self.clip_param, 1 + self.clip_param) * At
                # update actor

                action_loss = -torch.min(surr1, surr2).mean()
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                self.actor_optimizer.step()

                # update critic

                value_loss = F.mse_loss(V, Gt_index)
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                self.critic_optimizer.step()

        self.buffer.memory = []


class ExperienceReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

if __name__ == "__main__":
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    env = env.unwrapped
    ppo = PPO(args, env)
    for i in range(args.epochs):
        s = env.reset()
        while True:
            env.render()
            a, a_prob = ppo.select_action(s)
            s_, r, done, _ = env.step(a)
            ppo.append_to_replay(s, a, a_prob, r, s_)
            s = s_

            if done:
                if ppo.buffer.__len__() >= args.batch_size:
                    ppo.update()
                    print("Finished")
                    break