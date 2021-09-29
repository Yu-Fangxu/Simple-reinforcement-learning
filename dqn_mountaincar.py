import numpy as np
import torch.nn as nn
import torch
import gym
import torch.nn.functional as F
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', '-e', type=int, default=1000)
parser.add_argument('--gamma', '-b' , default=0.9)
parser.add_argument('--lr', default=1e-4)
parser.add_argument('--target_net_update_freq', default=100)
parser.add_argument('--replay_size', default=1024)
parser.add_argument('--batch_size', default=512)
parser.add_argument('--epsilon', default=0.7)
args = parser.parse_args()

args.device = "cuda" if torch.cuda.is_available() else "cpu"
class Net(nn.Module):
    def __init__(self, num_feats, num_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_feats, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, num_actions)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQN(nn.Module):
    def __init__(self, env, args):
        super(DQN, self).__init__()
        self.device = args.device
        self.gamma = args.gamma
        self.target_net_update_freq = args.target_net_update_freq
        self.batch_size = args.batch_size
        self.num_feats = env.observation_space.shape[0]
        self.num_actions = env.action_space.n
        self.env = env
        self.lr = args.lr
        self.replay_size = args.replay_size
        self.target_model = Net(self.num_feats, self.num_actions)
        self.model = Net(self.num_feats, self.num_actions)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        self.memory = ExperienceReplayMemory(self.replay_size)
        self.epsilon = args.epsilon
        self.learn_step_counter = 0

        self.model.to(self.device)
        self.target_model.to(self.device)

    def append_to_replay(self, s, a, r, s_):
        self.memory.push((s, a, r, s_))

    def pre_minibatch(self):
        transitions = self.memory.sample(self.batch_size)

        batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)
        # print(batch_state)
        shape = (-1, self.num_feats)
        batch_state = torch.tensor(batch_state, device=self.device, dtype=torch.float).view(shape)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).squeeze().view(-1, 1)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).squeeze().view(-1, 1)
        batch_next_state = torch.tensor(batch_next_state, device=self.device, dtype=torch.float).view(shape)

        return batch_state, batch_action, batch_reward, batch_next_state

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        x = x.to(self.device)
        if np.random.uniform() < self.epsilon:
            actions_val = self.model(x).detach()
            # print(actions_val.shape)
            action = torch.argmax(actions_val).data.item()
        else:
            action = np.random.randint(0, self.num_actions)
        # print(action)
        return action

    def learn(self):
        self.target_model.eval()
        if self.learn_step_counter % self.target_net_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        self.learn_step_counter += 1
        batch_state, batch_action, batch_reward, batch_next_state = self.pre_minibatch()

        current_q_values = self.model(batch_state).gather(1, batch_action)  # for each data, choose q[i][a]
        next_q_values = self.target_model(batch_next_state)

        ### double dqn
        next_q_values_1 = self.model(batch_next_state)
        next_actions = torch.argmax(next_q_values_1, dim=1).view(self.batch_size, 1)
        q_target = batch_reward + self.gamma * next_q_values.gather(1, next_actions).view(self.batch_size, 1)

        ### vanilla dqn

        # q_target = batch_reward + self.gamma * next_q_values.max(1)[0].view(self.batch_size, 1)
        loss = self.loss(current_q_values, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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

if __name__ == '__main__':
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    env = env.unwrapped
    dqn = DQN(env, args)
    for i in range(args.epochs):
        s = env.reset()
        total_r = 0
        while True:
            env.render()
            action = dqn.choose_action(s)

            s_, r, done, _ = env.step(action)

            dqn.append_to_replay(s, action, r, s_)

            total_r += r
            if dqn.memory.__len__() >= args.replay_size:
                dqn.learn()
                if done:
                    print(i, '| total reward: ', round(total_r))
            if done:
                break
            s = s_
