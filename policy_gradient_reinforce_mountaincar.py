import torch.nn as nn
import torch
import numpy as np
import gym
import torch.nn.functional as F
import argparse
from torch.distributions import Categorical
from tqdm import tqdm
### attention:
#   reinforce algorithm is hard to solve mountaincar problem, the better choice is actor critic
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', '-e', type=int, default=1000)
parser.add_argument('--gamma', '-b' , default=0.99)
parser.add_argument('--lr', default=1e-3)

args = parser.parse_args()
args.device = "cuda" if torch.cuda.is_available() else "cpu"
eps = np.finfo(np.float32).eps.item()
class Net(nn.Module):
    def __init__(self, num_actions, num_feats):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_feats, 24)
        self.fc3 = nn.Linear(24, num_actions)

        self.rewards = []
        self.log_probs = []

    def forward(self, s):
        s = F.relu(self.fc1(s))
        a = self.fc3(s)
        return F.softmax(a, dim=1)

class Policy(nn.Module):
    def __init__(self, env, args):
        super(Policy, self).__init__()
        self.env = env
        self.gamma = args.gamma

        self.device = args.device
        self.model = Net(env.action_space.n, env.observation_space.shape[0]).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
    def choose_action(self, s):
        s = torch.from_numpy(s).float().unsqueeze(0).to(self.device)
        actions = self.model(s)

        dis = Categorical(actions)
        action = dis.sample()
        self.model.log_probs.append(dis.log_prob(action))
        return action.item()

    def learn(self):

        policy_loss = []
        returns = []
        R = 0
        for r in self.model.rewards[::-1]:
            R = r + R * self.gamma
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps) # baseline
        for log_prob, R in zip(self.model.log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        self.model.rewards = []
        self.model.log_probs = []

if __name__ == "__main__":
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    env = env.unwrapped
    policy = Policy(env, args)
    for i in tqdm(range(args.epochs)):
        s = env.reset()
        j = 0
        while True:
            j += 1
            env.render()
            action = policy.choose_action(s)

            s, r, done, _ = env.step(action)
            policy.model.rewards.append(r)
            if done:
                print("Finished")
                break
            if j % 100 == 99:
            # print(i, "start learn")
                policy.learn()
