import numpy as np
import torch.nn as nn
import torch
import gym
import torch.nn.functional as F
import random
import argparse
n_states = 40
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
args.device = "cpu"
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

# import torch
# import torch.nn as nn
# import numpy as np
# import gym
# import torch.autograd
# import random
#
#
# class MyNet(nn.Module):
#     def __init__(self):
#         super(MyNet,self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(2,24),    ##两个输入
#             nn.ReLU(),
#             nn.Linear(24,24),
#             nn.ReLU(),
#             nn.Linear(24,3)    ##三个输出
#         )
#         self.mls = nn.MSELoss()
#         self.opt = torch.optim.Adam(self.parameters(),lr = 0.001)
#     def forward(self,x):
#         return self.fc(x)
# env = gym.envs.make('MountainCar-v0')
# env = env.unwrapped
# net = MyNet()   #实例化
# net2 = MyNet()
#
#
# store_count = 0
# store_size = 2000
# decline = 0.6   # epsilo
# learn_time = 0
# updata_time = 20  #目标值网络更新步长
# gama = 0.9
# b_size = 1000
# store = np.zeros((store_size,6))    ###[s,a,s_,r]，其中s占两个，a占一个，r占一个
# start_study = False
#
# for i in range(50000):
#     s = env.reset()  ##
#     while True:
#         ###根据 state 产生动作
#         if random.randint(0,100) < 100 * (decline ** learn_time):  # 相当于epsilon
#             a = random.randint(0,2)
#         else:
#             out = net(torch.Tensor(s)).detach()    ##detch()截断反向传播的梯度，[r1,r2]
#             a = torch.argmax(out).data.item()      ##[取最大，即取最大值的index]
#         s_, r, done, info = env.step(a)               ##环境返回值，可查看step函数
#
#         store[store_count % store_size][0:2] = s    ##覆盖老记忆
#         store[store_count % store_size][2:3] = a
#         store[store_count % store_size][3:5] = s_
#         store[store_count % store_size][5:6] = r
#         store_count +=1
#         s = s_
#   #####反复试验然后存储数据，存满后，就每次取随机部分采用sgd
#         if store_count > store_size:
#             if learn_time % updata_time ==0:
#                 net2.load_state_dict(net.state_dict())  ##延迟更新
#
#             index = random.randint(0,store_size - b_size - 1)
#             b_s = torch.Tensor(store[index:index + b_size,0:2])
#             b_a = torch.Tensor(store[index:index + b_size, 2:3]).long()  ##  因为gather的原因，索引值必须是longTensor
#             b_s_ = torch.Tensor(store[index:index + b_size, 3:5])
#             b_r = torch.Tensor(store[index:index + b_size, 5:6 ])  #取batch数据
#
#             q = net(b_s).gather(1,b_a) #### 聚合形成一张q表    根据动作得到的预期奖励是多少
#             q_next = net2(b_s_).detach().max(1)[0].reshape(b_size,1)  #值和索引，延迟更新
#             tq = b_r+gama * q_next
#             loss = net.mls(q,tq)
#             net.opt.zero_grad()
#             loss.backward()
#             net.opt.step()
#
#             learn_time += 1
#             if not start_study:
#                 print('start_study')
#                 start_study  = True
#                 break
#         if done:
#             print(i)
#             break
#
#         env.render()
#
#
