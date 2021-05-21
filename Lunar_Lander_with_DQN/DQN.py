import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque
import random

# Neural Network for Q Learning
class DQN(nn.Module):

    def __init__(self, state_size, action_size, seed=0):
        super(DQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.l1 = nn.Linear(state_size, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))

        return self.l3(x)


# Memory
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory():

    def __init__(self, memory_size, batch_size):

        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)

    def push(self, state, action, reward, next_state, done):
        self.memory.append(Transition(state, action, reward, next_state, done))

    def sample_batch(self):

        ts = random.sample(self.memory, k=self.batch_size)
        return Transition(*zip(*ts))

    def __len__(self):
        return len(self.memory)