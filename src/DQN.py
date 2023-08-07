import simpy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import namedtuple, deque

torch.manual_seed(1)
np.random.seed(1)

Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class DQNAgent:
    def __init__(self, state_size, action_space, discount_factor=1,
                 epsilon_greedy=1.0, epsilon_min=0.01, epsilon_decay=0.99995,
                 learning_rate=0.001, max_memory_size=2000):

        self.state_size = state_size
        self.action_size = len(action_space)
        self.action_space = action_space
        self.max_memory_size = max_memory_size
        self.memory = deque(maxlen=max_memory_size)
        self.gamma = discount_factor
        self.epsilon = epsilon_greedy
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = learning_rate
        self._build_nn_model()

        # Attributes to track total cost per day and daily reward
        self.total_cost_per_day = []
        self.daily_reward = 0

    def _build_nn_model(self):
        self.model = nn.Sequential()

        # 은닉층
        self.model.add_module(f'hidden_{0}',
                              nn.Linear(self.state_size, 32))
        self.model.add_module(f'activation_{0}', nn.ReLU())

        self.model.add_module(f'hidden_{1}',
                              nn.Linear(32, 32))
        self.model.add_module(f'activation_{1}', nn.ReLU())

        self.model.add_module(f'hidden_{2}',
                              nn.Linear(32, 32))
        self.model.add_module(f'activation_{2}', nn.ReLU())

        # 마지막 층
        self.model.add_module('output', nn.Linear(32, self.action_size))

        # 모델 빌드 & 컴파일
        self.model.to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(self.action_space)
        else:
            q_values = self.model(torch.FloatTensor(state).to(device))
            # print(q_values.argmax())
            return q_values.argmax()

        # action_idx = np.argmax(q_values)
        # return self.action_space[action_idx]

    def _learn(self, batch_samples):
        batch_states, batch_targets = [], []
        for transition in batch_samples:
            s, a, r, next_s, done = transition
            state_tensor = torch.FloatTensor(s).to(device)
            next_state_tensor = torch.FloatTensor(next_s).to(device)
            q_values = self.model(state_tensor)
            next_q_values = self.model(next_state_tensor)
            if done:
                target = r
            else:
                target = (r + self.gamma * torch.max(next_q_values))
            if type(a) == list:
                for i in range(len(self.action_space)):
                    if self.action_space[i] == a:
                        q_values[0][i] = target
            else:
                q_values[0][a] = target
            batch_states.append(state_tensor.flatten())
            batch_targets.append(q_values)

            self._adjust_epsilon()

        batch_states = torch.stack(batch_states)
        batch_targets = torch.stack(batch_targets)

        loss = nn.MSELoss()(self.model(batch_states), batch_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def replay(self, batch_size):
        samples = random.choices(self.memory, k=batch_size)
        loss = self._learn(samples)
        return loss

    def _adjust_epsilon(self):
        if self.epsilon > self.epsilon_min:
            if self.epsilon > 0.01:
                self.epsilon *= self.epsilon_decay

    def remember(self, transition):
        self.memory.append(transition)


# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
