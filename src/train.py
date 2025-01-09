from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch.nn as nn
import torch
from copy import deepcopy
import random
import numpy as np
import os

env_df = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.

env_rd = TimeLimit(
    env=HIVPatient(domain_randomization=True), max_episode_steps=200
)

env = env_df

state_dim = env.observation_space.shape[0]

config = {'nb_actions': env.action_space.n,
          'max_episode': 200,
          'learning_rate': 0.0007,
          'gamma': 0.98,
          'buffer_size': 1000000,
          'epsilon_min': 0.15,
          'epsilon_max': 1.,
          'epsilon_decay_period': 20000,
          'epsilon_delay_decay': 700,
          'batch_size': 256,
          'gradient_steps': 5,
          # 'criterion': torch.nn.MSELoss(), 
          'criterion': torch.nn.HuberLoss(),
          'update_target': 700,
          'nb_neurons': 256}


class ReplayBuffer():
    def __init__(self, capacity, device):
        self.capacity = capacity # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)

class DQN(nn.Module):
    def __init__(self, state_dim, nb_neurons, n_action):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons), # *2 C15 / 14
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons), # *2 C15 / 14
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, n_action)
        )

    def forward(self, x):
        return self.layers(x)



# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.nb_actions = config['nb_actions']
        self.memory = ReplayBuffer(config['buffer_size'], device)
        self.epsilon_max = config['epsilon_max']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_stop = config['epsilon_decay_period']
        self.epsilon_delay = config['epsilon_delay_decay']
        self.epsilon_step = (self.epsilon_max - self.epsilon_min) / self.epsilon_stop
        self.model = DQN(state_dim, config['nb_neurons'], self.nb_actions).to(device)
        self.target_model = DQN(state_dim, config['nb_neurons'], self.nb_actions).to(device).eval()
        self.criterion = config['criterion']
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.nb_gradient_steps = config['gradient_steps']
        self.update_target = config['update_target']

    def act(self, observation, use_random=False):
        device = "cuda" if next(self.model.parameters()).is_cuda else "cpu"
        with torch.no_grad():
            Q = self.model(torch.Tensor(observation).unsqueeze(0).to(device))
            return torch.argmax(Q).item()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self):
        device = torch.device('cpu')
        self.model = DQN(state_dim, config['nb_neurons'], self.nb_actions).to(device)
        path = os.getcwd() + "/model_C16.pt"
        self.model.load_state_dict(torch.load(path, map_location=device))
        self.model.eval()

    def greedy_action(self, network, state):
        device = "cuda" if next(network.parameters()).is_cuda else "cpu"
        with torch.no_grad():
            Q = network(torch.Tensor(state).unsqueeze(0).to(device))
            return torch.argmax(Q).item()

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self):

        device = "cuda" if torch.cuda.is_available() else "cpu"
        score = 0
        max_episode = config['max_episode']
        test_episode = 50
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        env = env_df
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0

        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = self.greedy_action(self.model, state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps):
                self.gradient_step()
            # replace target
            if step % self.update_target == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            step += 1
            if done or trunc:
                episode += 1
                if episode > test_episode:
                    test_score = evaluate_HIV(agent=self, nb_episode=1)
                    test_dr_score = evaluate_HIV_population(agent=self, nb_episode=1)
                else :
                    test_score = 0
                    test_dr_score = 0
                print("Epi ", '{:3d}'.format(episode),
                      ", epsilon", '{:6.2f}'.format(epsilon),
                      ", batch size", '{:5d}'.format(len(self.memory)),
                      ", episode return ", '{:.2e}'.format(episode_cum_reward),
                      ", test score ", '{:.2e}'.format(test_score),
                      ", test dr score ", '{:.2e}'.format(test_dr_score),
                      sep='')
                # train on both environnement
                if np.random.choice(2):
                  env = env_df
                else:
                  env = env_rd
                state, _ = env.reset()
                # save the best model based on the test score
                if test_score + test_dr_score > score:
                    score = test_score + test_dr_score
                    self.best_model = deepcopy(self.model).to(device)
                    self.save("model_C15.pt")
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
            else:
                state = next_state
