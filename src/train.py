from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from replay_buffer import ReplayBuffer_1
import torch.optim.lr_scheduler as lr_scheduler
from DQN import DQN
from copy import deepcopy
import random
import os
from pathlib import Path


env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
def greedy_action(network, state):
    device = next(network.parameters()).device
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()
class ProjectAgent:
    def __init__(self, use_dueling=True, use_per=True, state_dim=6, n_actions=4):
        """
        DQN Agent that optionally uses:
          - Dueling Q-Network architecture
          - Prioritized Experience Replay
        """
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.gamma = 0.85 
        self.save_path = str(Path(__file__).parent / "best_reward_agent_1.pth")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.replay_buffer = ReplayBuffer_1(capacity=60000,device = self.device)


        self.lr = 1e-3 
        self.batch_size = 1024 
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.996  

        self.target_update_freq = 1000
        self.update_count = 0

        # initialize models
        self.model = DQN().to(self.device)
        self.target_model = DQN().to(self.device)

        self.target_model.load_state_dict(self.model.state_dict())  # We must have the same model at the beginning
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(),lr=self.lr)   
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=350, gamma=0.5) # could remove it

    def act(self, observation, use_random=False):
        if use_random and random.random() < self.epsilon:
                return env.action_space.sample()
        else:
            Q = self.model(torch.Tensor(observation).unsqueeze(0).to(self.device))
            return torch.argmax(Q).item()
    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved in {path}")

    def load(self):
        self.model.load_state_dict(torch.load(self.save_path, map_location=self.device))
        self.target_model = deepcopy(self.model).to(self.device)
        self.model.eval()
        print("Model loaded")

    def step_scheduler(self):
        self.scheduler.step()

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def gradient_step(self):
        """Perform one training step of DQN."""
        if len(self.replay_buffer) < self.batch_size:
            return  # We can't train without enough samples in the buffer


        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)


        model_values = self.model(states)
        model_values = model_values.gather(1, actions.unsqueeze(1)).squeeze(1)


        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1)
            max_next_q = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_value = rewards + self.gamma * max_next_q * (1 - dones)

        loss = (model_values - target_value) ** 2

        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def train(self,max_episodes):
        reward_history = []
        Best_reward = 0
        for episode in range(max_episodes):
            state, _ = env.reset()
            epoch_r = 0.0
            for _ in range(200):
                action = agent.act(state, use_random=True)
                next_state, reward, done, truncated, _info = env.step(action)

                agent.replay_buffer.append(state, action, reward, next_state, done)
                agent.gradient_step()

                state = next_state
                epoch_r += reward
                if done or truncated:
                    break
            agent.update_epsilon()
            agent.step_scheduler()
            print(f"Episode {episode:4d}, Reward: {int(epoch_r):11d}, Epsilon: {agent.epsilon:.2f}")
            reward_history.append(epoch_r)
            if epoch_r > Best_reward:
                Best_reward = epoch_r
                agent.save("best_reward_agent_1.pth")
        return reward_history
        

