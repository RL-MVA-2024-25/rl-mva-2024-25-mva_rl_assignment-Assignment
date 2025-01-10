import os
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from gymnasium.wrappers import TimeLimit

from env_hiv import HIVPatient
from evaluate import evaluate_HIV, evaluate_HIV_population
from replay_buffer import ReplayBuffer


env = TimeLimit(
    env=HIVPatient(domain_randomization=True), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = self.config = {
            "nb_actions": env.action_space.n,
            "learning_rate": 0.001,
            "gamma": 0.98,
            "buffer_size": 100000,
            "epsilon_min": 0.02,
            "epsilon_max": 1.0,
            "epsilon_decay_period": 20000,
            "epsilon_delay_decay": 100,
            "batch_size": 200,
            "gradient_steps": 3,
            "update_target_strategy": "replace",
            "update_target_freq": 400,
            "update_target_tau": 0.005,
            "criterion": torch.nn.SmoothL1Loss(),
        }
        self.gamma = self.config["gamma"]
        self.batch_size = self.config["batch_size"]
        self.nb_actions = self.config["nb_actions"]
        # epsilon greedy strategy
        self.epsilon_max = self.config["epsilon_max"]
        self.epsilon_min = self.config["epsilon_min"]
        self.epsilon_stop = (
            self.config["epsilon_decay_period"]
            if "epsilon_decay_period" in self.config.keys()
            else 1000
        )
        self.epsilon_delay = (
            self.config["epsilon_delay_decay"]
            if "epsilon_delay_decay" in self.config.keys()
            else 20
        )
        self.epsilon_step = (self.epsilon_max - self.epsilon_min) / self.epsilon_stop
        # memory buffer
        self.memory = ReplayBuffer(self.config["buffer_size"], self.device)

        self.criterion = (
            self.config["criterion"]
            if "criterion" in self.config.keys()
            else torch.nn.MSELoss()
        )
        self.lr = (
            self.config["learning_rate"]
            if "learning_rate" in self.config.keys()
            else 0.001
        )
        self.optimizer = (
            self.config["optimizer"]
            if "optimizer" in self.config.keys()
            else torch.optim.Adam(self.model.parameters(), lr=self.lr)
        )
        self.nb_gradient_steps = (
            self.config["gradient_steps"]
            if "gradient_steps" in self.config.keys()
            else 1
        )
        self.update_target_strategy = (
            self.config["update_target_strategy"]
            if "update_target_strategy" in self.config.keys()
            else "replace"
        )
        self.update_target_freq = (
            self.config["update_target_freq"]
            if "update_target_freq" in self.config.keys()
            else 20
        )
        self.update_target_tau = (
            self.config["update_target_tau"]
            if "update_target_tau" in self.config.keys()
            else 0.005
        )
        self.model = self.DQN().to(self.device)
        self.target_model = deepcopy(self.model).to(self.device)

    def act(self, observation, use_random=False):
        with torch.no_grad():
            Q = self.model(torch.Tensor(observation).unsqueeze(0).to(self.device))
            return torch.argmax(Q).item()

    def save(self, path):
        torch.save(self.model.state_dict(), path + "/best_model.pt")

    def load(self):
        self.model = self.DQN().to(self.device)
        self.model.load_state_dict(
            torch.load(os.getcwd() + "/best_model.pt", map_location=self.device)
        )
        self.model.eval()

    def act_greedy(self, state):
        with torch.no_grad():
            Q = self.model(torch.Tensor(state).unsqueeze(0).to(self.device))
            return torch.argmax(Q).item()

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1 - D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def DQN(self):
        state_dim = env.observation_space.shape[0]
        n_action = env.action_space.n
        nb_neurons = 256

        DQN_model = torch.nn.Sequential(
            nn.Linear(state_dim, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, n_action),
        )

        return DQN_model

    def train(self):
        print("Using device:", self.device)

        # after 200 episode, all experiments suffer from a decline in reward
        max_episode = 250
        episode_return = []
        episode = 0
        step = 0
        episode_cum_reward = 0
        best_val_reward = 0
        state, _ = env.reset()
        epsilon = epsilon_max

        while episode < max_episode:
            # update epsilon
            if step > epsilon_delay:
                epsilon = max(epsilon_min, epsilon - epsilon_step)
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = self.act_greedy(state)

            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward

            for _ in range(self.nb_gradient_steps):
                self.gradient_step()

            if self.update_target_strategy == "replace":
                if step % self.update_target_freq == 0:
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == "ema":
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = (
                        tau * model_state_dict[key] + (1 - tau) * target_state_dict[key]
                    )
                self.target_model.load_state_dict(target_state_dict)

            step += 1
            if done or trunc:
                episode += 1

                # evaluate on the evaluate_HIV
                eposide_val_reward = evaluate_HIV(agent=self, nb_episode=1)

                # need to evaluate on the evaluate_HIV_population

                print(
                    f"Episode {episode:3d} | "
                    f"Epsilon {epsilon:6.2f} | "
                    f"Batch Size {len(self.memory):5d} | "
                    f"Episode Cummulative Reward {episode_cum_reward:.4e} | "
                    f"Evaluation Reward {eposide_val_reward:.4e}"
                )
                state, _ = env.reset()

                # save the best model
                if eposide_val_reward > best_val_reward:
                    best_val_reward = eposide_val_reward
                    self.best_model = deepcopy(self.model).to(self.device)
                    path = os.getcwd()
                    self.save(path)
                episode_return.append(episode_cum_reward)

                episode_cum_reward = 0
            else:
                state = next_state

        self.model.load_state_dict(self.best_model.state_dict())
        path = os.getcwd()
        self.save(path)
        return episode_return
