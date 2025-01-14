from gymnasium.wrappers import TimeLimit
from env_hiv_ import FastHIVPatient as HIVPatient
from functools import partial
from evaluate import evaluate_HIV, evaluate_HIV_population
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from tqdm import tqdm
import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# env = TimeLimit(
#     env=HIVPatient(domain_randomization=True), max_episode_steps=200
# )  # The time wrapper limits the number of steps in an episode at 200.
# # Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

class VanillaAgent:
    def act(self, observation, use_random=False):
        return 0

    def save(self, path):
        pass

    def load(self):
        pass


class ForestAgent:
    
    def __init__(self, config=None, env=None):
        self.config = config
        self.env = env
        self.indiv_env = TimeLimit(HIVPatient(domain_randomization=False), max_episode_steps=200)
        # self.random_env = TimeLimit(HIVPatient(domain_randomization=True), max_episode_steps=200)
        if config != None:
            self.config_str = ''.join(f'_{value}' for key, value in config.items())
        
    def collect_samples(self, horizon, disable_tqdm=False, print_done_states=False):
        s, _ = self.env.reset()
        #dataset = []
        S = []
        A = []
        R = []
        S2 = []
        D = []
        for _ in tqdm(range(horizon), disable=disable_tqdm):
            a = self.env.action_space.sample()
            s2, r, done, trunc, _ = self.env.step(a)
            #dataset.append((s,a,r,s2,done,trunc))
            S.append(s)
            A.append(a)
            R.append(r)
            S2.append(s2)
            D.append(done)
            if done or trunc:
                s, _ = self.env.reset()
                if done and print_done_states:
                    print("done!")
            else:
                s = s2
        S = np.array(S)
        A = np.array(A).reshape((-1,1))
        R = np.array(R)
        S2= np.array(S2)
        D = np.array(D)
        return S, A, R, S2, D
    
    def rf_fqi(self, S, A, R, S2, D, iterations, nb_actions, gamma, disable_tqdm=False):
        nb_samples = S.shape[0]
        Qfunctions = []
        indiv_test = 0
        random_test = 0
        SA = np.append(S,A,axis=1)
        for iter in tqdm(range(iterations), disable=disable_tqdm):
            if iter==0:
                value=R.copy()
            else:
                Q2 = np.zeros((nb_samples,nb_actions))
                for a2 in range(nb_actions):
                    A2 = a2*np.ones((S.shape[0],1))
                    S2A2 = np.append(S2,A2,axis=1)
                    Q2[:,a2] = Qfunctions[-1].predict(S2A2)
                max_Q2 = np.max(Q2,axis=1)
                value = R + gamma*(1-D)*max_Q2
            Q = XGBRegressor(n_estimators=self.config["n_estim"], max_depth=self.config["max_depth"])
            Q.fit(SA,value)
            self.model = Q
            Qfunctions.append(Q)
            if iter % 10 == 0:
                indiv_eval = evaluate_HIV(agent=self, nb_episode=5)
                random_eval = evaluate_HIV_population(agent=self, nb_episode=20)
                if indiv_test <= indiv_eval:
                    indiv_test = indiv_eval
                    Q.save_model(f"./models/Q{self.config_str}.json")
                elif indiv_test == indiv_eval and random_test <= random_eval:
                    random_test = random_eval
                    Q.save_model(f"./models/Q{self.config_str}.json")
        return Qfunctions
    
    def train(self, horizon, iterations, gamma, disable_tqdm=False, print_done_states=False):
        S, A, R, S2, D = self.collect_samples(horizon, disable_tqdm, print_done_states)
        self.Qfunctions = self.rf_fqi(S, A, R, S2, D, iterations, self.env.action_space.n, gamma, disable_tqdm)
        return self.Qfunctions[-1]


    def act(self, observation):
        Q = np.zeros(self.indiv_env.action_space.n)
        for a in range(self.indiv_env.action_space.n):
            obs =  np.expand_dims(observation, axis=0)
            act =  np.array([[a]])
            SA = np.append(obs,act,axis=1)
            Q[a] = self.model.predict(SA)
        return np.argmax(Q)

    def save(self, path):
        pass

    def load(self):
        self.model = XGBRegressor()
        self.model.load_model("./models/Q_100_5.json")

model = ForestAgent({"n_estim":400, "max_depth":14}, TimeLimit(HIVPatient(domain_randomization=True), max_episode_steps=200))
model.train(100000, 500, 0.96, disable_tqdm=False)

"""
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity) # capacity of the buffer
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
    
class DQNAgent:
    
    def __init__(self, config):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dim = env.observation_space.shape[0]
        n_action = env.action_space.n 
        layers = []
        for layer in range(config["n_layers"]-1):
            layers.append(nn.Linear(state_dim if layer==0 else config["hidden_dim"], config["hidden_dim"]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
        layers.append(nn.Linear(config["hidden_dim"], n_action))
        self.model = nn.Sequential(*layers).to(device)
        self.target_model = deepcopy(self.model).to(device)
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.nb_actions = env.action_space.n
        self.memory = ReplayBuffer(config['buffer_size'], device)
        self.epsilon_max = config['epsilon_max']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_stop = config['epsilon_decay_period']
        self.epsilon_delay = config['epsilon_delay_decay']
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.max_episode = config['max_episode']
        self.pre_fill_buffer = config['pre_fill_buffer'] if 'pre_fill_buffer' in config.keys() else False
        self.config_str = ''.join(f'_{value}' for key, value in config.items())

        self.train(env, self.max_episode)

    def greedy_action(self, state):
        device = "cuda" if next(self.model.parameters()).is_cuda else "cpu"
        with torch.no_grad():
            Q = self.model(torch.Tensor(state).unsqueeze(0).to(device))
            return torch.argmax(Q).item()
    
    
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.model(Y).max(1)[0].detach()
            #update = torch.addcmul(R, self.gamma, 1-D, QYmax)
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 

    def train(self, env, max_episode):
        episode_return = []
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0

        if self.pre_fill_buffer:
            print("Pre-filling buffer")
            while len(self.memory) < self.memory.capacity:
                for _ in range(200):
                    action = env.action_space.sample()
                    next_state, reward, done, trunc, _ = env.step(action)
                    self.memory.append(state, action, reward, next_state, done)
                    state = next_state
                    if done or trunc:
                        state, _ = env.reset()


        for episode in tqdm(range(max_episode)):
            for _ in range(200):
                # update epsilon
        
                if step > self.epsilon_delay:
                    epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)

                # select epsilon-greedy action
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = self.greedy_action(state)

                # step
                next_state, reward, done, trunc, _ = env.step(action)
                self.memory.append(state, action, reward, next_state, done)
                episode_cum_reward += reward

                for _ in range(self.nb_gradient_steps): 
                    self.gradient_step()
                # update target network if needed
                if step % self.update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())

                state = next_state
                step += 1
                if done or trunc:
                    break

            # print("Episode ", '{:3d}'.format(episode), 
            #             ", epsilon ", '{:6.2f}'.format(epsilon), 
            #             ", Memory ", '{:5d}'.format(len(self.memory)), 
            #             ", episode return ", '{:4.1f}'.format(episode_cum_reward),
            #             ", step ", '{:5d}'.format(step),
            #             sep='')
            state, _ = env.reset()
            episode_return.append(episode_cum_reward)
            episode_cum_reward = 0
            env.reset()
        moving_average = np.convolve(episode_return, np.ones(5)/5, mode='same')
        # plot episode reward
        plt.plot(moving_average)
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.title('Training')
        plt.savefig('training' + self.config_str + '.png')
        plt.clf()
        self.model.eval()
        return episode_return

    def act(self, observation, use_random=False):
        if use_random:
            return env.action_space.sample()
        else:
            return self.greedy_action(observation)

    def save(self, path):
        pass

    def load(self):
        pass
    
"""