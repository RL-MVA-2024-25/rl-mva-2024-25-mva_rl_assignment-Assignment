from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
import matplolib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from replay_buffer import ReplayBuffer
import torch.optim.lr_scheduler as lr_scheduler
from DQN import QNetwork
from copy import deepcopy

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
    def __init__(self, env,model):
        self.n_actions = 4
        self.state_dim = 6
        self.gamma = 0.85 
        self.device = "cuda" if next(model.parameters()).is_cuda else "cpu"
        self.save_path = "agent.pt"

        self.replay_buffer = ReplayBuffer(capacity=60000,device = self.device)
        self.model = model
        self.lr = 1e-3 
        self.batch_size = 1024

        self.epsilon_max = 1
        self.epsilon_min =  0.01
        self.epsilon_stop = 1000
        self.epsilon_delay = 20
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop

        
        self.update_count = 0
        self.target_update_freq = 100
        self.update_target_tau = 0.005
        self.update_target_strategy = 'ema'
        self.nb_gradient_steps = 2
 
        #self.q_network = QNetwork(self.state_dim, self.n_actions).to(self.device)
        self.target_network = DQN(self.state_dim, self.n_actions).to(self.device)
           
        self.target_network.load_state_dict(self.model.state_dict())
        self.target_network.eval()
        
        self.monitoring_nb_trials = 50
        self.monitor_every =  50

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(),lr=self.lr)                     
        #self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=350, gamma=0.5)
    
    def MC_eval(self, env, nb_trials):   
        MC_total_reward = []
        MC_discounted_reward = []
        for _ in range(nb_trials):
            x,_ = env.reset()
            done = False
            trunc = False
            total_reward = 0
            discounted_reward = 0
            step = 0
            while not (done or trunc):
                a = greedy_action(self.model, x)
                y,r,done,trunc,_ = env.step(a)
                x = y
                total_reward += r
                discounted_reward += self.gamma**step * r
                step += 1
            MC_total_reward.append(total_reward)
            MC_discounted_reward.append(discounted_reward)
        return np.mean(MC_discounted_reward), np.mean(MC_total_reward)

    def V_initial_state(self, env, nb_trials):   
        with torch.no_grad():
            for _ in range(nb_trials):
                val = []
                x,_ = env.reset()
                val.append(self.model(torch.Tensor(x).unsqueeze(0).to(self.device)).max().item())
        return np.mean(val)
    
    def gradient_step(self):
        if len(self.replay_buffer) > self.batch_size:
            X, A, R, Y, D = self.replay_buffer.sample(self.batch_size)
            QYmax = self.target_network(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 

    def act(self, observation, use_random=False):
        if use_random:
            return env.action_space.sample()
        else:
            return greedy_action(self.model, observation)

    def train(self, env, max_episode):
        episode_return = []
        MC_avg_total_reward = []   # NEW NEW NEW
        MC_avg_discounted_reward = []   # NEW NEW NEW
        V_init_state = []   # NEW NEW NEW
        episode = 0
        episode_cum_reward = 0
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
                action = greedy_action(self.model, state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.replay_buffer.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()
            # update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0: 
                    self.target_network.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_network.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                self.target_network.load_state_dict(target_state_dict)
            # next transition
            step += 1
            if done or trunc:
                episode += 1
                # Monitoring
                if self.monitoring_nb_trials>0:
                    MC_dr, MC_tr = self.MC_eval(env, self.monitoring_nb_trials)    # NEW NEW NEW
                    V0 = self.V_initial_state(env, self.monitoring_nb_trials)   # NEW NEW NEW
                    MC_avg_total_reward.append(MC_tr)   # NEW NEW NEW
                    MC_avg_discounted_reward.append(MC_dr)   # NEW NEW NEW
                    V_init_state.append(V0)   # NEW NEW NEW
                    episode_return.append(episode_cum_reward)   # NEW NEW NEW
                    print("Episode ", '{:2d}'.format(episode), 
                          ", epsilon ", '{:6.2f}'.format(epsilon), 
                          ", batch size ", '{:4d}'.format(len(self.replay_buffer)), 
                          ", ep return ", '{:4.1f}'.format(episode_cum_reward), 
                          ", MC tot ", '{:6.2f}'.format(MC_tr),
                          ", MC disc ", '{:6.2f}'.format(MC_dr),
                          ", V0 ", '{:6.2f}'.format(V0),
                          sep='')
                else:
                    episode_return.append(episode_cum_reward)
                    print("Episode ", '{:2d}'.format(episode), 
                          ", epsilon ", '{:6.2f}'.format(epsilon), 
                          ", batch size ", '{:4d}'.format(len(self.replay_buffer)), 
                          ", ep return ", '{:4.1f}'.format(episode_cum_reward), 
                          sep='')

                
                state, _ = env.reset()
                episode_cum_reward = 0
            else:
                state = next_state
        return episode_return, MC_avg_discounted_reward, MC_avg_total_reward, V_init_state

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved in {path}")

    def load(self):
        self.model.load_state_dict(torch.load(self.save_path, map_location=self.device))
        self.target_network = deepcopy(self.model).to(self.device)
        self.model.eval()


    def collect_sample(self,nb_sample):
        s, _ = env.reset()
        for _ in range(nb_sample):
            a = self.act(s)
            s2, r, done, trunc, _ = env.step(a)
            self.replay_buffer.append(s, a, r, s2, done)
            if done or trunc :
                s, _ = env.reset()
            else:
                s = s2
        print('end of collection')



