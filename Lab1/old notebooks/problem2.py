
#### https://chatgpt.com/share/672df28c-dab8-8008-b199-937a95e1637b


# Copyright [2024] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 1 for EL2805 - Reinforcement Learning.

# Load packages
import numpy as np
import gymnasium as gym
import torch
import random
import matplotlib.pyplot as plt

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()
k = env.action_space.n      # tells you the number of actions
low, high = env.observation_space.low, env.observation_space.high

# Parameters
N_episodes = 100        # Number of episodes to run for training
discount_factor = 1.    # Value of gamma

# Reward
episode_reward_list = []  # Used to save episodes reward

# Functions used during training
def running_average(x, N):
    ''' Function used to compute the running mean
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

def scale_state_variables(s, low=env.observation_space.low, high=env.observation_space.high):
    ''' Rescaling of s to the box [0,1]^2 '''
    x = (s - low) / (high - low)
    return x



####VARIABLES QUE CAL DEFINIR
m = ### ????
state_dim = 2
gamma = 1
lamda = ####???
alpha = ####???
n = np.array(m, state_dim)
z = np.zeros(k, m)
w = np.array(k,m)



# Training process
for i in range(N_episodes):
    # Reset enviroment data
    done = False
    truncated = False
    state = scale_state_variables(env.reset()[0])
    total_episode_reward = 0.

    w = np.zeros(k,2 )
    z = np.zeros(k,2 )

    while not (done or truncated):
        # Take a random action
        # env.action_space.n tells you the number of actions
        # available
        action = np.random.randint(0, k) ## epsilon greedy volem que faci uniform amb probability eposilon i greeyd amb prob 1-epsilon
            
        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise. Truncated is true if you reach 
        # the maximal number of time steps, False else.
        next_state, reward, done, truncated, _ = env.step(action)
        next_state = scale_state_variables(next_state)
        
        

        # Update episode reward
        total_episode_reward += reward
            
        # Update state for next iteration
        state = next_state

    # Append episode reward
    episode_reward_list.append(total_episode_reward)

    # Close environment
    env.close()
    

# Plot Rewards
plt.plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
plt.plot([i for i in range(1, N_episodes+1)], running_average(episode_reward_list, 10), label='Average episode reward')
plt.xlabel('Episodes')
plt.ylabel('Total reward')
plt.title('Total Reward vs Episodes')
plt.legend()
plt.grid(alpha=0.3)
plt.show()


class Fourier_basis:

    def __init__(self, n_states, max_non_zero, p, type_coeffs, defined_coeffs):
        self.n_states = n_states #input size
        self.m = m #output size
        self.p = p
        self.max_non_zero = max_non_zero
        self.coeffs = self.define_coeffs(type_coeffs, defined_coeffs) 


    def define_coeffs(self, type_coeffs, defined_coeffs):
        
        if type_coeffs == "predefined":
            coeffs = defined_coeffs
        
        else:
            coeffs = np.array(np.zeros(self.n_states)) 

            for i in range(1, self.max_non_zero + 1):
                for indices in combinations(range(self.n_states), i):
                    for c in product(range(1, self.p + 1), repeat=i):
                        coef = np.zeros(self.n_states)
                        coef[list(indices)] = list(c)
                        coeffs = np.vstack((coeffs, coef))
        
        return coeffs

    def get_phi(self, state): ## to_basis
        return np.cos(np.pi*np.dot(self.n_coeffs, state))
    
    def scale_learning_rates(self, lr):
        norm = np.linalg.norm(self.coeffs, axis=1)
        norm[norm == 0.] = 1.  # When the norm is zero do not scale
        return lr / norm




class Sarsa_algorithm:

    def __init__(self, state_space, action_space,  min_max_norm=False, alpha=0.0001, lamb=0.9, gamma=0.99, epsilon=0.05, fourier_order=2, max_non_zero_fourier=2):
        self.alpha = alpha
        self.lr = alpha
        self.lamb = lamb
        self.gamma = gamma
        self.epsilon = epsilon

        self.state_space = state_space
        self.state_dim = self.state_space.shape[0]
        self.action_space = action_space
        self.action_dim = self.action_space.n
        self.min_max_norm = min_max_norm
        self.p = fourier_order
        self.non_zero = max_non_zero_fourier
        self.basis = Fourier_basis(self.state_dim, self.action_dim, self.non_zero, self.p, "nothing", None)
        self.scaled_lr = self.basis.scale_learning_rate(self.alpha)

        self.num_basis = self.basis.get_num_basis()

        self.z = {a: np.zeros(self.num_basis) for a in range(self.action_dim)} ## np.zeros((self.action_dim, self.num_basis))
        self.w = {a: np.zeros(self.num_basis) for a in range(self.action_dim)} ## np.zeros((self.action_dim, self.num_basis))

        self.q_old = None
        self.action = None

        ###SGD
        self.velocity =  np.zeros((self.action_dim, self.num_basis))
        self.momentum = 0.0 ## ???


    def get_q_value(self, w, action, state):
        return np.dot(w[action], basis.get_phi(state))


    def epsilon_greedy(self, state):
        
        if random.random()<self.epsilon:
            action = np.random.randint(0, self.k)

        else:
            Q_s = [self.get_q_value(w, a, state) for a in range(self.action_dim)]
            action = np.argmax(Q_s)

        return action


    def update_eligibility_trace(self, q, action):

        for a in range(self.action_dim):
            if a == action:
                self.z[a] = self.gamma*self.lamb*self.z[a] + q[a]###gradient wrt to w_a of Q_w(st,a)##
            else:
                self.z[a] = self.gamma*self.lamb*self.z[a]
    

    def update_weights_SGD(self, reward, q, next_q):

        delta = (reward + self.gamma*next_q - q)

        ###### en algun moment he d'actualitzar velocity i momentum
        self.velocity = self.velocity*self.momentum+self.scaled_lr*delta*self.z

        self.w = self.w + self.velocity*self.momentum + self.scaled_lr*delta*self.z


    def reset_z_v(self):
        for a in range(self.action_dim):
            self.z[a].fill(0.0)
            self.velocity[a].fill(0.0)

    def scale_state_variables(s, low=env.observation_space.low, high=env.observation_space.high):
        ''' Rescaling of s to the box [0,1]^2 '''
        x = (s - low) / (high - low)
        return x

    def sarsa_learning_step(self, state, action, reward, next_state, done):

        state, next_state = scale_state_variables(state), scale_state_variables(next_state)
        
        q = self.get_q_value(w, state, action)

        if not done:
            next_action = self.epsilon_greedy(next_state)
            next_q = self.get_q_value(w, next_action, next_state)
        else:
            next_q = 0.0

        self.update_eligibility_trace(z,q, action)

        self.update_weights_SGD(reward, q, next_q, z)

        if done:
            self.reset_z()




        
            














