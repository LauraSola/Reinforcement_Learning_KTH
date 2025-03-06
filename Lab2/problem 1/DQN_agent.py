# Copyright [2024] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 2 for EL2805 - Reinforcement Learning.


# Load packages
import numpy as np
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import random


class Agent(nn.Module):
    ''' Base agent class, used as a parent class

        Args:
            n_actions (int): number of actions

        Attributes:
            n_actions (int): where we store the number of actions
            last_action (int): last action taken by the agent
    '''
    def __init__(self, n_actions: int, input_size:int, n_hidden_layers = 2, hidden_size = 64, dueling = False):
        super().__init__()
        self.n_actions = n_actions
        self.last_action = None
        self.n_hidden_layers = n_hidden_layers
        self.dueling = dueling

     
        self.input_layer = nn.Linear(input_size, hidden_size)  # First layer: state -> hidden layer

        self.activation = nn.ReLU()  # ReLU activation function for hidden layers
        
        self.hidden_layers = nn.ModuleList()

        for _ in range(self.n_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))

        if not dueling:
            self.output_layer = nn.Linear(hidden_size, self.n_actions)  # Output layer: hidden -> Q-values

        if dueling:
            self.value_funct_layer = nn.Linear(hidden_size, 1)

            self.advantage_layer = nn.Linear(hidden_size, self.n_actions)
        
        


    def forward(self, state: np.ndarray):
        ''' Performs a forward computation '''

        #print(state.shape)
        state = torch.tensor(state, dtype=torch.float32)

        x = self.activation(self.input_layer(state))  # Apply input layer and ReLU

        for layer in self.hidden_layers:
            x = self.activation(layer(x))  # Apply hidden layer and ReLU

        if not self.dueling:
            return  self.output_layer(x)  # Return Q-values for all actions
        
        else:
            advantage = self.advantage_layer(x)

            #print(advantage.shape)            
            value_funct = self.value_funct_layer(x) 
            #print(value_funct.shape) 
            advAvg = torch.mean(advantage, dim=1, keepdim=True)
            return value_funct + advantage - advAvg
    
        


    def backward(self):
        ''' Performs a backward pass on the network '''

        ### if C steps have passed --> copy
        #new_model = copy.deepcopy(model)


        pass

    
    def select_action(self, q_values, epsilon):

        if random.random()<epsilon:
                action = np.random.randint(0, self.n_actions)

        else:
            action = torch.argmax(q_values).item()

        return action


class RandomAgent(Agent):
    ''' Agent taking actions uniformly at random, child of the class Agent'''
    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__(n_actions)

    def forward(self, state: np.ndarray) -> int:
        ''' Compute an action uniformly at random across n_actions possible
            choices

            Returns:
                action (int): the random action
        '''
        self.last_action = np.random.randint(0, self.n_actions)
        return self.last_action


# Define Experience tuple
# Experience represents a transition in the environment, including the current state, action taken,
# received reward, next state, and whether the episode is done.
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ExperienceReplayBuffer:
    """Replay buffer for storing experiences.
    
       The experience replay buffer stores past experiences so that the agent can learn from them later.
       By sampling randomly from these experiences, the agent avoids overfitting to the most recent 
       transitions and helps stabilize training.
       - The buffer size is limited, and older experiences are discarded to make room for new ones.
       - Experiences are stored as tuples of (state, action, reward, next_state, done).
       - A batch of experiences is sampled randomly during each training step for updating the Q-values."""

    def __init__(self, maximum_length, CER):
        self.buffer = deque(maxlen=maximum_length)  # Using deque ensures efficient removal of oldest elements
        self.CER = CER

    def append(self, experience):
        """Add a new experience to the buffer"""
        self.buffer.append(experience)

    def __len__(self):
        """Return the current size of the buffer"""
        return len(self.buffer)

    def sample_batch(self, n):
        """Randomly sample a batch of experiences with combined experience replay"""
        if n > len(self.buffer):
            raise IndexError('Sample size exceeds buffer size!')
        
        if self.CER:
            indices = np.random.choice(len(self.buffer)-1, size=n-1, replace=False)  # Random sampling
            batch = [self.buffer[i] for i in indices]  # Create a batch from sampled indices
            batch.append(self.buffer[-1])  
            
        
        else:
            indices = np.random.choice(len(self.buffer), size=n, replace=False)  # Random sampling
            batch = [self.buffer[i] for i in indices]  # Create a batch from sampled indices

        return zip(*batch)  # Unzip batch into state, action, reward, next_state, and done




def decay_epsilon(k, n_episodes, type_decay, e_max = 0.99, e_min = 0.05):

    z = 0.95*n_episodes #0.9 - 0.95

    if type_decay == "linear":

        aux = (e_max - e_min)*(k-1)/(z-1)
        return max(e_min, e_max - aux)

    elif type_decay == "exponential":

        return max(e_min, e_max*(e_min/e_max)**((k-1)/(z-1)))

    else:
        raise ValueError("Not valid decay type")
    

