# Done by: 
# - Benet Ramió i Comas (20031026T177)
# - Laura Sola Garcia (20031119T225)


# Copyright [2024] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 2 for EL2805 - Reinforcement Learning.


# Load packages
import numpy as np
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim

class Actor(nn.Module):
    ''' Actor network for DDPG

        Args:
            n_actions (int): actions dimensionality

        Attributes:
            n_actions (int): where we store the dimensionality of an action

        The actor network is a neural network that takes the state as input and outputs
        the action. We suggest to use an actor network with 3 layers. Use 400 neurons in
        the first layer with ReLU activation, and 200 neurons in the second layer with ReLU
        activation. Remember to add the tanh activation to the output of the actor network
        to constraint the action to be between [−1, 1].
    '''
    def __init__(self, n_actions: int):
        super(Actor, self).__init__()
        self.n_actions = n_actions

        self.input_layer = nn.Linear(8, 400)
        self.hidden_layer1 = nn.Linear(400, 200)
        self.output_layer = nn.Linear(200, n_actions)
        self.activation = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, state: np.ndarray) -> np.ndarray:
        ''' Performs a forward computation '''
        # if is a numpy array, convert it to a tensor
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        
        x = self.activation(self.input_layer(state))
        x = self.activation(self.hidden_layer1(x))
        x = self.tanh(self.output_layer(x))
        return x
    

class Critic(nn.Module):
    ''' Critic network for DDPG

        Args:
            n_actions (int): actions dimensionality

        Attributes:
            n_actions (int): where we store the dimensionality of an action

        The critic network is a neural network that takes the state and action as input
        and outputs the Q-value. The critic network is similar to the actor network, but:
        (1) you feed only the state to the input layer; (2) you concatenate the output of
        the input layer with the action.
    '''
    def __init__(self, n_actions: int):
        super(Critic, self).__init__()
        self.n_actions = n_actions

        self.input_layer = nn.Linear(8, 400)
        self.hidden_layer1 = nn.Linear(400 + n_actions, 200)
        self.output_layer = nn.Linear(200, 1)
        self.activation = nn.ReLU()

    def forward(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        ''' Performs a forward computation '''
        x = self.activation(self.input_layer(state))
        x = torch.cat((x, action), dim=1)
        x = self.activation(self.hidden_layer1(x))
        x = self.output_layer(x)
        return x


class Agent(object):
    ''' Base agent class to DDPG

        Args:
            n_actions (int): actions dimensionality

        Attributes:
            n_actions (int): where we store the dimensionality of an action
            actor (Actor): actor network
            critic (Critic): critic network
            target_actor (Actor): target actor network
            target_critic (Critic): target critic network


    '''
    def __init__(self, n_actions: int):
        self.n_actions = n_actions

        # Initialize the networks
        self.actor = Actor(n_actions)
        self.critic = Critic(n_actions)
        self.target_actor = Actor(n_actions)
        self.target_critic = Critic(n_actions)

        # Copy the weights of the actor and critic networks to the target networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    # def forward(self, state: np.ndarray):
    #     ''' Performs a forward computation '''
    #     pass

    def forward_actor(self, state: np.ndarray) -> np.ndarray:
        ''' Forward pass on the actor network '''
        return self.actor.forward(state)
    
    def forward_target_actor(self, state: np.ndarray) -> np.ndarray:
        ''' Forward pass on the target actor network '''
        return self.target_actor.forward(state)
    
    def forward_critic(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        ''' Forward pass on the critic network '''
        return self.critic.forward(state, action)
    
    def forward_target_critic(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        ''' Forward pass on the target critic network '''
        return self.target_critic.forward(state, action)
    
    def backward_critic(self, states, actions, rewards, next_states, dones, discount_factor, optimizer, clipping_value):
        """Perform a backward pass for the critic network."""
        # Compute the target Q value
        with torch.no_grad():
            target_q_values = rewards + discount_factor * self.forward_target_critic(next_states, self.forward_target_actor(next_states)) * (1 - dones)

        q_values = self.forward_critic(states, actions)
        loss_critic = nn.MSELoss()(q_values, target_q_values)

        # Backpropagate the critic loss
        optimizer.zero_grad()
        loss_critic.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), clipping_value)
        optimizer.step()

    def backward_actor(self, states, optimizer, clipping_value):
        """Perform a backward pass for the actor network."""
        loss_actor = -self.forward_critic(states, self.forward_actor(states)).mean()

        # Backpropagate the actor loss
        optimizer.zero_grad()
        loss_actor.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), clipping_value)
        optimizer.step()




class RandomAgent(Agent):
    ''' Agent taking actions uniformly at random, child of the class Agent'''
    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__(n_actions)

    def forward(self, state: np.ndarray) -> np.ndarray:
        ''' Compute a random action in [-1, 1]

            Returns:
                action (np.ndarray): array of float values containing the
                    action. The dimensionality is equal to self.n_actions from
                    the parent class Agent.
        '''
        return np.clip(-1 + 2 * np.random.rand(self.n_actions), -1, 1) # Random action in [-1, 1]


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



class OrnsteinUhlenbeckNoise:
    """
    Ornstein-Uhlenbeck noise generator for temporally correlated noise.

    Attributes:
        action_dim (int): Dimensionality of the action space.
        mu (float): Decay factor for the noise (default 0.15, should be in [0, 1)).
        sigma (float): Standard deviation of the noise.
        state (np.ndarray): Internal state for generating correlated noise.
    """
    
    def __init__(self, action_dim, mu=0.15, sigma=0.2):
        """
        Initializes the noise generator with the given parameters.
        
        Parameters:
            action_dim (int): Dimensionality of the action space.
            mu (float): Decay factor for the noise (default 0.15).
            sigma (float): Standard deviation of the noise.
        """
        self.action_dim = action_dim
        self.mu = mu
        self.sigma = sigma
        self.state = np.zeros(action_dim)
    
    def reset(self):
        """
        Resets the internal state to zeros.
        """
        self.state = np.zeros(self.action_dim)
    
    def generate(self):
        """
        Generates Ornstein-Uhlenbeck noise and updates the internal state.

        Returns:
            np.ndarray: Noise sample of shape (action_dim,).
        """
        noise = np.random.normal(0, self.sigma, self.action_dim)
        self.state = -self.mu * self.state + noise
        return self.state

