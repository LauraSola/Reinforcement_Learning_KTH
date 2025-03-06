# Copyright [2024] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 2 for EL2805 - Reinforcement Learning.

# Load packages
import numpy as np
import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from DQN_agent import RandomAgent, Agent, ExperienceReplayBuffer, Experience, decay_epsilon
import torch.nn as nn
import torch.optim as optim
import copy
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

# Import and initialize the discrete Lunar Lander Environment
env = gym.make('LunarLander-v2')
# If you want to render the environment while training run instead:
# env = gym.make('LunarLander-v2', render_mode = "human")


env.reset()

# Parameters
N_episodes = 100 #100- 1000                             # Number of episodes
discount_factor = 0.95                       # Value of the discount factor
n_ep_running_average = 50                    # Running average of 50 episodes
n_actions = env.action_space.n               # Number of available actions
dim_state = len(env.observation_space.high)  # State dimensionality
type_decay = "exponential" #"exponential" / "linear"

# We will use these variables to compute the average episodic reward and
# the average number of steps per episode
episode_reward_list = []       # this list contains the total reward per episode
episode_number_of_steps = []   # this list contains the number of steps per episode

# Random agent initialization
input_size = 8
output_size = n_actions
num_hidden_layers = 1
hidden_size = 64
dueling = True

agent = Agent(n_actions, input_size, num_hidden_layers, hidden_size, dueling )

### Training process
optimizer = optim.Adam(agent.parameters(), lr=0.001) #0.001 - 0.0001
L = 5000 #5000 - 30000
N = 32 #4-128
buffer = ExperienceReplayBuffer(maximum_length=L) 

### fill buffer with random experiences
state = env.reset()[0]
for i in range(N):
    action = np.random.randint(0, n_actions)
    next_state, reward, done, truncated, _ = env.step(action)

    buffer.append(Experience(state, action, reward, next_state, done))

    if not done or truncated:
        state = next_state
    else:
        state = env.reset()[0]



clipping_value = 0.5 #0.5 - 2

# trange is an alternative to range in python, from the tqdm library
# It shows a nice progression bar that you can update with useful information
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

for i in EPISODES:
    # Reset enviroment data and initialize variables
    done, truncated = False, False
    state = env.reset()[0]
    total_episode_reward = 0.
    t = 0
    
    epsilon = decay_epsilon(i, N_episodes, type_decay)

   
    C = L/N
    while not (done or truncated):

        if t%C == 0:
            target_agent = copy.deepcopy(agent)


        q_values = agent.forward([state])

        action = agent.select_action(q_values, epsilon)#torch.argmax(q_values).item()

        # Get next state and reward
        next_state, reward, done, truncated, _ = env.step(action)

        # Update episode reward
        total_episode_reward += reward

        buffer.append(Experience(state, action, reward, next_state, done))

        ## sample a random batch of N experiences
        states, actions, rewards, next_states, dones = buffer.sample_batch(N)


        # Convert the batch data into tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)  # Unsqueeze for correct shape
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)


        #Take a random action
        q_values = agent.forward(states)
        q_values = q_values.gather(1, actions).squeeze()

        # Compute the target Q-values for the next states
        with torch.no_grad():  # No need to compute gradients for target Q-values
        
            next_q_values = target_agent.forward(next_states).max(1)[0]# Max Q-value for next state
            
            targets = rewards + 0.9 * next_q_values * (1 - dones)  # Target: Bellman equation

        loss = nn.functional.mse_loss(q_values, targets)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), clipping_value)  # Clip gradients to avoid exploding gradients
        optimizer.step()


        # Update state for next iteration
        state = next_state
        t+= 1

    

    # Append episode reward and total number of steps
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)


    # Updates the tqdm update bar with fresh information
    # (episode number, total reward of the last episode, total number of Steps
    # of the last episode, average reward, average number of steps)
    avg_reward_last_episodes = running_average(episode_reward_list, n_ep_running_average)[-1]
    EPISODES.set_description(
        "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
        i, total_episode_reward, t,
        avg_reward_last_episodes,
        running_average(episode_number_of_steps, n_ep_running_average)[-1]))
    
    if avg_reward_last_episodes > 50:
        break

# Close environment
env.close()