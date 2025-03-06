
from DQN_agent import RandomAgent, Agent, ExperienceReplayBuffer, Experience, decay_epsilon
import copy
import numpy as np
import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from tqdm import trange
import torch.nn as nn
import torch.optim as optim
import itertools
import warnings
import os
warnings.simplefilter(action='ignore', category=FutureWarning)
current_dir = os.path.dirname(os.path.abspath(__file__))



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

class Trainer:


    def __init__(self):

        # CAMBIAR A DEFAULT EL MILLOR QUAN EL TROBEM
        self.N_episodes = None #100-1000
        self.discount_factor = None
        self.type_decay = None
        self.L = None  #buffer size 5000-30000
        self.N = None  #training batch 4-128
        self.learning_rate = None  #10^-3 - 10^-4
        self.clipping_value = None #0.5 - 2
        self.n_hidden_layers = None  #1-2
        self.hidden_size = None  #8-128
        self.dueling = None 
        self.CER = None 

        ###
        self.n_ep_running_average = 50
        self.label = None

        ###
        self.env = gym.make('LunarLander-v3') # change to v2 if v3 does not work
        self.n_actions = self.env.action_space.n               # Number of available actions
        self.dim_state = len(self.env.observation_space.high)  # State dimensionality

        ## keep best results
        self.best_agents = {"agent": [], "avg_reward":[], "label":[]}


    def update_best_results(self, agent, avg_reward_last, size_lists = 3):
        # save the best agents
        if len(self.best_agents["agent"]) < size_lists:
            self.best_agents["agent"].append(agent)
            self.best_agents["avg_reward"].append(avg_reward_last)
            self.best_agents["label"].append(self.label)
        else:
            if avg_reward_last > min(self.best_agents["avg_reward"]):
                idx = self.best_agents["avg_reward"].index(min(self.best_agents_rewards["avg_reward"]))
                self.best_agents["agent"][idx] = agent
                self.best_agents["avg_reward"][idx] = avg_reward_last
                self.best_agents["label"][idx] = self.label


    def plot(self, episode_reward_list, episode_number_of_steps, avg_reward_last_episodes, avg_steps_last_episodes, show = False):
        # Ensure the directory exists in the current path
        os.makedirs(f"{current_dir}/figs", exist_ok=True)
        
        # Plot Rewards and steps
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
        ax[0].plot([i for i in range(1, self.N_episodes+1)], episode_reward_list, label='Episode reward')
        ax[0].plot([i for i in range(1, self.N_episodes+1)], avg_reward_last_episodes, label='Avg. episode reward')
        ax[0].set_xlabel('Episodes')
        ax[0].set_ylabel('Total reward')
        ax[0].set_title('Total Reward vs Episodes')
        ax[0].legend()
        ax[0].grid(alpha=0.3)

        ax[1].plot([i for i in range(1, self.N_episodes+1)], episode_number_of_steps, label='Steps per episode')
        ax[1].plot([i for i in range(1, self.N_episodes+1)],avg_steps_last_episodes, label='Avg. number of steps per episode')
        ax[1].set_xlabel('Episodes')
        ax[1].set_ylabel('Total number of steps')
        ax[1].set_title('Total number of steps vs Episodes')
        ax[1].legend()
        ax[1].grid(alpha=0.3)

        plt.savefig(f"{current_dir}/figs/{self.label}.png")

        if show:
            plt.show()


    def initialize_buffer(self):

        buffer = ExperienceReplayBuffer(maximum_length=self.L, CER = self.CER) 

        ### fill buffer with random experiences
        state = self.env.reset()[0]
        for i in range(self.N):
            action = np.random.randint(0, self.n_actions)
            next_state, reward, done, truncated, _ = self.env.step(action)

            buffer.append(Experience(state, action, reward, next_state, done))

            if not done or truncated:
                state = next_state
            else:
                state = self.env.reset()[0]

        return buffer



    def train_DQN(self, agent, show = False):

        self.env.reset()

        buffer = self.initialize_buffer()
        optimizer = optim.Adam(agent.parameters(), lr=self.learning_rate) 

        episode_reward_list = []       # this list contains the total reward per episode
        episode_number_of_steps = []   # this list contains the number of steps per episode


        EPISODES = trange(self.N_episodes, desc='Episode: ', leave=True)

        for i in EPISODES:
            # Reset enviroment data and initialize variables
            done, truncated = False, False
            state = self.env.reset()[0]
            total_episode_reward = 0.
            t = 0
            
            epsilon = decay_epsilon(i, self.N_episodes, self.type_decay)

        
            C = self.L/self.N

            while not (done or truncated):

                if t%C == 0:
                    target_agent = copy.deepcopy(agent)


                q_values = agent.forward([state])

                action = agent.select_action(q_values, epsilon)#torch.argmax(q_values).item()

                # Get next state and reward
                next_state, reward, done, truncated, _ = self.env.step(action)

                # Update episode reward
                total_episode_reward += reward

                buffer.append(Experience(state, action, reward, next_state, done))

                ## sample a random batch of N experiences
                states, actions, rewards, next_states, dones = buffer.sample_batch(self.N)


                # Convert the batch data into tensors
                states = torch.tensor(np.array(states), dtype=torch.float32)
                actions = torch.tensor(np.array(actions), dtype=torch.int64).unsqueeze(1)  # Unsqueeze for correct shape
                rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
                next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
                dones = torch.tensor(np.array(dones), dtype=torch.float32)


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
                nn.utils.clip_grad_norm_(agent.parameters(), self.clipping_value)  # Clip gradients to avoid exploding gradients
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
            avg_reward_last_episodes = running_average(episode_reward_list, self.n_ep_running_average)#[-1]
            avg_steps_last_episodes = running_average(episode_number_of_steps, self.n_ep_running_average)#[-1]
            EPISODES.set_description(
                "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
                i, total_episode_reward, t,
                avg_reward_last_episodes[-1],
                avg_steps_last_episodes[-1]))
            
            if avg_reward_last_episodes[-1] > 50:
                print("Early Stopping, exiting training")
                break
        print(f"Training finished with average reward of {avg_reward_last_episodes[-1]}")
        self.update_best_results(agent, avg_reward_last_episodes[-1])
        self.plot(episode_reward_list, episode_number_of_steps, avg_reward_last_episodes, avg_steps_last_episodes, show)


    def hyperparameter_search(self, hyperparameters):

        # Separate fixed and variable hyperparameters
        fixed_hyperparams = {k: v for k, v in hyperparameters.items() if not isinstance(v, list)}
        variable_hyperparams = {k: v for k, v in hyperparameters.items() if isinstance(v, list)}


        # Generate all combinations of variable hyperparameters
        keys, values = zip(*variable_hyperparams.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        # Loop over all hyperparameter combinations
        for i, params in enumerate(param_combinations):
            print(f"Training model {i+1}/{len(param_combinations)}")

            # Combine fixed and variable hyperparameters
            current_hyperparams = {**fixed_hyperparams, **params}

            for key, value in current_hyperparams.items():
                setattr(self, key, value)

            
            self.label = "_".join(f"{key}={value}" for key, value in current_hyperparams.items())
            print(f"Training model with hyperparameters: {self.label}")

            # Initialize DQN agent
            agent = Agent(
                n_actions = self.n_actions,
                input_size = self.dim_state,
                n_hidden_layers = self.n_hidden_layers,
                hidden_size = self.hidden_size,
                dueling = self.dueling
            )

            self.train_DQN(agent)

        os.makedirs(f"{current_dir}/models", exist_ok=True)

        for i in range(len(self.best_agents["agent"])):
            print(f"Average reward = {self.best_agents['avg_reward'][i]}")
            print(f"Model = {self.best_agents['label'][i]}")
            torch.save(self.best_agents['agent'][i], f"{current_dir}/models/{self.best_agents['label'][i]}.pt")
            
            

if __name__ == "__main__":

    hyperparameters = {
        "N_episodes": [100, 101],
        "discount_factor": 0.9, #[0.5, 0.7, 0.9, 0.95, 0.99],
        "type_decay": "exponential",  #["exponential", "linear"],
        "L": 5000, #[5000, 10000, 20000, 30000],
        "N": [4,16], #[4, 16, 32, 64, 128],
        "learning_rate": 0.001, #[0.001, 0.0005, 0.0001],
        "clipping_value": 0.5, #[0.5, 1, 2],
        "n_hidden_layers": 1, #[1, 2],
        "hidden_size": 8, #[8, 16, 32, 64, 128],
        "dueling": True, #[True, False],
        "CER": True, #[True, False]
    }

    trainer = Trainer()
    trainer.hyperparameter_search(hyperparameters)


