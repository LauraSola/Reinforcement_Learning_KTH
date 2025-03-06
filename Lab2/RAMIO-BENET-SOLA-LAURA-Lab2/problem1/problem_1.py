# Done by: 
# - Benet Ramió i Comas (20031026T177)
# - Laura Sola Garcia (20031119T225)


from DQN_agent import Agent, ExperienceReplayBuffer, Experience, decay_epsilon
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
import random
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

        
        self.N_episodes = 800 #100-1000
        self.discount_factor = 0.99
        self.type_decay = "exponential"
        self.L = 10000  #buffer size 5000-30000
        self.N = 128  #training batch 4-128
        self.learning_rate = 0.0001  #10^-3 - 10^-4
        self.clipping_value = 1 #0.5 - 2
        self.n_hidden_layers = 1  #1-2
        self.hidden_size = 64  #8-128
        self.dueling = False 
        self.CER = True 

        ###
        self.n_ep_running_average = 50
        self.label = None

        ###
        self.env = gym.make('LunarLander-v2') # change to v2 if v3 does not work
        self.n_actions = self.env.action_space.n               # Number of available actions
        self.dim_state = len(self.env.observation_space.high)  # State dimensionality

        ## keep best results
        self.best_agents = {"agent": [], "avg_reward":[], "label":[]}

        ## store results
        self.avg_rewards = []
        self.avg_steps = []


    def update_best_results(self, agent, avg_reward_last, size_lists = 3):
        # save the best agents
        if len(self.best_agents["agent"]) < size_lists:
            self.best_agents["agent"].append(agent)
            self.best_agents["avg_reward"].append(avg_reward_last)
            self.best_agents["label"].append(self.label)
        else:
            if avg_reward_last > min(self.best_agents["avg_reward"]):
                idx = self.best_agents["avg_reward"].index(min(self.best_agents["avg_reward"]))
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

    def plot_smoothed_results(self,results, label):
        """Plot smoothed rewards with confidence intervals."""
    
        x_vals = np.arange(len(results))
        plt.plot(x_vals, results, label=label)
        plt.xlabel("Episodes")
        
        
        if label:
            plt.legend(loc='upper left')

    def plot_all_smoothed_rewards(self, title, name_file, labels=None):
        """Plot smoothed rewards with confidence intervals """
        plt.figure(figsize=(12, 6)) 
        plt.grid()
        plt.title(title)
        plt.ylabel("Average Reward")
        for i, rewards in enumerate(self.avg_rewards):
            label = labels[i] if labels else None
            self.plot_smoothed_results(rewards, label)

        plt.savefig(f"{current_dir}/figs/rewards_{name_file}.png")
        plt.show()

    def plot_all_smoothed_steps(self, title, name_file, labels=None):
        """Plot smoothed rewards with confidence intervals """
        plt.figure(figsize=(12, 6)) 
        plt.grid()
        plt.title(title)
        plt.ylabel("Average Number of Steps")
        for i, steps in enumerate(self.avg_steps):
            label = labels[i] if labels else None
            self.plot_smoothed_results(steps, label)

        plt.savefig(f"{current_dir}/figs/steps_{name_file}.png")
        plt.show()


    def initialize_buffer(self):

        buffer = ExperienceReplayBuffer(maximum_length=self.L, CER = self.CER) 

        ### fill buffer with random experiences
        state = self.env.reset()[0]
        for i in range(int(0.2*self.L)):
            action = np.random.randint(0, self.n_actions)
            next_state, reward, done, truncated, _ = self.env.step(action)

            buffer.append(Experience(state, action, reward, next_state, done or truncated))

            if not (done or truncated):
                state = next_state
            else:
                state = self.env.reset()[0]

        return buffer
    
    def select_next_action(self, state, epsilon, agent):
        with torch.no_grad():
            q_values = agent.forward([state])
            if random.random()<epsilon:
                action = np.random.randint(0, self.n_actions)

            else:
                action = torch.argmax(q_values).item()

            return action



    def train_DQN(self, agent, show = False):

        self.env.reset()

        buffer = self.initialize_buffer()
        optimizer = optim.Adam(agent.parameters(), lr=self.learning_rate) 

        episode_reward_list = []       # this list contains the total reward per episode
        episode_number_of_steps = []   # this list contains the number of steps per episode

        best_avg_reward = -1000 #arbitrary low value
        best_model = copy.deepcopy(agent)

        EPISODES = trange(self.N_episodes, desc='Episode: ', leave=True)

        for i in EPISODES:
            # Reset enviroment data and initialize variables
            done, truncated = False, False
            state = self.env.reset()[0]
            total_episode_reward = 0.
            t = 0

            epsilon = decay_epsilon(i, self.N_episodes, self.type_decay)

        
            C = self.L//self.N

            while not (done or truncated):

                if t%C == 0:
                    target_agent = copy.deepcopy(agent)


                action = self.select_next_action(state, epsilon, agent)

                # Get next state and reward
                next_state, reward, done, truncated, _ = self.env.step(action)

                # Update episode reward
                total_episode_reward += reward

                buffer.append(Experience(state, action, reward, next_state, done or truncated))

                ## sample a random batch of N experiences
                states, actions, rewards, next_states, dones = buffer.sample_batch(self.N)


                # Convert the batch data into tensors
                states = torch.tensor(np.array(states), dtype=torch.float64)
                actions = torch.tensor(np.array(actions), dtype=torch.int64).unsqueeze(1)  # Unsqueeze for correct shape
                rewards = torch.tensor(np.array(rewards), dtype=torch.float64)
                next_states = torch.tensor(np.array(next_states), dtype=torch.float64)
                dones = torch.tensor(np.array(dones), dtype=torch.int64)


                #Take a random action
                q_values = agent.forward(states)                
                q_values = q_values.gather(1, actions).squeeze()



                # Compute the target Q-values for the next states
                with torch.no_grad():  # No need to compute gradients for target Q-values
                
                    next_q_values = target_agent.forward(next_states).max(1)[0]# Max Q-value for next state

                    targets = rewards + self.discount_factor * next_q_values * (1 - dones)  # Target: Bellman equation
                    

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

            if avg_reward_last_episodes[-1]>best_avg_reward:
                best_avg_reward = avg_reward_last_episodes[-1]
                best_model = copy.deepcopy(agent)


            EPISODES.set_description(
                "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
                i, total_episode_reward, t,
                avg_reward_last_episodes[-1],
                avg_steps_last_episodes[-1]))
           
        self.avg_rewards.append(avg_reward_last_episodes)
        self.avg_steps.append(avg_steps_last_episodes)
        print(f"Training finished with average reward of {avg_reward_last_episodes[-1]}")
        print(f"Training finished with the best average reward of {best_avg_reward}")
        self.update_best_results(best_model, best_avg_reward)
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
            ).double()

            self.train_DQN(agent)

        
        os.makedirs(f"{current_dir}/models", exist_ok=True)

        for i in range(len(self.best_agents["agent"])):
            print(f"Average reward = {self.best_agents['avg_reward'][i]}")
            print(f"Model = {self.best_agents['label'][i]}")
            torch.save(self.best_agents['agent'][i], f"{current_dir}/models/{self.best_agents['label'][i]}.pth")
            


## FUNCTIONS FOR THE DIFFERENT EXERCISES OF THE FIRST SECTION


def hyperparameter_tuning():

    hyperparameters = {
        "N_episodes": [800],#[100, 1000],
        "discount_factor": 0.99, #[0.5, 0.7, 0.9, 0.95, 0.99],
        "type_decay": "exponential",  #["exponential", "linear"],
        "L": 10000, #[5000, 10000, 20000, 30000],
        "N": 128, #[4, 16, 32, 64, 128],
        "learning_rate": 0.0001, #[0.001, 0.0005, 0.0001],
        "clipping_value": 1, #[0.5, 1, 2],
        "n_hidden_layers": 1, #[1, 2],
        "hidden_size": 64, #[8, 16, 32, 64, 128],
        "dueling": False, #[True, False],
        "CER": True, #[True, False]
    }

    trainer = Trainer()
    trainer.hyperparameter_search(hyperparameters)


def discount_factor():
    disc_factor = {
        "discount_factor": [0.5, 0.99, 1]
    }

    trainer = Trainer()

    trainer.hyperparameter_search(disc_factor)
    trainer.plot_all_smoothed_rewards("Effect of discount factor on average reward", "disc_factor", ["Discount Factor = 0.5", "Discount Factor = 0.99", "Discount Factor = 1"])
    trainer.plot_all_smoothed_steps("Effect of discount factor on average number of steps", "disc_factor", ["Discount Factor = 0.5", "Discount Factor = 0.99", "Discount Factor = 1"])

def num_episodes():
    n_eps = {
        "N_epsiodes": [400, 600, 800, 1000]
    }

    trainer = Trainer()

    trainer.hyperparameter_search(n_eps)
    trainer.plot_all_smoothed_rewards("Effect of the number of episodes on the average reward", "eps", ["Num Episodes = 400", "Num Episodes = 600", "Num Episodes = 800", "Num Episodes = 1000"])
    trainer.plot_all_smoothed_steps("Effect of the number of episodes on the average number of steps", "eps", ["Num Episodes = 400", "Num Episodes = 600", "Num Episodes = 800", "Num Episodes = 1000"])


def mem_size():

    mem = {
        "L": [5000, 10000, 20000]
    }

    trainer = Trainer()

    trainer.hyperparameter_search(mem)
    trainer.plot_all_smoothed_rewards("Effect of the memory size on the average reward", "eps", ["Memory Size = 5000", "Memory Size = 10000", "Memory Size = 20000"])
    trainer.plot_all_smoothed_steps("Effect of the memory size on the average number of steps", "eps", ["Memory Size = 5000", "Memory Size = 10000", "Memory Size = 20000"])

    
if __name__ == '__main__':
    task = "hyperparameter tuning"
    
    if task == "hyperparameter tuning":
        hyperparameter_tuning()
    elif task == "discount factor":
        discount_factor()
    elif task == "number episodes":
        num_episodes()
    elif task == "memory size":
        mem_size()
        


