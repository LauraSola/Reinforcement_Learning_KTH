# Copyright [2024] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 2 for EL2805 - Reinforcement Learning.


# Load packages
import numpy as np
import gymnasium as gym
import torch, torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import trange
from DDPG_agent import RandomAgent, Agent, ExperienceReplayBuffer, Experience, OrnsteinUhlenbeckNoise
from DDPG_soft_updates import soft_updates
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os


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

def initialize_buffer(L, env, m):
    ''' Function used to initialize the buffer with random experiences
    '''
    buffer = ExperienceReplayBuffer(maximum_length=L, CER=False)
    state = env.reset()[0]
    for _ in range(L):
        action = np.clip(-1 + 2 * np.random.rand(m), -1, 1)
        next_state, reward, done, truncated, _ = env.step(action)
        buffer.append(Experience(state, action, reward, next_state, done or truncated))
        if done:
            state = env.reset()[0]
        else:
            state = next_state
    return buffer


class Trainer_DDGP:
    def __init__(self, TE=300, discount_factor=0.99, n_ep_running_average=50, clipping_value=1, tau=10e-3, L=30000, N=64, d=2, lr_actor=5e-5, lr_critic=5e-4, noise_mu=0.15, noise_sigma=0.2):
        self.TE = TE
        self.discount_factor = discount_factor
        self.n_ep_running_average = n_ep_running_average
        self.clipping_value = clipping_value
        self.tau = tau
        self.L = L
        self.N = N
        self.d = d
        
        self.env = gym.make('LunarLanderContinuous-v3')
        self.env.reset()

        self.m = len(self.env.action_space.high)
        self.n = len(self.env.observation_space.high)
        
        self.agent = Agent(self.m)

        self.actor_optimizer = torch.optim.Adam(self.agent.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.agent.critic.parameters(), lr=lr_critic)
        self.buffer = initialize_buffer(self.L, self.env, self.m)

        self.noise_generator = OrnsteinUhlenbeckNoise(action_dim=self.m, mu=noise_mu, sigma=noise_sigma)

        self.best_agent = (None, None, 0, float('-inf'), 0) # Best agent: (Actor, Critic, Episode, Avg. Reward, Avg. Steps)
        self.episode_reward_list = []  # Used to save episodes reward
        self.episode_number_of_steps = []

    def train(self):
        EPISODES = trange(self.TE, desc='Episode: ', leave=True)

        for i in EPISODES:
            done, truncated = False, False
            state = self.env.reset()[0]
            total_episode_reward = 0.
            t = 0
            self.noise_generator.reset()

            while not (done or truncated):
                # Take action
                noise = self.noise_generator.generate()
                action = (self.agent.forward_actor(state)).detach().numpy() + noise
                action = np.clip(action, -1, 1)

                # Get next state and reward, update episode reward and append experience to the buffer
                next_state, reward, done, truncated, _ = self.env.step(action)
                total_episode_reward += reward
                self.buffer.append(Experience(state, action, reward, next_state, done or truncated))

                # Sample a random batch of N transitions and convert them to PyTorch tensors
                states, actions, rewards, next_states, dones = self.buffer.sample_batch(self.N)
                states = torch.tensor(np.array(states), dtype=torch.float)
                actions = torch.tensor(np.array(actions), dtype=torch.float)
                rewards = torch.tensor(np.array(rewards), dtype=torch.float).view(self.N, 1)
                next_states = torch.tensor(np.array(next_states), dtype=torch.float)
                dones = torch.tensor(np.array(dones), dtype=torch.float).view(self.N, 1)

                # Update the critic
                self.agent.backward_critic(states, actions, rewards, next_states, dones, self.discount_factor, self.critic_optimizer, self.clipping_value)

                # Update the actor and target networks every 'd' steps
                if t % self.d == 0:
                    self.agent.backward_actor(states, self.actor_optimizer, self.clipping_value)
                    
                    self.agent.target_actor = soft_updates(self.agent.actor, self.agent.target_actor, self.tau)
                    self.agent.target_critic = soft_updates(self.agent.critic, self.agent.target_critic, self.tau)            

                # Update state for next iteration
                state = next_state
                t+= 1

            # Append episode reward
            self.episode_reward_list.append(total_episode_reward)
            self.episode_number_of_steps.append(t)

            # Average metrics
            avg_reward = running_average(self.episode_reward_list, self.n_ep_running_average)[-1]
            avg_steps = running_average(self.episode_number_of_steps, self.n_ep_running_average)[-1]

            # Updates the tqdm update bar with fresh information
            # (episode number, total reward of the last episode, total number of Steps
            # of the last episode, average reward, average number of steps)
            EPISODES.set_description(
                "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
                i, total_episode_reward, t, avg_reward, avg_steps))

            # Save best agent
            if avg_reward > self.best_agent[3]:
                self.best_agent = (self.agent.actor, self.agent.critic, i, avg_reward, avg_steps)

        # Close environment
        self.env.close()

        # Print and save best agent
        print(f"Best agent found at episode {self.best_agent[2]} with an average reward of {self.best_agent[3]:.2f} and an average number of steps of {self.best_agent[4]}")
        current_path = os.path.dirname(os.path.abspath(__file__))
        if not os.path.exists(current_path + '/outputs'):
            os.makedirs(current_path + '/outputs')
        torch.save(self.best_agent[0], f"{current_path}/outputs/neural-network-2-actor.pth")
        torch.save(self.best_agent[1], f"{current_path}/outputs/neural-network-2-critic.pth")


    def plot(self, return_variables=False):
        if return_variables:
            return self.TE, self.episode_reward_list, self.episode_number_of_steps, self.n_ep_running_average, self.best_agent
        else:
            current_path = os.path.dirname(os.path.abspath(__file__))
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
            ax[0].plot([i for i in range(1, self.TE+1)], self.episode_reward_list, label='Episode reward')
            ax[0].plot([i for i in range(1, self.TE+1)], running_average(
                self.episode_reward_list, self.n_ep_running_average), label='Avg. episode reward')
            ax[0].set_xlabel('Episodes')
            ax[0].set_ylabel('Total reward')
            ax[0].set_title('Total Reward vs Episodes')
            ax[0].set_ylim(-800, 400)
            ax[0].legend()
            ax[0].grid(alpha=0.3)

            ax[1].plot([i for i in range(1, self.TE+1)], self.episode_number_of_steps, label='Steps per episode')
            ax[1].plot([i for i in range(1, self.TE+1)], running_average(
                self.episode_number_of_steps, self.n_ep_running_average), label='Avg. number of steps per episode')
            ax[1].set_xlabel('Episodes')
            ax[1].set_ylabel('Total number of steps')
            ax[1].set_title('Total number of steps vs Episodes')
            ax[1].legend()
            ax[1].grid(alpha=0.3)
            plt.savefig(f"{current_path}/outputs/rewards-steps-2-{self.best_agent[3]:.2f}.png")
            plt.show()

def plot_varing_hyperparameter(discount_factor_plot, label_plot='Discount factor', hyperparameter_name='discount_factors'):
    # Plot Rewards and steps
    current_path = os.getcwd()
    if not os.path.exists(f"{current_path}/outputs"):
        os.makedirs(f"{current_path}/outputs")

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
    for i in range(len(discount_factor_plot[f'{hyperparameter_name}'])):
        ax[0].plot([i for i in range(1, discount_factor_plot['TEs'][i]+1)], running_average(
            discount_factor_plot['episode_reward_lists'][i], discount_factor_plot['n_ep_running_averages'][i]), label=f'{label_plot}: {discount_factor_plot[f"{hyperparameter_name}"][i]}')
        ax[1].plot([i for i in range(1, discount_factor_plot['TEs'][i]+1)], running_average(
            discount_factor_plot['episode_number_of_stepss'][i], discount_factor_plot['n_ep_running_averages'][i]), label=f'{label_plot}: {discount_factor_plot[f"{hyperparameter_name}"][i]}')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Total reward')
    ax[0].set_title('Total Reward vs Episodes')
    ax[0].set_ylim(-800, 400) #limit the y-axes
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].set_xlabel('Episodes')
    ax[1].set_ylabel('Total number of steps')
    ax[1].set_title('Total number of steps vs Episodes')
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    plt.savefig(f"{current_path}/outputs/rewards-steps-2-{hyperparameter_name}.png")
    plt.show()    

class Experiments:
    def __init__(self):
        self.current_path = os.path.dirname(os.path.abspath(__file__))

    def task_e1(self):
        print("Training with default parameters")
        trainer = Trainer_DDGP()
        trainer.train()
        trainer.plot()

    def task_e2(self):
        print("Training with different discount factors")
        print("Discount factor = 0.99")
        trainer = Trainer_DDGP(discount_factor=0.99)
        trainer.train()
        TE, episode_reward_list, episode_number_of_steps, n_ep_running_average, best_agent = trainer.plot(return_variables=True)

        print("Discount factor = 1")
        trainer1 = Trainer_DDGP(discount_factor=1)
        trainer1.train()
        TE_1, episode_reward_list_1, episode_number_of_steps_1, n_ep_running_average_1, best_agent_1 = trainer1.plot(return_variables=True)

        print("Discount factor = 0.5")
        trainer2 = Trainer_DDGP(discount_factor=0.5)
        trainer2.train()
        TE_2, episode_reward_list_2, episode_number_of_steps_2, n_ep_running_average_2, best_agent_2 = trainer2.plot(return_variables=True)

        print("Plotting results")
        discount_factor_plot = {
            'discount_factors': [0.99, 1, 0.5],
            'TEs': [TE, TE_1, TE_2],
            'episode_reward_lists': [episode_reward_list, episode_reward_list_1, episode_reward_list_2],
            'episode_number_of_stepss': [episode_number_of_steps, episode_number_of_steps_1, episode_number_of_steps_2],
            'n_ep_running_averages': [n_ep_running_average, n_ep_running_average_1, n_ep_running_average_2],
            'best_agents': [best_agent, best_agent_1, best_agent_2],
            }
        
        plot_varing_hyperparameter(discount_factor_plot, label_plot='Discount factor', hyperparameter_name='discount_factors')

    def task_e3(self):
        print("Training with different memory sizes")
        print("Memory size = 30000")
        trainer = Trainer_DDGP(L=30000)
        trainer.train()
        TE, episode_reward_list, episode_number_of_steps, n_ep_running_average, best_agent = trainer.plot(return_variables=True)

        print("Memory size = 5000")
        trainer1 = Trainer_DDGP(L=1000)
        trainer1.train()
        TE_1, episode_reward_list_1, episode_number_of_steps_1, n_ep_running_average_1, best_agent_1 = trainer1.plot(return_variables=True)

        print("Memory size = 50000")
        trainer2 = Trainer_DDGP(L=50000)
        trainer2.train()
        TE_2, episode_reward_list_2, episode_number_of_steps_2, n_ep_running_average_2, best_agent_2 = trainer2.plot(return_variables=True)

        print("Plotting results")
        memory_size_plot = {
            'memory_sizes': [30000, 1000, 50000],
            'TEs': [TE, TE_1, TE_2],
            'episode_reward_lists': [episode_reward_list, episode_reward_list_1, episode_reward_list_2],
            'episode_number_of_stepss': [episode_number_of_steps, episode_number_of_steps_1, episode_number_of_steps_2],
            'n_ep_running_averages': [n_ep_running_average, n_ep_running_average_1, n_ep_running_average_2],
            'best_agents': [best_agent, best_agent_1, best_agent_2],
            }
        
        plot_varing_hyperparameter(memory_size_plot, label_plot='Memory size', hyperparameter_name='memory_sizes')

class Plots_Actor_Critic:
    def __init__(self):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(f"{self.current_dir}/outputs", exist_ok=True)
        
        self.actor = torch.load(os.path.join(self.current_dir, 'outputs/neural-network-2-actor.pth'))
        self.critic = torch.load(os.path.join(self.current_dir, 'outputs/neural-network-2-critic.pth'))
        print('Network actor model: {}'.format(self.actor))
        print('Network critic model: {}'.format(self.critic))
        
        self.environment = gym.make('LunarLanderContinuous-v3')

    
    # Main task function
    def task_f_DDPG(self):
        """
        Generate and save plots for the value function and policy for a continuous Lunar Lander task.

        """

        # Prepare grid of states
        n_steps = 100
        omega = torch.linspace(start=-torch.pi, end=torch.pi, steps=n_steps, dtype=torch.float64)
        y = torch.linspace(start=0, end=1.5, steps=n_steps, dtype=torch.float64)
        omega_grid, y_grid = torch.meshgrid(omega, y, indexing="ij")

        # Create state tensors with fixed values for unused dimensions
        n_states = len(y_grid.reshape(-1))
        state_dim = len(self.environment.observation_space.low)
        states = torch.zeros((n_states, state_dim), dtype=torch.float64)
        states[:, 1] = y_grid.reshape(-1)  # Height
        states[:, 4] = omega_grid.reshape(-1)  # Angle

        # Compute the policy and value functions
        with torch.no_grad():
            z_values_policy = self.policy_function(states)
            z_values_value = self.value_function(states, z_values_policy)
            z_values_policy = z_values_policy[:,1].reshape(y_grid.shape) # Get the second action value (engine direction)
            z_values_value = z_values_value.reshape(y_grid.shape)
        
        # Generate the value function plot
        self.plot_solution_f_DDPG(
            omega_grid=omega_grid,
            y_grid=y_grid,
            z_grid=z_values_value,
            z_label=r"$Q_{\theta}(s, \pi_{\theta}(s))$",
            filename="value"
        )

        # Generate the policy plot
        self.plot_solution_f_DDPG(
            omega_grid=omega_grid,
            y_grid=y_grid,
            z_grid=z_values_policy,
            z_label=r"$\pi_{\theta}(s)_2$",
            filename="policy"
        )
    
    # Define the agent functions for value and policy
    def value_function(self, states, actions):
        states = states.clone().detach().float()
        actions = actions.clone().detach().float()
        q_values = self.critic(states, actions)  # Get Q-values from the network
        return q_values  # Return Q-values

    def policy_function(self, states):
        states = states.clone().detach().float()
        actions = self.actor(states)  # Get actions from the network
        return actions  # Return actions
    
    # Function to analyze the agent and plot results
    def plot_solution_f_DDPG(self, omega_grid, y_grid, z_grid, z_label, filename):
        """
        Generate and save a 3D plot for the given results.

        Parameters:
            omega_grid (torch.Tensor): Grid of omega values.
            y_grid (torch.Tensor): Grid of y values.
            z_grid (torch.Tensor): Grid of z values.
            z_label (str): Label for the z-axis.
            filename (str): Name of the file to save the plot.
        """


        # Plot results
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        plot = ax.plot_surface(omega_grid, y_grid, z_grid, cmap="coolwarm")
        ax.set_xlabel(r"$\omega$ (angle)")
        ax.set_ylabel(r"$y$ (height)")
        ax.set_zlabel(z_label)
        fig.colorbar(plot, ax=ax, shrink=0.5, aspect=10, label=z_label)
        plt.title(f"{z_label} vs. $\omega$ and $y$")
        plt.savefig(f"{self.current_dir}/outputs/{filename}-2.png")
        plt.show()


class Test_DDPG():
    def __init__(self, N_episodes=50):
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        self.N_episodes = N_episodes

        self.environment = gym.make('LunarLanderContinuous-v3')
        
        self.actor = torch.load(os.path.join(self.current_path, f'outputs/neural-network-2-actor.pth'))
        self.random_agent = RandomAgent(len(self.environment.action_space.high))

    def test(self, random):
        episode_reward_list = []
        episode_number_of_steps = []

        EPISODES = trange(self.N_episodes, desc='Episode: ', leave=True)

        for i in EPISODES:
            # Reset the environment
            done, truncated = False, False
            state = self.environment.reset()[0]
            total_episode_reward = 0.
            t = 0
            while not (done or truncated):
                # Take action
                if random:
                    action = self.random_agent.forward(state)
                else:
                    action = self.actor.forward(state).detach().numpy()

                # Get next state and reward
                next_state, reward, done, truncated, _ = self.environment.step(action)

                # Update episode reward
                total_episode_reward += reward

                # Update state for next iteration
                state = next_state
                t+= 1

            # Append episode reward
            episode_reward_list.append(total_episode_reward)
            episode_number_of_steps.append(t)



            # Updates the tqdm update bar with fresh information
            # (episode number, total reward of the last episode, total number of Steps
            # of the last episode, average reward, average number of steps)
            EPISODES.set_description(
                "Episode {} - Reward/Steps: {:.1f}/{}".format(
                i, total_episode_reward, t))
            
        # Close environment
        self.environment.close()

        return episode_reward_list, episode_number_of_steps
    
    def plot(self, random_episode_reward_list, random_episode_number_of_steps, actor_episode_reward_list, actor_episode_number_of_steps):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
        ax[0].plot([i for i in range(1, self.N_episodes+1)], random_episode_reward_list, label='Random agent')
        ax[0].plot([i for i in range(1, self.N_episodes+1)], actor_episode_reward_list, label='Actor agent')
        ax[0].set_xlabel('Episodes')
        ax[0].set_ylabel('Total reward')
        ax[0].set_title('Total Reward vs Episodes')
        ax[0].legend()
        ax[0].grid(alpha=0.3)

        ax[1].plot([i for i in range(1, self.N_episodes+1)], random_episode_number_of_steps, label='Random agent')
        ax[1].plot([i for i in range(1, self.N_episodes+1)], actor_episode_number_of_steps, label='Actor agent')
        ax[1].set_xlabel('Episodes')
        ax[1].set_ylabel('Total number of steps')
        ax[1].set_title('Total number of steps vs Episodes')
        ax[1].legend()
        ax[1].grid(alpha=0.3)
        plt.savefig(f"{self.current_path}/outputs/rewards-steps-2-test.png")
        plt.show()
    
    def test_random(self):
        episode_reward_list, episode_number_of_steps = self.test(random=True)
        print(f"Random agent after {self.N_episodes}: Average reward: {np.mean(episode_reward_list):.2f}, Average number of steps: {np.mean(episode_number_of_steps):.2f}")
        return episode_reward_list, episode_number_of_steps
    
    def test_actor(self):
        episode_reward_list, episode_number_of_steps = self.test(random=False)
        print(f"Actor agent after {self.N_episodes}: Average reward: {np.mean(episode_reward_list):.2f}, Average number of steps: {np.mean(episode_number_of_steps):.2f}")
        return episode_reward_list, episode_number_of_steps
    
    def test_both(self):
        reward_random, steps_random = self.test_random()
        reward_actor, steps_actor = self.test_actor()
        self.plot(reward_random, steps_random, reward_actor, steps_actor)
            


if __name__ == '__main__':
    task = "test"
    
    if task == "default":
        trainer = Trainer_DDGP()
        trainer.train()
        trainer.plot()
    elif task == "varying discount factors":
        experiments = Experiments()
        experiments.task_e2()
    elif task == "varying memory sizes":
        experiments = Experiments()
        experiments.task_e3()
    elif task == "plots policy and value functions":
        plots_f = Plots_Actor_Critic()
        plots_f.task_f_DDPG()
    elif task == "test":
        test = Test_DDPG()
        test.test_both()
