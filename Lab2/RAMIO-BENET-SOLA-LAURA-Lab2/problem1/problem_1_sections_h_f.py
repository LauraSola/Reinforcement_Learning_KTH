# Done by: 
# - Benet Ramió i Comas (20031026T177)
# - Laura Sola Garcia (20031119T225)


import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import trange
import os
from problem_1 import running_average
from DQN_agent import Agent, RandomAgent

model = torch.load('neural-network-1.pth')
env = gym.make('LunarLander-v2')

import matplotlib.cm as cm

# Plotting
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))


# Define the agent functions for value and policy
def value_function(states):
    q_values = model(torch.tensor(states))  # Get Q-values from the network
    return q_values.max(dim=1).values  # Return max Q-value (state value)

def policy_function(states):
    q_values = model(torch.tensor(states))  # Get Q-values from the network
    return q_values.argmax(dim=1)  # Return action with max Q-value


# Function to analyze the agent and plot results
def plot_solution(agent_function, environment, z_label):
    """
    Generate and save a 3D plot for the given agent function.

    Parameters:
        agent_function (callable): Function to compute the value or policy.
        environment (gym.Env): Environment to determine state dimensions.
        z_label (str): Label for the Z-axis of the plot.
        filepath (str or Path): Path to save the plot.
    """
    # Prepare grid of states
    n_steps = 1000
    omega = torch.linspace(start=-torch.pi, end=torch.pi, steps=n_steps, dtype=torch.float64)
    y = torch.linspace(start=0, end=1.5, steps=n_steps, dtype=torch.float64)
    omega_grid, y_grid = torch.meshgrid(omega, y, indexing="ij")

    # Create state tensors with fixed values for unused dimensions
    n_states = len(y_grid.reshape(-1))
    state_dim = len(environment.observation_space.low)
    states = torch.zeros((n_states, state_dim), dtype=torch.float64)
    states[:, 1] = y_grid.reshape(-1)  # Height
    states[:, 4] = omega_grid.reshape(-1)  # Angle

    # Compute agent output for the grid
    with torch.no_grad():
        z_values = agent_function(states)  # Evaluate the agent function
        z_grid = z_values.reshape(y_grid.shape)

    # Plot results
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    plot = ax.plot_surface(omega_grid, y_grid, z_grid, cmap="coolwarm")
    ax.set_xlabel(r"$\omega$ (angle)")
    ax.set_ylabel(r"$y$ (height)")
    ax.set_zlabel(z_label)
    fig.colorbar(plot, ax=ax, shrink=0.5, aspect=10, label=z_label)
    plt.title(f"{z_label} vs. $\omega$ and $y$")
    plt.savefig(f"{current_dir}/results/{agent_function}.png")
    plt.show()

# Main task function
def task_f():
    """
    Generate and save plots for the value function and policy for a Lunar Lander agent.

    Parameters:
        results_dir (Path): Directory to save the plots.
        agent_path (str): Path to the trained agent file.
    """
    os.makedirs(f"{current_dir}/results", exist_ok=True)

    # Generate the value function plot
    plot_solution(
        agent_function=value_function,
        environment=env,
        z_label=r"$V_{\theta}(s)$",
        
    )

    # Generate the policy plot
    plot_solution(
        agent_function=policy_function,
        environment=env,
        z_label=r"$\pi_{\theta}(s)$",
    )

#task_f()


###############################

def comparison_agents(environment, agent, random):
    # Parameters
    N_episodes = 50               # Number of episodes to run for training
    n_ep_running_average = 50      # Running average of 50 episodes

    # Reward
    episode_reward_list = []  # Used to save episodes reward
    episode_number_of_steps = []


    # Testing process
    EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

    for i in EPISODES:
        # Reset enviroment data
        done, truncated = False, False
        state = environment.reset()[0]
        total_episode_reward = 0.
        t = 0
        while not (done or truncated):
            # Take action
            if random:
                action = agent.forward(state)
            else:
                q_values = model(torch.tensor(state))
                _, action = torch.max(q_values, dim=0)
                action = action.item()

            # Get next state and reward
            next_state, reward, done, truncated, _ = environment.step(action)

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
            "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
            i, total_episode_reward, t,
            running_average(episode_reward_list, n_ep_running_average)[-1],
            running_average(episode_number_of_steps, n_ep_running_average)[-1]))
        
    # Close environment
    environment.close()

    return episode_reward_list, episode_number_of_steps

def plot_comparison(episode_reward_list_random, episode_number_of_steps_random, episode_reward_list, episode_number_of_steps):
    """
    Plots a comparison of rewards and number of steps for two agents over 50 episodes.

    Args:
        episode_reward_list_random (list): Rewards for the random agent across episodes.
        episode_number_of_steps_random (list): Steps taken by the random agent across episodes.
        episode_reward_list (list): Rewards for the other agent across episodes.
        episode_number_of_steps (list): Steps taken by the other agent across episodes.
    """

    episodes = range(1, 51)  # 50 episodes

    # Plot rewards comparison
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)  # First subplot: rewards
    plt.plot(episodes, episode_reward_list_random, label='Random Agent', color='blue', linestyle='--')
    plt.plot(episodes, episode_reward_list, label='Trained Agent', color='green', linestyle='-')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Comparison of Rewards over 50 Episodes')
    plt.legend()
    plt.grid()

    # Plot number of steps comparison
    plt.subplot(1, 2, 2)  # Second subplot: number of steps
    plt.plot(episodes, episode_number_of_steps_random, label='Random Agent', color='blue', linestyle='--')
    plt.plot(episodes, episode_number_of_steps, label='Trained Agent', color='green', linestyle='-')
    plt.xlabel('Episodes')
    plt.ylabel('Number of Steps')
    plt.title('Comparison of Steps over 50 Episodes')
    plt.legend()
    plt.grid()

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.savefig(f"{current_dir}/figs/comparison_rand.png")
    plt.show()

def task_h():
    # Create the random agent
    random_agent = RandomAgent(env.action_space.n)

    # Call function
    episode_reward_list_random, episode_number_of_steps_random = comparison_agents(env, random_agent, True)
    episode_reward_list, episode_number_of_steps = comparison_agents(env, model, False)

    # print the average reward of the random agent and the trained agent after 50 episodes
    print("Average reward of the random agent after 50 episodes: ", np.mean(episode_reward_list_random))
    print("Average reward of the trained agent after 50 episodes: ", np.mean(episode_reward_list))

    print("Average num steps of the random agent after 50 episodes: ", np.mean(episode_number_of_steps_random))
    print("Average num steps of the trained agent after 50 episodes: ", np.mean(episode_number_of_steps))

    plot_comparison(episode_reward_list_random, episode_number_of_steps_random, episode_reward_list, episode_number_of_steps)


if __name__ == '__main__':
    task = "plots policy and value functions"
    
    if task == "plots policy and value functions":
        task_f()
    elif task == "compare random agent":
        task_h()
        