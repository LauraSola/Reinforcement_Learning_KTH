import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import os
from DQN_agent import Agent

model = torch.load('/Users/laurasolagarcia/Documents/REINFORCEMENT LEARNING/lab2/problem 1/results_kaggle/models/L=10000_clipping_value=1_n_hidden_layers=1_dueling=False_CER=True_N_episodes=800_discount_factor=0.99_type_decay=exponential_N=128_learning_rate=0.0001_hidden_size=64.pt')
env = gym.make('LunarLander-v2')

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
    n_steps = 100
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

# Example usage
task_f()
