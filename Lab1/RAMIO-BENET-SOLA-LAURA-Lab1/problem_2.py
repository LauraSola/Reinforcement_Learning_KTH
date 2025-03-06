# Done by: 
# - Benet Ramió i Comas (20031026T177)
# - Laura Sola Garcia (20031119T225)

# Copyright [2024] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 1 for EL2805 - Reinforcement Learning.

# This file contains the implementation of all the classes, methods and functions needed to solve the Mountain Car environment implementing SARSA(lambda) with function aproximation.
# The notebook 'problem1_experiments.ipynb' contains the code to run the experiments and the analysis of the results.


# Load packages
import numpy as np
import gymnasium as gym
import random
import matplotlib.pyplot as plt
from tqdm import trange
import itertools
from typing import List
import pickle
from itertools import combinations, product


##############################################
########## SARSA(lambda) ALGORITHM ###########
############################################## 


class Fourier_basis:

    def __init__(self, n_states, max_non_zero, p, type_coeffs, defined_coeffs):
        self.n_states = n_states #input size
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

    def get_phi(self, state): 
        return np.cos(np.pi*np.dot(self.coeffs, state))
    
    def scale_learning_rate(self, lr):
        norm = np.linalg.norm(self.coeffs, axis=1)
        norm[norm == 0.] = 1.  # When the norm is zero do not scale
        return lr / norm
    
    def get_num_basis(self):
        """Return the number of basis functions."""
        return self.coeffs.shape[0]



class SarsaLambda:
    def __init__(self, state_space, action_space, min_max_norm=False, alpha=0.0001, lamb=0.9, gamma=0.99, epsilon=0.05, fourier_order=2, max_non_zero_fourier=2, momentum = 0.8, initialization = "zeros", type_coeffs="nothing", defined_coeffs=None, reduction_factor= 1, tau = 1.0, expl_strategy = "e-greedy"):
        # Hyperparameters
        self.alpha = alpha
        self.lamb = lamb
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Environment and state/action spaces
        self.env = gym.make('MountainCar-v0')
        self.state_space = state_space
        self.state_dim = self.state_space.shape[0]
        self.action_space = action_space
        self.action_dim = self.action_space.n

        # Fourier basis
        self.min_max_norm = min_max_norm
        self.p = fourier_order
        self.non_zero = max_non_zero_fourier
        self.basis = Fourier_basis(self.state_dim, self.non_zero, self.p, type_coeffs, defined_coeffs)
        self.scaled_lr = self.basis.scale_learning_rate(self.alpha)
        self.num_basis = self.basis.get_num_basis()

        # Weights and eligibility traces
        self.z = {a: np.zeros(self.num_basis) for a in range(self.action_dim)}

        if initialization == "zeros":
            self.w = {a: np.zeros(self.num_basis) for a in range(self.action_dim)}
        elif initialization == "random uniform":
            low, high = -1.0, 1.0  # Define the range for random initialization
            self.w = {a: np.random.uniform(low, high, self.num_basis) for a in range(self.action_dim)}
        elif initialization == "random gaussian":
            mean, std_dev = 0.0, 1.0
            self.w = {a: np.random.normal(mean, std_dev, self.num_basis) for a in range(self.action_dim)}
     
     
        # Momentum-based SGD variables
        self.velocity = {a: np.zeros(self.num_basis) for a in range(self.action_dim)}
        self.momentum = momentum

        #lr updates
        self.last_threshold = float('-inf')
        self.reduction_factor = reduction_factor

        #exploration
        self.expl_strategy = expl_strategy
        self.tau = tau

    def get_q_value(self, w, action, state):
        """Compute Q-value for a given state and action."""
        return np.dot(w[action], self.basis.get_phi(state))
    

    def get_v_value(self, state):
        """Compute Q-value for a given state and action."""
        return np.max([np.dot(self.w[action], self.basis.get_phi(state)) for action in range(self.action_dim)])
    
    def discretize_state_space(self, n_points=100):
        """
        Discretize the state space into a grid for visualization.

        Args:
            n_points (int): Number of discretization points along each dimension.

        Returns:
            tuple: position_grid, velocity_grid representing the state space grid.
        """
        # Define a grid over the state space
        position_space = np.linspace(self.env.observation_space.low[0], self.env.observation_space.high[0], n_points)
        velocity_space = np.linspace(self.env.observation_space.low[1], self.env.observation_space.high[1], n_points)
        position_grid, velocity_grid = np.meshgrid(position_space, velocity_space)
        
        return position_grid, velocity_grid
    
    
    
    def get_v_function(self, n_points = 100):
        """
        Compute the value function over a discretized state space.

        Args:
            n_points (int): Number of points to discretize along each dimension.

        Returns:
            tuple: position_grid, velocity_grid, v_values representing the value function.
        """
        # Discretize the state space
        position_grid, velocity_grid = self.discretize_state_space(n_points)

        # Compute V-values for each state in the grid
        v_values = np.zeros_like(position_grid)
        for i in range(position_grid.shape[0]):
            for j in range(position_grid.shape[1]):
                state = np.array([position_grid[i, j], velocity_grid[i, j]])
                scaled_state = self.scale_state_variables(state)
                v_values[i, j] = self.get_v_value(scaled_state)

        return position_grid, velocity_grid, v_values




    def get_policy(self, n_points = 100):
        """Compute the optimal policy over the state space."""
    # Discretize the state space
        position_grid, velocity_grid = self.discretize_state_space(n_points)

        # Compute optimal action for each state in the grid
        optimal_policy = np.zeros_like(position_grid, dtype=int)
        for i in range(position_grid.shape[0]):
            for j in range(position_grid.shape[1]):
                state = np.array([position_grid[i, j], velocity_grid[i, j]])
                scaled_state = self.scale_state_variables(state)
                Q_values = [self.get_q_value(self.w, a, scaled_state) for a in range(self.action_dim)]
                optimal_policy[i, j] = np.argmax(Q_values)

        return position_grid, velocity_grid, optimal_policy
            

    def softmax(self, q_values):
        """Softmax function for action selection."""
        exp_q = np.exp(np.array(q_values) / self.tau) #tau = temperature parameter
        return exp_q / np.sum(exp_q)


    def exploration_strategy(self, state, test=False):
        """Choose action using epsilon-greedy policy."""

        Q_s = [self.get_q_value(self.w, a, state) for a in range(self.action_dim)]

        if self.expl_strategy == "boltzmann":
            probabilities = self.softmax(Q_s)
            return np.random.choice(len(Q_s), p=probabilities)
    
        else:
            if not test and random.random() < self.epsilon: #exploration
                if self.expl_strategy == "worst":
                    if random.random() < 0.5:
                        return np.argmin(Q_s) 
                    else:
                        return np.random.randint(0, self.action_dim)                 
                else:
                    return np.random.randint(0, self.action_dim)
                
            else: # return best action wrt Q
                return np.argmax(Q_s)

    def update_eligibility_trace(self, action, state):
        """Update eligibility traces."""

        phi_state = self.basis.get_phi(state)
        for a in range(self.action_dim):
            if a == action:
                self.z[a] = self.gamma * self.lamb * self.z[a] + phi_state
            else:
                self.z[a] = self.gamma * self.lamb * self.z[a]

    def update_weights_SGD(self, reward, q, next_q):
        """Update weights using SGD with momentum."""

        delta = reward + self.gamma * next_q - q  # Temporal difference error
        for a in range(self.action_dim):
            self.velocity[a] = self.momentum * self.velocity[a] + self.scaled_lr * delta * self.z[a]
            self.w[a] += self.velocity[a]

    def update_epsilon(self, decay=0.9999): 
        """Decay epsilon."""
        self.epsilon *= decay

    def update_learning_rate(self, avg_reward):
        """
        Reduce learning rate based on average reward thresholds.
        Args:
            avg_reward (float): The average reward over recent episodes.
            reduction_factor (float): Factor by which to reduce the learning rate.
        """
        thresholds = [-140, -135, -130, -125, -120, -115, -110, -105, -100]  # Define thresholds for learning rate reduction

        for threshold in thresholds:
            if avg_reward > threshold > self.last_threshold and self.alpha > 1e-6:  # Avoid reducing alpha below a very small value
                self.alpha *= self.reduction_factor
                self.scaled_lr = self.basis.scale_learning_rate(self.alpha)
                self.last_threshold = threshold
                break  # Apply one reduction at a time to avoid excessive lowering


    def reset_z_v(self):
        """Reset eligibility traces and velocity vectors"""
        for a in range(self.action_dim):
            self.z[a].fill(0.0)
            self.velocity[a].fill(0.0)

    def scale_state_variables(self, s ):
        """Rescale state variables to [0, 1]^n."""
        low=self.env.observation_space.low
        high=self.env.observation_space.high

        return (s - low) / (high - low)

    def sarsa_learning_step(self, state, action, reward, next_state, next_action, done):
        """Perform one step of SARSA(λ)."""
        
        q = self.get_q_value(self.w, action, state)

        if not done:
            next_q = self.get_q_value(self.w, next_action, next_state)
        else:
            next_q = 0.0

        self.update_eligibility_trace(action, state)
        self.update_weights_SGD(reward, q, next_q)
        self.update_epsilon()

        if done:
            self.reset_z_v()

class Trainer:
    def __init__(self, environment: gym.Env,
                 agent: SarsaLambda,
                 epsilon: float = 0.3,
                 number_episodes: int = 200,
                 episode_reward_trigger: float = -135,
                 early_stopping=False
                 ):
        
        ## SET PARAMETERS
        self.early_stopping = early_stopping
        self.episode_reward_trigger = episode_reward_trigger
        self.number_episodes = number_episodes
        self.env = environment
        self.agent = agent
        self.epsilon = epsilon
    

        ## SET VARIABLES
        self.episode_reward_list = []

    # Functions used during training
    def running_average(self, x, N):
        ''' Function used to compute the running mean
            of the last N elements of a vector x
        '''
        if len(x) >= N:
            y = np.copy(x)
            y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
        else:
            y = np.zeros_like(x)
        return y

    def decay_lr(self):
    # Compute average reward over last 30 episodes
        if len(self.episode_reward_list) >= 30:
            avg_reward = np.mean(self.episode_reward_list[-30:])
        else:
            avg_reward = np.mean(self.episode_reward_list)

        # Update learning rate and epsilon
        self.agent.update_learning_rate(avg_reward)

    def train(self):
        ### RESET ENVIRONMENT ###
        self.env.reset() ########

        time = 0

        for e in trange(self.number_episodes):
            done = False
            terminated = False
            state = self.agent.scale_state_variables(self.env.reset()[0])
            total_episode_reward = 0.

            while not (done or terminated):
                ### TAKE ACTION ###
                action = self.agent.exploration_strategy(state)

                ### USE ACTION ###
                next_state, reward, done, terminated, _ = self.env.step(action)
                next_state = self.agent.scale_state_variables(next_state)

                ### COMPUTE NEXT ACTION
                next_action = self.agent.exploration_strategy(next_state)

                ### USE SARSA
                self.agent.sarsa_learning_step(state, action, reward, next_state, next_action, done)

                ## UPDATE DATA
                total_episode_reward += reward
                state = next_state


            self.episode_reward_list.append(total_episode_reward)
            
            self.decay_lr()
        
        # Create dictionary object
        w_dict = self.agent.w
        w =  np.stack([w_dict[a] for a in sorted(w_dict.keys())], axis=0)
        data = {'W': w, 'N': self.agent.basis.coeffs}

        # Save dictionary to a file
        with open('weights.pkl', 'wb') as file:
            pickle.dump(data, file)


   
    def test(self, N=50, verbose=False):

        N_EPISODES = N  # Number of episodes to run for trainings
        CONFIDENCE_PASS = -135
        print('Checking solution...')
        EPISODES = trange(N_EPISODES, desc='Episode: ', leave=True)
        episode_reward_list = []
        for i in EPISODES:
            # EPISODES.set_description("Episode {}".format(i))
            # Reset enviroment data
            done = False
            truncated = False
            state = self.agent.scale_state_variables(self.env.reset()[0])
            total_episode_reward = 0.

            action = self.agent.exploration_strategy(state, test=True)

            while not (done or truncated):
                # Get next state and reward.  The done variable
                # will be True if you reached the goal position,
                # False otherwise
                next_state, reward, done, truncated, *_ = self.env.step(action)
                next_state = self.agent.scale_state_variables(next_state)
                next_action = self.agent.exploration_strategy(state, test=True)

                # Update episode reward
                total_episode_reward += reward

                # Update state for next iteration
                state = next_state
                action = next_action

            # Append episode reward
            episode_reward_list.append(total_episode_reward)

            # Close environment
            self.env.close()

        avg_reward = np.mean(episode_reward_list)
        confidence = np.std(episode_reward_list) * 1.96 / np.sqrt(N_EPISODES)

        if verbose:
            print('Policy achieves an average total reward of {:.1f} +/- {:.1f} with confidence 95%.'.format(
                avg_reward,
                confidence))

        if avg_reward - confidence >= CONFIDENCE_PASS:
            if verbose:
                print('Your policy passed the test!')
            return True, avg_reward, confidence
        else:
            if verbose:
                print(
                    'Your policy did not pass the test! The average reward of your policy needs to be greater than {} with 95% confidence'.format(
                        CONFIDENCE_PASS))
            return False, avg_reward, confidence


##############################################
######### HYPERPARAMETER EXPERIMENTS #########
############################################## 

def run_hyperparameter_search(hyperparameters, num_episodes=200, test_episodes=50, early_stopping=True):
    """
    Run a hyperparameter search for a SARSA(λ) agent on the MountainCar-v0 environment.

    Args:
        hyperparameters (dict): Dictionary of hyperparameters. Keys with lists will be treated as variable.
        num_episodes (int): Number of episodes to train.
        test_episodes (int): Number of episodes to test.
        early_stopping (bool): Whether to use early stopping during training.
    
    Returns:
        tuple: (all_rewards, all_labels)
            all_rewards (list of lists): Episode rewards for each hyperparameter combination.
            all_labels (list of str): Labels describing each hyperparameter combination.
    """
    # Separate fixed and variable hyperparameters
    fixed_hyperparams = {k: v for k, v in hyperparameters.items() if not isinstance(v, list)}
    variable_hyperparams = {k: v for k, v in hyperparameters.items() if isinstance(v, list)}

    # Generate all combinations of variable hyperparameters
    keys, values = zip(*variable_hyperparams.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Initialize lists to store results
    all_rewards = []
    all_labels = []

    test_avg_rewards = []
    test_confidences = []

    # Loop over all hyperparameter combinations
    for i, params in enumerate(param_combinations):
        # Combine fixed and variable hyperparameters
        current_hyperparams = {**fixed_hyperparams, **params}

        # Extract hyperparameters for this iteration
        alpha = current_hyperparams["alpha"]
        lamb = current_hyperparams["λ"]
        gamma = current_hyperparams["gamma"]
        epsilon = current_hyperparams["epsilon"]
        fourier_order = current_hyperparams["fourier_order"]
        max_non_zero_fourier = current_hyperparams["max_non_zero_fourier"]
        reduction_factor = current_hyperparams["reduction factor"]
        initialization = current_hyperparams["initialization"]
        expl_strategy = current_hyperparams["expl_strategy"]
        type_coeffs = current_hyperparams["type_coeffs"]
        defined_coeffs = current_hyperparams["defined_coeffs"]

        # Initialize environment
        env = gym.make("MountainCar-v0")

        # Initialize SARSA(λ) agent
        agent = SarsaLambda(
            state_space=env.observation_space,
            action_space=env.action_space,
            alpha=alpha,
            lamb=lamb,
            gamma=gamma,
            epsilon=epsilon,
            fourier_order=fourier_order,
            max_non_zero_fourier=max_non_zero_fourier,
            reduction_factor=reduction_factor,
            initialization=initialization,
            expl_strategy=expl_strategy,
            type_coeffs=type_coeffs,
            defined_coeffs=defined_coeffs
        )

        # Initialize trainer
        trainer = Trainer(
            environment=env,
            agent=agent,
            epsilon=epsilon,
            number_episodes=num_episodes,
            episode_reward_trigger=-150,
            early_stopping=early_stopping,
        )

        # Train and test the agent
        print(f"Starting training for combination {i + 1}/{len(param_combinations)}: {current_hyperparams}")
        trainer.train()
        print(f"Training for combination {i + 1} completed!")
        _, avg_reward, confidence = trainer.test(N=test_episodes, verbose=True)
        test_avg_rewards.append(avg_reward)
        test_confidences.append(confidence)
        print(f"Testing for combination {i + 1} completed!")

        # Store results
        all_rewards.append(trainer.episode_reward_list)
        
        # Create label only for variable hyperparameters
        variable_label = ", ".join(f"{k}={params[k]}" for k in variable_hyperparams)
        all_labels.append(variable_label)

        # Close environment
        env.close()


    return all_rewards, all_labels, test_avg_rewards, test_confidences

##############################################
####### PLOT HYPERPARAMETER EXPERIMENTS ######
##############################################           


def plot_smoothed_rewards(rewards, window_size=20):
    """
    Plots smoothed rewards using a moving average.

    Parameters:
        rewards (list or np.array): The list of rewards for each episode.
        window_size (int): The size of the sliding window for averaging. Default is 20.
    """
    # Compute the moving average
    smoothed_rewards = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
    
    # Plot the results
    plt.plot(smoothed_rewards)
    plt.title(f"Smoothed Episode Rewards (Window Size: {window_size})")
    plt.xlabel("Episodes")
    plt.ylabel("Smoothed Total Reward")
    plt.show()

def compute_smoothed_rewards(rewards, window_size=20):
    """Compute smoothed rewards using a moving average."""

    rewards = [-200 for i in range(window_size)] + rewards

    smoothed = np.convolve(rewards, np.ones(window_size) / window_size, mode="valid")
    std_devs = [
        np.std(rewards[max(0, i - window_size + 1):i + 1])
        for i in range(len(smoothed))
    ]
    return smoothed, np.array(std_devs)

def plot_smoothed_rewards_with_confidence(rewards, cf,window_size=20, label=None):
    """Plot smoothed rewards with confidence intervals."""

    
    smoothed_rewards, std_devs = compute_smoothed_rewards(rewards, window_size)
    x_vals = np.arange(len(smoothed_rewards))
    plt.plot(x_vals, smoothed_rewards, label=label)
    if cf:
        plt.fill_between(
            x_vals,
            smoothed_rewards - std_devs,
            smoothed_rewards + std_devs,
            alpha=0.3
        )
        plt.title("Smoothed Rewards with Confidence Interval")
    else:
        plt.title("Smoothed Rewards")
    plt.xlabel("Episodes")
    plt.ylabel("Smoothed Reward")
    
    if label:
        plt.legend()

def plot_all_smoothed_rewards(all_rewards, window_size=20, labels=None, cf = True):
    """Plot smoothed rewards with confidence intervals """
    plt.figure(figsize=(12, 6)) 
    for i, rewards in enumerate(all_rewards):
        label = labels[i] if labels else None
        plot_smoothed_rewards_with_confidence(rewards,cf, window_size, label)
    plt.show()

def plot_test_results_lr(test_avg_rewards, test_confidences, values):
    fig = plt.figure(figsize=(10, 6))
    plt.errorbar(values, test_avg_rewards, yerr=test_confidences, fmt='o-', capsize=5, label='Mean Reward ± Std Dev')
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Average Reward')
    plt.xscale('log')
    plt.title(f'Effect of the Learning Rate on Performance')
    plt.legend()
    plt.grid()
    plt.show()

def plot_test_results_lamb(test_avg_rewards, test_confidences, values):
    plt.figure(figsize=(10, 6))
    plt.errorbar(values, test_avg_rewards, yerr=test_confidences, fmt='o-', capsize=5, label='Mean Reward ± Std Dev')
    plt.xlabel('Lambda')
    plt.ylabel('Average Reward')
    plt.title(f'Effect of Lambda (Elegibility Trace) on Performance')
    plt.ylim((-150, -110))
    plt.legend()
    plt.grid()
    plt.show()


##############################################
############## PLOT SOLUTION #################
##############################################

def plot_policy(position_grid, velocity_grid, optimal_policy):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    surface = ax.plot_surface(position_grid, velocity_grid, optimal_policy, cmap='coolwarm')
    ax.set_title("Optimal Policy")
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_zlabel("Action")
        
    plt.tight_layout()

    plt.show()


def plot_v_function(position_grid, velocity_grid, v_values):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    surface = ax.plot_surface(position_grid, velocity_grid, v_values, cmap='viridis')
    ax.set_title("Value Function")
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_zlabel("Value Function")

    plt.tight_layout()


    plt.show()

