# Done by: 
# - Benet Ramió i Comas (20031026T177)
# - Laura Sola Garcia (20031119T225)

# Copyright [2024] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 1 for EL2805 - Reinforcement Learning.

# This file contains the implementation of all the classes, methods and functions needed to solve the Minotaur Maze problem.
# The notebook 'problem1_experiments.ipynb' contains the code to run the experiments and the analysis of the results.

import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display
import random
# RANDOM SEED
np.random.seed(17844)


# Implemented methods
methods = ['DynProg', 'ValIter']

# Some colours
LIGHT_RED    = '#FFC4CC'
LIGHT_GREEN  = '#95FD99'
BLACK        = '#000000'
WHITE        = '#FFFFFF'
LIGHT_PURPLE = '#E8D0FF'
LIGHT_ORANGE = '#FFDBBB'


#############################################
#### Maze environment class with no keys ####
#############################################

class Maze_NoKeys:

    # Actions
    STAY       = 0
    MOVE_LEFT  = 1
    MOVE_RIGHT = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values 
    STEP_REWARD = -100          #TODO
    GOAL_REWARD = 100          #TODO
    IMPOSSIBLE_REWARD = -100   #TODO
    MINOTAUR_REWARD = -100      #TODO

    def __init__(self, maze, stay=False):
        """ Constructor of the environment Maze.
        """
        self.maze                     = maze
        self.stay                     = stay
        self.actions                  = self.__actions()
        self.states, self.map         = self.__states()
        self.n_actions                = len(self.actions)
        self.n_states                 = len(self.states)
        self.transition_probabilities = self.__transitions()
        self.rewards                  = self.__rewards()

    def __actions(self):
        actions = dict()
        actions[self.STAY]       = (0, 0)
        actions[self.MOVE_LEFT]  = (0,-1)
        actions[self.MOVE_RIGHT] = (0, 1)
        actions[self.MOVE_UP]    = (-1,0)
        actions[self.MOVE_DOWN]  = (1,0)
        return actions

    def __states(self):
        
        states = dict()
        map = dict()
        s = 0
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                for k in range(self.maze.shape[0]):
                    for l in range(self.maze.shape[1]):
                        if self.maze[i,j] != 1:
                            states[s] = ((i,j), (k,l))
                            map[((i,j), (k,l))] = s
                            s += 1
        
        states[s] = 'Eaten'
        map['Eaten'] = s
        s += 1
        
        states[s] = 'Win'
        map['Win'] = s
        
        return states, map

    def __move(self, state, action):               
        """ Makes a step in the maze, given a current position and an action. 
            If the action STAY or an inadmissible action is used, the player stays in place.
        
            :return list of tuples next_state: Possible states ((x,y), (x',y')) on the maze that the system can transition to.
        """

        if self.states[state] == 'Eaten' or self.states[state] == 'Win': # In these states, the game is over
            return [self.states[state]]
        
        else: # Compute the future possible positions given current (state, action)
            row_player = self.states[state][0][0] + self.actions[action][0] # Row of the player's next position 
            col_player = self.states[state][0][1] + self.actions[action][1] # Column of the player's next position 
            
            # Is the player getting out of the limits of the maze or hitting a wall?
            impossible_action_player = (row_player == -1 or row_player == self.maze.shape[0] or \
                                         col_player == -1 or col_player == self.maze.shape[1] or \
                                         self.maze[row_player, col_player] == 1 )
            
            actions_minotaur = [[0, -1], [0, 1], [-1, 0], [1, 0]] # Possible moves for the Minotaur
            if self.stay:
                actions_minotaur.append([0, 0])

            rows_minotaur, cols_minotaur = [], []
            for i in range(len(actions_minotaur)):
                # Is the minotaur getting out of the limits of the maze?
                impossible_action_minotaur = (self.states[state][1][0] + actions_minotaur[i][0] == -1) or \
                                             (self.states[state][1][0] + actions_minotaur[i][0] == self.maze.shape[0]) or \
                                             (self.states[state][1][1] + actions_minotaur[i][1] == -1) or \
                                             (self.states[state][1][1] + actions_minotaur[i][1] == self.maze.shape[1])
            
                if not impossible_action_minotaur:
                    rows_minotaur.append(self.states[state][1][0] + actions_minotaur[i][0])
                    cols_minotaur.append(self.states[state][1][1] + actions_minotaur[i][1])  
          

            # Based on the impossiblity check return the next possible states.
            if impossible_action_player: # The action is not possible, so the player remains in place
                states = []
                for i in range(len(rows_minotaur)):
                    
                    if (self.states[state][0][0] == rows_minotaur[i] and self.states[state][0][1] == cols_minotaur[i]): # TODO: We met the minotaur
                        states.append('Eaten')
                    
                    elif (self.maze[self.states[state][0][0], self.states[state][0][1]] == 2 ): # TODO: We are at the exit state, without meeting the minotaur
                        states.append('Win')
                
                    else: # The player remains in place, the minotaur moves randomly
                        states.append(((self.states[state][0][0], self.states[state][0][1]), (rows_minotaur[i], cols_minotaur[i])))
                
                return states
          
            else: # The action is possible, the player and the minotaur both move
                states = []
                for i in range(len(rows_minotaur)):
                
                    if (row_player == rows_minotaur[i] and col_player == cols_minotaur[i]): # TODO: We met the minotaur
                        states.append('Eaten')
                    
                    elif self.maze[row_player, col_player] == 2: # TODO:We are at the exit state, without meeting the minotaur
                        states.append('Win')
                    
                    else: # The player moves, the minotaur moves randomly
                        states.append(((row_player, col_player), (rows_minotaur[i], cols_minotaur[i])))
              
                return states
        
        
        

    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states,self.n_states,self.n_actions)
        transition_probabilities = np.zeros(dimensions)

        # TODO: Compute the transition probabilities.
  
        for s in range(self.n_states):
                for a in range(self.n_actions):
                    next_possible_states = self.__move(s,a)
                    prob_next = 1.0/len(next_possible_states)

                    for t in next_possible_states:
                        transition_probabilities[self.map[t],s,a] = prob_next
    
        return transition_probabilities



    def __rewards(self):
        
        """ Computes the rewards for every state action pair """

        rewards = np.zeros((self.n_states, self.n_actions))
        
        for s in range(self.n_states):
            for a in range(self.n_actions):
                
                if self.states[s] == 'Eaten': # The player has been eaten
                    rewards[s, a] = self.MINOTAUR_REWARD
                
                elif self.states[s] == 'Win': # The player has won
                    rewards[s, a] = self.GOAL_REWARD
                
                else:                
                    next_states = self.__move(s,a)
                    next_s = next_states[0] # The reward does not depend on the next position of the minotaur, we just consider the first one
                    
                    if self.states[s][0] == next_s[0] and a != self.STAY: # The player hits a wall
                        rewards[s, a] = self.IMPOSSIBLE_REWARD
                    
                    else: # Regular move
                        rewards[s, a] = self.STEP_REWARD

        return rewards




    def simulate(self, start, policy, method):
        
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods)
            raise NameError(error)

        path = list()
        
        if method == 'DynProg':
            horizon = policy.shape[1] # Deduce the horizon from the policy shape
            t = 0 # Initialize current time
            s = self.map[start] # Initialize current state 
            path.append(start) # Add the starting position in the maze to the path
            
            while t < horizon - 1:
                a = int(policy[s, t]) # Move to next state given the policy and the current state       
                next_states = self.__move(s, a) # Move to next state given the policy and the current state
                map_next_states = [self.map[state] for state in next_states] # Map the next states
                next_s = np.random.choice(map_next_states, p=self.transition_probabilities[map_next_states, s, a])  # Choose the next state given the transition probabilities
                path.append(self.states[next_s]) # Add the next state to the path
                t +=1 # Update time and state for next iteration
                s = next_s
                
        if method == 'ValIter': 
            t = 1 # Initialize current state, next state and time
            s = self.map[start]
            path.append(start) # Add the starting position in the maze to the path
            next_states = self.__move(s, policy[s]) # Move to next state given the policy and the current state
            map_next_states = [self.map[state] for state in next_states] # Map the next states
            next_s = np.random.choice(map_next_states, p=self.transition_probabilities[map_next_states, s, policy[s]])  # Choose the next state given the transition probabilities
            path.append(self.states[next_s]) # Add the next state to the path
            
            # horizon geometric mean 30
            horizon = np.random.geometric(1/30) # Sample the horizon from a geometric distribution with mean 30
            
            # Loop while state is not the goal state
            while s != next_s and t <= horizon:
                s = next_s # Update state
                next_states = self.__move(s, policy[s]) # Move to next state given the policy and the current state
                map_next_states = [self.map[state] for state in next_states] # Map the next states
                next_s = np.random.choice(map_next_states, p=self.transition_probabilities[map_next_states, s, policy[s]])  # Choose the next state given the transition probabilities
                path.append(self.states[next_s]) # Add the next state to the path
                t += 1 # Update time for next iteration
        
        return [path, horizon] # Return the horizon as well, to plot the histograms for the VI



    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.rewards)



############################################
##### Maze environment class with keys #####
############################################

class Maze:

    # Actions
    STAY       = 0
    MOVE_LEFT  = 1
    MOVE_RIGHT = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values 
    STEP_REWARD = 0          #TODO
    GOAL_REWARD = 10          #TODO
    KEY_REWARD = 1           #TODO
    IMPOSSIBLE_REWARD = -100    #TODO
    MINOTAUR_REWARD = -100      #TODO

    def __init__(self, maze):
        """ Constructor of the environment Maze.
        """
        self.maze                     = maze
        self.actions                  = self.__actions()
        self.states, self.map         = self.__states()
        self.n_actions                = len(self.actions)
        self.n_states                 = len(self.states)
        self.transition_probabilities = self.__transitions()
        self.rewards                  = self.__rewards()

    def __actions(self):
        actions = dict()
        actions[self.STAY]       = (0, 0)
        actions[self.MOVE_LEFT]  = (0,-1)
        actions[self.MOVE_RIGHT] = (0, 1)
        actions[self.MOVE_UP]    = (-1,0)
        actions[self.MOVE_DOWN]  = (1,0)
        return actions

    def __states(self):
        
        states = dict()
        map = dict()
        s = 0
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                for k in range(self.maze.shape[0]):
                    for l in range(self.maze.shape[1]):
                        if self.maze[i,j] != 1:
                            states[s] = ((i,j), (k,l), False)
                            map[((i,j), (k,l), False)] = s
                            s += 1
                            states[s] = ((i,j), (k,l), True)
                            map[((i,j), (k,l), True)] = s
                            s += 1
        
        states[s] = 'Eaten'
        map['Eaten'] = s
        s += 1
        
        states[s] = 'Win'
        map['Win'] = s
        
        return states, map


    def __move(self, state, action):               
        """ Makes a step in the maze, given a current position and an action. 
            If the action STAY or an inadmissible action is used, the player stays in place.
        
            :return list of tuples next_state: Possible states ((x,y), (x',y')) on the maze that the system can transition to.
        """

        if self.states[state] == 'Eaten' or self.states[state] == 'Win': # In these states, the game is over
            return [self.states[state]]
        
        # if the player has the key and is in the 'False', he goes to true and the minotaur does not move
        elif self.maze[self.states[state][0][0], self.states[state][0][1]] == 3 and self.states[state][-1] == False:
            return [(self.states[state][0], self.states[state][1], True)]
        
        else: # Compute the future possible positions given current (state, action)
            row_player = self.states[state][0][0] + self.actions[action][0] # Row of the player's next position 
            col_player = self.states[state][0][1] + self.actions[action][1] # Column of the player's next position 
            
            # Is the player getting out of the limits of the maze or hitting a wall?
            impossible_action_player = (row_player == -1 or row_player == self.maze.shape[0] or \
                                         col_player == -1 or col_player == self.maze.shape[1] or \
                                         self.maze[row_player, col_player] == 1 ) #TODO
            
            actions_minotaur = [[0, -1], [0, 1], [-1, 0], [1, 0]] # Possible moves for the Minotaur
            rows_minotaur, cols_minotaur = [], []
            for i in range(len(actions_minotaur)):
                # Is the minotaur getting out of the limits of the maze?
                impossible_action_minotaur = (self.states[state][1][0] + actions_minotaur[i][0] == -1) or \
                                             (self.states[state][1][0] + actions_minotaur[i][0] == self.maze.shape[0]) or \
                                             (self.states[state][1][1] + actions_minotaur[i][1] == -1) or \
                                             (self.states[state][1][1] + actions_minotaur[i][1] == self.maze.shape[1])
            
                if not impossible_action_minotaur:
                    rows_minotaur.append(self.states[state][1][0] + actions_minotaur[i][0])
                    cols_minotaur.append(self.states[state][1][1] + actions_minotaur[i][1])  
          

            # Based on the impossiblity check return the next possible states.
            if impossible_action_player: # The action is not possible, so the player remains in place
                states = []
                for i in range(len(rows_minotaur)):
                    
                    if (self.states[state][0][0] == rows_minotaur[i] and self.states[state][0][1] == cols_minotaur[i]): # TODO: We met the minotaur
                        states.append('Eaten')
                    
                    elif (self.maze[self.states[state][0][0], self.states[state][0][1]] == 2 and self.states[state][-1]): # TODO: We are at the exit state, without meeting the minotaur
                        states.append('Win')
                
                    else: # The player remains in place, the minotaur moves randomly
                        states.append(((self.states[state][0][0], self.states[state][0][1]), (rows_minotaur[i], cols_minotaur[i]), self.states[state][-1]))
                
                return states
          
            else: # The action is possible, the player and the minotaur both move
                states = []
                for i in range(len(rows_minotaur)):
                
                    if (row_player == rows_minotaur[i] and col_player == cols_minotaur[i]): # TODO: We met the minotaur
                        states.append('Eaten')
                    
                    elif (self.maze[row_player, col_player] == 2 and self.states[state][-1]): # TODO:We are at the exit state, without meeting the minotaur
                        states.append('Win')
                    
                    elif (self.maze[row_player, col_player] == 3): # The player gets the key
                        # states.append(((row_player, col_player), (rows_minotaur[i], cols_minotaur[i]), True))
                        states.append(((row_player, col_player), (rows_minotaur[i], cols_minotaur[i]), self.states[state][-1]))
                    
                    else: # The player moves, the minotaur moves randomly
                        states.append(((row_player, col_player), (rows_minotaur[i], cols_minotaur[i]), self.states[state][-1]))                        
                
                return states
        
        
        

    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states,self.n_states,self.n_actions)
        transition_probabilities = np.zeros(dimensions)
  
        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_possible_states = self.__move(s,a)
                uniform_prob = 1/len(next_possible_states)
                
                if 'Eaten' in next_possible_states or 'Win' in next_possible_states:
                    min = next_possible_states.index('Eaten') if 'Eaten' in next_possible_states else next_possible_states.index('Win')
                    
                else:
                    dist = [np.linalg.norm(np.array(next_state[1])-np.array(next_state[0])) for next_state in next_possible_states]
                    min = np.argmin(dist)

                # 35% chance of the minotaur moving in the same direction as the player, 65% chance of moving in a random direction
                if next_possible_states[min] == 'Win': # If the player wins, uniform probability
                    transition_probabilities[self.map[next_possible_states[min]],s,a] = uniform_prob
                else:
                    for t in next_possible_states:
                        if t == next_possible_states[min]:
                            if t == 'Win':
                                print(t, next_possible_states[min])
                            transition_probabilities[self.map[t],s,a] = 0.35 + 0.65*uniform_prob
                        else:
                            transition_probabilities[self.map[t],s,a] = 0.65*uniform_prob
                   
        return transition_probabilities

    def __rewards(self):
        
        """ Computes the rewards for every state action pair """

        rewards = np.zeros((self.n_states, self.n_actions))
        
        for s in range(self.n_states):
            for a in range(self.n_actions):
                    
                if self.states[s] == 'Eaten': # The player has been eaten
                    rewards[s, a] = self.MINOTAUR_REWARD
                
                elif self.states[s] == 'Win': # The player has won
                    rewards[s, a] = self.GOAL_REWARD

                
                else:                
                    next_states = self.__move(s,a)
                    next_s = next_states[0] # The reward does not depend on the next position of the minotaur, we just consider the first one
                    
                    if self.states[s][-1] == False and next_s[-1] == True: # In the previous state, the player did not have the key (there is only one possible state if the player has the key)
                        rewards[s, a] = self.KEY_REWARD

                    elif self.states[s][0] == next_s[0] and a != self.STAY: # The player hits a wall
                        rewards[s, a] = self.IMPOSSIBLE_REWARD
                    
                    else: # Regular move
                        rewards[s, a] = self.STEP_REWARD

        return rewards



    def simulate(self, start, policy, method, mean=30):
        
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods)
            raise NameError(error)

        path = list()
        
        if method == 'DynProg':
            horizon = policy.shape[1] # Deduce the horizon from the policy shape
            t = 0 # Initialize current time
            s = self.map[start] # Initialize current state 
            path.append(start) # Add the starting position in the maze to the path
            
            while t < horizon - 1:
                a = policy[s, t] # Move to next state given the policy and the current state
                next_states = self.__move(s, a)
                map_next_states = [self.map[state] for state in next_states]
                # next_s = np.random.choice(map_next_states, p=self.transition_probabilities[map_next_states, s, a])[0]  # Choose the next state given the transition probabilities
                next_s = np.random.choice(map_next_states) # Choose the next state randomly

                path.append(self.states[next_s]) # Add the next state to the path
                t +=1 # Update time and state for next iteration
                s = next_s
                
        if method == 'ValIter': 
            t = 1 # Initialize current state, next state and time
            s = self.map[start]
            path.append(start) # Add the starting position in the maze to the path
            next_states = self.__move(s, policy[s]) # Move to next state given the policy and the current state
            map_next_states = [self.map[state] for state in next_states] # Map the next states
            next_s = np.random.choice(map_next_states, p=self.transition_probabilities[map_next_states, s, policy[s]])  # Choose the next state given the transition probabilities
            path.append(self.states[next_s]) # Add the next state to the path
            
            # horizon geometric mean
            horizon = np.random.geometric(1/mean) # Sample the horizon from a geometric distribution
            
            # Loop while state is not the goal state
            while s != next_s and t <= horizon:
                s = next_s # Update state
                next_states = self.__move(s, policy[s]) # Move to next state given the policy and the current state
                map_next_states = [self.map[state] for state in next_states] # Map the next states
                next_s = np.random.choice(map_next_states, p=self.transition_probabilities[map_next_states, s, policy[s]])  # Choose the next state given the transition probabilities
                # next_s = np.random.choice(map_next_states) # Choose the next state randomly
                path.append(self.states[next_s]) # Add the next state to the path
                t += 1 # Update time for next iteration
        
        return [path, horizon] # Return the horizon as well, to plot the histograms for the VI



    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.rewards)


    def possible_actions(self, state):

        """
        Determine the possible actions the player can take from a given state.

        Args:
            state (int): The current state index representing the player's position in the maze.

        Returns:
            list: A list of valid actions (each as a tuple of row and column deltas) 
                that the player can take without exceeding maze boundaries or hitting walls.
        """

        possible_actions = [0] #we can always stay in place

        if self.states[state]!= 'Eaten' and self.states[state]!= 'Win':

            for action in range(1,self.n_actions):

                row_player = self.states[state][0][0] + self.actions[action][0] # Row of the player's next position 
                col_player = self.states[state][0][1] + self.actions[action][1] # Column of the player's next position 
                
                # Is the player getting out of the limits of the maze or hitting a wall?
                impossible_action_player = (row_player == -1 or row_player == self.maze.shape[0] or \
                                            col_player == -1 or col_player == self.maze.shape[1] or \
                                            self.maze[row_player, col_player] == 1 )                

                if not impossible_action_player:
                    possible_actions.append(action)

        return possible_actions
    


    def observations(self, s, a):
        """ Q learning algorithm:
            Given the current state, and action, returns the next state and reward. If the state is terminal, returns done = True.
                :return tuple next_state: next state
                :return float reward: reward
                :return bool done: True if the next state is terminal, False otherwise
        """

        next_states = self.__move(s, a) #possible next states


        map_next_states = [self.map[state] for state in next_states] # Map the next states
        
        next_s = np.random.choice(map_next_states, p=self.transition_probabilities[map_next_states, s, a])  # Choose the next state given the transition probabilities
        # next_s = np.random.choice(map_next_states) # Choose the next state randomly
        
        done = self.states[s] == 'Eaten' or self.states[s] == 'Win' # Check if the next state is terminal
        reward = self.rewards[s, a] # Get the reward

        #if reward > 0 print everything
        # if reward > 0:
        #     print('State:', self.states[s], 'Action:', self.actions_names[a], 'Next state:', self.states[next_s], 'Reward:', reward)

        return next_s, reward, done



############################################
#### Reinforcement Learning Agent class ####
############################################

class RL_agent:

    def __init__(self, env, epsilon, discount_factor, alpha, init, delta=None):
        self.state = None
        self.discount_factor = discount_factor
        self.alpha = alpha
        self.env = env
        self.epsilon = epsilon
        self.done = False
        self.delta = delta
        self.episode = 1

        self.counter = np.zeros((self.env.n_states, self.env.n_actions))
        # initialize Q
        if init == '0':
            self.Q = np.zeros((self.env.n_states, self.env.n_actions))
        elif init == '1':
            self.Q = np.ones((self.env.n_states, self.env.n_actions))
        elif init == 'rand':
            self.Q = np.random.rand(self.env.n_states, self.env.n_actions)
        elif init == '3':
            self.Q = np.ones((self.env.n_states, self.env.n_actions))*3
        else:
            raise ValueError('Unknown initialization, choose between 0, 1, rand, 10')
        # set the terminal states to 0 nomatter the initialization
        self.Q[self.env.map['Eaten'], :] = -100
        self.Q[self.env.map['Win'], :] = 10


    def behavior_policy(self, s): ##epsilon-greedy policy

        possible_act = self.env.possible_actions(s)

        if self.delta is not None:
            self.epsilon = 1/(self.episode ** self.delta)

        if np.random.rand()<self.epsilon: ## explore
            a = np.random.choice(possible_act)

        else: ## greedy action
            max_value = np.max(self.Q[s, possible_act])
            max_actions = [action for action in possible_act if self.Q[s, action] == max_value]
            a = np.random.choice(max_actions)

        return a


    def get_policy(self):
        # the policy is the action that maximizes the Q value of hte possible actions
        policy = []
        for s in range(self.env.n_states):
            if self.env.states[s] != 'Eaten' and self.env.states[s] != 'Win':
                possible_act = self.env.possible_actions(s)
                max_value = np.max(self.Q[s, possible_act])
                max_actions = [action for action in possible_act if self.Q[s, action] == max_value]
                policy.append(max_actions[0])
            else:
                policy.append(0)
        return policy


    def step_Q_learning(self):
        ## observations
        s = self.state
        a = self.behavior_policy(s)
        s_next, reward, self.done =  self.env.observations(s,a)

        ## update Q function
        if not self.done:
            self.counter[s,a] +=1
            lr = 1/(self.counter[s,a]**self.alpha)
            self.Q[s, a] += lr * (reward + self.discount_factor * np.max(self.Q[s_next, :]) - self.Q[s, a])
    
        self.state = s_next

    
    def step_SARSA(self):
        ## observations
        s = self.state
        a = self.behavior_policy(s)
        s_next, reward, self.done =  self.env.observations(s,a)
        a_next = self.behavior_policy(s_next)


        ## update Q function
        if not self.done:
            self.counter[s,a] +=1
            lr = 1/(self.counter[s,a]**self.alpha)
            self.Q[s, a] += lr * (reward + self.discount_factor * self.Q[s_next, a_next] - self.Q[s, a])
            # if s == ini:
                # print('State:', self.env.states[s], 'Action:', self.env.actions_names[a], 'Next state:', self.env.states[s_next], 'Next action:', a_next, 'Reward:', reward, "Q", self.Q[s, a], "Q_next", self.Q[s_next, a_next])
            # print if Q value is updated in the terminal states
            # if reward > 1:
            #     print('State:', self.env.states[s], 'Action:', self.env.actions_names[a], 'Next state:', self.env.states[s_next], 'Reward:', reward, "Q", self.Q[s, a], "Q_next", self.Q[s_next, a_next])
            # if s == self.env.map['Win']:
            #     print('State:', self.env.states[s], 'Action:', self.env.actions_names[a], 'Next state:', self.env.states[s_next], 'Reward:', reward, "Q", self.Q[s, a], "Q_next", self.Q[s_next, a_next])
        # if self.done: # print everything to debug
        #     print('State:', self.env.states[s], 'Action:', self.env.actions_names[a], 'Next state:', self.env.states[s_next], 'Reward:', reward, "Q", self.Q[s, a], "Q_next", self.Q[s_next, a_next])

        # save the Q values in a csv file
        ## update next state
        self.state = s_next
        if self.done:
            self.episode += 1



############################################
############ Extra functions ###############
############################################

def dynamic_programming(env, horizon):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    #TODO

    ### MDP
    P         = env.transition_probabilities
    r         = env.rewards
    n_states  = env.n_states
    n_actions = env.n_actions
    T         = horizon

    ### objectives to compute
    V = np.zeros((env.n_states, horizon+1))
    policy = np.zeros((env.n_states, horizon+1))

    #initialization
    V[env.map['Win'], -1] = 1
    V[env.map['Eaten'], -1] = -100

    for t in range(T-1, -1, -1):
        Q = np.zeros((n_states, n_actions))
        for s in range(n_states):
            for a in range(n_actions):
                Q[s,a] = r[s,a]
                for j in range(env.n_states):
                    Q[s,a] += P[j,s,a]*V[j,t+1]
    
            V[s,t] = np.max(Q[s, :])
            policy[s,t] = np.argmax(Q[s,:])

    return V, policy


def value_iteration(env, gamma, epsilon):
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """

    P = env.transition_probabilities  # Shape: (n_states, n_states, n_actions)
    r = env.rewards                   # Shape: (n_states, n_actions)
    n_states = env.n_states
    n_actions = env.n_actions

    V = np.zeros(n_states)  # Initialize value function
    policy = np.zeros(n_states, dtype=int)  # Initialize policy

    tolerance_thr = epsilon * (1 - gamma) / gamma
    delta = tolerance_thr + 1  # Force at least one iteration

    while delta > tolerance_thr:
        Q = np.zeros((n_states, n_actions))  # Store Q-values for all states and actions

        # Compute Q-values for each action in each state
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = r[s, a] + gamma * np.dot(P[:, s, a], V)

        # Get the best value and action for each state
        V_next = np.max(Q, axis=1)

        # Update convergence criterion
        delta = np.max(np.abs(V_next - V))
        V = V_next
    
    policy = np.argmax(Q, axis=1)

    return V, policy


def train_agent(agent, maze, start, algorithm, epsilon=0.2, discount_factor=49/50, alpha=2/3, num_episodes=50000, init='zeros', delta=None):

    ### initialize Q values and other parameters
    env = Maze(maze)
    agent = RL_agent(env, epsilon, discount_factor, alpha, init, delta)
    v_values = []

    for i in range(num_episodes):
        s = env.map[start]
        agent.state = s
        agent.done = False

        while not agent.done:
            if algorithm == 'Q_learning':
                agent.step_Q_learning()
            elif algorithm == 'SARSA':
                agent.step_SARSA()
            else:
                raise ValueError('Unknown algorithm, choose between Q_learning and SARSA')

        # calculate the value function based on the initial state
        v_values.append(np.max(agent.Q[s, agent.env.possible_actions(s)]))

        #print every 5000 episodes
        if i % 5000 == 0:
            print('Episode:', i, 'Value:', v_values[-1])
        

    policy = agent.get_policy()

    return policy, v_values


def animate_solution(maze, path):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -1: LIGHT_RED, -2: LIGHT_PURPLE, 3: LIGHT_ORANGE}
    
    rows, cols = maze.shape # Size of the maze
    fig = plt.figure(1, figsize=(cols, rows)) # Create figure of the size of the maze

    # Remove the axis ticks and add title
    ax = plt.gca()
    ax.set_title('Policy simulation')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)]

    # Create a table to color
    grid = plt.table(
        cellText = None, 
        cellColours = colored_maze, 
        cellLoc = 'center', 
        loc = (0,0), 
        edges = 'closed'
    )
    
    # Modify the height and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)

    for i in range(0, len(path)):
        if path[i-1] != 'Eaten' and path[i-1] != 'Win':
            grid.get_celld()[(path[i-1][0])].set_facecolor(col_map[maze[path[i-1][0]]])
            grid.get_celld()[(path[i-1][1])].set_facecolor(col_map[maze[path[i-1][1]]])
        if path[i] != 'Eaten' and path[i] != 'Win':
            grid.get_celld()[(path[i][0])].set_facecolor(col_map[-2]) # Position of the player
            grid.get_celld()[(path[i][1])].set_facecolor(col_map[-1]) # Position of the minotaur
        display.display(fig)
        time.sleep(0.5)
        display.clear_output(wait = True)