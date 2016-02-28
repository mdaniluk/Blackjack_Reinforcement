import numpy as np
from random import random
from utils import  Action
from environment import Environment
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

class Agent:
    def __init__(self, environment):
        self.env = env = environment
        self.N0 = 10.0
        
        # Number of time that action has been selected from state s
        self.N = np.zeros((env.dealer_values, env.player_values, env.action_values))
        
        # expected reward
        self.Q = np.zeros((env.dealer_values, env.player_values, env.action_values))
        
        # Policy
        self.V = np.zeros((env.dealer_values, env.player_values))

    def epsilon_greedy(self, state):
        min_num_action = min(self.N[state.dealer_card - 1, state.player_sum - 1, :])
        eps = self.N0 / (self.N0 + min_num_action)
#        print (eps)
        if random() < eps:
            return Action.getRandomAction()
        else:
            action_value = np.argmax(self.Q[state.dealer_card - 1, state.player_sum - 1,:])
            return Action.get_action(action_value)
            
    def monte_carlo_control(self, iters):       
        for episode in range(0, iters):
            state_episode = self.env.get_initial_state()
            reward_episode = 0
            history = []
            #sample episode
            while not state_episode.terminal:
                action = self.epsilon_greedy(state_episode)
                
                history.append([state_episode, action, reward_episode])
                #update number of visits
                self.N[state_episode.dealer_card - 1, state_episode.player_sum - 1, Action.get_value(action)] += 1
                
                [reward_episode, state_episode] = self.env.step(state_episode, action)
            
            #update Q
            for state, action, reward in history:
                step_size = 1.0 / self.N[state.dealer_card - 1, state.player_sum - 1, Action.get_value(action)]
                Gt = reward_episode
                error = Gt - self.Q[state.dealer_card - 1, state.player_sum - 1, Action.get_value(action)]
                self.Q[state.dealer_card - 1, state.player_sum - 1, Action.get_value(action)] += step_size * error
            
        #update policy based on action-value function
        for (dealer_sum, player_sum), value in np.ndenumerate(self.V):
            self.V[dealer_sum, player_sum] = max(self.Q[dealer_sum, player_sum, :])
    
    def plot_optimal_value_function(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = np.arange(0,self.env.dealer_values, 1)
        y = np.arange(0,self.env.player_values, 1)
        X, Y = np.meshgrid(x, y)
        ax.plot_surface(X+1, Y+1, self.V[X,Y], rstride=1, cstride=1, cmap= 'hot', linewidth=0, antialiased=False)
        plt.show()
            

if __name__ == '__main__':
    env = Environment()
    agent = Agent(env)
    agent.monte_carlo_control(1000000)
    agent.plot_optimal_value_function()
    print ('a')