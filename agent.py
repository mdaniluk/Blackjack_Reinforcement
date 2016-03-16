import numpy as np
from random import random
from utils import  Action, State
from environment import Environment
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import compute_mse
import random as r
r.seed(1)

class Agent:
    def __init__(self, environment):
        self.env = env = environment
        self.N0 = 100.0
        
        # Number of time that action has been selected from state s
        self.N = np.zeros((env.dealer_values, env.player_values, env.action_values))
        
        # expected reward
        self.Q = np.zeros((env.dealer_values, env.player_values, env.action_values))
        
        # Policy
        self.V = np.zeros((env.dealer_values, env.player_values))
        self.weights = np.random.uniform(low=-0.00, high=0.0, size=(36))
#        self.Q = np.random.uniform(low=-0, high=0, size=(420))
#        self.Q = np.reshape(self.Q, (env.dealer_values, env.player_values, env.action_values))

        
    def reset(self):
        self.N = np.zeros((self.env.dealer_values, self.env.player_values, self.env.action_values))
        self.Q = np.zeros((self.env.dealer_values, self.env.player_values, self.env.action_values))
        self.V = np.zeros((self.env.dealer_values, self.env.player_values))
        
    def epsilon_greedy(self, state):   
        if state.dealer_card > self.N.shape[0] or state.player_sum > self.N.shape[1]:
            min_num_action = 0
        else:
            min_num_action = min(self.N[state.dealer_card - 1, state.player_sum - 1, :]) 
                
        eps = self.N0 / (self.N0 + min_num_action)

#        print (eps)
        if random() < eps:
            return Action.getRandomAction()
        else:
            action_value = np.argmax(self.Q[state.dealer_card - 1, state.player_sum - 1,:])
            return Action.get_action(action_value)
     
           
    def monte_carlo_control(self, iters):   
        num_wins = 0
        optimal_policy = np.zeros((self.env.dealer_values, self.env.player_values))
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
            if (Gt == 1):
                num_wins = num_wins + 1
            
        print ("Percentage of win %.3f" % (num_wins / iters * 100.0))
        #update policy based on action-value function
        for (dealer_sum, player_sum), value in np.ndenumerate(self.V):
            if self.Q[dealer_sum, player_sum, 1] > self.Q[dealer_sum, player_sum, 0]:
                optimal_policy[dealer_sum, player_sum] = 1
            self.V[dealer_sum, player_sum] = max(self.Q[dealer_sum, player_sum, :])
        plt.pcolor(optimal_policy)
        plt.xlabel('Player sum')
        plt.ylabel('Dealer showing')
        plt.show()
    
    def td_learning(self, iters, lambda_, compare_to_monctecarlo = False):
        if compare_to_monctecarlo:
            monte_carlo_iterations = 1000000
            env = Environment()
            agent = Agent(env)
            agent.monte_carlo_control(monte_carlo_iterations)
            Q_monte_carlo = agent.Q
            mse_all = []
            
        for episode in range(0, iters):
            E = np.zeros(((self.env.dealer_values, self.env.player_values, self.env.action_values)))  
            
            #initialize state and action          
            state = self.env.get_initial_state()
            reward = 0
            action = self.epsilon_greedy(state)
#            self.N[state.dealer_card - 1, state.player_sum - 1, Action.get_value(action)] += 1 
            while not state.terminal:                   
#                update number of visits
                self.N[state.dealer_card - 1, state.player_sum - 1, Action.get_value(action)] += 1              
                [reward, state_forward] = self.env.step(state, action)                 
                action_forward = self.epsilon_greedy(state_forward)  
                
                if not state_forward.terminal:
                    current_estimate = reward + self.Q[state_forward.dealer_card - 1, state_forward.player_sum - 1, Action.get_value(action_forward)]
#                    self.N[state_forward.dealer_card - 1, state_forward.player_sum - 1, Action.get_value(action_forward)] += 1
                else:
                    current_estimate = reward
                    
                previous_estimate = self.Q[state.dealer_card - 1, state.player_sum - 1, Action.get_value(action)]
                delta = current_estimate - previous_estimate
                
                E[state.dealer_card - 1, state.player_sum - 1, Action.get_value(action)] += 1
                
                step_size = 1.0 / self.N[state.dealer_card - 1, state.player_sum - 1, Action.get_value(action)]
#                print (step_size)
                self.Q += step_size * delta * E
                E = lambda_ * E
                #update Q
#                if not state_forward.terminal:
#                    update = reward + (self.Q[state_forward.dealer_card - 1, state_forward.player_sum - 1, Action.get_value(action_forward)] -
#                            self.Q[state.dealer_card - 1, state.player_sum - 1, Action.get_value(action)])
#                else:
#                    update = reward - self.Q[state.dealer_card - 1, state.player_sum - 1, Action.get_value(action)]
#
#                self.Q[state.dealer_card - 1, state.player_sum - 1, Action.get_value(action)] += (alpha * update)
                action = action_forward
                state = state_forward
            
            if compare_to_monctecarlo:
#                print (compute_mse(self.Q, Q_monte_carlo))
                mse_all.append(compute_mse(self.Q, Q_monte_carlo))
#                print(compute_mse(self.Q, Q_monte_carlo))
  
        if compare_to_monctecarlo:
#            print (mse_all[-1])
            plt.plot(range(0, iters), mse_all, 'r-')
            plt.show()
                                 
        #update policy based on action-value function
        for (dealer_sum, player_sum), value in np.ndenumerate(self.V):
            self.V[dealer_sum, player_sum] = max(self.Q[dealer_sum, player_sum, :])
        print ('a')

#
    def get_feature_vector(self, state, action):
        feature_vector = np.zeros((3,6,2))
        dealer_cuboids = [[1,4], [4,7], [7,10]]
        player_cuboids = [[1,6], [4,9], [7,12], [10,15], [13,18], [16,21]]
        action_cuboids = [0,1]
        for d_idx, d in enumerate(dealer_cuboids):
            for p_idx, p in enumerate(player_cuboids):
                for a_idx, a in enumerate(action_cuboids):
                    if state.dealer_card >= d[0] and state.dealer_card <= d[1] and state.player_sum >= p[0] and state.player_sum <= p[1] and action.get_value() == a_idx:
                        feature_vector[d_idx,p_idx,a_idx] = 1

#        feature_vector[state.dealer_card-1, state.player_sum-1, action.get_value()] = 1
        return np.reshape(feature_vector, 36)
    
    def epsilon_greedy_linear_constant(self, state, eps_ = 0.1): 
        eps = eps_
        if random() < eps:
            return Action.getRandomAction()
        else:
            actionHit_value = sum(self.get_feature_vector(state, Action.hit) * self.weights)
            actionStick_value = sum(self.get_feature_vector(state, Action.stick) * self.weights)
            action = Action.hit if actionHit_value > actionStick_value else Action.stick
            return action
    
    def estimate_Q(self, state, action):
          return np.dot(self.get_feature_vector(state, action), self.weights)
#        return self.Q[state.dealer_card - 1, state.player_sum - 1, Action.get_value(action)]
    
    def approximation_to_Q(self):
        Q = np.zeros((self.env.dealer_values, self.env.player_values, self.env.action_values)) 
        for (dealer_sum, player_sum), value in np.ndenumerate(self.V):
            s = State(dealer_sum+1, player_sum+1)
            Q[dealer_sum, player_sum ,0] = np.dot(self.get_feature_vector(s, Action.hit), self.weights)
            Q[dealer_sum, player_sum ,1] = np.dot(self.get_feature_vector(s, Action.stick), self.weights)
        return Q
    def linear_sarsa(self, iters, lambda_, compare_to_monctecarlo = False):          
        if compare_to_monctecarlo:
            monte_carlo_iterations = 1000000
            env = Environment()
            agent = Agent(env)
            agent.monte_carlo_control(monte_carlo_iterations)
            Q_monte_carlo = agent.Q
            mse_all = []
            
        for episode in range(0, iters):
            if (episode % 1000 == 0):
                print (episode)
#            E = np.zeros(((self.env.dealer_values, self.env.player_values, self.env.action_values))) 
            E = np.zeros(36) 
            #initialize state and action          
            state = self.env.get_initial_state()
            reward = 0
            action = self.epsilon_greedy_linear_constant(state)
#            self.N[state.dealer_card - 1, state.player_sum - 1, Action.get_value(action)] += 1 
            while not state.terminal:                   
#                update number of visits
                self.N[state.dealer_card - 1, state.player_sum - 1, Action.get_value(action)] += 1              
                [reward, state_forward] = self.env.step(state, action)                 
                action_forward = self.epsilon_greedy_linear_constant(state_forward)  
                
                if not state_forward.terminal:
                    current_estimate = reward + self.estimate_Q(state_forward, action_forward)
#                    self.N[state_forward.dealer_card - 1, state_forward.player_sum - 1, Action.get_value(action_forward)] += 1
                else:
                    current_estimate = reward
                    
                previous_estimate = self.estimate_Q(state, action)
                delta = current_estimate - previous_estimate
                
#                E[self.get_feature_vector(state, action) == 1] += 1
                E = np.add(E, self.get_feature_vector(state, action))
#                step_size = 1.0 / self.N[state.dealer_card - 1, state.player_sum - 1, Action.get_value(action)]
                step_size = 0.01                
#                print (step_size)
                self.weights += step_size * delta * E
                E = lambda_ * E

                action = action_forward
                state = state_forward
            if compare_to_monctecarlo:
#                print (compute_mse(self.Q, Q_monte_carlo))
                mse_all.append(compute_mse(self.approximation_to_Q(), Q_monte_carlo))
#                print(compute_mse(self.Q, Q_monte_carlo))
  
        if compare_to_monctecarlo:
#            print (mse_all[-1])
            plt.plot(range(0, iters), mse_all, 'r-')
            plt.show()
            
        for (dealer_sum, player_sum), value in np.ndenumerate(self.V):
            s = State(dealer_sum+1, player_sum+1)
            self.Q[dealer_sum, player_sum ,0] = np.dot(self.get_feature_vector(s, Action.hit), self.weights)
            self.Q[dealer_sum, player_sum ,1] = np.dot(self.get_feature_vector(s, Action.stick), self.weights)
            self.V[dealer_sum, player_sum] = max(self.estimate_Q(s,Action.hit), self.estimate_Q(s,Action.stick))
        print ('a')      
        
    def plot_optimal_value_function(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = np.arange(0,self.env.dealer_values, 1)
        y = np.arange(0,self.env.player_values, 1)
        X, Y = np.meshgrid(x, y)
        ax.plot_surface(X+1, Y+1, self.V[X,Y], rstride=1, cstride=1, cmap= 'hot', linewidth=0, antialiased=False)
        ax.set_xlabel('Dealer showing')
        ax.set_ylabel('Player sum')       
        plt.show()
            

if __name__ == '__main__':
    env = Environment()
    agent = Agent(env)
    agent.monte_carlo_control(1000000)
#    agent.td_learning(10000, 0.5, False)
#    agent.linear_sarsa(10000,1, True)
    agent.plot_optimal_value_function()
#    print ('a')