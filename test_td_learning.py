from environment import Environment
from agent import Agent
import matplotlib.pyplot as plt
import random
random.seed(1)
import numpy as np
from utils import compute_mse
    
if __name__ == '__main__':
    monte_carlo_iterations = 1000000
    td_iterations = 10000
    env = Environment()
    agent = Agent(env)
    agent.monte_carlo_control(monte_carlo_iterations)
    Q_monte_carlo = agent.Q
    
    alphas = np.linspace(0,1,11)
    mse_all = []
    for alpha in alphas:
#        print (alpha)
        agent.reset()
        agent.td_learning(td_iterations, alpha)
        Q_tf = agent.Q
        mse_all.append(compute_mse(Q_tf, Q_monte_carlo, True))
    
    plt.plot(alphas, mse_all, 'r-')
    plt.show()