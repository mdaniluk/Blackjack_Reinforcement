from environment import Environment
from agent import Agent
import matplotlib.pyplot as plt
import random
random.seed(1)
import numpy as np
from utils import compute_mse, Trace

if __name__ == '__main__':
    """
    test sarsa lambda algorithm
    """
    
    """ the learning curve of mean-squared error
        against episode number for lambda = 0 and lambda = 1
    """
    env = Environment()
    agent = Agent(env)
    print ('the learning curve of mean-squared error against episode number for')
    print ('lambda = 0')
    agent.td_learning(10000, 0.0, True, trace = Trace.accumulating)
    
    agent.reset()
    print ('lambda = 1')
    agent.td_learning(10000, 1.0, True, trace = Trace.accumulating)
    
    agent.reset()
    print ('The mean-squared error against lambda')
    monte_carlo_iterations = 1000000
    td_iterations = 10000
    
    agent.monte_carlo_control(monte_carlo_iterations)
    Q_monte_carlo = agent.Q
    
    alphas = np.linspace(0,1,11)
    mse_all = []
    avg_iters = 1 # change it to average over more iterations
    for alpha in alphas:
        mse_current = 0
        for i in range (0,avg_iters):
            agent.reset()
            agent.td_learning(td_iterations, alpha)
            Q_tf = agent.Q           
            mse_current += compute_mse(Q_tf, Q_monte_carlo, False)
            
        mse_all.append(mse_current / avg_iters)
    
    plt.plot(alphas, mse_all, 'r-')
    plt.xlabel("lambda")
    plt.ylabel("MSE")
    plt.show()