from environment import Environment
from agent import Agent
import matplotlib.pyplot as plt
import random
random.seed(1)
import numpy as np
from utils import compute_mse, Trace

if __name__ == '__main__':
    monte_carlo_iterations = 1000000
    td_iterations = 10000
    env = Environment()
    agent = Agent(env)
    agent.monte_carlo_control(monte_carlo_iterations)
    Q_monte_carlo = agent.Q
    
    alphas = np.linspace(0,1,11)
    mse_all_acc = []
    mse_all_replace = []
    mse_all_dutch = []
    avg_iters = 10
    for alpha in alphas:
        mse_current = 0
        for i in range (0,avg_iters):
            agent.reset()
            agent.td_learning(td_iterations, alpha, trace = Trace.accumulating)
            Q_tf = agent.Q           
            mse_current += compute_mse(Q_tf, Q_monte_carlo, True)
            
        mse_all_acc.append(mse_current / avg_iters)
        
        mse_current = 0
        for i in range (0,avg_iters):
            agent.reset()
            agent.td_learning(td_iterations, alpha, trace = Trace.replacing)
            Q_tf = agent.Q           
            mse_current += compute_mse(Q_tf, Q_monte_carlo, True)
            
        mse_all_replace.append(mse_current / avg_iters)
        
        mse_current = 0
        for i in range (0,avg_iters):
            agent.reset()
            agent.td_learning(td_iterations, alpha, trace = Trace.dutch)
            Q_tf = agent.Q           
            mse_current += compute_mse(Q_tf, Q_monte_carlo, True)
            
        mse_all_dutch.append(mse_current / avg_iters)
    
    p1, = plt.plot(alphas, mse_all_acc, 'r-')
    p2, = plt.plot(alphas, mse_all_replace, 'b-')
    p3, = plt.plot(alphas, mse_all_dutch, 'g-')
    plt.xlabel("lambda")
    plt.ylabel("MSE")
    plt.legend([p1, p2, p3], ['accumulating trace', 'replacing trace', 'dutch trace'], loc="best")

    plt.show()