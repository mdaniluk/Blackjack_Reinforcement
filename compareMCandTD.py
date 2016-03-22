from environment import Environment
from agent import Agent
import random
random.seed(2)

    
if __name__ == '__main__':
    monte_carlo_iterations = 100000
    td_iterations = 100000
    env = Environment()
    agent = Agent(env)
    agent.monte_carlo_control(monte_carlo_iterations)
    Q_monte_carlo = agent.Q
    
    agent.reset()
    agent.td_learning(td_iterations, 0)
    Q_tf = agent.Q
    
    agent.reset()
    agent.linear_sarsa(td_iterations, 0)
    Q_linear = agent.Q
    
    agent.play(Q_monte_carlo, 100000)
    agent.play(Q_tf, 100000)
    agent.play(Q_linear, 100000)