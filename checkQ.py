from __future__ import print_function
from environment import Environment
from agent import Agent
import itertools
import os

if __name__ == '__main__':
    if os.path.isfile("output/checkQ.txt"):
        os.remove("output/checkQ.txt")
    env = Environment()
    agent = Agent(env)
    agent.monte_carlo_control(1000000)

    for dealer, player, action in itertools.product(range(env.dealer_values), range(env.player_values), range(env.action_values)):
        with open("output/checkQ.txt", "a") as f:
            print("%d\t %d\t %d\t %.5f" % (dealer+1, player+1, action, agent.Q[dealer, player, action]), file = f)
        