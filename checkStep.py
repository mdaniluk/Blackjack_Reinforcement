from environment import Environment
from utils import State
from collections import Counter

import numpy as np
#
#class Result:
#    def __init__(self, dealer_card, player_sum, reward):
#        self.dealer_card = dealer_card
#        self.player_sum = player_sum
#        self.reward = reward
        
def chech_step(dealer_card, player_sum, action):
    env = Environment()
    print ('DealerCard\t PlayerSum\t reward\t frequency')
    iterations = 100000
    output = Counter()
    for i in range(0, iterations):
        [reward, state_res] = check(env, dealer_card, player_sum, action)
        output[state_res.dealer_card, state_res.player_sum, reward] += 1
    compute_freq(output, iterations)
    
    
def check(env, dealer_card, player_sum, action):
    state = State(dealer_card, player_sum)
    return env.step(state, action)
    
def compute_freq(output, iterations):
    output_sorted = sorted(output.items(), key=lambda i: i[1], reverse=True)
#    for out in output:
#        print("%d\t %d\t %d\t %.3f" % (out[0], out[1], out[2], output[out] / iterations))   
    for out in output_sorted:
        print("%d\t %d\t %d\t %.3f" % (out[0][0], out[0][1], out[0][2], out[1] / iterations))
    