from __future__ import print_function
from environment import Environment
from utils import State, Action
from collections import Counter
import os

import numpy as np
    
def chech_step(dealer_card, player_sum, action, file_name):
    if os.path.isfile("output/" + file_name):
        os.remove("output/" + file_name)
    env = Environment()
    print ('DealerCard\t PlayerSum\t reward\t frequency')
    iterations = 1000
    output = Counter()
    for i in range(0, iterations):
        [reward, state_res] = check(env, dealer_card, player_sum, action)
        if state_res.terminal:
            output[0, 0, reward] += 1
        else:
            output[state_res.dealer_card, state_res.player_sum, reward] += 1
    compute_freq(output, iterations, file_name)
    
    
def check(env, dealer_card, player_sum, action):
    state = State(dealer_card, player_sum)
    return env.step(state, action)
    
def compute_freq(output, iterations, file_name):
    output_sorted = sorted(output.items(), key=lambda i: i[1], reverse=True)
#    for out in output:
#        print("%d\t %d\t %d\t %.3f" % (out[0], out[1], out[2], output[out] / iterations))   
    for out in output_sorted:
        print("%d\t %d\t %d\t %.3f" % (out[0][0], out[0][1], out[0][2], out[1] / iterations))
        with open("output/" + file_name, "a") as f:
            print("%d\t %d\t %d\t %.3f" % (out[0][0], out[0][1], out[0][2], out[1] / iterations), file = f)
    
if __name__ == '__main__':
    chech_step(1,18,Action.stick, "checkStepDealer1Player18Action1.txt")
    chech_step(10,15,Action.stick, "checkStepDealer10Player15Action1.txt")
    chech_step(1,10,Action.hit, "checkStepDealer1Player10Action0.txt")
    chech_step(1,1,Action.hit, "checkStepDealer1Player1Action0.txt")
    