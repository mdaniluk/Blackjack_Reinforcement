from enum import Enum
from random import random

def compute_mse(Q1, Q2, print_it = False):
    """
    compute mean squre error
    """
    mse = ((Q1 - Q2) ** 2).mean(axis = None)
    if print_it:
        print (mse)
        print ('\n')
    return mse
    
class Color(Enum):
    """
    Enum which represent a Color
    """
    red = -1
    black = 1

class Card:
    """
    Card object
    """
    def __init__(self, value, color):
        self.value = value
        self.color = color  
        
class Action(Enum):
    """
    Class represent Action
    """
    hit = 0
    stick = 1
    
    def getRandomAction():
        r = random()
        return Action.hit if r < 0.5 else Action.stick
    def get_action(n):
        return Action.hit if n == 0 else Action.stick
        
    def get_value(action):
        return 0 if action == Action.hit else 1
    
class State:
    """
    Class represent State
    """
    def __init__(self, dealer_card, player_sum, terminal = False):
        self.dealer_card = dealer_card
        self.player_sum = player_sum
        self.terminal = terminal

class Trace(Enum):
    accumulating = 0
    replacing = 1
    dutch = 2     
        
        