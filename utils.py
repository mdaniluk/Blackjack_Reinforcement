from enum import Enum

class Color(Enum):
    red = -1
    black = 1

class Card:
    def __init__(self, value, color):
        self.value = value
        self.color = color  
        
class Action(Enum):
    hit = 0
    stick = 1
    
class State:
    def __init__(self, dealer_card, player_sum, terminal = False):
        self.dealer_card = dealer_card
        self.player_sum = player_sum
        self.terminal = terminal
        
        
        