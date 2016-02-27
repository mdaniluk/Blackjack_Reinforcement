from random import random, randint
from utils import Color, Card, Action, State

class Environment:
    
    def __init__(self):
        self.probability_red = 1.0 / 3.0
        self.card_min = 1
        self.card_max = 10
        self.win = 21
        self.loss = 1
        self.dealer_stick = 17
        
    def draw(self):
        if random() < self.probability_red:
            return Card(randint(self.card_min, self.card_max), Color.red)
        else:
            return Card(randint(self.card_min, self.card_max), Color.black)
            
    def step(self, state, action):
        if(action == Action.hit):
            card = self.draw()
            player_sum = self.add_card(state.player_sum, card)
            state_result = State(state.dealer_card, player_sum)
            if (state_result.player_sum > self.win or state_result.player_sum < self.loss):
                reward = -1
                state_result.terminal = True               
            else:
                reward = 0                         
        elif(action == Action.stick):
            state_result = State(state.dealer_card, state.player_sum)
            dealer_sum = state.dealer_card
            while(dealer_sum < self.dealer_stick):
                card = self.draw()
                dealer_sum = self.add_card(dealer_sum, card)
                state_result.dealer_card = dealer_sum
                if (dealer_sum > self.win or dealer_sum < self.loss):
                    reward = 1
                    state_result.terminal = True
                    return [reward, state_result] 
                    
            if (dealer_sum > state.player_sum):
                reward = -1
            elif(dealer_sum == state.player_sum):
                reward = 0
            else:
                reward = 1
                     
        return [reward, state_result]   
                    
    def add_card(self, value, card):
        if (card.color == Color.black):
            value += card.value
        elif(card.color == Color.red):
            value -= card.value
        return value
                
    
if __name__ == '__main__':
    env = Environment()
    env.draw()