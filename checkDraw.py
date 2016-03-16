from __future__ import print_function
from environment import Environment
from collections import Counter
from utils import Color
import os

def sample_draw(env):    
    card = env.draw()
    return [card.value, card.color]

def compute_freq(red, black, colors, iterations, to_file):
    print('Color frequency - red cards %.3f , black cards %.3f' % (colors[Color.red] / iterations, 
                                                                  colors[Color.black] / iterations) )
    print('Card value\t Color(red -1, black 1)\t frequency') 
    mse_red= []  
    mse_black= []                                                     
    for val in range (1, 11):
        if to_file:
            with open("output/chechDraw.txt", "a") as f:
                print("%d\t %d\t %.3f" % (val, -1, (red[val] / iterations)), file = f) #red
                print("%d\t %d\t %.3f" % (val, 1, (black[val] / iterations)), file = f) #black
                mse_red.append(abs(red[val] / iterations - 0.033))
                mse_black.append(abs(black[val] / iterations - 0.066))
        print("%d\t %d\t %.3f" % (val, -1, (red[val] / iterations))) #red
        print("%d\t %d\t %.3f" % (val, 1, (black[val] / iterations))) #black
    if to_file:
        print (sum(mse_red) / len(mse_red))    
        print (sum(mse_black) / len(mse_black))    
if __name__ == '__main__':
    if os.path.isfile("output/chechDraw.txt"):
        os.remove("output/chechDraw.txt")
    print ('Test draw method')
    iterations = 1000
    red_values = Counter()
    black_values = Counter()
    card_colors = Counter()
    env = Environment()
    for i in range(0,iterations):
        val, color = sample_draw(env)
        card_colors[color] += 1
        if (color == Color.red):
            red_values[val] += 1
        elif(color == Color.black):
            black_values[val] += 1
        
    compute_freq(red_values, black_values, card_colors, iterations, True)
    
    