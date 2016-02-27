from environment import Environment
from collections import Counter
from utils import Color

def sample_draw(env):    
    card = env.draw()
    return [card.value, card.color]

def compute_freq(red, black, colors, iterations):
    print('Color frequency - red cards %.3f , black cards %.3f' % (colors[Color.red] / iterations, 
                                                                  colors[Color.black] / iterations) )
    print('Card value\t Color(red -1, black 1)\t frequency')                                                        
    for val in range (1, 11):
        print("%d\t %d\t %.3f" % (val, -1, (red[val] / iterations) )) #red
        print("%d\t %d\t %.3f" % (val, 1, (black[val] / iterations) )) #black
      
if __name__ == '__main__':
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
        
    compute_freq(red_values, black_values, card_colors, iterations)
    
    