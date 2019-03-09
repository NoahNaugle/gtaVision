# balance_data.py

import cv2
import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle

train_data = np.load('C:/Users/Noah/Desktop/Training_Data/fresh_training_data/test-1.npy')

df = pd.DataFrame(train_data)
#print(Counter(df[1].apply(str)))


forward = []
backward = []
left = []
right = []
forward_left = []
forward_right = []
backward_left = []
backward_right = []
nokey = [] 

shuffle(train_data)     # shuffle the data before chopping it up

for data in train_data:
    img = data[0]
    choice = data[1]

    if choice == [1,0,0,0,0,0,0,0,0]:
        forward.append([img,choice])
    elif choice == [0,1,0,0,0,0,0,0,0]:
        backward.append([img,choice])
    elif choice == [0,0,1,0,0,0,0,0,0]:
        left.append([img,choice])
    elif choice == [0,0,0,1,0,0,0,0,0]:
        right.append([img,choice])
    elif choice == [0,0,0,0,1,0,0,0,0]:
        forward_left.append([img,choice])
    elif choice == [0,0,0,0,0,1,0,0,0]:
        forward_right.append([img,choice])
    elif choice == [0,0,0,0,0,0,1,0,0]:
        backward_left.append([img,choice])
    elif choice == [0,0,0,0,0,0,0,1,0]:
        backward_right.append([img,choice])
    elif choice == [0,0,0,0,0,0,0,0,1]:
        nokey.append([img,choice])
    else:
        print('no matches')

# start balancing the data
# I have customized this to my data results as backward left and right 
# are rarely used I do not balance those keys since there are so few and 
# if I did balance them it would take forever to get enough data

# also I commented out the print statements as this is more of a save file
# in the next step you can see how much data you balanced out

forward = forward[:len(left)][:len(right)][:len(forward_left)][:len(forward_right)][:len(nokey)]
#print('forward:', (len(forward)), '             [1,0,0,0,0,0,0,0,0]')

backward = backward[:len(forward)]
#print('backward:', (len(backward)), '            [0,1,0,0,0,0,0,0,0]')

left = left[:len(forward)]
#print('left:', (len(left)), '               [0,0,1,0,0,0,0,0,0]')

right = right[:len(forward)]
#print('right:', (len(right)), '              [0,0,0,1,0,0,0,0,0]')

forward_left = forward_left[:len(forward)]
#print('forward_left:', (len(forward_left)), '       [0,0,0,0,1,0,0,0,0]')

forward_right = forward_right[:len(forward)]
#print('forward_right:', (len(forward_right)), '      [0,0,0,0,0,1,0,0,0]')

backward_left = backward_left      
#print('backward_left:', (len(backward_left)), '       [0,0,0,0,0,0,1,0,0]')

backward_right = backward_right   
#print('backward_right:', (len(backward_right)), '      [0,0,0,0,0,0,0,1,0]')

nokey = nokey[:len(forward)]
#print('nokey:', (len(nokey)), '               [0,0,0,0,0,0,0,0,1]')


final_data = forward + backward + left + right + forward_left + forward_right + nokey + backward_left + backward_right 
shuffle(final_data)
#print('Final data #:', (len(final_data)))



np.save('C:/gtaVision/Training/data/balanced/balance_data-94.npy', final_data) 



