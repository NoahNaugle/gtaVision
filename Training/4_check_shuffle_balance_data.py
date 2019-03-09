import cv2
import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle


# this file only checks how much training data you actually have after 
# step 3 which took a lot of data away to balance the data
# this way you can get an idea of how much training you really need
# reminder: Good results come with good data

train_data = np.load('C:/gtaVision/Training/data/balanced/test-1.npy')

df = pd.DataFrame(train_data)

forward = []
backward = []
left = []
right = []
forward_left = []
forward_right = []
backward_left = []
backward_right = []
nokey = [] 

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

forward = forward
print('forward:', (len(forward)), '             [1,0,0,0,0,0,0,0,0]')
backward = backward
print('backward:', (len(backward)), '            [0,1,0,0,0,0,0,0,0]')
left = left
print('left:', (len(left)), '               [0,0,1,0,0,0,0,0,0]')
right = right
print('right:', (len(right)), '              [0,0,0,1,0,0,0,0,0]')
forward_left = forward_left
print('forward_left:', (len(forward_left)), '       [0,0,0,0,1,0,0,0,0]')
forward_right = forward_right
print('forward_right:', (len(forward_right)), '      [0,0,0,0,0,1,0,0,0]')
backward_left = backward_left
print('backward_left:', (len(backward_left)), '       [0,0,0,0,0,0,1,0,0]')
backward_right = backward_right
print('backward_right:', (len(backward_right)), '      [0,0,0,0,0,0,0,1,0]')
nokey = nokey
print('nokey:', (len(nokey)), '               [0,0,0,0,0,0,0,0,1]')

final_data = forward + backward + left + right + forward_left + forward_right + nokey + backward_left + backward_right 
print('Final data #:', (len(final_data)))