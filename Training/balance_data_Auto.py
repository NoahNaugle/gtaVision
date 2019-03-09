# balance_data.py

import cv2
import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle

# NOTE: This is a work in progress file
# it loops through all the training files and creates a new
# large file for tensorflow training. 
# It takes a long time to process check notes at the very bottom.

forward = []
backward = []
left = []
right = []
forward_left = []
forward_right = []
backward_left = []
backward_right = []
nokey = [] 

FILE_I_END = 10

for i in range(9,FILE_I_END+1):      # start, stop, step loop for the training data
    train_data = np.load('C:/gtaVision/Training/data/raw/training_data-{}.npy'.format(i)) # location to load the files
    shuffle(train_data)             # shuffle the data before getting the lenth of choices
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

    
    forward = forward[:len(left)][:len(right)][:len(forward_left)][:len(forward_right)][:len(nokey)]
    backward = backward[:len(forward)]
    left = left[:len(forward)]
    right = right[:len(forward)]
    forward_left = forward_left[:len(forward)]
    forward_right = forward_right[:len(forward)]
    backward_left = backward_left      
    backward_right = backward_right    
    nokey = nokey[:len(forward)]

    final_data = forward + backward + left + right + forward_left + forward_right + nokey + backward_left + backward_right 
    shuffle(final_data) # shuffle only the data for that file in range of i

shuffle(final_data) # shuffle all the data
np.save('C:/gtaVision/Training/data/combined/big-file-5.npy'.format(i), final_data) # location to save the data

# beware this can take a very long time to process the file. It takes alot to combine the data files into one file for training.

