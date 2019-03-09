# balance_data.py

import cv2
import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle

# NOTE: this file and balance_data_Auto.py are very similar. 
# not sure which I used in the past so both are here
# I will delete one of them soon

forward = []
backward = []
left = []
right = []
forward_left = []
forward_right = []
backward_left = []
backward_right = []
nokey = [] 

FILE_I_END = 2

for i in range(1,FILE_I_END+1):      # start, stop, step loop for the training data
    train_data = np.load('C:/gtaVision/Training/data/balanced/test-{}.npy'.format(i)) # location to load the files
    shuffle(train_data) # shuffle the data before getting the lenth of choices
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
    backward = backward
    left = left
    right = right
    forward_left = forward_left
    forward_right = forward_right
    backward_left = backward_left
    backward_right = backward_right
    nokey = nokey

    final_data = forward + backward + left + right + forward_left + forward_right + nokey + backward_left + backward_right 
    shuffle(final_data) # shuffle only the data for that file in range of i

shuffle(final_data) # shuffle all the data
np.save('C:/gtaVision/Training/data/combined/training-master-file.npy'.format(i), final_data) # location to save the data

# beware this can take a very long time to process the file. It takes alot to combine the data files into one file for training.

