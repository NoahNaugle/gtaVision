import numpy as np 
import pandas as pd 
from collections import Counter
from random import shuffle
import cv2

# if you want to check to make sure what you recorded is correct
# run this file and it will open up a box of each frame recorded and 
# the recorded key press for each frame in the command prompt

check_data = np.load('C:/gtaVision/Training/data/raw/training_data-1.npy')

for data in check_data: 
    img = data[0]
    choice = data[1]
    cv2.imshow('test',img)
    print(choice)
    if cv2.waitKey(25) & 0xFF == ord('q'):  # press q to quit while on frame box
        cv2.destroyAllWindows()
        break