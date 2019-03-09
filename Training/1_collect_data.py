import numpy as np
from grabscreen import grab_screen
import cv2
import time
from getkeys import key_check
import os

# this file is going to record a part of your screen and your key inputs below.

w = [1,0,0,0,0,0,0,0,0]
s = [0,1,0,0,0,0,0,0,0]
a = [0,0,1,0,0,0,0,0,0]
d = [0,0,0,1,0,0,0,0,0]
wa = [0,0,0,0,1,0,0,0,0]
wd = [0,0,0,0,0,1,0,0,0]
sa = [0,0,0,0,0,0,1,0,0]
sd = [0,0,0,0,0,0,0,1,0]
nk = [0,0,0,0,0,0,0,0,1]

starting_value = 163    # if you are starting fresh set this to 0

# change directory below to where you're going to be creating data or loading previous data
while True:
    file_name = 'C:/gtaVision/Training/raw/training_data-{}.npy'.format(starting_value)

    if os.path.isfile(file_name):
        print('File exists, moving along',starting_value)
        starting_value += 1
    else:
        print('File does not exist, starting fresh!',starting_value)
        break


def keys_to_output(keys):
    '''
    Convert keys to a ...multi-hot... array
     0  1  2  3  4   5   6   7    8
    [W, S, A, D, WA, WD, SA, SD, NOKEY] boolean values.
    '''
    output = [0,0,0,0,0,0,0,0,0]

    if 'W' in keys and 'A' in keys:
        output = wa
    elif 'W' in keys and 'D' in keys:
        output = wd
    elif 'S' in keys and 'A' in keys:
        output = sa
    elif 'S' in keys and 'D' in keys:
        output = sd
    elif 'W' in keys:
        output = w
    elif 'S' in keys:
        output = s
    elif 'A' in keys:
        output = a
    elif 'D' in keys:
        output = d
    else:
        output = nk
    return output


def main(file_name, starting_value):
    training_data = []  
    for i in list(range(4))[::-1]:  # countdown timer to get setup to record data
        print(i+1)
        time.sleep(1)

    last_time = time.time()
    paused = False
    print('STARTING!!!')
    while(True):
        
        if not paused:
            # 1600 x 900 windowed mode 16:9 ratio
            screen = grab_screen(region=(0,40,1600,900))
            last_time = time.time()
            # resize to something a bit more acceptable for a CNN
            screen = cv2.resize(screen, (480,270))
            # run a color convert:
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            
            keys = key_check()
            output = keys_to_output(keys)
            training_data.append([screen,output])

            #print('loop took {} seconds'.format(time.time()-last_time))
            last_time = time.time()
##            cv2.imshow('window',cv2.resize(screen,(640,360)))
##            if cv2.waitKey(25) & 0xFF == ord('q'):
##                cv2.destroyAllWindows()
##                break

            if len(training_data) % 2500 == 0:
                print(len(training_data))
                
                if len(training_data) == 5000:
                    np.save(file_name,training_data)
                    print('SAVED')                      # new .npy file is saved
                    training_data = []
                    print(file_name[31:])              # not necessary if you don't want see what number you are on
                    # or print(starting_value)
                    starting_value += 1                 # add +1 to each new training data file
                    # change to your directory location to save the file
                    file_name = 'C:/gtaVision/Training/raw/training_data-{}.npy'.format(starting_value)
                    
                    

                

        keys = key_check()
        if 'T' in keys:     # if you need to pause recording data press 'T' to pause and 'T' again to unpause
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)


if __name__ == "__main__":
    main(file_name, starting_value)