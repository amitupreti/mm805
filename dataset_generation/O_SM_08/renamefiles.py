import os
import cv2
import matplotlib.pyplot as plt
import gc
from dense_flow import dense_flow
from time import sleep

FRAMES_PATH = ['./O_SM_08/', 'O_SM_08-GT/']

if __name__ == '__main__':
    counter = 0
    for path in FRAMES_PATH:
        files = os.listdir(path)
        # files = sorted(files)
        # breakpoint()
        # files = files[3977:]  # doing small for test. Will regenerate the whole later
        for i in range(0, len(files)):
            file_name = files[i]
            file_new_name = file_name.split('-')[-1]

            print(f'Renamed {path+file_name} to {path+file_new_name}')
            # breakpoint()
            os.rename(path + file_name, path + file_new_name)