import os
import cv2
import matplotlib.pyplot as plt
import gc
from dense_flow import dense_flow
from time import sleep
import numpy as np
from PIL import Image
from numpy import asarray

FRAMES_PATH = './O_SM_08-GT/'
DENSE_PATH = './labels/'


maxval = 255
thresh = 10# IF PIXEL VALUE IS MORE THAN 30 MAKE IT WHITE ELSE BLACK

try:
    os.mkdir(DENSE_PATH)
except FileExistsError:
    pass

if __name__ == '__main__':
    counter = 0
    files = os.listdir(FRAMES_PATH)
    files.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))

    # breakpoint()
    # files = files[3977:]  # doing small for test. Will regenerate the whole later
    for i in range(0, len(files), 2):
        # i=150
        name1 = files[i]
        name2 = files[i + 1]
        frame1 = f'{FRAMES_PATH}{name1}'
        frame2 = f'{FRAMES_PATH}{name2}'

        # PIL images into NumPy arrays
        numpydata1 = asarray(Image.open(frame1))
        numpydata2 = asarray(Image.open(frame2))
        combined = numpydata1 + numpydata2
        im_gray = np.array(Image.fromarray(combined).convert('L'))
        im_bin = (im_gray > thresh) * maxval

        name1__no_ext = name1.split('.')[0].split('_')[-1]
        name2__no_ext = name2.split('.')[0].split('_')[-1]
        filename = f'{DENSE_PATH}{counter}__source_{name1__no_ext}_{name2__no_ext}.gif'

        print(f'Processing Frame1 {frame1} and Frame2 {frame2} --> {filename}')
        # breakpoint()
        plt.imsave(filename, im_bin, cmap='gray')
        # Image.fromarray(im_bin.astype(np.uint8)).save(filename)
