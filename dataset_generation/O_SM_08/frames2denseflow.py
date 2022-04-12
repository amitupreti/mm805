import os
import cv2
import matplotlib.pyplot as plt
import gc
from dense_flow import dense_flow
from time import sleep
from PIL import Image

FRAMES_PATH = './O_SM_08/'
# FRAMES_PATH = './O_SM_08-GT/'
DENSE_PATH = './optical_flow/'
# DENSE_PATH = './optical_flow1/'

try:
    os.mkdir(DENSE_PATH)
except FileExistsError:
    pass

if __name__ == '__main__':
    counter = 0
    files = os.listdir(FRAMES_PATH)
    files.sort(key = lambda x: int(x.split('.')[0]))
    # files.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))

    # breakpoint()
    # files = files[3977:]  # doing small for test. Will regenerate the whole later
    for i in range(0, len(files), 2):
        name1 = files[i]
        name2 = files[i + 1]
        frame1 = f'{FRAMES_PATH}{name1}'
        frame2 = f'{FRAMES_PATH}{name2}'

        f1 = cv2.imread(frame1)
        f2 = cv2.imread(frame2)
        try:
            res = dense_flow(f1, f2)
        except:
            continue

        # plt.imshow(res)
        # plt.show()
        name1__no_ext = name1.split('.')[0]
        name2__no_ext = name2.split('.')[0]

        filename = f'{DENSE_PATH}{counter}__source_{name1__no_ext}_{name2__no_ext}.tif'
        print(f'Processing Frame1 {frame1} and Frame2 {frame2} --> {filename}')
        # breakpoint()
        # plt.imsave(filename, res)
        Image.fromarray(res).save(filename)
        gc.collect()
        counter += 1
        sleep(0.01)
        # breakpoint()
