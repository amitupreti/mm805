import os
import cv2
import matplotlib.pyplot as plt

from dense_flow import dense_flow

FRAMES_PATH = './frames/'
DENSE_PATH = './optical_flow/'

try:
    os.mkdir(DENSE_PATH)
except FileExistsError:
    pass

if __name__ == '__main__':
    counter = 0
    files = os.listdir(FRAMES_PATH)
    sorted(files)
    files = files[:100]  # doing small for test. Will regenerate the whole later
    for i in range(0, len(files), 2):
        name1 = files[i]
        name2 = files[i + 1]
        frame1 = f'{FRAMES_PATH}{name1}'
        frame2 = f'{FRAMES_PATH}{name2}'

        print(f'Processing Frame1 {frame1} and Frame2 {frame2}')

        f1 = cv2.imread(frame1)
        f2 = cv2.imread(frame2)

        res = dense_flow(f1, f2)

        plt.imshow(res)
        # plt.show()
        name1__no_ext = name1.split('.')[0]
        name2__no_ext = name2.split('.')[0]

        plt.imsave(f'{DENSE_PATH}/dense_flow{counter}__source_{name1__no_ext}_{name2__no_ext}.png', res)
        counter += 1
        # breakpoint()
