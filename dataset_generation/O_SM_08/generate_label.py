import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from dense_flow import dense_flow

DENSE_PATH = './optical_flow/'
LABEL_PATH = './labels/'
pathlib.Path(LABEL_PATH).mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    # read all images in optical flow folder and then generate mask.
    # Mask will be an image of same size as the original image with all pixels set to 255.
    # mask will be just a 2D array (height, width)  i.e no color channels.
    # The mask will be saved in the mask folder.
    # mask format is .gif(following chengui's format)

    for i, file in enumerate(os.listdir(DENSE_PATH)):
        if file.endswith(".png"):
            img = plt.imread(os.path.join(DENSE_PATH, file))
            breakpoint()
            label = np.zeros_like(img[:, :, 0])
            # breakpoint()
            # TODO After iris confirms the logic for label generation, implement here
            #it has to be a binary image
            # label[:,:] = 255

            # save mask
            mask_name = file.split('.')[0] + '.gif'
            plt.imsave(os.path.join(LABEL_PATH, mask_name), label, cmap='binary')
