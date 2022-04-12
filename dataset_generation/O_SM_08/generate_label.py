import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from dense_flow import dense_flow
from PIL import Image

DENSE_PATH = './O_SM_08-GT/' #already labeled image
LABEL_PATH = './labels/'
pathlib.Path(LABEL_PATH).mkdir(parents=True, exist_ok=True)

maxval = 255
thresh = 10# IF PIXEL VALUE IS MORE THAN 30 MAKE IT WHITE ELSE BLACK

if __name__ == '__main__':
    # read all images in optical flow folder and then generate mask.
    # Mask will be an image of same size as the original image with all pixels set to 255.
    # mask will be just a 2D array (height, width)  i.e no color channels.
    # The mask will be saved in the mask folder.
    # mask format is .gif(following chengui's format)

    for i, file in enumerate(os.listdir(DENSE_PATH)):
        if file.endswith(".png"):
            file = 'GT_260.png'
            img = plt.imread(os.path.join(DENSE_PATH, file))
            # https://stackoverflow.com/questions/40449781/convert-image-np-array-to-binary-image

            im_gray = np.array(Image.open(os.path.join(DENSE_PATH, file)).convert('L'))

            im_bin = (im_gray > thresh) * maxval
            # plt.imshow(im_bin, cmap='gray')
            # plt.show()



            # save mask
            mask_name = file.split('.')[0] + '.gif'

            plt.imsave(os.path.join(LABEL_PATH, mask_name), im_bin, cmap='gray')
            # breakpoint()
            # Image.fromarray(np.uint8(im_bin)).save(os.path.join(LABEL_PATH, mask_name))

