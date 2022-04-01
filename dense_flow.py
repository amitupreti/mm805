import numpy as np
from scipy import signal
import cv2
import matplotlib.pyplot as plt


def dense_flow(f1, f2):
    """

    :param f1: cv2 BGR image frame 1
    :param f2: cv2 BGR image frame 2
    :return: rgb dense flow image
    """
    mask = np.zeros_like(f1)
    # Sets image saturation to maximum
    mask[..., 1] = 255
    f1_gray = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    f2_gray = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(f1_gray, f2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # Sets image hue according to the optical flow direction
    mask[..., 0] = angle * 180 / np.pi / 2
    # Sets image value according to the optical flow magnitude (normalized)
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    # Converts HSV to RGB (BGR) color representation
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    # Opens a new window and displays the output frame
    # cv2.imshow("dense optical flow", rgb)
    return rgb


if __name__ == '__main__':
    f1 = cv2.imread('ezgif-frame-012.jpg')
    f2 = cv2.imread('ezgif-frame-013.jpg')

    res = dense_flow(f1, f2)
    print(res.shape)
    plt.imshow(res)
    plt.show()
    plt.imsave('dense_flow.png', res)
