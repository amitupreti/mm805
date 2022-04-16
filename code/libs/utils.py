import numpy as np
from scipy import signal
import cv2
import matplotlib.pyplot as plt


def randn_permutate(data):
    data = data.transpose(0, 2 ,3, 1)

    nums, row, column, byte = data.shape

    for i in range(nums):
        block = data[i, :, :, :]
        block = np.reshape(block, (row*column, byte))
        np.random.shuffle(block)

        data[i, :, :, :] = np.reshape(block, (row, column, byte) )


    data = data.transpose(0, 3, 1, 2)

    return data

# def randn_permutate_pytorch(data, device):
#     data = data.transpose(0, 2 ,3, 1)
#     nums, row, column, byte = data.shape
#
#     with torch.no_grad():
#         data = torch.tensor(data).to(device)
#         for i in range(nums):
#             block = data[i, :, :, :]
#             block = block.view( ( row*column, byte) )
#             indices = torch.randperm(row*column)
#             block = block[indices, :]
#
#             data[i, :, :, :] = block.view( (row, column, byte) )
#
#
#     data = data.cpu().detach().numpy()
#     data = data.transpose(0, 3, 1, 2)
#
#     return data


