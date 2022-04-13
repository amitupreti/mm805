
import argparse

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import imageio

import time

import torch.utils.data
import cv2

import random as rd

# print all the elements
np.set_printoptions(threshold=np.inf)

# global control variable for radius 
g_radiussizes = [0,3,6,9,12,15]
g_boxsizes = np.array( [ (2*r+1)**2   for r in g_radiussizes] )
g_boxsizes[0] = 0

# global variable for cude number 
g_cuda = "cuda:8" 

def loadFiles_plus(path_im, keyword = ""):
    re_fs = []
    re_fullfs = []

    files = os.listdir(path_im)
    files = sorted(files)

    for file in files:
        if file.find(keyword) != -1:
            re_fs.append(file)
            re_fullfs.append(path_im + "/" + file)

    return re_fs, re_fullfs


def img2patches(img, radius):
    re_data = []

    row, column, byte = img.shape

    for r in range(radius, row - radius):
        for c in range(radius, column - radius):
#            print("i, j", i, j)
            patches = img[r - radius:r + radius, c - radius:c + radius, :]

            re_data.append(patches)

            print("r, c = ", r, c)

#             showim = img - img + img
#             showim[r - radius:r + radius, c - radius:c + radius, 0] = 255
#             showim[r - radius:r + radius, c - radius:c + radius, 1] = 0
#             showim[r - radius:r + radius, c - radius:c + radius, 2] = 0
#
#
#             print("r, c = ", r, c)
#             plt.subplot(1, 2, 1)
#             plt.imshow(patches)
#             plt.subplot(1, 2, 2)
#             plt.imshow(showim)
#             plt.pause(0.01)

    return re_data


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 20, 10, 1)
#         self.conv2 = nn.Conv2d(20, 50, 10, 1)
#         self.fc1 = nn.Linear(8*8*50, 500)
#         self.fc2 = nn.Linear(500, 2)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = x.view(-1, 8*8*50)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)
#

'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()


        self.conv1 = nn.Conv2d(15, 10024, 7)
        self.fc1 = nn.Linear(10024, 2)

    def forward(self, x):
#        print("x.shape", x.shape)
        x = F.relu(self.conv1(x))
#        print("x.shape", x.shape)
        x = x.view(-1, 10024)
#        print("x.shape", x.shape)
        x = self.fc1(x)
#        print("x.shape", x.shape)

        return F.log_softmax(x, dim=1)
'''

class Multi_Net(nn.Module):
    def __init__(self):
        super(Multi_Net, self).__init__()
        # 1024 * 5 = 5120
        self.fc1 = nn.Sequential( nn.Linear(5120, 1920),
                                nn.Linear(1920, 2))

        # for size 7    end with 1*1
        self.conv1 = nn.Sequential( nn.Conv2d(3, 512, 1 + 2 * g_radiussizes[1]),
                                    nn.BatchNorm2d(512),
                                    nn.Conv2d(512, 1024, 1),
                                    nn.BatchNorm2d(1024),
                                    nn.ReLU(True) )

        # for size 13
        self.conv2 = nn.Sequential( nn.Conv2d(3, 512, 1 + 2 * g_radiussizes[2]),
                                    nn.BatchNorm2d(512),
                                    nn.Conv2d(512, 1024, 1),
                                    nn.BatchNorm2d(1024),
                                    nn.ReLU(True) )

        # for size 19
        self.conv3 = nn.Sequential( nn.Conv2d(3, 512, 1 + 2 * g_radiussizes[3]),
                                    nn.BatchNorm2d(512),
                                    nn.Conv2d(512, 1024, 1),
                                    nn.BatchNorm2d(1024),
                                    nn.ReLU(True) )

         # for size 25
        self.conv4 = nn.Sequential( nn.Conv2d(3, 512, 1 + 2 * g_radiussizes[4]),
                                    nn.BatchNorm2d(512),
                                    nn.Conv2d(512, 1024, 1),
                                    nn.BatchNorm2d(1024),
                                    nn.ReLU(True) )

         # for size 31
        self.conv5 = nn.Sequential( nn.Conv2d(3, 512, 1 + 2 * g_radiussizes[5]),
                                    nn.BatchNorm2d(512),
                                    nn.Conv2d(512, 1024, 1),
                                    nn.BatchNorm2d(1024),
                                    nn.ReLU(True) )

    def forward(self, x):
    #       print("x.shape 1", x.shape)

        # x 输入的是三个不同size的 patch 的组合, 分别是 9 * 9 * 3, 17 * 17 * 3 和 33 * 33 * 3
        x1 = x[:,:, np.arange(np.sum( g_boxsizes[:1]) , np.sum( g_boxsizes[:2]) )].reshape( x.size(0), x.size(1), 1 + 2*g_radiussizes[1], 1 + 2*g_radiussizes[1])
        x2 = x[:,:, np.arange(np.sum( g_boxsizes[:2]) , np.sum( g_boxsizes[:3]) )].reshape( x.size(0), x.size(1), 1 + 2*g_radiussizes[2], 1 + 2*g_radiussizes[2])
        x3 = x[:,:, np.arange(np.sum( g_boxsizes[:3]) , np.sum( g_boxsizes[:4]) )].reshape( x.size(0), x.size(1), 1 + 2*g_radiussizes[3], 1 + 2*g_radiussizes[3])
        x4 = x[:,:, np.arange(np.sum( g_boxsizes[:4]) , np.sum( g_boxsizes[:5]) )].reshape( x.size(0), x.size(1), 1 + 2*g_radiussizes[4], 1 + 2*g_radiussizes[4])
        x5 = x[:,:, np.arange(np.sum( g_boxsizes[:5]) , np.sum( g_boxsizes[:6]) )].reshape( x.size(0), x.size(1), 1 + 2*g_radiussizes[5], 1 + 2*g_radiussizes[5])

        # 三个 batchsize *1024* 1*1
        #print(x1.shape)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x4 = self.conv4(x4)
        x5 = self.conv5(x5)

        # 三个 batchsize *1024 
        x1 = x1.view (x1.size(0), -1) 
        x2 = x2.view (x2.size(0), -1) 
        x3 = x3.view (x3.size(0), -1) 
        x4 = x4.view (x4.size(0), -1) 
        x5 = x5.view (x5.size(0), -1) 

        # 用cat 连成  5   (batchsize * 1024)的vector
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        # 然后使用fully connected layer
        x = self.fc1(x)

        return F.log_softmax(x, dim=1)








class FakeDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        img, target = self.images[index], self.labels[index]

        return img, target

    def __len__(self):
        return len(self.images)



def img2patches_mask(img, lab, radius, msk):
    re_imgs = []
    re_labs = []

    row_im, column_im, byte_im = img.shape

    row_mk, column_mk = msk.shape
    row_lb, column_lb = lab.shape

    for r in range(radius, row_im - radius):
        for c in range(radius, column_im - radius):
            if msk[r, c] == 255 :
                patch_img = img[r - radius:r + radius + 1, c - radius:c + radius + 1, :]
                patch_msk = msk[r - radius:r + radius + 1, c - radius:c + radius + 1]
#                patch_labmsk = lab[r - radius:r + radius + 1, c - radius:c + radius + 1]
                patch_lab = lab[r, c]

                center_value = img[r, c, :]

                indices = patch_msk == 255
                img_vals = patch_img[indices, :]
                nums, byte = img_vals.shape
                np.random.shuffle(img_vals)

                img_vals = np.pad(img_vals, ((0,   (2*radius + 1)**2 - nums  ), (0,0)), 'symmetric')
                patch_fake = np.reshape(img_vals, ((2*radius + 1), (2*radius + 1), byte  ) )

                patch_sub = patch_fake - center_value

                re_imgs.append(patch_sub)
                re_labs.append(patch_lab)

        print("r = ", r)


    re_imgs = np.asarray(re_imgs)
    re_labs = np.asarray(re_labs)

    re_imgs = re_imgs.transpose(0, 3, 1, 2)
    re_imgs = re_imgs.dot(1.0/255.0)
    re_labs = re_labs.dot(1.0/255.0)
    re_labs.astype(int)

    return re_imgs, re_labs


def img2patches_plus(img, lab, radius, msk):
    re_imgs = []
    re_labs = []

    row_im, column_im, byte_im = img.shape

    row_mk, column_mk = msk.shape
    row_lb, column_lb = lab.shape

    for r in range(radius, row_im - radius):
        for c in range(radius, column_im - radius):
            if msk[r, c] == 255 :
                patch_img = img[r - radius:r + radius + 1, c - radius:c + radius + 1, :]
                patch_msk = msk[r - radius:r + radius + 1, c - radius:c + radius + 1]
#                patch_labmsk = lab[r - radius:r + radius + 1, c - radius:c + radius + 1]
                patch_lab = lab[r, c]

                center_value = img[r, c, :]

#                judge_idx = patch_msk == 0
#                flag = np.sum(judge_idx)
                flag = np.sum(patch_msk == 0)

                if flag != 0:
                    indices = patch_msk == 255
                    img_vals = patch_img[indices, :]
                    nums, byte = img_vals.shape
                    np.random.shuffle(img_vals)

                    img_vals = np.pad(img_vals, ((0,   (2*radius + 1)**2 - nums  ), (0,0)), 'symmetric')
                    patch_fake = np.reshape(img_vals, ((2*radius + 1), (2*radius + 1), byte  ) )

                else:
                    indices = patch_msk == 255
                    img_vals = patch_img[indices, :]


                    nums, byte = img_vals.shape
                    np.random.shuffle(img_vals)

                    patch_fake = np.reshape(img_vals, patch_img.shape )


                patch_sub = patch_fake - center_value

                re_imgs.append(patch_sub)
                re_labs.append(patch_lab)


    re_imgs = np.asarray(re_imgs)
    re_labs = np.asarray(re_labs)

    re_imgs = re_imgs.transpose(0, 3, 1, 2)
    re_imgs = re_imgs.dot(1.0/255.0)
    re_labs = re_labs.dot(1.0/255.0)
    re_labs.astype(int)

    return re_imgs, re_labs



def img2patches_multi(img, lab, msk, radius_in, radius_out):

    for i in range(len(radius_out)):
        print(i)




    return img



def random_pad(vals, tar_nums):

    nums, byte = vals.shape

    if tar_nums > nums:

        tar_idx = torch.randperm(tar_nums)
        tar_idx = tar_idx - tar_idx - 1

    #    print("tar_idx = ", tar_idx)

        idx = torch.randperm(nums)

    #    print("tar_idx = ", tar_idx.shape)
    #    print("idx = ", idx.shape)

        cnt = tar_nums // nums

    #    print("cnt = ", cnt)
        for i in range(cnt):
            tar_idx[i*nums:(i + 1)*nums] = idx


    #     print("(i + 1)*nums = ", (i + 1)*nums)
    #     print("tar_nums = ", tar_nums)
    #     print(" (tar_nums - (i + 1)*nums ) = ",  (tar_nums - (i + 1)*nums )  )
        tar_idx[(i + 1)*nums:tar_nums] = idx[0:(tar_nums - (i + 1)*nums )]

    #    print("tar_idx = ", tar_idx)

        re_vals = vals[tar_idx, :]
    else:
        re_vals = vals[0:nums]

    return re_vals

#    return tar_idx
# def img2patches_files(fs_im, fs_lb, fs_mk, radius):
#
#     im = imageio.imread(fs_im[0])
#     lb = imageio.imread(fs_lb[0])
#     mk = imageio.imread(fs_mk[0])
#     mk = mask_extend(mk, 5)
#
#     adv_im = np.pad(im, ((radius, radius), (radius, radius), (0,0)), 'edge')
#     adv_lb = np.pad(lb, ((radius, radius), (radius, radius)), 'edge')
#     adv_mk = np.pad(mk, ((radius, radius), (radius, radius)), 'edge')
#
#
#     re_imgs, re_labs = img2patches_fast(adv_im, adv_lb, radius, adv_mk)
#
#
#     for i in range(1, len(fs_im)):
#         im = imageio.imread(fs_im[i])
#         lb = imageio.imread(fs_lb[i])
#         mk = imageio.imread(fs_mk[i])
#         mk = mask_extend(mk, 5)
#
#         adv_im = np.pad(im, ((radius, radius), (radius, radius), (0,0)), 'edge')
#         adv_lb = np.pad(lb, ((radius, radius), (radius, radius)), 'edge')
#         adv_mk = np.pad(mk, ((radius, radius), (radius, radius)), 'edge')
#
#
#         imgs, labs = img2patches_fast(adv_im, adv_lb, radius, adv_mk)
#
#         re_imgs = np.concatenate((re_imgs, imgs), axis=0)
#         re_labs = np.concatenate((re_labs, labs), axis=0)
#
#     return re_imgs, re_labs


def img2patches_in_out(img, lab, msk, radius_in, radius_out):



    img = torch.tensor(img, dtype=torch.float)
    lab = torch.tensor(lab, dtype=torch.float)
    msk = torch.tensor(msk, dtype=torch.float)

    row_img_src, column_img_src, byte_img_src = img.shape

    img = F.pad(img, ((0, 0, radius_out, radius_out, radius_out, radius_out) ), 'constant')
    lab = F.pad(lab, ((radius_out, radius_out, radius_out, radius_out) ), 'constant')
    msk = F.pad(msk, ((radius_out, radius_out, radius_out, radius_out) ), 'constant')


    totallen = torch.sum(msk == 255)

    re_imgs = torch.zeros(totallen, 2*radius_in + 1, 2*radius_in + 1, byte_img_src)
    re_labs = torch.zeros(totallen)

    row_im, column_im, byte_im = img.shape

    count = 0

    for r in range(radius_out, row_im - radius_out):
        for c in range(radius_out, column_im - radius_out):
            if msk[r, c] == 255:
                patch_img = img[r - radius_out:r + radius_out + 1, c - radius_out:c + radius_out + 1, :]
                patch_msk = msk[r - radius_out:r + radius_out + 1, c - radius_out:c + radius_out + 1]
                patch_lab = lab[r, c]

                center_value = img[r, c, :]


                flag = torch.sum(patch_msk == 0)

                if flag != 0:
                    indices = patch_msk == 255
                    img_vals = patch_img[indices, :]

                    nums, byte = img_vals.shape

                    vals = random_pad(img_vals, (2*radius_out + 1)**2 )

                    patch_fake = vals.view((2*radius_out) + 1, (2*radius_out) + 1, byte_im)
                else:
                    patch_fake = patch_img


                patch_fake = patch_fake.reshape( (2*radius_out + 1)**2, byte_im   )


                indices = torch.randperm( (2*radius_out + 1)**2  )
                indices = indices[0:(2*radius_in + 1)**2]
                patch_fake = patch_fake[indices, :].view(2*radius_in + 1, 2*radius_in + 1, byte_im)


                patch_sub = patch_fake - center_value


                re_imgs[count, :, :, :] = patch_sub
                re_labs[count] = patch_lab

                count = count + 1

    re_imgs = re_imgs * (1.0/255.0)
    re_labs = re_labs * (1.0/255.0)

    re_imgs = re_imgs.transpose(1, 3) 

    return re_imgs, re_labs



def img2patches_fast(img, lab, radius, msk):
    re_imgs = []
    re_labs = []

    row_im, column_im, byte_im = img.shape

    row_mk, column_mk = msk.shape
    row_lb, column_lb = lab.shape

    for r in range(radius, row_im - radius):
        for c in range(radius, column_im - radius):
            if msk[r, c] == 255 :
                patch_img = img[r - radius:r + radius + 1, c - radius:c + radius + 1, :]
                patch_msk = msk[r - radius:r + radius + 1, c - radius:c + radius + 1]
#                patch_labmsk = lab[r - radius:r + radius + 1, c - radius:c + radius + 1]
                patch_lab = lab[r, c]

                center_value = img[r, c, :]

#                judge_idx = patch_msk == 0
#                flag = np.sum(judge_idx)
                flag = np.sum(patch_msk == 0)

                if flag != 0:
                    indices = patch_msk == 255
                    img_vals = patch_img[indices, :]
                    nums, byte = img_vals.shape
                    np.random.shuffle(img_vals)

                    img_vals = np.pad(img_vals, ((0,   (2*radius + 1)**2 - nums  ), (0,0)), 'symmetric')
                    patch_fake = np.reshape(img_vals, ((2*radius + 1), (2*radius + 1), byte  ) )

                else:

                    patch_fake = patch_img


                patch_sub = patch_fake - patch_fake
                for b in range(3):
#                    patch_sub[:, :, b] = patch_fake[:, :, b] / max(1.0, center_value[b] )
                    patch_sub[:, :, b] = patch_fake[:, :, b] - center_value[b]

#                print("patch_sub = ", patch_sub)
#                 patch_sub = patch_fake - center_value
#                 print("patch_sub = ", patch_sub)
                re_imgs.append(patch_sub)
                re_labs.append(patch_lab)

#        print("r = ", r)


    re_imgs = np.asarray(re_imgs)
    re_labs = np.asarray(re_labs)

    re_imgs = re_imgs.transpose(0, 3, 1, 2)
    re_imgs = re_imgs.dot(1.0/255.0)
    re_labs = re_labs.dot(1.0/255.0)
    re_labs.astype(int)

#    print("re_imgs = ", re_imgs )


    return re_imgs, re_labs



def img2patches_files_pytorch(fs_im, fs_lb, fs_mk, radius_in, radius_out):

    im = imageio.imread(fs_im[0])
    lb = imageio.imread(fs_lb[0])
    mk = imageio.imread(fs_mk[0])
    mk = mask_extend(mk, 5)


    re_imgs, re_labs = img2patches_in_out(im, lb, mk, radius_in, radius_out)

    for i in range(1, len(fs_im)):
        im = imageio.imread(fs_im[i])
        lb = imageio.imread(fs_lb[i])
        mk = imageio.imread(fs_mk[i])
        mk = mask_extend(mk, 5)

        imgs, labs = img2patches_in_out(im, lb, mk, radius_in, radius_out)

        re_imgs = torch.cat( (re_imgs, imgs), 0)
        re_labs = torch.cat( (re_labs, labs), 0)


    return re_imgs, re_labs


def img2patches_files_multi_pytorch(fs_im, fs_lb, fs_mk, radius_in, radius_out):


    print("loading files", fs_im[0])
    im = imageio.imread(fs_im[0])
    lb = imageio.imread(fs_lb[0])
    mk = imageio.imread(fs_mk[0])
    mk = mask_extend(mk, 5)

    # first radius size
    re_imgs, re_labs = img2patches_in_out(im, lb, mk, g_radiussizes[1], g_radiussizes[1]  )
    # flat for concatination
    re_imgs = re_imgs.reshape(re_imgs.size(0),re_imgs.size(1),-1)

    for r in range(2, len(g_radiussizes)):
        re_imgs_temp, re_labs_temp = img2patches_in_out(im, lb, mk, g_radiussizes[r], g_radiussizes[r]  )
        # flat for concatination
        re_imgs_temp = re_imgs_temp.reshape(re_imgs_temp.size(0),re_imgs_temp.size(1),-1)
        # concatinate
        re_imgs = torch.cat((re_imgs, re_imgs_temp), 2)

    #re_labs = re_labs5


    for i in range(1, len(fs_im)):
        print("loading files", fs_im[i])

        im = imageio.imread(fs_im[i])
        lb = imageio.imread(fs_lb[i])
        mk = imageio.imread(fs_mk[i])
        mk = mask_extend(mk, 5)

        # first radius size
        imgs, labs = img2patches_in_out(im, lb, mk, g_radiussizes[1], g_radiussizes[1]  )
        # flat for concatination
        imgs = imgs.reshape(imgs.size(0),imgs.size(1),-1)

        for r in range(2, len(g_radiussizes)):
            imgs_temp, labs_temp = img2patches_in_out(im, lb, mk, g_radiussizes[r], g_radiussizes[r]  )
            # flat for concatination
            imgs_temp = imgs_temp.reshape(imgs_temp.size(0),imgs_temp.size(1),-1)
            # concatinate
            imgs = torch.cat((imgs, imgs_temp), 2)


        re_imgs = torch.cat( (re_imgs, imgs), 0)
        re_labs = torch.cat( (re_labs, labs), 0)



    return re_imgs, re_labs




def img2patches_files(fs_im, fs_lb, fs_mk, radius):

    im = imageio.imread(fs_im[0])
    lb = imageio.imread(fs_lb[0])
    mk = imageio.imread(fs_mk[0])
    mk = mask_extend(mk, 5)

    adv_im = np.pad(im, ((radius, radius), (radius, radius), (0,0)), 'edge')
    adv_lb = np.pad(lb, ((radius, radius), (radius, radius)), 'edge')
    adv_mk = np.pad(mk, ((radius, radius), (radius, radius)), 'edge')


    re_imgs, re_labs = img2patches_fast(adv_im, adv_lb, radius, adv_mk)


    for i in range(1, len(fs_im)):
        im = imageio.imread(fs_im[i])
        lb = imageio.imread(fs_lb[i])
        mk = imageio.imread(fs_mk[i])
        mk = mask_extend(mk, 5)

        adv_im = np.pad(im, ((radius, radius), (radius, radius), (0,0)), 'edge')
        adv_lb = np.pad(lb, ((radius, radius), (radius, radius)), 'edge')
        adv_mk = np.pad(mk, ((radius, radius), (radius, radius)), 'edge')


        imgs, labs = img2patches_fast(adv_im, adv_lb, radius, adv_mk)

        re_imgs = np.concatenate((re_imgs, imgs), axis=0)
        re_labs = np.concatenate((re_labs, labs), axis=0)

    return re_imgs, re_labs


#def ConvertToBox(data, )



def train(args, model, device, train_loader, optimizer, epoch):
    print( " Taining ...", epoch)
    class_weights = torch.FloatTensor([0.4, 0.6]).to(device)
    loss_func = torch.nn.NLLLoss(weight=class_weights, reduction='sum').to(device)

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.int64)

#        print(data.shape)
        optimizer.zero_grad()
        output = model(data)
#        loss = F.nll_loss(output, target, (1, 9))
        loss = loss_func(output, target)
#        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
#        print("up     borderline ----------------------")
        tar_pos = torch.tensor([0])
        tar_neg = torch.tensor([0])

        pre_pos = torch.tensor([0])
        pre_neg = torch.tensor([0])

        for data, target in test_loader:
            data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.int64)

#             print("data.shape = ",data.shape)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
#            print("pred = ", pred)
            tar_pos += torch.sum(target == 1)
            tar_neg += torch.sum(target == 0)

            pre_pos += torch.sum(pred == 1)
            pre_neg += torch.sum(pred == 0)
#             print("target: pos-", torch.sum(target == 1), "   neg-", torch.sum(target == 0))
#             print("pred:   pos-", torch.sum(pred == 1), "   neg-", torch.sum(pred == 0))

#            print("pred = ", pred)
            correct += pred.eq(target.view_as(pred)).sum().item()

#        print("bottom borderline ----------------------")

    test_loss /= len(test_loader.dataset)


    print("")
    print("")
    print("")
    print("up     borderline -------------------------")
    print("tar:", "postive-", tar_pos, "  negative-",tar_neg)
    print("pre:", "postive-", pre_pos, "  negative-",pre_neg)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    print("bottom borderline -------------------------")


def detect(imgs, model,device, batch_size):
    model.eval()
#    test_loss = 0

    nums, byte, totalSizeVal = imgs.shape

    re_labs = np.zeros(nums)

    l_idx = 0

    for r_idx in range(0, nums, 1000):
        if r_idx == 0:
            l_idx = r_idx

        else:
            data = torch.tensor( imgs[l_idx:r_idx, :,  :] ).to(device, dtype=torch.float)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)

            re_labs[l_idx:r_idx] = pred.cpu().detach().squeeze()
            l_idx = r_idx


    r_idx = nums + 1

    data = torch.tensor( imgs[l_idx:r_idx, :, :] ).to(device, dtype=torch.float)
    output = model(data)
    pred = output.argmax(dim=1, keepdim=True)

    re_labs[l_idx:r_idx] = pred.cpu().detach().squeeze()



    return re_labs



def detect_tensor(imgs, model,device, batch_size):
    model.eval()

    nums, byte, totalSizeVal = imgs.shape

#    re_labs = np.zeros(nums)
    re_labs = torch.zeros(nums)

    l_idx = 0

    for r_idx in range(0, nums, 1000):
        if r_idx == 0:
            l_idx = r_idx

        else:
            data = imgs[l_idx:r_idx, :, :].to(device, dtype=torch.float)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)

            re_labs[l_idx:r_idx] = pred.squeeze()
#            re_labs[l_idx:r_idx] = pred.cpu().detach().squeeze()
            l_idx = r_idx


    r_idx = nums + 1

#    data = torch.tensor( imgs[l_idx:r_idx, :, :, :] ).to(device, dtype=torch.float)
    data = imgs[l_idx:r_idx, :, :].to(device, dtype=torch.float)

    output = model(data)
    pred = output.argmax(dim=1, keepdim=True)

    re_labs[l_idx:r_idx] = pred.squeeze()




    return re_labs





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






def randn_permutate_pytorch(data, device):

    data = data.transpose(0, 2 ,3, 1)

    nums, row, column, byte = data.shape



    with torch.no_grad():
#        torch.cuda.clear_memory_allocated()
#        data = torch.tensor(data).to(device)
        data = torch.tensor(data)

        for i in range(nums):
            block = data[i, :, :, :]
            block = block.view( ( row*column, byte) )
            indices = torch.randperm(row*column)
            block = block[indices, :]

            data[i, :, :, :] = block.view( (row, column, byte) )


    data = data.cpu().detach().numpy()
    data = data.transpose(0, 3, 1, 2)

    return data


def rand_permutate_plus(data, shuffledInd):

    data = data.transpose(1, 2)
    nums, totalsizeval, byte = data.shape

    for i in range(nums):
        block = data[i, :, :]


        block = block[shuffledInd, :]

        data[i, :, :] = block.view( (totalsizeval, byte) )


    data = data.transpose(1, 2)

    return data



def evaluation_labs_tensor(prelabs, trulabs):

    prelabs = prelabs.reshape(prelabs.numel())
    trulabs = trulabs.reshape(trulabs.numel())

    TP = 0
    TN = 0
    FP = 0
    FN = 0


    for i in range(prelabs.numel()):
        prolab = prelabs[i]
        trulab = trulabs[i]


        if prolab == 1 and trulab == 1:
            TP += 1

        if prolab == 1 and trulab == 0:
            FP += 1

        if prolab == 0 and trulab == 0:
            TN += 1

        if prolab == 0 and trulab == 1:
            FN += 1

    Re = TP/max((TP + FN), 1)
    Pr = TP/max((TP + FP), 1)

    Fm = (2*Pr*Re)/max((Pr + Re), 0.0001)


    return Re, Pr, Fm


def evaluation_entry_tensor(prelabs, trulabs):

    prelabs = prelabs.reshape(prelabs.numel())
    trulabs = trulabs.reshape(trulabs.numel())

    TP = 0
    TN = 0
    FP = 0
    FN = 0


    for i in range(prelabs.numel()):
        prolab = prelabs[i]
        trulab = trulabs[i]


        if prolab == 1 and trulab == 1:
            TP += 1

        if prolab == 1 and trulab == 0:
            FP += 1

        if prolab == 0 and trulab == 0:
            TN += 1

        if prolab == 0 and trulab == 1:
            FN += 1

    return TP, TN, FP, FN




def evaluation_labs(prelabs, trulabs):
    prelabs = np.reshape(prelabs, (np.size(prelabs), 1))
    trulabs = np.reshape(trulabs, (np.size(trulabs), 1))

#     TP = np.sum( (prelabs == 1) & (trulabs == 1) )
#     FP = np.sum( (prelabs == 1) & (trulabs == 0) )
#     TN = np.sum( (prelabs == 0) & (trulabs == 0) )
#     FN = np.sum( (prelabs == 0) & (trulabs == 0) )
#
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    count = np.size(prelabs)

    for i in range(count):
        prolab = int(prelabs[i])
        trulab = int(trulabs[i])

#        print("prolab:", prolab, "trulab:", trulab)

        if prolab == 1 and trulab == 1:
            TP += 1
#            print("TP")

        if prolab == 1 and trulab == 0:
            FP += 1
#            print("FP")

        if prolab == 0 and trulab == 0:
            TN += 1
#            print("TN")

        if prolab == 0 and trulab == 1:
            FN += 1
#            print("FN")





    Re = TP/max((TP + FN), 1)
    Pr = TP/max((TP + FP), 1)

    Fm = (2*Pr*Re)/max((Pr + Re), 1)


    return Re, Pr, Fm


def mask_extend(mask, radius):
    row, column = mask.shape

    re_mask = mask - mask
    for r in range(radius, row - radius):
        for c in range(radius, column - radius):
            if np.sum(mask[r - radius:r + radius + 1, c - radius:c + radius + 1] == 0) != 0:
                re_mask[r, c] = 0
            else:
                re_mask[r, c] = 255

    return re_mask



def main():

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for training (default: 1000)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=40, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    print("------------")
    print(use_cuda)
    print("------------")

    torch.manual_seed(args.seed)

    device = torch.device(g_cuda if use_cuda else "cpu")
    torch.cuda.set_device( g_cuda )

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    

    path_im = '../data/training/images'
    path_lb = '../data/training/1st_manual'
    path_mk = '../data/training/mask'


    net_pa = '../'

    model = Multi_Net().to(device)
#     optimizer = optim.Adam(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    
    fs_im, fullfs_im = loadFiles_plus(path_im)
    fs_lb, fullfs_lb = loadFiles_plus(path_lb)
    fs_mk, fullfs_mk = loadFiles_plus(path_mk)


#    fullfs_im = fullfs_im[13:18]
#    fullfs_lb = fullfs_lb[13:18]
#    fullfs_mk = fullfs_mk[13:18]


#     im = imageio.imread(fullfs_im[0])
#     lb = imageio.imread(fullfs_lb[0])
#     mk = imageio.imread(fullfs_mk[0])
#     mk = mask_extend(mk, 5)


    index = np.random.permutation(len(fullfs_im)).astype(int)

    fullfs_im = np.asarray(fullfs_im)[index]
    fullfs_lb = np.asarray(fullfs_lb)[index]
    fullfs_mk = np.asarray(fullfs_mk)[index]

    fullfs_im = fullfs_im[0:20]
    fullfs_lb = fullfs_lb[0:20]
    fullfs_mk = fullfs_mk[0:20]



    radius_in  = 5
    radius_out = 5
#    img2patches_files_multi_pytorch(fs_im, fs_lb, fs_mk, radius_in, radius_out)



    print('generating training data ...')
    starttime = time.time()
    with torch.no_grad():
        imgs, labs = img2patches_files_multi_pytorch(fullfs_im, fullfs_lb, fullfs_mk, radius_in, radius_out)

#        imgs, labs = img2patches_in_out(im, lb, mk, radius_in, radius_out)

        nums_im, byte_im, totalSize= imgs.shape
        indices = torch.randperm(nums_im)

        imgs = imgs[indices, :, :]
        labs = labs[indices]
    endtime = time.time()

    print('genarating training data completed, total time:', endtime - starttime, "  data size:", imgs.shape)
    print(" Save training data for future use.")
    torch.save(imgs, 'imgTensor.pt')
    torch.save(labs, 'labTensor.pt')
    
    # print(" Load training data for training.")
    # imgs = torch.load('imgTensor.pt')
    # labs = torch.load( 'labTensor.pt')



#    imgs = rand_permutate_plus(imgs)

    print("start training")

    shuffledInd = np.arange(np.sum( g_boxsizes[:1]) , np.sum( g_boxsizes[:2]) )
    np.random.shuffle(shuffledInd)

    for i in range(2,len(g_boxsizes)): 
        ind = np.arange(np.sum( g_boxsizes[:i]) , np.sum( g_boxsizes[:i+1]) )
        np.random.shuffle(ind)
        shuffledInd = np.concatenate( (shuffledInd, ind), axis = 0)

    for epoch in range(1, args.epochs + 1):
        print("re-permuting training data ...")


        starttime = time.time()

        with torch.no_grad():
            imgs = rand_permutate_plus(imgs, shuffledInd)

            print("imgs.shape = ", imgs.shape)
#        imgs = randn_permutate(imgs)

        traindata = FakeDataset(imgs, labs)

        train_loader = torch.utils.data.DataLoader(traindata,
            batch_size=args.batch_size, shuffle=True, **kwargs)


        endtime = time.time()
        print("re-permuting training data completed, total time:", endtime - starttime, "  data size:", imgs.shape)


        train(args, model, device, train_loader, optimizer, epoch)
        test( args, model, device, train_loader)

        pre_labs = detect_tensor(imgs, model, device, 2000)
        Re, Pr, Fm = evaluation_labs_tensor(pre_labs, labs)

        print("")
        print("")
        print("borderline *************************************")
        print("")

        print("Re = ", Re)
        print("Pr = ", Pr)
        print("Fm = ", Fm)

        print("")
        print("borderline *************************************")
        print("")
        print("")


        if epoch % 5 == 0:
            name = net_pa + "network_dis_" + str(epoch).zfill(4) + ".pt"
            torch.save(model.state_dict(), name)
            print("\n\n save model competed \n\n")
    





    print("starting evaluation ---------------------------------------------------------------")
    path_im = '../data/test/images'
    path_lb = '../data/test/1st_manual'
    path_mk = '../data/test/mask'


    net_pa = '../'


    fs_im, fullfs_im = loadFiles_plus(path_im)
    fs_lb, fullfs_lb = loadFiles_plus(path_lb)
    fs_mk, fullfs_mk = loadFiles_plus(path_mk)




#     im = imageio.imread(fullfs_im[0])
#     lb = imageio.imread(fullfs_lb[0])
#     mk = imageio.imread(fullfs_mk[0])
#     mk = mask_extend(mk, 5)


    radius_in  = 5
    radius_out = 5
#    img2patches_files_multi_pytorch(fs_im, fs_lb, fs_mk, radius_in, radius_out)


    model = Multi_Net().to(device)
    print("Loading trained model...")
    model.load_state_dict(torch.load('../network_dis_0040.pt'))

    TP_sum = 0
    TN_sum = 0
    FP_sum = 0
    FN_sum = 0

    Re_sum = 0
    Pr_sum = 0
    Fm_sum = 0

    for i in range(len(fullfs_im)):
        im = imageio.imread(fullfs_im[i])
        lb = imageio.imread(fullfs_lb[i])
        mk = imageio.imread(fullfs_mk[i])
        mk = mask_extend(mk, 5)

        print('Loadinging test image', i+1)
        with torch.no_grad():
            # first radius size
            imgs, labs = img2patches_in_out(im, lb, mk, g_radiussizes[1], g_radiussizes[1]  )
            # flat for concatination
            imgs = imgs.reshape(imgs.size(0),imgs.size(1),-1)

            for r in range(2, len(g_radiussizes)):
                imgs_temp, labs_temp = img2patches_in_out(im, lb, mk, g_radiussizes[r], g_radiussizes[r]  )
                # flat for concatination
                imgs_temp = imgs_temp.reshape(imgs_temp.size(0),imgs_temp.size(1),-1)
                # concatinate
                imgs = torch.cat((imgs, imgs_temp), 2)

        pre_labs = detect_tensor(imgs, model, device, 2000)
        TP, TN, FP, FN = evaluation_entry_tensor(pre_labs, labs)

        TP_sum += TP
        TN_sum += TN
        FP_sum += FP
        FN_sum += FN


        Re = TP/max((TP + FN), 1)
        Pr = TP/max((TP + FP), 1)

        Fm = (2*Pr*Re)/max((Pr + Re), 0.0001)

        Re_sum = Re_sum + Re
        Pr_sum = Pr_sum + Pr
        Fm_sum = Fm_sum + Fm


        print("")
        print("")
        print("borderline *************************************")
        print("")

        print("Re = ", Re)
        print("Pr = ", Pr)
        print("Fm = ", Fm)

        print("")
        print("borderline *************************************")
        print("")
        print("")


    Re = TP_sum/max((TP_sum + FN_sum), 1)
    Pr = TP_sum/max((TP_sum + FP_sum), 1)

    Fm = (2*Pr*Re)/max((Pr + Re), 0.0001)

    print("\n\n\n\n\n")
    print("final result: borderline -------------------------------")
    print("Re = ", Re)
    print("Pr = ", Pr)
    print("Fm = ", Fm)
    print("final result: borderline -------------------------------")

    print("avg: Re, Pr, Fm: ", Re_sum/20, " ", Pr_sum/20, " ", Fm_sum/20)





if __name__=='__main__':
    main()
