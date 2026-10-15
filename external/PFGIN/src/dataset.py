import json
import os
import random

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from skimage.color import gray2rgb

from torch.utils.data import DataLoader
from skimage import io
import scipy

class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, flist, mask_flist, augment=True, training=True):
        super(Dataset, self).__init__()
        self.augment = augment
        self.training = training
        self.data = self.load_flist(flist)
        self.mask_data = self.load_flist(mask_flist)

        self.input_size = config.INPUT_SIZE
        self.sigma = config.SIGMA
        self.mask = config.MASK
        self.nms = config.NMSMASK_REVERSE

        self.reverse_mask = config.MASK_REVERSE
        self.mask_threshold = config.MASK_THRESHOLD

        print('training:{}  mask:{}  mask_list:{}  data_list:{}'.format(training, self.mask, mask_flist, flist))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)

        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):

        size = self.input_size

        # load image
        img = Image.open(self.data[index])
        img = np.array(img)

        # gray to rgb
        if len(img.shape) < 3:
            img = gray2rgb(img)

        # resize/crop if needed
        if size != 0:
            img = self.resize(img, size, size)

        # load mask
        mask = self.load_mask(img, index % len(self.mask_data))

        if self.reverse_mask == 1:
            mask = 255 - mask


        # augment data
        if self.augment and np.random.binomial(1, 0.5) > 0:
            img = img[:, ::-1, ...]
            mask = mask[:, ::-1, ...]

        return self.to_tensor(img), self.to_tensor(mask)


    def load_mask(self, img, index):
        imgh, imgw = img.shape[0:2]

        if self.training:
            mask_index = random.randint(0, len(self.mask_data) - 1)
        else:
            mask_index = index

        mask = Image.open(self.mask_data[mask_index])
        mask = np.array(mask)
        mask = self.resize(mask, imgh, imgw)
        mask = (mask > self.mask_threshold).astype(np.uint8) * 255       # threshold due to interpolation


        return mask

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        img = cv2.resize(img, dsize=(height, width))

        return img

    def load_flist(self, flist):
        if flist is None:
            return []
        with open(flist, 'r') as j:
            f_list = json.load(j)
            return f_list


    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item

class ImgDataset (Dataset):
    def __init__(self, x=None, y=None, transform=None):
        self.x = x
        self.y = y

        self.transform = transform

    def __len__(self):
        return len (self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x, y


def junk_band_remover(img_with_junk_band0):
    h,w,c = img_with_junk_band0.shape
    band1 = np.ones((h,w,c))
    band1[:,:,0] = img_with_junk_band0[:,:,0]
    j=0
    for i in range(c):
        if np.max(img_with_junk_band0[:,:,i])!=0:

            band1[:,:,j]=img_with_junk_band0[:, :, i]
            j=j+1
    return band1[:,:,:j]


def Nor(data):
    data = data.astype('float32')
    shape = data.shape
    shape = np.array(shape)
    x = data
    if shape.shape[0] == 2:
        x = (x-np.min(x))/(np.max(x)-np.min(x))
    else:
        for i in range(shape[2]):
            a = x[:,:,i]
            if np.max(a)!=0:
                    x[:,:,i] = (x[:,:,i]-np.min(a))/(np.max(a)-np.min(a))
    return x


def arr_to_tensor(data1):
    # print(np.max(data1))
    data = Nor(data1)
    data = np.transpose(data,[2,0,1])
    data = torch.tensor(data)
    data = data.float()
    return data


def clipdc(step, data1, data2):
    """
    This funciton clip the input two images into two list which have small patches in
    :param step1: step size of first dim
    :param step2: step size of second dim
    :param band_num: step size of band dim
    :param data1: first data to be  clipped (noisy image)
    :param data2: second  data to be  clipped (noise free image defult)
    :return:
    dataset_noisy: list of patches for data1
    dataset_pure: list of patches for data2
    mpsnr: the mean of the psnr of the two patch list
    """
    dataset_noisy = []
    dataset_pure = []

    band_num = step[2]
    step1=step[0]
    step2 = step[1]
    shape_data = data1.shape
    spatial_step1 = np.arange(0,shape_data[0],step1)
    spatial_step1[-1] =shape_data[0]-step1
    spatial_step2 = np.arange(0,shape_data[1],step1)
    spatial_step2[-1] = shape_data[1]-step2
    step_len1 = spatial_step1.shape[0]
    step_len2 = spatial_step2.shape[0]
    if band_num == data1.shape[2]:
        for i in range(step_len1):
            for j in range(step_len2):
                noisy = data1[spatial_step1[i]:spatial_step1[i]+ step1, spatial_step2[j]:spatial_step2[j]+step2,:]
                pure= data2[spatial_step1[i]:spatial_step1[i]+ step1, spatial_step2[j]:spatial_step2[j]+step2,:]
                noisy2 = noisy.copy ( )
                # noisy2 = np.concatenate ((noisy, noisy2, noisy), axis=2)
                dataset_noisy.append (arr_to_tensor(noisy2))
                dataset_pure.append (arr_to_tensor(pure))

    else:
        band_step=  np.arange(0,shape_data[2],band_num)
        band_step[-1] = shape_data[2]-band_num
        for i in range(step_len1):
            for j in range(step_len2):
                for z in range(band_step.shape[0]):
                    # print(spatial_step1[i],spatial_step1[i]+ step1, spatial_step2[j],spatial_step2[j]+step2,
                    #         band_step[z],band_step[z]+band_num)
                    noisy = data1[ spatial_step1[i]:spatial_step1[i]+ step1, spatial_step2[j]:spatial_step2[j]+step2,
                            band_step[z]:band_step[z]+band_num]
                    mask= data2[ spatial_step1[i]:spatial_step1[i]+ step1, spatial_step2[j]:spatial_step2[j]+step2,
                          band_step[z]:band_step[z]+band_num]

                    # noisy2 = np.concatenate((noisy,noisy,noisy),axis=2)
                    noisy2 = noisy.copy ( )
                    dataset_noisy.append (arr_to_tensor(noisy2))
                    dataset_pure.append (arr_to_tensor(mask))

    return dataset_noisy, dataset_pure,
def array_to_uint8(data):
    x = Nor(data)
    x = x*255
    x  =x.astype('uint8')
    return x
def deadlines(img, line_num=1, line_pattern=1, line_wide_range=10, rand=False):
    """
    add deadline sparse noise to hsi data
    :param img: h*w uint16 hsi data,
    :param line_num: int; number of lines
    :param line_pattern: 2 is vertical,1 is horizontal
    :param noise_pattern: deadline or brightline
    :param line_wide_range: width of added deadlines, int, 1*1
    :return: deadlines influenced image and noise mask
    """

    if rand:
        linde_wid = np.random.randint (1, line_wide_range, line_num, dtype=int)
    else:
        linde_wid = np.random.randint (1, line_wide_range, line_num, dtype=int)
        linde_wid[:] = line_wide_range



    if line_num == 0:
        img = img
    else:
        size_img = img.shape
        noise_mask = np.zeros ([size_img[0], size_img[1]])

        index_poss = np.random.randint (1, size_img[0] - np.max (linde_wid), line_num, dtype=int)
        index_pos = list(index_poss)
    if line_pattern == 1:

        for i in range (line_num):
            index_dead = index_pos[i]
            img[index_dead:index_dead + linde_wid[i] + 1, :] = 0
            noise_mask[index_dead:index_dead + linde_wid[i] + 1, :] = 1
    elif line_pattern == 2:
        for i in range (line_num):
            index_dead = index_pos[i]
            img[:, index_dead:index_dead + linde_wid[i] + 1] = 0
            noise_mask[:, index_dead:index_dead + linde_wid[i] + 1] = 1
    return img, noise_mask


def dead(img, line_num_dead=1, line_num_undead=0, pattern_dead=1, pattern_undead=1, line_wide_range=10,rand=False):
    """
    :param img:  h*w uint16 hsi data,
    :param line_num_dead: num of deadlines in the image; int 1*1
    :param line_num_undead: num of bright lines in the image;int 1*1
    :param pattern_dead: Horizontal or vertical of deadlines; 2 is vertical,1 is horizontal
    :param pattern_undead:Horizontal or vertical of Bright Lines;2 is vertical,1 is horizontal
    :param line_wide_range: width of added deadlines, int, 1*1
    :return: deadlines influenced image and noise mask data range does not  change, while data type is float32
    """
    if line_num_dead != 0:
        noisy1, mask1 = deadlines (img,line_num= line_num_dead, line_pattern=pattern_dead, line_wide_range=line_wide_range, rand=rand)
        if line_num_undead != 0:

            noisy2, mask2 = deadlines (img,line_num= line_num_undead, line_pattern=pattern_dead, line_wide_range=line_wide_range, rand=rand)
        else:
            noisy2 = noisy1
            mask2 = mask1
        noisy = noisy2
        mask = mask1 + mask2
        mask = mask
    elif line_num_undead != 0:
        noisy2, mask2 = deadlines (img, line_num_undead, pattern_undead, 1, line_wide_range,rand)
        noisy = noisy2
        mask = mask2
    else:
        noisy = img
        mask = img * 0
    return noisy, mask


def dead_all(img,size_img,rand=False,line_num=1,wideth=1,pattern=1):
    s1 = size_img[0]
    if rand:
        linde_wid = np.random.randint (1, wideth, line_num, dtype=int)
    else:
        linde_wid = np.random.randint (1, wideth, line_num, dtype=int)
        linde_wid[:] = wideth
    index = np.random.randint (1, s1 - np.max (wideth, line_num, dtype=int))
    mask = img.copy()
    mask = mask * 0
    img_noise = img.copy ( )

    for i in range(line_num):
        if pattern==1:
            img_noise = img_noise[:,index[i]:index[i]+linde_wid[i],:]
            mask[:,index[i]:index[i]+linde_wid[i],:]=1
        else:
            img_noise = img_noise[index[i]:index[i] + linde_wid[i], :,:]
            mask[index[i]:index[i] + linde_wid[i], :,:] = 1

    return img_noise, mask


def mask_gen(img, deadlin_num=1, line_wide=10, line_pattern=1,rand = True,all=False):
    """
    img: iuput image size:h*w*B
    para deadlin_num :numbers of deadlines which shall be inpaintinged

    """
    img2 = img.copy()
    size_img = img.shape

    img = Nor(img)
    mask = img.copy()
    mask = mask * 0

    # noisy = np.zeros([size_img[0],size_img[1],size_img[2]])
    img_noise = img.copy ( )
    if all is True:
        img_noise,mask = dead_all(img,size_img,line_num=deadlin_num,wideth=line_wide,pattern=line_pattern,rand=rand)
    else:
        for i in range (size_img[2]):
            img_band = img[:, :, i]

            img_dead, deal_mask = dead (img_band, line_num_dead=deadlin_num, line_wide_range=line_wide,
                                        pattern_dead=line_pattern, rand=rand)
            mask[:, :, i] = deal_mask
            noisy_band = img_dead
            img_noise[:, :, i] = noisy_band
    img2 = Nor(img2)
    return img2, img_noise, mask


def loaddata(path='D:\\hsi_data\\HSI_dehazing_dataset\\GF5', batch_size =16,line_wide=30, line_num=10,step_size = [256,256,32],config = None):
    """
    :param path:  path of data
    :param batch_size:
    :param par: par of noise adding five
    : 1:gau_noise_sigma;
    2:stipe lin number
    3:stripe sigjma
    4:number of deadlines
    5:poisson niose para
    :return:
    """
    batch_size = config.batch_size
    line_num  = config.line_num
    step_size = config.step_size
    line_wide = config.line_wide
    path = config.Data_path
    data_dir = sorted(os.listdir(path+'\\tr_data'))
    val_dir = sorted (os.listdir (path + '\\val_data'))
    tr_img = []
    tr_mask = []
    val_img = []
    val_mask = []


    print('Starting load training data')
    for file in data_dir:
        data = io.imread(path+'\\tr_data\\'+file)
        data = np.transpose (data, [1, 2, 0])
        data =  junk_band_remover((data))
        data = data[:,:,:]
        #  generate mask and create missing area with given line numbers: line_num and wide: line_wide
        img_gt,noisy_t, mask_t = mask_gen(data,deadlin_num=line_num,line_wide=line_wide,rand=True)
        #  clip the masks and GT image into patch with given step size in three dimensions,
        train_GT, train_noise_img= clipdc (step=step_size,data1=img_gt,data2= mask_t)

        tr_img = tr_img+train_GT
        tr_mask = tr_mask+train_noise_img
    # x = scipy.io.loadmat ('E:\\hsi_data\\KSC.mat')
    # img = x['KSC']
    # Dc = junk_band_remover (img)
    # Dc = Dc[:, :512, :]
    # img = np.abs (Dc)
    # img_gt, noisy_t, mask_t = mask_gen(img,deadlin_num=line_num,line_wide=line_wide,rand=True)
    # train_GT, train_noise_img= clipdc (step=[step_size, step_size,bands],data1=img_gt, data2=mask_t)
    # tr_img = tr_img + train_GT
    # tr_mask = tr_mask + train_noise_img
    # Dc = io.imread ('E:\hsi_data\DC\DC.tif')
    # Dc = np.transpose (Dc, [1, 2, 0])
    # Dc = junk_band_remover (Dc)
    # Dc = Dc[:, :, :191]
    # img = np.abs (Dc)
    # img_gt, noisy_t, mask_t = mask_gen(img,deadlin_num=line_num,line_wide=line_wide,rand=True)
    # train_GT, train_noise_img= clipdc (step=[step_size, step_size,bands],data1=img_gt, data2=mask_t)
    # tr_img = tr_img + train_GT
    # tr_mask = tr_mask + train_noise_img

    print( 'number of training data is %04d ' %(len(tr_img)))
    train_dataset = ImgDataset (tr_img, tr_mask)
    train_loader = DataLoader (train_dataset, batch_size=batch_size,shuffle=True)
    print('training data load finished, load validation data next')
    for file in val_dir:
        data = io.imread (path + '\\val_data\\' + file)
        data = np.transpose (data, [1, 2, 0])
        data =  junk_band_remover((data))
        data = data[:,:,5:300]
        img_gt,noisy_t, mask_t = mask_gen(data,deadlin_num=line_num,line_wide=line_wide,rand=True)
        train_GT, train_noise_img= clipdc (step=step_size,data1=img_gt, data2=mask_t)
        val_img = val_img+train_GT
        val_mask = val_mask+train_noise_img
    print( 'number of validation data is %04d ' %(len(val_mask)))
    val_dataset = ImgDataset (val_img,val_mask)

    val_dataloader = DataLoader (val_dataset , batch_size=batch_size, shuffle=True)
    print ('Data load finished')
    return train_loader, val_dataloader



if __name__ == "__main__":


    print(np.random.randint (1, 11, 1, dtype=int))
    path_t= 'D:\\hsi_data\\HSI_dehazing_dataset\\GF5\\test\\'
    path_1 = 'D:\\hsi_data\\HSI_dehazing_dataset\\GF5'
    path_2 = 'D:\\hsi_data\\HSI_dehazing_dataset\\AVIRIS'
    path = path_t
    # data_dir = sorted(os.listdir(path+'\\tr_data'))
    # img = io.imread(path+'\\tr_data\\'+data_dir[0])
    # data = np.transpose (img, [2, 1, 0])
    # data = junk_band_remover ((data))
    # data = Nor(data)
    # data,data_noise,mask = mask_gen(data,deadlin_num=1,line_wide=10)
    # img = data[:,:,10:13]
    #
    # # img1 = cv2.bitwise_and(img,img,mask=mask)
    # cv2.imshow('1',img)
    # cv2.imshow('2',mask[:,:,10:13])
    #
    # cv2.waitKey(0)
    a = loaddata(path)

