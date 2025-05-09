import os
import json
import random
import glob
from torch.utils.data import Dataset
import cv2
from PIL import Image, ImageDraw
import numpy as np
import math
import torch, torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.utils import save_image


class FaceRetouchingDataset_video_mask(Dataset):
    def __init__(self, path, resolution=700, data_type="train", get_mask=False, data_percentage=1, frame_num=6):
        self.resolution = resolution
        self.data_type = data_type
        self.frame_num = frame_num
        self.imgs = glob.glob(os.path.join(path, data_type, 'source', '*.*'))
        self.imgs_r = glob.glob(os.path.join(path, data_type, 'target', '*.*'))
        self.masks = glob.glob(os.path.join(path, data_type, 'mask', '*.*'))
        self.imgs = sorted(self.imgs, key=lambda x: os.path.basename(x))
        self.imgs_r = sorted(self.imgs_r, key=lambda x: os.path.basename(x))
        self.masks = sorted(self.masks, key=lambda x: os.path.basename(x))
        self.toTensor = transforms.Compose([  # 先resize成700*700，再随机裁剪为512*512，模拟视频中人物的变换
            transforms.Resize(self.resolution),
            transforms.ToTensor()
        ])
        self.Normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.toTensor_512_Normalize = transforms.Compose([  # 直接resize为512*512，模拟镜头的拉伸
            transforms.Resize(512),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.toTensor_512 = transforms.Compose([  # 直接resize为512*512，模拟镜头的拉伸
            transforms.Resize(512),
            transforms.ToTensor()
        ])
        assert len(self.imgs) == len(self.imgs_r), "Can not match the FFHQ and FFHQR!"
        for p, p_r in zip(self.imgs, self.imgs_r):
            assert os.path.basename(p) == os.path.basename(p_r), "Can not match the FFHQ and FFHQR!"
        self.length = len(self.imgs)

        if data_type == 'train':
            self.length = int(self.length * data_percentage)
            self.imgs = self.imgs[:self.length]
            self.imgs_r = self.imgs_r[:self.length]
        print(f"Data number: {len(self.imgs)}")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert("RGB")
        img_r = Image.open(self.imgs_r[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("L")
        img, img_r, mask = self.toTensor(img), self.toTensor(img_r), self.toTensor(mask)
        img, img_r = self.Normalize(img), self.Normalize(img_r)
        img_list = []
        img_list_r = []
        mask_list = []
        for i in range(self.frame_num - 1):
            position = [random.randint(0, self.resolution - 512), random.randint(0, self.resolution - 512)]
            sub_img = TF.crop(img, position[0], position[1], 512, 512)
            sub_img_r = TF.crop(img_r, position[0], position[1], 512, 512)
            sub_mask = TF.crop(mask, position[0], position[1], 512, 512)
            img_list.append(sub_img)
            img_list_r.append(sub_img_r)
            mask_list.append(sub_mask)

        img_list.append(self.toTensor_512_Normalize(Image.open(self.imgs[index]).convert("RGB")))
        img_list_r.append(self.toTensor_512_Normalize(Image.open(self.imgs_r[index]).convert("RGB")))
        mask_list.append(self.toTensor_512(Image.open(self.masks[index]).convert("L")))
        if self.data_type == 'train':
            flip = random.random()
            if flip < 0.5:
                for i in range(self.frame_num):
                    img_list[i] = TF.hflip(img_list[i])
                    img_list_r[i] = TF.hflip(img_list_r[i])
                    mask_list[i] = TF.hflip(mask_list[i])

        return (os.path.basename(self.imgs[index])[:-4], img_list, img_list_r, mask_list)


class FaceRetouchingDataset_video_new(Dataset):
    def __init__(self, path, resolution=700, data_type="train", get_mask=False, data_percentage=1, frame_num=8):
        self.resolution = resolution
        self.data_type = data_type
        self.frame_num = frame_num
        self.imgs = glob.glob(os.path.join(path, data_type, 'source', '*.*'))
        self.imgs_r = glob.glob(os.path.join(path, data_type, 'target', '*.*'))
        self.imgs = sorted(self.imgs, key=lambda x: os.path.basename(x))
        self.imgs_r = sorted(self.imgs_r, key=lambda x: os.path.basename(x))
        self.toTensor = transforms.Compose([       # 先resize成700*700，再随机裁剪为512*512，模拟视频中人物的变换
            transforms.Resize(self.resolution),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.toTensor_512 = transforms.Compose([   # 直接resize为512*512，模拟镜头的拉伸
            transforms.Resize(512),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        assert len(self.imgs) == len(self.imgs_r), "Can not match the FFHQ and FFHQR!"
        for p, p_r in zip(self.imgs, self.imgs_r):
            assert os.path.basename(p) == os.path.basename(p_r), "Can not match the FFHQ and FFHQR!"
        self.length = len(self.imgs)
        
        if data_type == 'train':
            self.length = int(self.length*data_percentage)
            self.imgs = self.imgs[:self.length]
            self.imgs_r = self.imgs_r[:self.length]
        print(f"Data number: {len(self.imgs)}")
        print(self.frame_num)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert("RGB")
        img_r = Image.open(self.imgs_r[index]).convert("RGB")
        img, img_r = self.toTensor(img), self.toTensor(img_r) 
        img_list = []
        img_list_r = []
        for i in range(self.frame_num-1):
            position = [random.randint(0, self.resolution-512), random.randint(0, self.resolution-512)]
            sub_img = TF.crop(img, position[0], position[1], 512, 512)
            sub_img_r = TF.crop(img_r, position[0], position[1], 512, 512)
            img_list.append(sub_img)
            img_list_r.append(sub_img_r)

        img_list.append(self.toTensor_512(Image.open(self.imgs[index]).convert("RGB")))
        img_list_r.append(self.toTensor_512(Image.open(self.imgs_r[index]).convert("RGB")))        
        if self.data_type == 'train':
            for i in range(self.frame_num):
                flip = random.random()
                if flip < 0.5: 
                    img_list[i] = TF.hflip(img_list[i])
                    img_list_r[i] = TF.hflip(img_list_r[i])
        
        return (os.path.basename(self.imgs[index])[:-4], img_list, img_list_r)


class VideoDataset_test(Dataset):
    def __init__(self, path, resolution=512, frame_num=6):
        self.resolution = resolution
        self.imgs = glob.glob(os.path.join(path, '*.*'))
        self.imgs = sorted(self.imgs, key=lambda x: os.path.basename(x))
        self.length = len(self.imgs)
        print(f"Data number: {len(self.imgs)}")
        self.frame_num = frame_num

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        normalize = transforms.Compose([transforms.ToTensor(), transforms.Resize(self.resolution), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        img_list = []
        for i in range(self.frame_num):
            img = Image.open(self.imgs[(index-(self.frame_num-1)+i)%self.length]).convert("RGB")
            h, w = img.size
            crop = transforms.CenterCrop(size=(min(h, w), min(h, w)))
            img = crop(img)
            img = normalize(img)
            img_list.append(img)
        return (os.path.basename(self.imgs[index])[:-4], img_list)
