"""
created by: Donghyeon Won
Modified by; Xiaofeng Lin
"""


##################
#This util is based on model of five variables: Mask, shelf, cart, paper, people
##################

import os
import numpy as np
import pandas as pd
from PIL import Image, ImageFile

from torch.utils.data import Dataset
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models


class MaskDataset(Dataset):
    """
    dataset for training and evaluation
    """
    def __init__(self, txt_file, img_dir, transform = None):
        """
        Args:
            txt_file: Path to txt file with annotation
            img_dir: Directory with images
            transform: Optional transform to be applied on a sample.
        """
        #Gives a Dataframe
        self.label_frame = pd.read_csv(txt_file, delimiter=",")
        #An image directory
        self.img_dir = img_dir
        #A transform function
        self.transform = transform
    def __len__(self):
        return len(self.label_frame)
    def __getitem__(self, idx):
        imgpath = os.path.join(self.img_dir,
                                self.label_frame.iloc[idx, 0])
        #Convert an image
        image = pil_loader(imgpath)
        
        mask = self.label_frame.iloc[idx, 1:2].values.astype('float')
        visattr = self.label_frame.iloc[idx, 2:].values.astype('float')
        label = {'mask':mask, 'visattr':visattr}
        
        sample = {"image":image, "label":label}
        if self.transform:
            sample["image"] = self.transform(sample["image"])
        return sample

class MaskDatasetEval(Dataset):
    """
    dataset for just calculating the output (does not need an annotation file)
    """
    def __init__(self, img_dir):
        """
        Args:
            img_dir: Directory with images
        """
        self.img_dir = img_dir
        self.transform = transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225]),
                                ])
        self.img_list = sorted(os.listdir(img_dir))
    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, idx):
        imgpath = os.path.join(self.img_dir,
                                self.img_list[idx])
        image = pil_loader(imgpath)
        # we need this variable to check if the image is Mask or not)
        sample = {"imgpath":imgpath, "image":image}
        sample["image"] = self.transform(sample["image"])
        return sample

class FinalLayer(nn.Module):
    """modified last layer for resnet50 for our dataset"""
    def __init__(self):
        super(FinalLayer, self).__init__()
        self.fc = nn.Linear(2048, 4) #Number of output variables!
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc(x)
        out = self.sigmoid(out)
        return out


def pil_loader(path):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
       	return img.convert('RGB')
def modified_resnet50():
    # load pretrained resnet50 with a modified last fully connected layer
    model = models.resnet50(pretrained = True)#This is a widely used pre-trained model imported from pytorch
    model.fc = FinalLayer()

    # uncomment following lines if you wnat to freeze early layers
    # i = 0
    # for child in model.children():
    #     i += 1
    #     if i < 4:
    #         for param in child.parameters():
    #             param.requires_grad = False


    return model

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count != 0:
            self.avg = self.sum / self.count

class Lighting(object):#Returns an image
    """
    Lighting noise(AlexNet - style PCA - based noise)
    https://github.com/zhanghang1989/PyTorch-Encoding/blob/master/experiments/recognition/dataset/minc.py
    """
    
    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))
