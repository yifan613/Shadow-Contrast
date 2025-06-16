import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, RandomAffine, RandomHorizontalFlip, RandomCrop
from PIL import Image


def ImageTransform(loadSize):
    return {"train": Compose([
        # RandomCrop(loadSize, pad_if_needed=True, padding_mode='constant', fill=255),
        RandomAffine(10, fill=255),
        RandomHorizontalFlip(p=0.2),
        ToTensor(),
    ]), "test": Compose([
        ToTensor(),
    ]), "train_gt": Compose([
        # RandomCrop(loadSize, pad_if_needed=True, padding_mode='constant', fill=255),
        RandomAffine(10, fill=255),
        RandomHorizontalFlip(p=0.2),
        ToTensor(),
    ])}


class DocData(Dataset):
    def __init__(self, path_img, loadSize, mode=1):
        super().__init__()
        self.path_img = path_img
        self.data_img = os.listdir(path_img)
        self.mode = mode
        self.load_size = loadSize
        self.ImgTrans = ImageTransform(loadSize)["test"]

    def __len__(self):
        return len(self.data_img)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.path_img, self.data_img[idx]))
        img = img.convert('RGB').resize((self.load_size[0], self.load_size[1]))
        img= self.ImgTrans(img)
        name = self.data_img[idx]
        return img, name
