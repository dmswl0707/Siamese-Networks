import torch
import torchvision
import torchvision.utils
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt
import random
from PIL import Image
import PIL.ImageOps
import numpy as np


train_dir = '/home/choieunji/다운로드/Facial-Similarity-with-Siamese-Networks-in-Pytorch-master/data/faces/training'
test_dir = '/home/choieunji/다운로드/Facial-Similarity-with-Siamese-Networks-in-Pytorch-master/data/faces/testing'
batch_size = 64
num_epochs = 100

# 데이터셋 전처리
class SiameNetworkDataset(Dataset):
    def __init__(self, imageFolderDataset,transform=None,should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        should_get_some_class = random.randint(0, 1)
        if should_get_some_class:
            while True:
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)

            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)

            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


def imshow(img, text=None, should_save = False):
    npimg = img.numpy()
    plt.axis('off')
    if text:
        plt.text(75, 8, text, style = 'italic', fontweight = 'bold',
            bbox = {'facecolor': 'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()



dataset = ImageFolder(root=train_dir)
siamese_dataset = SiameNetworkDataset(imageFolderDataset=dataset,
                                      transform=transforms.Compose([transforms.Resize((100,100)), transforms.ToTensor()])
                                      ,should_invert=False)

train_dataloader = DataLoader(siamese_dataset, shuffle = True, num_workers = 8, batch_size = 8)

dataiter = iter(train_dataloader)
example_batch = next(dataiter)
concatenated = torch.cat((example_batch[0], example_batch[1]), 0)
imshow(torchvision.utils.make_grid(concatenated))

print(example_batch[2].numpy())