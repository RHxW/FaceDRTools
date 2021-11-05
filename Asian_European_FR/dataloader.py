from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import cv2
import random


class MyDataset(Dataset):

    def __init__(self, data_path,train=True):
        self.data_path = data_path
        self.train = train

        folder_names = ['asian','european']


        self.image_paths = []
        self.labels = []
        for i,folder_name in enumerate(folder_names):
            for j in [os.path.join(data_path,folder_name,i.strip()) for i in os.listdir(os.path.join(data_path,folder_name))]:
                self.image_paths.append(j)
                self.labels.append(i)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        label = self.labels[index]

        image = cv2.imdecode(np.fromfile(image_path,dtype=np.uint8),-1)
        if self.train:
            image = self._mirror(image)
            image = self._distort(image)

        image = torch.Tensor(image)/255
        image = ((image - torch.Tensor([0.406,0.456,0.485]))/torch.Tensor([0.225,0.224,0.229])).permute(2,0,1).contiguous()
        # label = torch.tensor(label)

        return image, label

    def __len__(self):
        return len(self.image_paths)

    def _mirror(self,image):
        if random.randrange(2):
            image = image[:, ::-1]
        return image

    def _distort(self,image):
        def _convert(image, alpha=1, beta=0):
            tmp = image.astype(float) * alpha + beta
            tmp[tmp < 0] = 0
            tmp[tmp > 255] = 255
            image[:] = tmp

        image = image.copy()

        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.6, 1.4))

        # image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        #
        # if random.randrange(2):
        #     tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
        #     tmp %= 180
        #     image[:, :, 0] = tmp
        #
        # if random.randrange(2):
        #     _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))
        #
        # image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        return image


if __name__ == '__main__':
    resize = 64
    dataset = MyDataset(data_path=r'G:\Asian_European\train')
    custom_loader = DataLoader(dataset, batch_size=10, shuffle=False)

    for epoch in range(1):
        for i, (images,labels) in enumerate(custom_loader):
            print(images.size())
            print(labels.size())
            print(labels)
            cv2.imshow('image',np.array(images[0].permute(1,2,0)))
            cv2.waitKey()
            cv2.destroyAllWindows()