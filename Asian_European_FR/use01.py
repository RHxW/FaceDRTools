from dataloader import MyDataset
import argparse
from torch.utils.data import DataLoader
import cv2
import numpy as np
import torch
from model import Attr_net
import torch.backends.cudnn as cudnn
import torch.nn as nn
import os
import shutil
import matplotlib.pyplot as plt
import time
from utils import warm_up_lr,FocalLoss

'''
    用于提取出face_emoer数据集中的亚洲人脸
'''

parser = argparse.ArgumentParser(description='asian face extract')
parser.add_argument('--data_source_path', default=r'G:\img64s')
parser.add_argument('--data_save_path', default=r'G:\face_emoer_asian_face')
parser.add_argument('--resize', default=64, type=int, help='network input size')
parser.add_argument('--attr_num', default=1, type=int, help='attridute number')
parser.add_argument('--resume_net', default=r'weights/parameter_epoch12_iter300_preci0.9781_loss0.003_.pth', help='resume iter for retraining')
parser.add_argument('--cls', default=['asian_face','european_face'],type=list)
args = parser.parse_args()

def preprocessing(image):
    image = torch.Tensor(image) / 255
    image = ((image - torch.Tensor([0.406, 0.456, 0.485])) / torch.Tensor([0.225, 0.224, 0.229])).permute(0, 3, 1, 2).contiguous()
    return image

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if str(device) == 'cuda':
        cudnn.benchmark = True
    net = Attr_net(attr_num=args.attr_num)
    net = net.to(device)
    net.eval()

    checkpoint = torch.load(args.resume_net)
    net.load_state_dict(checkpoint['net'])

    folder_names = [i.strip() for i in os.listdir(args.data_source_path)]

    for folder_name in folder_names:
        images = []
        print(folder_name)

        if int(folder_name) <= 145470:
            continue
        if len(os.listdir(os.path.join(args.data_source_path,folder_name))) < 2:
            continue
        image_paths = [os.path.join(args.data_source_path,folder_name,i.strip()) for i in os.listdir(os.path.join(args.data_source_path,folder_name))]

        _ = [images.append(cv2.imread(j)) for i,j in enumerate(image_paths) if i < 30]
        imgs = np.array(images)
        imgs_ = preprocessing(imgs)
        imgs_ = imgs_.to(device)

        with torch.no_grad():
            out = net(imgs_)
            out = out.cpu()
            _, indexs = torch.Tensor.max(out, dim=1)

            if torch.Tensor.sum(indexs).item() <= len(indexs) // 2:
                if not os.path.exists(os.path.join(args.data_save_path, folder_name)):
                    os.makedirs(os.path.join(args.data_save_path, folder_name))
                _ = [shutil.move(i, os.path.join(args.data_save_path, folder_name)) for i in image_paths]

