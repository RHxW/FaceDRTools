from Asian_European_FR.dataloader import MyDataset
import argparse
from torch.utils.data import DataLoader
import cv2
import numpy as np
import torch
from Asian_European_FR.model import Attr_net
import torch.backends.cudnn as cudnn
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import time
from Asian_European_FR.utils import warm_up_lr,FocalLoss
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Attribute demo')
parser.add_argument('--test_data_path', default=r'/home/traindata/facedata/train_glint360_96/imgs/', help='test data path')
parser.add_argument('--resize', default=64, type=int, help='network input size')
parser.add_argument('--attr_num', default=1, type=int, help='attridute number')
parser.add_argument('--resume_net', default=r'weights/parameter_epoch12_iter300_preci0.9781_loss0.003_.pth', help='resume iter for retraining')
parser.add_argument('--cls', default=['asian_face', 'european_face'], type=list)
args = parser.parse_args()

def preprocessing(image):
    image = torch.Tensor(image) / 255
    image = ((image - torch.Tensor([0.406, 0.456, 0.485])) / torch.Tensor([0.225, 0.224, 0.229])).permute(2, 0, 1).contiguous().unsqueeze(0)
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

    id_list = os.listdir(args.test_data_path)

    f_in = open('./european_idlist.txt', 'w')
    f_Asian = open('./asian_idlist.txt', 'w')
    print(len(id_list))
    for id_name in tqdm(id_list):
        
        img_list = os.listdir(os.path.join(args.test_data_path, id_name))

        for img_name in img_list[:1]:
            img_path = os.path.join(args.test_data_path, id_name, img_name)
            image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
            image = cv2.resize(image,(args.resize,args.resize))
            img = preprocessing(image)
            img = img.to(device)
            out = net(img)
            out = out.cpu()
            _, indexs = torch.Tensor.max(out, dim=1)

            cls = args.cls[indexs[0].item()]
            # print(cls)
            if cls == 'asian_face':
                f_Asian.writelines([str(id_name), '\n'])
                break
            else:
                f_in.writelines([str(id_name), '\n'])
                break

    f_in.close()
    f_Asian.close()

            # image = cv2.resize(image, (400, 400))
            # cv2.putText(image, args.cls[indexs[0].item()], (10, 50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1)
            # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            # cv2.imshow('image', image)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
