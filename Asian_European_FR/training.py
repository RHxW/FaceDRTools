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

parser = argparse.ArgumentParser(description='Attribute Training')
parser.add_argument('--train_data_path', default=r'G:\Asian_European\train', help='train data path')
parser.add_argument('--test_data_path', default=r'G:\Asian_European\test', help='test data path')
parser.add_argument('--resize', default=64, type=int, help='network input size')
parser.add_argument('--batch_size', default=512, type=int, help='Iteration size')
parser.add_argument('--num_workers', default=4, type=int, help='number workers')
parser.add_argument('--attr_num', default=1, type=int, help='attridute number')
parser.add_argument('--lr', default=1e-4, type=float, help='start learning rate')
parser.add_argument('--Epoch', default=500, type=int)
parser.add_argument('--NUM_EPOCH_WARM_UP', default=1, type=int)
parser.add_argument('--weights_sava_path', default='weights', help='weights save path')
parser.add_argument('--resume_net', default=r'weights/parameter_epoch9_iter300_preci0.9768_loss0.003_.pth', help='resume iter for retraining')
args = parser.parse_args()

def train(train_loader,device,net):
    net = net.to(device)
    if str(device) == 'cuda':
        cudnn.benchmark = True
    net.train()

    weight_ = torch.ones(args.batch_size,args.attr_num).to(device)
    weight_[:,0] = weight_[:,0] * 1.5

    # optimizer = torch.optim.SGD(params=net.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    optimizer = torch.optim.Adam(params=net.parameters(),lr=args.lr,weight_decay=0.0005)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=400,cooldown=160,min_lr=1e-8)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[4,7,10],gamma=0.1)
    focalLoss = FocalLoss()

    start_epoch = 0
    loss = 0

    if args.resume_net:
        checkpoint = torch.load(args.resume_net)
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1

    x = []
    y = []
    precision_ = 0
    # plt.ion()
    batch = 0
    for epoch in range(start_epoch,args.Epoch):
        # scheduler.step(loss)
        # scheduler.step()
        for i,(images,labels) in enumerate(train_loader):

            # batch += 1
            # if epoch < args.NUM_EPOCH_WARM_UP:
            #     warm_up_lr(batch, args.NUM_EPOCH_WARM_UP*(len(train_dataset)//args.batch_size), args.lr, optimizer)

            images = images.to(device)
            labels = labels.to(device).long()

            start_time = time.time()
            output = net(images)
            train_loss = focalLoss(output,labels)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            end_time = time.time()

            loss = train_loss.detach().item()

            if i%10 == 0:
                print('epoch/iter==>', ' ', epoch, '/', i,' ', 'loss==>', loss, ' ', 'lr==>',optimizer.state_dict()['param_groups'][0]['lr'],' ','time==>',end_time-start_time)

                # x.append((len(train_loader)*epoch) + i)
                # y.append(loss)
                # plt.clf()
                # plt.xlabel('iteration')
                # plt.ylabel('loss')
                # plt.plot(x, y, color='b', lw=1)
                # plt.pause(0.01)

            if i%100 == 0:
                precision = test(net,device)
                print('精度==>', precision)
                if (precision + 0.004) >= precision_:
                    precision_ = precision

                    state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                    torch.save(state, os.path.join(args.weights_sava_path, 'parameter_epoch{}_iter{}_preci'.format(epoch,i) + '%.4f'%precision + '_loss%.3f'%loss + '_.pth'))

        state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, os.path.join(args.weights_sava_path, 'parameter_epoch{}_preci'.format(epoch) + '_loss%.3f' % loss + '_.pth'))

    # plt.ioff()

def test(net,device):
    test_dataset = MyDataset(data_path=args.test_data_path,train=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    net.eval()

    tp = 0
    sum_num = 0
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)

        with torch.no_grad():
            out = net(images)
            out = out.cpu()

            _,indexs = torch.Tensor.max(out,dim=1)
            tp += float(torch.Tensor.sum(indexs == labels).item())

        sum_num += float(labels.size(0))

    precision = tp / sum_num

    net.train()
    return precision


if __name__ == '__main__':
    train_dataset = MyDataset(data_path=args.train_data_path,train=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True ,num_workers=args.num_workers,drop_last=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Attr_net(attr_num=args.attr_num)
    train(train_loader,device,net)
