from torchvision.models.resnet import model_urls
import torchvision.models as models
from torchvision import transforms
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import init

class Attr_net(nn.Module):
    def __init__(self,attr_num=2):
        super(Attr_net, self).__init__()
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs,2)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self,x):
        out = self.model(x)
        return out

if __name__ == '__main__':
    x = torch.Tensor(2,3,64,64)
    net = Attr_net()
    print(net)
    y = net(x)
    print(y.size())