import torch
from torch import nn
from torch.nn import functional as F
import torchsummary
import torchvision

class ResBlk(nn.Module):
    '''
    ResNet Block
    '''
    def __init__(self, ch_in, ch_out, stride=1):
        '''

        :param ch_in:
        :param ch_out:
        '''

        super(ResBlk,self).__init__()
        self.in_feature = ch_in
        self.out_feature = ch_out

        _features = ch_out

        if self.in_feature != self.out_feature:
            # 在输入通道和输出通道不相等的情况下计算通道是否为2倍差值
            if self.out_feature / self.in_feature == 2.0:
                stride = 2  # 在输出特征是输入特征的2倍的情况下 要想参数不翻倍 步长就必须翻倍
            else:
                raise ValueError("输出特征数最多为输入特征数的2倍！")

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)


        self.extra = None if self.in_feature == self.out_feature else nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=2),
            nn.BatchNorm2d(ch_out)
        )

    def forward(self, x):
        '''

        :param x:[b, ch, h, w]
        :return:
        '''
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.in_feature != self.out_feature:
            identity = self.extra(x)
        # shortcut
        # element-wise add
        #print('extra', self.extra(x).shape)
        #print('out', out.shape)

        out = identity + out
        out = F.relu(out)
        return out

class ResNet18(nn.Module):

    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        # followed 4 blocks
        # [b, 64, h, w] => [b, 128, h, w]
        self.blk1 = nn.Sequential(
            ResBlk(64, 64),
            ResBlk(64, 64)
        )

        # [b, 128, h, w] => [b, 256, h, w]
        self.blk2 = nn.Sequential(
            ResBlk(64, 128),
            ResBlk(128, 128)
        )
        # [b, 256, h, w] => [b, 512, h, w]
        self.blk3 = nn.Sequential(
            ResBlk(128, 256),
            ResBlk(256, 256)
        )
        # [b, 512, h, w] => [b, 1024, h, w]
        self.blk4 = nn.Sequential(
            ResBlk(256, 512),
            ResBlk(512, 512)
        )
        #self.avgpool = F.adaptive_avg_pool2d(output_size=[1, 1])
        #self.blk5 = ResBlk(512, 512, stride=2)
        #self.fc1 = nn.Linear(512, 128)

        self.outlayer = nn.Linear(512, 4)
    def forward(self, x):
        '''

        :param x:
        :return:
        '''
        x = F.relu(self.conv1(x))
        # [b, 64, h, w] => [b, 1024, h, w]
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        #x = self.blk5(x)
        #print('after conv:', x.shape) # [b, 512, 2, 2]
        #x = F.adaptive_avg_pool2d(x, [1, 1])# [b, 512, 1, 1]
        #print('after pool:', x.shape)
        x = F.adaptive_avg_pool2d(x, [1, 1])
        x = torch.flatten(x, 1)
        x = self.outlayer(x)
        return x

def main():
    #blk = ResBlk(64, 128, stride=2)
    #tmp = torch.randn(2, 64, 32, 32)
    #out = blk(tmp)
    #print(out.shape)
    device = torch.device('cuda')
    x = torch.randn(128, 3, 32, 32)
    x = x.to(device='cuda')
    model = torchvision.models.resnet18(weights=None)
    model = model.to(device)
    input = (3, 32, 32)

    torchsummary.summary(model, input)
    out = model(x)
    print(out.shape)

#main()

