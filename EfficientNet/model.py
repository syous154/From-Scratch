#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from torchsummary import summary
from torch import optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader, random_split
from torchvision.ops import stochastic_depth
from torch.optim.lr_scheduler import StepLR
import math
from torchvision.transforms import AutoAugmentPolicy


# In[2]:


def round_repeats(repeats, depth_coefficient):
    return int(math.ceil(depth_coefficient * repeats))

def round_filters(filters, width_coefficient, depth_divisor):
    filters *= width_coefficient
    new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


# In[3]:


class SEBlock(nn.Module):
    def __init__(self, in_channel, r, expand):                                                                        ### 아래 EfficientNet의 SE 과정을 보면 r=0.04인것 처럼 진행함(논문에선 0.25이것도 모름)
        super().__init__()
        
        ## 축소될 채널 수 계산
        if expand == 1:
          r = 0.25
        sq_channel = max(1,int(in_channel * r))
            
        
        self.se = nn.Sequential(
            ## squeeze
            nn.AdaptiveAvgPool2d(1),
            
            ## Exitation
            nn.Conv2d(in_channel, sq_channel, kernel_size = 1),
            nn.SiLU(inplace = True),
            nn.Conv2d(sq_channel, in_channel,kernel_size = 1),
            nn.Sigmoid()
        )
        
        
    
    def forward(self, x):
        return x * self.se(x)
    
# Chaeck용
if __name__ == '__main__':
    x = torch.randn(3, 3, 224, 224)
    model = SEBlock(x.size(1), 0.04, 1)
    output = model(x)
    print('output size:', output.size())


# In[4]:


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, expand, padding_size ,stride=1 ,r=0.04, dropout_rate=0.2, p = 0.0):
        super().__init__()
        
        # 변수 설정
        self.dropout_rate = dropout_rate
        self.expand = expand
        self.p = p
        # skip connection 사용을 위한 조건 지정
        self.use_residual = in_channels == out_channels and stride == 1        #stride가 1이면서 input 채널과 ouput 채널의 수가 같으면 use_residual의 값은 True
        
        expand_channels = in_channels * expand                                  #확장된 채널 수 계산

        ## Expand Phase
        self.expasion = nn.Sequential(
            nn.Conv2d(in_channels, expand_channels,kernel_size= 1, stride = 1),
            nn.BatchNorm2d(expand_channels, momentum=0.99),
            nn.SiLU()
        )
        ## Depthwise Conv Phase
        self.depthwise = nn.Sequential(
            nn.Conv2d(expand_channels, expand_channels, kernel_size= kernel_size, 
                      stride=stride, padding = padding_size, groups = expand_channels),
            nn.BatchNorm2d(expand_channels, momentum=0.99),
            nn.SiLU()
        )
        ## SE Phase
        self.seblock = SEBlock(expand_channels, r, expand)
        ## Pointwise Conv Phase
        self.pointwise = nn.Sequential(
            nn.Conv2d(expand_channels, out_channels,kernel_size= 1, stride = 1),
            nn.BatchNorm2d(out_channels, momentum=0.99)
        )
    def forward(self, x):
        if self.training and self.use_residual:
            if torch.rand(1)[0] < self.p:
                return x
        res = x
        
        if self.expand != 1:                                        ## 확장 비율이 1인경우는 확장이 되지 않는 것이기 때문에 생략함
            x = self.expasion(x)
            
        x = self.depthwise(x)
        x = self.seblock(x)
        x = self.pointwise(x)
        
        if self.training and (self.dropout_rate is not None):       ## self.training은 훈련중인지를 판단 -> 훈련 중이면 드롭아웃 실시
                x = F.dropout2d(input=x, p=self.dropout_rate, training=self.training, inplace=True)
        
        if self.use_residual:
            x = x + res
        return x
    
    
# Chaeck용
#if __name__ == '__main__':
#    x = torch.randn(3, 16, 17, 17)
#    model = MBConv(16,24,3,6)
#    output = model(x)
#    print('output size:', output.size())


# In[5]:


class EfficientNet(nn.Module):
    def __init__(self, num_classes=1000, width_coef=1., depth_coef=1., resolution_coef=1., dropout=0.2):
        super().__init__()
        channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        repeats = [1, 2, 2, 3, 3, 4, 1]
        strides = [1, 2, 2, 2, 1, 2, 1]
        kernel_size = [3, 3, 5, 3, 5, 5, 3]
        depth = depth_coef
        width = width_coef
        depth_divisor = 8

        repeats = [round_repeats(x, depth) for x in repeats]
        self.p = []
        for i in range(sum(repeats)):
            self.p.append(i * 0.008695652173913044)
        self.p_idx = 0
        
        #print(channels)
        #print(repeats)
        # efficient net

        #self.upsample = lambda x: F.interpolate(x, scale_factor=resolution_coef, mode='bilinear', align_corners=False)      # 채널 수는 그대로 H x W의 크기를 resolution_coef 값을 곱한만큼으로 조정, 
        # upsample된 크기의 이미지를 바랄 뿐이고 내가 직접 건드릴 필요는 없다.(keras에서 그렇게 구현함)

        self.stage1 = nn.Sequential(
            nn.Conv2d(3, channels[0],3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0], momentum=0.99),
            nn.SiLU(inplace=True)

        )

        self.stage2 = self._make_Block(MBConv, repeats[0], channels[0], round_filters(channels[1],width,depth_divisor), kernel_size[0], strides[0], expand = 1, padding_size = 1)

        self.stage3 = self._make_Block(MBConv, repeats[1], round_filters(channels[1],width,depth_divisor), round_filters(channels[2],width,depth_divisor), kernel_size[1], strides[1], expand = 6, padding_size = 1)

        self.stage4 = self._make_Block(MBConv, repeats[2], round_filters(channels[2],width,depth_divisor), round_filters(channels[3],width,depth_divisor), kernel_size[2], strides[2], expand = 6, padding_size = 2)

        self.stage5 = self._make_Block(MBConv, repeats[3], round_filters(channels[3],width,depth_divisor), round_filters(channels[4],width,depth_divisor), kernel_size[3], strides[3], expand = 6, padding_size = 1)

        self.stage6 = self._make_Block(MBConv, repeats[4], round_filters(channels[4],width,depth_divisor), round_filters(channels[5],width,depth_divisor), kernel_size[4], strides[4], expand = 6, padding_size = 2)

        self.stage7 = self._make_Block(MBConv, repeats[5], round_filters(channels[5],width,depth_divisor), round_filters(channels[6],width,depth_divisor), kernel_size[5], strides[5], expand = 6, padding_size = 2)

        self.stage8 = self._make_Block(MBConv, repeats[6], round_filters(channels[6],width,depth_divisor), round_filters(channels[7],width,depth_divisor), kernel_size[6], strides[6], expand = 6, padding_size = 1)

        self.stage9 = nn.Sequential(
            nn.Conv2d(round_filters(channels[7],width,depth_divisor), round_filters(channels[8],width,depth_divisor), kernel_size = 1, stride=1, bias=False),
            nn.BatchNorm2d(round_filters(channels[8],width,depth_divisor), momentum=0.99),
            nn.SiLU()
        ) 

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(round_filters(channels[8],width,depth_divisor), num_classes)

    def forward(self, x):
        #print(f'input = {x.size()} ')
        #x = self.upsample(x)
        #print(f'upsample = {x.size()}')
        x = self.stage1(x)
        #print(f'stage 1 = {x.size()}')
        x = self.stage2(x)
        #print(f'stage 2 = {x.size()}')
        x = self.stage3(x)
        #print(f'stage 3 = {x.size()}')
        x = self.stage4(x)
        #print(f'stage 4 = {x.size()}')
        x = self.stage5(x)
        #print(f'stage 5 = {x.size()}')
        x = self.stage6(x)
        #print(f'stage 6 = {x.size()}')
        x = self.stage7(x)
        #print(f'stage 7 = {x.size()}')
        x = self.stage8(x)
        #print(f'stage 8 = {x.size()}')
        x = self.stage9(x)
        #print(f'stage 9 = {x.size()}')
        x = self.avgpool(x)
        #print(f'stage pool= {x.size()}')
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.linear(x)
        return x


    def _make_Block(self, block, repeats, in_channels, out_channels, kernel_size, stride, expand, padding_size):
        strides = [stride] + [1] * (repeats - 1)    #[stride, 1, 1, 1, 1] 이런 식으로 만들어줌
        layers = []
        
        for stride in strides:
            layers.append(block(in_channels, out_channels, kernel_size, expand, stride = stride, padding_size = padding_size, p=self.p[self.p_idx]))
            in_channels = out_channels
            self.p_idx += 1
            #print(layers)
        return nn.Sequential(*layers)


# In[7]:


def EfficientNetB0(num_class = 1000):
  return EfficientNet(num_classes = num_class, width_coef=1.0, depth_coef= 1.0, resolution_coef = 224/224, dropout=0.2)

def EfficientNetB1(num_class = 1000):
  return EfficientNet(num_classes = num_class, width_coef=1.0, depth_coef= 1.1, resolution_coef = 240/224, dropout=0.2)

def EfficientNetB2(num_class = 1000):
  return EfficientNet(num_classes = num_class, width_coef=1.1, depth_coef= 1.2, resolution_coef = 260/224, dropout=0.3)

def EfficientNetB3(num_class = 1000):
  return EfficientNet(num_classes = num_class, width_coef=1.2, depth_coef= 1.4, resolution_coef = 300/224, dropout=0.3)

def EfficientNetB4(num_class = 1000):
  return EfficientNet(num_classes = num_class, width_coef=1.4, depth_coef= 1.8, resolution_coef = 380/224, dropout=0.4)

def EfficientNetB5(num_class = 1000):
  return EfficientNet(num_classes = num_class, width_coef=1.6, depth_coef= 2.2, resolution_coef = 456/224, dropout=0.4)

def EfficientNetB6(num_class = 1000):
  return EfficientNet(num_classes = num_class, width_coef=1.8, depth_coef= 2.6, resolution_coef = 528/224, dropout=0.5)

def EfficientNetB7(num_class = 1000):
  return EfficientNet(num_classes = num_class, width_coef=2.0, depth_coef= 3.1, resolution_coef = 600/224, dropout=0.5)

