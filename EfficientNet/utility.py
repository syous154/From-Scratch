#!/usr/bin/env python
# coding: utf-8

# In[3]:


import import_ipynb ## ipynb파일을 import 하게 도와주는 모듈
import model
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


# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


# ### 이미지 전처리

# In[ ]:


from PIL import Image

class decode_and_center_crop(object):
    def __init__(self, image_size=224, crop_fraction=0.875):
        self.image_size = image_size
        self.crop_fraction = crop_fraction

    def __call__(self, img):
        # Get image size
        image_width, image_height = img.size

        # Calculate crop parameters
        crop_padding = round(self.image_size * (1 / self.crop_fraction - 1))
        padded_center_crop_size = int((self.image_size / (self.image_size + crop_padding)) * min(image_height, image_width))

        # Calculate crop window
        offset_height = (image_height - padded_center_crop_size + 1) // 2
        offset_width = (image_width - padded_center_crop_size + 1) // 2
        crop_window = (offset_width, offset_height, offset_width + padded_center_crop_size, offset_height + padded_center_crop_size)

        # Apply crop
        img = img.crop(crop_window)

        # Apply resize
        img = img.resize((self.image_size, self.image_size), Image.BICUBIC)    

        return img

class decode_and_random_crop:
    def __init__(self, image_size=224):
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(self.image_size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)),
        ])

    def __call__(self, image):
        return self.transform(image)


# In[ ]:


data_path = 'D:/CIFAR_10/'

train_transform = transforms.Compose([
    decode_and_center_crop(),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(policy=AutoAugmentPolicy.CIFAR10),  # AutoAugment 적용
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_transform = transforms.Compose([
    decode_and_center_crop(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# Load the entire training data
full_trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                             download=True, transform=train_transform)

# Split the full training set into a training set and a validation set
train_size = 4000
val_size = 1000
trainset, remaining = torch.utils.data.random_split(full_trainset, [train_size, len(full_trainset) - train_size])
valset, _ = torch.utils.data.random_split(remaining, [val_size, len(remaining) - val_size])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=0)
valloader = torch.utils.data.DataLoader(valset, batch_size=32,
                                        shuffle=False, num_workers=0)

# Load the entire test data
full_testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                            download=True, transform=test_transform)

# Use only the first 10000 test data
test_size = 1000
_, testset = torch.utils.data.random_split(full_testset, [len(full_testset) - test_size, test_size])

testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False)



# In[ ]:


criterion = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

def train_model(model, epochs):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()

            # 정확도 계산
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        # 스케줄러 업데이트
        #flag = False
        #if epoch > 0 and epoch % 2 == 0 and flag == True:
        #  optimizer.param_groups[0]['lr'] *= 0.97
        #  flag = False
        #if epoch > 0 and epoch % 3 == 0 and flag == False:
        #  optimizer.param_groups[0]['lr'] *= 0.97
        #  flag = True
        
        accuracy = 100 * correct / total
        #train_losses.append(running_loss / len(trainloader))
        #train_accuracies.append(accuracy)
        
        print('[%d/%d] loss: %.3f, accuracy: %.2f %%' % (epoch + 1, epochs, running_loss / len(trainloader), accuracy))
        
        # 검증 세트에서 손실과 정확도 계산
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for i, data in enumerate(valloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        #val_losses.append(val_running_loss / len(valloader))
        #val_accuracies.append(val_accuracy)
        print('[%d/%d] val loss: %.3f, val accuracy: %.2f %%' % (epoch + 1, epochs, val_running_loss / len(valloader), val_accuracy))


# In[ ]:


# 테스트 데이터에 대한 정확도 확인
def test_model(model):
  model.eval()
  correct = 0
  total = 0

  with torch.no_grad():
      for data in valloader:
          images, labels = data[0].to(device), data[1].to(device)
          outputs = model(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

  #print('Accuracy on the test images: %.2f %%' % (100 * correct / total))
  return correct


# In[ ]:


import numpy as np

# 가능한 값의 범위를 정의합니다.
min_value = 1
max_value = 2
alphas = []
betas = []
gammas = []
# 가능한 모든 조합을 반복합니다.
for alpha in np.arange(min_value, max_value, 0.05):
    for beta in np.arange(min_value, max_value, 0.05):
        for gamma in np.arange(min_value, max_value, 0.05):
            # 식을 계산합니다.
            value = alpha * (beta ** 2) * (gamma ** 2)
            # 식이 2에 가까운지 확인합니다.
            if np.isclose(value, 2, atol=0.1):
                alpha = round(alpha, 2)
                beta = round(beta, 2)
                gamma = round(gamma, 2)

                alphas.append(alpha)
                betas.append(beta)
                gammas.append(gamma)
                print(f"알파: {alpha}, 베타: {beta}, 감마: {gamma}, value = {value}")

                
            if alpha == 1.2 and beta == 1.1 and gamma == 1.15:
                
                print(f"알파: {alpha}, 베타: {beta}, 감마: {gamma}, value = {value}")


# In[ ]:


models_acc = [] ## 파라미터 별 val acc를 저장
criterion = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

for i in range(len(alphas)):
    model = model.EfficientNetB0(num_classes = 10,width_coef = betas[i], depth_coef = alphas[i], resolution_coef = gammas[i]).to(device)

    train_model(model = model, epochs = 10)
  
    models_acc.append(test_model(model = model))
    del model
    torch.cuda.empty_cache()  # GPU 메모리 해제 시도
    
    print(f'{i+1}회 완료')
    
max_acc = max(models_acc)
max_index = models_acc.index(max_acc)
alpha = alphas[max_index]
beta = betas[max_index]
gamma = gammas[max_index]


# In[ ]:


print(f'가장 높은 acc를 보이는 파라미터의 조합 alpha : {alpha}, beta : {beta}, gamma : {gamma}')


# In[ ]:


phis = [0, 1, 2, 3, 4, 5, 6, 7]  #b0 ~ b7까지의 파이 값
a = 1.2
b = 1.1
c = 1.15
index = 0
for phi in phis:
  print(f'efficientnet-b{index} : ({round(b**(phi),1)}, {round(a**(phi),1)}, {round(c**(phi)*224,0)})')
  index += 1


# In[ ]:


phis = [0, 0.5, 1, 2, 3.8, 5.15, 6.2, 7.2]  #파이 값 수정
a = 1.17                                    #알파 값 수정 (케라스 EfficientNet 문서 기반(https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/)) 
b = 1.1
c = 1.15
index = 0
for phi in phis:
  print(f'efficientnet-b{index} : ({round(b**(phi),1)}, {round(a**(phi),1)}, {round(round(c**(phi),2)*224,0)})')
  index += 1

