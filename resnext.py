#!/usr/bin/env python
# coding: utf-8

# In[34]:


import torch


# In[35]:


import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse


# In[94]:


from flask import Flask, request, jsonify


# In[37]:


from PIL import Image
import torchvision.transforms.functional as TF


# In[38]:


import numpy as np
import requests
from io import BytesIO


# In[ ]:





# In[39]:


class Block(nn.Module):
    #Grouped convolution block
    expansion = 2

    def __init__(self, in_planes, cardinality=32, bottleneck_width=4, stride=1):
        super(Block, self).__init__()
        group_width = cardinality * bottleneck_width
        self.conv1 = nn.Conv2d(in_planes, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)
        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(group_width)
        self.conv3 = nn.Conv2d(group_width, self.expansion*group_width, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*group_width)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*group_width:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*group_width, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*group_width)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNeXt(nn.Module):
    def __init__(self, num_blocks, cardinality, bottleneck_width, num_classes=10):
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(num_blocks[0], 1)
        self.layer2 = self._make_layer(num_blocks[1], 2)
        self.layer3 = self._make_layer(num_blocks[2], 2)
        # self.layer4 = self._make_layer(num_blocks[3], 2)
        self.linear = nn.Linear(cardinality*bottleneck_width*8, num_classes)

    def _make_layer(self, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(Block(self.in_planes, self.cardinality, self.bottleneck_width, stride))
            self.in_planes = Block.expansion * self.cardinality * self.bottleneck_width
        # Increase bottleneck_width by 2 after each stage.
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.layer4(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNeXt29_2x64d():
    return ResNeXt(num_blocks=[3,3,3], cardinality=2, bottleneck_width=64)

def ResNeXt29_4x64d():
    return ResNeXt(num_blocks=[3,3,3], cardinality=4, bottleneck_width=64)

def ResNeXt29_8x64d():
    return ResNeXt(num_blocks=[3,3,3], cardinality=8, bottleneck_width=64)

def ResNeXt29_32x4d():
    return ResNeXt(num_blocks=[3,3,3], cardinality=32, bottleneck_width=4)


# In[96]:


PORT = 8090

app = Flask(__name__)
@app.route("/")
def hello():
    return "Image classification example\n"
@app.route('/get_prediction', methods=['GET'])
def get_prediction():
    img_path_ = request.args['url']
    img_path = requests.get(img_path_)
    img = Image.open(BytesIO(img_path.content)).convert('RGB')
    image = img.resize((32, 32))
    width,height = image.size
    #print(width,height)
    img = TF.to_tensor(image)
    img.unsqueeze_(0)
    with torch.no_grad():
        model = torch.load('net.pth',map_location='cpu')
        outputs = model(img)
        # loss = criterion(outputs, targets)
        # test_loss += loss.item()
        # _, predicted = outputs.max(1)
        # total += targets.size(0)
        # correct += predicted.eq(targets).sum().item()
        softmax = torch.exp(outputs).cpu()
        prob = list(softmax.numpy())
        predictions = np.argmax(prob, axis=1)
        #print(predictions)
        op_val = None
        dic = {0 :'Airplane', 1:'automobile',2:'bird',3:'cat',4:'deer',5:'dog',6:'frog',7:'horse',8:'ship',9:'truck'}
        for i in dic :
            if i == predictions:
                op_val = dic[i]
        return jsonify(op_val)
        #print(predictions_1)
        
if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=PORT)


# In[73]:





# In[97]:


#temp = get_prediction("frog.jpg")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




