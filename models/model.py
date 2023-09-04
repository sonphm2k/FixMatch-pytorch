import math
import torch
from torch.nn import init
import torch.nn as nn
import timm

class WideEffnet(nn.Module):
    def __init__(self, num_class=10, model_name="", droprate=0.5, linear_num=512, circle=False, pretrained=False):
        super().__init__()
        self.output_num = [4,2,1]
        self.model_name = model_name
        model_ft = timm.create_model(self.model_name, pretrained = pretrained, drop_path_rate = 0.2) # Backbone
        model_ft.head = nn.Sequential() # save memory
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.classifier = nn.Sequential()
        self.model = model_ft
        # For EfficientNet, the feature dim is not fixed
        # for efficientnet_b0 1280
        # for efficientnet_b4 1792
        self.classifier = ClassBlock(26880, num_class, droprate, linear=linear_num, return_f = circle)

    # reference: https://github.com/yueruchen/sppnet-pytorch/blob/master/spp_layer.py
    def spatial_pyramid_pool(self, previous_conv, previous_conv_size, out_pool_size=[4, 2, 1]):
        '''
          previous_conv: a tensor vector of previous convolution layer
          previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
          out_pool_size: a int vector of expected output size of max pooling layer

          returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
        '''
        num_sample = previous_conv.shape[0]
        for i in range(len(out_pool_size)):
          h_wid = int(math.ceil(previous_conv_size[0]/out_pool_size[i]))
          w_wid = int(math.ceil(previous_conv_size[1]/out_pool_size[i]))
          h_pad = (h_wid*out_pool_size[i]-previous_conv_size[0]+1)/2
          w_pad = (w_wid*out_pool_size[i]-previous_conv_size[1]+1)/2
          maxpool = torch.nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(int(h_pad), int(w_pad)))
          x = maxpool(previous_conv)
          if(i == 0):
            spp = x.view(num_sample, -1)
          else:
            spp = torch.cat((spp,x.view(num_sample, -1)), 1)
        return spp

    def forward(self, x):
        x = self.model.forward_features(x)
        # x = self.model.avgpool(x)
        x = self.spatial_pyramid_pool(x, [int(x.size(2)), int(x.size(3))])
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x

class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, linear=512, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear>0:
            add_block += [nn.Linear(input_dim, linear)]
        else:
            linear = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(linear)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(linear, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
          f = x
          x = self.classifier(x)
          return [x,f]
        else:
          x = self.classifier(x)
          return x

def weights_init_kaiming(m):
  classname = m.__class__.__name__
  # print(classname)
  if classname.find('Conv') != -1:
      init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
  elif classname.find('Linear') != -1:
      init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
  elif classname.find('BatchNorm1d') != -1:
      init.normal_(m.weight.data, 1.0, 0.02)
  if hasattr(m, 'bias') and m.bias is not None:
      init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
  classname = m.__class__.__name__
  if classname.find('Linear') != -1:
      init.normal_(m.weight.data, std=0.001)
      init.constant_(m.bias.data, 0.0)

# Test
# model = WideEffnet(num_class=10, model_name="efficientnet_b0", pretrained=False)