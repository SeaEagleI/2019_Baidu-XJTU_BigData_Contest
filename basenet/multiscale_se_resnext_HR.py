from torch import nn
import torch
from torchvision import models,transforms,datasets
import torch.nn.functional as F
from basenet.senet import se_resnet50,se_resnext101_32x4d,se_resnext50_32x4d, se_resnext26_32x4d

class multiscale_se_resnext_HR(nn.Module):
    def __init__(self,num_class, pretrain = True):
        super(multiscale_se_resnext_HR,self).__init__()

        self.base_model = se_resnext50_32x4d(9, None)

        if pretrain == True :
            MODEL_PATH = './pretrained_seresnet/se_resnext50_32x4d-a260b3a4.pth'
            print("load model from {}".format(MODEL_PATH))
            state_dict = torch.load(MODEL_PATH)

            state_dict.pop('last_linear.bias')
            state_dict.pop('last_linear.weight')
            self.base_model.load_state_dict(state_dict, strict = False)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)

        #self.classifier = nn.Linear(2048,256)

    def forward(self, x):
        input_size = x.size()[2]
        self.interp = nn.UpsamplingBilinear2d(size = (int(input_size*1.25)+1,  int(input_size*1.25)+1))

        x2 = self.interp(x)

        x = self.base_model(x)

        x2 = self.base_model(x2)

        out = x + x2

        return out
