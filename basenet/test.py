import torch
import torch.nn as nn
#from senet import se_resnext50_32x4d
from basenet.multimodal import MultiModalNet

from basenet.senet import se_resnet50,se_resnext101_32x4d,se_resnext50_32x4d
#from oct_resnet import oct_resnet26,oct_resnet101
from basenet.nasnet import nasnetalarge
from basenet.multiscale_resnet import multiscale_resnet
from basenet.DPN import DPN92, DPN26
from basenet.multiscale_se_resnext import multiscale_se_resnext

import torch.backends.cudnn as cudnn
torch.set_default_tensor_type('torch.cuda.FloatTensor')
cudnn.benchmark = True
model = multiscale_se_resnext(9)
print(model(torch.randn(16,3,100,100).float().cuda()).size())
"""
model = MultiModalNet('se_resnext26_32x4d', 'DPN26', 0.5)
print(model(torch.randn(16,3,100,100).float().cuda(),torch.randn(16,7,24,26).float().cuda()))
"""
