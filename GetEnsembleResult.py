import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import os,os.path as op
from Dataloader.MultiModal_BDXJTU2019 import BDXJTU2019_test
from basenet.multimodal import MultiModalNet

CLASSES = ['001', '002', '003', '004', '005', '006', '007', '008', '009']

def GetEnsembleResult():
    # Priors
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(0)

    # Dataset
    Dataset = BDXJTU2019_test(root = 'data',TEST_IMAGE_DIR = 'test')
    Dataloader = data.DataLoader(Dataset,1,num_workers=1,shuffle=False,pin_memory=True)

    # Network
    cudnn.benchmark = True    

    net1 = MultiModalNet('se_resnext50_32x4d', 'DPN26', 0.5)
    net1.load_state_dict(torch.load('./weights/best_models/se_resnext50_32x4d_SGD_1_20.pth'))
    net1.eval()

    net2 = MultiModalNet('multiscale_se_resnext_HR', 'DPN26', 0.5)
    net2.load_state_dict(torch.load('./weights/best_models/multiscale_se_resnext_HR_SGD_16.pth'))
    net2.eval()

#    filename = './submission/MM_epoch26_25_all_pretrained_2HR_616v2.txt'
    submit_file = './submission/submission.txt'
    f = open(submit_file, 'w+')

    for (image_tensor, visit_tensor, anos) in tqdm(Dataloader):
        Tensor_1   = net1.forward(image_tensor.cuda(), visit_tensor.cuda())
        Tensor_HR  = net2.forward(image_tensor.cuda(), visit_tensor.cuda())
        preds = torch.nn.functional.normalize(Tensor_1) \
                + torch.nn.functional.normalize(Tensor_HR)
        _, pred = preds.data.topk(1, 1, True, True)
        #f.write(anos[0] + ',' + CLASSES[4] + '\r\n')
#        print('{}\t{}'.format(anos[0][:-4],CLASSES[pred[0][0]]))
        f.writelines('{}\t{}\n'.format(anos[0][:-4],CLASSES[pred[0][0]]))
    f.close()

if __name__ == '__main__':
    GetEnsembleResult()
