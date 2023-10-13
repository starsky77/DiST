#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

""" Compute saliency maps of images from dataset folder
    and dump them in a results folder """

import torch
from torchvision import datasets, transforms, utils, models
import os
import torch.nn as nn
import torch.nn.functional as F

# Import saliency methods
from saliency.fullgrad import FullGrad
from saliency.simple_fullgrad import SimpleFullGrad
from saliency.smooth_fullgrad import SmoothFullGrad

from saliency.gradcam import GradCAM
from saliency.grad import InputGradient
from saliency.smoothgrad import SmoothGrad

import argparse

from misc_functions import *

class BlurPoolConv2d(nn.Module):
    def __init__(self, conv):
        super().__init__()
        default_filter = torch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
        filt = default_filter.repeat(conv.in_channels, 1, 1, 1)
        self.conv = conv
        self.register_buffer('blur_filter', filt)

    def forward(self, x):
        blurred = F.conv2d(x, self.blur_filter, stride=1, padding=(1, 1),
                           groups=self.conv.in_channels, bias=None)
        return self.conv.forward(blurred)

def apply_blurpool(mod: nn.Module):
    for (name, child) in mod.named_children():
        if isinstance(child, nn.Conv2d) and (np.max(child.stride) > 1 and child.in_channels >= 16):
            setattr(mod, name, BlurPoolConv2d(child))
        else: apply_blurpool(child)


parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', default='resnet50_DiSTinguish', type=str)
parser.add_argument('--binary', default=1, type=int)
args = parser.parse_args()

# PATH variables
PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
dataset = PATH + 'test_image/'


batch_size = 1
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

# Dataset loader for sample images
sample_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(dataset, transform=transforms.Compose([
                       transforms.Resize((224,224)),
                       transforms.ToTensor(),
                       transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                        std = [0.229, 0.224, 0.225])
                   ])),
    batch_size= batch_size, shuffle=False)

unnormalize = NormalizeInverse(mean = [0.485, 0.456, 0.406],
                           std = [0.229, 0.224, 0.225])


model = models.resnet50(weights=False)
for (name, child) in model.named_children():
    if isinstance(child, nn.Conv2d) and (np.max(child.stride) > 1 and child.in_channels >= 16):
        setattr(model, name, BlurPoolConv2d(child))
    else: apply_blurpool(child)
if args.model == "resnet50_DiSTinguish" or args.model == "resnet50_DiSTinguish_Sty_aug":
    num_ftrs = model.fc.in_features
    num_classes = 2000
    model.fc = nn.Linear(num_ftrs, num_classes)


if args.model=="resnet50_DiSTinguish":
    model_path = PATH + "../../model/DiSTinguish/weights_ep_89.pt"
elif args.model=="resnet50_DiSTinguish_Sty_aug":
    model_path = PATH + "../../model/DiSTinguish_Style_aug/weights_ep_89.pt"
elif args.model=="resnet50_Sty_aug":
    model_path = PATH + "../../model/Style_aug/weights_ep_89.pt"
checkpoint = torch.load(model_path)
new_checkpoint = {}
for k in checkpoint:
    new_k = k.replace("module.", "")
    new_checkpoint[new_k] = checkpoint[k]
model.load_state_dict(new_checkpoint)
model = torch.nn.DataParallel(model).cuda()

# Initialize saliency methods
saliency_methods = {
'smoothgrad': SmoothGrad(model)
}

def compute_saliency_and_save(isbinary):
    for batch_idx, (data, _) in enumerate(sample_loader):
        data = data.to(device).requires_grad_()

        # Compute saliency maps for the input data
        for s in saliency_methods:
            saliency_map = saliency_methods[s].saliency(data)

            # Save saliency maps
            for i in range(data.size(0)):
                filename = save_path + str( (batch_idx+1) * (i+1))
                image = unnormalize(data[i].cpu())
                if isbinary:
                    save_saliency_map_no_image(image, saliency_map[i], filename + '_full_' + s + '.jpg',binaryMask=True, threshold=0.15)
                    save_saliency_map(image, saliency_map[i], filename + '_' + s + '.jpg',binaryMask=True, threshold=0.15)
                else:
                    save_saliency_map_no_image(image, saliency_map[i], filename + '_full_' + s + '.jpg')
                    save_saliency_map(image, saliency_map[i], filename + '_' + s + '.jpg')


if __name__ == "__main__":
    # Create folder to saliency maps
    save_path = PATH + f'results/{args.model}_{"binary" if args.binary else "original"}/'
    create_folder(save_path)
    compute_saliency_and_save(args.binary)
    print('Saliency maps saved.')







