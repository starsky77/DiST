import os
import numpy as np
from PIL import Image
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import resnet50,resnet18, ResNet50_Weights,AlexNet_Weights,VGG19_Weights,ResNet18_Weights
from torch.utils import model_zoo
from torch.hub import load_state_dict_from_url

from scipy.spatial import distance
from scipy.special import softmax

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import argparse
from tqdm import tqdm

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


def pairwise_cosine_distance(features):
    n = len(features)
    cosine_distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = distance.cosine(features[i], features[j])
            cosine_distances[i, j] = dist
            cosine_distances[j, i] = dist
    return cosine_distances

def mean_cosine_distance(cosine_distances):
    n = len(cosine_distances)
    mean_distances = np.zeros(n)
    for i in range(n):
        mean_distances[i] = np.mean(cosine_distances[i, np.arange(n) != i])
    return mean_distances

def load_model_ffcv(model_path, is2K):
    model = models.resnet50(weights=False)
    for (name, child) in model.named_children():
        if isinstance(child, nn.Conv2d) and (np.max(child.stride) > 1 and child.in_channels >= 16):
            setattr(model, name, BlurPoolConv2d(child))
        else: apply_blurpool(child)
    if is2K:
        num_ftrs = model.fc.in_features
        num_classes = 2000
        model.fc = nn.Linear(num_ftrs, num_classes)
    checkpoint = torch.load(model_path)
    new_checkpoint = {}
    for k in checkpoint:
        new_k = k.replace("module.", "")
        new_checkpoint[new_k] = checkpoint[k]
    model.load_state_dict(new_checkpoint)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model = torch.nn.DataParallel(model).cuda()
    return model

parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', default='resnet50', type=str)
parser.add_argument('--model-path', default='', type=str)
parser.add_argument('--is2K', default=0, type=int)
args = parser.parse_args()


if args.model=="vit":
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.head = torch.nn.Identity()
    model = torch.nn.DataParallel(model).cuda()
elif args.model=="vit_small":
    model = timm.create_model('vit_small_patch16_224', pretrained=True)
    model.head = torch.nn.Identity()
    model = torch.nn.DataParallel(model).cuda()
elif args.model=="vit_large":
    model = timm.create_model('vit_large_patch16_224', pretrained=True)
    model.head = torch.nn.Identity()
    model = torch.nn.DataParallel(model).cuda()
elif args.model=="resnet50":
    weights = ResNet50_Weights.IMAGENET1K_V1 
    model = resnet50(weights=weights)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model = torch.nn.DataParallel(model).cuda()
elif args.model=="vgg19":
    model = models.vgg19(weights=VGG19_Weights.DEFAULT)
    model.classifier = torch.nn.Identity()
    model = torch.nn.DataParallel(model).cuda()
elif args.model=="alexnet":
    model = models.alexnet(weights=AlexNet_Weights.DEFAULT)
    model.classifier = torch.nn.Identity()
    model = torch.nn.DataParallel(model).cuda()
elif args.model=="resnet50_SIN":
    resnet50_trained_on_SIN='https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar'
    model = models.resnet50(weights=None)
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = model_zoo.load_url(resnet50_trained_on_SIN)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.module
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model = torch.nn.DataParallel(model).cuda()
elif args.model=="resnet50_DiSTinguish":
    model=load_model_ffcv("../model/DiSTinguish/weights_ep_89.pt", is2K = 1)
elif args.model=="resnet50_DiSTinguish_Sty_aug":
    model=load_model_ffcv("../model/DiSTinguish_Style_aug/weights_ep_89.pt", is2K = 1)
elif args.model=="resnet50_Sty_aug":
    model=load_model_ffcv("../model/Style_aug/weights_ep_89.pt", is2K = 0)

model.eval()
transform_texform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # This line ensures all images have 3 channels
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


main_folders = ['../data/original', '../data/set1', '../data/set2']
image_extensions = ['.JPEG', '.jpg', '.jpg']
sub_folders = os.listdir(main_folders[0])


correctCount = 0
total = 0
failed_images=[]
image_files = os.listdir(main_folders[1])
pbar = tqdm(image_files, total=len(image_files))


all_features=[]
for image_file in pbar:
    features = []
    original_image_name=""
    for main_folder, image_extension in zip(main_folders, image_extensions):
        image_path = os.path.join(main_folder, image_file.replace('.jpg', image_extension))
        if image_extension == ".JPEG":
            original_image_name=image_path

        image = transform_texform(Image.open(image_path)).unsqueeze(0)

        with torch.no_grad():
            feature = model(image).cpu().numpy().flatten()

        mean = feature.mean()
        std = feature.std()
        feature_normalized = (feature - mean) / std
        features.append(feature_normalized)
        if len(all_features)==0:
            all_features=feature
        else:
            all_features = np.vstack([all_features, feature])

    
    # Calculate the cosine distance
    cosine_distances = pairwise_cosine_distance(features)
    mean_distances = mean_cosine_distance(cosine_distances)
    probabilities = softmax(mean_distances)
    prediction = np.argmax(probabilities)
    if prediction==0:
        correctCount+=1
    else:
        failed_images.append(original_image_name)
    total+=1

n_pairs = len(image_files)  
n_classes = 2  
labels = np.array([c for _ in range(n_pairs) for c in [0] + [1]*2])
tsne = TSNE(n_components=2, random_state=0)
features_2d = tsne.fit_transform(all_features)

plt.figure(figsize=(8, 6))
for i, label in enumerate(set(labels)):
    idx = labels == label
    if label==0:
        plt_label="Original Shape"
    else:
        plt_label=f"Distorted Shape"
    plt.scatter(features_2d[idx, 0], features_2d[idx, 1], label=plt_label, alpha=0.25)

plt.legend(fontsize='large')
plt.title("t-SNE Visualization of Features")
plot_path = f'../plot/tsne_{args.model}.png'
plt.savefig(plot_path)


acc_texform=correctCount/total*1.0
print(acc_texform)