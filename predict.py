# Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

# Basic usage: python predict.py /path/to/image checkpoint
# Options:
# Return top KK most likely classes: python predict.py input checkpoint --top_k 3
# Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
# Use GPU for inference: python predict.py input checkpoint --gpu

import argparse

from collections import OrderedDict

import json

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

import numpy as np

import os

import pickle

from PIL import Image

import time

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.transforms.functional as TF

parser = argparse.ArgumentParser()

parser.add_argument('input',
                    action='store',
                    default="flowers/valid/1/image_06739.jpg",
                    help='Path to image, default should be <flowers/valid/1/image_06739.jpg>')

parser.add_argument('checkpoint',
                    action='store',
                    default="save_dir/checkpoint.pth",
                    help='checkpoint file, default should be <save_dir/checkpoint.pth>')

parser.add_argument('--topk', 
                    action='store',
                    default=5,
                    dest="gpu",
                    help='Use GPU for inference')

parser.add_argument('--category_names',
                    action='store',
                    default="cat_to_name.json",
                    dest='category_names',
                    help='Mapping of categories to real names')

parser.add_argument('--gpu', 
                    action='store',
                    default=False,
                    dest="gpu",
                    help='Use GPU for inference')

cli_args = parser.parse_args()

print('input          = {!r}'.format(cli_args.input))
print('checkpoint     = {!r}'.format(cli_args.checkpoint))
print('top k          = {!r}'.format(cli_args.topk))
print('category_names = {!r}'.format(cli_args.category_names))
print('gpu            = {!r}'.format(cli_args.gpu))

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):

    path = 'checkpoint.pth'
    
    state = torch.load(path)
    
    print(config_dictionary)

    model = config_dictionary["model"]
    optimizer.state_dict = config_dictionary["optimizer_state"]

##### Commented Out to keep workspace size down #####
load_checkpoint(cli_args.checkpoint)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(model)
print(optimizer)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    im = image
    p_im = transform(im)
    np_im = np.array(p_im)

    # PyTorch expects the color channel to be the first dimension 
    # but it's the third dimension in the PIL image and Numpy array
    np_im.transpose((2, 0, 1))
    
    return p_im, np_im


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


def predict(image_path, model, topk=cli_args.topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # Evaluation Mode
    model.eval()
    model = model.to(device)
    
    # TODO: Implement the code to predict the class from an image file
    im = Image.open(image_path)
    p_im, np_im = process_image(im)
    
    # Get torch
    tnp_im = torch.from_numpy(np_im)

    # https://discuss.pytorch.org/t/expected-stride-to-be-a-single-integer-value-or-a-list/17612/4
    tnp_im = tnp_im.unsqueeze(0)
    
    tnp_im = tnp_im.to(device) 
    
    output = model.forward(tnp_im) 

    ps = torch.exp(output)

    probs, classes = ps.topk(topk)
    
    return probs, classes


im = Image.open(cli_args.input)
    
plt.imshow(im)
plt.show()
    
probs, classes = predict(img_path, model)

np_class = classes.cpu().detach().numpy()
classes = np_class.tolist()[0]

np_probs = probs.cpu().detach().numpy()
probs = np_probs.tolist()[0]
    
np_names = np.array([])

y = []
    
for x in range (0, cli_args.topk):
    y = int(np_class[0,x])
    name = [ categories_to_name.get(str(y)) ]
    np_names = np.append(np_names, name)

print("Top k = "+cli_args.topk)

for x in range (0, cli_args.topk)
    print(np_names[x]+": "+probs[x])
