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
                    default="save_dir",
                    help='checkpoint file, default should be <save_dir>')

parser.add_argument('--topk', 
                    action='store',
                    default=5,
                    dest="topk",
                    help='Use GPU for inference')

parser.add_argument('--category_names',
                    action='store',
                    default="cat_to_name.json",
                    dest='category_names',
                    help='Mapping of categories to real names')

parser.add_argument('--gpu', 
                    action='store_true',
                    default=False,
                    dest="gpu",
                    help='Use GPU for inference')

cli_args = parser.parse_args()

print('input          = {!r}'.format(cli_args.input))
print('checkpoint     = {!r}'.format(cli_args.checkpoint))
print('top k          = {!r}'.format(cli_args.topk))
print('category_names = {!r}'.format(cli_args.category_names))
print('gpu            = {!r}'.format(cli_args.gpu))

model = models.vgg16(pretrained=True)
optimizer = {}
epochs = 0

categories_to_name = {}

transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],                            
            [0.229, 0.224, 0.225]
        )
    ]
)

with open('cat_to_name.json', 'r') as f:
    categories_to_name = json.load(f)
    num_of_categories = len(categories_to_name)
    print("Number of Categories: ", num_of_categories)
    print(json.dumps(categories_to_name, sort_keys=True, indent=4))
    

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    
    config_dictionary = torch.load(filepath)
    
    print(config_dictionary)
    
    model.state_dict = config_dictionary["model_state"]
    model.eval()
    
    optimizer_state = config_dictionary["optimizer_state"]
    learning_rate = config_dictionary["learning_rate"]
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    # optimizer.state_dict = config_dictionary["optimizer_state"]
    epochs = config_dictionary["epochs"]

    model.class_to_idx = config_dictionary['class_to_idx'] 
   
##### Commented Out to keep workspace size down #####
load_checkpoint(cli_args.checkpoint+"/checkpoint.pth")

device = "cpu"

if cli_args.gpu == True:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(model)
print(optimizer)
print(epochs)
    
model = model.to(device)

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
    
probs, classes = predict(cli_args.input, model)

np_class = classes.cpu().detach().numpy()
print(np_class)

classes = np_class.tolist()[0]
print(classes)

np_probs = probs.cpu().detach().numpy()
probs = np_probs.tolist()[0]
    

y = []

print(str('Top k = ')+str(cli_args.topk))

for x in range (0, cli_args.topk):
    y = int(np_class[0,x])
    name = categories_to_name[str(y)]

    print(str(name)+": "+str(probs[x]))
