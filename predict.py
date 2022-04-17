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
                    default="error.png",
                    dest='input',
                    required=True,
                    help='Path to image')

parser.add_argument('checkpoint',
                    action='store',
                    default="checkpoint.pth",
                    dest='checkpoint',
                    required=True,
                    help='checkpoint file')

parser.add_argument('--category_names',
                    action='store',
                    default="cat_to_name.json",
                    dest='category_names',
                    help='Mapping of categories to real names')

parser.add_argument('--gpu', 
                    action='store',
                    default=False,
                    dest="gpu"
                    help='Use GPU for inference')

cli_args = parser.parse_args()

print('input          = {!r}'.format(cli_args.input))
print('checkpoint     = {!r}'.format(cli_args.checkpoint))
print('category_names = {!r}'.format(cli_args.category_names))
print('gpu            = {!r}'.format(cli_args.gpu))
