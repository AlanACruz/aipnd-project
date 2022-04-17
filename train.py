# Train a new network on a data set with train.py

# Basic usage: python train.py data_directory
# Prints out training loss, validation loss, and validation accuracy as the network trains
# Options:
# Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
# Choose architecture: python train.py data_dir --arch "vgg13"
# Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
# Use GPU for training: python train.py data_dir --gpu

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

parser.add_argument('data_dir',
                    action='store',
                    default="data_dir",
                    dest='data_dir',
                    required=True,
                    help='Data directory')

parser.add_argument('--save_dir',
                    action='store',
                    default="~/opt",
                    dest='save_dir',
                    help='Set directory to save checkpoints')

parser.add_argument('--arch', 
                    action='store',
                    default="vgg13",
                    dest="arch",
                    help='Choose architecture')

parser.add_argument('--learning_rate', 
                    action='store',
                    default=0.01,
                    dest='learning_rate',
                    help='hyperparameter')

parser.add_argument('--hidden_units', 
                    action='store',
                    default=512,
                    dest='hidden_units',
                    help='hyperparameter')

parser.add_argument('--epochs', 
                    action='store',
                    default=1,
                    dest='epochs',
                    help='hyperparameter')

cli_args = parser.parse_args()

print('data_dir      = {!r}'.format(results.data_dir))
print('save_dir      = {!r}'.format(results.save_dir))
print('arch          = {!r}'.format(results.arch))
print('learning_rate = {!r}'.format(results.learning_rate))
print('hidden_units  = {!r}'.format(results.hidden_units))
print('epochs        = {!r}'.format(results.epochs))
