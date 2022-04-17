# Train a new network on a data set with train.py

# Basic usage: python train.py data_directory
# Prints out training loss, validation loss, and validation accuracy as the network trains
# Options:
# Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
# Choose architecture: python train.py data_dir --arch "vgg16"
# Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs20
# Use  GPU for training: python train.py data_dir --gpu

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

parser.add_argument('data_dir',
                    action='store',
                    default="flowers",
                    help='Data directory, default should be <flowers>')

parser.add_argument('--save_dir',
                    action='store',
                    default="~/opt",
                    dest='save_dir',
                    help='Set directory to save checkpoints, default should be <~/opt>')

parser.add_argument('--arch', 
                    action='store',
                    default="vgg16",
                    dest="arch",
                    help='Choose architecture')

parser.add_argument('--learning_rate', 
                    action='store',
                    default=0.001,
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

print('data_dir      = {!r}'.format(cli_args.data_dir))
print('save_dir      = {!r}'.format(cli_args.save_dir))
print('arch          = {!r}'.format(cli_args.arch))
print('learning_rate = {!r}'.format(cli_args.learning_rate))
print('hidden_units  = {!r}'.format(cli_args.hidden_units))
print('epochs        = {!r}'.format(cli_args.epochs))

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
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

# TODO: Load the datasets with ImageFolder
train_datasets = datasets.ImageFolder(
    train_dir, 
    transform=transform
)

valid_datasets = datasets.ImageFolder(
    valid_dir, 
    transform=transform
)

test_datasets = datasets.ImageFolder(
    test_dir, 
    transform=transform
)

# TODO: Using the image datasets and the trainforms, define the dataloaders
train_dataloader = torch.utils.data.DataLoader(
    train_datasets, 
    batch_size=32,
    shuffle=True
)

valid_dataloader = torch.utils.data.DataLoader(
    valid_datasets, 
    batch_size=32, 
    shuffle=True
)

test_dataloader = torch.utils.data.DataLoader(
    test_datasets, 
    batch_size=32, 
    shuffle=True
)

with open('cat_to_name.json', 'r') as f:
    categories_to_name = json.load(f)
    num_of_categories = len(categories_to_name)
    print("Number of Categories: ", num_of_categories)
    print(json.dumps(categories_to_name, sort_keys=True, indent=4))

# load a pre trained model
# vgg variants classifiers have 25088
# as input_features of 1st FC layer
if cli_args.arch == "resnet18":
    model = models.resnet18(pretrained=True)
elif cli_args.arch == "alexnet":
    model = models.alexnet(pretrained=True)
else:
    model = models.vgg16(pretrained=True)
    
print(model)

img_input = 25088
recommended_hidden = cli_args.hidden_units # 1024
output_size=102

# TODO: Build and train your network

# freeze pre-trained parameters
for param in model.parameters():
    param.requires_grad = False

# create custom classifier
recommended_classifier = nn.Sequential(
    OrderedDict(
        [
            # in_feature in fc1 should match in_features
            # from the first FC layer of a pre-trained classifier
            ('fc1', nn.Linear(img_input, recommended_hidden)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout()),

            # in_features of fc2 should
            # match out_feature of fc1, and so on.
            # 
            # out_features on the last layer should
            # match the classes (102) you to want to predict
            ('fc2', nn.Linear(recommended_hidden, output_size)),
            ('output', nn.LogSoftmax(dim=1))
        ]
    )
)

model.classifier = recommended_classifier

print(model)

# Use GPU if it's available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=cli_args.learning_rate)

steps = 0
tot_train_loss = 0
print_every = 5

train_losses, test_losses = [], []

model.train()

for epoch in range(cli_args.epochs):
    
    for inputs, labels in train_dataloader:
        
        steps += 1
        
        # Move input and label tensors to the default device
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()

        # Forward
        log_ps = model.forward(inputs)
        loss = criterion(log_ps, labels)

        # Backward
        loss.backward()
        optimizer.step()

        tot_train_loss += loss.item()

        train_loss = tot_train_loss / len(train_dataloader.dataset)
        train_losses.append(train_loss)

        if steps % print_every == 0:
            print(f"Epoch {epoch+1}/{cli_args.epochs}.. "
                  f"Train loss: {tot_train_loss/print_every:.3f}.. ")
            tot_train_loss = 0.0

print('Finished Training')

# TODO: Do validation on the test set
steps = 0
tot_test_loss = 0
test_correct = 0  # Number of correct predictions on the test set

# Turn off gradients for validation, saves memory and computations
with torch.no_grad():
    for images, labels in test_dataloader:

        steps += 1

        # Move input and label tensors to the default device
        inputs = inputs.to(device)
        labels = labels.to(device)
        images = images.to(device)
                
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        tot_test_loss += loss.item()

        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        test_correct += equals.sum().item()

        # Get mean loss to enable comparison between train and test sets
        # train_loss = tot_train_loss / len(train_dataloader.dataset)
        test_loss = tot_test_loss / len(test_dataloader.dataset)

        # At completion of epoch
        test_losses.append(test_loss)

        print("Epoch: {}/{}.. ".format(1, cli_args.epochs),
                "Test Loss: {:.3f}.. ".format(test_loss),
                "Test Accuracy: {:.3f}".format(test_correct / len(test_dataloader.dataset)))
            
print("Done Testing")

# TODO: Save the checkpoint 
model.class_to_idx = train_datasets.class_to_idx

config_dictionary = {
  'model': model,
  'optimizer_state': optimizer.state_dict,
  'epochs': cli_args.epochs
}

os.mkdir(cli_args.save_dir)

torch.save(config_dictionary, cli_args.save_dir+"/checkpoint.pth")
