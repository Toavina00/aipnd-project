import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings

from datetime import datetime
from collections import OrderedDict

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets

warnings.simplefilter("ignore")

# Argument parser

parser = argparse.ArgumentParser()

parser.add_argument("data_directory")
parser.add_argument("--save_dir", default="./")
parser.add_argument("--arch", default="vgg11", choices=["vgg11", "vgg13", "vgg16", "vgg19", "densenet121", "densenet161", "densenet169", "densenet201" "alexnet"])
parser.add_argument("--learning_rate", default=0.001, type=float)
parser.add_argument("--hidden_units", default=512, type=int)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--gpu", action="store_true", default=False)

args = parser.parse_args()

# Variables initialization 

save_path = os.path.join(args.save_dir, "model_" + args.arch + datetime.now().strftime("_%Y-%m-%d_%H:%M:%S") + ".pth")

device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")

data_dir = args.data_directory
input_units = 0
output_units = 102
hidden_units = args.hidden_units
model = None
epoch = args.epochs
learning_rate = args.learning_rate

# Model initialization

match args.arch:
    case "vgg11":
        model = models.vgg11(pretrained=True)
        input_units = 25088
    case "vgg13":
        model = models.vgg13(pretrained=True)
        input_units = 25088
    case "vgg16":
        model = models.vgg16(pretrained=True)
        input_units = 25088
    case "vgg19":
        model = models.vgg19(pretrained=True)
        input_units = 25088
    case "densenet121":
        model = models.densenet121(pretrained=True)
        input_units = 1024
    case "densenet161":
        model = models.densenet161(pretrained=True)
        input_units = 2208
    case "densenet169":
        model = models.densenet169(pretrained=True)
        input_units = 1664
    case "densenet201":
        model = models.densenet201(pretrained=True)
        input_units = 1920
    case _:
        model = models.alexnet(pretrained=True)
        input_units = 9216


for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(
    OrderedDict({
        "fc0": nn.Linear(in_features=input_units, out_features=hidden_units, bias=True),
        "relu0": nn.ReLU(),
        "dropout": nn.Dropout(0.4),
        "fc1": nn.Linear(in_features=hidden_units, out_features=output_units, bias=True),
        "output": nn.LogSoftmax(dim=1)
    })
)

model.classifier = classifier
model.to(device)

# Data loading

train_dir = os.path.join(data_dir, 'train')
valid_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'train')

data_transforms = {
    "train": transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.299, 0.224, 0.225])
    ]),
    "valid": transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.299, 0.224, 0.225])
    ]),
    "test": transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.299, 0.224, 0.225])
    ]),
}

image_datasets = {
    "train": datasets.ImageFolder(train_dir, transform=data_transforms["train"]),
    "valid": datasets.ImageFolder(valid_dir, transform=data_transforms["valid"]),
    "test": datasets.ImageFolder(test_dir, transform=data_transforms["test"])
}

dataloaders = {
    "train": DataLoader(image_datasets["train"], batch_size=64, shuffle=True),
    "valid": DataLoader(image_datasets["valid"], batch_size=64, shuffle=True),
    "test": DataLoader(image_datasets["test"], batch_size=64, shuffle=True),
}

# Model training

model.train()

optimizer = optim.Adam(model.classifier.parameters(), learning_rate)
criterion = nn.NLLLoss()

print("----------")
print(" Training ")
print("----------")
print("\n")
print("device:", device)
print("\n")

for e in range(epoch):
    running_loss = 0
    for images, labels in dataloaders["train"]:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        logps = model.forward(images)
        loss = criterion(logps, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
    else:

        validation_loss = 0
        accuracy = 0

        with torch.no_grad():
            model.eval()
            for images, labels in dataloaders["valid"]:
                images, labels = images.to(device), labels.to(device)

                logps = model.forward(images)
                loss = criterion(logps, labels)

                validation_loss += loss.item()
                accuracy += (logps.argmax(dim=1) == labels).type(torch.FloatTensor).mean()


        model.train()

        print(f"Epoch: {e+1}/{epoch}")
        print(f"Train loss: {running_loss/len(dataloaders['train'])}")
        print(f"Validation loss: {validation_loss/len(dataloaders['valid'])}")
        print(f"Validation accuracy: {accuracy/len(dataloaders['valid'])}")


print("\n")
print("----------")
print(" Testing  ")
print("----------")
print("\n")

test_loss = 0
accuracy = 0

with torch.no_grad():
    model.eval()
    for images, labels in dataloaders["test"]:
        images, labels = images.to(device), labels.to(device)

        logps = model.forward(images)
        loss = criterion(logps, labels)
        test_loss += loss.item()

        accuracy += (logps.argmax(dim=1) == labels).type(torch.FloatTensor).mean()

print(f"Test loss: {validation_loss/len(dataloaders['test'])}")
print(f"Test accuracy: {accuracy/len(dataloaders['test'])}")

print("\n")
print("----------")
print("  Saving  ")
print("----------")
print("\n")

checkpoint = {
    "arch": args.arch,
    "input_size": input_units,
    "output_size": output_units,
    "hidden_layers": [hidden_units],
    "state_dict": model.state_dict(),
    "training_state": {
        "epoch": epoch,
        "optimizer_state": optimizer.state_dict()
    },
    "class_to_idx": image_datasets["train"].class_to_idx
}

torch.save(checkpoint, save_path)

print("Model saved as", save_path)