import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings
import json
import PIL

from datetime import datetime
from collections import OrderedDict

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets

warnings.simplefilter("ignore")

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
    ax.set_title(title)
    ax.axis('off')

    return ax

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    img = PIL.Image.open(image)
    img.thumbnail((256, 256))

    x0, y0 = img.width // 2, img.height // 2
    left, upper = x0 - 112, y0 - 112
    right, lower = x0 + 112, y0 + 112

    img = img.crop((left, upper, right, lower))

    np_image = np.array(img).astype(np.float32) / 255.0
    np_image -= np.array([0.485, 0.456, 0.406])
    np_image /= np.array([0.229, 0.224, 0.225])

    return np_image.transpose((2, 0, 1))

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    model.eval()

    img = process_image(image_path)
    image = torch.tensor(img)
    image = image.to(device)
    image = image.unsqueeze(0)
    logps = model.forward(image)
    prob = torch.exp(logps)

    top_prob, top_class = prob.topk(topk)
    top_prob = top_prob.flatten().tolist()
    top_class = top_class.flatten().tolist()

    idx_to_class = {idx: clss for clss, idx in model.class_to_idx.items()}
    top_class = [idx_to_class[idx] for idx in top_class]

    return top_prob, top_class

def display_prediction(image_path, model, cat_to_name):
    ps, cl = predict(image_path, model)
    image = process_image(image_path)
    image = torch.tensor(image)

    fig, (ax1, ax2) = plt.subplots(figsize=(5,10), nrows=2)

    title = "Image"
    imshow(image, ax=ax1, title=title)

    ax2.barh([cat_to_name[c] for c in cl] if cat_to_name else cl, ps)
    ax2.set_title("Prediction")
    ax2.set_xlabel("Probability")
    ax2.set_ylabel("Categories")
    ax2.invert_yaxis()

# Argument parser

parser = argparse.ArgumentParser()

parser.add_argument("input")
parser.add_argument("model")
parser.add_argument("--top_k", default=5, type=int)
parser.add_argument("--category_names")

args = parser.parse_args()

# Variables initialization

device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")

input_path = args.input
model_path = args.model
top_k = args.top_k

cat_to_name = None

if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

# Model loading

print("----------")
print(" Loading  ")
print("----------")
print("\n")

checkpoint = torch.load(model_path)

classifier = nn.Sequential(
    OrderedDict({
        "fc0": nn.Linear(checkpoint["input_size"], checkpoint["hidden_layers"][0], bias=True),
        "relu0": nn.ReLU(),
        "dropout0": nn.Dropout(0.4),
        "fc1": nn.Linear(checkpoint["hidden_layers"][0], checkpoint["output_size"], bias=True),
        "output": nn.LogSoftmax(dim=1)
    })
)

model = None

match checkpoint["arch"]:
    case "vgg11":
        model = models.vgg11(pretrained=True)
    case "vgg13":
        model = models.vgg13(pretrained=True)
    case "vgg16":
        model = models.vgg16(pretrained=True)
    case "vgg19":
        model = models.vgg19(pretrained=True)
    case "densenet121":
        model = models.densenet121(pretrained=True)
    case "densenet161":
        model = models.densenet161(pretrained=True)
    case "densenet169":
        model = models.densenet169(pretrained=True)
    case "densenet201":
        model = models.densenet201(pretrained=True)
    case _:
        model = models.alexnet(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.classifier = classifier

model.load_state_dict(checkpoint["state_dict"])
model.class_to_idx = checkpoint["class_to_idx"]

model.to(device)

print("Model loaded\n")

# Prediction

print("----------")
print("Prediction")
print("----------")

display_prediction(input_path, model, cat_to_name)