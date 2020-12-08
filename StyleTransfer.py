import torch
import torch.nn.functional as F
from torch import optim,nn
from torchvision import models,transforms,datasets
import requests
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time


class NeuralStyleTransfer:
    def __init__(self, content_image, style_image):
        self.model = models.vgg19(pretrained=True).features
        for pram in self.model.parameters():
            pram.requires_grad(False)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.content = self.load_image(content_image).to(self.device)
        self.target = self.content.clone().requires_grad(True).to(self.device)
        self.style = self.load_image(style_image).to(self.device)

    def load_image(self, path, max_size=400, shape=None, gray=False):
        image = Image.open(path).convert('RGB')

        if shape is not None:
            size = shape
        elif max(image.size) > max_size:
            size = max_size
        else:
            size = max(image.size)

        if gray:
            image = image.convert('L')
            image = image.conver('RGB')
        in_transform = transforms.Compose([transforms.Resize(size),
                                           transforms.ToTensor(),
                                           transforms.Normalize(
                                               (0.485, 0.456, 0.406),
                                               (0.229, 0.224, 0.225))])
        return in_transform(image).unsqueeze(0)

    def get_features(self, model, image):
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',
            '28': 'conv5_1'
        }

        features = {}

        x = image
        for name, layer in self.model._modules.items():
            x=layer(x)
            if name in layers:
                features[layers[name]]=x
        return features

    def gramian(tensor):
        # Compute the gramian matrix of a single channel from a single conv layer.

        t = tensor.view(tensor.shape[1], -1)
        return t @ t.T

    def content_loss(c_features, t_features):
        # Compute mean squared content loss of all feature maps.

        loss = 0.5 * (t_features['conv4_2'] - c_features['conv4_2']) ** 2
        return torch.mean(loss)

    def style_loss(self, s_grams, t_features, weights):

        # Compute style loss, i.e. the weighted sum of MSE of all layers.
        # for each style feature, get target and style gramians, compare
        loss = 0
        for layer in weights:
            _, d, h, w = s_features[layer].shape
            t_gram = gramian(t_features[layer])

            layer_loss = torch.mean((t_gram - s_grams[layer]) ** 2) / (d * h * w)
            loss += layer_loss * weights[layer]

        return loss

    def im_convert(tensor):
        # Display a tensor as an image.
        image = tensor.to("cpu").clone().detach()
        image = image.numpy().squeeze(0)
        image = image.transpose(1, 2, 0)
        image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
        image = image.clip(0, 1)

        return image

