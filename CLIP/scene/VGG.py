import torch.nn as nn
from collections import namedtuple
import torchvision.models as models
import torchvision


def get_features(image, model, layers=None):
    if layers is None:
        layers = {'21': 'conv4_2', '31': 'conv5_2'}
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features