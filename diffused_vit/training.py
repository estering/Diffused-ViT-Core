import math
import random

import numpy as np
import torch
from torch import nn

from .model import INFERENCE_XI


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name="auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def run_epoch(model, loader, device, epoch, epochs, diffusion_weight, optimizer=None):
    training = optimizer is not None
    model.train(training)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    weight = diffusion_weight * (1.0 + math.cos(epoch * math.pi / epochs))
    total = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().to(device)
        with torch.set_grad_enabled(training):
            output = model(images)
            classification_loss = bce(output.logits, labels)
            reconstruction_loss = mse(
                output.refined_feature, output.clean_feature.detach()
            )
            loss = classification_loss + weight * reconstruction_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total += loss.item() * images.shape[0]

    return total / len(loader.dataset)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    labels, scores = [], []
    for images, target in loader:
        output = model(images.to(device))
        labels.extend(target.numpy())
        scores.extend(torch.sigmoid(output.logits / INFERENCE_XI).cpu().numpy())
    return np.asarray(labels), np.asarray(scores)
