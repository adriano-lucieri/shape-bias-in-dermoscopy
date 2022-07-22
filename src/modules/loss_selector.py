import torch.nn as nn


def loss_selector(name):
    if name == 'CE':
        criterion = nn.CrossEntropyLoss()

    elif name == 'BCE':
        criterion = nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')

    elif name == 'MSE':
        criterion = nn.MSELoss(size_average=None, reduce=None, reduction='mean')

    return criterion