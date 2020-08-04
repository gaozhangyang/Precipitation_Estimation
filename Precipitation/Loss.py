import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from pathlib import Path

toCPU = lambda x: x.detach().cpu().numpy()
toCUDA = lambda x: torch.tensor(x).cuda()


class gaussian_kde(nn.Module):
    def __init__(self, X_min, X_max, bins, sigma):
        super(gaussian_kde, self).__init__()
        self.X_min = X_min
        self.X_max = X_max
        self.bins = bins
        self.sigma = sigma
        self.delta = float(X_max - X_min) / float(bins)
        self.centers = float(X_min) \
                       + self.delta * (torch.arange(bins).float() + 0.5).cuda()

    def forward(self, x):
        '''
        get the density of #self.centers suppoted by points #x
        '''
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = torch.exp(-0.5 * (x / self.sigma)**2) \
                      / (self.sigma * np.sqrt(np.pi * 2) * self.centers.shape[0])
        x = x.sum(dim=1)
        return x


epsilon = 1e-10
class KL_loss(nn.Module):
    def __init__(self):
        super(KL_loss, self).__init__()

    def forward(self, P, Q):
        '''
        get the KL of two density series #P and #Q
        '''
        P = P + epsilon
        Q = Q + epsilon
        KL = (P * torch.log(P) - P * torch.log(Q))
        KL = torch.sum(KL)
        return KL


class Euclidean_distance(nn.Module):
    def __init__(self):
        super(Euclidean_distance, self).__init__()

    def forward(self, pred, y_true):
        return torch.norm(pred - y_true)


class Huber_Loss(nn.Module):
    def __init__(self, delta):
        super(Huber_Loss, self).__init__()
        self.delta = delta

    def forward(self, pred, true):
        mask = (torch.abs(pred - true) <= self.delta).float()
        loss = mask*0.5*(true-pred)**2 \
               + (1-mask)*(self.delta*torch.abs(pred-true)-0.5*self.delta**2)
        loss = torch.mean(loss)
        return loss


class Estimation_Loss(nn.Module):
    def __init__(self, X_min, X_max, bins, sigma, delta=None, mse=False):
        super(Estimation_Loss, self).__init__()
        if mse:
            self.ed_loss = nn.MSELoss()
        else:
            self.ed_loss = Euclidean_distance()

        if delta is not None:
            self.ed_loss = Huber_Loss(delta)

        self.KL = KL_loss()
        self.KDE = gaussian_kde(X_min=X_min,
                                X_max=X_max,
                                bins=bins,
                                sigma=sigma)

    def forward(self, pred, y_true):
        '''
        get the KL loss and ed loss of #true precipitation and #pred precipitation
        '''
        self.P = self.KDE(y_true)
        self.Q = self.KDE(pred)

        kl_loss = self.KL(self.P, self.Q)
        ed_loss = self.ed_loss(pred, y_true)

        return kl_loss, ed_loss