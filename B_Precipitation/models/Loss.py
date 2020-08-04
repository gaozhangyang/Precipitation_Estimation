import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from pathlib import Path

toCPU=lambda x: x.detach().cpu().numpy()
toCUDA=lambda x: torch.tensor(x).cuda()

class gaussian_kde(nn.Module):
    def __init__(self,X_min,X_max,bins,sigma):
        super(gaussian_kde, self).__init__()
        self.X_min=X_min
        self.X_max=X_max
        self.bins=bins
        self.sigma=sigma
        self.delta = float(X_max - X_min) / float(bins)
        self.centers = float(X_min) + self.delta * (torch.arange(bins).float() + 0.5).cuda()
    
    def forward(self,x):
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = torch.exp(-0.5*(x/self.sigma)**2) / (self.sigma * np.sqrt(np.pi*2)*self.centers.shape[0])
        x = x.sum(dim=1)
        return x

epsilon=1e-10
class KL_loss(nn.Module):
    def __init__(self):
        super(KL_loss,self).__init__()
    
    def forward(self,P,Q):
        P=P+epsilon
        Q=Q+epsilon
        KL = (P*torch.log(P)-P*torch.log(Q))
        KL = torch.sum(KL)
        return KL


class Euclidean_distance(nn.Module):
    def __init__(self):
        super(Euclidean_distance, self).__init__()
    
    def forward(self,pred,y_true,weight):
        return torch.sqrt(torch.sum(weight*(pred-y_true)**2))

class Estimation_Loss(nn.Module):
    def __init__(self,X_min,X_max,bins,sigma,mse=False):
        super(Estimation_Loss, self).__init__()
        if mse:
            self.ed_loss=nn.MSELoss()
        else:
            self.ed_loss=Euclidean_distance()

        self.KL = KL_loss()
        self.KDE=gaussian_kde(X_min=X_min,X_max=X_max,bins=bins,sigma=sigma)
    
    def forward(self,pred,y_true,weight):
        self.P = self.KDE(y_true)
        self.Q = self.KDE(pred)

        # def hook(grad):
        #     print('Q {}'.format(grad))
        #     return grad
        # self.Q.register_hook(hook)

        kl_loss = self.KL(self.P,self.Q)
        ed_loss = self.ed_loss(pred,y_true,weight)
        return kl_loss, ed_loss

class Huber_Loss(nn.Module):
    def __init__(self,delta) -> None:
        super(Huber_Loss,self).__init__()
        self.delta=delta
    
    def forward(self,pred,true):
        mask = (torch.abs(pred-true)<=self.delta).float()
        # loss = mask*0.5*(true-pred)**2 \
        #        + (1-mask)*(self.delta*torch.abs(pred-true)-0.5*self.delta**2)
        loss = mask*0.5*(true-pred)**2 \
               + (1-mask)*(torch.abs(pred-true)+0.5*self.delta**2-self.delta)
        loss = torch.mean(loss)
        return loss


OneHot=lambda label,C: torch.zeros(label.shape[0],C).cuda().scatter_(1,label.view(-1,1),1)
class SoftmaxLoss(nn.Module):
    def __init__(self):
        super(SoftmaxLoss,self).__init__()

    def forward(self,score,label,sample_weights):
        onehot_label = OneHot(label,2)
        logsoftmax=F.log_softmax(score)
        loss=torch.mean(-torch.sum(onehot_label*logsoftmax,dim=1)*sample_weights)
        return loss



if __name__ == '__main__':
    # KL=KL_loss(h=0.1,X_min=0,X_max=60,bins=100)
    Loss_func=Estimation_Loss(w=0.1,h=0.1,X_min=0,X_max=60,bins=100)
    pred=nn.Parameter(torch.rand(1024)).cuda()*60
    pred.retain_grad()
    y_true=torch.rand(1024).cuda()*60

    loss=Loss_func(pred,y_true)
    print(loss)