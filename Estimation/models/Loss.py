import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class gaussian_kde(nn.Module):
    def __init__(self,h):
        super(gaussian_kde, self).__init__()
        self.h=h
    
    def ker(self,x,X):
        return torch.sum( torch.exp(-(x-X)**2/(2*self.h**2)) )
    
    def forward(self,x,X):
        P=[]
        for i in range(x.shape[0]):
            P.append(self.ker(x[i],X).view(1,-1))
        
        P=torch.cat(P).view(-1)
        P=torch.clamp(P,1e-10)
        P=P/torch.sum(P)
        return P

class KL_loss(nn.Module):
    def __init__(self,h,X_min,X_max,bins):
        super(KL_loss, self).__init__()
        self.X_min=X_min
        self.X_max=X_max
        self.bins=bins
        self.P=gaussian_kde(h)
        self.Q=gaussian_kde(h)
    
    def forward(self,pred,y_true):
        x=torch.linspace(self.X_min,self.X_max,self.bins)
        P=self.P(x,y_true.view(-1))
        Q=self.Q(x,pred.view(-1))
        KL=torch.sum(P*torch.log(P/Q))
        return KL,x,P,Q

class Euclidean_distance(nn.Module):
    def __init__(self):
        super(Euclidean_distance, self).__init__()
    
    def forward(self,pred,y_true):
        return torch.norm(pred-y_true)

class Estimation_Loss(nn.Module):
    def __init__(self,w,h,X_min,X_max,bins):
        super(Estimation_Loss, self).__init__()
        self.kl_loss=KL_loss(h,X_min,X_max,bins)
        self.ed_loss=Euclidean_distance()
        # self.ed_loss=nn.MSELoss()
        self.w=w
    
    def forward(self,pred,y_true):
        kl_loss,x,P,Q=self.kl_loss(pred,y_true)
        ed_loss=self.ed_loss(pred,y_true)
        loss=self.w*kl_loss+ed_loss
        return loss

if __name__ == '__main__':
    # KL=KL_loss(h=0.1,X_min=0,X_max=60,bins=100)
    Loss_func=Estimation_Loss(w=0.1,h=0.1,X_min=0,X_max=60,bins=100)
    pred=nn.Parameter(torch.rand(1024)).cuda()*60
    pred.retain_grad()
    y_true=torch.rand(1024).cuda()*60

    loss=Loss_func(pred,y_true)
    print(loss)