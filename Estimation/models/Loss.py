import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# class gaussian_kde(nn.Module):
#     def __init__(self,h):
#         super(gaussian_kde, self).__init__()
#         self.h=h
    
#     def ker(self,x,X):
#         return torch.sum( torch.exp(-(x-X)**2/(2*self.h**2)) )
    
#     def forward(self,x,X):
#         P=[]
#         for i in range(x.shape[0]):
#             P.append(self.ker(x[i],X).view(1,-1))
        
#         P=torch.cat(P).view(-1)
#         P=torch.clamp(P,1e-10)
#         P=P/torch.sum(P)
#         return P

# class KL_loss(nn.Module):
#     def __init__(self,h,X_min,X_max,bins):
#         super(KL_loss, self).__init__()
#         self.X_min=X_min
#         self.X_max=X_max
#         self.bins=bins
#         self.P=gaussian_kde(h)
#         self.Q=gaussian_kde(h)
    
#     def forward(self,pred,y_true):
#         x=torch.linspace(self.X_min,self.X_max,self.bins)
#         P=self.P(x,y_true.view(-1))
#         Q=self.Q(x,pred.view(-1))
#         KL=torch.sum(P*torch.log(P/Q))
#         return KL,x,P,Q


class GaussianHistogram(nn.Module):
    def __init__(self, bins, X_min, X_max, sigma):
        super(GaussianHistogram, self).__init__()
        self.bins = bins
        self.min = X_min
        self.max = X_max
        self.sigma = sigma
        self.delta = float(X_max - X_min) / float(bins)
        self.centers = float(X_min) + self.delta * (torch.arange(bins).float() + 0.5).cuda()

    def forward(self, x):
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = torch.exp(-0.5*(x/self.sigma)**2) / (self.sigma * np.sqrt(np.pi*2)) * self.delta
        x = x.sum(dim=1)
        x = x/torch.sum(x)
        return x

class KL_loss(nn.Module):
    def __init__(self,X_min,X_max,bins,sigma):
        super(KL_loss, self).__init__()
        self.gausshist = GaussianHistogram(bins=bins, X_min=X_min, X_max=X_max, sigma=sigma)
    
    def forward(self,pred,y_true):
        
        def hook(grad):
            if torch.isnan(grad).any():
                return torch.zeros_like(grad).cuda()
            else:
                return grad
        if pred.requires_grad:
            pred.register_hook(hook)
        
        P=self.gausshist(y_true)
        Q=self.gausshist(pred)
        tmp=P/Q
        mask=torch.isinf(tmp)
        KL=torch.sum((P*torch.log(P/Q))[~mask])
        
        return KL


class Euclidean_distance(nn.Module):
    def __init__(self):
        super(Euclidean_distance, self).__init__()
    
    def forward(self,pred,y_true):
        return torch.norm(pred-y_true)

class Estimation_Loss(nn.Module):
    def __init__(self,w,X_min,X_max,bins,sigma):
        super(Estimation_Loss, self).__init__()
        self.kl_loss=KL_loss(X_min,X_max,bins,sigma)
        self.ed_loss=Euclidean_distance()
        # self.ed_loss=nn.MSELoss()
        self.w=w
    
    def forward(self,pred,y_true):
        kl_loss=self.kl_loss(pred,y_true)
        ed_loss=self.ed_loss(pred,y_true)
        loss = self.w*kl_loss + ed_loss
        return loss


if __name__ == '__main__':
    # KL=KL_loss(h=0.1,X_min=0,X_max=60,bins=100)
    Loss_func=Estimation_Loss(w=0.1,h=0.1,X_min=0,X_max=60,bins=100)
    pred=nn.Parameter(torch.rand(1024)).cuda()*60
    pred.retain_grad()
    y_true=torch.rand(1024).cuda()*60

    loss=Loss_func(pred,y_true)
    print(loss)