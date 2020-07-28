import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


# class KL_loss(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, P, Q):
#         ctx.save_for_backward(P,Q)
#         KL = (P*torch.log(P)-P*torch.log(Q))
#         mask1=torch.isinf(KL)
#         mask2=torch.isnan(KL)
#         mask=mask1|mask2
#         KL = torch.sum( KL[~mask] )
#         return KL

#     @staticmethod
#     def backward(ctx, grad_output):
#         P,Q = ctx.saved_tensors
#         grad=-P/Q
#         mask1=torch.isinf(grad)
#         mask2=torch.isnan(grad)
#         mask=mask1|mask2
#         grad[mask]=0
#         return None, grad


epsilon=1e-10
class KL_loss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, P, Q):
        Q=Q+epsilon
        ctx.save_for_backward(P,Q)
        KL = (P*torch.log(P)-P*torch.log(Q))
        mask1=torch.isinf(KL)
        mask2=torch.isnan(KL)
        mask=mask1|mask2
        KL = torch.sum( KL[~mask] )
        return KL

    @staticmethod
    def backward(ctx, grad_output):
        P,Q = ctx.saved_tensors
        grad=-P/Q
        mask1=torch.isinf(grad)
        mask2=torch.isnan(grad)
        mask=mask1|mask2
        grad[mask]=0
        return None, grad


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
        self.KL = KL_loss.apply
        self.KDE=gaussian_kde(X_min=X_min,X_max=X_max,bins=bins,sigma=sigma)
    
    def forward(self,pred,y_true):
        P=self.KDE(y_true)
        Q=self.KDE(pred)
        kl_loss=self.KL(P,Q)
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