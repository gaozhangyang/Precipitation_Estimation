import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
def SetSeed(seed):
    """function used to set a random seed
    Arguments:
        seed {int} -- seed number, will set to torch, random and numpy
    """
    SEED = seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
SetSeed(2020)

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
        # 计算self.centers到每一个x的距离
        # x是网络的输出，或ground truth降雨
        # 输出每一个center处的概率密度
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = torch.exp(-0.5*(x/self.sigma)**2) / (self.sigma * np.sqrt(np.pi*2)*self.bins)#TODO 除以样本数量而不是bins
        x = x.sum(dim=1)
        return x

# class KL_loss(nn.Module):
#     def __init__(self,X_min,X_max,bins,sigma):
#         super(KL_loss, self).__init__()
#         self.KDE=gaussian_kde(X_min,X_max,bins,sigma)
    
#     def forward(self,pred,y_true):
#         P=self.KDE(y_true)# ground truth下的概率密度
#         Q=self.KDE(pred)  # 模型输出下的概率密度
#         Q.retain_grad()
#         KL=torch.sum( (P*torch.log(P)-P*torch.log(Q)) )
#         return KL,self.KDE.centers,P,Q

class KL_loss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, P, Q):
        ctx.save_for_backward(P,Q)
        KL=torch.sum( (P*torch.log(P)-P*torch.log(Q)) )
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


KL = KL_loss.apply
KDE=gaussian_kde(X_min=-10,X_max=60,bins=70,sigma=0.5)

##准备数据
pred=nn.Parameter(torch.rand(1024)).cuda()*60
pred.retain_grad()
y_true=torch.rand(1024).cuda()*60

## 手动更新
for epoch in range(100):
    P=KDE(y_true)
    Q=KDE(pred)
    Q.retain_grad()

    kl=KL(P,Q)
    kl.backward(retain_graph=True)
    if torch.isnan(pred.grad).any():
        print()
    pred.data=pred.data-1000*pred.grad
    pred.grad.zero_()
    