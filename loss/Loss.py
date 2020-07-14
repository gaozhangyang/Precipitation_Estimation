import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class KL_loss(nn.Module):
    def __init__(self,w=1000,reduction='mean'):
        super(KL_loss, self).__init__()
        self.reduction=reduction
        self.w=w
        
    
    def forward(self,P,Q):#P:target Q:pred
        P=P.view(P.shape[0],-1)
        Q=Q.view(Q.shape[0],-1)
        L=P*(torch.log(P)-torch.log(Q))
        if self.reduction=='mean':
            # KL=torch.mean(torch.mean(L,dim=1))
            KL=F.kl_div(Q,P,reduction='batchmean')
        
        if self.reduction=='sum':
            # KL=torch.mean(torch.sum(L,dim=1))
            KL=F.kl_div(Q,P,reduction='sum')

        D=torch.mean(F.pairwise_distance(P,Q,p=2))
        loss=KL*self.w+D
        return loss