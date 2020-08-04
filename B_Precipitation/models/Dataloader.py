
import numpy as np
import random
import torch

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


from torch.utils.data import Dataset
from datetime import date
from datetime import timedelta
import os


class IR_Split:
    def __init__(self,X,Y,task='identification',seed=2020,win_size=14,sampling_step=7,R_w=1,NR_w=1,evaluate=False):
        self.X=X
        self.Y=Y
        self.win_size=win_size
        self.task=task
        self.R_w=R_w
        self.NR_w=NR_w
        self.evaluate=evaluate
        self.sampling_step=sampling_step
        
    
    def split_R_NR(self,StageIV):
        R_samples=[]
        NR_samples=[]
        for T in range(StageIV.shape[0]):
            for row in range(self.win_size,StageIV.shape[1]-self.win_size-1,self.sampling_step):
                for col in range(self.win_size,StageIV.shape[2]-self.win_size-1,self.sampling_step):
                    if StageIV[T,row,col]>0.1:
                        R_samples.append((T,row,col))
                    else:
                        NR_samples.append((T,row,col))

                        
        R_samples=np.array(R_samples)
        NR_samples=np.array(NR_samples)
        return R_samples,NR_samples
    
    def split_dataset(self,sp1=0.6,sp2=0.8):
        self.R_samples,self.NR_samples=self.split_R_NR(self.Y)
        N = self.R_samples.shape[0]
        
        if self.task=='identification':
            self.samples=np.vstack([random.choices(self.NR_samples,k=N),self.R_samples])
            self.weights=np.ones(self.samples.shape[0])
            self.weights[:N]*=self.NR_w
            self.weights[N:]*=self.R_w
        
        if self.task=='estimation':
            self.samples=np.vstack([random.choices(self.NR_samples,k=N),self.R_samples])
            self.weights=np.ones(self.samples.shape[0])
            self.weights[:N]*=self.NR_w
            self.weights[N:]*=self.R_w
        
        L=len(self.samples)
        
        return self.samples, self.weights


class IRDataset(Dataset):
    def __init__(self,samples,weights,X,Y,win_size=14,seed=2020):
        self.X=X
        self.Y=Y
        self.win_size=win_size
        self.samples=samples
        self.weights=weights
        self.L=len(self.samples)

    @classmethod
    def safe_crop_center(self,img,x,y,cropx,cropy):
        startx = x-(cropx)
        endx=x+(cropx)+1
        starty = y-(cropy)   
        endy= y+(cropy)+1  
        
        if len(img.shape)==3:
            _,H,W=img.shape
            if startx<0 or starty<0 or endx>=H or endy>=H:
                return None
            return img[:,startx:endx,starty:endy]
            
        if len(img.shape)==2:
            H,W=img.shape
            if startx<0 or starty<0 or endx>=H or endy>=H:
                return None
            return img[startx:endx,starty:endy]
    
    @classmethod
    def unsafe_crop_center(self,img,x,y,cropx,cropy):
        startx = x-(cropx)
        endx=x+(cropx)+1
        starty = y-(cropy)   
        endy= y+(cropy)+1
        if len(img.shape)==2:
            return img[startx:endx,starty:endy]
        
        if len(img.shape)==3:
            return img[:,startx:endx,starty:endy]
        

    def __getitem__(self, idx):
        T,row,col=self.samples[idx]
        X_croped=self.unsafe_crop_center(self.X[T],row,col,self.win_size,self.win_size)
        # Y_croped=self.Y[T,row,col]
        Y_croped=self.unsafe_crop_center(self.Y[T],row,col,self.win_size,self.win_size)
        return X_croped,Y_croped,self.weights[idx],T,row,col, self.X[T], self.Y[T]


    def __len__(self):
        return self.L

    def name(self):
        return 'IRDataset'



class CustomDatasetDataLoader(object):
    def __init__(self, X, Y, weights, batchSize,selected_samples,win_size, nThreads=8,seed=2020):
        self.dataset = IRDataset(selected_samples,weights,X=X,Y=Y,win_size=win_size,seed=seed)
        self.batchSize = batchSize

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batchSize,
            shuffle=True,
            num_workers=int(nThreads),
            drop_last=False)
        

    def __iter__(self):
        return self.dataloader.__iter__()

    def __len__(self):
        return self.dataloader.__len__()

    def name(self):
        return 'CustomDatasetDataLoader'


# if __name__ =='__main__':
#     IRS=IR_Split(task='identification',shuffle=True,win_size=14)
#     samples, train_sample_idx, test_sample_idx, val_sample_idx = IRS.split_dataset()
#     dataloader=CustomDatasetDataLoader(1024,samples[train_sample_idx],win_size=14)
#     print()