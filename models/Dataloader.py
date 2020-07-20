
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
    def __init__(self,X,Y,task='identification',shuffle=False,seed=2020,win_size=14):
        self.X=X
        self.Y=Y
        self.win_size=win_size
        self.task=task
        self.shuffle=shuffle
        
    
    def split_R_NR(self,StageIV):
        R_samples=[]
        NR_samples=[]
        for T in range(StageIV.shape[0]):
            for row in range(self.win_size,StageIV.shape[1]-self.win_size-1,self.win_size):
                for col in range(self.win_size,StageIV.shape[2]-self.win_size-1,self.win_size):
                    if StageIV[T,row,col]>0.1:
                        R_samples.append((T,row,col))
                    else:
                        NR_samples.append((T,row,col))

                        
        R_samples=np.array(R_samples)
        NR_samples=np.array(NR_samples)
        return R_samples,NR_samples
    
    def split_dataset(self,sp1=0.6,sp2=0.8):
        self.R_samples,self.NR_samples=self.split_R_NR(self.Y)
        
        if self.task=='identification':
            self.samples=np.vstack([np.array(random.choices(self.R_samples,k=340000)),
                                    np.array(random.choices(self.NR_samples,k=340000))])
        
        if self.task=='estimation':
            self.samples=np.array(random.choices(self.R_samples,k=470000))
        
        
        if self.shuffle:
            np.random.shuffle(self.samples)
        L=len(self.samples)
        
        
        self.train_sample_idx=range(0,int(L*sp1))
        self.test_sample_idx=range(int(L*sp1),int(L*sp2))
        self.val_sample_idx=range(int(L*sp2),int(L*1))
        
        return self.samples, self.train_sample_idx, self.test_sample_idx, self.val_sample_idx


class IRDataset(Dataset):
    def __init__(self,samples,X,Y,win_size=14,seed=2020):
        self.X=X
        self.Y=Y
        self.win_size=win_size
        self.samples=samples
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
        Y_croped=self.Y[T,row,col]
        return X_croped,Y_croped,T,row,col


    def __len__(self):
        return self.L

    def name(self):
        return 'IRDataset'



class CustomDatasetDataLoader(object):
    def __init__(self, X, Y, batchSize,selected_samples,win_size, nThreads=8,seed=2020):
        self.dataset = IRDataset(selected_samples,X=X,Y=Y,win_size=win_size,seed=seed)
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