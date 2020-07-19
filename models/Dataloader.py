from torch.utils.data import Dataset
from datetime import date
from datetime import timedelta
import numpy as np
import random
import torch

from torch.utils.data import Dataset
from datetime import date
from datetime import timedelta
import numpy as np
import random
import os

root_path='/usr/commondata/weather/IR_data/IR_dataset_QingHua/'
GOSE=np.load(root_path+'X_train_hourly.npz')['arr_0']
StageIV=np.load(root_path+'Y_train_hourly.npz')['arr_0']


class IRDataset(Dataset):
    def __init__(self,task='identification',mode='train',balance=True):
        self.X=GOSE
        self.Y=StageIV
        
        self.R_samples=np.load('/usr/commondata/weather/New/WCE/R_samples_toy.npy')
        self.NR_samples=np.load('/usr/commondata/weather/New/WCE/NR_samples_toy.npy')
        
        
        if task=='identification':
            R_samples=np.array(random.choices(self.R_samples,k=3400))
            NR_samples=np.array(random.choices(self.NR_samples,k=3400))
        
        if task=='estimation':
            R_samples=np.array(random.choices(self.R_samples,k=3400))
            NR_samples=np.array(random.choices(self.NR_samples,k=3400))
        
        self.samples=np.vstack([R_samples,NR_samples])
        np.random.shuffle(self.samples)
        L=len(self.samples)
        
        
        self.mode=mode
        if mode=='train':
            self.sample_idx=range(0,int(L*0.6))
        
        if mode=='test':
            self.sample_idx=range(int(L*0.6),int(L*0.8))
        
        if mode=='val':
            self.sample_idx=range(int(L*0.8),int(L*1))
        
        self.L=len(self.sample_idx)
        
        
        self.mean=np.array([407.1981814386521,905.3917506083453,1041.6140561764744]).reshape(-1,1)
        self.std_var=np.sqrt(np.array([412.5176029578715,20423.3857524064,16250.988775375401])).reshape(-1,1)

    
    @classmethod
    def crop_center(self,img,x,y,cropx,cropy):
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

    def __getitem__(self, idx):
        key,i,j=self.samples[idx]
        X,Y=self.X[key],self.Y[key]
        i,j=int(i),int(j)
        X_croped=self.crop_center(X,i,j,14,14)
        Y_croped=Y[i,j]
        for chennel in range(3):
            X_croped[chennel,:,:]=(X_croped[chennel,:,:]-self.mean[chennel])/self.std_var[chennel]
        return X_croped,Y_croped,i,j,key


    def __len__(self):
        return self.L

    def name(self):
        return 'IRDataset'



class CustomDatasetDataLoader(object):
    def __init__(self, batchSize, nThreads=8, task='identification', mode='train'):
        self.dataset = IRDataset(task,mode)
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


if __name__ =='__main__':
    dataloader=CustomDatasetDataLoader(1024)
    print()