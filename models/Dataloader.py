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


GOSE=np.load('/usr/commondata/weather/New/GOSE.npy',allow_pickle=True).item()
StageIV=np.load('/usr/commondata/weather/New/StageIV.npy',allow_pickle=True).item()

def date2num(start_date,end_dates):
    result=[]
    for T in end_dates:
        end_date = date(int(T[:4]), int(T[4:6]), int(T[6:8]))
        delta = (end_date - start_date)
        day=delta.days
        hour=T[8:]
        result.append('{}.{}'.format(day,hour))
    return result
    
start_date = date(2011, 12, 31)
end_dates=list(StageIV.keys())
StageIV_keys=date2num(start_date,end_dates)
for i in range(len(StageIV_keys)):
    StageIV[StageIV_keys[i]]=StageIV.pop(end_dates[i])


balance=True
if balance:
    global_samples=np.load('/usr/commondata/weather/New/samples_B.npy')
else:
    global_samples=np.load('/usr/commondata/weather/New/samples.npy')



class IRDataset(Dataset):
    def __init__(self,mode='train',balance=True):
        self.X=GOSE
        self.Y=StageIV
        self.samples=global_samples
        
        
        L=len(self.samples)
        self.mode=mode
        if mode=='train':
            self.sample_idx=range(0,int(L*0.6))
        
        if mode=='test':
            self.sample_idx=range(int(L*0.6),int(L*0.8))
        
        if mode=='val':
            self.sample_idx=range(int(L*0.8),int(L*1))
        
        self.L=len(self.sample_idx)

        self.mean=np.array([422.6626917616589,442.8474696204027,442.8474696204027,-199.98306597322008,442.8474696204027]).reshape(-1,1)
        self.std_var=np.sqrt(np.array([489.64054904540046,1279.0312110649393,1279.0312110649393,3892.523190931211,1279.0312110649393])).reshape(-1,1)
    
    
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
            
    def sampling(self,key,img):
        R_samples=[]
        NR_samples=[]
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                Y=self.crop_center(img,i,j,14,14)
                if Y is not None:
                    if Y[14,14]>0.1:
                        R_samples.append((key,i,j))
                    else:
                        NR_samples.append((key,i,j))
        NR_samples_B=random.sample(NR_samples, len(R_samples))
        return R_samples+NR_samples,R_samples+NR_samples_B
    
    
    def get_samples(self):
        useful_keys=list(set(self.X.keys())&set(self.Y.keys()))
        self.useful_keys=sorted(useful_keys)
        
        samples=[]
        samples_B=[]
        for key in tqdm.tqdm(self.useful_keys):
            samples_tmp,samples_B_tmp=self.sampling(key,self.Y[key])
            samples+=samples_tmp
            samples_B+=samples_B_tmp
        return samples,samples_B
    
    def save_samples(self):
        samples,samples_B=dataset.get_samples()
        samples=np.array(samples)
        samples_B=np.array(samples_B)
        np.save('/usr/commondata/weather/New/samples_B.npy',samples_B)
        np.save('/usr/commondata/weather/New/samples.npy',samples)
        

    def __getitem__(self, idx):
        key,i,j=self.samples[idx]
        X,Y=self.X[key],self.Y[key]
        i,j=int(i),int(j)
        X_croped=self.crop_center(X,i,j,14,14)
        Y_croped=self.crop_center(Y,i,j,0,0)
        X_croped[np.isnan(X_croped)]=0
        Y_croped[np.isnan(Y_croped)]=0
        for chennel in range(5):
            X_croped[chennel,:,:]=(X_croped[chennel,:,:]-self.mean[chennel])/self.std_var[chennel]
        return X_croped,Y_croped,i,j,key


    def __len__(self):
        return self.L

    def name(self):
        return 'IRDataset'



class CustomDatasetDataLoader(object):
    def __init__(self, batchSize, nThreads=8, mode='train'):
        self.dataset = IRDataset(mode)
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