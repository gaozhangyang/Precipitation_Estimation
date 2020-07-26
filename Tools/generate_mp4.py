from multiprocessing import Pool,Manager,Process
from pathlib import Path
import os
import tqdm
import multiprocessing
import signal
import numpy as np
import abc
import imageio
import time
import matplotlib.pyplot as plt
import sys
sys.path.append('/usr/data/gzy/climate/Precipitation_Estimation/')
import Identification.models.Dataloader as Dataloader
import matplotlib.patches as patches
import torch

toCPU=lambda x: x.detach().cpu().numpy()
toCUDA=lambda x: torch.tensor(x).cuda()

processes=[]

def term(sig_num, addtion):
    print('terminate process {}'.format(os.getpid()))
    try:
        print('the processes is {}'.format(processes) )
        for p in processes:
            print('process {} terminate'.format(p.pid))
            p.terminate()
            # os.kill(p.pid, signal.SIGKILL)
    except Exception as e:
        print(str(e))


class GIFPloter():
    def __init__(self,root):
        self.root=root

    @abc.abstractmethod
    def callback(self,task,pid,Process_state):
        # TODO 此处增加画图操作
        Process_state[str(pid)]=True
    

    def run(self,pnum,tasks):
        signal.signal(signal.SIGTERM, term)

        ###############################多线程处理##########################
        Process_state=Manager().dict({str(i):True for i in range(pnum)})
        idx=0
        while idx<len(tasks):
            #查询是否有可用线程
            for pid in range(pnum):
                if Process_state[str(pid)]==True:
                    Process_state[str(pid)]=False #占用当前线程
                    p=Process(target=self.callback,args=(tasks[idx],pid,Process_state))
                    print(idx)
                    idx+=1
                    p.start()
                    processes.append(p)
                    break

        for p in processes:
            p.join()


    def SaveGIF(self,name,fps=1):
        path = self.root
        gif_images_path = os.listdir(path+'/')

        gif_images_path.sort()
        gif_images = []
        for i, path_ in enumerate(gif_images_path):
            print(path_)
            if '.png' in path_:
                if i % 1 == 0:
                    gif_images.append(imageio.imread(path+'/'+path_))
                    
        imageio.mimsave(path+'/'+"{}.mp4".format(name), gif_images, fps=fps)


class Plot_XY(GIFPloter):
    def __init__(self,root):
        super(Plot_XY,self).__init__(root)
    
    def callback(self,task,pid,Process_state):
        x,y,X,Y,T,row,col,idx=task
        ##大图
        fig= plt.figure(constrained_layout=True,figsize=(30, 25))
        gs = fig.add_gridspec(5,6)
        H=X[0].shape[0]
        x_pos=col-15
        y_pos=row-15

        ax1=fig.add_subplot(gs[0:2, 0:2])
        ax1.imshow(X[0])
        rect = patches.Rectangle((x_pos,y_pos),29,29,linewidth=1,edgecolor='r',facecolor='none')
        ax1.add_patch(rect)

        ax2=fig.add_subplot(gs[0:2, 2:4])
        ax2.imshow(X[1])
        rect = patches.Rectangle((x_pos,y_pos),29,29,linewidth=1,edgecolor='r',facecolor='none')
        ax2.add_patch(rect)

        ax3=fig.add_subplot(gs[0:2, 4:6])
        ax3.imshow(X[2])
        rect = patches.Rectangle((x_pos,y_pos),29,29,linewidth=1,edgecolor='r',facecolor='none')
        ax3.add_patch(rect)

        ax4=fig.add_subplot(gs[2:4, 0:2])
        ax4.imshow(X[0]-X[1])
        rect = patches.Rectangle((x_pos,y_pos),29,29,linewidth=1,edgecolor='r',facecolor='none')
        ax4.add_patch(rect)

        ax5=fig.add_subplot(gs[2:4, 2:4])
        ax5.imshow(X[1]-X[2])
        rect = patches.Rectangle((x_pos,y_pos),29,29,linewidth=1,edgecolor='r',facecolor='none')
        ax5.add_patch(rect)

        ax6=fig.add_subplot(gs[2:4, 4:6])
        ax6.imshow(Y)
        rect = patches.Rectangle((x_pos,y_pos),29,29,linewidth=1,edgecolor='r',facecolor='none')
        ax6.add_patch(rect)


        ## 子图
        ax7=fig.add_subplot(gs[4, 0])
        ax7.imshow(x[0])

        ax8=fig.add_subplot(gs[4, 1])
        ax8.imshow(x[1])

        ax9=fig.add_subplot(gs[4, 2])
        ax9.imshow(x[2])

        ax10=fig.add_subplot(gs[4, 3])
        ax10.imshow(x[0]-x[1])

        ax11=fig.add_subplot(gs[4, 4])
        ax11.imshow(x[1]-x[2])

        ax12=fig.add_subplot(gs[4, 5])
        ax12.imshow(y)

        ax9.set_title('{}-{}-{}'.format(T,row,col))
        ax12.set_title('{}'.format(y[14,14]))
        
        path=Path(self.root)
        path.mkdir(exist_ok=True,parents=True)
        plt.savefig(path/('{}.png'.format(idx)))
        plt.close()

        Process_state[str(pid)]=True


class Plot_pred_surface(GIFPloter):
    def __init__(self,root):
        super(Plot_pred_surface,self).__init__(root)
    
    def callback(self,task,pid,Process_state):
        x, pred, y_true, T, H, W=task
        N=int(pred.shape[0]**0.5)
        pred=pred.reshape(H,W)
        y_true=y_true.reshape(H,W)


        fig= plt.figure(constrained_layout=True,figsize=(20, 14))
        gs = fig.add_gridspec(14,20)

        ax1=fig.add_subplot(gs[0:4, 0:4])
        ax1.imshow(x[0])

        ax2=fig.add_subplot(gs[0:4, 4:8])
        ax2.imshow(x[1])

        ax3=fig.add_subplot(gs[0:4, 8:12])
        ax3.imshow(x[2])

        ax4=fig.add_subplot(gs[0:4, 12:16])
        ax4.imshow(x[0]-x[1])

        ax5=fig.add_subplot(gs[0:4, 16:20])
        ax5.imshow(x[1]-x[2])

        ax6=fig.add_subplot(gs[4:14, 0:10])
        ax6.imshow(y_true)

        ax7=fig.add_subplot(gs[4:14, 10:20])
        ax7.imshow(pred)
        ax7.set_title('{}'.format(T))

        path=Path(self.root)
        path.mkdir(exist_ok=True,parents=True)
        plt.savefig(path/('{}.png'.format(T)))
        plt.close()

        Process_state[str(pid)]=True




class Draw:
    def __init__(self):
        pass

    def generate_XY_MP4(self):
        train_path='/usr/commondata/weather/dataset_release/IR_dataset_QingHua/'
        GOSE_train=np.load(train_path+'X_val_hourly.npz')['arr_0']
        StageIV_train=np.load(train_path+'Y_val_hourly.npz')['arr_0']

        train_samples=Dataloader.IR_Split(     
                                    X=GOSE_train, 
                                    Y=StageIV_train,
                                    task='estimation',
                                    seed=2020,
                                    shuffle=True,
                                    win_size=14,
                                    k_num=10000,
                                ).split_dataset()

        train_loader= Dataloader.CustomDatasetDataLoader(  
                                                X=GOSE_train, 
                                                Y=StageIV_train,
                                                batchSize=1024,
                                                selected_samples=train_samples,
                                                win_size=14,
                                                nThreads=0,
                                                seed=2020,
                                                )

        tasks=[]
        idx=0
        for xs,ys,Ts,rows,cols,Xs,Ys in train_loader:
            for N in range(xs.shape[0]):
                if (ys[N]>10).any():
                    tasks.append((xs[idx],ys[idx],Xs[idx],Ys[idx],Ts[idx],rows[idx],cols[idx],idx))
                    idx+=1
                    print(idx)
                
                if idx==200:
                    break
                    
            if idx==200:
                break
        
        XY_MP4=Plot_XY('/usr/data/gzy/Precipitation_Estimation/Visualization/val_center_above_10')
        XY_MP4.run(30,tasks)
        XY_MP4.SaveGIF('XY',fps=0.5)
    
    def generate_pred_surface_MP4(self,model_path,step=14):
        import os
        import matplotlib.pyplot as plt
        from Tools.torchtool import SetSeed

        SetSeed(2020)
        
        # TODO load data
        train_path='/usr/commondata/weather/'
        GOSE=np.load(train_path+'X_val_hourly.npz')['arr_0']
        StageIV=np.load(train_path+'Y_val_hourly.npz')['arr_0']

        H,W=round((GOSE.shape[2]-29)/step), round((GOSE.shape[3]-29)/step)

        tasks=[]
        for i in range(0, GOSE.shape[0], int(GOSE.shape[0]/150)):
            X=torch.tensor(GOSE[i]).float().cuda()
            Y=StageIV[i]
            pred,y_true=self.test_model(model_path,X=X,Y=Y,step=step)
            tasks.append((toCPU(X),pred,y_true,i,H,W))
        
        # TODO save result
        pred_MP4=Plot_pred_surface('/usr/data/gzy/climate/Precipitation_Estimation/Visualization/pred_val_surface_ex3')
        pred_MP4.run(30,tasks)
        pred_MP4.SaveGIF('pred_val',fps=0.5)


    def test_model(self,model_path,X,Y,multi_gpu=True,batch_size=1024,step=14):
        from Identification.models.IPEC_model import IPECNet
        import torch.nn as nn
        from collections import OrderedDict

        ########################load model######################
        model=IPECNet(nc=[1,16,16,32,32],padding_type='zero',norm_layer=nn.BatchNorm2d,task='identification')
        model = torch.nn.DataParallel(model.to('cuda'), device_ids=[0])
        state_dict = torch.load(model_path)
        if multi_gpu:
            model.load_state_dict(state_dict)
        else:
            # create new OrderedDict that does not contain `module.`
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v

            model.load_state_dict(new_state_dict)
            
        model=model.cuda()
        
        
        #######################generate samples################
        L=len(range(14,375-15,step))**2
        test_data=torch.zeros(L,3,29,29)
        y_true=np.zeros(L)

        N=0
        for i in range(14,375-15,step):
            for j in range(14,375-15,step):
                tmpX=Dataloader.IRDataset.unsafe_crop_center(X,i,j,14,14)
                test_data[N,:,:,:]=tmpX
                y_true[N]=(Y[i,j]>0.1)
                N+=1
        
        #######################get pred###################
        with torch.no_grad():
            pred=[]
            for i in range(0,L//batch_size+1):
                scope=range(i*batch_size,min((i+1)*batch_size,L))
                tmpX=test_data[scope].float().cuda()
                tmp_pred=np.argmax(model(tmpX).detach().cpu().numpy(),axis=1)
                pred.append(tmp_pred)

        pred=np.hstack(pred)
        
        return pred,y_true



if __name__ == '__main__':
    draw=Draw()
    # draw.generate_XY_MP4()
    draw.generate_pred_surface_MP4( model_path='/usr/data/gzy/climate/Precipitation_Estimation/Identification/ex2/results/003/epoch_5_step_5.pt',
                                    step=14)
