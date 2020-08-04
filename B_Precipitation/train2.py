from os import close
import torch
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
import torch.utils.data as data
import torchnet as tnt
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import os
import json
import pickle as pkl
import argparse
import pprint
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from models.Dataloader import CustomDatasetDataLoader,IR_Split
from models.IPEC_model import IPECNet
from models.Meters import BinaryClsMeter
import json
import tqdm
from torch.optim import lr_scheduler
import sys
sys.path.append('/usr/commondata/weather/code/Precipitation_Estimation/')
from Tools.torchtool import EarlyStopping
from Tools.log import logfunc,plot_estimation
from B_Precipitation.models.Loss import Estimation_Loss,SoftmaxLoss
from tensorboardX import SummaryWriter
from Tools.control_parm import LinearSchedular
import matplotlib.pyplot as plt
from sklearn import metrics

# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

OneHot=lambda label,C: torch.zeros(label.shape[0],C).scatter_(1,label.view(-1,1),1)
toCPU=lambda x: x.detach().cpu().numpy()
toCUDA=lambda x: torch.tensor(x).cuda()

train_path='/usr/commondata/weather/dataset_release/IR_dataset_QingHua/'
# GOSE_test=GOSE_val=GOSE_train=np.load(train_path+'X_train_hourly_toy.npz')['arr_0']
# StageIV_test=StageIV_val=StageIV_train=np.load(train_path+'Y_train_hourly_toy.npz')['arr_0']

GOSE_train=np.load(train_path+'X_train_hourly.npz')['arr_0']
StageIV_train=np.load(train_path+'Y_train_hourly.npz')['arr_0']

GOSE_val=np.load(train_path+'X_val_hourly.npz')['arr_0']
StageIV_val=np.load(train_path+'Y_val_hourly.npz')['arr_0']

GOSE_test=np.load(train_path+'X_test_C_summer_hourly.npz')['arr_0']
StageIV_test=np.load(train_path+'Y_test_C_summer_hourly.npz')['arr_0']


def save_info(filename,loginfo):
    with open(filename,'w') as file_obj:
        json.dump( loginfo, file_obj)


#####################identification###################
def train_epoch_identification(model,writer, optimizer, scheduler, criterion, data_loader, device, config):
    acc_meter = BinaryClsMeter(config['task'])
    loss_meter = tnt.meter.AverageValueMeter()

    for idx, (x,y,weight,T,row,col,X,Y) in enumerate(data_loader):
        x = x.to(device).float()
        y=y[:,14,14]
        y = (y>0.1).to(device).long().view(-1)
        weight = weight.to(device).float()
        optimizer.zero_grad()

        out = model(x)

        loss = criterion(out,y,weight)
        loss.backward()
        optimizer.step()
        scheduler.step()

        acc_meter.add(torch.argmax(out,dim=1), y)
        loss_meter.add(loss.item())

        writer.add_scalar('train_loss', loss.item(), global_step=writer.train_step)
        if y.shape[0]==1024:
            writer.add_figure('identification',plot_estimation(toCPU(torch.argmax(out,dim=1)),toCPU(y)),global_step=writer.train_step,close=True)
            plt.close()
        writer.train_step+=1

        if (idx + 1) % config['display_step'] == 0:
            print('Step [{}/{}], Loss: {:.4f}'.format(idx + 1, len(data_loader), loss.item()))
        

    indicate=acc_meter.value()
    epoch_metrics = {'train_loss': loss_meter.value()[0],
                     'train_acc0': indicate[0],
                     'train_acc1': indicate[1],
                     'train_POD': indicate[2],
                     'train_FAR': indicate[3],
                     'train_CSI': indicate[4]
                     }

    writer.add_scalar('train_acc0', indicate[0], global_step=writer.train_step)
    writer.add_scalar('train_acc1', indicate[1], global_step=writer.train_step)
    writer.add_scalar('train_POD', indicate[2], global_step=writer.train_step)
    writer.add_scalar('train_FAR', indicate[3], global_step=writer.train_step)
    writer.add_scalar('train_CSI', indicate[4], global_step=writer.train_step)

    return epoch_metrics

def evaluation_identification(model, writer, criterion, loader, device, config, mode='val'):
    acc_meter = BinaryClsMeter(config['task'])
    loss_meter = tnt.meter.AverageValueMeter()
    for idx, (x,y,weight,T,row,col,X,Y) in enumerate(loader):
        x = x.to(device).float()
        y=y[:,14,14]
        y = (y>0.1).to(device).long().view(-1)
        weight = weight.to(device).float()
        with torch.no_grad():
            out = model(x)
            loss = criterion(out,y,weight)

        acc_meter.add(torch.argmax(out,dim=1), y)
        loss_meter.add(loss.item())

    indicate=acc_meter.value()
    metrics = {'{}_loss'.format(mode): loss_meter.value()[0],
               '{}_acc0'.format(mode): indicate[0],
               '{}_acc1'.format(mode): indicate[1],
               '{}_POD'.format(mode): indicate[2],
               '{}_FAR'.format(mode): indicate[3],
               '{}_CSI'.format(mode): indicate[4],
               }
    
    writer.add_scalar('test_acc0', indicate[0], global_step=writer.train_step)
    writer.add_scalar('test_acc1', indicate[1], global_step=writer.train_step)
    writer.add_scalar('test_POS', indicate[2], global_step=writer.train_step)
    writer.add_scalar('test_FAR', indicate[3], global_step=writer.train_step)
    writer.add_scalar('test_CSI', indicate[4], global_step=writer.train_step)

    if mode == 'val':
        return metrics

    elif mode == 'test':
        return metrics

######################estimation######################
def train_epoch_estimation(model, writer, optimizer, scheduler, criterion, data_loader, device, config, controlers):
    w_kl_controler,w_ed_controler = controlers
    acc_meter = BinaryClsMeter(config['task'])
    kl_loss_meter = tnt.meter.AverageValueMeter()
    ed_loss_meter = tnt.meter.AverageValueMeter()

    for idx, (x,y,weight,T,row,col,X,Y) in enumerate(data_loader):
        config['w_kl'] = w_kl_controler.Step()
        config['w_ed'] = w_ed_controler.Step()

        x = x.to(device).float()
        y=y[:,14,14]
        y = y.to(device).float().view(-1)
        weight = weight.to(device).float()

        optimizer.zero_grad()
        out = model(x).view(-1)
        # def hook(grad):
        #     print('pred {}'.format(grad))
        #     return grad
        # out.register_hook(hook)

        kl_loss, ed_loss = criterion(out,y,weight)
        loss=config['w_kl']*kl_loss+ed_loss*config['w_ed']

        loss.backward()
        optimizer.step()
        scheduler.step()

        acc_meter.add(out, y)
        kl_loss_meter.add(kl_loss.item())
        ed_loss_meter.add(ed_loss.item())

        writer.add_scalar('train_loss', loss.item(), global_step=writer.train_step)
        writer.add_scalar('train_kl_loss', kl_loss.item(), global_step=writer.train_step)
        writer.add_scalar('train_ed_loss', ed_loss.item(), global_step=writer.train_step)
        writer.add_scalar('train_mse',metrics.mean_squared_error(toCPU(y), toCPU(out)) , global_step=writer.train_step)
        writer.add_scalar('w_kl', config['w_kl'], global_step=writer.train_step)
        writer.add_scalar('w_ed', config['w_ed'], global_step=writer.train_step)
        writer.add_histogram('hist_true', toCPU(y), global_step=writer.train_step)
        writer.add_histogram('hist_pred', toCPU(out), global_step=writer.train_step)
        if y.shape[0]==1024:
            writer.add_figure('estimate',plot_estimation(toCPU(out),toCPU(y)),global_step=writer.train_step,close=True)
            plt.close()
        writer.train_step+=1
        

        if (idx + 1) % config['display_step'] == 0:
            print('Step [{}/{}], Loss: {:.4f}'.format(idx + 1, len(data_loader), loss.item()))
        
            

    indicate=acc_meter.value()
    kl_loss=kl_loss_meter.value()[0]
    ed_loss=ed_loss_meter.value()[0]

    writer.add_scalar('train_CC', indicate[0], global_step=writer.train_step)
    writer.add_scalar('train_BIAS', indicate[1], global_step=writer.train_step)
    writer.add_scalar('train_MSE', indicate[2], global_step=writer.train_step)

    epoch_metrics = {'train_kl_loss': kl_loss,
                     'train_ed_loss': ed_loss,
                     'train_loss': config['w_kl']*kl_loss+config['w_ed']*ed_loss,
                     'train_CC': indicate[0],
                     'train_BIAS': indicate[1],
                     'train_MSE': indicate[2],
                     }

    return epoch_metrics


def evaluation_estimation(model, writer, criterion, loader, device, config, mode='val'):
    acc_meter = BinaryClsMeter(config['task'])
    kl_loss_meter = tnt.meter.AverageValueMeter()
    ed_loss_meter = tnt.meter.AverageValueMeter()

    for idx, (x,y,weight,T,row,col,X,Y) in enumerate(loader):
        x = x.to(device).float()
        y=y[:,14,14]
        y = y.to(device).float().view(-1)
        weight = weight.to(device).float()

        with torch.no_grad():
            out = model(x).view(-1)
            kl_loss, ed_loss = criterion(out,y.long(),weight)

        acc_meter.add(out, y)
        kl_loss_meter.add(kl_loss.item())
        ed_loss_meter.add(ed_loss.item())

    indicate=acc_meter.value()
    kl_loss=kl_loss_meter.value()[0]
    ed_loss=ed_loss_meter.value()[0]
    loss = config['w_kl']*kl_loss+config['w_ed']*ed_loss

    writer.add_scalar('test_CC', indicate[0], global_step=writer.train_step)
    writer.add_scalar('test_BIAS', indicate[1], global_step=writer.train_step)
    writer.add_scalar('test_MSE', indicate[2], global_step=writer.train_step)

    valmetrics = {
                '{}_kl_loss'.format(mode): kl_loss,
                '{}_ed_loss'.format(mode): ed_loss,
                '{}_loss'.format(mode): loss,
                '{}_CC'.format(mode): indicate[0],
                '{}_BIAS'.format(mode): indicate[1],
                '{}_MSE'.format(mode): indicate[2],
                }
    
    if mode == 'val':
        return valmetrics

    elif mode == 'test':
        return valmetrics


#####################data loaders#####################
def load_data(config):
    train_samples,train_weight=IR_Split(     X=GOSE_train, 
                                Y=StageIV_train,
                                task=config['task'],
                                seed=config['rdm_seed'],
                                win_size=14,
                                sampling_step=config['sampling_step'],
                                R_w=config['R_w'],
                                NR_w=config['NR_w'],
                            ).split_dataset()

    train_loader= CustomDatasetDataLoader(  X=GOSE_train, 
                                            Y=StageIV_train,
                                            weights=train_weight,
                                            batchSize=config['batch_size'],
                                            selected_samples=train_samples,
                                            win_size=14,
                                            nThreads=1,
                                            seed=config['rdm_seed'],
                                            )
    
    val_samples,val_weight=IR_Split(   X=GOSE_val, 
                            Y=StageIV_val,
                            task=config['task'],
                            seed=config['rdm_seed'],
                            win_size=14,
                            sampling_step=config['sampling_step'],
                            R_w=config['R_w'],
                            NR_w=config['NR_w'],
                            # evaluate=True
                        ).split_dataset()
    val_loader  = CustomDatasetDataLoader(  X=GOSE_val, 
                                            Y=StageIV_val,
                                            weights=val_weight,
                                            batchSize=config['batch_size'],
                                            selected_samples=val_samples,  
                                            win_size=14,
                                            nThreads=1,
                                            seed=config['rdm_seed']
                                            )



    test_samples,test_weight=IR_Split(  X=GOSE_val, 
                            Y=StageIV_val,
                            task=config['task'],
                            seed=config['rdm_seed'],
                            win_size=14,
                            R_w=config['R_w'],
                            NR_w=config['NR_w'],
                            evaluate=True
                        ).split_dataset()
    test_loader  = CustomDatasetDataLoader( X=GOSE_val, 
                                            Y=StageIV_val,
                                            weights=test_weight,
                                            batchSize=config['batch_size'],
                                            selected_samples=test_samples,  
                                            win_size=14,
                                            nThreads=1,
                                            seed=config['rdm_seed']
                                            )

    return train_loader,val_loader,test_loader


def load_model(config,model):
    res_dir = Path(config['res_dir'])
    model.load_state_dict( torch.load(res_dir/'epoch_{}.pt'.format(config['epoch_s']-1)) )
    with open(res_dir/'train_val_info.json','r') as fp:
        previous_log = json.load(fp)
    return model,previous_log


def main(config,writer):
    device = torch.device(config['device'])

    ###################load model###################
    model=IPECNet(nc=[1,16,16,32,32],padding_type='zero',norm_layer=nn.BatchNorm2d,task=config['task'])
    if config['gpus']==1:
        model = torch.nn.DataParallel(model.to(device), device_ids=[0])
    else:
        model = torch.nn.DataParallel(model.to(device), device_ids=[0,1])
    

    optimizer = torch.optim.SGD(model.parameters(),lr=config['lr'])
    if config['task']=='identification':
        criterion = SoftmaxLoss()
    if config['task']=='estimation':
        criterion = Estimation_Loss(config['X_min'],config['X_max'],config['bins'],config['sigma'],config['mse'])


    if config['epoch_s']>1:
        model,previous_log = load_model(config,model)
        trainlog=previous_log['train_metrics']
        writer.train_step = trainlog['train_step']
        best_epoch = trainlog[str(config['epoch_s']-1)]['best_epoch']
        early_stopping.best_score=-trainlog[str(best_epoch)]['val_loss']
        scheduler = lr_scheduler.StepLR(optimizer, step_size=config['lr_step'], gamma=0.1,last_epoch=writer.train_step)
    else:
        trainlog = {}
        writer.train_step=0
        scheduler = lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

    #####################control parameters#########
    w_kl_controler = LinearSchedular(config['w_kl_C'])
    w_ed_controler = LinearSchedular(config['w_ed_C'])

    #####################start train################
    for epoch in tqdm.tqdm(range(config['epoch_s'], config['epoch_e'])):
        train_loader,val_loader,test_loader=load_data(config)

        print('EPOCH {}/{}'.format(epoch, config['epoch_e']))
        ##############
        print('Train . . . ')
        model.train()
        if config['task']=='identification':
            train_metrics = train_epoch_identification(
                                                        model, 
                                                        writer, 
                                                        optimizer, 
                                                        scheduler, 
                                                        criterion, 
                                                        train_loader, 
                                                        device=device, 
                                                        config=config,
                                                        )
        
        if config['task']=='estimation':
            train_metrics = train_epoch_estimation(
                                                        model, 
                                                        writer, 
                                                        optimizer, 
                                                        scheduler, 
                                                        criterion, 
                                                        train_loader, 
                                                        device=device, 
                                                        config=config,
                                                        controlers=(w_kl_controler,w_ed_controler),
                                                        )

        print(train_metrics)
        
        ##############
        print('Validation . . . ')
        model.eval()
        if config['task']=='identification':
            val_metrics = evaluation_identification(model, writer, criterion, val_loader, device=device, config=config, mode='val')
        
        if config['task']=='estimation':
            val_metrics = evaluation_estimation(model, writer, criterion, val_loader, device=device, config=config, mode='val')

        val_metrics['best_epoch']=early_stopping.best_epoch
        print(val_metrics)

        ##############
        trainlog[epoch] = {**train_metrics, **val_metrics}
        trainlog['train_step'] = writer.train_step
        filename=os.path.join(config['res_dir'], 'train_val_info.json')
        with open(filename,'w') as file_obj:
            json.dump(  {'train_metrics': trainlog,'val_metrics':val_metrics}, file_obj)
        
        early_stopping(val_metrics['val_loss'],model,epoch)
        if early_stopping.early_stop:
            break

    #############
    print('Testing best epoch . . .')
    model.load_state_dict(
            torch.load( os.path.join(config['res_dir'],'epoch_{}.pt'.format(early_stopping.best_epoch)) )
        )
    model.eval()

    if config['task']=='identification':
        test_metrics = evaluation_identification(model,writer, criterion, test_loader, device=device, mode='test', config=config)
    if config['task']=='estimation':
        test_metrics = evaluation_estimation(model,writer, criterion, test_loader, device=device, mode='test', config=config)

    print(test_metrics)

    filename=os.path.join(config['res_dir'], 'test_info.json')
    with open(filename,'w') as file_obj:
        json.dump(  {'test_metrics':test_metrics},file_obj)


if __name__ == '__main__':
    def coords(s):
        try:
            x, y, z = s.split(',')
            return float(x), float(y), int(z)
        except:
            raise argparse.ArgumentTypeError("Coordinates must be x,y,z")

    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str,
                        help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--rdm_seed', default=1, type=int, help='Random seed')
    parser.add_argument('--display_step', default=10, type=int,
                        help='Interval in batches between display of training metrics')
    parser.add_argument('--res_dir', default='/usr/commondata/weather/code/Precipitation_Estimation/B_Precipitation', type=str)
    parser.add_argument('--ex_name', default='test', type=str)

    # dataset parameters
    parser.add_argument('--task', default='identification', type=str)
    parser.add_argument('--sampling_step',default=14,type=int)
    parser.add_argument('--R_w', default=1, type=float)
    parser.add_argument('--NR_w', default=1, type=float)

    # Training parameters
    parser.add_argument('--epoch_s', default=1, type=int, help='start epoch')
    parser.add_argument('--epoch_e', default=10, type=int, help='end epoch')
    parser.add_argument('--batch_size', default=1024, type=int, help='Batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--lr_step',default=1000,type=int)
    parser.add_argument('--patience', default=100, type=int)
    parser.add_argument('--delta', default=0, type=float)
    parser.add_argument('--gpus',default=1,type=int)
   

    # estimation parameters
    # parser.add_argument('--w_kl', default=0.01, type=float, help='weight of KL loss')
    # parser.add_argument('--w_ed', default=1, type=float, help='weight of ED loss')
    parser.add_argument('--w_kl_C', default=[(0,0,100),(0,1,100)], dest="w_kl_C", type=coords, nargs='+', help='control vector of kl loss')
    parser.add_argument('--w_ed_C', default=[(1,1,1000)], dest="w_ed_C", type=coords, nargs='+', help='control vector of ed loss')
    parser.add_argument('--sigma', default=2.5, type=float, help='window size of density estimation')
    parser.add_argument('--X_min', default=-10, type=float, help='minimum of rainfull')
    parser.add_argument('--X_max', default=60, type=float, help='maximum of rainfull')
    parser.add_argument('--bins', default=70, type=float, help='bins of rainfull')
    parser.add_argument('--mse', default=False, help='use mse loss')
    

    args = parser.parse_args()
    config = args.__dict__
    config['res_dir']=config['res_dir']+'/{}/{}'.format(config['task'],config['ex_name'])
    res=Path(config['res_dir'])
    res.mkdir(parents=True, exist_ok=True )
    writer = SummaryWriter(logdir=res/'logging')
    pprint.pprint(config)

    filename=os.path.join(config['res_dir'], 'model_param.json')
    with open(filename,'w') as file_obj:
        json.dump(  config,file_obj)

    early_stopping = EarlyStopping( patience=config['patience'],
                                    delta=config['delta'], 
                                    root=config['res_dir'],
                                    verbose=True)
    main(config,writer)