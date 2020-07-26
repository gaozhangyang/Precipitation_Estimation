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
sys.path.append('/usr/data/gzy/Precipitation_Estimation/')
from Tools.torchtool import EarlyStopping
from Estimation.models.Loss import Estimation_Loss

OneHot=lambda label,C: torch.zeros(label.shape[0],C).scatter_(1,label.view(-1,1),1)

train_path='/usr/commondata/weather/dataset_release/IR_dataset_QingHua/'
GOSE_train=np.load(train_path+'X_train_hourly.npz')['arr_0']
StageIV_train=np.load(train_path+'Y_train_hourly.npz')['arr_0']

GOSE_val=np.load(train_path+'X_val_hourly.npz')['arr_0']
StageIV_val=np.load(train_path+'Y_val_hourly.npz')['arr_0']

GOSE_test=np.load(train_path+'X_test_C_summer_hourly.npz')['arr_0']
StageIV_test=np.load(train_path+'Y_test_C_summer_hourly.npz')['arr_0']


def save_info(filename,loginfo):
    with open(filename,'w') as file_obj:
        json.dump( loginfo, file_obj)


def train_epoch_identification(model, optimizer, scheduler, criterion, data_loader, device, config):
    acc_meter = BinaryClsMeter(config['task'])
    loss_meter = tnt.meter.AverageValueMeter()

    for idx, (x,y,T,row,col,X,Y) in enumerate(data_loader):
        x = x.to(device).float()
        y=y[:,14,14]
        y = (y>0.1).to(device).long().view(-1)
        optimizer.zero_grad()
        # x[torch.isnan(x)]=0
        if torch.isnan(x).any():
            print('NO1')
        out = model(x)

        loss = criterion(out,y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        acc_meter.add(torch.argmax(out,dim=1), y)
        loss_meter.add(loss.item())

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
    return epoch_metrics


def train_epoch_estimation(model, optimizer, scheduler, criterion, data_loader, device, config):
    acc_meter = BinaryClsMeter(config['task'])
    loss_meter = tnt.meter.AverageValueMeter()

    for idx, (x,y,T,row,col,X,Y) in enumerate(data_loader):
        x = x.to(device).float()
        y = y.to(device).float().view(-1)

        optimizer.zero_grad()
        x[torch.isnan(x)]=0
        out = model(x).view(-1)

        loss = criterion(out,y)

        loss.backward()
        optimizer.step()
        scheduler.step()

        acc_meter.add(out, y)
        loss_meter.add(loss.item())


        if (idx + 1) % config['display_step'] == 0:
            print('Step [{}/{}], Loss: {:.4f}'.format(idx + 1, len(data_loader), loss.item()))

    indicate=acc_meter.value()
    epoch_metrics = {'train_loss': loss_meter.value()[0],
                     'train_CC': indicate[0],
                     'train_BIAS': indicate[1],
                     'train_MSE': indicate[2],
                     }
    return epoch_metrics





def evaluation_identification(model, criterion, loader, device, config, mode='val'):
    acc_meter = BinaryClsMeter(config['task'])
    loss_meter = tnt.meter.AverageValueMeter()
    for idx, (x,y,T,row,col,X,Y) in enumerate(loader):
        x = x.to(device).float()
        y=y[:,14,14]
        y = (y>0.1).to(device).long().view(-1)
        with torch.no_grad():
            # x[torch.isnan(x)]=0
            if torch.isnan(x).any():
                print('NO2')
            out = model(x)
            loss = criterion(out,y)

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

    if mode == 'val':
        return metrics
    elif mode == 'test':
        return metrics


def evaluation_estimation(model, criterion, loader, device, config, mode='val'):
    acc_meter = BinaryClsMeter(config['task'])
    loss_meter = tnt.meter.AverageValueMeter()

    for idx, (x,y,T,row,col,X,Y) in enumerate(loader):
        x = x.to(device).float()
        y = y.to(device).float().view(-1)

        with torch.no_grad():
            if torch.isnan(x).any():
                print('NO2')
            # x[torch.isnan(x)]=0
            out = model(x).view(-1)
            loss = criterion(out,y.long())

        acc_meter.add(out, y)
        loss_meter.add(loss.item())

    indicate=acc_meter.value()
    metrics = {'{}_loss'.format(mode): loss_meter.value()[0],
               '{}_CC'.format(mode): indicate[0],
               '{}_BIAS'.format(mode): indicate[1],
               '{}_MSE'.format(mode): indicate[2],
               }

    if mode == 'val':
        return metrics
    elif mode == 'test':
        return metrics


def load_data(config):
    train_samples=IR_Split(     X=GOSE_train, 
                                Y=StageIV_train,
                                task=config['task'],
                                seed=config['rdm_seed'],
                                shuffle=True,
                                win_size=14,
                                R_num=config['train_R'],
                                NR_num=config['train_NR'],
                            ).split_dataset()
    train_loader= CustomDatasetDataLoader(  X=GOSE_train, 
                                            Y=StageIV_train,
                                            batchSize=config['batch_size'],
                                            selected_samples=train_samples,
                                            win_size=14,
                                            nThreads=1,
                                            seed=config['rdm_seed'],
                                            )
    
    val_samples=IR_Split(   X=GOSE_val, 
                            Y=StageIV_val,
                            task=config['task'],
                            seed=config['rdm_seed'],
                            shuffle=True,
                            win_size=14,
                            R_num=config['val_R'],
                            NR_num=config['val_NR'],
                            # evaluate=True
                        ).split_dataset()
    val_loader  = CustomDatasetDataLoader(  X=GOSE_val, 
                                            Y=StageIV_val,
                                            batchSize=config['batch_size'],
                                            selected_samples=val_samples,  
                                            win_size=14,
                                            nThreads=1,
                                            seed=config['rdm_seed']
                                            )


    test_samples=IR_Split(  X=GOSE_val, 
                            Y=StageIV_val,
                            task=config['task'],
                            seed=config['rdm_seed'],
                            shuffle=True,
                            win_size=14,
                            R_num=config['test_R'],
                            NR_num=config['test_NR'],
                            # evaluate=True
                        ).split_dataset()
    test_loader  = CustomDatasetDataLoader( X=GOSE_val, 
                                            Y=StageIV_val,
                                            batchSize=config['batch_size'],
                                            selected_samples=test_samples,  
                                            win_size=14,
                                            nThreads=1,
                                            seed=config['rdm_seed']
                                            )

    return train_loader,val_loader,test_loader



def main(config):
    device = torch.device(config['device'])
    train_loader,val_loader,test_loader=load_data(config)

    model=IPECNet(nc=[1,16,16,32,32],padding_type='zero',norm_layer=nn.BatchNorm2d,task=config['task'])
    model = torch.nn.DataParallel(model.to(device), device_ids=[0,1])
    optimizer = torch.optim.SGD(model.parameters(),lr=config['lr'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
    if config['task']=='identification':
        criterion = nn.CrossEntropyLoss()
    if config['task']=='estimation':
        criterion = Estimation_Loss(config['w'],config['h'],config['X_min'],config['X_max'],config['bins'])

    trainlog = {}
    for epoch in tqdm.tqdm(range(1, config['epochs'] + 1)):
        print('EPOCH {}/{}'.format(epoch, config['epochs']))
        print('Train . . . ')
        model.train()
        if config['task']=='identification':
            train_metrics = train_epoch_identification(model, optimizer, scheduler, criterion, train_loader, device=device, config=config)
        
        if config['task']=='estimation':
            train_metrics = train_epoch_estimation(model, optimizer, scheduler, criterion, train_loader, device=device, config=config)

        print(train_metrics)
        

        print('Validation . . . ')
        model.eval()
        if config['task']=='identification':
            val_metrics = evaluation_identification(model, criterion, val_loader, device=device, config=config, mode='val')
        
        if config['task']=='estimation':
            val_metrics = evaluation_estimation(model, criterion, val_loader, device=device, config=config, mode='val')

        print(val_metrics)


        trainlog[epoch] = {**train_metrics, **val_metrics}
        
        filename=os.path.join(config['res_dir'], 'train_val_info.json')
        with open(filename,'w') as file_obj:
            json.dump(  {'train_metrics': trainlog,'val_metrics':val_metrics}, file_obj)
        
        early_stopping(val_metrics['val_loss'],model,epoch)
        if early_stopping.early_stop:
            break

    print('Testing best epoch . . .')
    model.load_state_dict(
            torch.load( os.path.join(config['res_dir'],'epoch_{}_step_{}.pt'.format(epoch,early_stopping.train_step)) )
        )
    model.eval()

    if config['task']=='identification':
        test_metrics = evaluation_identification(model, criterion, test_loader, device=device, mode='test', config=config)
    if config['task']=='estimation':
        test_metrics = evaluation_estimation(model, criterion, test_loader, device=device, mode='test', config=config)

    print(test_metrics)

    filename=os.path.join(config['res_dir'], 'test_info.json')
    with open(filename,'w') as file_obj:
        json.dump(  {'test_metrics':test_metrics},file_obj)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str,
                        help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--rdm_seed', default=1, type=int, help='Random seed')
    parser.add_argument('--display_step', default=10, type=int,
                        help='Interval in batches between display of training metrics')
    parser.add_argument('--res_dir', default='/usr/data/gzy/Precipitation_Estimation/results', type=str)
    parser.add_argument('--ex_name', default='001', type=str)

    # dataset parameters
    parser.add_argument('--task', default='identification', type=str)
    parser.add_argument('--train_R', default=50000, type=int)
    parser.add_argument('--train_NR', default=50000, type=int)

    parser.add_argument('--val_R', default=10000, type=int)
    parser.add_argument('--val_NR', default=10000, type=int)

    parser.add_argument('--test_R', default=10000, type=int)
    parser.add_argument('--test_NR', default=10000, type=int)


    # Training parameters
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs per fold')
    parser.add_argument('--batch_size', default=1024, type=int, help='Batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--patience', default=8, type=int)
    parser.add_argument('--delta', default=0, type=float)

    # estimation parameters
    parser.add_argument('--w', default=1, type=float, help='weight of KL loss')
    parser.add_argument('--h', default=2.5, type=float, help='window size of density estimation')
    parser.add_argument('--X_min', default=-10, type=float, help='minimum of rainfull')
    parser.add_argument('--X_max', default=60, type=float, help='maximum of rainfull')
    parser.add_argument('--bins', default=70, type=float, help='bins of rainfull')




    args = parser.parse_args()
    config = args.__dict__
    config['res_dir']=config['res_dir']+'/{}/{}'.format(config['task'],config['ex_name'])
    res=Path(config['res_dir'])
    res.mkdir(parents=True, exist_ok=True )
    pprint.pprint(config)

    filename=os.path.join(config['res_dir'], 'model_param.json')
    with open(filename,'w') as file_obj:
        json.dump(  config,file_obj)

    early_stopping = EarlyStopping( patience=config['patience'],
                                    delta=config['delta'], 
                                    root=config['res_dir'],
                                    verbose=True)
    main(config)