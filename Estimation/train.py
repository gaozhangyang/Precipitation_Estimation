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
from loss.Loss import Estimation_Loss
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from models.Dataloader import CustomDatasetDataLoader,IR_Split
from models.IPEC_model import IPECNet
from models.Meters import BinaryClsMeter
import json
import tqdm
from torch.optim import lr_scheduler

OneHot=lambda label,C: torch.zeros(label.shape[0],C).scatter_(1,label.view(-1,1),1)

train_path='/usr/commondata/weather/dataset_release/IR_dataset_QingHua/'
GOSE_train=np.load(train_path+'X_train_hourly.npz')['arr_0']
StageIV_train=np.load(train_path+'Y_train_hourly.npz')['arr_0']

train_path='/usr/commondata/weather/dataset_release/IR_dataset_QingHua/'
GOSE_val=np.load(train_path+'X_val_hourly.npz')['arr_0']
StageIV_val=np.load(train_path+'Y_val_hourly.npz')['arr_0']


def train_epoch(model, optimizer, scheduler, criterion, data_loader, device, config):
    acc_meter = BinaryClsMeter()
    loss_meter = tnt.meter.AverageValueMeter()

    for idx, (x,y,T,row,col,X,Y) in enumerate(data_loader):
        x = x.to(device).float()
        y = y.to(device).float().view(-1)

        optimizer.zero_grad()
        x[torch.isnan(x)]=0
        out = model(x).view(-1)

        loss = criterion(out,y)
        # print(loss)
        if torch.isnan(loss):
            print()

        loss.backward()
        optimizer.step()
        scheduler.step()

        acc_meter.add(out, y)
        loss_meter.add(loss.item())


        if (idx + 1) % config['display_step'] == 0:
            print('Step [{}/{}], Loss: {:.4f}'.format(idx + 1, len(data_loader), loss_meter.value()[0]))

    indicate=acc_meter.value()
    epoch_metrics = {'train_loss': loss_meter.value()[0],
                     'train_CC': indicate[0],
                     'train_BIAS': indicate[1],
                     'train_MSE': indicate[2],
                     }

    return epoch_metrics


def evaluation(model, criterion, loader, device, config, mode='val'):
    acc_meter = BinaryClsMeter()
    loss_meter = tnt.meter.AverageValueMeter()

    for idx, (x,y,T,row,col,X,Y) in enumerate(loader):
        x = x.to(device).float()
        y = y.to(device).float().view(-1)

        with torch.no_grad():
            x[torch.isnan(x)]=0
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



def save_results(epoch, metrics, config):
    with open(os.path.join(config['res_dir'], 'epoch_{}'.format(epoch), 'test_metrics.json'), 'w') as outfile:
        json.dump(metrics, outfile, indent=4)


def main(config):
    device = torch.device(config['device'])

    train_samples=IR_Split(     X=GOSE_train, 
                                Y=StageIV_train,
                                task='estimation',
                                seed=config['rdm_seed'],
                                shuffle=True,
                                win_size=14,
                                k_num=470000
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
                            task='estimation',
                            seed=config['rdm_seed'],
                            shuffle=True,
                            win_size=14,
                            k_num=10000
                        ).split_dataset()


    val_loader  = CustomDatasetDataLoader(  X=GOSE_val, 
                                            Y=StageIV_val,
                                            batchSize=config['batch_size'],
                                            selected_samples=val_samples,  
                                            win_size=14,
                                            nThreads=1,
                                            seed=config['rdm_seed']
                                            )
    
    
    model=IPECNet(nc=[1,16,16,32,32],padding_type='zero',norm_layer=nn.BatchNorm2d,task='estimation')
    model = torch.nn.DataParallel(model.to(device), device_ids=[0,1,2,3])
    optimizer = torch.optim.SGD(model.parameters(),lr=config['lr'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config['lr_stepsize'], gamma=0.1)
    criterion = Estimation_Loss(w=config['w'],
                                h=config['h'],
                                X_min=config['X_min'],
                                X_max=config['X_max'],
                                bins=config['bins']
                                )

    trainlog = {}
    for epoch in tqdm.tqdm(range(1, config['epochs'] + 1)):
        print('EPOCH {}/{}'.format(epoch, config['epochs']))
        print('Train . . . ')
        model.train()
        train_metrics = train_epoch(model, optimizer, scheduler, criterion, train_loader, device=device, config=config)
        print(train_metrics)

        print('Validation . . . ')
        model.eval()
        val_metrics = evaluation(model, criterion, val_loader, device=device, config=config, mode='val')
        print(val_metrics)

        trainlog[epoch] = {**train_metrics, **val_metrics}
        

        if True:
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                        os.path.join(config['res_dir'], 'Epoch_{}.pth.tar'.format(epoch + 1)),
                        )
        
        filename=os.path.join(config['res_dir'], 'train_val_info.json')
        with open(filename,'w') as file_obj:
            json.dump(  {'train_metrics': trainlog,'val_metrics':val_metrics}, file_obj)

    print('Testing best epoch . . .')
    model.load_state_dict(
        torch.load(os.path.join(config['res_dir'], 'Epoch_{}.pth.tar'.format(epoch + 1) ))['state_dict'])
    model.eval()

    test_metrics = evaluation(model, criterion, val_loader, device=device, mode='test', config=config)

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
    parser.add_argument('--res_dir', default='./Estimation/results/like_qinghua', type=str)


    # Training parameters
    parser.add_argument('--epochs', default=30, type=int, help='Number of epochs per fold')
    parser.add_argument('--batch_size', default=1024, type=int, help='Batch size')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')

    # loss parameters
    parser.add_argument('--w', default=1, type=float)
    parser.add_argument('--h', default=2.5, type=float)
    parser.add_argument('--X_min', default=-10, type=float)
    parser.add_argument('--X_max', default=60, type=float)
    parser.add_argument('--bins', default=70, type=int)
    parser.add_argument('--lr_stepsize', default=1000, type=int)
    
    args = parser.parse_args()
    config = args.__dict__

    res=Path(config['res_dir'])
    res.mkdir(parents=True, exist_ok=True )
    pprint.pprint(config)

    filename=os.path.join(config['res_dir'], 'model_param.json')
    with open(filename,'w') as file_obj:
        json.dump(  config,file_obj)

    main(config)