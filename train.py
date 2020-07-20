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
from loss.Loss import KL_loss
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from models.Dataloader import CustomDatasetDataLoader,IR_Split
from models.IPEC_model import IPECNet
from models.Meters import BinaryClsMeter
import json
import tqdm

OneHot=lambda label,C: torch.zeros(label.shape[0],C).scatter_(1,label.view(-1,1),1)

train_path='/usr/commondata/weather/dataset_release/IR_dataset_QingHua/'
GOSE_train=np.load(train_path+'X_train_hourly.npz')['arr_0']
StageIV_train=np.load(train_path+'Y_train_hourly.npz')['arr_0']

train_path='/usr/commondata/weather/dataset_release/IR_dataset_QingHua/'
GOSE_val=np.load(train_path+'X_val_hourly.npz')['arr_0']
StageIV_val=np.load(train_path+'Y_val_hourly.npz')['arr_0']


def train_epoch(model, optimizer, criterion, data_loader, device, config):
    acc_meter = BinaryClsMeter()
    loss_meter = tnt.meter.AverageValueMeter()
    y_true = []
    y_pred = []

    for idx, (x,y,i,j,key) in enumerate(data_loader):
        x = x.to(device).float()
        y = (y>0.1).to(device).long().view(-1)

        optimizer.zero_grad()
        x[torch.isnan(x)]=0
        out = model(x)

        # loss = criterion( out.view(-1),(y>0.1).float().view(-1) )
        loss = criterion(out,y)
        loss.backward()
        optimizer.step()

        acc_meter.add(torch.argmax(out,dim=1), y)
        loss_meter.add(loss.item())

        if (idx + 1) % config['display_step'] == 0:
            print('Step [{}/{}], Loss: {:.4f}'.format(idx + 1, len(data_loader), loss_meter.value()[0]))

    indicate=acc_meter.value()
    epoch_metrics = {'train_loss': loss_meter.value()[0],
                     'train_acc0': indicate[0],
                     'train_acc1': indicate[1],
                     'train_POD': indicate[2],
                     'train_FAR': indicate[3],
                     'train_CSI': indicate[4]
                     }

    return epoch_metrics


def evaluation(model, criterion, loader, device, config, mode='val'):
    y_true = []
    y_pred = []

    acc_meter = BinaryClsMeter()
    loss_meter = tnt.meter.AverageValueMeter()

    for idx, (x,y,T,row,col) in enumerate(loader):
        y_true.extend(list(map(int, y)))
        x = x.to(device).float()
        y = (y>0.1).to(device).long().view(-1)

        with torch.no_grad():
            x[torch.isnan(x)]=0
            out = model(x)
            loss = criterion(out,y.long())

        acc_meter.add(torch.argmax(out,dim=1), y)
        loss_meter.add(loss.item())

    indicate=acc_meter.value()
    metrics = {'{}_loss'.format(mode): loss_meter.value()[0],
               '{}_train_acc0'.format(mode): indicate[0],
               '{}_train_acc1'.format(mode): indicate[1],
               '{}_POD'.format(mode): indicate[2],
               '{}_FAR'.format(mode): indicate[3],
               '{}_CSI'.format(mode): indicate[4],
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
                                task='identification',
                                seed=config['rdm_seed'],
                                shuffle=True,
                                win_size=14,
                                k_num=340000
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
                            task='identification',
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

    samples1=[]
    for x,y,T,row,col in train_loader:
        samples1.append((T,row,col))
    

    samples2=[]
    for x,y,T,row,col in train_loader:
        samples2.append((T,row,col))
    
    
    model=IPECNet(nc=[1,16,16,32,32],padding_type='zero',norm_layer=nn.BatchNorm2d,task='identification')
    model = torch.nn.DataParallel(model.to(device), device_ids=[0,1,2,3])
    optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'])
    criterion = nn.CrossEntropyLoss()

    trainlog = {}
    for epoch in tqdm.tqdm(range(1, config['epochs'] + 1)):
        print('EPOCH {}/{}'.format(epoch, config['epochs']))
        print('Train . . . ')
        model.train()
        train_metrics = train_epoch(model, optimizer, criterion, train_loader, device=device, config=config)
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

    print('Testing best epoch . . .')
    model.load_state_dict(
        torch.load(os.path.join(config['res_dir'], 'Epoch_{}.pth.tar'.format(epoch + 1) ))['state_dict'])
    model.eval()

    test_metrics = evaluation(model, criterion, val_loader, device=device, mode='test', config=config)

    print(test_metrics)

    filename=os.path.join(config['res_dir'], 'loginfo.json')
    with open(filename,'w') as file_obj:
        json.dump(  {'train_metrics': trainlog,'test_metrics':test_metrics},
                    file_obj)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str,
                        help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--rdm_seed', default=1, type=int, help='Random seed')
    parser.add_argument('--display_step', default=10, type=int,
                        help='Interval in batches between display of training metrics')
    parser.add_argument('--res_dir', default='./results', type=str)


    # Training parameters
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs per fold')
    parser.add_argument('--batch_size', default=1024, type=int, help='Batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    
    args = parser.parse_args()
    config = args.__dict__
    res=Path(config['res_dir'])
    res.mkdir(parents=True, exist_ok=True )
    pprint.pprint(config)
    main(config)