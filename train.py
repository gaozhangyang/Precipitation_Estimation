import torch
import torch.utils.data as data
import torchnet as tnt
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import random
import os
import json
import pickle as pkl
import argparse
import pprint
from models.Dataloader import CustomDatasetDataLoader
from models.IPEC_model import IPECNet
from loss.Loss import KL_loss
import torch.nn as nn
import torch.nn.functional as F
from models.Meters import BinaryClsMeter
from pathlib import Path


def SetSeed(seed):
    """function used to set a random seed
    Arguments:
        seed {int} -- seed number, will set to torch, random and numpy
    """
    SEED = seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

def train_epoch(model, optimizer, criterion, data_loader, device, config):
    loss_meter = tnt.meter.AverageValueMeter()
    y_true = []
    y_pred = []

    for idx, (x,y,i,j,key) in enumerate(data_loader):
        x = x.to(device).float()
        y = y.to(device).float().view(-1,1)

        optimizer.zero_grad()
        x[torch.isnan(x)]=0
        out = model(x)

        loss = criterion(F.sigmoid(out).view(-1),(y>0.1).float().view(-1))
        loss.backward()
        optimizer.step()
        pred = out.detach()

        loss_meter.add(loss.item())

        if (idx + 1) % config['display_step'] == 0:
            print('Step [{}/{}], Loss: {:.4f}'.format(idx + 1, len(data_loader), loss_meter.value()[0]))

    epoch_metrics = {'train_loss': loss_meter.value()[0]}

    return epoch_metrics


def evaluation(model, criterion, loader, device, config, mode='val'):
    y_true = []
    y_pred = []

    acc_meter = BinaryClsMeter()
    loss_meter = tnt.meter.AverageValueMeter()

    for idx, (x,y,i,j,key) in enumerate(loader):
        y_true.extend(list(map(int, y)))
        x = x.to(device).float()
        y = y.to(device).float().view(-1,1)

        with torch.no_grad():
            prediction = F.sigmoid( model(x) )
            loss = criterion( prediction, (y>0.1).float().view(-1))

        acc_meter.add(prediction>0.5, y>0.1)
        loss_meter.add(loss.item())

    metrics = {'{}_loss'.format(mode): loss_meter.value()[0],
               '{}_score'.format(mode): acc_meter.value()}

    if mode == 'val':
        return metrics
    elif mode == 'test':
        return metrics



def save_results(epoch, metrics, config):
    with open(os.path.join(config['res_dir'], 'epoch_{}'.format(epoch), 'test_metrics.json'), 'w') as outfile:
        json.dump(metrics, outfile, indent=4)



def main(config):
    device = torch.device(config['device'])
    train_loader=CustomDatasetDataLoader(batchSize=config['batch_size'], task='identification', mode='train')
    val_loader=CustomDatasetDataLoader(batchSize=config['batch_size'], task='identification',mode='val')
    test_loader=CustomDatasetDataLoader(batchSize=config['batch_size'], task='identification',mode='test')

    model=IPECNet(nc=[1,16,16,32,32],padding_type='zero',norm_layer=nn.BatchNorm2d)
    model = torch.nn.DataParallel(model.to(device), device_ids=[0,1,2,3])
    optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'])
    criterion = nn.BCELoss()
    # criterion = nn.CrossEntropyLoss()
    # criterion = KL_loss(w=0)

    trainlog = {}
    best_score = 0
    for epoch in range(1, config['epochs'] + 1):
        print('EPOCH {}/{}'.format(epoch, config['epochs']))

        model.train()
        train_metrics = train_epoch(model, optimizer, criterion, train_loader, device=device, config=config)

        print('Validation . . . ')
        model.eval()
        val_metrics = evaluation(model, criterion, val_loader, device=device, config=config, mode='val')

        print('Loss {:.4f},  Score {:.2f}'.format(val_metrics['val_loss'], val_metrics['val_score']))

        trainlog[epoch] = {**train_metrics, **val_metrics}
        

        if True:
            best_score = val_metrics['val_score']
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                        os.path.join(config['res_dir'], 'Epoch_{}.pth.tar'.format(epoch + 1)))

    print('Testing best epoch . . .')
    model.load_state_dict(
        torch.load(os.path.join(config['res_dir'], 'Epoch_{}.pth.tar'.format(epoch + 1) ))['state_dict'])
    model.eval()

    test_metrics = evaluation(model, criterion, test_loader, device=device, mode='test', config=config)

    print('Loss {:.4f},  Acc {:.2f}'.format(test_metrics['test_loss'], test_metrics['test_score']))


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
    parser.add_argument('--epochs', default=1, type=int, help='Number of epochs per fold')
    parser.add_argument('--batch_size', default=1024, type=int, help='Batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    
    args = parser.parse_args()
    config = args.__dict__
    res=Path(config['res_dir'])
    res.mkdir(parents=True, exist_ok=True )
    SetSeed(config['rdm_seed'])
    pprint.pprint(config)
    main(config)