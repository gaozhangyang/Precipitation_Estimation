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
        if (y<0).any():
            print('NO')

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(y, out)
        loss.backward()
        optimizer.step()

        pred = out.detach()

        loss_meter.add(loss.item())

        if (idx + 1) % config['display_step'] == 0:
            print('Step [{}/{}], Loss: {:.4f}'.format(idx + 1, len(data_loader), loss_meter.value()[0]))

    epoch_metrics = {'train_loss': loss_meter.value()[0]}

    return epoch_metrics


# def evaluation(model, criterion, loader, device, config, mode='val'):
#     y_true = []
#     y_pred = []

#     acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
#     loss_meter = tnt.meter.AverageValueMeter()

#     for (x, y) in loader:
#         y_true.extend(list(map(int, y)))
#         x = x.to(device)
#         y = y.to(device)

#         with torch.no_grad():
#             prediction = model(x)
#             loss = criterion(prediction, y)

#         acc_meter.add(prediction, y)
#         loss_meter.add(loss.item())

#         y_p = prediction.argmax(dim=1).cpu().numpy()
#         y_pred.extend(list(y_p))

#     metrics = {'{}_loss'.format(mode): loss_meter.value()[0]}

#     if mode == 'val':
#         return metrics
#     elif mode == 'test':
#         return metrics


def save_results(epoch, metrics, config):
    with open(os.path.join(config['res_dir'], 'epoch_{}'.format(epoch), 'test_metrics.json'), 'w') as outfile:
        json.dump(metrics, outfile, indent=4)



def main(config):
    device = torch.device(config['device'])
    train_loader=CustomDatasetDataLoader(batchSize=config['batch_size'],mode='train')
    # val_loader=CustomDatasetDataLoader(mode='val')
    # test_loader=CustomDatasetDataLoader(mode='test')

    model=IPECNet(nc=[1,16,16,32,32],padding_type='zero',norm_layer=nn.BatchNorm2d)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'])
    # criterion = nn.CrossEntropyLoss()
    criterion = KL_loss(w=0)

    trainlog = {}
    best_score = 0
    for epoch in range(1, config['epochs'] + 1):
        print('EPOCH {}/{}'.format(epoch, config['epochs']))

        model.train()
        train_metrics = train_epoch(model, optimizer, criterion, train_loader, device=device, config=config)

        # print('Validation . . . ')
        # model.eval()
        # val_metrics = evaluation(model, criterion, val_loader, device=device, config=config, mode='val')

        # print('Loss {:.4f},  Score {:.2f}'.format(val_metrics['val_loss'], val_metrics['score']))

        # trainlog[epoch] = {**train_metrics, **val_metrics}
        

        # if val_metrics['Acc'] >= best_score:
        #     best_score = val_metrics['score']
        #     torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
        #                 'optimizer': optimizer.state_dict()},
        #                 os.path.join(config['res_dir'], 'Epoch_{}'.format(epoch + 1), 'model.pth.tar'))

    # print('Testing best epoch . . .')
    # model.load_state_dict(
    #     torch.load(os.path.join(config['res_dir'], 'Epoch_{}'.format(epoch + 1), 'model.pth.tar'))['state_dict'])
    # model.eval()

    # test_metrics = evaluation(model, criterion, test_loader, device=device, mode='test', config=config)

    # print('Loss {:.4f},  Acc {:.2f}'.format(test_metrics['test_loss'], test_metrics['test_accuracy']))



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str,
                        help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--rdm_seed', default=1, type=int, help='Random seed')
    parser.add_argument('--display_step', default=10, type=int,
                        help='Interval in batches between display of training metrics')


    # Training parameters
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs per fold')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    
    args = parser.parse_args()
    config = args.__dict__
    SetSeed(config['rdm_seed'])
    pprint.pprint(config)
    main(config)