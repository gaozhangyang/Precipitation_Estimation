
import torch
from sklearn.metrics import confusion_matrix
import numpy as np

class BinaryClsMeter(object):
    '''Meters provide a way to keep track of important statistics in an online manner.

    This class is abstract, but provides a standard interface for all meters to follow.

    '''
    def __init__(self):
        self.reset()

    def reset(self):
        '''Resets the meter to default settings.'''
        self.N=0
        self.pos=0
        self.y_true=[]
        self.y_pred=[]


    def add(self, pred,target):
        '''Log a new value to the meter

        Args:
            value: Next restult to include.

        '''
        self.N+=pred.shape[0]
        self.pos+=torch.sum(pred==target).detach().cpu().numpy()
        self.y_true+=target.detach().cpu().numpy().tolist()
        self.y_pred+=pred.detach().cpu().numpy().tolist()



    def value(self):
        '''Get the value of the meter in the current state.'''
        if self.N==0:
            return 0
        else:
            tn, fp, fn, tp=confusion_matrix(self.y_true,self.y_pred ).ravel()
            POD=tp/(tp+fn)
            FAR=fp/(tp+fp)
            CSI=tp/(tp+fn+fp)
            acc0=tp/(tp+fn)
            acc1=tn/(tn+fp)
            return acc0, acc1, POD, FAR, CSI, 