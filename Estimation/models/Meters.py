
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
        self.y_true+=target.view(-1).detach().cpu().numpy().tolist()
        self.y_pred+=pred.view(-1).detach().cpu().numpy().tolist()



    def value(self):
        '''Get the value of the meter in the current state.'''
        if self.N==0:
            return 0
        else:
            conv=np.cov([self.y_true,self.y_pred])
            CC=conv[0,1]/np.sqrt(conv[0,0]*conv[1,1])
            BIAS=np.sum(np.array(self.y_pred)-np.array(self.y_true))/np.sum(self.y_true)
            MSE=np.mean((np.array(self.y_pred)-np.array(self.y_true))**2)
            return CC, BIAS, MSE