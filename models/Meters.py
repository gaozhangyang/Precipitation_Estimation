
import torch

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


    def add(self, pred,target):
        '''Log a new value to the meter

        Args:
            value: Next restult to include.

        '''
        self.N+=pred.shape[0]
        self.pos+=torch.sum(pred==target).detach().cpu().numpy()


    def value(self):
        '''Get the value of the meter in the current state.'''
        if self.N==0:
            return 0
        else:
            return self.pos/self.N