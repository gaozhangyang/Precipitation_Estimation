import numpy as np
import random
import torch


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

from torch.utils.data import Dataset

class IR_Split:
    def __init__(self,
                 X,
                 Y,
                 task='identification',
                 shuffle=False,
                 seed=2020,
                 win_size=14,
                 sampling_step=7,
                 R_num=10000,
                 NR_num=10000,
                 evaluate=False):
        self.X = X
        self.Y = Y
        self.win_size = win_size
        self.task = task
        self.shuffle = shuffle
        self.R_num = R_num
        self.NR_num = NR_num
        self.evaluate = evaluate
        self.sampling_step = sampling_step

    def split_R_NR(self, StageIV):
        '''
        generate index of R/NR samples
        '''
        R_samples = []
        NR_samples = []

        for T in range(StageIV.shape[0]):
            for row in range(self.win_size,
                             StageIV.shape[1] - self.win_size - 1,
                             self.sampling_step):
                for col in range(self.win_size,
                                 StageIV.shape[2] - self.win_size - 1,
                                 self.sampling_step):
                    if StageIV[T, row, col] > 0.1:
                        R_samples.append((T, row, col))
                    else:
                        NR_samples.append((T, row, col))

        R_samples = np.array(R_samples)
        NR_samples = np.array(NR_samples)

        return R_samples, NR_samples


    def split_dataset(self):
        '''
        generate samples
        '''
        self.R_samples, self.NR_samples = self.split_R_NR(self.Y)

        if self.task == 'identification':
            if self.evaluate:
                self.samples = np.vstack([self.R_samples, self.NR_samples])
            else:
                if self.NR_num > 0:
                    self.samples = np.vstack([
                        np.array(random.choices(self.R_samples, k=self.R_num)),
                        np.array(random.choices(self.NR_samples,
                                                k=self.NR_num))
                    ])
                else:
                    self.samples = np.array(
                        random.choices(self.R_samples, k=self.R_num))

        if self.task == 'estimation':
            if self.evaluate:
                self.samples = np.vstack([self.R_samples, self.NR_samples])
            else:
                self.samples = np.array(
                    random.choices(self.R_samples, k=self.R_num))

        if self.shuffle:
            np.random.shuffle(self.samples)
        L = len(self.samples)

        return self.samples


class IRDataset(Dataset):
    def __init__(self, samples, X, Y, win_size=14, seed=2020):
        self.X = X
        self.Y = Y
        self.win_size = win_size
        self.samples = samples
        self.L = len(self.samples)


    @classmethod
    def safe_crop_center(self, img, x, y, cropx, cropy):
        '''
        get cropping images with boundary constraints
        '''
        startx = x - (cropx)
        endx = x + (cropx) + 1
        starty = y - (cropy)
        endy = y + (cropy) + 1

        if len(img.shape) == 3:
            _, H, W = img.shape
            if startx < 0 or starty < 0 or endx >= H or endy >= H:
                return None
            return img[:, startx:endx, starty:endy]

        if len(img.shape) == 2:
            H, W = img.shape
            if startx < 0 or starty < 0 or endx >= H or endy >= H:
                return None
            return img[startx:endx, starty:endy]


    @classmethod
    def unsafe_crop_center(self, img, x, y, cropx, cropy):
        '''
        get cropping images without boundary constraints
        '''
        startx = x - (cropx)
        endx = x + (cropx) + 1
        starty = y - (cropy)
        endy = y + (cropy) + 1
        if len(img.shape) == 2:
            return img[startx:endx, starty:endy]

        if len(img.shape) == 3:
            return img[:, startx:endx, starty:endy]

    def __getitem__(self, idx):
        T, row, col = self.samples[idx]
        X_croped = self.unsafe_crop_center(self.X[T], row, col, self.win_size,
                                           self.win_size)
        # Y_croped=self.Y[T,row,col]
        Y_croped = self.unsafe_crop_center(self.Y[T], row, col, self.win_size,
                                           self.win_size)
        return X_croped, Y_croped, T, row, col, self.X[T], self.Y[T]

    def __len__(self):
        return self.L

    def name(self):
        return 'IRDataset'


class CustomDatasetDataLoader(object):
    def __init__(self,
                 X,
                 Y,
                 batchSize,
                 selected_samples,
                 win_size,
                 nThreads=8,
                 seed=2020):
        self.dataset = IRDataset(selected_samples,
                                 X=X,
                                 Y=Y,
                                 win_size=win_size,
                                 seed=seed)
        self.batchSize = batchSize

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batchSize,
            shuffle=True,
            num_workers=int(nThreads),
            drop_last=False)


    def __iter__(self):
        return self.dataloader.__iter__()


    def __len__(self):
        return self.dataloader.__len__()


    def name(self):
        return 'CustomDatasetDataLoader'
