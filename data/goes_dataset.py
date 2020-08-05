import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader


class GOES(Dataset):
    def __init__(self, path:str, opt):
        super().__init__()
        self.X = np.load(path+'X_{}_hourly.npz'.format(opt.mode))['arr_0']
        self.Y = np.load(path+'Y_{}_hourly.npz'.format(opt.mode))['arr_0']
        
        # def discretize(y_raw):
        #         y = np.zeros_like(y_raw, dtype=np.int)
        #         y[y_raw == 0] = 0
        #         y[(y_raw > 0) & (y_raw <= 2.5)] = 1
        #         y[(y_raw > 2.5) & (y_raw <= 8.0)] = 2
        #         y[(y_raw > 8.0) & (y_raw <= 16.0)] = 3
        #         y[(y_raw > 16.0)] = 4
        #         return y

        def discretize(y_raw):
            y = np.zeros_like(y_raw, dtype=np.int)
            y[y_raw <= 0.1] = 0
            y[y_raw > 0.1] = 1
            return y
        
        self.Y = discretize(self.Y)

        self.label2color = np.array([(0, 0, 0),
                                     (135, 206, 250),
                                     (135, 206, 235),
                                     (0, 191, 255),
                                     (30, 144, 255),
                                     (65, 105, 225),
                                     (0, 0, 255)])

        self.label2name = np.array(['0', '1', '2', '3', '4', '5', '6'])

    def __getitem__(self, index: int):
        image, target = torch.FloatTensor(self.X[index]), torch.FloatTensor(self.Y[index])
        sample = {'A': image, 'B': target}
        sample['issup'] = True
        return sample

    def __len__(self):
        return len(self.X)
    
    def name(self):
        return 'GOESDataset'

if __name__ == "__main__":
    path = '/usr/commondata/weather/dataset_release/IR_dataset_QingHua/'
    train_set = GOES(path, 'train')
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=1)
    for i, data in enumerate(train_loader):
        X, y = data[0], data[1]
        print(X.shape, y.shape)
        break