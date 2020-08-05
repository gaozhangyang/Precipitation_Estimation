import numpy as np
from dataloaders import custom_transforms as tr
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split


class MODIS(Dataset):
    def __init__(self, path, train):
        super().__init__()
        self.train = train
        self.modis_dic = np.load(path, allow_pickle='True').item()
        self.X_train, self.y_train, self.X_test, self.y_test = self.data_split(self.modis_dic)

    def __getitem__(self, idx):
        if self.train:
            image = self.X_train[idx]
            target = self.y_train[idx]
        else:
            image = self.X_test[idx]
            target = self.y_test[idx]
        sample = {'image': image, 'label': target}
        return self.transform(sample)

    def __len__(self):

        return len(self.X_train[0])

    def transform(self, sample):

        composed_transforms = transforms.Compose([
            tr.Normalize(
                mean=[4.8003e+00, 2.5270e+03, 4.4159e+02, 2.4647e+00, 1.9367e+04, 7.8704e+00],
                std=[4.5098e+00, 9.4051e+02, 9.4881e+01, 1.4661e+00, 1.4378e+04, 5.9816e+00]
            ),
            tr.ToTensor()
        ])

        return composed_transforms(sample)

    def data_split(self, modis_dic):

        data_set = np.array(list(modis_dic.values()))
        day_idx = np.array(list(modis_dic.keys()))
        images = data_set[:, :-1]
        targets = data_set[:, -1].reshape(-1, 1, 48, 72)

        day_idx = [int(i/31) for i in day_idx]

        X_train, X_test, y_train, y_test = train_test_split(images, targets, test_size=0.2, random_state=0, stratify=day_idx)

        return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    path = 'dataloaders/datasets/dataset.npy'
    modis_train = MODIS(path, False)
    dataloader = DataLoader(modis_train, batch_size=4, shuffle=True, num_workers=0)

    for i, sample in enumerate(dataloader):
        image, target = sample['image'], sample['label']
        print(image.shape)
        print(target.shape)
        break