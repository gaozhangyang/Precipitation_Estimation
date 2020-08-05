import sys
sys.path.append('/usr/commondata/weather/code/Precipitation_Estimation')
import matplotlib.cm as cm
from matplotlib import colors
import json
from Precipitation.Meters import BinaryClsMeter
import torch
import matplotlib.patches as patches
import Precipitation.Dataloader as Dataloader
from multiprocessing import Pool, Manager, Process
from pathlib import Path
import os
import tqdm
import signal
import numpy as np
import abc
import imageio
import matplotlib.pyplot as plt
import sys
sys.path.append('/usr/commondata/weather/code/Precipitation_Estimation/')


def toCPU(x):
    return x.detach().cpu().numpy()


def toCUDA(x):
    return torch.tensor(x).cuda()


processes = []


def term(sig_num, addtion):
    print('terminate process {}'.format(os.getpid()))
    try:
        print('the processes is {}'.format(processes))
        for p in processes:
            print('process {} terminate'.format(p.pid))
            p.terminate()
            # os.kill(p.pid, signal.SIGKILL)
    except Exception as e:
        print(str(e))


class GIFPloter():
    def __init__(self, root):
        self.root = root

    @abc.abstractmethod
    def callback(self, task, pid, Process_state):
        # TODO 此处增加画图操作
        Process_state[str(pid)] = True

    def run(self, pnum, tasks):
        signal.signal(signal.SIGTERM, term)

        ###############################多线程处理##########################
        Process_state = Manager().dict({str(i): True for i in range(pnum)})
        idx = 0
        while idx < len(tasks):
            # 查询是否有可用线程
            for pid in range(pnum):
                if Process_state[str(pid)] == True:
                    Process_state[str(pid)] = False  # 占用当前线程
                    p = Process(target=self.callback,
                                args=(tasks[idx], pid, Process_state))
                    print(idx)
                    idx += 1
                    p.start()
                    processes.append(p)
                    break

        for p in processes:
            p.join()

    def SaveGIF(self, name, fps=1):
        path = self.root
        gif_images_path = os.listdir(path + '/')

        gif_images_path = [
            img for img in gif_images_path if img[-4:] == '.png'
        ]
        gif_images_path = sorted(gif_images_path, key=lambda x: int(x[:-4]))
        gif_images = []
        for i, path_ in enumerate(gif_images_path):
            print(path_)
            if '.png' in path_:
                if i % 1 == 0:
                    gif_images.append(imageio.imread(path + '/' + path_))

        imageio.mimsave(path + '/' + "{}.mp4".format(name),
                        gif_images,
                        fps=fps)


class Plot_XY(GIFPloter):
    def __init__(self, root):
        super(Plot_XY, self).__init__(root)

    def callback(self, task, pid, Process_state):
        x, y, X, Y, T, row, col, idx = task
        # 大图
        fig = plt.figure(constrained_layout=True, figsize=(30, 25))
        gs = fig.add_gridspec(5, 6)
        H = X[0].shape[0]
        x_pos = col - 15
        y_pos = row - 15

        ax1 = fig.add_subplot(gs[0:2, 0:2])
        ax1.imshow(X[0])
        rect = patches.Rectangle((x_pos, y_pos),
                                 29,
                                 29,
                                 linewidth=1,
                                 edgecolor='r',
                                 facecolor='none')
        ax1.add_patch(rect)

        ax2 = fig.add_subplot(gs[0:2, 2:4])
        ax2.imshow(X[1])
        rect = patches.Rectangle((x_pos, y_pos),
                                 29,
                                 29,
                                 linewidth=1,
                                 edgecolor='r',
                                 facecolor='none')
        ax2.add_patch(rect)

        ax3 = fig.add_subplot(gs[0:2, 4:6])
        ax3.imshow(X[2])
        rect = patches.Rectangle((x_pos, y_pos),
                                 29,
                                 29,
                                 linewidth=1,
                                 edgecolor='r',
                                 facecolor='none')
        ax3.add_patch(rect)

        ax4 = fig.add_subplot(gs[2:4, 0:2])
        ax4.imshow(X[0] - X[1])
        rect = patches.Rectangle((x_pos, y_pos),
                                 29,
                                 29,
                                 linewidth=1,
                                 edgecolor='r',
                                 facecolor='none')
        ax4.add_patch(rect)

        ax5 = fig.add_subplot(gs[2:4, 2:4])
        ax5.imshow(X[1] - X[2])
        rect = patches.Rectangle((x_pos, y_pos),
                                 29,
                                 29,
                                 linewidth=1,
                                 edgecolor='r',
                                 facecolor='none')
        ax5.add_patch(rect)

        ax6 = fig.add_subplot(gs[2:4, 4:6])
        ax6.imshow(Y)
        rect = patches.Rectangle((x_pos, y_pos),
                                 29,
                                 29,
                                 linewidth=1,
                                 edgecolor='r',
                                 facecolor='none')
        ax6.add_patch(rect)

        # 子图
        ax7 = fig.add_subplot(gs[4, 0])
        ax7.imshow(x[0])

        ax8 = fig.add_subplot(gs[4, 1])
        ax8.imshow(x[1])

        ax9 = fig.add_subplot(gs[4, 2])
        ax9.imshow(x[2])

        ax10 = fig.add_subplot(gs[4, 3])
        ax10.imshow(x[0] - x[1])

        ax11 = fig.add_subplot(gs[4, 4])
        ax11.imshow(x[1] - x[2])

        ax12 = fig.add_subplot(gs[4, 5])
        ax12.imshow(y)

        ax9.set_title('{}-{}-{}'.format(T, row, col))
        ax12.set_title('{}'.format(y[14, 14]))

        path = Path(self.root)
        path.mkdir(exist_ok=True, parents=True)
        plt.savefig(path / ('{}.png'.format(idx)))
        plt.close()

        Process_state[str(pid)] = True


class Plot_pred_surface(GIFPloter):
    def __init__(self, root):
        super(Plot_pred_surface, self).__init__(root)

    def callback(self, task, pid, Process_state):
        cmap = 'cool'  # 'cool'
        images = []

        x, pred, y_true, T, H, W, specify_task = task
        N = int(pred.shape[0]**0.5)
        pred = pred.reshape(H, W)
        y_true = y_true.reshape(H, W)

        fig = plt.figure(constrained_layout=True, figsize=(20, 14))
        gs = fig.add_gridspec(14, 20)

        ax1 = fig.add_subplot(gs[0:4, 0:4])
        ax1.imshow(x[0])

        ax2 = fig.add_subplot(gs[0:4, 4:8])
        ax2.imshow(x[1])

        ax3 = fig.add_subplot(gs[0:4, 8:12])
        ax3.imshow(x[2])

        ax4 = fig.add_subplot(gs[0:4, 12:16])
        ax4.imshow(x[0] - x[1])

        ax5 = fig.add_subplot(gs[0:4, 16:20])
        ax5.imshow(x[1] - x[2])

        ax6 = fig.add_subplot(gs[4:14, 0:10])
        images.append(ax6.imshow(y_true))
        images.append(ax6.imshow(y_true, cmap=cmap))

        ax7 = fig.add_subplot(gs[4:14, 10:20])
        images.append(ax7.imshow(pred))
        images.append(ax7.imshow(pred, cmap=cmap))
        ax7.set_title('{}'.format(T))

        # vmin = min(image.get_array().min() for image in images)
        # vmax = max(image.get_array().max() for image in images)
        if specify_task=='estimation':
            norm = colors.Normalize(vmin=0, vmax=20)
        else:
            norm = colors.Normalize(vmin=0, vmax=1)
            
        for im in images:
            im.set_norm(norm)

        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                     ax=[ax6, ax7],
                     orientation='horizontal',
                     fraction=.1)

        path = Path(self.root)
        path.mkdir(exist_ok=True, parents=True)
        plt.savefig(path / ('{}.png'.format(T)))
        plt.close()

        Process_state[str(pid)] = True


class Draw:
    def __init__(self):
        pass

    def generate_XY_MP4(
            self,
            train_path='/usr/commondata/weather/dataset_release/IR_dataset_QingHua/',
            data_X='X_val_hourly.npz',
            data_Y='Y_val_hourly.npz',
            save_path='/usr/commondata/weather/code/Precipitation_Estimation/results/Visualization/val_center_above_10',
            save_name='XY'):
        GOSE_train = np.load(train_path + data_X)['arr_0']
        StageIV_train = np.load(train_path + data_Y)['arr_0']

        train_samples = Dataloader.IR_Split(
            X=GOSE_train,
            Y=StageIV_train,
            task='estimation',
            seed=2020,
            shuffle=True,
            win_size=14,
            R_num=1000,
            NR_num=10,
        ).split_dataset()

        train_loader = Dataloader.CustomDatasetDataLoader(
            X=GOSE_train,
            Y=StageIV_train,
            batchSize=1024,
            selected_samples=train_samples,
            win_size=14,
            nThreads=0,
            seed=2020,
        )

        tasks = []
        idx = 0
        for xs, ys, Ts, rows, cols, Xs, Ys in train_loader:
            for N in range(xs.shape[0]):
                if (ys[N] > 10).any():
                    tasks.append((xs[idx], ys[idx], Xs[idx], Ys[idx], Ts[idx],
                                  rows[idx], cols[idx], idx))
                    idx += 1
                    print(idx)

                if idx == 200:
                    break

            if idx == 200:
                break

        XY_MP4 = Plot_XY(save_path)
        XY_MP4.run(30, tasks)
        XY_MP4.SaveGIF(save_name, fps=0.5)


    def generate_final_surface_MP4(
            self,
            iden_model_path,
            esti_model_path,
            train_path='/usr/commondata/weather/dataset_release/IR_dataset_QingHua/',
            step=14,
            data_X='X_val_hourly.npz',
            data_Y='Y_val_hourly.npz',
            save_path='/usr/commondata/weather/code/Precipitation_Estimation/results/Visualization/pred_val_surface_ex3',
            save_name='pred_val',
            specify_task='estimate',
            logstep=None):
        import os
        from Tools.torchtool import SetSeed

        save_path2 = Path(save_path)
        save_path2.mkdir(parents=True, exist_ok=True)
        filename = os.path.join(save_path, 'setting.json')
        with open(filename, 'w') as file_obj:
            json.dump(
                {
                    'iden_model': iden_model_path,
                    'esti_model': esti_model_path,
                    'data_X': data_X,
                    'data_Y': data_Y,
                    'specify_task': specify_task,
                    'step': step
                }, file_obj)

        SetSeed(2020)

        GOSE = np.load(train_path + data_X)['arr_0'][:800]
        StageIV = np.load(train_path + data_Y)['arr_0'][:800]

        H, W = round((GOSE.shape[2] - 29) / step), round(
            (GOSE.shape[3] - 29) / step)

        iden_meter = BinaryClsMeter(task='identification')
        if specify_task == 'estimation':
            esti_meter = BinaryClsMeter(task='estimation')

        #######################generate samples################
        H = len(range(14, 375 - 15, step))
        L = H**2 * GOSE.shape[0]
        test_X = torch.zeros(L, 3, 29, 29)
        test_Y = np.zeros(L)
        N = 0
        for T in tqdm.tqdm( range(0, GOSE.shape[0]) ):
            X = torch.tensor(GOSE[T]).float().cuda()
            Y = StageIV[T]
            for i in range(14, 375 - 15, step):
                for j in range(14, 375 - 15, step):
                    tmpX = Dataloader.IRDataset.unsafe_crop_center(
                        X, i, j, 14, 14)
                    test_X[N, :, :, :] = tmpX
                    test_Y[N] = Y[i, j]
                    N += 1

        pred, y_true = self.final_test_model(X=test_X,
                                             Y=test_Y,
                                             iden_model_path=iden_model_path,
                                             esti_model_path=esti_model_path,
                                             step=step,
                                             save_path=save_path,
                                             save_name=save_name,
                                             specify_task=specify_task)

        iden_meter.add(pred > 0.1, y_true > 0.1)
        if specify_task == 'estimation':
            esti_meter.add(pred, y_true)

        pred = pred.reshape(GOSE.shape[0], H, H)
        y_true = y_true.reshape(GOSE.shape[0], H, H)
        tasks = []
        if logstep is None:
            logstep = int(GOSE.shape[0] / 150)

        for i in range(0, GOSE.shape[0]):
            if i % logstep == 0:
                print(i)
                tasks.append((GOSE[i], pred[i], y_true[i], i, H, W, specify_task))

        # log test information
        iden_indicate = iden_meter.value()
        metrics = {
            'test_acc0': iden_indicate[0],
            'test_acc1': iden_indicate[1],
            'test_POD': iden_indicate[2],
            'test_FAR': iden_indicate[3],
            'test_CSI': iden_indicate[4],
            'n': iden_meter.n.tolist()
        }

        if specify_task == 'estimation':
            esti_indicate = esti_meter.value()
            metrics = {
                'test_CC': esti_indicate[0],
                'test_BIAS': esti_indicate[1],
                'test_MSE': esti_indicate[2],
                'test_acc0': iden_indicate[0],
                'test_acc1': iden_indicate[1],
                'test_POD': iden_indicate[2],
                'test_FAR': iden_indicate[3],
                'test_CSI': iden_indicate[4],
                'n': iden_meter.n.tolist()
            }

        pred_MP4 = Plot_pred_surface(save_path)
        pred_MP4.run(30, tasks)
        pred_MP4.SaveGIF(save_name, fps=0.5)

        filename = os.path.join(save_path, 'test_info.json')
        with open(filename, 'w') as file_obj:
            json.dump({'test_metrics': metrics}, file_obj)

    def final_test_model(self,
                         X,
                         Y,
                         iden_model_path,
                         esti_model_path,
                         batch_size=100000,
                         step=14,
                         save_path='',
                         save_name='',
                         specify_task='estimation',
                         multi_gpu=True):
        from Precipitation.IPEC_model import IPECNet
        import torch.nn as nn
        from collections import OrderedDict

        ########################load model######################
        iden_model = IPECNet(nc=[1, 16, 16, 32, 32],
                             padding_type='zero',
                             norm_layer=nn.BatchNorm2d,
                             task='identification')
        iden_model = torch.nn.DataParallel(iden_model.to('cuda'),
                                           device_ids=[0,1,2,3,4,5,6,7])
        iden_state_dict = torch.load(iden_model_path)
        if specify_task == 'estimation':
            esti_model = IPECNet(nc=[1, 16, 16, 32, 32],
                                 padding_type='zero',
                                 norm_layer=nn.BatchNorm2d,
                                 task='estimation')
            esti_model = torch.nn.DataParallel(esti_model.to('cuda'),
                                               device_ids=[0,1,2,3,4,5,6,7])
            esti_state_dict = torch.load(esti_model_path)

        if multi_gpu:
            iden_model.load_state_dict(iden_state_dict)
            if specify_task == 'estimation':
                esti_model.load_state_dict(esti_state_dict)
        else:
            # create new OrderedDict that does not contain `module.`
            new_iden_state_dict = OrderedDict()
            for k, v in iden_state_dict.items():
                name = k[7:]  # remove `module.`
                new_iden_state_dict[name] = v
            iden_model.load_state_dict(new_iden_state_dict)

            if specify_task == 'estimation':
                new_esti_state_dict = OrderedDict()
                for k, v in esti_state_dict.items():
                    name = k[7:]  # remove `module.`
                    new_esti_state_dict[name] = v
                esti_model.load_state_dict(new_esti_state_dict)

        iden_model = iden_model.cuda()
        iden_model.eval()
        if specify_task == 'estimation':
            esti_model = esti_model.cuda()
            esti_model.eval()

        #######################get pred###################
        with torch.no_grad():
            pred = []
            L = X.shape[0]
            for i in range(0, L // batch_size + 1):
                scope = range(i * batch_size, min((i + 1) * batch_size, L))
                tmpX = X[scope].float().cuda()

                rain_mask = iden_model(tmpX).detach().cpu().numpy()
                rain_mask = np.argmax(rain_mask, axis=1)
                if specify_task == 'estimation':
                    tmp_pred = esti_model(tmpX).detach().cpu().numpy()
                    tmp_pred = tmp_pred.reshape(-1) * rain_mask.reshape(-1)
                    tmp_pred[tmp_pred < 0.1] = 0
                else:
                    tmp_pred = rain_mask.astype(np.float)
                    Y = (Y > 0.1).astype(np.float)
                pred.append(tmp_pred)

        pred = np.hstack(pred)

        return pred, Y


if __name__ == '__main__':
    draw = Draw()
    draw.generate_XY_MP4(train_path='/usr/commondata/weather/dataset_release/IR_dataset_QingHua/',
                        data_X='X_val_hourly.npz',
                        data_Y='Y_val_hourly.npz',
                        save_path='/usr/commondata/weather/code/Precipitation_Estimation/results/Visualization/val_center_above_10',
                        save_name='XY_val')

    # cmd=[
    #         {'iden_model_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/identification/001/epoch_2_step_2.pt',
    #         'esti_model_path':'',
    #         'train_path':'/usr/commondata/weather/dataset_release/IR_dataset_QingHua/',
    #         'step':14,
    #         'data_X':'X_val_hourly.npz',
    #         'data_Y':'Y_val_hourly.npz',
    #         'save_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/Visualization/iden_val1',
    #         'save_name':'val',
    #         'specify_task':'identification'
    #         },

    #         {'iden_model_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/identification/002/epoch_12_step_12.pt',
    #         'esti_model_path':'',
    #         'train_path':'/usr/commondata/weather/dataset_release/IR_dataset_QingHua/',
    #         'step':14,
    #         'data_X':'X_val_hourly.npz',
    #         'data_Y':'Y_val_hourly.npz',
    #         'save_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/Visualization/iden_val2',
    #         'save_name':'val',
    #         'specify_task':'identification'
    #         },

    #         {'iden_model_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/identification/003/epoch_3_step_3.pt',
    #         'esti_model_path':'',
    #         'train_path':'/usr/commondata/weather/dataset_release/IR_dataset_QingHua/',
    #         'step':14,
    #         'data_X':'X_val_hourly.npz',
    #         'data_Y':'Y_val_hourly.npz',
    #         'save_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/Visualization/iden_val3',
    #         'save_name':'val',
    #         'specify_task':'identification'
    #         },

    #         {'iden_model_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/identification/004/epoch_4_step_4.pt',
    #         'esti_model_path':'',
    #         'train_path':'/usr/commondata/weather/dataset_release/IR_dataset_QingHua/',
    #         'step':14,
    #         'data_X':'X_val_hourly.npz',
    #         'data_Y':'Y_val_hourly.npz',
    #         'save_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/Visualization/iden_val4',
    #         'save_name':'val',
    #         'specify_task':'identification'
    #         },

    #         {'iden_model_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/identification/005/epoch_28_step_28.pt',
    #         'esti_model_path':'',
    #         'train_path':'/usr/commondata/weather/dataset_release/IR_dataset_QingHua/',
    #         'step':14,
    #         'data_X':'X_val_hourly.npz',
    #         'data_Y':'Y_val_hourly.npz',
    #         'save_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/Visualization/iden_val5',
    #         'save_name':'val',
    #         'specify_task':'identification'
    #         },

    #         {'iden_model_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/identification/006/epoch_4_step_4.pt',
    #         'esti_model_path':'',
    #         'train_path':'/usr/commondata/weather/dataset_release/IR_dataset_QingHua/',
    #         'step':14,
    #         'data_X':'X_val_hourly.npz',
    #         'data_Y':'Y_val_hourly.npz',
    #         'save_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/Visualization/iden_val6',
    #         'save_name':'val',
    #         'specify_task':'identification'
    #         },

    #         {'iden_model_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/identification/007/epoch_6_step_6.pt',
    #         'esti_model_path':'',
    #         'train_path':'/usr/commondata/weather/dataset_release/IR_dataset_QingHua/',
    #         'step':14,
    #         'data_X':'X_val_hourly.npz',
    #         'data_Y':'Y_val_hourly.npz',
    #         'save_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/Visualization/iden_val7',
    #         'save_name':'val',
    #         'specify_task':'identification'
    #         },
    #     ]

    # cmd=[
    #         {'iden_model_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/identification/005/epoch_28_step_28.pt',
    #          'esti_model_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/estimation/changew3/001/epoch_59.pt',
    #          'train_path':'/usr/commondata/weather/dataset_release/IR_dataset_QingHua/',
    #          'step':14,
    #          'data_X':'X_val_hourly.npz',
    #          'data_Y':'Y_val_hourly.npz',
    #          'save_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/Visualization/esti_val1',
    #          'save_name':'val',
    #          'specify_task':'identification'
    #         },

    #         {'iden_model_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/identification/005/epoch_28_step_28.pt',
    #          'esti_model_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/estimation/changew3/002/epoch_50.pt',
    #          'train_path':'/usr/commondata/weather/dataset_release/IR_dataset_QingHua/',
    #          'step':14,
    #          'data_X':'X_val_hourly.npz',
    #          'data_Y':'Y_val_hourly.npz',
    #          'save_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/Visualization/esti_val2',
    #          'save_name':'val',
    #          'specify_task':'estimation'
    #         },

    #         {'iden_model_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/identification/005/epoch_28_step_28.pt',
    #          'esti_model_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/estimation/changew3/003/epoch_59.pt',
    #          'train_path':'/usr/commondata/weather/dataset_release/IR_dataset_QingHua/',
    #          'step':14,
    #          'data_X':'X_val_hourly.npz',
    #          'data_Y':'Y_val_hourly.npz',
    #          'save_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/Visualization/esti_val3',
    #          'save_name':'val',
    #          'specify_task':'estimation'
    #         },

    #         {'iden_model_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/identification/005/epoch_28_step_28.pt',
    #          'esti_model_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/estimation/changew3/004/epoch_39.pt',
    #          'train_path':'/usr/commondata/weather/dataset_release/IR_dataset_QingHua/',
    #          'step':14,
    #          'data_X':'X_val_hourly.npz',
    #          'data_Y':'Y_val_hourly.npz',
    #          'save_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/Visualization/esti_val4',
    #          'save_name':'val',
    #          'specify_task':'estimation'
    #         },

    #         {'iden_model_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/identification/005/epoch_28_step_28.pt',
    #          'esti_model_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/estimation/changew3/005/epoch_39.pt',
    #          'train_path':'/usr/commondata/weather/dataset_release/IR_dataset_QingHua/',
    #          'step':14,
    #          'data_X':'X_val_hourly.npz',
    #          'data_Y':'Y_val_hourly.npz',
    #          'save_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/Visualization/esti_val5',
    #          'save_name':'val',
    #          'specify_task':'estimation'
    #         },

    #         {'iden_model_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/identification/005/epoch_28_step_28.pt',
    #          'esti_model_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/estimation/changew3/006/epoch_15.pt',
    #          'train_path':'/usr/commondata/weather/dataset_release/IR_dataset_QingHua/',
    #          'step':14,
    #          'data_X':'X_val_hourly.npz',
    #          'data_Y':'Y_val_hourly.npz',
    #          'save_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/Visualization/esti_val6',
    #          'save_name':'val',
    #          'specify_task':'estimation'
    #         },

    #         {'iden_model_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/identification/005/epoch_28_step_28.pt',
    #          'esti_model_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/estimation/changew3/007/epoch_45.pt',
    #          'train_path':'/usr/commondata/weather/dataset_release/IR_dataset_QingHua/',
    #          'step':14,
    #          'data_X':'X_val_hourly.npz',
    #          'data_Y':'Y_val_hourly.npz',
    #          'save_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/Visualization/esti_val7',
    #          'save_name':'val',
    #          'specify_task':'estimation'
    #         },

    #         {'iden_model_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/identification/005/epoch_28_step_28.pt',
    #          'esti_model_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/estimation/changew3/008/epoch_15.pt',
    #          'train_path':'/usr/commondata/weather/dataset_release/IR_dataset_QingHua/',
    #          'step':14,
    #          'data_X':'X_val_hourly.npz',
    #          'data_Y':'Y_val_hourly.npz',
    #          'save_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/Visualization/esti_val8',
    #          'save_name':'val',
    #          'specify_task':'estimation'
    #         },

    #         {'iden_model_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/identification/005/epoch_28_step_28.pt',
    #          'esti_model_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/estimation/changew3/009/epoch_28.pt',
    #          'train_path':'/usr/commondata/weather/dataset_release/IR_dataset_QingHua/',
    #          'step':14,
    #          'data_X':'X_val_hourly.npz',
    #          'data_Y':'Y_val_hourly.npz',
    #          'save_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/Visualization/esti_val9',
    #          'save_name':'val',
    #          'specify_task':'estimation'
    #         },

    #         {'iden_model_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/identification/005/epoch_28_step_28.pt',
    #          'esti_model_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/estimation/changew3/010/epoch_27.pt',
    #          'train_path':'/usr/commondata/weather/dataset_release/IR_dataset_QingHua/',
    #          'step':14,
    #          'data_X':'X_val_hourly.npz',
    #          'data_Y':'Y_val_hourly.npz',
    #          'save_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/Visualization/esti_val10',
    #          'save_name':'val',
    #          'specify_task':'estimation'
    #         },

    #         {'iden_model_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/identification/005/epoch_28_step_28.pt',
    #          'esti_model_path':'//usr/commondata/weather/code/Precipitation_Estimation/results/estimation/changew3/011/epoch_29.pt',
    #          'train_path':'/usr/commondata/weather/dataset_release/IR_dataset_QingHua/',
    #          'step':14,
    #          'data_X':'X_val_hourly.npz',
    #          'data_Y':'Y_val_hourly.npz',
    #          'save_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/Visualization/esti_val11',
    #          'save_name':'val',
    #          'specify_task':'estimation'
    #         },

    #         {'iden_model_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/identification/005/epoch_28_step_28.pt',
    #          'esti_model_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/estimation/changew3/012/epoch_75.pt',
    #          'train_path':'/usr/commondata/weather/dataset_release/IR_dataset_QingHua/',
    #          'step':14,
    #          'data_X':'X_val_hourly.npz',
    #          'data_Y':'Y_val_hourly.npz',
    #          'save_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/Visualization/esti_val12',
    #          'save_name':'val',
    #          'specify_task':'estimation'
    #         },

    #         {'iden_model_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/identification/005/epoch_28_step_28.pt',
    #          'esti_model_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/estimation/changew3/013/epoch_75.pt',
    #          'train_path':'/usr/commondata/weather/dataset_release/IR_dataset_QingHua/',
    #          'step':14,
    #          'data_X':'X_val_hourly.npz',
    #          'data_Y':'Y_val_hourly.npz',
    #          'save_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/Visualization/esti_val13',
    #          'save_name':'val',
    #          'specify_task':'estimation'
    #         },

    #         {'iden_model_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/identification/005/epoch_28_step_28.pt',
    #          'esti_model_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/estimation/changew3/014/epoch_59.pt',
    #          'train_path':'/usr/commondata/weather/dataset_release/IR_dataset_QingHua/',
    #          'step':14,
    #          'data_X':'X_val_hourly.npz',
    #          'data_Y':'Y_val_hourly.npz',
    #          'save_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/Visualization/esti_val14',
    #          'save_name':'val',
    #          'specify_task':'estimation'
    #         },

    #         {'iden_model_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/identification/005/epoch_28_step_28.pt',
    #          'esti_model_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/estimation/changew3/015/epoch_75.pt',
    #          'train_path':'/usr/commondata/weather/dataset_release/IR_dataset_QingHua/',
    #          'step':14,
    #          'data_X':'X_val_hourly.npz',
    #          'data_Y':'Y_val_hourly.npz',
    #          'save_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/Visualization/esti_val15',
    #          'save_name':'val',
    #          'specify_task':'estimation'
    #         },
    #     ]

    # cmd=[
    #         {'iden_model_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/identification/005/epoch_28_step_28.pt',
    #         'esti_model_path':'',
    #         'train_path':'/usr/commondata/weather/dataset_release/IR_dataset_QingHua/',
    #         'step':14,
    #         'data_X':'X_val_hourly.npz',
    #         'data_Y':'Y_val_hourly.npz',
    #         'save_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/Visualization/final_iden_val',
    #         'save_name':'val',
    #         'specify_task':'identification',
    #         'logstep':1
    #         },

    #         {'iden_model_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/identification/005/epoch_28_step_28.pt',
    #          'esti_model_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/estimation/changew3/013/epoch_75.pt',
    #          'train_path':'/usr/commondata/weather/dataset_release/IR_dataset_QingHua/',
    #          'step':14,
    #          'data_X':'X_val_hourly.npz',
    #          'data_Y':'Y_val_hourly.npz',
    #          'save_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/Visualization/final_esti_val',
    #          'save_name':'val',
    #          'specify_task':'estimation',
    #          'logstep':1
    #         },
    # ]

    # for i in range(0,2):
    #     draw.generate_final_surface_MP4(**cmd[i])