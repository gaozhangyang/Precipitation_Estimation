import matplotlib.pyplot as plt
from pathlib import Path
import time
import torch
import numpy as np

toCPU=lambda x: x.detach().cpu().numpy()
toCUDA=lambda x: torch.tensor(x).cuda()

def logfunc(config,criterion):
    plt.plot(toCPU(criterion.KDE.centers),toCPU(criterion.P),label='P')
    plt.plot(toCPU(criterion.KDE.centers),toCPU(criterion.Q),label='Q')
    plt.legend()
    figpath=Path(config['res_dir'])/'fig'
    figpath.mkdir(exist_ok=True,parents=True)
    plt.savefig(figpath/'{}.png'.format(int(time.time())))
    plt.close()