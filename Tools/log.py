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

def plot_estimation(pred,true):
    from matplotlib import colors

    fig= plt.figure(constrained_layout=True,figsize=(8, 8))
    gs = fig.add_gridspec(2,2)

    images=[]
    cmap='rainbow'

    ax1=fig.add_subplot(gs[0:1, 0:1])
    images.append( ax1.imshow(pred.reshape(32,32),cmap=cmap) )

    ax2=fig.add_subplot(gs[0:1, 1:2])
    images.append( ax2.imshow(true.reshape(32,32),cmap=cmap) )

    ax3=fig.add_subplot(gs[1:2, 0:1])
    images.append( ax3.imshow((pred-true).reshape(32,32),cmap=cmap) )

    ax4=fig.add_subplot(gs[1:2, 1:2])
    ax4.hist(pred-true)

    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for im in images:
        im.set_norm(norm)

    fig.colorbar(images[0], ax=[ax1,ax2,ax3,ax4], orientation='horizontal', fraction=.1)
    return fig