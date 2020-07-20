from ftplib import FTP  
from fnmatch import fnmatch
from math import radians
from multiprocessing import Pool,Manager,Process
from tokenize import group
from satpy.dataset import dataset_walker
from scipy import interpolate
from pathlib import Path
from satpy.scene import Scene
from datetime import date
import os
import glob
import tqdm
import time
import argparse
import signal
import numpy as np
import warnings
import pygrib
warnings.filterwarnings("ignore")


processes=[]

def dotask(args,task,pid,Process_state): 
# def dotask(args,task): 
    #-------------此处写下对于task的操作
    # TODO
    if args.mode == 'IR_2012':
        data, file_name = cat_channel(args.read_root, task, args.long_range, args.lat_range)
    elif args.mode =='StageIV':
        data, file_name = get_proj_data_StageIV(args.read_root, task, args.long_range, args.lat_range)
    else:
        raise Exception('No Implementation')
    np.savez_compressed(args.save_root+file_name, data)
    #-------------释放当前的线程
    Process_state[str(pid)]=True


def term(sig_num, addtion):
    print('terminate process {}'.format(os.getpid()))
    try:
        print('the processes is {}'.format(processes) )
        for p in processes:
            print('process {} terminate'.format(p.pid))
            p.terminate()
            # os.kill(p.pid, signal.SIGKILL)
    except Exception as e:
        print(str(e))


def run(args):

    ###############################多线程处理##########################
    Process_state=Manager().dict({str(i):True for i in range(args.pnum)})
    filenames = os.listdir(args.read_root) # list all files
    processed = [x[:10] for x in os.listdir(args.save_root)]

    if args.mode == 'IR_2012':
        groups = {}
        for f in filenames:
            name_info = f.split('.')
            key = name_info[2]+'.'+name_info[3]
            if key not in groups.keys():
                groups[key] = []
                groups[key].append(f)
            else:
                groups[key].append(f)
        tasks = list(groups.values())

    elif args.mode == 'StageIV':
        tasks = os.listdir(args.read_root)

    # dotask(args, tasks[0])

    idx=0
    while idx<len(tasks):
        #查询是否有可用线程
        for pid in range(args.pnum):
            if Process_state[str(pid)]==True:
                Process_state[str(pid)]=False #占用当前线程
                p=Process(target=dotask,args=(args,tasks[idx],pid,Process_state))
                idx+=1
                p.start()
                processes.append(p)
                break

    for p in processes:
        p.join()
    
    while(1):
        check1=(np.array(Process_state.values())==True).all()
        time.sleep(1)
        check2=(np.array(Process_state.values())==True).all()
        if (check1==True) and (check2==True):
            print('-----------end------------')
            break

#------------------------------------------------------------------------------

def cat_channel(dir_path: str ,group: list, long_range: tuple, lat_range: tuple):

    data = np.zeros((3, 375, 875))
    new_name = ''
    for filename in group:
        file_info= filename.split('.')
        new_name = file_info[2]+'.'+file_info[3]+'.GOSE'
        channel = int(file_info[4][-2:])
        channel = channel_map(channel)
        print(filename)

        radiance = get_proj_data_GOSE(dir_path+filename, long_range, lat_range)
        data[channel] = radiance
    return data, new_name

def date2num(start_date, end_date):
    ed = date(int(end_date[:4]), int(end_date[4:6]), int(end_date[6:8]))
    delta = (ed - start_date)
    day=str(delta.days)
    hour=end_date[8:]
    result = day+'.'+hour
    return result

def get_proj_data_GOSE(filename: str, long_range: tuple, lat_range: tuple):
    # read nc data
    scn = Scene(filenames=[filename], reader='goes-imager_nc')

    available = scn.available_dataset_names()
    scn.load(available)
    radiance = scn[available[0]].values
    lat = scn[available[1]].values
    long = scn[available[2]].values
    
    # select Reg
    mask_lat_C = (np.min(lat_range) <= lat) *  (lat <= np.max(lat_range))
    mask_long_C = (np.min(long_range) <= long) *  (long <= np.max(long_range))
    mask = mask_lat_C * mask_long_C
    y, x = np.nonzero(mask) # indices of selected region
    
    # proj to regular grid
    toGridIndex = lambda x, rang: ((x - x.min()) / (x.max() - x.min()) * rang).astype(int)
    width = int((np.max(lat_range) - np.min(lat_range)) / 0.04)
    length = int((np.max(long_range) - np.min(long_range)) / 0.04)
    lons_grid_index = toGridIndex(long[mask], length)
    lats_grid_index = toGridIndex(lat[mask], width)
    grid = np.zeros((lats_grid_index.max(), lons_grid_index.max())) # regular grid
    N = np.zeros_like(grid) # counter
    
    for i in range(x.shape[0]):
        grid[lats_grid_index[i]-1, lons_grid_index[i]-1] += radiance[y[i],x[i]]
        N[lats_grid_index[i]-1, lons_grid_index[i]-1] += 1
    projected = np.flip(grid / (N+1e-10), 0)
    
    # interpolation with linear method
    a, b = np.nonzero(projected)
    values = projected[a, b]
    grid_x, grid_y = np.mgrid[0:374:375j, 0:874:875j]
    grid_z = interpolate.griddata((a, b), values, (grid_x, grid_y), method='linear')
    return grid_z

def get_proj_data_StageIV(dir_path: str, filename: str, long_range: tuple, lat_range: tuple):
    gr = pygrib.open(dir_path+filename)
    data = gr[1].values.data
    data[data > 9000] = 0
    lat = np.array(gr[1].latlons()[0])
    long = np.array(gr[1].latlons()[1])

    start_date = date(2011, 12, 31)
    time = date2num(start_date, filename.split('.')[1])
    file_name = time+'.'+'StageIV'
    print(file_name)

    # select Reg
    mask_lat_C = (np.min(lat_range) <= lat) *  (lat <= np.max(lat_range))
    mask_long_C = (np.min(long_range) <= long) *  (long <= np.max(long_range))
    mask = mask_lat_C * mask_long_C
    y, x = np.nonzero(mask) # indices of selected region
    
    # proj to regular grid
    toGridIndex = lambda x, rang: ((x - x.min()) / (x.max() - x.min()) * rang).astype(int)
    width = int((np.max(lat_range) - np.min(lat_range)) / 0.04)
    length = int((np.max(long_range) - np.min(long_range)) / 0.04)
    lons_grid_index = toGridIndex(long[mask], length)
    lats_grid_index = toGridIndex(lat[mask], width)
    grid = np.zeros((lats_grid_index.max(), lons_grid_index.max())) # regular grid
    grid[grid == 0] = -1
    N = np.zeros_like(grid) # counter
    
    for i in range(x.shape[0]):
        if grid[lats_grid_index[i]-1, lons_grid_index[i]-1] == -1:
            grid[lats_grid_index[i]-1, lons_grid_index[i]-1] = 0
            grid[lats_grid_index[i]-1, lons_grid_index[i]-1] += data[y[i],x[i]]
            N[lats_grid_index[i]-1, lons_grid_index[i]-1] += 1
        else:
            grid[lats_grid_index[i]-1, lons_grid_index[i]-1] += data[y[i],x[i]]
            N[lats_grid_index[i]-1, lons_grid_index[i]-1] += 1
    
    projected = grid / (N + 1e-10)
    mask = projected < 0
    projected[mask] = 0
    
    # interpolation with linear method
    a, b = np.nonzero(~mask)
    values = projected[a, b]
    grid_x, grid_y = np.mgrid[0:374:375j, 0:874:875j]
    grid_z = interpolate.griddata((a, b), values, (grid_x, grid_y), method='linear')
    return grid_z, file_name

if __name__ =='__main__':
    signal.signal(signal.SIGTERM, term)#注册信号量，使得在终端杀死主进程时，子进程也被杀死

    parser = argparse.ArgumentParser() 
    parser.add_argument('--pnum',type=int,default=80,help='the number of sub-processes')
    parser.add_argument('--read_root',type=str,default='/usr/commondata/weather/StageIV/StageIV/')
    parser.add_argument('--save_root',type=str,default='/usr/commondata/weather/IR_data/output/')
    parser.add_argument('--long_range',type=tuple,default=(-115,-80))
    parser.add_argument('--lat_range',type=tuple,default=(30,45))
    parser.add_argument('--mode',type=str,default='StageIV',choices=['IR_2012', 'StageIV'])
    args = parser.parse_args()

    args.save_root = args.save_root + args.mode + '/'
    width = int((np.max(args.lat_range) - np.min(args.lat_range)) / 0.04)
    length = int((np.max(args.long_range) - np.min(args.long_range)) / 0.04)
    channel_dict = {3:0, 4:1, 6:2}
    channel_map = lambda x: channel_dict[x]

    run(args)