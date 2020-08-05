import numpy as np
import os
from multiprocessing import Process, Manager
import signal
import time
import pynvml
pynvml.nvmlInit()


cmd=[
    'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py     --gpus 1 \
                                                                        --task identification \
                                                                        --train_R 50000   \
                                                                        --train_NR 200000   \
                                                                        --batch_size 1024   \
                                                                        --res_dir /usr/commondata/weather/code/Precipitation_Estimation/results \
                                                                        --ex_name 001',

    'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py     --gpus 1 \
                                                                        --task identification \
                                                                        --train_R 100000   \
                                                                        --train_NR 200000   \
                                                                        --batch_size 1024   \
                                                                        --res_dir /usr/commondata/weather/code/Precipitation_Estimation/results \
                                                                        --ex_name 002',

    'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py     --gpus 1 \
                                                                        --task identification \
                                                                        --train_R 340000   \
                                                                        --train_NR 340000   \
                                                                        --batch_size 1024   \
                                                                        --res_dir /usr/commondata/weather/code/Precipitation_Estimation/results \
                                                                        --ex_name 003',

    'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py     --gpus 1 \
                                                                        --task identification \
                                                                        --train_R 200000   \
                                                                        --train_NR 400000   \
                                                                        --batch_size 1024   \
                                                                        --res_dir /usr/commondata/weather/code/Precipitation_Estimation/results \
                                                                        --ex_name 004',

    'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py     --gpus 1 \
                                                                        --task identification \
                                                                        --train_R 50000   \
                                                                        --train_NR 250000   \
                                                                        --batch_size 1024   \
                                                                        --res_dir /usr/commondata/weather/code/Precipitation_Estimation/results \
                                                                        --ex_name 005',

    'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py     --gpus 1 \
                                                                        --task identification \
                                                                        --train_R 50000   \
                                                                        --train_NR 300000   \
                                                                        --batch_size 1024   \
                                                                        --res_dir /usr/commondata/weather/code/Precipitation_Estimation/results \
                                                                        --ex_name 006',

    'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py     --gpus 1 \
                                                                        --task identification \
                                                                        --train_R 100000   \
                                                                        --train_NR 500000   \
                                                                        --batch_size 1024   \
                                                                        --res_dir /usr/commondata/weather/code/Precipitation_Estimation/results \
                                                                        --ex_name 007',
]



cmd = [
    'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py     --gpus 1 \
                                                                        --task estimation  \
                                                                        --w_kl_C 0,0,100 \
                                                                        --w_ed_C 1,1,200 \
                                                                        --train_R 100000   \
                                                                        --batch_size 1024   \
                                                                        --res_dir /usr/commondata/weather/code/Precipitation_Estimation/results \
                                                                        --ex_name changew3/001 \
                                                                        --epoch_s 1 \
                                                                        --epoch_e 100 \
                                                                        --patience 100 \
                                                                        --hdelta 2.5',

    'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py     --gpus 1 \
                                                                        --task estimation  \
                                                                        --w_kl_C 0,0,100 \
                                                                        --w_ed_C 1,1,200 \
                                                                        --train_R 100000   \
                                                                        --batch_size 1024   \
                                                                        --res_dir /usr/commondata/weather/code/Precipitation_Estimation/results \
                                                                        --ex_name changew3/002 \
                                                                        --epoch_s 1 \
                                                                        --epoch_e 100 \
                                                                        --patience 100 \
                                                                        --hdelta 5',

    'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py     --gpus 1 \
                                                                        --task estimation  \
                                                                        --w_kl_C 0,0,100 \
                                                                        --w_ed_C 1,1,200 \
                                                                        --train_R 100000   \
                                                                        --batch_size 1024   \
                                                                        --res_dir /usr/commondata/weather/code/Precipitation_Estimation/results \
                                                                        --ex_name changew3/003 \
                                                                        --epoch_s 1 \
                                                                        --epoch_e 100 \
                                                                        --patience 100 \
                                                                        --hdelta 7.5',

    'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py     --gpus 1 \
                                                                        --task estimation  \
                                                                        --w_kl_C 0,0,100 \
                                                                        --w_ed_C 1,1,200 \
                                                                        --train_R 100000   \
                                                                        --batch_size 1024   \
                                                                        --res_dir /usr/commondata/weather/code/Precipitation_Estimation/results \
                                                                        --ex_name changew3/004 \
                                                                        --epoch_s 1 \
                                                                        --epoch_e 100 \
                                                                        --patience 100 \
                                                                        --hdelta 10',


    'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py     --gpus 1 \
                                                                        --task estimation  \
                                                                        --w_kl_C 0,0,100 \
                                                                        --w_ed_C 1,1,200 \
                                                                        --train_R 100000   \
                                                                        --batch_size 1024   \
                                                                        --res_dir /usr/commondata/weather/code/Precipitation_Estimation/results \
                                                                        --ex_name changew3/005 \
                                                                        --epoch_s 1 \
                                                                        --epoch_e 100 \
                                                                        --patience 100 \
                                                                        --hdelta 12.5\
                                                                        ',

    'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py     --gpus 1 \
                                                                        --task estimation  \
                                                                        --w_kl_C 0,0,100 \
                                                                        --w_ed_C 1,1,200 \
                                                                        --train_R 100000   \
                                                                        --batch_size 1024   \
                                                                        --res_dir /usr/commondata/weather/code/Precipitation_Estimation/results \
                                                                        --ex_name changew3/006 \
                                                                        --epoch_s 1 \
                                                                        --epoch_e 100 \
                                                                        --patience 100 \
                                                                        --hdelta 15\
                                                                        ',

    'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py     --gpus 1 \
                                                                        --task estimation  \
                                                                        --w_kl_C 0,0,100 \
                                                                        --w_ed_C 1,1,200 \
                                                                        --train_R 100000   \
                                                                        --batch_size 1024   \
                                                                        --res_dir /usr/commondata/weather/code/Precipitation_Estimation/results \
                                                                        --ex_name changew3/007 \
                                                                        --epoch_s 1 \
                                                                        --epoch_e 100 \
                                                                        --patience 100 \
                                                                        --hdelta 20\
                                                                        ',

    'cd Precipitation\nCUDA_VISIBLE_DEVICES={}' + ' python train2.py     --gpus 1 \
                                                                        --task estimation  \
                                                                        --w_kl_C 0,0,100 \
                                                                        --w_ed_C 1,1,200 \
                                                                        --train_R 100000   \
                                                                        --batch_size 1024   \
                                                                        --res_dir /usr/commondata/weather/code/Precipitation_Estimation/results \
                                                                        --ex_name changew3/008 \
                                                                        --epoch_s 1 \
                                                                        --epoch_e 100 \
                                                                        --patience 100 \
                                                                        --hdelta 25\
                                                                        ',

    'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py     --gpus 1 \
                                                                        --task estimation  \
                                                                        --w_kl_C 0,0,300 0,10,300 \
                                                                        --w_ed_C 1,1,200 \
                                                                        --train_R 100000   \
                                                                        --batch_size 1024   \
                                                                        --res_dir /usr/commondata/weather/code/Precipitation_Estimation/results \
                                                                        --ex_name changew3/009 \
                                                                        --epoch_s 1 \
                                                                        --epoch_e 100 \
                                                                        --patience 100 \
                                                                        --hdelta 12.5',


    'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py     --gpus 1 \
                                                                        --task estimation  \
                                                                        --w_kl_C 0,0,300 0,10,300 \
                                                                        --w_ed_C 1,1,200 \
                                                                        --train_R 100000   \
                                                                        --batch_size 1024   \
                                                                        --res_dir /usr/commondata/weather/code/Precipitation_Estimation/results \
                                                                        --ex_name changew3/010 \
                                                                        --epoch_s 1 \
                                                                        --epoch_e 100 \
                                                                        --patience 100 \
                                                                        --hdelta 20\
                                                                        ',

    'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py     --gpus 1 \
                                                                        --task estimation  \
                                                                        --w_kl_C 0,0,300 0,10,300 \
                                                                        --w_ed_C 1,1,200 \
                                                                        --train_R 100000   \
                                                                        --batch_size 1024   \
                                                                        --res_dir /usr/commondata/weather/code/Precipitation_Estimation/results \
                                                                        --ex_name changew3/011 \
                                                                        --epoch_s 1 \
                                                                        --epoch_e 100 \
                                                                        --patience 100 \
                                                                        --hdelta 25',

    'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py     --gpus 1 \
                                                                        --task estimation  \
                                                                        --w_kl_C 0,0,300 0,10,300 \
                                                                        --w_ed_C 1,1,200 \
                                                                        --train_R 100000   \
                                                                        --batch_size 1024   \
                                                                        --res_dir /usr/commondata/weather/code/Precipitation_Estimation/results \
                                                                        --ex_name changew3/012 \
                                                                        --epoch_s 1 \
                                                                        --epoch_e 100 \
                                                                        --patience 100 \
                                                                        --hdelta 2.5',
    
    'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py     --gpus 1 \
                                                                        --task estimation  \
                                                                        --w_kl_C 0,0,300 0,10,300 \
                                                                        --w_ed_C 1,1,200 \
                                                                        --train_R 100000   \
                                                                        --batch_size 1024   \
                                                                        --res_dir /usr/commondata/weather/code/Precipitation_Estimation/results \
                                                                        --ex_name changew3/013 \
                                                                        --epoch_s 1 \
                                                                        --epoch_e 100 \
                                                                        --patience 100 \
                                                                        --hdelta 5',
    
    'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py     --gpus 1 \
                                                                        --task estimation  \
                                                                        --w_kl_C 0,0,300 0,10,300 \
                                                                        --w_ed_C 1,1,200 \
                                                                        --train_R 100000   \
                                                                        --batch_size 1024   \
                                                                        --res_dir /usr/commondata/weather/code/Precipitation_Estimation/results \
                                                                        --ex_name changew3/014 \
                                                                        --epoch_s 1 \
                                                                        --epoch_e 100 \
                                                                        --patience 100 \
                                                                        --hdelta 7.5',
    
    'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py     --gpus 1 \
                                                                        --task estimation  \
                                                                        --w_kl_C 0,0,300 0,10,300 \
                                                                        --w_ed_C 1,1,200 \
                                                                        --train_R 100000   \
                                                                        --batch_size 1024   \
                                                                        --res_dir /usr/commondata/weather/code/Precipitation_Estimation/results \
                                                                        --ex_name changew3/015 \
                                                                        --epoch_s 1 \
                                                                        --epoch_e 100 \
                                                                        --patience 100 \
                                                                        --hdelta 10',

]

min_gpu_memory = 12


def run(command, gpuid, gpustate):
    os.system(command.format(gpuid))
    gpustate[gpuid] += min_gpu_memory


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


def get_memory(gpuid):
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpuid)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    memory_GB = meminfo.free/1024**3
    return memory_GB


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, term)  # 注册信号量，使得在终端杀死主进程时，子进程也被杀死

    gpus = [0,1,2,3,4,5,6,7]
    gpustate = Manager().dict({i: get_memory(i) for i in gpus})
    processes = []
    idx = 0
    while idx < len(cmd):
        # 查询是否有可用gpu
        for gpuid in gpus:
            if gpustate[gpuid] > min_gpu_memory:
                p = Process(target=run, args=(
                    cmd[idx], gpuid, gpustate), name=str(gpuid))
                p.start()
                print('run {} with gpu {}'.format(cmd[idx], gpuid))
                processes.append(p)
                idx += 1
                gpustate[gpuid] -= min_gpu_memory

                if idx == len(cmd):
                    break

        time.sleep(600)

    for p in processes:
        p.join()

    while(1):
        pass
