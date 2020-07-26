import os
from multiprocessing import Process,Manager
import numpy as np
import signal
import time


## parameter analysis for SAGloss

# cmd=[
#     'cd Identification\nCUDA_VISIBLE_DEVICES={}'+ ' python train.py --train_R 10000 --train_NR 10000 --batch_size 1024 --res_dir ./ex1/results/001',
#     'cd Identification\nCUDA_VISIBLE_DEVICES={}'+ ' python train.py --train_R 5000 --train_NR 10000 --batch_size 1024 --res_dir ./ex1/results/002',
#     'cd Identification\nCUDA_VISIBLE_DEVICES={}'+ ' python train.py --train_R 10000 --train_NR 5000 --batch_size 1024 --res_dir ./ex1/results/003',

#     'cd Identification\nCUDA_VISIBLE_DEVICES={}'+ ' python train.py --train_R 10000 --train_NR 10000 --batch_size 512 --res_dir ./ex1/results/004',
#     'cd Identification\nCUDA_VISIBLE_DEVICES={}'+ ' python train.py --train_R 5000 --train_NR 10000 --batch_size 512 --res_dir ./ex1/results/005',
#     'cd Identification\nCUDA_VISIBLE_DEVICES={}'+ ' python train.py --train_R 10000 --train_NR 5000 --batch_size 512 --res_dir ./ex1/results/006',

#     'cd Identification\nCUDA_VISIBLE_DEVICES={}'+ ' python train.py --train_R 10000 --train_NR 10000 --batch_size 256 --res_dir ./ex1/results/007',
#     'cd Identification\nCUDA_VISIBLE_DEVICES={}'+ ' python train.py --train_R 5000 --train_NR 10000 --batch_size 256 --res_dir ./ex1/results/008',
#     'cd Identification\nCUDA_VISIBLE_DEVICES={}'+ ' python train.py --train_R 10000 --train_NR 5000 --batch_size 256 --res_dir ./ex1/results/009',
#    ]



# cmd=[
#     'cd Identification\nCUDA_VISIBLE_DEVICES={}'+ ' python train.py --train_R 1000 --train_NR 1000 --batch_size 1024 --res_dir ./ex2/results/002',
#     'cd Identification\nCUDA_VISIBLE_DEVICES={}'+ ' python train.py --train_R 100000 --train_NR 100000 --batch_size 1024 --res_dir ./ex2/results/001',
#     'cd Identification\nCUDA_VISIBLE_DEVICES={}'+ ' python train.py --train_R 300000 --train_NR 300000 --batch_size 1024 --res_dir ./ex2/results/003',
#     'cd Identification\nCUDA_VISIBLE_DEVICES={}'+ ' python train.py --train_R 5000 --train_NR 5000 --batch_size 1024 --res_dir ./ex2/results/004',
#     'cd Identification\nCUDA_VISIBLE_DEVICES={}'+ ' python train.py --train_R 10000 --train_NR 10000 --batch_size 1024 --res_dir ./ex2/results/005',
#     'cd Identification\nCUDA_VISIBLE_DEVICES={}'+ ' python train.py --train_R 50000 --train_NR 50000 --batch_size 1024 --res_dir ./ex2/results/006',
#    ]


# cmd=[
#     'cd Identification\nCUDA_VISIBLE_DEVICES={}'+ ' python train.py --train_R 50000 --train_NR 50000 --batch_size 1024 --lr 0.001 --res_dir ./ex3/results/001',
#     'cd Identification\nCUDA_VISIBLE_DEVICES={}'+ ' python train.py --train_R 50000 --train_NR 50000 --batch_size 1024 --lr 0.0001 --res_dir ./ex3/results/002',
#     'cd Identification\nCUDA_VISIBLE_DEVICES={}'+ ' python train.py --train_R 50000 --train_NR 50000 --batch_size 1024 --lr 0.00001 --res_dir ./ex3/results/003',
#     'cd Identification\nCUDA_VISIBLE_DEVICES={}'+ ' python train.py --train_R 50000 --train_NR 50000 --batch_size 1024 --lr 0.000001 --res_dir ./ex3/results/004',
#    ]


def run(command,gpuid,gpustate):
    os.system(command.format(gpuid))
    gpustate[str(gpuid)]=True

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

if __name__ =='__main__':
    signal.signal(signal.SIGTERM, term)#注册信号量，使得在终端杀死主进程时，子进程也被杀死

    gpustate=Manager().dict({str(i):True for i in range(0,4)})
    processes=[]
    idx=0
    while idx<len(cmd):
        #查询是否有可用gpu
        for gpuid in range(0,4):
            if gpustate[str(gpuid)]==True:
                print(idx)
                gpustate[str(gpuid)]=False
                p=Process(target=run,args=(cmd[idx],gpuid,gpustate),name=str(gpuid))
                p.start()
                print(gpustate)
                processes.append(p)
                idx+=1
                break

    for p in processes:
        p.join()
    
    while(1):
        pass
