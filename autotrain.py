import numpy as np
import os
from multiprocessing import Process,Manager
import signal
import time
import pynvml
pynvml.nvmlInit()

## parameter analysis for SAGloss

# cmd=[
    # 'cd Estimation\nCUDA_VISIBLE_DEVICES={}'+ ' python train.py --task identification --train_R 10000   --train_NR 10000    --batch_size 1024   --res_dir ./ex1 --ex_name 001',
    # 'cd Estimation\nCUDA_VISIBLE_DEVICES={}'+ ' python train.py --task identification --train_R 5000    --train_NR 10000    --batch_size 1024   --res_dir ./ex1 --ex_name 002',
    # 'cd Estimation\nCUDA_VISIBLE_DEVICES={}'+ ' python train.py --task identification --train_R 10000   --train_NR 5000     --batch_size 1024   --res_dir ./ex1 --ex_name 003',
    # 'cd Estimation\nCUDA_VISIBLE_DEVICES={}'+ ' python train.py --task identification --train_R 10000   --train_NR 10000    --batch_size 512    --res_dir ./ex1 --ex_name 004',
    # 'cd Estimation\nCUDA_VISIBLE_DEVICES={}'+ ' python train.py --task identification --train_R 5000    --train_NR 10000    --batch_size 512    --res_dir ./ex1 --ex_name 005',
    # 'cd Estimation\nCUDA_VISIBLE_DEVICES={}'+ ' python train.py --task identification --train_R 10000   --train_NR 5000     --batch_size 512    --res_dir ./ex1 --ex_name 006',
    # 'cd Estimation\nCUDA_VISIBLE_DEVICES={}'+ ' python train.py --task identification --train_R 10000   --train_NR 10000    --batch_size 256    --res_dir ./ex1 --ex_name 007',
    # 'cd Estimation\nCUDA_VISIBLE_DEVICES={}'+ ' python train.py --task identification --train_R 5000    --train_NR 10000    --batch_size 256    --res_dir ./ex1 --ex_name 008',
    # 'cd Estimation\nCUDA_VISIBLE_DEVICES={}'+ ' python train.py --task identification --train_R 10000   --train_NR 5000     --batch_size 256    --res_dir ./ex1 --ex_name 009',

    # 'cd Estimation\nCUDA_VISIBLE_DEVICES={}'+ ' python train.py --task identification --train_R 1000    --train_NR 1000     --batch_size 1024   --res_dir ./ex2 --ex_name 001',
    # 'cd Estimation\nCUDA_VISIBLE_DEVICES={}'+ ' python train.py --task identification --train_R 100000  --train_NR 100000   --batch_size 1024   --res_dir ./ex2 --ex_name 002',
    # 'cd Estimation\nCUDA_VISIBLE_DEVICES={}'+ ' python train.py --task identification --train_R 300000  --train_NR 300000   --batch_size 1024   --res_dir ./ex2 --ex_name 003',
    # 'cd Estimation\nCUDA_VISIBLE_DEVICES={}'+ ' python train.py --task identification --train_R 5000    --train_NR 5000     --batch_size 1024   --res_dir ./ex2 --ex_name 004',
    # 'cd Estimation\nCUDA_VISIBLE_DEVICES={}'+ ' python train.py --task identification --train_R 10000   --train_NR 10000    --batch_size 1024   --res_dir ./ex2 --ex_name 005',
    # 'cd Estimation\nCUDA_VISIBLE_DEVICES={}'+ ' python train.py --task identification --train_R 50000   --train_NR 50000    --batch_size 1024   --res_dir ./ex2 --ex_name 006',
    # 'cd Estimation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py --gpus 1 --task identification --train_R 340000   --train_NR 340000    --batch_size 1024   --lr 0.001      --res_dir ./ex2 --ex_name 007',
    

    # 'cd Estimation\nCUDA_VISIBLE_DEVICES={}'+ ' python train.py --task identification --train_R 50000   --train_NR 50000    --batch_size 1024   --lr 0.001      --res_dir ./ex3 --ex_name 001',
    # 'cd Estimation\nCUDA_VISIBLE_DEVICES={}'+ ' python train.py --task identification --train_R 50000   --train_NR 50000    --batch_size 1024   --lr 0.0001     --res_dir ./ex3 --ex_name 002',
    # 'cd Estimation\nCUDA_VISIBLE_DEVICES={}'+ ' python train.py --task identification --train_R 50000   --train_NR 50000    --batch_size 1024   --lr 0.00001    --res_dir ./ex3 --ex_name 003',
    # 'cd Estimation\nCUDA_VISIBLE_DEVICES={}'+ ' python train.py --task identification --train_R 50000   --train_NR 50000    --batch_size 1024   --lr 0.000001   --res_dir ./ex3 --ex_name 004',

    # 'cd Estimation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py  --gpus 1 --task identification --train_R 50000   --train_NR 50000    --batch_size 1024   --lr 0.001   --res_dir ./ex4 --ex_name 001',
#    ]



# cmd=[
    # 'cd Estimation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py --gpus 1 --task estimation     --train_R 470000   --train_NR 470000    --batch_size 1024   --w 1000     --res_dir ./ex4 --ex_name 001',
    # 'cd Estimation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py --gpus 1 --task estimation     --train_R 470000   --train_NR 470000    --batch_size 1024   --w 100      --res_dir ./ex4 --ex_name 002',
    # 'cd Estimation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py --gpus 1 --task estimation     --train_R 470000   --train_NR 470000    --batch_size 1024   --w 10       --res_dir ./ex4 --ex_name 003',
    # 'cd Estimation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py --gpus 1 --task estimation     --train_R 470000   --train_NR 470000    --batch_size 1024   --w 1        --res_dir ./ex4 --ex_name 004',
    # 'cd Estimation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py --gpus 1 --task estimation     --train_R 470000   --train_NR 470000    --batch_size 1024   --w 100  --lr 0.0001     --res_dir ./ex4 --ex_name 005',
    # 'cd Estimation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py --gpus 1 --task estimation     --train_R 470000   --train_NR 470000    --batch_size 1024   --w 1000 --lr 0.0001      --res_dir ./ex4 --ex_name 006',
    # 'cd Estimation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py --gpus 1 --task estimation     --train_R 470000   --train_NR 470000    --batch_size 1024   --w 100  --lr 0.00001       --res_dir ./ex4 --ex_name 007',
    # 'cd Estimation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py --gpus 1 --task estimation     --train_R 470000   --train_NR 470000    --batch_size 1024   --w 1000 --lr 0.00001        --res_dir ./ex4 --ex_name 008',


    # 'cd Estimation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py --gpus 1 --task estimation     --train_R 100000   --train_NR 100000    --batch_size 1024   --w 1000      --res_dir ./ex5 --ex_name 001',
    # 'cd Estimation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py --gpus 1 --task estimation     --train_R 100000   --train_NR 100000    --batch_size 1024   --w 100       --res_dir ./ex5 --ex_name 002',
    # 'cd Estimation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py --gpus 1 --task estimation     --train_R 470000   --train_NR 470000    --batch_size 1024   --w 1000      --res_dir ./ex5 --ex_name 003',
    # 'cd Estimation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py --gpus 1 --task estimation     --train_R 470000   --train_NR 470000    --batch_size 1024   --w 100       --res_dir ./ex5 --ex_name 004',

    # 'cd Estimation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py --gpus 1 --task estimation     --train_R 100000   --train_NR 100000    --batch_size 1024   --w 1000    --sigma 2  --res_dir ./ex6 --ex_name 001',
    # 'cd Estimation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py --gpus 1 --task estimation     --train_R 100000   --train_NR 100000    --batch_size 1024   --w 1000    --sigma 5  --res_dir ./ex6 --ex_name 002',
    # 'cd Estimation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py --gpus 1 --task estimation     --train_R 100000   --train_NR 100000    --batch_size 1024   --w 1000    --sigma 10  --res_dir ./ex6 --ex_name 003',
    # 'cd Estimation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py --gpus 1 --task estimation     --train_R 100000   --train_NR 100000    --batch_size 1024   --w 1000    --sigma 15  --res_dir ./ex6 --ex_name 004',
#    ]



# cmd=[
    # 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py  --gpus 1 --task identification --train_R 50000   --train_NR 200000   --batch_size 1024   --res_dir ./results --ex_name 001',
    # 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py  --gpus 1 --task identification --train_R 100000   --train_NR 200000   --batch_size 1024   --res_dir ./results --ex_name 002',
    # 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py  --gpus 1 --task identification --train_R 340000   --train_NR 340000   --batch_size 1024   --res_dir ./results --ex_name 003',
    # 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py  --gpus 1 --task identification --train_R 200000   --train_NR 400000   --batch_size 1024   --res_dir ./results --ex_name 004',

    # 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py  --gpus 1 --task identification --train_R 50000   --train_NR 250000   --batch_size 1024   --res_dir ./results --ex_name 005',
    # 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py  --gpus 1 --task identification --train_R 50000   --train_NR 300000   --batch_size 1024   --res_dir ./results --ex_name 006',
    # 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py  --gpus 1 --task identification --train_R 100000   --train_NR 500000   --batch_size 1024   --res_dir ./results --ex_name 007',
    # 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py  --gpus 1 --task identification --train_R 100000   --train_NR 700000   --batch_size 1024   --res_dir ./results --ex_name 008',
    # ]


cmd=[
    'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py  --gpus 1 --task estimation  --w 1000 --train_R 100000   --batch_size 1024   --res_dir ./results --ex_name 001',
    'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py  --gpus 1 --task estimation  --w 100  --train_R 100000   --batch_size 1024   --res_dir ./results --ex_name 002',
    'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py  --gpus 1 --task estimation  --w 10   --train_R 100000   --batch_size 1024   --res_dir ./results --ex_name 003',
    'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py  --gpus 1 --task estimation  --w 1    --train_R 100000   --batch_size 1024   --res_dir ./results --ex_name 004',
    'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py  --gpus 1 --task estimation  --w 0.1  --train_R 100000   --batch_size 1024   --res_dir ./results --ex_name 005',

    # 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py  --gpus 1 --task estimation  --w 1000 --train_R 100000   --batch_size 1024   --res_dir ./results --ex_name 001',
    # 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py  --gpus 1 --task estimation  --w 1000 --train_R 200000   --batch_size 1024   --res_dir ./results --ex_name 002',
    # 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py  --gpus 1 --task estimation  --w 1000 --train_R 470000   --batch_size 1024   --res_dir ./results --ex_name 003',
    # 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py  --gpus 1 --task estimation  --w 100  --train_R 100000   --batch_size 1024   --res_dir ./results --ex_name 004',
    # 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py  --gpus 1 --task estimation  --w 100  --train_R 200000   --batch_size 1024   --res_dir ./results --ex_name 005',
    # 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py  --gpus 1 --task estimation  --w 100  --train_R 470000   --batch_size 1024   --res_dir ./results --ex_name 006',
    # 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py  --gpus 1 --task estimation  --w 10   --train_R 100000   --batch_size 1024   --res_dir ./results --ex_name 007',
    # 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py  --gpus 1 --task estimation  --w 10   --train_R 200000   --batch_size 1024   --res_dir ./results --ex_name 008',
    # 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py  --gpus 1 --task estimation  --w 10   --train_R 470000   --batch_size 1024   --res_dir ./results --ex_name 009',
    ]


min_gpu_memory=12

def run(command,gpuid,gpustate):
    os.system(command.format(gpuid))
    gpustate[gpuid]+=min_gpu_memory

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

def get_memory(gpuid):
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpuid)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    memory_GB=meminfo.free/1024**3
    return memory_GB

if __name__ =='__main__':
    signal.signal(signal.SIGTERM, term)#注册信号量，使得在终端杀死主进程时，子进程也被杀死
    
    gpus=[0,1,2,3,4,5,6,7]
    gpustate=Manager().dict({i:get_memory(i) for i in gpus})
    processes=[]
    idx=0
    while idx<len(cmd):
        #查询是否有可用gpu
        for gpuid in gpus:
            if  gpustate[gpuid]> min_gpu_memory:
                p=Process(target=run,args=(cmd[idx],gpuid,gpustate),name=str(gpuid))
                p.start()
                print('run {} with gpu {}'.format(cmd[idx],gpuid))
                processes.append(p)
                idx+=1       
                gpustate[gpuid]-=min_gpu_memory

                if idx==len(cmd):
                    break

        time.sleep(600)

    # for p in processes:
    #     p.join()
    
    while(1):
        pass
