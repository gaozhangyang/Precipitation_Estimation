import os
from multiprocessing import Process,Manager
import numpy as np
import signal
import time


## parameter analysis for SAGloss

cmd=[
    'cd Identification\nCUDA_VISIBLE_DEVICES={}'+ ' python train.py -train_R 10000 -train_NR 10000 ',
   ]



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

    gpustate=Manager().dict({str(i):True for i in range(2,8)})
    processes=[]
    idx=0
    while idx<len(cmd):
        #查询是否有可用gpu
        for gpuid in range(2,8):
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
