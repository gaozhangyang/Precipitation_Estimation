from ftplib import FTP  
from fnmatch import fnmatch
from multiprocessing import Pool,Manager,Process
from pathlib import Path
import os
import tqdm
import time
import multiprocessing
import argparse
import signal
import numpy as np


download_num=multiprocessing.Value("d",0.0) # d表示数值,主进程与子进程共享这个value。（主进程与子进程都是用的同一个value）
processes=[]

def download(args,infile,myfile,pid,Process_state,name): 
    # download_num.value+=1

    ftp=FTP(args.server)                        #设置变量
    ftp.login("anonymous","user@internet")      #连接的用户名，密码
    ftp.cwd('{}/{}'.format(args.cd_path,name))

    
    myfile=myfile/infile
    with open(myfile,'wb') as myfile:
        ftp.retrbinary('RETR '+infile,myfile.write,1024*1024)
        
    ftp.close()

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

def autodownload(args):
    for name in ['001','002','003','004']:
        ##########################拿取服务器文件数据###################
        ftp=FTP(args.server)                         #设置变量
        ftp.login("anonymous","user@internet")      #连接的用户名，密码
        print(ftp.getwelcome())            #打印出欢迎信息

        ftp.cwd('{}/{}'.format(args.cd_path,name))
        files_info=[]
        ftp.dir('.', files_info.append)
        
        ##############################文件过滤########################
        infiles=[]
        for i in range(len(files_info)):
            file_name = files_info[i].split()[-1]
            file_size = files_info[i].split()[4]
            if not (fnmatch(file_name, '*BAND_03.nc') or fnmatch(file_name, '*BAND_04.nc') or fnmatch(file_name, '*BAND_06.nc')):
                continue
            
            myfile=Path('./IR_2014')/file_name
            if os.path.exists(myfile):
                if os.path.getsize(myfile) == int(file_size):
                    continue
            
            infiles.append(file_name)
        
        ftp.quit()
        print(len(infiles))
        
        myfile=Path('./IR_2014')
        myfile.mkdir(parents=True,exist_ok=True)
        for infile in infiles:
            if not os.path.exists(myfile/infile):
                f=open(myfile/infile,'w')
                f.close()

        
        ###############################多线程下载########################
        Process_state=Manager().dict({str(i):True for i in range(pnum)})
        idx=0
        while idx<len(infiles):
            #查询是否有可用线程
            for pid in range(pnum):
                if Process_state[str(pid)]==True:
                    Process_state[str(pid)]=False
                    p=Process(target=download,args=(args,infiles[idx],myfile,pid,Process_state,name))
                    print(idx)
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

# name='004'
if __name__ =='__main__':
    signal.signal(signal.SIGTERM, term)#注册信号量，使得在终端杀死主进程时，子进程也被杀死

    parser = argparse.ArgumentParser() 
    parser.add_argument('--server', type=str,default='ftp.bou.class.noaa.gov')
    parser.add_argument('--cd_path', type=str,default='155954/7338375394')
    parser.add_argument('--save_file',type=str,default='./IR_2014')
    parser.add_argument('--pnum',type=int,default=30,'the number of subprocess')
    args = parser.parse_args()
    pnum=args.pnum
    # name=args.name
    autodownload(args)
    
