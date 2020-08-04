import numpy as np
import os
from multiprocessing import Process, Manager
import signal
import time
import pynvml
pynvml.nvmlInit()

# parameter analysis for SAGloss

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


# cmd=[
# 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py  --gpus 1 --task estimation  --w_kl 0     --w_ed 1 --train_R 100000   --batch_size 1024   --res_dir ./results --ex_name 001  --patience 8',
# 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py  --gpus 1 --task estimation  --w_kl 0.01  --w_ed 1 --train_R 100000   --batch_size 1024   --res_dir ./results --ex_name 002  --patience 8',
# 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py  --gpus 1 --task estimation  --w_kl 0.001 --w_ed 1 --train_R 100000   --batch_size 1024   --res_dir ./results --ex_name 003  --patience 8',
# 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py  --gpus 1 --task estimation  --w_kl 0.01 --w_ed 0.001 --train_R 100000   --batch_size 1024   --res_dir ./results --ex_name 005  --patience 8',
# 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py  --gpus 1 --task estimation  --w_kl 0.01 --w_ed 0.0001 --train_R 100000   --batch_size 1024   --res_dir ./results --ex_name 006  --patience 8',
# 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py  --gpus 1 --task estimation  --w_kl 0.01 --w_ed 0.00001 --train_R 100000   --batch_size 1024   --res_dir ./results --ex_name 007  --patience 8',

# 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py  --gpus 1 --task estimation  --w_kl 0.1  --w_ed 1 --train_R 100000   --batch_size 1024   --res_dir ./results --ex_name onlyKL0.1  --patience 8',
# 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py  --gpus 1 --task estimation  --w_kl 0.01   --w_ed 0    --train_R 100000   --batch_size 1024   --res_dir ./results --ex_name onlyKL0.01   --patience 10',
# 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py  --gpus 1 --task estimation  --w_kl 0.001  --w_ed 0    --train_R 100000   --batch_size 1024   --res_dir ./results --ex_name onlyKL0.001  --patience 10',
# 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py  --gpus 1 --task estimation  --w_kl 0.0001  --w_ed 0    --train_R 100000   --batch_size 1024   --res_dir ./results --ex_name onlyKL0.0001  --patience 10',
# 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py  --gpus 1 --task estimation  --w_kl 0.00001  --w_ed 0    --train_R 100000   --batch_size 1024   --res_dir ./results --ex_name onlyKL0.00001  --patience 10',

# 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py  --gpus 1 --task estimation  --w 1000 --train_R 100000   --batch_size 1024   --res_dir ./results --ex_name 001',
# 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py  --gpus 1 --task estimation  --w 1000 --train_R 200000   --batch_size 1024   --res_dir ./results --ex_name 002',
# 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py  --gpus 1 --task estimation  --w 1000 --train_R 470000   --batch_size 1024   --res_dir ./results --ex_name 003',
# 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py  --gpus 1 --task estimation  --w 100  --train_R 100000   --batch_size 1024   --res_dir ./results --ex_name 004',
# 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py  --gpus 1 --task estimation  --w 100  --train_R 200000   --batch_size 1024   --res_dir ./results --ex_name 005',
# 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py  --gpus 1 --task estimation  --w 100  --train_R 470000   --batch_size 1024   --res_dir ./results --ex_name 006',
# 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py  --gpus 1 --task estimation  --w 10   --train_R 100000   --batch_size 1024   --res_dir ./results --ex_name 007',
# 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py  --gpus 1 --task estimation  --w 10   --train_R 200000   --batch_size 1024   --res_dir ./results --ex_name 008',
# 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py  --gpus 1 --task estimation  --w 10   --train_R 470000   --batch_size 1024   --res_dir ./results --ex_name 009',

# 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train.py  --gpus 1 \
#                                                                 --task estimation  \
#                                                                 --hdelta 0.01 \
#                                                                 --train_R 100000   \
#                                                                 --batch_size 1024   \
#                                                                 --res_dir ./results \
#                                                                 --ex_name huber0.01  \
#                                                                 --patience 8 \
#                                                                 --epoch_s 1 \
#                                                                 --epoch_e 100',

# 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train.py  --gpus 1 \
#                                                                 --task estimation  \
#                                                                 --hdelta 0.1 \
#                                                                 --train_R 100000   \
#                                                                 --batch_size 1024   \
#                                                                 --res_dir ./results \
#                                                                 --ex_name huber0.1  \
#                                                                 --patience 8 \
#                                                                 --epoch_s 1 \
#                                                                 --epoch_e 100',

# 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train.py  --gpus 1 \
#                                                                 --task estimation  \
#                                                                 --hdelta 0.5 \
#                                                                 --train_R 100000   \
#                                                                 --batch_size 1024   \
#                                                                 --res_dir ./results \
#                                                                 --ex_name huber0.5  \
#                                                                 --patience 8 \
#                                                                 --epoch_s 1 \
#                                                                 --epoch_e 100',

# 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train.py  --gpus 1 \
#                                                                 --task estimation  \
#                                                                 --hdelta 1.0 \
#                                                                 --train_R 100000   \
#                                                                 --batch_size 1024   \
#                                                                 --res_dir ./results \
#                                                                 --ex_name huber1.0  \
#                                                                 --patience 8 \
#                                                                 --epoch_s 1 \
#                                                                 --epoch_e 100',

# 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py     --gpus 1 \
#                                                                     --task estimation  \
#                                                                     --w_kl_C 0,0,1000 0,100,1000 \
#                                                                     --w_ed_C 1,1,200 \
#                                                                     --train_R 100000   \
#                                                                     --batch_size 1024   \
#                                                                     --res_dir ./results \
#                                                                     --ex_name changew001 \
#                                                                     --epoch_s 1 \
#                                                                     --epoch_e 100',

# 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py     --gpus 1 \
#                                                                     --task estimation  \
#                                                                     --w_kl_C 0,0,100 0,10,100 \
#                                                                     --w_ed_C 1,1,200 \
#                                                                     --train_R 100000   \
#                                                                     --batch_size 1024   \
#                                                                     --res_dir ./results \
#                                                                     --ex_name changew002 \
#                                                                     --epoch_s 1 \
#                                                                     --epoch_e 100',

# 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py     --gpus 1 \
#                                                                     --task estimation  \
#                                                                     --w_kl_C 0,0,100 0,10,100 \
#                                                                     --w_ed_C 1,1,200 \
#                                                                     --train_R 100000   \
#                                                                     --batch_size 1024   \
#                                                                     --res_dir ./results \
#                                                                     --ex_name changew/001 \
#                                                                     --epoch_s 1 \
#                                                                     --epoch_e 100 \
#                                                                     --patience 100',

# 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py     --gpus 1 \
#                                                                     --task estimation  \
#                                                                     --w_kl_C 0,0,100 0,10,100 10,100,1200 \
#                                                                     --w_ed_C 1,1,200 \
#                                                                     --train_R 100000   \
#                                                                     --batch_size 1024   \
#                                                                     --res_dir ./results \
#                                                                     --ex_name changew/002 \
#                                                                     --epoch_s 1 \
#                                                                     --epoch_e 100 \
#                                                                     --patience 100',

# 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py     --gpus 1 \
#                                                                     --task estimation  \
#                                                                     --w_kl_C 0,0,100 0,10,100 \
#                                                                     --w_ed_C 1,1,200 \
#                                                                     --train_R 470000   \
#                                                                     --batch_size 1024   \
#                                                                     --res_dir ./results \
#                                                                     --ex_name changew/003 \
#                                                                     --epoch_s 1 \
#                                                                     --epoch_e 100 \
#                                                                     --patience 100 \
#                                                                     --sampling_step 4',

# 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py     --gpus 1 \
#                                                                     --task estimation  \
#                                                                     --w_kl_C 0,0,100 0,10,100 10,100,1200 \
#                                                                     --w_ed_C 1,1,200 \
#                                                                     --train_R 470000   \
#                                                                     --batch_size 1024   \
#                                                                     --res_dir ./results \
#                                                                     --ex_name changew/004 \
#                                                                     --epoch_s 1 \
#                                                                     --epoch_e 100 \
#                                                                     --patience 100\
#                                                                     --sampling_step 4',
# ]


# cmd=[
# 'cd B_Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py --gpus 1 \
#                                                                 --task identification \
#                                                                 --R_w 2600000   \
#                                                                 --NR_w 2600000   \
#                                                                 --batch_size 1024   \
#                                                                 --res_dir /usr/commondata/weather/code/Precipitation_Estimation/B_Precipitation \
#                                                                 --ex_name 001 \
#                                                                 --sampling_step 14',

# 'cd B_Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py --gpus 1 \
#                                                                 --task identification \
#                                                                 --R_w 10    \
#                                                                 --NR_w 10    \
#                                                                 --batch_size 1024   \
#                                                                 --res_dir /usr/commondata/weather/code/Precipitation_Estimation/B_Precipitation/results \
#                                                                 --ex_name 001 \
#                                                                 --sampling_step 5\
#                                                                 --epoch_s 1\
#                                                                 --epoch_e 100',

# 'cd B_Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py --gpus 1 \
#                                                                 --task identification \
#                                                                 --R_w 10    \
#                                                                 --NR_w 30    \
#                                                                 --batch_size 1024   \
#                                                                 --res_dir /usr/commondata/weather/code/Precipitation_Estimation/B_Precipitation/results \
#                                                                 --ex_name 002 \
#                                                                 --sampling_step 5\
#                                                                 --epoch_s 1\
#                                                                 --epoch_e 100',

# 'cd B_Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py --gpus 1 \
#                                                                 --task identification \
#                                                                 --R_w 10    \
#                                                                 --NR_w 50    \
#                                                                 --batch_size 1024   \
#                                                                 --res_dir /usr/commondata/weather/code/Precipitation_Estimation/B_Precipitation/results \
#                                                                 --ex_name 003 \
#                                                                 --sampling_step 5\
#                                                                 --epoch_s 1\
#                                                                 --epoch_e 100',

# 'cd B_Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py --gpus 1 \
#                                                                 --task identification \
#                                                                 --R_w 30    \
#                                                                 --NR_w 10    \
#                                                                 --batch_size 1024   \
#                                                                 --res_dir /usr/commondata/weather/code/Precipitation_Estimation/B_Precipitation/results \
#                                                                 --ex_name 004 \
#                                                                 --sampling_step 5\
#                                                                 --epoch_s 1\
#                                                                 --epoch_e 100',

# 'cd B_Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py --gpus 1 \
#                                                                 --task identification \
#                                                                 --R_w 50    \
#                                                                 --NR_w 10    \
#                                                                 --batch_size 1024   \
#                                                                 --res_dir /usr/commondata/weather/code/Precipitation_Estimation/B_Precipitation/results \
#                                                                 --ex_name 005 \
#                                                                 --sampling_step 5\
#                                                                 --epoch_s 1\
#                                                                 --epoch_e 100',
# ]


cmd = [
    # 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py     --gpus 1 \
    #                                                                     --task estimation  \
    #                                                                     --w_kl_C 0,0,100 0,10,100 \
    #                                                                     --w_ed_C 1,1,200 \
    #                                                                     --train_R 100000   \
    #                                                                     --batch_size 1024   \
    #                                                                     --res_dir ./results \
    #                                                                     --ex_name changew2/001 \
    #                                                                     --epoch_s 1 \
    #                                                                     --epoch_e 100 \
    #                                                                     --patience 100 \
    #                                                                     --hdelta 2.5',

    # 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py     --gpus 1 \
    #                                                                     --task estimation  \
    #                                                                     --w_kl_C 0,0,100 0,10,100 \
    #                                                                     --w_ed_C 1,1,200 \
    #                                                                     --train_R 100000   \
    #                                                                     --batch_size 1024   \
    #                                                                     --res_dir ./results \
    #                                                                     --ex_name changew2/002 \
    #                                                                     --epoch_s 1 \
    #                                                                     --epoch_e 100 \
    #                                                                     --patience 100 \
    #                                                                     --hdelta 5',

    # 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py     --gpus 1 \
    #                                                                     --task estimation  \
    #                                                                     --w_kl_C 0,0,100 0,10,100 \
    #                                                                     --w_ed_C 1,1,200 \
    #                                                                     --train_R 100000   \
    #                                                                     --batch_size 1024   \
    #                                                                     --res_dir ./results \
    #                                                                     --ex_name changew2/003 \
    #                                                                     --epoch_s 1 \
    #                                                                     --epoch_e 100 \
    #                                                                     --patience 100 \
    #                                                                     --hdelta 7.5',

    # 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py     --gpus 1 \
    #                                                                     --task estimation  \
    #                                                                     --w_kl_C 0,0,100 0,10,100 \
    #                                                                     --w_ed_C 1,1,200 \
    #                                                                     --train_R 100000   \
    #                                                                     --batch_size 1024   \
    #                                                                     --res_dir ./results \
    #                                                                     --ex_name changew2/004 \
    #                                                                     --epoch_s 1 \
    #                                                                     --epoch_e 100 \
    #                                                                     --patience 100 \
    #                                                                     --hdelta 10',


    # 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py     --gpus 1 \
    #                                                                     --task estimation  \
    #                                                                     --w_kl_C 0,0,100 \
    #                                                                     --w_ed_C 1,1,200 \
    #                                                                     --train_R 100000   \
    #                                                                     --batch_size 1024   \
    #                                                                     --res_dir ./results \
    #                                                                     --ex_name changew3/001 \
    #                                                                     --epoch_s 1 \
    #                                                                     --epoch_e 100 \
    #                                                                     --patience 100 \
    #                                                                     --hdelta 2.5',

    # 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py     --gpus 1 \
    #                                                                     --task estimation  \
    #                                                                     --w_kl_C 0,0,100 \
    #                                                                     --w_ed_C 1,1,200 \
    #                                                                     --train_R 100000   \
    #                                                                     --batch_size 1024   \
    #                                                                     --res_dir ./results \
    #                                                                     --ex_name changew3/002 \
    #                                                                     --epoch_s 1 \
    #                                                                     --epoch_e 100 \
    #                                                                     --patience 100 \
    #                                                                     --hdelta 5',

    # 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py     --gpus 1 \
    #                                                                     --task estimation  \
    #                                                                     --w_kl_C 0,0,100 \
    #                                                                     --w_ed_C 1,1,200 \
    #                                                                     --train_R 100000   \
    #                                                                     --batch_size 1024   \
    #                                                                     --res_dir ./results \
    #                                                                     --ex_name changew3/003 \
    #                                                                     --epoch_s 1 \
    #                                                                     --epoch_e 100 \
    #                                                                     --patience 100 \
    #                                                                     --hdelta 7.5',

    # 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py     --gpus 1 \
    #                                                                     --task estimation  \
    #                                                                     --w_kl_C 0,0,100 \
    #                                                                     --w_ed_C 1,1,200 \
    #                                                                     --train_R 100000   \
    #                                                                     --batch_size 1024   \
    #                                                                     --res_dir ./results \
    #                                                                     --ex_name changew3/004 \
    #                                                                     --epoch_s 1 \
    #                                                                     --epoch_e 100 \
    #                                                                     --patience 100 \
    #                                                                     --hdelta 10',


    # 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py     --gpus 1 \
    #                                                                     --task estimation  \
    #                                                                     --w_kl_C 0,0,100 \
    #                                                                     --w_ed_C 1,1,200 \
    #                                                                     --train_R 100000   \
    #                                                                     --batch_size 1024   \
    #                                                                     --res_dir ./results \
    #                                                                     --ex_name changew3/005 \
    #                                                                     --epoch_s 1 \
    #                                                                     --epoch_e 100 \
    #                                                                     --patience 100 \
    #                                                                     --hdelta 12.5\
    #                                                                     ',

    # 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py     --gpus 1 \
    #                                                                     --task estimation  \
    #                                                                     --w_kl_C 0,0,100 \
    #                                                                     --w_ed_C 1,1,200 \
    #                                                                     --train_R 100000   \
    #                                                                     --batch_size 1024   \
    #                                                                     --res_dir ./results \
    #                                                                     --ex_name changew3/006 \
    #                                                                     --epoch_s 1 \
    #                                                                     --epoch_e 100 \
    #                                                                     --patience 100 \
    #                                                                     --hdelta 15\
    #                                                                     ',

    # 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}'+ ' python train2.py     --gpus 1 \
    #                                                                     --task estimation  \
    #                                                                     --w_kl_C 0,0,100 \
    #                                                                     --w_ed_C 1,1,200 \
    #                                                                     --train_R 100000   \
    #                                                                     --batch_size 1024   \
    #                                                                     --res_dir ./results \
    #                                                                     --ex_name changew3/007 \
    #                                                                     --epoch_s 1 \
    #                                                                     --epoch_e 100 \
    #                                                                     --patience 100 \
    #                                                                     --hdelta 20\
    #                                                                     ',

    # 'cd Precipitation\nCUDA_VISIBLE_DEVICES={}' + ' python train2.py     --gpus 1 \
    #                                                                     --task estimation  \
    #                                                                     --w_kl_C 0,0,100 \
    #                                                                     --w_ed_C 1,1,200 \
    #                                                                     --train_R 100000   \
    #                                                                     --batch_size 1024   \
    #                                                                     --res_dir ./results \
    #                                                                     --ex_name changew3/008 \
    #                                                                     --epoch_s 1 \
    #                                                                     --epoch_e 100 \
    #                                                                     --patience 100 \
    #                                                                     --hdelta 25\
    #                                                                     ',

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

    gpus = [1,2,3]
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
