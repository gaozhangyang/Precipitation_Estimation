# Infrared Precipitation Estimation

# 1. Framework

<div style="align: center">

![](https://github.com/gaozhangyang/Precipitation_Estimation/blob/master/gitfigure/pipline.png "pipeline")

![](https://github.com/gaozhangyang/Precipitation_Estimation/blob/master/gitfigure/network_structure.png "network structure")

</div>

# 2. Dataset

![](https://github.com/gaozhangyang/Precipitation_Estimation/blob/master/gitfigure/XY_val.gif "XY_val")

<div style="align: center">

| region | scope |
|:--:|:--:|
| RegC | 30N-45N, 90W-105W |
| RegW | 35N-40N, 110W-115W|
| RegE | 35N-40N, 80W-85W  |


| training set | evaluating set | testing set |
|:--:|:--:|:--:|
| RegC, June to July 2012   | RegC, August,2012 |   |
|                           |                       | RegC,June to August, 2014 <br> RegC,December 2012 to February 2013 |
|                           |                       | RegW and RegE, June to August 2012 |

</div>

# 3. Usage

 1. train
	```python
	python autotrain.py
	```
 2. evaluate
	 ```python 
	 cd Tools
	 python generate_mp4.py
	 ```

# 4. Performence

<div style="align: center">

![](https://github.com/gaozhangyang/Precipitation_Estimation/blob/master/gitfigure/val_iden_005.gif "val_iden_005") 

[supplementary materials of iden experiments](https://westlakeu-my.sharepoint.com/:f:/g/personal/gaozhangyang_westlake_edu_cn/ErUPhGHNTTlNlyWDbHbiNV0Bt50DFCll9JZPkBGzQ4y_og?e=PcyYrO)

| Name(iden,val) | Acc0 | Acc1 |
|:--|:--:|:--:|
| 001(NR/R=200000/50000)    |	0.8898 |	0.8950 |
| 002(200000/100000)        |	0.8894 |	0.9022 |
| 003(340000/340000)        |	0.8635 |	0.9297 |
| 004(400000/200000)        |	0.8887 |	0.9027 |
| 005(250000/50000)         |	0.9335 |	0.8044 |
| 006(300000/50000)         |	0.9389 |	0.7840 |
| 007(500000/100000)        |	0.9320 |	0.8092 |

</div>



<div style="align: center">

![](https://github.com/gaozhangyang/Precipitation_Estimation/blob/master/gitfigure/val_esti_013.gif "val_esti_013")

[supplementary materials of iden experiments](https://westlakeu-my.sharepoint.com/:f:/g/personal/gaozhangyang_westlake_edu_cn/Emr80jHzY2JOi35Puq2tQo4BBN0t39A7caTFXlja1qwUrQ?e=lzHaNN)


| Name(esti,val) | CC | BIAS | MSE | Acc0 | Acc1 |
|:--|:--:|:--:|:--:|:--:|:--:|
| 001(huber 2.5)    |0.3579	|0.5256	|0.5683	|0.9336	|0.8044|
| 002(huber 5)	    |0.3627	|0.7709	|0.5934	|0.9336	|0.8044|
| 003(huber 7.5)	|0.3634	|0.8883	|0.6126	|0.9337	|0.8041|
| 004(huber 10)	    |0.3625	|0.9633	|0.6277	|0.9339	|0.8032|
| 005(huber 12.5)   |0.3621 |1.027	|0.6420	|0.9339	|0.8028|
| 006(huber 15)     |0.3621 |1.060	|0.6501	|0.9341	|0.8008|
| 007(huber 20)	    |0.3616	|1.0858	|0.6604	|0.9344	|0.7994|
| 008(huber 25)	    |0.3590	|1.1206	|0.6732	|0.9352	|0.7920|
| 009(huber 12.5+KL)|0.3043	|0.2022	|0.7038	|0.9508	|0.6528|
| 010(huber20+KL)	|0.2928	|0.2763	|0.7715	|0.9464	|0.6994|
| 011(huber25+KL)   |0.2241 |0.6564 |1.091	|0.9463	|0.6875|
| 012(huber 2.5+KL)	|0.2684	|0.2143	|0.8625	|0.9494	|0.6877|
| 013(huber 5+KL)	|0.2919	|0.2316	|0.6972	|0.9429	|0.7326|
| 014(huber 7.5+KL)	|0.2447	|0.3118	|0.8018	|0.9494	|0.6736|
| 015(huber 10+KL)	|0.2974	|0.2311	|0.7527	|0.9458	|0.7113|

</div>

# 5. citation
```
{
    %0 Journal Article
    %T Infrared Precipitation Estimation Using Convolutional Neural Network
    %P 1-14
    %U https://ieeexplore.ieee.org/document/9085928/
    %G en
    %J IEEE Transactions on Geoscience and Remote Sensing
    %A Wang, Cunguang
    %A Xu, Jing
    %A Tang, Guoqiang
    %A Yang, Yi
    %A Hong, Yang
    %D 2020
}
```