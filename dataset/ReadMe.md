# dataset

## 文件结构
- IR
    - IR_2012_raw
    - IR_2014_raw
    - IR_2012_Qinghua
    - IR_2014_Qinghua

- StageIV
    - StageIV_raw
    - StageIV_Qinghua

- dataset
    - IR_StageIV_Qinghua
        - X_train_hourly.npz
        - Y_train_hourly.npz
        - train.csv
        - X_test_hourly.npz
        - Y_test_hourly.npz
        - test.csv
        - X_val_hourly.npz
        - Y_val_hourly.npz
        - val.csv

## 1.download
> https://www.avl.class.noaa.gov/saa/products/search?sub_id=0&datatype_family=GVAR_IMG&submit.x=29&submit.y=8

- 数据存储格式: IR_2012/***.nc
- 从服务器下载数据: python download.py 

## 1.preprocessing
- 基本要求：以天为单位存储数据,使用h5py的"gzip"压缩方法存储为".nc"文件。
- 存储的的key为:2014.105, value为一个矩阵