# dataset

## 1.download
> https://www.avl.class.noaa.gov/saa/products/search?sub_id=0&datatype_family=GVAR_IMG&submit.x=29&submit.y=8

- 数据存储格式: IR_2012/***.nc
- 从服务器下载数据: python download.py 

## 1.preprocessing
- 基本要求：以天为单位存储数据,使用h5py的"gzip"压缩方法存储为".nc"文件。
- 存储的的key为:2014.105, value为一个矩阵