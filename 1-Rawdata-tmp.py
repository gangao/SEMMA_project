# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 12:45:49 2019

@author: ginge
"""

"""""""""""""""""""""""""""""""""
        源数据准备
"""""""""""""""""""""""""""""""""

"""
1 读取sas数据
"""
import pandas as pd
rd=pd.read_sas('D:\DATA\S_MODEL\model_data_20170606.sas7bdat')
#pandas.read_sas(filepath_or_buffer, format=None, index=None, encoding=None, 
#               chunksize=None, iterator=False)
#chunksize: int,Read file chunksize lines at a time, returns iterator.
#iterator: bool,If True, returns an iterator for reading the file incrementally.
#Returns: DataFrame if iterator=False and chunksize=None, else SAS7BDATReader or XportReader
import scorecardpy as sc
dat = sc.germancredit()

"""
2 查看原始数据的一些属性
"""

index1 = rd.index
dtype1 = rd.dtypes
columns1 = rd.columns
rd.size
rd.shape
ndim1 = rd.ndim
type(rd.axes)
print(rd.axes)
rd.memory_usage(deep=True) # 每列的内存使用量（以字节为单位）
rd.info
rd.values

rd.isna()
rd.bool()

转换列类型 astype
rd.astype({'Stkcd':'category'}).dtypes


rd.head(5)

观察数量
列名和数量
索引情况
列的数据类型
数值型变量的相关统计量&绘图
分类型变量的频数&绘图


# 对每一列做描述性统计（去除了缺失值）
# Generate descriptive statistics that summarize the central tendency, dispersion 
# and shape of a dataset’s distribution, excluding NaN values.
# DataFrame.describe(self, percentiles=None, include=None, exclude=None)
# percentiles默认 [.25, .5, .75]
# include默认None为数值型（numeric）；include='all'；
# include=[np.number]；include=[np.object]；include=['category']
describe = rd.describe()
describe = rd.describe(percentiles=[.10, .25, .5, .75, .9], include='all')

# 对于非数值型的列，如category，object，timestamp输出频数分布，且缺失值也纳入统计
# 对于数值型的列，除统计缺失值外，也做描述性统计
# sas中的.B,.C这种不同类型的缺失值如何在pandas中实现？
# 用给定的format对数值和非数值型的列做频数分布

"""
3 空值"",缺失值NaN、None
"""
from numpy import NaN
test = {'id':[11,12,13,14,15], 'birthday':['2000-01-01','',None,'2000-01-19',NaN],
        'name':['王欧派','周士大',NaN,None,'林离开'], 'score':[100,0,NaN,98,None]}

test = pd.DataFrame(test)
test.dtypes
test['name'].count()
test.count()
test = test.set_index('id')
test = test.astype({'birthday':'datetime64'})
test.dtypes

isnull = test.isna()
test.isna().any()
test.isna().all()

# 1 先得判断是否有空值''或空格（包括多个空格）,然后将空值填充为NaN
test = {'id':[11,12,13,14,15,16], 'birthday':['2000-01-01','  ',None,'','2000-01-19',NaN],
        'name':['王欧派','  ','周士大',NaN,None,''], 'score':[100,11,0,NaN,98,None]}
test = pd.DataFrame(test)

# 1.1 方法1：applymap和replace
import numpy as np
# 全是空格转为NaN，其中apply方法要取列，再循环做，applymap则不用
test['birthday'] = test['birthday'].apply(lambda x: np.NaN if str(x).isspace() else x) 
test1 = test.applymap(lambda x: np.NaN if str(x).isspace() else x)
# 空值''转为NaN
test1 = test1.replace('',np.NaN)

# 1.2 方法2：replace + 正则，同时做两件事情：把空格和空值''转为NaN
test = test.replace(to_replace=r'\s+', value=np.NaN, regex=True).replace('', np.NaN) # \s+表示一个或多个空格
test = test.replace(None, np.NaN)

#1.3 None转为NaN或相反
test = test.where((pd.notnull(test)), None) # pd.where():Replace values where the condition is False.
test = test.where((pd.notnull(test)), np.NaN) # pd.where():Replace values where the condition is False.


# 2 频数、统计量分析，带缺失值



import time
test['timestamp'] = test['birthday'].apply(lambda x:time.mktime(x.timetuple())) 
# 日期——>时间元组——>时间戳


"""
3 更改列类型
"""