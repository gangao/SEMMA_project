# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 09:56:52 2019

@author: ginge
"""

"""
1 独热编码（one-hot encoding）
"""
# =============================================================================
# 1.1 方法1：sklearn的OneHotEncoder
# =============================================================================
from sklearn import preprocessing  
enc = preprocessing.OneHotEncoder(categories='auto')
# 加入categories='auto'是直接按输入值的独一性来确定类比，这在0.22版本中是默认的。
# 不加的话，目前是按[0,MAX(VALUE)]的范围来确认最大值的。
# 现在不需要用LabelEncoder将字符先转为整数了
# exp1
enc.fit([[2],[3],[4]]) # fit学习编码：1个变量，3个类别
array = enc.transform([[3]]).toarray() # 编码转换
print(array)
# exp2
enc.fit([[0,0,3],[1,1,0],[0,2,1],[1,0,2]]) # fit学习编码：3个变量，2-4个类别
array = enc.transform([[1,2,3]]).toarray() # 编码转换
print(array)
# exp3
enc = preprocessing.OneHotEncoder()
enc.fit([['a','b','c'],['b','c','b'],['a','a','e'],['a','a','s']]) # fit学习编码：1个变量，3个类别
array = enc.transform([['b','c','s']]).toarray() # 编码转换
print(array)


# =============================================================================
# 1.2 方法2：pandas的get_dummies方法
# pandas.get_dummies(data, prefix=None, prefix_sep='_', dummy_na=False, 
# columns=None, sparse=False, drop_first=False)
# =============================================================================

import pandas as pd
import numpy as np
# data-输入的数据;prefix-转换后列名的前缀 ;dummy_na-是否增加一列表示空缺值
# exp1
s1=pd.Series(list('abcb'))
df1=pd.get_dummies(s1)
df1=pd.get_dummies(s1,sparse=True)
# exp2
s2=['a','b','c',np.nan]
df2=pd.get_dummies(s2,dummy_na=True)
# exp3
df3=pd.DataFrame({'A':['a','b','a'],'B':['b','a','c'],'C':[1,2,3]})
df4=pd.get_dummies(df3,prefix=['col1_','col2_'])


"""
2 归一化和标准化（normalization and standardization）
"""
from sklearn import preprocessing
from scipy.stats import rankdata

x=[[1],[2],[131],[25],[25],[19],[77],[79]]
norm_x = preprocessing.MinMaxScaler().fit_transform(x) # 归一化，x-min/max-min
stdd_x = preprocessing.StandardScaler().fit_transform(x) # 标准化，x-μ/σ
print(norm_x)
print(stdd_x)
print('原始顺序：', rankdata(x))
print('归一化顺序：', rankdata(norm_x))
print('标准化顺序：', rankdata(stdd_x))




















