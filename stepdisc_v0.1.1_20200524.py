# -*- coding: utf-8 -*-
"""
@name：stepdisc
@author: cshuo
@version: 0.1.1
@describe：逐步判别分析程序及相关子程序
    1）尝试实现sas中proc stepdisc过程步的功能，该功能用线性判别分析(LDA)中的wilks lambda统计量的前后差异（除法）来筛选入模变量，
        这种方法是多变量方法，并非单变量方法，考虑了变量之间的关系。
    2）目前实现的算法是实践中最多使用的stepwise方式，即有进有出的算法。
    3）具体的算法参照《多元统计分析》（于秀林 任雪松 中国统计出版社 1999版）3.1节、6.5节及6.2节中的例子数据编写而成
    4）该过程步的主要瓶颈在于样本量过大时计算两组好坏样本的协方差矩阵，即s1_cov = np.cov(s1_data.T, fweights=s1_wt)这里
        后续有一个计算样本各列均值，可能计算量要小很多。如果协方差矩阵计算出来了，比如1000个变量就是1000*1000的矩阵，需要后续
        计算多次逆矩阵，以及一些矩阵乘法，后面这些可能没有前面耗资源厉害。具体性能要大数据量调试了再知道。
@date：20200524
@to_do_list:
    1）把返回值keepvar由字典格式改为df格式其中一列是进入的step排序，一列应该是var的原序号，
        一列是操作enter/delete，一列是enter的变量名 ，一列是delete的变量名
    2）检查计算是否正确。已与sas proc stepdisc对比，完美！！！一模一样，哈哈哈哈！！
    3) include功能
    4）可以考虑做一个random keep功能，即随机选iv值前10的变量作为keep的第一个变量，然后再stepdisc，以形成
        多个不同的强变量组合集，放到下一步逐步逻辑回归中去。

# example: 来自《多元统计分析》p108-人文发展指数例
undp = np.array([[76,99,5374,0],[79.5,99,5359,0],[78,99,5372,0],[72.1,95.9,5242,0],[73.8,77.7,5370,0],
                [71.2,93,4250,1],[75.3,94.9,3412,1],[70,91.2,3390,1],[72.8,99,2300,1],[62.9,80.6,3799,1]])
undp_wt = np.column_stack((undp,[2,2,2,2,2,3,3,3,3,3]))
undp_wt = np.column_stack((undp,[1,1,1,1,1,1,1,1,1,1]))
df_undp = pd.DataFrame(undp_wt, columns=['var1','var2','var3','perf','wt'])

"""

import numpy as np
import pandas as pd
from scipy import stats

"""
stepdisc大逻辑
"""

def stepdisc(indata, target, weight=None, varlist=None, include=None, maxstep=None, sle=None, sls=None):
    '''
    stepdisc main function
    
    Params
    ------
    indata: dataframe type，建模样本数据，必须包含目标变量target
    target: indata中的二分类变量名，最好已处理为1和0
    weight: indata中的权重列名，如果缺失则造一个全为1的权重列
    varlist: list type，总变量名单，如果缺失则直接取indata的全部列名
    include：list type，筛选变量结果中默认已包含的变量名单，目前版本未实现      
    maxstep: 最大stepwise的步数，比如设为100，则最后最多选出100个变量。
    sle: 变量选取的f统计值对应的置信概率，行业默认0.15，即15%
    sls: 变量剔除的f统计值对应的置信概率，行业默认0.15，即15%
        
    Returns
    ------
    dict： 返回保留的变量名单，字典格式
    '''
    #1 预处理
        #如果target不是1，0改为10
        #如果一些检查报错，报错
        #确定maxstep
    if sle == None:
        sle = 0.15
    if sls == None:
        sls = 0.15
    #2 制作varlist、vardict、varkeep、varleft
    if varlist == None:
        varlist = list(indata.columns)
        varlist.remove(target)
        if weight != None: #如原数据有wt字段，则去除
            varlist.remove(weight)
    else:
        #如果有定制好的varlist，要做几件事
        #1）indata的数据集根据varlist再重新挑选出来
        #2）vardict的key要根据原数据集和varlist重新制作
        pass #待补充
    if maxstep == None:
        maxstep = len(varlist) * 2
    vardict = {index:item for index, item in enumerate(varlist)}
    varkeep = {} #选取的变量列表组成的字典，key是在原数据集中的列序号
    varleft = vardict #原数据集中备选的变量列表组成的字典，key是在原数据集中的列序号
    #3 数据处理和基础值计算
    #3.1 分别取出两组数据和对应varlist的列
    s1_data = indata[indata[target] == 0][varlist]
    s2_data = indata[indata[target] == 1][varlist]
    print('s1_data : ',s1_data)
    print('s2_data : ',s2_data)
    #3.2 制作weight列
    if weight == None: #如原数据无wt。则做一列weight全为1
        sum1_nowt = len(indata[indata[target] == 0])
        sum2_nowt = len(indata[indata[target] == 1])
        s1_wt = pd.Series(list(1 for i in range(0,sum1_nowt)))
        s2_wt = pd.Series(list(1 for i in range(0,sum2_nowt)))
        s_wt = pd.Series(list(1 for i in range(0,sum1_nowt+sum2_nowt)))
    else:
        s1_wt = indata[indata[target] == 0][weight]
        s2_wt = indata[indata[target] == 1][weight]
        s_wt = indata[weight]
    print('s_wt : ',s_wt)
    print('s1_wt : ',s1_wt)
    print('s2_wt : ',s2_wt)
    #3.3 分别计算两组的协方差矩阵e
    s1_cov = np.cov(s1_data.T, fweights=s1_wt)
    s2_cov = np.cov(s2_data.T, fweights=s2_wt)
    print('s1_cov : ',s1_cov)
    print('s2_cov : ',s2_cov)
    #3.4 分别计算两组权重和
    sum_wt1 = s1_wt.sum()
    sum_wt2 = s2_wt.sum()
    #3.5 e是加权后的协方差矩阵之和，也就是组内离差矩阵 --需要保留，一直使用
    e_matrix = s1_cov * (sum_wt1 - 1) + s2_cov * (sum_wt2 - 1) #这里因是样本要n-1
    print('e_matrix : ',e_matrix)
    #3.5 然后算组间离差矩阵a
    s_mean = np.average(indata[varlist].T, axis=1, weights=s_wt) #加权平均
    s1_mean = np.average(s1_data.T, axis=1, weights=s1_wt) #加权平均
    s2_mean = np.average(s2_data.T, axis=1, weights=s2_wt) #加权平均
    s1_mean_diff = s1_mean - s_mean
    s2_mean_diff = s2_mean - s_mean
    s1_matrix = sum_wt1 * s1_mean_diff[:,None] * s1_mean_diff #这里没有n-1
    s2_matrix = sum_wt2 * s2_mean_diff[:,None] * s2_mean_diff
    a_matrix = s1_matrix + s2_matrix
    #3.6  然后算矩阵t   T = E + A --也需要保留，一直使用
    t_matrix = e_matrix + a_matrix
    print('t_matrix : ',t_matrix)
    #4 循环
    for step in range(0,maxstep):
        print('step: ', step)
        backward_step(step, varkeep, sls, sum_wt1, sum_wt2, e_matrix, t_matrix) #检查是否有变量需要剔除
        varkeep_bf = len(varkeep)
        forward_step(step, varkeep, varleft, sle, sum_wt1, sum_wt2, e_matrix, t_matrix) #检查是否有变量需要进入
        varkeep_aft = len(varkeep)
        if len(varkeep) == 0: #一个没进入？
            print('No any Variable entered!')
            break
        if len(varleft) == 0: #没有剩余变量了
            print('No Variables left!')
            break
        if varkeep_bf == varkeep_aft: #这次循环没有变量进入，则跳出
            print('This variable can not be entered, Break!')
            break
    return varkeep

 
"""
backward step
"""
def backward_step(step, varkeep, sls, sum_wt1, sum_wt2, e_matrix, t_matrix):
    print('B-STEP: ', step)
    if len(varkeep) <= 1: #0时为未进入任何变量，1时为只进入一个变量，都不需要检查退出条件
        print('#0时为未进入任何变量，1时为只进入一个变量，都不需要检查退出条件')
        print_log_wilk_lambda(0, 'backward', step, varkeep=None, df=None)
    else:
        print('in backward_step: else ~')
        sls_varlist = pd.DataFrame(columns=['varno','varname','F_stats_wilk_lambda','Pr_F']) #放wilks lambda相除部分差值的数据表
        varkeep_num = len(varkeep) #varkeep_num是前一次循环引入留下的变量数
        dfd_stay = sum_wt1 + sum_wt2 - (varkeep_num - 1) - 2 # n - k - (L - 1),n是总样本数，k是组数，二分类为2，L是上一次循环留下的变量数
        for n, i in enumerate(varkeep.keys()): #i是从已选择变量中挑
            varkeep_left = varkeep.copy()
            del varkeep_left[i]
            keep_index = list(varkeep_left.keys()) #已选取变量的序号，用于选取e及t中的矩阵E11/T11
            print('varkeep - i : ', i)
            print('varkeep_left: ', keep_index)
            side_index = list(i for x in range(0,len(keep_index))) #根据当前判断变量i，制作i列的列表，长度为已选取变量的个数，用于选取ER1
            err_l = calc_Mrr(i, e_matrix, keep_index, side_index) #err_l = err - Er1*np.linalg.inv(E11)*E1r
            trr_l = calc_Mrr(i, t_matrix, keep_index, side_index) #trr_l = trr - Tr1*np.linalg.inv(T11)*T1r
            
            wilk_lambda_ratio = err_l/trr_l #实际这里是err_(l-1)/trr_(l-1)
            print('wilk_lambda_ratio: ', wilk_lambda_ratio)
            F_stats_wilk_lambda =  (1 - wilk_lambda_ratio) / wilk_lambda_ratio * dfd_stay / 1 #计算最终的wl(l)/wl(l-1)的统计量即(1-Ar/Ar)*[(n-k-L+1)/(k-1)]
            Pr_F = 1-stats.f.cdf(F_stats_wilk_lambda,dfn=1,dfd=dfd_stay) if 1-stats.f.cdf(F_stats_wilk_lambda,dfn=1,dfd=dfd_stay) >= 0.0001 else 0.0001
            
            sls_varlist = sls_varlist.append({'varno': i, 'varname': varkeep[i],\
                                              'F_stats_wilk_lambda': F_stats_wilk_lambda, 'Pr_F': Pr_F},\
                                             ignore_index = True)
        
        #跳出循环后，选择最大的统计量（即最小Ar）进行检验，然后删除或不删除   
        sls_varlist.sort_values(by=['F_stats_wilk_lambda'], ascending=True, inplace=True) #sls_varlist按F_stats_wilk_lambda从小到大排序，取最小的进行判断
        if sls_varlist.iloc[0].loc['F_stats_wilk_lambda'] <= stats.f.ppf(1-sls,dfn=1,dfd=dfd_stay): #剔除是判断小于等于，stats来自scipy
            print('in backward_step-else: if <=')
            print_log_wilk_lambda(1, 'backward', step, varkeep=None, df=sls_varlist)
            drop_out(varkeep, sls_varlist) #剔除该变量
        else:
            print('in backward_step-else: else')
            print_log_wilk_lambda(0, 'backward', step, varkeep=None, df=sls_varlist)

"""
drop_out()
"""
def drop_out(varkeep, sls_varlist):
    drop_var = sls_varlist.iloc[0].loc['varno']
    del varkeep[drop_var]
    return None

"""
calc_Mrr()
"""
def calc_Mrr(i, matrix, keep_index, side_index):
    #err_l = err - Er1*np.linalg.inv(E11)*E1r
    Mrr = matrix[i,i] # 即xi
    M11 = matrix[keep_index][:,keep_index]
    Mr1 = matrix[keep_index,side_index] # 某行1列
    M1r = Mr1.T
    M1 = np.dot(Mr1,np.linalg.inv(M11))
    M2 = np.dot(M1,M1r)
    Mrr_l = Mrr - M2
    print(' calc_Mrr - i : ', i)
    print(' Mrr : ', Mrr)
    print(' M11 : ', M11)
    print(' Mr1 : ', Mr1)
    print(' M2 : ', M2)
    print(' Mrr_l : ', Mrr_l)
    return Mrr_l


"""
forward step
"""
def forward_step(step, varkeep, varleft, sle, sum_wt1, sum_wt2, e_matrix, t_matrix):
    if len(varleft) < 1: #即已没有剩下的变量了
        print_log_wilk_lambda(-1, 'forward', step, varkeep=varkeep, df=None)
        print('即已没有剩下的变量了')
    else:
        sle_varlist = pd.DataFrame(columns=['varno','varname','F_stats_wilk_lambda','Pr_F']) #放wilks lambda相除部分差值的数据表
        varkeep_num = len(varkeep) #varkeep_num是前一次循环引入留下的变量数
        dfd_entry = sum_wt1 + sum_wt2 - varkeep_num - 2 #n-L-k，n是总样本，k是组数，二分类就是k=2，L是上一次循环留下的变量数
        print('dfd_entry: ', dfd_entry)
        print('varkeep_num: ', varkeep_num)
        for n, i in enumerate(varleft.keys()): #i是从未选取变量里面挑,i是对应的原始变量的列序号，与n不一定相同
            keep_index = list(varkeep.keys()) #已选取变量的序号，用于选取e及t中的矩阵E11/T11
            print('varleft - i : ', i)
            print('keep_index: ', keep_index)
            side_index = list(i for x in range(0,len(keep_index))) #根据当前判断变量i，制作i列的列表，长度为已选取变量的个数，用于选取ER1
            print('side_index: ', side_index)
            err_l = calc_Mrr(i, e_matrix, keep_index, side_index) #err_l = err - Er1*np.linalg.inv(E11)*E1r
            trr_l = calc_Mrr(i, t_matrix, keep_index, side_index) #trr_l = trr - Tr1*np.linalg.inv(T11)*T1r
            print('err_l: ', err_l)
            print('trr_l: ', trr_l)
            wilk_lambda_ratio = err_l/trr_l 
            print('wilk_lambda_ratio: ', wilk_lambda_ratio)
            F_stats_wilk_lambda = (1 - wilk_lambda_ratio) / wilk_lambda_ratio * dfd_entry / 1 #计算最终的wl(l)/wl(l-1)的统计量即(1-Ar/Ar)*[(n-k-L)/(k-1)]
            print('enumerate(varleft.keys()): ', i)
            print('F_stats_wilk_lambda: ', F_stats_wilk_lambda)

            Pr_F = 1-stats.f.cdf(F_stats_wilk_lambda,dfn=1,dfd=dfd_entry) if 1-stats.f.cdf(F_stats_wilk_lambda,dfn=1,dfd=dfd_entry) >= 0.0001 else 0.0001
            print('Pr_F: ', Pr_F)
            sle_varlist = sle_varlist.append({'varno': i, 'varname': varleft[i],\
                                              'F_stats_wilk_lambda': F_stats_wilk_lambda, 'Pr_F': Pr_F},\
                                             ignore_index = True)
            print('sle_varlist: ', sle_varlist)
        #跳出循环后，选择最大的统计量（即最小Ar）进行检验，然后删除或不删除   
        sle_varlist.sort_values(by=['F_stats_wilk_lambda'], ascending=False, inplace=True) #sle_varlist按F_stats_wilk_lambda从大到小排序，取最大的进行判断
        print('stats.f.ppf(1-sle,dfn=1,dfd=dfd_entry): ',stats.f.ppf(1-sle,dfn=1,dfd=dfd_entry))
        print("sle_varlist.iloc[0].loc['F_stats_wilk_lambda']: ",sle_varlist.iloc[0].loc['F_stats_wilk_lambda'])
        if sle_varlist.iloc[0].loc['F_stats_wilk_lambda'] > stats.f.ppf(1-sle,dfn=1,dfd=dfd_entry): #加入是判断大于，stats来自scipy
            print('add_in！！！！ ', sle_varlist.iloc[0])
            add_in(varkeep, varleft, sle_varlist) #先在varkeep中加入该变量
            print_log_wilk_lambda(1, 'forward', step, varkeep=varkeep, df=sle_varlist) #再打印出来
        else:
            print('no_add_in!!! ')
            print_log_wilk_lambda(0, 'forward', step, varkeep=varkeep, df=sle_varlist)

"""
drop_out()
"""
def add_in(varkeep, varleft, sle_varlist):
    add_var = sle_varlist.iloc[0].loc['varno']
    varkeep[add_var] = sle_varlist.iloc[0].loc['varname']
    del varleft[add_var]
    return None

"""
print_log_wilk_lambda()
"""
def print_log_wilk_lambda(tag, steptype, step, varkeep=None, df=None):
    if steptype == 'backward':
        print('********  STEPWISE SELECTION: ',step,' ********')
        print('********    Sub-Step: backward    ********')
        if tag == 0:
            if df is not None:
                print('******** Statistics for removeal ********')
                print(' ')
                print(df.head(100)) #暂时最多只打印前100个变量
                print(' ')
                print('No variables can be removed')
            else:
                print('No need to check, skip')
        elif tag == 1:
            print('******** Statistics for removeal ********')
            print(' ')
            print(df.head(100))
            print(' ')
            print('Variable ',df.iloc[0].loc['varname'],' will be removed')
    elif steptype == 'forward':
        print('********  STEPWISE SELECTION: ',step,' ********')
        print('********    Sub-Step: forward    ********')
        if tag == 0:
            print('******** Statistics for entry ********')
            print(' ')
            print(df.head(100))
            print(' ')
            print('This variable can not be entered')
            print('Variables have been entered')
            for i in varkeep.keys(): #打印变量名在同一行
                print('%s,'%(varkeep[i]), end=' ') 
                if(i % 10 == 0 and i!= 0):
                    print('') #换行输出
            print('')
        elif tag == 1:
            print('******** Statistics for entry ********')
            print(' ')
            print(df.head(100))
            print(' ')
            print('Variable ',df.iloc[0].loc['varname'],' will be entered')
            print('Variables have been entered')
            for i in varkeep.keys(): #打印变量名在同一行
                print('%s,'%(varkeep[i]), end=' ') 
                if(i % 10 == 0 and i!= 0):
                    print('') #换行输出
            print('')
        elif tag == -1:
            print('******** No Variables Left, Break! ********')
        
    
'''
test

'''
undp = np.array([[76,99,5374,0],[79.5,99,5359,0],[78,99,5372,0],[72.1,95.9,5242,0],[73.8,77.7,5370,0],
                [71.2,93,4250,1],[75.3,94.9,3412,1],[70,91.2,3390,1],[72.8,99,2300,1],[62.9,80.6,3799,1]])
#undp_wt = np.column_stack((undp,[2,2,2,2,2,3,3,3,3,3]))
undp_wt = np.column_stack((undp,[1,1,1,1,1,1,1,1,1,1]))
df_undp = pd.DataFrame(undp_wt, columns=['var1','var2','var3','perf','wt'])

test_var_keep = stepdisc(df_undp, 'perf', 'wt', sle=0.2, sls=0.2)

