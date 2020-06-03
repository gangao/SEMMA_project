# -*- coding: utf-8 -*-
"""
Created on Tue May 26 09:10:33 2020

@author: ginge

尝试用statsmodel里面的logit回归实现stepwise功能：
1、forward基于score chi-squire统计量 ——目前未实现成功，按理说是如下计算，但值不对
            U = mod.score(res.params)
            info_matrix = -1 * mod.hessian(res.params)# 信息矩阵 = 负的hessian矩阵 logit的information未实现
            I = np.linalg.inv(info_matrix)
            Score1 = np.dot(U.T,I)
            Score = np.dot(Score1,U)
            
2、backward基于wald chi-squire统计量
"""



'''
varlist
varlist_keep
varlist_left
varlist_entry



0.set some of the arguments:
  varlist_keep = varlist_include
  varlist_left = varlist - varlist_entry
    
1.step_loop start:
    2.forward_loop start: n= len(varlist_left)
        2.1.varlist_keep + loop(1 of varlist_left )
        2.2.fit model, output the new var's wald stats to a wald_var_df
    3.sort wald_var_df, check if the max one meet the threshold
        3.1 if meet: add the new var into varlist_entry
                    delete from varlist_left
        3.2 if not meet: break loop
    4.backward_loop start: n= len(varlist_entry)
        4.1 fit model?? output every wald stats to a ward_var_df
    5.sort wald_var_df, check if the max one meet the threshold
        if meet: delete the var
                 go back to 4?
        if not meet: pass


要打印出来一些统计结果，
要出来一张表，记录每个step的决策
最终模型结果的呈现？
加入vif和corr的判断
手工剔除变量的设定
'''


import statsmodels.api as sm
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from patsy import dmatrices

def logistic_stepwise(indata, target, weight=None, varlist=None, maxstep=None, \ARMED
                      include=None, exclude=None, sle=None, sls=None, corr_check=0.75, vif_check=10)
    '''
    logistic_stepwise main function
    
    Params
    ------
    indata: dataframe type，建模样本数据，必须包含目标变量target
    target: indata中的二分类变量名，最好已处理为1和0
    weight: indata中的权重列名，如果缺失则造一个全为1的权重列
    varlist: list type，总变量名单，如果缺失则直接取indata的全部列名
    include：list type，筛选变量结果中默认已包含的变量名单
    exclude: list type，默认剔除的变量名单
    maxstep: 最大stepwise的步数，比如设为100，则最后最多选出100个变量。
    sle: 变量选取的f统计值对应的置信概率，行业默认0.15，即15%
    sls: 变量剔除的f统计值对应的置信概率，行业默认0.15，即15%
        
    Returns
    ------
    dict： 返回保留的变量名单，字典格式
    
    main logic
    ------
    0 目前向前回归所用的score统计量还无法实现，只能用wald统计量代替。
    1 逐步加入单个变量进行逻辑回归，包括向前回归forward和向后回归backward两步
    2 向前回归的逻辑
    2.1 向前回归时，将所有的剩余变量单个循环加入已保存变量中，计算回归后的score统计量，
        判断单个模型的所有变量vif是否均满足标准，以及判断单个变量的所有变量之间的相关系数是否满足标准；
    2.2 向前回归完成后，排序和检查score统计量最高的那个，如满足所有条件则选入已保存变量。
    2.3 如果未加入任何变量，则代表变量挑选结束，不进行下一步向后回归，直接跳出循环。
    3 向后回归的逻辑
    3.1 向前回归完成后，进行向后回归，主要是统计检查上一步已保存变量组成的模型中，所有变量的wald统计量是否满足标准，
        只要有一个任一变量不满足，则从已保存变量中剔除上一步加入的那个特定变量。
    3.2 如果均满足条件，则不变动已保持变量。
    4 每一次执行完向前和向后回归后，将结果记录在同一张表中
    
    '''
    #1 主要变量预处理
    varlist = varlist_check(indata, target, weight, varlist)
    varkeep, varleft = varlist_initial(varlist, include, exclude)
    if maxstep == None:
        maxstep = len(varlist) * 2
    for step in range(0, maxstep):
        #1 制作一个stepwise_record空表
        stepwise_record = pd.DataFrame(columns=['step','serial','operation','varname', 'corr_check', 'vif_check'])
 
        varkeep_bf = len(varkeep)
        
        #2 先选入模变量
        stepwise_record, varkeep, varleft = forward_logit()
        
        if varkeep_bf == varkeep_aft: #这次循环没有变量进入，则跳出
            print('This variable can not be entered, Break!')
            break
        
        #2 再查看可剔除变量
        stepwise_record, varkeep, varleft = backward_logit()
        
        varkeep_aft = len(varkeep)
        if len(varleft) == 0: #没有剩余变量了
            print('No Variables left!')
            break



"""
"""        
def backward_logit(step, varkeep):
    #1 用已入模的变量列表进行回归，只一次
    mod = sm.Logit(target, varkeep, freq_weights=indata[weight]) 
    res = mod.fit()
    #2 制作backward_loop_record表
    df = res.wald_test_terms(skip_single=False).table
    backward_loop_record = df[~(df.index == 'Intercept')].reset_index().reset_index()
    backward_loop_record.rename(columns={"level_0": "varno", "index": "varname", "statistic": "wald_stats"})
    #3 查看所有已入模变量的wald统计量，从小到达排序
    backward_loop_record.sort_values(by=['wald_stats'], ascending=True, inplace=True) #wald_stats从小到大
    if backward_loop_record.iloc[0].loc['pvalue'] <= sls: #判断
        print('backward, out')
        backward_logit_drop_out(varkeep, varleft, forward_loop_record.iloc[0].loc['varname'])
        return stepwise_record
        
    else:
        print('backward, no out')
        return stepwise_record
        

    
"""
"""

def forward_logit(step, varkeep, varleft, vif_check=10, corr_check=0.75):
    """
    逻辑：
    1 以varkeep为基础，循环检查varleft里面的单变量进行逻辑回归，并记录每次循环的结果进入一张表，列内容有：
    1.1）输出该单变量的score test统计量
    1.2）vif检查结果：如有设置，则加入单个变量回归时，会检查该模型所有变量的vif情况，如不满足vif阈值条件，则输出'fail'
    1.3）corr检查结果：如有设置，则加入单个变量回归时，会检查该模型所有变量之间的相关系数（实际只受该变量与其他变量的影响），
        如不满足corr阈值条件，则输出'fail' ———— corr这步应该可以放到开头一起做？？效率更高？
    1.4）逻辑回归不断加入变量后，一定可以收敛吗？
    2 所有循环完了之后，剔除vif和corr不满足条件的（如有设置），按score统计量从大到小排序，然后取满足条件的最大一个。如均不满足条件，说明无变量可入模。   
    """
    #1 制作一个forward_loop_record空表
    forward_loop_record = pd.DataFrame(columns=['varno', 'varname', 'wald_stats', 'pvalue', 'df_constraint', 'vif_check' ,'corr_check'])
    #2 开始循环
    for i, varname in enumerate(varleft):
        #2.1 已入模变量（varkeep）与新变量varname组成新的x变量列表
        x_var = varkeep + varname
        #2.2 用新的x变量列表进行回归
        mod = sm.Logit(target, x_var, freq_weights=indata[weight]) 
        res = mod.fit()
            
        #2.3 当前变量集X_VAR中所有变量的单变量VIF，记录该变量集中是否有大于vif阈值的变量
        vif_table = pd.DataFrame(columns = ['varno', 'varname' , 'vif'])
        reg_data = indata[x_var] #需包含常量
        col = list(range(data.shape[1]))
        if weight != None:
            obs_wt = indata[weight]
        else:
            obs_wt = None
        for i in range(len(x_var)- 1):
            vif = _compute_vif(reg_data.iloc[:,col].values, i+1, weights=obs_wt)
            # i = 2
            # x_var = list1
            # vif = 11
            vif_table = vif_table.append({'varno':i+1, 'varname':x_var[i+1], 'vif':vif}, ignore_index = True)
            
        print(vif_table)
        vif_threshold = vif_check
        if len(vif_table[vif_table.vif >= vif_threshold]) > 0:
            print('VIF > ',vif_threshold)
            print(vif_table[vif_table.vif >= vif_threshold])
            pass # 将vif信息加入forward_loop_record作为排除条件
            vif_check = 'pass'
        else:
            vif_check = 'fail'
            pass
        
        #2.4 当前变量集X_VAR中所有变量的CORR矩阵，记录该变量集中是否有大于corr阈值的变量
        corr_check(indata_x, corr_threshold=corr_check, weight=weight):
    
        #2.5 新变量varname的单变量wald
        i=0
        varname = 'selfLR'
        statistic = res.wald_test_terms(skip_single=False).table.loc[varname, 'statistic']#从回归结果中取出特定变量的wald统计量相关结果
        pvalue = res.wald_test_terms(skip_single=False).table.loc[varname, 'pvalue']#从回归结果中取出特定变量的wald统计量相关结果
        df_constraint = res.wald_test_terms(skip_single=False).table.loc[varname, 'df_constraint']#从回归结果中取出特定变量的wald统计量相关结果
        #插入forward_loop_record表
        forward_loop_record = forward_loop_record.append({'varno': i, 'varname': varname,\
                                              'wald_stats': statistic, 'pvalue': pvalue,\
                                                  'df constraint': df_constraint},\
                                             ignore_index = True)
        
        #2.6  整体变量wald--是否要进行判断？？
        A = np.identity(len(res.params))
        A = A[1:,:]
        print('Wald test for all vars: ', res.wald_test(A))
        
    #3  跳出循环后，按wald统计量从大到小排序，选最大的进行检验，然后选取变量或不选取
    forward_loop_record.sort_values(by=['wald_stats'], ascending=False, inplace=True) #wald_stats从大到小
    if forward_loop_record.iloc[0].loc['pvalue'] <= sle: #判断
        print('forward, enter')
        forward_logit_add_in(varkeep, varleft, forward_loop_record.iloc[0].loc['varname'])
        return stepwise_record
        
    else:
        print('forward, not enter')
        return stepwise_record


"""
""" 
def forward_logit_add_in(varkeep, varleft, varname):
    

    


"""
问题1：
VIF -- WEIGHT怎么解决？？--已解决，用_compute_vif即可，权重单独列输入
与sas进行对比分析过，有以下结论：
1）某变量的vif是用该变量与剩余变量进行直接线性回归得到的，有截距项
2）vif是用1/（1-R方）计算的，不是用调整后的R方
3）加了权重后的算法就是普通的加权重的线性回归

""" 
import statsmodels
from statsmodels.stats.outliers_influence import variance_inflation_factor
#用sas进行测试，使用的是未调整过的r方进行的计算。
data = spector_data_df[['GPA', 'TUCE', 'PSI']]
data = sm.add_constant(data)
col = list(range(data.shape[1]))
variance_inflation_factor(data.iloc[:,col].values, 1) # 此方法与sas计算完全一致 --1.1761582341579837
variance_inflation_factor(data.iloc[:,col].values, 2) # 此方法与sas计算完全一致 --1.1894350280708073
variance_inflation_factor(data.iloc[:,col].values, 3) # 此方法与sas计算完全一致 --1.012902241028604

# sas/*参数估计 */
# /*变量 自由度 参数估计 标准误差 t值 Pr>|t| 方差膨胀 */
# /*Intercept 1 -1.49802 0.52389 -2.86 0.0079 0 */
# /*GPA 1 0.46385 0.16196 2.86 0.0078 1.17616 */
# /*TUCE 1 0.01050 0.01948 0.54 0.5944 1.18944 */
# /*PSI 1 0.37855 0.13917 2.72 0.0111 1.01290 */

#加权的数据
#该方法与sas算的也一样，不过要注意数据里面不带权重，权重是另外加的。
_compute_vif(data.iloc[:,col].values, 1, weights=spector_data_df['wt']) #--1.1071832363814058
_compute_vif(data.iloc[:,col].values, 2, weights=spector_data_df['wt']) #--1.177906420323179
_compute_vif(data.iloc[:,col].values, 3, weights=spector_data_df['wt']) #--1.07916579276853


# sas/*参数估计 */
# /*变量 自由度 参数估计 标准误差 t值 Pr>|t| 方差膨胀 */
# /*Intercept 1 -1.03932 0.51610 -2.01 0.0537 0 */
# /*GPA 1 0.41306 0.14697 2.81 0.0089 1.10718 */
# /*TUCE 1 -0.00506 0.01938 -0.26 0.7959 1.17791 */
# /*PSI 1 0.48714 0.13891 3.51 0.0015 1.07917 */


_compute_vif(data.iloc[:,col].values, 1, weights=None) # 1.1761582341579837
_compute_vif(data.iloc[:,col].values, 2, weights=None) # 1.1894350280708073
_compute_vif(data.iloc[:,col].values, 3, weights=None) # 1.012902241028604



def vif_check(weight)




dir(sm.stats)


variance_inflation_factor(trainingdata_x.iloc[:,col].values, 0 )
variance_inflation_factor(trainingdata_x.iloc[:,col].values, 0, freq_weights='wt' )

## 参考
## 每轮循环中计算各个变量的VIF，并删除VIF>threshold 的变量
def vif(X, thres=10.0):
    col = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:,col].values, ix)
               for ix in range(X.iloc[:,col].shape[1])]
        
        maxvif = max(vif)
        maxix = vif.index(maxvif)
        if maxvif > thres:
            del col[maxix]
            print('delete=',X_train.columns[col[maxix]],'  ', 'vif=',maxvif )
            dropped = True
    print('Remain Variables:', list(X.columns[col]))
    print('VIF:', vif)
    return list(X.columns[col]) 


vif(data.exog)
X = data.exog
ix = [ix for ix in range(X.iloc[:,col].shape[1])]
variance_inflation_factor(X.iloc[:,col].values, 0)

dir(statsmodels.stats.outliers_influence)
dir(statsmodels.stats.diagnostic)
X.iloc[:,col].shape[1]

col = list(range(data.exog.shape[1]))
data.exog.iloc[:,col].shape[1]

from statsmodels.api import OLS
from statsmodels.api import WLS
def _compute_vif(exog, exog_idx, weights=None, model_config=None):
    """
    Compute variance inflation factor, VIF, for one exogenous variable
    for OLS and WLS that allows weights.
    Parameters
    ----------
    exog: X features [X_1, X_2, ..., X_n]
    exog_idx: ith index for features
    weights: weights
    model_config: {"hasconst": True,
    "cov_type": "HC3"} by default
    
    Returns: vif
    -------
    """
    if model_config is None:
        model_config = {"hasconst": True,
                        "cov_type": "HC3"}
    k_vars = exog.shape[1]
    x_i = exog[:, exog_idx]
    mask = np.arange(k_vars) != exog_idx
    x_noti = exog[:, mask]
    if weights is None:
        r_squared_i = OLS(x_i,
                          x_noti,
                          hasconst=model_config["hasconst"]).fit().rsquared
    else:
        r_squared_i = WLS(x_i,
                          x_noti,
                          hasconst=model_config["hasconst"],
                          weights=weights).fit(
            cov_type=model_config["cov_type"]).rsquared
    vif = 1. / (1. - r_squared_i)
    return vif


_compute_vif(trainingdata_x.iloc[:,col].values, 0, weights='wt')
_compute_vif(data.exog.iloc[:,col].values, 0)
variance_inflation_factor(data.exog.iloc[:,col].values, 0)
variance_inflation_factor(exog=data.exog.iloc[:,col].values, exog_idx=0) 


def variance_inflation_factor(exog, exog_idx):
    """variance inflation factor, VIF, for one exogenous variable

    The variance inflation factor is a measure for the increase of the
    variance of the parameter estimates if an additional variable, given by
    exog_idx is added to the linear regression. It is a measure for
    multicollinearity of the design matrix, exog.

    One recommendation is that if VIF is greater than 5, then the explanatory
    variable given by exog_idx is highly collinear with the other explanatory
    variables, and the parameter estimates will have large standard errors
    because of this.

    Parameters
    ----------
    exog : ndarray
        design matrix with all explanatory variables, as for example used in
        regression
    exog_idx : int
        index of the exogenous variable in the columns of exog

    Returns
    -------
    vif : float
        variance inflation factor

    Notes
    -----
    This function does not save the auxiliary regression.

    See Also
    --------
    xxx : class for regression diagnostics  TODO: does not exist yet

    References
    ----------
    https://en.wikipedia.org/wiki/Variance_inflation_factor
    """
    k_vars = exog.shape[1]
    x_i = exog[:, exog_idx]
    mask = np.arange(k_vars) != exog_idx
    x_noti = exog[:, mask]
    r_squared_i = OLS(x_i, x_noti).fit().rsquared
    vif = 1. / (1. - r_squared_i)
    return vif



"""
"""              
        pd.DataFrame(['1','22','333'],columns=(['a','b','c']))
        
        forward_loop_record['serial'] = i
        forward_loop_record['varname', 'wald_stats', 'pvalue', 'df constraint'] = 
        
        DataFrame.append(self, other, ignore_index=False, verify_integrity=False, sort=False)
        
        wald_stats = res.wald_test_terms(skip_single=False).table.loc\
            [:,['statistic', 'pvalue', 'df_constraint']]        
        
        
        
        wald_stats = res.wald_test_terms(skip_single=False).table.loc\
            [varname,['statistic', 'pvalue', 'df_constraint']]
         
      wald_stats.to_frame().stack()
        wald_stats.reset_index(drop=True)
         wald_stats.to_frame().reset_index(drop=True)
        
        df = pd.DataFrame({'serial':[i], 'varname':[varname]})
        result = pd.concat([df, wald_stats],  axis=1, ignore_index=True, sort=False)
        
        res.wald_test_terms(skip_single=False).table.loc[varname, ['statistic', 'pvalue', 'df constraint']] 
        
        res.wald_test_terms(skip_single=False).table.loc[varname, 'statistic'] 
         res.wald_test_terms(skip_single=False).table.loc[varname, ['statistic', 'pvalue', 'df_constraint']].tolist()
       #
        
        
        
        
        
       
        print(res.wald_test_terms(skip_single=False))
        dir(res.wald_test_terms(skip_single=False))
        print(mod.score(res.params))

    pass
    return stepwise_record, varkeep, varleft


def backward_logit():
    pass
    if backward == 1:
        backward_logit()
    return wald_stat_df


    
    varlist = ['a','b','c']
    include = ['a']
    varlist_keep = list()
    if include != None:
        varlist_keep.append(include)
    varlist_left = list().clear()
    
    
    


'''
'''
def varlist_check(indata, target, weight, varlist):


def varlist_initial(indata, target, weight, varlist):
    
#1 getting started
import statsmodels
dir(statsmodels.base)
print(statsmodels.base._model_params_doc)
base._missing_param_doc


"""
问题2：
逻辑回归加权重解决方案：
1)好像logit函数本身不支持。
2)目前可用下列GLM大类函数计算，但是有警告
3）其他解决手段？ 1-用scipy 2-看看statsmodels的作者有什么手段？？
"""
import statsmodels.api as sm
import numpy as np
spector_data = sm.datasets.spector.load_pandas()
spector_data_df = spector_data.data
spector_data_df['wt'] = spector_data_df.apply(lambda x: np.random.randint(1,10), axis=1)
spector_data_df.to_csv("D:\\Analysis\\SEMMA_project\\spector_data.csv")
spector_data_df = sm.add_constant(spector_data_df)

spector_data.exog = sm.add_constant(spector_data.exog)
trainingdata_x = spector_data.exog
trainingdata_y = spector_data.endog

#下面这种写法，weight不起作用
res = sm.Logit(spector_data_df['GRADE'], \
               spector_data_df[['const', 'PSI']], \
                   freq_weights=spector_data_df['wt']).fit()
    
print(res.summary())
print(res.summary2())

#这种写法的结果与sas一致，但报警告：
#__main__:3: DeprecationWarning: Calling Family(..) with a link class as argument is deprecated.
# Use an instance of a link class instead.
logmodel=sm.GLM(spector_data_df['GRADE'], \
                spector_data_df[['const', 'PSI']], \
                    family=sm.families.Binomial(sm.families.links.logit),\
                        freq_weights=spector_data_df['wt']).fit()

print(logmodel.summary())
print(logmodel.summary2())


trainingdata_y = pd.DataFrame()
trainingdata_y['Successes'] = spector_data.endog.apply(lambda x: x*np.random.randint(1,10) \
                      if x == 1 else 0)
trainingdata_y['Failures'] = spector_data.endog.apply(lambda x: np.random.randint(1,10) \
                      if x == 0 else 0)
    
df['true_cum']=df['a'].map(lambda x: if_true(x)).cumsum()

import statsmodels.api as sm
logmodel=sm.GLM(trainingdata_y[['Successes', 'Failures']], \
                trainingdata_x, \
                    family=sm.families.Binomial(sm.families.links.logit)).fit()
print(logmodel.summary())
print(logmodel.summary2())

trainingdata_x['wt'] = trainingdata_x.apply(lambda x: np.random.randint(1,10), axis=1)

logmodel=sm.GLM(trainingdata_y, \
                trainingdata_x[['const', 'GPA', 'TUCE', 'PSI']], \
                    family=sm.families.Binomial(sm.families.links.logit),\
                        freq_weights=trainingdata_x['wt']).fit()

print(logmodel.summary())
print(logmodel.summary2())

trainingdata_x.wt.sum()





"""
问题3：corr的权重计算法，和筛选法：

1）需和sas比对，看是否存在样本计算修正的问题（/n-1）
2）如何查看核输出结果。

"""
from statsmodels.stats.weightstats import DescrStatsW
def corr_check(indata_x, corr_threshold=0.75, weights=None):
    mask = list(indata_x.columns)
    if 'const' in mask
        mask.remove('const')
    if weights in mask:
        mask.remove(weights)
    d1_wt = DescrStatsW(indata_x[mask], weights=indata_x[weights])
    d1_wt_corr = d1_wt.corrcoef #相关系数
    corr_check = 'pass'
    for i in range(d1_wt_corr.shape[0]):
        for j in range(d1_wt_corr.shape[0] - i):
            if i != j and d1_wt_corr[i,j] > corr_threshold:
                var_i = mask[i]
                var_j = mask[j]
                print('correlation of %s and %s is higher than %.2f !'%(var_i, var_j, corr_threshold))
                corr_check = 'fail'
    print(corr_check)    

# example 1
np.random.seed(0)
x1_2d = 1.0 + np.random.randn(20, 3)
w1 = np.random.randint(1, 4, 20)
d1 = DescrStatsW(x1_2d, weights=w1)
d1.mean
d1.var
d1.std_mean


# example 2
mask = list(data1.columns)
mask.remove('wt')
d1_wt = DescrStatsW(data1[mask], weights=data1['wt'])

d1_wt_corr = d1_wt.corrcoef #相关系数

d1_wt_corr[d1_wt_corr > 0.2]

corr_threshold = 0.2
corr_threshold = 0.3
corr_threshold = 0.5

corr_check = 0
for i in range(d1_wt_corr.shape[0]):
    for j in range(d1_wt_corr.shape[0] - i):
        if i != j and d1_wt_corr[i,j] > corr_threshold:
            var_i = mask[i]
            var_j = mask[j]
            print('correlation of %s and %s is higher than %.2f !'%(var_i, var_j, corr_threshold))
            corr_check = 1
print(corr_check)    
        
var1 = mask[0]
var2 = mask[1]
print('correlation of % and s% is higher than %.2f !'%(var1, var2, corr_threshold))

"""
"""

%matplotlib inline
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std









np.random.seed(9876789)
nsample = 100
x = np.linspace(0, 10, 100)
X1 = np.column_stack((x)).T
X2 = np.column_stack((x, x**2))
beta = np.array([1, 0.1, 10])
e = np.random.normal(size=nsample)
X1 = sm.add_constant(X1)
X2 = sm.add_constant(X2)
y = np.dot(X2, beta) + e
model1 = sm.OLS(y, X1)
model2 = sm.OLS(y, X2)
results1 = model1.fit()
results2 = model2.fit()
results1.compare_lm_test(results2)

print(results.summary())
dir(results)
dir(model)
sm.regression.linear_model.RegressionResults.compare_lm_test(model)
sm.regression.linear_model.RegressionResults.compare_f_test(results)

results.compare_lm_test(results)

X, y = load_iris(return_X_y=True)


print(sm.datasets.__doc__)

dir(sm.datasets)

data = sm.datasets.anes96.load_pandas()

df = sm.datasets.anes96.load_pandas().data

y, X = dmatrices('vote ~ logpopul + TVnews + selfLR + ClinLR + age + educ + income',\
                 data=df, return_type='dataframe')
    

mod = sm.Logit(y, X)
res = mod.fit()
print(res.summary())
print(res.summary2())
print(res.wald_test.__doc__)

dir(res)
dir(mod)
print(mod.score(res.params))
print(mod.score_obs(res.params))

df.to_csv("D:\\Analysis\\SEMMA_project\\anes96.csv")


y, X = dmatrices('vote ~ selfLR',\
                 data=df, return_type='dataframe')

y, X = dmatrices('vote ~ selfLR + ClinLR',\
             data=df, return_type='dataframe')
  
mod = sm.Logit(y, X)
res = mod.fit()
print(res.summary())
print(res.summary2())

r = np.zeros_like(res.params)
r[1:] = [1]

A = np.identity(len(res.params))
A = A[1:,:]
print(res.wald_test(A))
print(res.t_test(A))
#这个wald可用
print(res.wald_test_terms(skip_single=False))
print(mod.score(res.params))

dir(res.wald_test_terms(skip_single=False))
res.wald_test_terms(skip_single=False).col_names
res.wald_test_terms(skip_single=False).statistic
res.wald_test_terms(skip_single=False).summary_frame
res.wald_test_terms(skip_single=False).table[index='selfLR']['statistic']
res.wald_test_terms(skip_single=False).dist_args

res.wald_test_terms(skip_single=False).table.loc['ClinLR', 'statistic']

#lm统计量
1）假设共x1-xn个自变量，要检验其中x1-xq个。原假设为该q个变量系数均为0；
2）y与x1-xq回归得到约束方程，和残差u
3）u与x1-xn回归，得到R^2，并计算样本容量n
4)LM=n*R^2 ~ chi-square(q)分布，比较其于该分布临界值c的关系，大于则推翻原假设。

# 信息矩阵 = 负的hessian矩阵 logit的information未实现
U = mod.score(res.params)
info_matrix = -1 * mod.hessian(res.params)
I = np.linalg.inv(info_matrix)

Score1 = np.dot(U.T,I)
Score = np.dot(Score1,U)

U_VAR1 = mod.score(res.params)[2:]
stats.diagnostic.linear_lm(resid= , exog=[selfLR + ClinLR])

dir(mod.information.__doc__)
print(res.llr_pvalue)

print(res.llnull)
print(res.llf)

print(r)
dir(mod.score(res.params))
dir(res.wald_test(A))
print(res.wald_test(A).conf_int)

print(res.t_test(r))
print(res.wald_test(r))
print(res.llr)
res.compare_lm_test(res)

dir(sm.regression.linear_model.RegressionResults.compare_lm_test())
dir(sm.test)
dir(sm.stats)
dir(res.wald_test)

res.wald_test()
sm.webdoc()
sm.webdoc('glm')


RegressionResults.compare_lm_test(restricted, demean=True, use_lr=False)


RegressionResults.compare_lm_test(restricted, demean=True, use_lr=False)[source]

statsmodels.regression.linear_model.RegressionResults.compare_lm_test


import numpy
import statsmodels.api as sm

# Random data with two (identical) groups with ten members each
# and there are 1000 repetitions of this setup
data = numpy.random.random( (20, 1000) )
model  = sm.add_constant(numpy.array([0]*10 + [1]*10))
restricted_model = numpy.ones((20,1))

fit = sm.OLS(data, model).fit()
print(fit.summary())
restricted_fit = sm.OLS(data, restricted_model).fit()

# The following raises a ValueError exception
# but should instead have the same results as the method shown below
fs, ps, dfs = fit.compare_f_test(restricted_fit)

## The current way you have to run this, running one at a time:
fs, ps, dfs = numpy.empty(1000), numpy.empty(1000), numpy.empty(1000)
for i in range(1000):
  fit = sm.OLS(data[:,i], model).fit()
  restricted_fit = sm.OLS(data[:,i], restricted_model).fit()
  fs[i], ps[i], dfs[i] = fit.compare_f_test(restricted_fit)


statsmodels.stats.diagnostic.linear_lm
statsmodels.stats.diagnostic.linear_lm(resid, exog, func=None)

exog = np.array(X.columns)
exog = data.exog.T
dir(res.resid_dev)
print(res.resid_dev.__doc__)
lm, lm_pval, ftest = sm.stats.diagnostic.linear_lm(res.resid_dev, exog, func=None)

 if func is None:
        def func(x):
            return np.power(x, 2)
exog_aux = np.column_stack((exog, func(exog[:, 1:])))

import numpy as np
import statsmodels.api as sm
data = sm.datasets.longley.load(as_pandas=False)
data.exog = sm.add_constant(data.exog)
results = sm.OLS(data.endog, data.exog).fit()

# r_matrix{array_like, str, tuple}
# One of:
# array : An r x k array where r is the number of restrictions to test and k is the number of regressors. It is assumed that the linear combination is equal to zero.
A = np.identity(len(results.params))
A = A[1:,:]
print(results.f_test(A))
print(results.wald_test(A))
B = np.array(([0,0,1,-1,0,0,0],[0,0,0,0,0,1,-1]))
# This tests that the coefficient on the 2nd and 3rd regressors are equal and jointly that the coefficient on the 5th and 6th regressors are equal.
print(results.f_test(B))

Alternatively, you can specify the hypothesis tests using a string

from statsmodels.datasets import longley
from statsmodels.formula.api import ols
dta = longley.load_pandas().data
formula = 'TOTEMP ~ GNPDEFL + GNP + UNEMP + ARMED + POP + YEAR'
results = ols(formula, dta).fit()
hypotheses = '(GNPDEFL = GNP), (UNEMP = 2), (YEAR/1829 = 1)'
hypotheses = '(GNP = UNEMP), (POP = YEAR)'
f_test = results.f_test(hypotheses)
print(f_test)

w_test = results.wald_test(hypotheses)
print(w_test)


results.tvalues

hypotheses = '(GNPDEFL = 0),(GNP = 0),(UNEMP = 0),(ARMED = 0),(POP = 0),(YEAR = 0)'
hypotheses = np.zeros_like(results.params)
hypotheses = np.ones_like(results.params)
hypotheses = np.identity(len(results.params))
hypotheses = hypotheses[1:,:]
t_test = results.t_test(hypotheses)
print(t_test)

hypotheses = '(GNPDEFL = GNP = UNEMP = ARMED = POP = YEAR = 0)'
hypotheses = '(GNPDEFL = 0)'
w_test = results.wald_test(hypotheses)
print(w_test)

dir(results)

f_test = results.f_test(hypotheses)
print(f_test)



data = sm.datasets.longley.load(as_pandas=False)
data.exog = sm.add_constant(data.exog)

#results = sm.OLS(data.endog, data.exog[:, 0:2]).fit()
olsm = sm.OLS(data.endog, data.exog)
dir(results)
results = sm.OLS(data.endog, data.exog).fit()
U1 = sm.OLS(data.endog, data.exog).score(results.params)

r = np.zeros_like(results.params)
r[1:] = [1]
print(r)
T_test = results.t_test(r)
print(T_test)

lm, lm_pval, ftest = sm.stats.diagnostic.linear_lm(resid=results.resid , exog=data.exog)

