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


to do:
    1) sign_check功能
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

to do:
    score test:可能是residual chi-square的比值

'''

import statsmodels.api as sm
import numpy as np
import pandas as pd

from statsmodels.api import OLS
from statsmodels.api import WLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.weightstats import DescrStatsW


def logistic_stepwise(indata, target, weight=None, varlist=None, maxstep=None, \
                      include=None, exclude=None, sle=0.15, sls=0.15, corr_check=0.75, vif_check=10, sign_check='minus'):
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
    corr_check:
    vif_check:
    sign_check: 变量系数的业务方向判断，默认预测的是坏的概率，则系数按woe业务含义来全为负值
    
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
    5 注意事项：
    5.1 indata中要加入const列
    '''
    #1 主要变量预处理
    indata = indata_var_check(indata, target, weight) #检查输入数据集中相关字段是否全乎，如没有const加上const
    varlist = varlist_check(indata, target, weight, varlist) # 检查或根据数据生成变量列表
    varkeep, varleft = varlist_initial(varlist, include, exclude) #根据变量列表、包含和排除变量，生成最初的已保存变量列表和剩余变量列表
    if maxstep == None:
        maxstep = len(varlist) * 2
    for step in range(0, maxstep):
        # 循环退出条件1： 没有剩余变量了
        if len(varleft) == 0: 
            print('No Variables left!')
            break
        #1 制作一个stepwise_record空表
        stepwise_record = pd.DataFrame(columns=['step', 'operation', 'varname', 'corr_check', 'vif_check'])
        #2 先选入模变量
        varnum_bf = len(varkeep)
        print('STEP %i : forward_logit start'%(step))
        frecord, varkeep, varleft = forward_logit(step, indata, target, weight, varkeep, varleft, sle, \
                                                          vif_check, corr_check)
        print('varkeep: ')
        print(varkeep)
        print('varleft: ')
        print(varleft)
        varnum_aft = len(varkeep)
        if varnum_bf == varnum_aft:
            print('No variables can be entered, Break!')
            break
        # 循环退出条件2： 这次循环没有变量进入，则跳出
        print('STEP %i : stepwise_record.append start'%(step))
        stepwise_record.append({'step':step, 'operation':'FORWARD', 'varname':frecord[0], 'corr_check':frecord[1], 'vif_check':frecord[2]}, ignore_index = True)
        #2 再查看可剔除变量
        print('STEP %i : back_logit start'%(step))
        brecord, varkeep = backward_logit(step, indata, target, weight, varkeep, sls)
        print('varkeep: ')
        print(varkeep)
        print('varleft: ')
        print(varleft)
        stepwise_record.append({'step':step, 'operation':'BACKWARD', 'varname':brecord[0], 'corr_check':'', 'vif_check':''}, ignore_index = True)
    if weight != None:
        final_model = sm.GLM(indata[target], \
                     indata[varkeep], \
                         family=sm.families.Binomial(sm.families.links.logit()), \
                             freq_weights=indata[weight])
    else:
        final_model = sm.Logit(indata[target], indata[varkeep]) 
    final_model_fit = final_model.fit()
    print(final_model_fit.summary())
    return final_model_fit


"""
""" 
def indata_var_check(indata, target, weight=None):
    #检查输入数据集中相关字段是否完整，如没有const加上const
    print('indata_var_check start')
    varlist_from_data = list(indata.columns) #从indata的列中取变量名
    if target not in varlist_from_data:
        raise Exception('Error: target var is not in %s'%(str(indata))) #报错并跳出
    if weight != None and weight not in varlist_from_data:
        raise Exception('Error: weight var is not in %s'%(str(indata))) #报错并跳出
    if 'const' not in varlist_from_data: # 如没有const，增加一个
        indata = sm.add_constant(indata)
    print('indata_var_check end')
    return indata


"""
""" 
def varlist_check(indata, target, weight, varlist):
    print('varlist_check start')
    if varlist == None:
        varlist_from_data = list(indata.columns) #从indata的列中取变量名
        varlist_from_data.remove(target) #去除y名称
        varlist_from_data.remove('const') #去除const
        if weight != None:
            varlist_from_data.remove(weight) #去除weight名称（如有）
        print('varlist_check end')
        return varlist_from_data
    else:
        if 'const' in varlist: # 去除const（如有）
            varlist.remove('const') #去除const
        if target in varlist: # 去除target
            varlist.remove(target) #去除const
        if weight in varlist: # 去除weight
            varlist.remove(weight) #去除const
        print('varlist_check end')
        return varlist

        
"""
""" 
def varlist_initial(varlist, include, exclude):
    print('varlist_initial start')
    if include != None and exclude != None: #如包含和排除变量均有定义，则先判断是否有冲突
        for var_include in include:
            for var_exclude in exclude:
                if var_include == var_exclude:
                    raise Exception('Error: var %s in both include and exclude'%(var_include)) #报错并跳出
    if include == None:
        varkeep = []
        varleft = varlist
    else:
        varkeep = include
        varleft = [val for var in varlist if val not in varkeep]
    if exclude != None:
        varleft = [val for var in varleft if var not in exclude]
    print('varlist_initial end')
    return varkeep, varleft


"""
"""
def forward_logit(step, indata, target, weight, varkeep, varleft, sle, vif_check, corr_check):
    """
    逻辑：
    1 以varkeep为基础，循环检查varleft里面的单变量进行逻辑回归，并记录每次循环的结果进入一张表，列内容有：
    1.1）输出该单变量的score test统计量
    1.2）vif检查结果：如有设置，则加入单个变量回归时，会检查该模型所有变量的vif情况，如不满足vif阈值条件，则输出'fail'
    1.3）corr检查结果：如有设置，则加入单个变量回归时，会检查该模型所有变量之间的相关系数（实际只受该变量与其他变量的影响），
        如不满足corr阈值条件，则输出'fail' ———— corr这步应该可以放到开头一起做？？效率更高？
    1.4）逻辑回归不断加入变量后，一定可以收敛吗？
    2 所有循环完了之后，剔除vif和corr不满足条件的!!（如有设置），按score统计量从大到小排序，然后取满足条件的最大一个。如均不满足条件，说明无变量可入模。   
    3 最后将该程序执行中的所有操作流水记录打印出来
    """
    #1 制作一个forward_loop_record空表
    forward_loop_record = pd.DataFrame(columns=['varno', 'varname', 'wald_stats', 'pvalue', 'df_constraint', 'vif_check', 'corr_check'])
    #2 开始循环
    for i, varname in enumerate(varleft):
        #2.1 已入模变量（varkeep）与新变量varname组成新的x变量列表
        x_var = ['const'] + [varname] + varkeep
        #2.2 用新的x变量列表进行回归
        if weight != None:
            mod = sm.GLM(indata[target], indata[x_var], \
                         family=sm.families.Binomial(sm.families.links.logit()), \
                             freq_weights=indata[weight])
        else:
            mod = sm.Logit(indata[target], indata[x_var]) 
        res = mod.fit()
        print('forward - res.summary()')
        print(res.summary())
        #2.3 当前变量集X_VAR中所有变量的单变量VIF，记录该变量集中是否有大于vif阈值的变量
        vif_result = 'None' #未检验的默认值
        if vif_check != None and len(x_var) >2: # 判断当前多于一个变量
            vif_table = pd.DataFrame(columns = ['varno', 'varname' , 'vif'])
            reg_data = indata[x_var] #需包含常量
            col = list(range(reg_data.shape[1]))
            if weight != None:
                obs_wt = indata[weight]
            else:
                obs_wt = None
            for i in range(len(x_var)- 1): #要对该模型中的所有变量进行判断
                vif = _compute_vif(reg_data.iloc[:,col].values, i+1, weights=obs_wt)
                vif_table = vif_table.append({'varno':i+1, 'varname':x_var[i+1], 'vif':vif}, ignore_index = True)
            print(vif_table)
            vif_threshold = vif_check
            if len(vif_table[vif_table.vif >= vif_threshold]) > 0:
                vif_result = 'fail' # 将vif信息加入forward_loop_record作为排除条件
                print('VIF > ',vif_threshold)
                print(vif_table[vif_table.vif >= vif_threshold])
            else:
                vif_result = 'pass'
        #2.4 当前变量集X_VAR中所有变量的CORR矩阵，记录该变量集中是否有大于corr阈值的变量
        corr_result = 'None' #未检验的默认值
        if corr_check != None and len(x_var) > 2: # 判断当前多于一个变量
           corr_result = corr_check_func(indata, x_var, corr_threshold=corr_check, weight=weight)
        #2.5 新变量varname的单变量wald
        print('res.wald_test_terms(skip_single=False).table :')
        print(res.wald_test_terms(skip_single=False).table)
        statistic = res.wald_test_terms(skip_single=False).table.loc[varname, 'statistic']#从回归结果中取出特定变量的wald统计量相关结果
        pvalue = res.wald_test_terms(skip_single=False).table.loc[varname, 'pvalue']#从回归结果中取出特定变量的wald统计量相关结果
        df_constraint = res.wald_test_terms(skip_single=False).table.loc[varname, 'df_constraint']#从回归结果中取出特定变量的wald统计量相关结果
        #插入forward_loop_record表
        forward_loop_record = forward_loop_record.append({'varno': i, 'varname': varname, 'wald_stats': statistic, \
                                                          'pvalue': pvalue, 'df_constraint': df_constraint, \
                                                          'vif_check': vif_result, 'corr_check': corr_result}, ignore_index = True)
        #2.6  整体变量wald--暂不输出
        # A = np.identity(len(res.params))
        # A = A[1:,:]
        # print('Wald test for all vars: ', res.wald_test(A))
    #3  跳出循环后，按wald统计量从大到小排序，选最大的进行检验，然后选取变量或不选取
    forward_loop_record.sort_values(by=['wald_stats'], ascending=False, inplace=True) #wald_stats从大到小
    print('forward_loop_record : ')
    print(forward_loop_record[:10])
    if vif_check != None and corr_check != None:
        fl_rec_filter = forward_loop_record[(forward_loop_record['vif_check'] != 'fail') & (forward_loop_record['corr_check'] != 'fail')]
    elif vif_check != None:
        fl_rec_filter = forward_loop_record[(forward_loop_record['vif_check'] != 'fail')]
    elif corr_check != None:
        fl_rec_filter = forward_loop_record[(forward_loop_record['corr_check'] != 'fail')]
    else:
        fl_rec_filter = forward_loop_record
    frecord = []
    if fl_rec_filter.iloc[0].loc['pvalue'] <= sle: #判断
        var_enter = fl_rec_filter.iloc[0].loc['varname']
        print('var_enter: ')
        print(var_enter)
        frecord.append(var_enter)
        frecord.append(fl_rec_filter.iloc[0].loc['vif_check'])
        frecord.append(fl_rec_filter.iloc[0].loc['corr_check'])
        varkeep = varkeep + [var_enter]
        varleft.remove(var_enter)
        return frecord, varkeep, varleft
    else:
        frecord = ['', '', '']
        print('forward, no variable entered!')
        return frecord, varkeep, varleft


"""
"""        
def backward_logit(step, indata, target, weight, varkeep, sls):
    """
    逻辑：
    1 以varkeep为基础，循环检查varleft里面的单变量进行逻辑回归，并记录每次循环的结果进入一张表，列内容有：
    1.1）输出该单变量的score test统计量
    1.2）vif检查结果：如有设置，则加入单个变量回归时，会检查该模型所有变量的vif情况，如不满足vif阈值条件，则输出'fail'
    1.3）corr检查结果：如有设置，则加入单个变量回归时，会检查该模型所有变量之间的相关系数（实际只受该变量与其他变量的影响），
        如不满足corr阈值条件，则输出'fail' ———— corr这步应该可以放到开头一起做？？效率更高？
    1.4）逻辑回归不断加入变量后，一定可以收敛吗？
    2 所有循环完了之后，剔除vif和corr不满足条件的（如有设置），按score统计量从大到小排序，然后取满足条件的最大一个。如均不满足条件，说明无变量可入模。   
    3 最后将该程序执行中的所有操作流水记录打印出来
    """
    x_var = ['const'] + varkeep
    #1 用已入模的变量列表进行回归，只一次
    if weight != None:
        mod = sm.GLM(indata[target], \
                     indata[x_var], \
                         family=sm.families.Binomial(sm.families.links.logit()), \
                             freq_weights=indata[weight])
    else:
        mod = sm.Logit(indata[target], indata[x_var]) 
    res = mod.fit()
    print('back - res.summary()')
    print(res.summary())
    #2 制作backward_loop_record表
    df = res.wald_test_terms(skip_single=False).table
    print(df)
    backward_loop_record = df[~(df.index == 'Intercept')].reset_index().reset_index()
    backward_loop_record = backward_loop_record.rename(columns={'level_0': 'varno', 'index': 'varname', 'statistic': 'wald_stats'})
    #3 查看所有已入模变量的wald统计量，从小到达排序
    backward_loop_record.sort_values(by=['wald_stats'], ascending=True, inplace=True) #wald_stats从小到大
    brecord = []
    if backward_loop_record.iloc[0].loc['pvalue'] >= sls: #判断
        print('backward check fail, last var out')
        #4 从已保存变量列表中删除上一轮进入的var，而非统计量不满足条件的var
        last_enter_var = varkeep[len(varkeep)-1]
        brecord.append(last_enter_var)
        varkeep.remove(last_enter_var)
        return brecord, varkeep
    else:
        print('backward check pass, last var keeped')
        brecord.append('')
        return brecord, varkeep

    
"""
VIF -- WEIGHT，用_compute_vif即可，权重单独列输入 -- 来源网上
与sas进行对比分析过，有以下结论：
1）某变量的vif是用该变量与剩余变量进行直接线性回归得到的，有截距项
2）vif是用1/（1-R方）计算的，不是用调整后的R方
3）加了权重后的算法就是普通的加权重的线性回归

""" 
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


"""
函数：corr的权重计算法，和筛选法：
to do:
1）需和sas比对，看是否存在样本计算修正的问题（/n-1）
2）如何查看核输出结果。
"""
def corr_check_func(indata, x_var, corr_threshold=0.75, weight=None):
    mask = x_var
    if 'const' in mask:
        mask.remove('const')
    if weight == None:
        d1_wt = DescrStatsW(indata[mask])
    else:
        if weight in mask:
            mask.remove(weight)
        d1_wt = DescrStatsW(indata[mask], weights=indata[weight])
    d1_wt_corr = d1_wt.corrcoef #相关系数
    print(d1_wt_corr)
    corr_check_result = 'pass'
    for i in range(d1_wt_corr.shape[0]):
        for j in range(d1_wt_corr.shape[0] - i):
            if i != j and d1_wt_corr[i,j] >= corr_threshold:
                var_i = mask[i]
                var_j = mask[j]
                print('correlation of %s and %s is EG than %.2f !'%(var_i, var_j, corr_threshold))
                corr_check_result = 'fail'
    print(corr_check_result)
    return corr_check_result


"""
小样本测试

导入数据
import statsmodels.api as sm
import numpy as np
spector_data = sm.datasets.spector.load_pandas()
spector_data_df = spector_data.data
spector_data_df['wt'] = spector_data_df.apply(lambda x: np.random.randint(1,10), axis=1)
spector_data_df = sm.add_constant(spector_data_df)
"""

logit_model_fit = logistic_stepwise(spector_data_df, 'GRADE', weight='wt')
   