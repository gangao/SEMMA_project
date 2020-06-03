# -*- coding: utf-8 -*-
"""
@name：woe_report
@author: cshuo
@version: 0.1.2
          fix(0.1.1):重新设计了format格式，即第1列为序号，第2列为分栏值，相应改写write_csv_format程序。
                如果要将特殊值和分栏值合并，或是任意多个分栏值合并，可以讲format的csv表中序号写成同一序号。
          fix(0.1.2):每个变量把转换woe时候的bins和label进行pickle化，用于convert woe时候读取使用。
@describe：细分栏主程序及相关子程序
@date：20191208
@to_do_list:
    1）woe_report程序输出ivlist
    2）woe零值智能判断和合并
    3）woe最后空值点去掉
    4）convert_woe
    5）step_disc
    6）logistic单步/corr/vif
    7）scorecard：直接生成code
    8）ks/roc/divergence/排序/画图


# example
import scorecardpy as sc
target = 'CREDITABILITY'
indata = sc.germancredit()
indata_check(indata)
indata[target] = indata[target].map({'good': 0, 'bad': 1})
fmt_file = 'D:\\Analysis\\SEMMA_project\\fmtlist.csv'
save_path = 'D:\\Analysis\\SEMMA_project\\woe_save\\'
splist1 = [-1001, -1002, 4, 5]
woe_report(indata, fmt_file, target, save_path, splist=splist1, varname=None, varlist=None, weight=None)  
"""


import re
import os
import csv
import pickle
import numpy as np
import pandas as pd

"""
"""
# 主程序 woe
def woe_report(indata, fmt_file, target, save_path, splist, varname=None, varlist=None, weight=None):
    '''
    woe main function
    woe存为pickle格式，同时存储在一张表里面
    
    Params
    ------
    indata:
    fmt_file: 
    target:
    save_path:
    varlist        
    weight:
        
    Returns
    ------
    DataFrame
    '''
    # 0 计算好人、坏人总数
    global total_good_raw 
    global total_bad_raw 
    global total_good_wt 
    global total_bad_wt 
    global total_num_wt
    if weight == None:
        indata['weight'] = 1
        weight = 'weight'
    total_good_raw = len(indata[indata[target]==0])
    total_bad_raw = len(indata[indata[target]==1])
    total_good_wt = indata.apply(lambda x: x[weight] if x[target]==0 else 0, axis=1).sum()
    total_bad_wt = indata.apply(lambda x: x[weight] if x[target]==1 else 0, axis=1).sum()
    total_num_wt = total_good_wt + total_bad_wt
    excel_path = save_path + "woe_report.xlsx"
    #print(indata)
    # 1 如果var==None, 则根据varlist进行循环，或根据indata里面的变量名单(去除target和weight)进行循环
        # 2 读取变量format，如未读取到，则计入警告表，并跳至下一个循环
        # 3 根据变量format生成labels
        # 4 取变量var，权重weight，表现target，将var转为label
        # 5 计算woe前的一些基本统计量，如加权/未加权总数、good数、bad数
        # 6 计算woe的子程序
    # 2 如果不为None, 则单个计算
    if varname == None: # 计算所有变量woe
        varlist = varlist_check(indata, varlist, target, weight)
        startrow = 1
        writer = pd.ExcelWriter(excel_path)
        for row in varlist.iterrows():
            varname = row[1]['varname']
            vartype = row[1]['vartype']
            try:
                format_dict = read_csv_format(fmt_file, varname, vartype)
                with open(save_path+varname+"_fmt.pkl", mode='wb') as pklfile:
                    pickle.dump(format_dict, pklfile)
            except:
                print("%s format not found" %(varname))
            #print(format_dict)
            woe = woe_report_var(indata, varname, vartype, format_dict, target, save_path, splist, weight=weight)
            woe.to_excel(writer, startcol = 1, startrow = startrow) # woe 放入excel
            startrow = startrow + len(woe) + 5
        writer.save() # save woe excel 
    else: # 计算单个变量woe
        try:
            format_dict = read_csv_format(fmt_file, varname)
        except:
            print("%s format not found" %(varname))
        #！！！！vartype此处未定义
        woe = woe_report_var(indata, varname, vartype, format_dict, target, save_path, splist, weight=weight)
        with pd.ExcelWriter(excel_path) as writer: # woe 放入excel
            woe.to_excel(writer, startcol = 1, startrow = startrow)        
    
        
            
"""
"""

# 1 search for particular variable by vname and return it's format
def read_csv_format(path, vname, vtype):
    '''
    read a variable's format from a csv file
    变量名格式
    变量格式为两列：序号+值
    读成字典格式: 值+序号
    
    Params
    ------
    path: csv file path, include the csv file name
    vname: variable name
    vtype:
        
    Returns
    ------
    dict：sorted, float type binning points
    '''
    flist = []
    nlist = []
    if os.access(path, os.F_OK):
        with open(path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            vname='$'+vname
            for row in reader:
                m = row[0].find(vname)
                if m != -1:
                    for row in reader:
                        n = re.match('^(?![\s\S])', str(row[0]).strip(), re.I) #判断是否为空行，不能放在外循环，要在内循环重新定义
                        if n: # or: if len(row[0]) == 0:
                            break
                        else:
                            flist.append(row[1])
                            nlist.append(row[0])
                else:
                    continue
        if vtype.lower() != 'category':
            print('number!')
            flist = list(map(float, flist))
            fmt_dict = dict(zip(flist,nlist))
        else:
            print('str!')
            fmt_dict = dict(zip(flist,nlist))
        return fmt_dict
    else:
        print('file %s not exist'%(path))
        return None
    


"""
"""
def woe_report_var(indata, varname, vartype, format_dict, target, save_path, splist, weight=None):
    '''
    caculate woe per var
    
    Params
    ------

    weight:
        
    Returns
    ------
    DataFrame
    '''    
    # 需求1、woe的保存应该是一个永久性的，是不是pickle化？而不仅仅是内存中的df
    # 需求2、woe计算中会有好坏缺失的情况，必须解决
    # 需求3、woe结果输出在一张大的excel表中，最好格式美观
    label_dict = bins_to_labels(format_dict, vartype, splist=splist)
    with open(save_path+varname+"_dict1.pkl", mode='wb') as pklfile:
        pickle.dump(label_dict, pklfile)
    label_map_dict = final_label(format_dict, label_dict, vartype)
    with open(save_path+varname+"_dict2.pkl", mode='wb') as pklfile:
        pickle.dump(label_map_dict, pklfile)
    rawdata = indata[[varname, weight, target]]
    if vartype.lower() != 'category':
        bins = list(format_dict.keys())
        bins = [float('-inf')] + bins + [float('inf')]
        rawdata[varname] = pd.cut(rawdata[varname], bins, \
                                   right=True, labels=list(label_dict.values()))
        rawdata[varname] = rawdata[varname].map(label_map_dict) # 把合并后的值的label转换为合并后的
    else: # 变量为字符或类别型，直接转换
        rawdata[varname] = rawdata[varname].map(label_dict).map(label_map_dict)
    # label列改名为变量名
    pass
    # 分组后算加权/未加权好坏
    # 其中要加入为0的特殊情况
    woe = rawdata.groupby(varname).apply(cacl_gb_num, target=target, weight=weight)
    woe['good_pct_wt'] = woe['good_num_wt'] / total_good_wt
    woe['bad_pct_wt'] = woe['bad_num_wt'] / total_bad_wt
    woe['woe_gb'] = woe.apply(lambda x: np.log(x['bad_pct_wt'] / x['good_pct_wt']\
                               if x['bad_pct_wt'] != 0 and x['good_pct_wt'] != 0 else np.nan), axis=1)
    woe['iv_value'] = woe.apply(lambda x: (x['bad_pct_wt'] - x['good_pct_wt'])*np.log(x['bad_pct_wt'] / x['good_pct_wt']\
                               if x['bad_pct_wt'] != 0 and x['good_pct_wt'] != 0 else np.nan), axis=1)
    woe['total_gb_pct'] = (woe['good_num_wt'] + woe['bad_num_wt']) / total_num_wt
    woe['bad_rate'] = woe['bad_num_wt'] / (woe['good_num_wt'] + woe['bad_num_wt'])
    # 生成最后一行
    woe = woe.reset_index()
    print('WOE1:')
    print(woe)
    dict1 = {varname:'TOTAL'}
    dict2 = woe[woe.columns.difference([varname])].sum(axis=0).to_dict() # 除去label栏其他全部加总再转为字典格式
    dict1.update(dict2)
    print(dict1)
    woe = woe.append(dict1, ignore_index=True)
    # woe pickcle化
    pickcle_path = save_path + varname + ".pkl"
    woe.to_pickle(pickcle_path)
    print('WOE2:')
    print(woe)
    # 返回woe df作为输出用
    return woe


"""
"""
def cacl_gb_num(x, target, weight):
    """基于dataframe做多列的汇总计算函数，并生成多个列"""
    d = {}
    d['good_num_raw'] = abs(x[target]-1).sum()
    d['bad_num_raw'] = x[target].sum()
    d['good_num_wt'] = (x[weight]*abs(x[target]-1)).sum()
    d['bad_num_wt'] = (x[weight]*x[target]).sum()
    return pd.Series(d, index=['good_num_raw', 'bad_num_raw', 'good_num_wt', 'bad_num_wt'])


"""
"""
def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z


"""
"""
# 2 bins to labels
def bins_to_labels(format_dict, vartype, splist = None):
    '''
    create labels from a format
    
    Params
    ------
    format_dict: dict, read from a csv format file previous
    vartype: Category or not Category
    splist: user self-defined, can be None, 
                        suggest to be a complete set of all special values
    
    Returns
    ------
    dict: str format labels    
    '''
    if vartype.lower() != 'category': # 数值型
        flist = list(map(float, format_dict.keys())) # format_list to float
        splist2 = []
        if splist != None:
            splist = sorted(list(map(float, splist))) # special_value_list to float
            # 把formatlist中包含的specialvalues单独拿出来，做出splist2
            for e in splist:
                try:
                    splist2.append(flist[flist.index(e)])
                    flist.remove(e)
                except:
                    continue
            # 把splist2 做成label:label_list1
            label_list1 = ['sp_value: {}'.format(splist2[i]) for i in range(len(splist2))]
            label_dict1 = dict(zip(splist2,label_list1))
        #把去除特殊值后的format_list加上无穷大、无穷小值，然后做成label_list2
        flist = [float('-inf')] + flist + [float('inf')]
        label_list2 = ['{} -< {}'.format(flist[i], flist[i+1]) for i in range(len(flist)-1)]
        label_dict2 = dict(zip(flist,label_list2))
        #label_list1和2合并
        if 'label_dict1' in locals().keys():
            label_dict = merge_two_dicts(label_dict1, label_dict2)
        else:
            label_dict = label_dict2
            
    else: # 字符型
        flist = list(format_dict.keys())
        splist2 = []
        if splist != None:
            splist = sorted(splist)
            for e in splist:
                try:
                    splist2.append(flist[flist.index(e)])
                    flist.remove(e)
                except:
                    continue
            # 把splist2 做成label:label_list1
            label_list1 = ['sp_value: {}'.format(splist2[i]) for i in range(len(splist2))]
            label_dict1 = dict(zip(splist2,label_list1))
        #把去除特殊值后的format_list做成label_list2
        label_list2 = ['{}'.format(flist[i]) for i in range(len(flist))]
        label_dict2 = dict(zip(flist,label_list2))
        #label_list1和2合并
        if 'label_dict1' in locals().keys():
            label_dict = merge_two_dicts(label_dict1, label_dict2)
        else:
            label_dict = label_dict2
            
    return label_dict



"""
"""
def final_label(format_dict, label_dict, vtype):
    '''
    实际上是根据format里面的序号，将两个序号相同的分栏，最终对应到一个分栏
    创建一个字典，是未按序号合并前的label和合并后label的对应
    用于将切分完（数值型）或转换完（类别型）并打上label的数据再次转为分栏合并后的数据
    
    Params
    ------
    format_dict: 
    label_dict: 
    vtype: 
    
    Returns
    ------
    dict: 
    '''    
    f_dict = format_dict.copy()
    if vtype.lower() != 'category':
        f_dict.update({float('-inf'):'0'})
    # 1 new_dict1为label和序号的对应
    new_dict1 = {}
    for k in label_dict.keys():
        new_dict1.update({label_dict[k]:f_dict[k]})
    # 2 new_dict2为去重后序号和按相同序号合并后label的对应
    new_dict2 = {}    
    for n in set(list(new_dict1.values())):
        v_list = []
        for v in new_dict1.keys():
            if new_dict1[v] == n:
                v_list.append(v)
        v_str = ', '.join(v_list)
        new_dict2.update({n:v_str})
    # 3 new_dict3为label和合并后label的对应
    l3 = []
    for k1 in new_dict1.keys():
        for k2 in new_dict2.keys():
            if new_dict1[k1] == k2:
                l3.append(new_dict2[k2])
    new_dict3 = dict(zip(new_dict1.keys(),l3))
    return new_dict3

