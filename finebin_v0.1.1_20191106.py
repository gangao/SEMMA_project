# -*- coding: utf-8 -*-
"""
@name：finebin
@author: cshuo
@version: 0.1.1
          fix:1)重新设计了format格式，即第1列为序号，第2列为分栏值，相应改写write_csv_format程序。
@describe：细分栏主程序及相关子程序
@date：20191106

    问题1：mssing值怎么弄？特殊值怎么分栏？ missing去除、特殊值先去除再加入
    问题2：分栏后，好坏有缺失怎么合并?包括特殊值
    问题3：最终数据集怎么使用format算woe？ 每个变量的format做成一个字典，format对应label，转换

"""
'''
# example
import scorecardpy as sc
indata = sc.germancredit()
indata_check(indata)
sepcial_value = [-1000,-1001,-1002,-1003]
save_path = 'D:\\Analysis\\SEMMA_project\\fmtlist.csv'
fine_bin(indata=indata, varlist=None, target=None, weight=None, method='quantile', \
         special_value=[-1000,-1001,-1002,-1003], bin_num=10, save_path=save_path)

'''


import numpy as np
import pandas as pd
import csv
import os

"""
"""
def read_csv_varlist(varlist_path):
    '''
    将某存储目录下的变量列表及变量类型的csv文件读为dict格式
    
    Params
    ------
    varlist_path: 目录字符串，如"D:\\varlist.csv"
            该csv有两列，分别为变量名和格式，无头。
    
    Returns
    ------   
    dict：key为变量名，value为格式
    
    '''
    try:
        df = pd.read_csv(varlist_path, header=None, prefix = 'col')
        sr = pd.Series(df['col1'].values, index = df['col0'])
        #dc = sr.to_dict()
    except FileNotFoundError as e:
        print("FileNotFoundError: %s" %e)
    return sr

"""
"""
def indata_check(indata):
    '''
    输入数据集清洗、改名主程序，在细分栏之前用
        注：此时的输入数据集已经不是源数据了，而是预测变量和表现变量集了，因此不应该有没有业务意义的缺失值
        而所谓的缺失值应已经是-999、'-999'之类的数据了，但为了谨慎起见，还是检查一下
    '''
    # 0 indata列名去空格转大写
    vlist_raw = [x for x in indata.columns.values]
    vlist_upper = [x.strip().upper() for x in indata.columns.values]    
    vlist_dict = dict(zip(vlist_raw, vlist_upper))
    indata.rename(columns=vlist_dict, inplace = True)
    # 1 清洗数据：去除缺失值(nan,null,'')，也就是没有业务意义的纯粹缺失值，并输出警告
    # 1.1 全是空格转为NaN & 空值''转为NaN
    indata.replace(to_replace=r'\s+', value=np.NaN, regex=True, inplace=True)
    indata.replace('', np.NaN, inplace=True)
    # 1.2 只要有缺失则去除行
    indata.dropna(inplace=True) # 除去缺失值和特殊值之外的数据
    return None

"""
细分栏主程序
"""
def fine_bin(indata, method, special_value, save_path, varlist=None, target=None, weight=None, bin_num=10):
    '''
    输入:数据集
    输出：细分栏格式
    
    Params
    ------
    indata:
        
    varlist: 默认dataframe格式，包括两列varname + vartype，其中varname有个大小写问题，在之前读为df时就应该解决
    
    target:
        
    weight:
        
    method: 
        quantile / tree / chimerge /quantile_smart
        
    vartype: 是一个varname与vartype对应的csv文件，读入后变为dataframe，
        varlist与之关联，取vartype，如缺失则自动置为numeric
        vartype的可用范围为：numeric / amount / ratio / month / category;
            
    special_value:
        
    bin_num:
        
    save_path:
    '''
    # 1 根据indata情况检查varlist情况，如varlist=None则自己制作一个
    varlist = varlist_check(indata, varlist, target, weight) # 包括变量是否在、去掉预测变量和weight、检查变量类型是否正确
        
    # 2 遍历varlist，进行变量细分栏
    for row in varlist.iterrows():
        varname = row[1]['varname']
        vartype = row[1]['vartype']
        # 判断method：
        if method.strip().upper() == 'QUANTILE':
        # 使用传统的分位数法进行变量的分栏
            fine_bin_quantile(indata, varname, vartype, weight, method, special_value, bin_num, save_path)
            
        elif method.strip().upper() == 'TREE':
            pass
        
        elif method.strip().upper() == 'CHIMERGE':
            pass
        
        elif method.strip().upper() == 'QUANTILE_SMART':
            pass
        
        else:
            pass
    
    return



"""
"""
def varlist_check(indata, varlist, target, weight):
    '''
    基于实际数据的varlist，检查输入的varlist（可为空），对varname和vartype进行改错，
    返回修改后的varlist
   
    Params
    ------
    indata:
    varlist: 默认dataframe格式，包括两列varname + vartype
    
    Returns
    ------
    DataFrame
    '''
    # 1 根据indata制作备用varlist
    varlist_bak = pd.DataFrame({'varname':indata.dtypes.index, 'vartype':indata.dtypes.values.astype('str')}) # 必须加astype不然变成categorytype
    if target != None: # 然后去除权重变量名
        try:
            varlist_bak.drop(varlist_bak[varlist_bak.varname == target].index, inplace=True) # 然后去除表现变量名
        except ValueError as e:
            print('target name error! error: %s'%e)
    if weight != None: # 然后去除权重变量名
        try:
            varlist_bak.drop(varlist_bak[varlist_bak.varname == weight].index, inplace=True)
        except ValueError as e:
            print('weight name error! error: %s'%e)
    # 2 改变varlist中vartype列
    varlist_bak['vartype'] = varlist_bak['vartype'].apply(lambda x: 'category'\
                               if str(x) in ['category', 'object'] else 'numeric')
    # 3 判断是否已设置了varlist输入
    if varlist == None: # 如果未定义varlist，则使用输入数据集的列做成的varlist
        varlist_out = varlist_bak     
    else: # 如果varlist有事先定义，那么还要和真实的数据集进行比较，包括变量和数据类型
        varlist_input = varlist
        # varlist去除weight和target
        if target != None: # 然后去除权重变量名
            try:
                varlist_input.drop(varlist_input[varlist_bak.varname == target].index) # 然后去除表现变量名
            except ValueError as e:
                print('target name error! error: %s'%e)
        if weight != None: # 然后去除权重变量名
            try:
                varlist_input.drop(varlist_input[varlist_bak.varname == weight].index)
            except ValueError as e:
                print('weight name error! error: %s'%e)        
        # varlist左连接varlist_bak
        varlist_input = varlist_input.merge(varlist_bak, how='left', on='varname', suffixes=('', '_1'))
        # 比较：1）varlist中未匹配到的变量(即实际数据varlist_bak中没有)：2
        #       2）变量定义类型与变量实际type不符合，定义为数值，结果为category ：3 （注：两category不是一个意思）
        conditions = [(varlist['vartype_1'].isnull()),
                      (varlist['vartype_1'] == 'category') & (varlist['vartype'] != 'category')]
        choices = [2, 3]
        varlist_input['check_result'] = np.select(conditions, choices, default=1)
        # 去除未匹配变量，并输出警告信息
        if len(varlist_input(varlist_input['check_result'] == 2)) >0:
            print('warning: s% variables of varlist not in indata!'\
                  %len(varlist_input[varlist_input['check_result'] == 2])
                  )
        varlist_input.drop(varlist_input[varlist_input['check_result'] == 2].index, inplace=True)
        # 改变不符合的变量类型，并输出警告信息
        if len(varlist_input(varlist_input['check_result'] == 3)) >0:
            print('warning: s% variables of varlist set wrong vartypes!changed!'\
                  %len(varlist_input[varlist_input['check_result'] == 3])
                  )
        varlist_input['vartype'].mask(varlist_input['check_result'] == 3, 'category', inplace=True)
        varlist_out = varlist_input.drop(['vartype_1', 'test'], axis=1)

    return varlist_out
        



"""
"""
def fine_bin_quantile(indata, varname, vartype, weight, method, special_value, bin_num, save_path):
    '''
    对一个变量进行分位数方法的细分栏
   
    Params
    ------
    indata:
    varlname：
    vartype：
    
    Returns
    ------
    None
    '''
    # 0 varname对应的vartype
    vartype = vartype.strip().upper() # 从varlist里面取，并已经去除空格和转为大写
    # 如果：变量实际的类型为数值型 & xtype 不为 category
    # 1.1 清洗数据：去除special_value；返回var+weight
    invar = clear_data(indata, varname, weight, special_value) # 除去缺失值和特殊值之外的数据
    # 1.2 找出原数据中的special_value
    special_value_list = spv_in_data(invar, special_value) # 要注意字符型特殊值后面是否有小数位
    # 2 判断：变量为数值型 或空
    if vartype != 'CATEGORY':
        # 2.1 取分位点
        finebin_fmt = binning_quantile(invar, varname, weight, bin_num)
        # 2.2 如果xtype并非numeric，则按对应类型优化分位点
        finebin_fmt_adj = binning_adjust(finebin_fmt, vartype)
        # 2.3 特殊值与分位点合并
        finebin_point = special_value_list + finebin_fmt_adj
    # 3 判断：变量实际的类型为字符型 | xtype 为 category    
    elif vartype == 'CATEGORY':
        # 3.1 直接取异同值作为分位点
        finebin_point = list(set(indata[varname].astype(str)))
        # 3.2 特殊值与分位点合并
        finebin_point = special_value_list + finebin_point
    # 4 输出成固定的format格式保存，格式为csv，在一张csv总表中后续写入
    write_csv_format(save_path, varname, finebin_point)


    
"""
"""
def clear_data(indata, varname, weight, special_value):
    '''
    清洗数据：去除special_value；
   
    Params
    ------
    invar:
    special_value: 
    
    Returns
    ------
    DataFrame
    '''
    if weight == None:
        invar = indata[varname].to_frame()
        invar['weight'] = 1
    else:
        invar = indata[[varname, weight]]
    spv_num = [x for x in special_value if type(x) != str]
    spv_str = [x for x in special_value if type(x) == str]
    if pd.api.types.is_numeric_dtype(invar):
        invar = invar[~invar.isin(spv_num)]
    else:
        invar = invar[~invar.isin(spv_str)]
    #vardata.replace(to_replace=r'\s+', value=np.NaN, regex=True).replace('', np.NaN)
    #vardata.dropna()
    return invar
    

"""
"""
def spv_in_data(invar, special_value):
    """
    根据特殊值列表找出该变量观测中实际存在的特征值。
    根据特征值是数值还是字符进行了分别判断。
    """
    spvlist_indata = []
    spv_num = [x for x in special_value if type(x) != str]
    spv_str = [x for x in special_value if type(x) == str]
    if pd.api.types.is_numeric_dtype(invar):
        spvlist = spv_num
    else:
        spvlist = spv_str
    for spv in spvlist:
        if spv in invar.values: # 
            spvlist_indata.append(spv)
    spvlist_indata.sort()
    return spvlist_indata



"""
"""
def binning_quantile(invar, varname, weight, bin_num):
    '''
    计算数值型变量的分位数点,
    
    Params
    ------
    invar: 输入数据集，dataframe格式，应做过筛选，只包含var和weight
        
    varname: str，列名
    
    weight: 权重，数值型，可为None
        
    bin_num: 最大分栏数，默认为10。即将全体分为几个人数相等的分栏，然后取对应的分位数点
    
    Returns
    ------
    list
    
    '''
    if weight == None: # 如果没有定义weight，则使用之前自己制作的weight字段
        weight = 'weight'
    invar = invar.sort_values(by=[varname]) # 1 按变量x排序
    invar['weight_cum'] = invar[weight].cumsum() # 2 weight累加
    invar['weight_cum_p'] = invar['weight_cum'] / invar['weight_cum'].iloc[-1] # 3 做weight累加的百分比
    q_list = list((x+1)/bin_num for x in range(0,bin_num)) # 4 根据分栏总数的设定，为quantile函数制作列表形式的百分位数的参数，
    invar_quantiles = invar['weight_cum_p'].quantile(q_list, interpolation='nearest').reset_index()
    # 5 用Series的quantile方法取分位数点，并且取分位数的插值方法设定为nearest。
    #   这样就一定与weight_cum_p取值有对应，然后就可以找到对应的x值作为真正的切分点。
    #   最后把索引做成列，即为后续保留了索引又变Series为df，导致可以使用merge
    invar_bins = invar_quantiles.merge(invar, how='left',
                                    left_on='weight_cum_p', 
                                    right_on='weight_cum_p') # 6 分位数点与原数据merge，找到对应的x值

    
    fmt_list = list(invar_bins[varname])
    return fmt_list # 是不是字典格式更好？



"""
"""
def binning_adjust(bin_format, vartype):
    '''
    根据变量类型进行对应的切分点调整
    还要加入：
    1）去除极值的影响
    2）包含str和category变量
    3）相同的切分点：说明该值很多，直接去重还是该值+1？+1的目的是让该变量作为单独的取值

    '''
    # 1 根据vartype进行切分点调整
    if vartype == 'MONTH':
        for i, bins in enumerate(bin_format):
            if bins > 120:
                bin_format[i] = round_x(bins, 24)
            elif bins > 72:
                bin_format[i] = round_x(bins, 12)
            elif bins > 36:
                bin_format[i] = round_x(bins, 6)
            elif bins > 3:
                bin_format[i] = round_x(bins, 3)
            else:
                bin_format[i] = round_x(bins, 1)
    elif vartype == 'RATIO':
        for i, bins in enumerate(bin_format):
            if abs(bins) > 5.0:
                bin_format[i] = round_x(bins, 1, 2)
            elif abs(bins) > 1.0:
                bin_format[i] = round_x(bins, 0.5, 2)
            elif abs(bins) > 0.5:
                bin_format[i] = round_x(bins, 0.1, 2)
            elif abs(bins) > 0.1:
                bin_format[i] = round_x(bins, 0.05, 2)
            else:
                bin_format[i] = round_x(bins, 0.01, 2)
    elif vartype == 'AMOUNT':
        for i, bins in enumerate(bin_format):
            if bins < -100000:
                bin_format[i] = round_x(bins, 10000)
            elif bins < -50000:
                bin_format[i] = round_x(bins, 5000)
            elif bins < -10000:
                bin_format[i] = round_x(bins, 1000)
            elif bins < -5000:
                bin_format[i] = round_x(bins, 500)
            elif bins < -1000:
                bin_format[i] = round_x(bins, 100)
            elif bins < -500:
                bin_format[i] = round_x(bins, 50)
            elif bins < -100:
                bin_format[i] = round_x(bins, 25)
            elif bins < -50:
                bin_format[i] = round_x(bins, 10)
            elif bins < 50:
                bin_format[i] = round_x(bins, 5)
            elif bins < 100:
                bin_format[i] = round_x(bins, 10)
            elif bins < 500:
                bin_format[i] = round_x(bins, 25)
            elif bins < 1000:
                bin_format[i] = round_x(bins, 50)
            elif bins < 5000:
                bin_format[i] = round_x(bins, 100)
            elif bins < 10000:
                bin_format[i] = round_x(bins, 500)
            elif bins < 50000:
                bin_format[i] = round_x(bins, 1000)
            elif bins < 100000:
                bin_format[i] = round_x(bins, 5000)
            else:
                bin_format[i] = round_x(bins, 10000)
    else:
        for i, bins in enumerate(bin_format):
            bin_format[i] = round_x(bins, 1)
        
    # 2 重复值调整：去除重复值，并加一个重复值+1
    fmt_dict = dict.fromkeys(set(bin_format))
    for keys in fmt_dict: # 统计重复数量
        fmt_dict[keys] = bin_format.count(keys)
        
    df = pd.DataFrame({'fmt':list(fmt_dict.keys()), 'cnt':list(fmt_dict.values())})
    df1 = df.copy()
    for index, row in df.iterrows(): # 循环df，大于1则增加一行
        if row['cnt'] > 1:
            df1 = df1.append({'fmt': row['fmt']+1, 'cnt': 99}, ignore_index=True)
    # 3 最后去重        
    bin_format_adj = sorted(list(set(df1['fmt'])))
    return bin_format_adj
    

"""
"""
def round_x(num, multi, d=None):
    numx = round(round(num/multi) * multi, d)
    return numx
    

"""
"""
def write_csv_format(path, vname, fmtlist):
    '''
    write a variable's format into a csv file
    
    Params
    ------
    path: csv file path, include the csv file name
    vname: variable name
    fmtlist: a list contain binning points
    
    Returns
    ------
    None
    '''
    if not os.access(path, os.F_OK): # os.F_OK: 检查文件是否存在;
        # if not exist, create csv file
        with open(str(path), 'w', newline='', encoding='utf-8') as csvfile: # 'w' open for writing, truncating the file first
            writer = csv.writer(csvfile)
            writer.writerow(['$'+vname]) # write variable name
            for i, rows in enumerate(fmtlist): # write variable name
                writer.writerow([i+1, str(rows)]) # -inf写入csv,打开会显示错误，但可以读取
            writer.writerow([''])    
    else:
        with open(str(path), 'a', newline='', encoding='utf-8') as csvfile: # 'a' open for writing, appending to the end of the file if it exists
            writer = csv.writer(csvfile)
            writer.writerow(['$'+vname]) # write variable name
            for i, rows in enumerate(fmtlist): # write variable name
                writer.writerow([i+1, str(rows)]) # -inf写入csv,打开会显示错误，但可以读取
            writer.writerow([''])
            
            
