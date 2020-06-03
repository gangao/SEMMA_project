# -*- coding: utf-8 -*-
"""
@name：convert_woe
@author: cshuo
@version: 0.1.2
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
indata = sc.germancredit()
indata_check(indata)
indata[target] = indata[target].map({'good': 0, 'bad': 1})

fmtfile = 'D:\\Analysis\\SEMMA_project\\fmtlist.csv'
out_path = 'D:\\Analysis\\SEMMA_project\\'
woe_path = 'D:\\Analysis\\SEMMA_project\\woe_save\\'
target = 'CREDITABILITY'
weight = 'weight'
outdata = 'D:\\Analysis\\SEMMA_project\\woedata'
woedata = convert_woe(indata, outdata, fmtfile, woe_path, target=target, weight=weight, varname=None, vartype=None, varlist=None)
"""

def convert_woe(indata, outdata, fmt_file, woe_path, target=None, weight=None, varname=None, vartype=None, varlist=None):
    '''
    将变量按woe报告转换为对应的woe值
    1、从原始数据进行woe转换，需使用之前woereport中pickle化的3个字典（与woe均位于一个文件夹woe_path下，一个变量4个pkl）
    2、不改变indata，输出woe转换后的数据集
    3、转换后的woe数据集pickle化到outdata
    4、几个问题未解决，1）单个变量转换woe？是不是先从woe的pkl中读出后再转？2）weight需不需要放进来？
    
    '''
    woedata = indata.copy()
    if varname == None: # 转换所有varlist里面的woe
        print('loop == yes')
        varlist = varlist_check(woedata, varlist, target, weight)
        for row in varlist.iterrows():
            print('looping!')
            varname = row[1]['varname']
            vartype = row[1]['vartype']
            #1 把当前变量转换为format格式
            with open(save_path+varname+"_fmt.pkl", mode='rb') as pklfile:
                format_dict = pickle.load(pklfile)
            with open(save_path+varname+"_dict1.pkl", mode='rb') as pklfile:
                label_dict = pickle.load(pklfile)
            with open(save_path+varname+"_dict2.pkl", mode='rb') as pklfile:
                label_map_dict = pickle.load(pklfile)
            if vartype.lower() != 'category':
                bins = list(format_dict.keys())
                bins = [float('-inf')] + bins + [float('inf')]
                woedata[varname] = pd.cut(woedata[varname], bins, \
                                           right=True, labels=list(label_dict.values()))
                woedata[varname].map(label_map_dict) # 把合并后的值的label转换为合并后的
            else: # 变量为字符或类别型，直接转换
                woedata[varname] = woedata[varname].map(label_dict).map(label_map_dict)
    
            #2 读出该变量的woe表，然后根据该表把format格式化后的变量再转变为woe值
            var_woe = pd.read_pickle(woe_path + varname + ".pkl")
            woe_map = var_woe.set_index(varname)['woe_gb'].to_dict()
            woedata[varname] = woedata[varname].map(woe_map)
    else: # 转换单个变量woe
        try:
            format_dict = read_csv_format(fmt_file, varname)
        except:
            print("%s format not found" %(varname))
        label_dict = bins_to_labels(format_dict, vartype, splist=splist)
        label_map_dict = final_label(format_dict, label_dict, vartype)
        if vartype.lower() != 'category':
            bins = list(format_dict.keys())
            bins = [float('-inf')] + bins + [float('inf')]
            woedata[varname] = pd.cut(woedata[varname], bins, \
                                       right=True, labels=list(label_dict.values()))
            woedata[varname].map(label_map_dict) # 把合并后的值的label转换为合并后的
        else: # 变量为字符或类别型，直接转换
            woedata[varname] = woedata[varname].map(label_dict).map(label_map_dict)
    #数据集pickle化
    print('before pickle')
    pickcle_data = outdata + ".pkl"
    print(pickcle_data)
    woedata.to_pickle(pickcle_data)
    
    return woedata










    