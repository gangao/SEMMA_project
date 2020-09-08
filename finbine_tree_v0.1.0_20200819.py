# -*- coding: utf-8 -*-
"""
@name：finebin_tree
@author: cshuo
@version: 0.1.0
@date：20200819
@describe：实现细分栏的决策树方法：
        1）决策树算法以CART方法为基础，优化指标为基尼不纯度Gini Impurity的下降， 即基尼增益Gini Gain。
        2）主分栏算法以递归方法实现
        3）决策树分栏的限制条件包括：层数、分割前父节点样本数、分割后左右子节点样本数、好坏数，包括加权和非加权的设置
        4）不纯度的计算是单独实现，也可以替换为其他方法如熵entropy、信息值iv
@todo：
    1）与finebin主程序的连接，输入（数据、fmt、brks怎么来）、输出是什么
    2）变量类型的定义，数值、字符、序数型、类别型等等
    3）效率问题，大数据测试和类似方法、包的对比
    4）是否有必要整体写成面向对象即类的形式
    5）自动智能coarsebin的逻辑和实现

"""
import scorecardpy as sc
target = 'CREDITABILITY'
indata = sc.germancredit()
indata_check(indata)
indata[target] = indata[target].map({'good': 0, 'bad': 1})
var = 'DURATION.IN.MONTH'
brks = [1,2,3,6,9,12,18,24,30,36,48,60,72,84,96]
tree_split_list = recrusive_tree(indata, var, target, brks)


'''
对一个变量进行决策树方法的细分栏
1）实际上先要进行分位数方法的细分栏，要做到以下功能：
    1.1）分得够细，最多要有50个栏位
    1.2）按照变量类型vartype来给出智能的分栏点位
2）贪心算法来遍历细分栏，先实现根据cart算法即gini impurity来进行决策树分组，要做到：
    2.1）可设置决策树层数：max_depth
    2.2）可设置叶子节点最小数量（包括加权和非加权的数量）、叶子节点中好坏的最小数量（包括加权和非加权的数量）
    2.3）可设置min_impurity_decrease的阈值
'''
import copy
def recrusive_tree(indata, var, target, brks, depth=0, weight=None, special_value=None, maxdepth=4, listo=None, imprmt_threshold=0,
                    min_lf_obs=0, min_lf_obs_wt=0, min_lf_g_obs=0, min_lf_b_obs=0,
                    min_lf_g_obs_wt=0, min_lf_b_obs_wt=0):
    global listn
    listn = []
    depth+=1
    print("depth: %s" %(depth))
    if depth==1 and listo is None:
        listo=[]
        leaf_cnt = len(indata)
        leaf_g_cnt = len(indata.loc[indata[target] == 0])
        leaf_b_cnt = leaf_cnt - leaf_g_cnt
        if weight is not None:
            leaf_cnt_wt = indata[weight].sum()
            leaf_g_cnt_wt = indata.loc[indata[target] == 0][weight].sum()
            leaf_b_cnt_wt = leaf_cnt_wt - leaf_g_cnt_wt
        else:
            leaf_cnt_wt = leaf_cnt
            leaf_g_cnt_wt = leaf_g_cnt
            leaf_b_cnt_wt = leaf_b_cnt
        gini = 1
        if special_value is None: # 如果没设置初始的listo，初始化循环列表，变量取值区间为正负无穷，初始gini值为1
            irange = [float('-inf')] + [float('inf')]
        else:
            irange = initial_range(special_value)
        listo.append({'vrange': irange, 'gini': gini, 'leaf_cnt': leaf_cnt, 'leaf_g_cnt': leaf_g_cnt, 'leaf_b_cnt': leaf_b_cnt, \
                      'leaf_cnt_wt': leaf_cnt_wt, 'leaf_g_cnt_wt': leaf_g_cnt_wt, 'leaf_b_cnt_wt': leaf_b_cnt_wt})
    for ele in listo:
        print("ele in listo: %s" %(ele))
        vrange = ele['vrange']
        data_in_range = indata[(indata[var] >= float(vrange[0])) & (indata[var] < float(vrange[1]))]
        #要先判断总、好坏的观测数，不满足分割条件不进行分割
        if (ele['leaf_cnt'] < 2*min_lf_obs or ele['leaf_cnt_wt'] < 2*min_lf_obs_wt or
            ele['leaf_g_cnt'] < 2*min_lf_g_obs or ele['leaf_b_cnt'] < 2*min_lf_b_obs or
            ele['leaf_g_cnt_wt'] < 2*min_lf_g_obs_wt or ele['leaf_b_cnt_wt'] < 2*min_lf_b_obs_wt):
            listn.append(ele)
            print("father leaf not meet criteria")
            continue
        #父叶子满足条件后进行切割 
        father_gini = ele['gini']
        best_imprmt = find_best_split(data_in_range, var, target, vrange, brks, father_gini, weight)
        print("best_imprmt: %s" %(best_imprmt))
        if best_imprmt is None: # 如果子叶子均不满足条件，导致找不到最佳分割点，则继续下一步循环
            listn.append(ele)
            print("best_imprmt is None")
            continue
        l_brk = {'vrange' : [float(vrange[0]), float(best_imprmt['brk'])], 'gini' : best_imprmt['gini_left'],\
                 'leaf_cnt': best_imprmt['left_cnt'], 'leaf_g_cnt': best_imprmt['left_g_cnt'], 'leaf_b_cnt': best_imprmt['left_b_cnt'], \
                 'leaf_cnt_wt': best_imprmt['left_cnt_wt'], 'leaf_g_cnt_wt': best_imprmt['left_g_cnt_wt'], 'leaf_b_cnt_wt': best_imprmt['left_b_cnt_wt']}
        r_brk = {'vrange' : [float(best_imprmt['brk']), float(vrange[1])], 'gini' : best_imprmt['gini_right'],\
                 'leaf_cnt': best_imprmt['right_cnt'], 'leaf_g_cnt': best_imprmt['right_g_cnt'], 'leaf_b_cnt': best_imprmt['right_b_cnt'], \
                 'leaf_cnt_wt': best_imprmt['right_cnt_wt'], 'leaf_g_cnt_wt': best_imprmt['right_g_cnt_wt'], 'leaf_b_cnt_wt': best_imprmt['right_b_cnt_wt']}
        listn.append(l_brk)
        listn.append(r_brk)
    print("listn is : %s" %(listn))
    listo = copy.deepcopy(listn)
    if depth < maxdepth:
        recrusive_tree(indata, var, target, brks, depth, listo=listo)
    return listn


def initial_range(special_value):
    # special_value是一个list，其元素应该为数值
    range_l = round(float(sorted(special_value,reverse=False)[0]))+1
    irange = [range_l, float('inf')]
    return irange
        

def find_best_split(indata, var, target, vrange, brks, father_gini, weight, imprmt_threshold=0,
                    min_lf_obs=0, min_lf_obs_wt=0, min_lf_g_obs=0, min_lf_b_obs=0,
                    min_lf_g_obs_wt=0, min_lf_b_obs_wt=0):
    # 单层，只根据fmtlist分割当前的dataset
    # 当前的fmtlist只是全部fmtlist的一部
    # imprvmt要包括很多信息，1）总体提升；2）总叶子数量；3）左右叶子的数量
    current_brks = brks_in_vrange(vrange, brks) #根据当前取值范围确定新的切分点
    if len(current_brks) == 0:
        print("current_brks is None")
        best_imprmt = None
    else:
        imprmt_list = []
        for brk in current_brks:
            imprmt = impurity_improvement(indata, var, target, brk, father_gini, weight)
            imprmt_list.append(imprmt)
        print('imprmt_list : %s'%(imprmt_list))
        print("before: imprmt_list lenth is :%d"%(len(imprmt_list)))
        imprmt_list = match_threshold(imprmt_list, imprmt_threshold, min_lf_obs, min_lf_obs_wt, min_lf_g_obs,
                                      min_lf_b_obs, min_lf_g_obs_wt, min_lf_b_obs_wt) #按条件筛去不符合条件的
        if len(imprmt_list) == 0:
            print("after: imprmt_list lenth is 0")
            best_imprmt = None
        else:
            best_imprmt = sorted(imprmt_list, key=lambda x:x['gini_imprvmt'], reverse=True)[0] #按提高值排序
    return best_imprmt

def match_threshold(imprmt_list, imprmt_threshold, min_lf_obs, min_lf_obs_wt, min_lf_g_obs, min_lf_b_obs, min_lf_g_obs_wt, min_lf_b_obs_wt):
    imprmt_list_filter = []
    for ele in imprmt_list:
        # print(type(ele))
        # print(ele)
        if (ele['gini_imprvmt'] > imprmt_threshold and ele['left_cnt'] > min_lf_obs and 
            ele['right_cnt'] > min_lf_obs and ele['left_cnt_wt'] > min_lf_obs_wt and 
            ele['right_cnt_wt'] > min_lf_obs_wt and ele['left_g_cnt'] > min_lf_g_obs and 
            ele['right_g_cnt'] > min_lf_g_obs and ele['left_b_cnt'] > min_lf_b_obs and 
            ele['right_b_cnt'] > min_lf_b_obs and ele['left_g_cnt_wt'] > min_lf_g_obs_wt and 
            ele['right_g_cnt_wt'] > min_lf_g_obs_wt and ele['left_g_cnt_wt'] > min_lf_b_obs_wt and 
            ele['right_g_cnt_wt'] > min_lf_b_obs_wt
            ):
            imprmt_list_filter.append(ele)
    return imprmt_list_filter

def impurity_improvement(indata, var, target, brk, father_gini, weight=None):
    # bins = [float('-inf')] + brkpoint + [float('inf')]
    # indata[varname] = pd.cut(indata[varname], bins, \
    #                                right=True, labels=['left', 'right'])
    # imprvmt要包括很多信息，1）总体提升；2）总叶子数量；3）左右叶子的数量
    # print(father_gini)
    imprvmt_result = {}
    # all_cnt = indata[weight].sum()
    # left_cnt = len(indata.loc[indata[var] < brk])
    # right_cnt = len(indata.loc[indata[var] >= brk])
    # if weight != None:
    #     all_cnt_wt = indata[weight].sum()
    #     left_cnt_wt = indata.loc[indata[var] < brk][weight].sum()
    #     right_cnt_wt = indata.loc[indata[var] >= brk][weight].sum()
    # else:
    #     all_cnt_wt = all_cnt
    #     left_cnt_wt = left_cnt
    #     right_cnt_wt = right_cnt
    print("brk is : %s"%(brk))
    gini_left = gini_impurity(indata.loc[indata[var] < brk], target, weight)
    gini_right = gini_impurity(indata.loc[indata[var]  >= brk], target, weight)
    left_cnt_wt = gini_left['cnt_wt']
    right_cnt_wt = gini_right['cnt_wt']
    all_cnt_wt = left_cnt_wt + right_cnt_wt
    if gini_left['gini'] is not None and gini_right['gini'] is not None:
        imprvmt = father_gini - gini_left['gini'] * (left_cnt_wt/all_cnt_wt) - gini_right['gini'] * (right_cnt_wt/all_cnt_wt)
    else:
        print("No gini, imprvmt = -1")
        imprvmt = -1 # 无法计算出gini则提升为-1
    imprvmt_result['brk'] = brk
    imprvmt_result['gini_imprvmt'] = imprvmt
    imprvmt_result['gini_left'] = gini_left['gini']
    imprvmt_result['gini_right'] = gini_right['gini']
    imprvmt_result['left_cnt'] = gini_left['cnt']
    imprvmt_result['left_cnt_wt'] = gini_left['cnt_wt']
    imprvmt_result['left_g_cnt'] = gini_left['good_cnt']
    imprvmt_result['left_g_cnt_wt'] = gini_left['good_cnt_wt']
    imprvmt_result['left_b_cnt'] = gini_left['bad_cnt']
    imprvmt_result['left_b_cnt_wt'] = gini_left['bad_cnt_wt']
    imprvmt_result['right_cnt'] = gini_right['cnt']
    imprvmt_result['right_cnt_wt'] = gini_right['cnt_wt']
    imprvmt_result['right_g_cnt'] = gini_right['good_cnt']
    imprvmt_result['right_g_cnt_wt'] = gini_right['good_cnt_wt']
    imprvmt_result['right_b_cnt'] = gini_right['bad_cnt']
    imprvmt_result['right_b_cnt_wt'] = gini_right['bad_cnt_wt']
    return imprvmt_result

def brks_in_vrange(vrange, brks):
    brks_in_vrange = []
    range_l = float(vrange[0])
    range_r = float(vrange[1])
    for brk in brks:
        if brk > range_l and brk < range_r:
            brks_in_vrange.append(brk)
    return brks_in_vrange

def gini_impurity(indata, target, weight=None):
    gini_dict = {}
    cnt = len(indata)
    good_cnt = len(indata.loc[indata[target] == 0])
    bad_cnt = cnt - good_cnt
    if weight is not None:
        cnt_wt = indata[weight].sum()
        good_cnt_wt = indata.loc[indata[target] == 0][weight].sum()
        bad_cnt_wt = cnt_wt - good_cnt_wt
    else:
        cnt_wt = cnt
        good_cnt_wt = good_cnt
        bad_cnt_wt = bad_cnt
    if cnt_wt != 0:
        good_pct = good_cnt_wt / cnt_wt
        gini_dict['gini'] = 2*good_pct*(1-good_pct)
    else: 
        gini_dict['gini'] = None
        print("gini is None")
    gini_dict['cnt'] = cnt
    gini_dict['cnt_wt'] = cnt
    gini_dict['good_cnt'] = good_cnt
    gini_dict['good_cnt_wt'] = good_cnt_wt
    gini_dict['bad_cnt'] = bad_cnt
    gini_dict['bad_cnt_wt'] = bad_cnt_wt
    return gini_dict




'''
递归函数测试
# import sys   
# sys.setrecursionlimit(1000)
    
def cut(list_ele):
    length = list_ele[1] - list_ele[0]
    brkpoint = list_ele[0] + round(length / 2)
    newlist_left = [list_ele[0] , brkpoint]
    newlist_right = [brkpoint , list_ele[1]]
    return newlist_left, newlist_right
    
def recrusive(listo, step):
    global listn
    step+=1
    print(step)
    listn = []
    for e in listo:
        print(e)
        l,r = cut(e)
        listn.append(l)
        listn.append(r)
    listo = listn
    if step < 3:
        recrusive(listo,step)

global listo 
global listn 
global step
listo = [[1,100]]
recrusive(listo, 0)
'''
