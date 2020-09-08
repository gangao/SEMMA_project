# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 20:35:44 2020

@author: ginge
"""
import scorecardpy as sc
target = 'CREDITABILITY'
indata = sc.germancredit()
indata_check(indata)
indata[target] = indata[target].map({'good': 0, 'bad': 1})

def recrusive_tree(indata, var, target, brks, depth, weight=None, maxdepth=3, listo=None, imprmt_threshold=0, \
                    min_lf_obs=0, min_lf_obs_wt=0, min_lf_g_obs=0, min_lf_b_obs=0,\
                    min_lf_g_obs_wt=0, min_lf_b_obs_wt=0):
    global listn
    depth+=1
    print(depth)
    listn = []
    if depth==1:
        leaf_cnt = len(indata)
        leaf_g_cnt = len(indata.loc[indata[target] == 0])
        leaf_b_cnt = leaf_cnt - leaf_g_cnt
        if weight != None:
            leaf_cnt_wt = indata[weight].sum()
            leaf_g_cnt_wt = indata.loc[indata[target] == 0][weight].sum()
            leaf_b_cnt_wt = leaf_cnt_wt - leaf_g_cnt_wt
        else:
            leaf_cnt = leaf_cnt
            leaf_g_cnt = leaf_g_cnt
            leaf_b_cnt = leaf_b_cnt
        gini = 1
        if listo=None: # 如果没设置初始的listo，初始化循环列表，变量取值区间为正负无穷，初始gini值为1
            irange = [float('-inf')] + [float('inf')]
        listo.append({'vrange': irange, 'gini': gini, 'leaf_cnt': leaf_cnt, 'leaf_g_cnt': leaf_g_cnt, 'leaf_b_cnt': leaf_b_cnt, \
                      'leaf_cnt_wt': leaf_cnt_wt, 'leaf_g_cnt_wt': leaf_g_cnt_wt, 'leaf_b_cnt_wt': leaf_b_cnt_wt})
    for ele in listo:
        print(ele)
        vrange = ele['vrange']
        data_in_range = indata[indata[var] >= float(vrange[0]) and indata[var] < float(vrange[1])]
        #要先判断总、好坏的观测数，不满足分割条件不进行分割
        if (ele['leaf_cnt'] < 2*min_lf_obs or ele['leaf_cnt_wt'] < 2*min_lf_obs_wt or
            ele['leaf_g_cnt'] < 2*min_lf_g_obs or ele['leaf_b_cnt'] < 2*min_lf_b_obs or
            ele['leaf_g_cnt_wt'] < 2*min_lf_g_obs_wt or ele['leaf_b_cnt_wt'] < 2*min_lf_b_obs_wt):
            continue
        # 
        father_gini = ele['gini']
        best_imprmt = find_best_split(data_in_range, var, target, vrange, brks, father_gini, weight)
        if best_imprmt == None:
            continue
        else:
            l_brk = {'vrange' : [range_l, float(best_imprmt['brk'])], 'gini' : best_imprmt['gini_left'],\
                     'leaf_cnt': best_imprmt['left_cnt'], 'leaf_g_cnt': best_imprmt['left_g_cnt'], 'leaf_b_cnt': best_imprmt['left_b_cnt'], \
                     'leaf_cnt_wt': best_imprmt['left_cnt_wt'], 'leaf_g_cnt_wt': best_imprmt['left_g_cnt_wt'], 'leaf_b_cnt_wt': best_imprmt['left_b_cnt_wt']}}
            r_brk = {'vrange' : [float(best_imprmt['brk']), range_r], 'gini' : best_imprmt['gini_right']\
                     'leaf_cnt': best_imprmt['right_cnt'], 'leaf_g_cnt': best_imprmt['right_g_cnt'], 'leaf_b_cnt': best_imprmt['right_b_cnt'], \
                     'leaf_cnt_wt': best_imprmt['right_cnt_wt'], 'leaf_g_cnt_wt': best_imprmt['right_g_cnt_wt'], 'leaf_b_cnt_wt': best_imprmt['right_b_cnt_wt']}}
            listn.append(l_brk)
            listn.append(r_brk)
    listo = listn.deepcopy()
    if depth < maxstep:
        recrusive(indata, var, target, brks, depth, listo=listo)
    return listo


def find_best_split(indata, var, target, vrange, brks, father_gini, weight, imprmt_threshold=0, \
                    min_lf_obs=0, min_lf_obs_wt=0, min_lf_g_obs=0, min_lf_b_obs=0,\
                    min_lf_g_obs_wt=0, min_lf_b_obs_wt=0):
    # 单层，只根据fmtlist分割当前的dataset
    # 当前的fmtlist只是全部fmtlist的一部
    # imprvmt要包括很多信息，1）总体提升；2）总叶子数量；3）左右叶子的数量
    current_brks = brks_in_vrange(vrange, brks) #根据当前取值范围确定新的切分点
    imprmt_list = []
    for brk in current_brks:
        imprmt = impurity_improvement(indata, var, target, brk, father_gini, weight)
        imprmt_list.append(imprmt)
    imprmt_list = match_threshold(imprmt_list, imprmt_threshold, min_lf_obs, min_lf_obs_wt, min_lf_g_obs,\
                                  min_lf_b_obs, min_lf_g_obs_wt, min_lf_b_obs_wt)
    best_imprmt = sort_the_best_imprmt(imprmt_list)
    return best_imprmt

def match_threshold(imprmt_list, imprmt_threshold, min_lf_obs, min_lf_obs_wt, min_lf_g_obs, min_lf_b_obs, min_lf_g_obs_wt, min_lf_b_obs_wt):
    imprmt_list_filter = {}
    for ele in imprmt_list:
        if (ele['gini'] > imprmt_threshold 
            and ele['left_cnt'] > min_lf_obs and ele['right_cnt'] > min_lf_obs
            and ele['left_cnt_wt'] > min_lf_obs_wt and ele['right_cnt_wt'] > min_lf_obs_wt
            and ele['left_g_cnt'] > min_lf_g_obs and ele['right_g_cnt'] > min_lf_g_obs
            and ele['left_b_cnt'] > min_lf_b_obs and ele['right_b_cnt'] > min_lf_b_obs
            and ele['left_g_cnt_wt'] > min_lf_g_obs_wt and ele['right_g_cnt_wt'] > min_lf_g_obs_wt
            and ele['left_g_cnt_wt'] > min_lf_b_obs_wt and ele['right_g_cnt_wt'] > min_lf_b_obs_wt
            )
            
        min_lf_obs, min_lf_obs_wt, min_lf_g_obs, min_lf_b_obs, min_lf_g_obs_wt, min_lf_b_obs_wt
        if ele['gini'] > 

def impurity_improvement(indata, var, target, brk, father_gini, weight=None):
    # bins = [float('-inf')] + brkpoint + [float('inf')]
    # indata[varname] = pd.cut(indata[varname], bins, \
    #                                right=True, labels=['left', 'right'])
    # imprvmt要包括很多信息，1）总体提升；2）总叶子数量；3）左右叶子的数量
    print(father_gini)
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
    gini_left = gini_impurity(indata.loc[indata[var] < brk], target, weight)
    gini_right = gini_impurity(indata.loc[indata[var]  >= brk], target, weight)
    left_cnt_wt = gini_left['cnt_wt']
    right_cnt_wt = gini_right['cnt_wt']
    all_cnt_wt = left_cnt_wt + right_cnt_wt
    imprvmt = father_gini - gini_left['gini'] * (left_cnt_wt/all_cnt_wt) - gini_right['gini'] * (right_cnt_wt/all_cnt_wt)
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
    return imprvmt

def brks_in_vrange(vrange, brks):
    new_brks = []
    range_l = float(vrange[0])
    range_r = float(vrange[1])
    for brk in brks:
        if brk > range_l and brk < range_r:
            new_brks.append(brk)
    return new_brks

def gini_impurity(indata, target, weight=None):
    gini_dict = {}
    cnt = len(indata)
    good_cnt = len(indata.loc[indata[target] == 0])
    bad_cnt = cnt - good_cnt
    if weight != None:
        cnt_wt = indata[weight].sum()
        good_cnt_wt = indata.loc[indata[target] == 0][weight].sum()
        bad_cnt_wt = cnt_wt - good_cnt_wt
    else:
        cnt_wt = cnt
        good_cnt_wt = good_cnt
        bad_cnt_wt = bad_cnt
    good_pct = good_cnt_wt / cnt_wt
    gini_dict['gini'] = 2*good_pct*(1-good_pct)
    gini_dict['cnt'] = cnt
    gini_dict['cnt_wt'] = cnt
    gini_dict['good_cnt'] = good_cnt
    gini_dict['good_cnt_wt'] = good_cnt_wt
    gini_dict['bad_cnt'] = bad_cnt
    gini_dict['bad_cnt_wt'] = bad_cnt_wt
    return gini_dict


vrange = ['-inf', 'inf']
vrange1 = [float('-inf')] + [float('inf')]
vrange1 = [0,100]
brks = [-100,0,1,5,10,20,30,40,100,1000]

dict1 = {'range':[100,1000],'gini':1}
dict2 = {'range':[1000,2000],'gini':0.5}
listo=[]
listo.append(dict1)
listo.append(dict2)
for e in listo:
    print(e['range'])
    print(e['gini'])


print(brks_in_vrange(vrange1, brks))

def finebin_tree(indata, varname, vartype, weight, special_value, bin_num, save_path):
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
    # 数据和切分点列表除去特殊值后，对于剩余的分割点循环计算出最佳分割点
    for i in fmtlist:
        tree_find_best_split()
    

def tree_find_best_split():
    if max_depth < maxdepth and min_leaf_wt >= min_leaf_wt and min_leaf >= min_leaf \
        and min_leaf_good >= min_leaf_good and min_leaf_good_wt >= min_leaf_good_wt \
        and min_leaf_bad >= min_leaf_bad and min_leaf_bad_wt >= min_leaf_bad_wt:
        brk_point = find_best_split(dataset, fmtlist)
        tree_brk_point.add(brk_point)
        dataset_list = [dataset_left, dataset_right] = data_split(brk_point)
        fmtlist = [fmtlist_left, fmtlist_right] = fmt_split(brk_point)
        data_fmt_dict = dict(zip(dataset_list,  fmt_list))
        for dataset, fmtlist in dataset_list:
            tree_find_best_split(dataset, fmtlist)
        max_depth +=1
    


import scorecardpy as sc
indata = sc.germancredit()
indata_check(indata)
target = 'CREDITABILITY'
indata[target] = indata[target].map({'good': 0, 'bad': 1})

var = 'DURATION.IN.MONTH'
brkpoint = 20
indata1 = indata.loc[indata[var] < brkpoint]


    
print(gini_impurity(indata1, 'CREDITABILITY'))
print(gini_impurity(indata.loc[indata[var] < brkpoint], 'CREDITABILITY'))
weight = 'DURATION.IN.MONTH'
print(gini_impurity(indata1, 'CREDITABILITY', 'DURATION.IN.MONTH'))

df = indata.loc[indata[target] == 0][weight]
brklist = {[-inf, 10]:gini1, [10:100]:gini2}
    

    

impurity_improvement(1, indata, 'DURATION.IN.MONTH', 20, 'CREDITABILITY', 'DURATION.IN.MONTH')

def tst(max_depth):
    max_depth +=1
    print(max_depth)
    
tst(1)

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


def find


def impurity_improvement(father_gini, indata, brkpoint):
    gini_all = gini_impurity(indata)
    gini_left = gini_impurity(indata_left)
    gini_right = gini_impurity(indata_right)
    









indata.columns
global list_all
list_all = [[1,100]]
recrusive(list_all, 0)

print(list_all)
print('1: %s'%(list_all))
leftnum=1
brkpoint=50
rightnum = 100
newlist_left = [leftnum , brkpoint]
newlist_right = [brkpoint , rightnum]
list_all = []
list_all.append(newlist_left)
list_all.append(newlist_right)
for list_ele in list_all:
    print(list_ele[0])




def coarsebin_auto():
    '''
    根据woe结果，对细分栏进行智能的粗分合并
    1）通过综合判断：woe差值、分栏人数进行合并，
        可设置最小分组人数（加权、非加权），最小分组好坏人数（加权非加权）
    2）可以对缺失值按靠近原则进行合并，依据woe靠近阈值和分栏人数，综合判断。
    3）可设置单调性智能合并，逻辑：？？
    '''
    
