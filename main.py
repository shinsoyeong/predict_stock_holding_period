import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
import sys, os
sys.path.append(os.pardir)
from sklearn.preprocessing import *


''' csv 파일을 load 해서 numpy 배열로 바꿔주는 함수 '''
def load_CSV(data):
    data1 = []
    data1 = pd.read_csv(data)
    data2 = data1.to_numpy()
    return data2

cus_info = pd.read_csv('cus_info.csv')
iem_info = pd.read_csv('iem_info_20210902.csv')
stk_bnc_hist = pd.read_csv('stk_bnc_hist.csv')
stk_hld_train = pd.read_csv('stk_hld_train.csv')
stk_hld_test = pd.read_csv('stk_hld_test.csv')

#정규화할 컬럼 추출
bnc_hist_norm = stk_bnc_hist[['bnc_qty', 'tot_aet_amt', 'stk_par_pr']]
stk_bnc_hist.drop(['bnc_qty', 'tot_aet_amt', 'stk_par_pr'], axis=1, inplace=True)

#데이터 정규화 코드
transformer = MinMaxScaler()
transformer.fit(bnc_hist_norm)
bnc_hist_norm = transformer.transform(bnc_hist_norm)

bnc_hist_norm = pd.DataFrame(bnc_hist_norm)
bnc_hist_norm.columns=(['bnc_qty', 'tot_aet_amt', 'stk_par_pr']) #컬럼에 레이블명 지정
print(bnc_hist_norm)

#정규화한 컬럼을 기존 DataFrame과 수평 결합
stk_bnc_hist = pd.concat([stk_bnc_hist, bnc_hist_norm], axis=1)
print(stk_bnc_hist)
