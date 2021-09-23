import pandas as pd  # 데이터를 로드하기 위한 라이브러리
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt  # plotting 하기 위한 라이브러리
from sklearn import preprocessing  # 데이터를 전처리하기 위한 라이브러리
import sys, os

sys.path.append(os.pardir)
from sklearn.preprocessing import *

''' csv 파일을 load해서 반환하는 함수 '''
def load_CSV(file):
    data = []
    data = pd.read_csv(file)
    return data



''' 각 파일의 이름 변수에 data 로드 '''
cus_info = load_CSV('cus_info.csv')  # 고객 및 주거래 계좌 정보
iem_info = load_CSV('iem_info_20210902.csv')  # 종목 정보 _주식 종목에 대한 코드 정보
stk_bnc_hist = load_CSV('stk_bnc_hist.csv')  # 국내 주식 잔고 이력 _일별 종목 잔고수량 및 금액, 액면가 정보
stk_hld_test = load_CSV('stk_hld_test.csv')  # 국내 주식 보유 기간(train) _고객에게 제공되는 과거 국내주식 보유기간 데이터 (681,472건)
stk_hld_train = load_CSV('stk_hld_train.csv')  # 국내 주식 보유 기간(test) _개발한 알고리즘 검증을 위한 문제지 (70,596건)

print(stk_bnc_hist.dtypes)
# 정규화할 컬럼 추출
bnc_hist_norm = stk_bnc_hist[['bnc_qty', 'tot_aet_amt', 'stk_par_pr']]
stk_bnc_hist.drop(['bnc_qty', 'tot_aet_amt', 'stk_par_pr'], axis=1, inplace=True)

# 데이터 정규화 코드
transformer = MinMaxScaler()
transformer.fit(bnc_hist_norm)
bnc_hist_norm = transformer.transform(bnc_hist_norm)

bnc_hist_norm = pd.DataFrame(bnc_hist_norm)
bnc_hist_norm.columns = (['bnc_qty', 'tot_aet_amt', 'stk_par_pr'])# 컬럼에 레이블명 지정
print(bnc_hist_norm)

# 정규화한 컬럼을 기존 DataFrame과 수평 결합
stk_bnc_hist = pd.concat([stk_bnc_hist, bnc_hist_norm], axis=1)
print(stk_bnc_hist)

stk_bnc_train = pd.concat([cus_info, stk_hld_train, stk_bnc_hist], axis=1)
#계좌ID를 기준으로 train data를 결합
stk_hld_test = pd.concat([cus_info, stk_hld_test], axis=1)
#계좌ID를 기준으로 test data를 결합




batch_size = 1000 #대량의 data를 처리하기 위한 mini batch

#시발어케짜지...........
