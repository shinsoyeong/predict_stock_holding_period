import pandas as pd



''' csv 파일을 load 해서 numpy 배열로 바꿔주는 함수 '''
def load_CSV(data):
    data1 = []
    data1 = pd.read_csv(data)
    data2 = data1.to_numpy()
    return data2

cus_info = load_CSV(cus_info.csv)
iem_info = load_CSV(iem_info.csv)
stk_bnc_hist = load_CSV(stk_bnc_hist.csv)
stk_hld_train = load_CSV(stk_hld_train.csv)
stk_hld_test = load_CSV(stk_hld_test.csv)
