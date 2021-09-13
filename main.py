import pandas as pd



''' csv 파일을 load 해서 numpy 배열로 바꿔주는 함수 '''
def load_CSV(data):
    data1 = []
    data1 = pd.read_csv(data)
    data2 = data1.to_numpy()
    return data2
