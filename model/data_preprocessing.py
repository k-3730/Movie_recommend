import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

__all__ = ["preprocess_data", "train_test_split_custom"]

# 범주형 변수 인코딩
def preprocess_data(data, sparse_features):
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    return data

# 데이터 나누기
def train_test_split_custom(data):
    return train_test_split(data, test_size=0.2, random_state=2020)
