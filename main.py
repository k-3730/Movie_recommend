import pandas as pd
from model import preprocess_data, train_test_split_custom
from model import train_model
from model import evaluate_model, save_predictions

if __name__ == "__main__":
    data = pd.read_csv('C:/Users/user/Desktop/1204 - 1208/Movie/movielens.csv')
    sparse_features = ['userId', 'title', 'genres', 'tag'] # 범주형 변수

    # 데이터 전처리
    data = preprocess_data(data, sparse_features)

    train, test = train_test_split_custom(data)

    # 모델 훈련
    model, train_model_input, test_model_input = train_model(train, test, sparse_features)

    # 모델 평가 및 결과 저장
    evaluate_model(model, test_model_input, test['target'])
    save_predictions(model, test, test_model_input, 'movielens.csv')
