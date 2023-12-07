# Movie_recommend

영화 데이터 : https://grouplens.org/datasets/movielens/

![Python 3.10.13](https://img.shields.io/badge/python-3.10.13-blue.svg)
![TensorFlow 2.10.0](https://img.shields.io/badge/TensorFlow-2.10.0-orange.svg)

원본 데이터는 영화 데이터 주소를 참고하면 되고 전처리되어 있는 데이터를 사용했습니다.

## 설명

영화제목, 장르, 태그, 평점 데이터를 Embedding 시키고 Concat 합니다.

그리고 전체 벡터 값을 각각 Factorization machine 모델, Neural Network 모델을 사용합니다.

최종적으로 각 사용자가 영화를 볼 확률을 Sigmoid로 계산해서 결과값을 pred_target에 저장합니다.

결과는 userId, title, pred_target 3개의 컬럼을 가지는 csv 파일로 저장됩니다.


## 모델
DeepFM : https://arxiv.org/abs/1703.04247

![1](https://github.com/k-3730/Movie_recommend/assets/45035923/efb3a554-db2f-46f7-a96d-39fd168f8391)

### Hyper Parameter

Batch size = 256

epoch = 30


## 데이터
userId : 사용자 ID (1000명)

title : 영화 제목 (1000개)

genres : 영화 장르

tag : 영화 태그

rating : 영화 평점

target : 영화 관람 여부 (0 = 관람 X, 1 = 관람 O)

## 결과 예시
LogLoss : 0.3966

AUC     : 0.8815

787번 사용자가 Cliffhanger 영화를 볼 확률은 0.002로 매우 낮은 결과를 볼 수 있습니다.

![image](https://github.com/k-3730/Movie_recommend/assets/45035923/50970e21-c60d-4ccb-b937-9c8234b96179)
