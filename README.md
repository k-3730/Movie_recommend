# Movie_recommend

영화 데이터 : https://grouplens.org/datasets/movielens/

![Python 3.10.13](https://img.shields.io/badge/python-3.10.13-blue.svg)
![TensorFlow 2.10.0](https://img.shields.io/badge/TensorFlow-2.10.0-orange.svg)

원본 데이터는 영화 데이터 주소를 참고하면 되고 전처리되어 있는 데이터를 사용했습니다.

각 사용자가 영화를 볼 확률에 대해서 계산하고 결과값을 pred_target에 저장합니다.

최종결과는 userId, title, pred_target 3개의 컬럼을 가지는 csv 파일로 저장됩니다.


## 모델
DeepFM : https://arxiv.org/abs/1703.04247

DeepFM 모델을 사용했습니다.


## 데이터 설명
userId : 사용자 ID (1000명)

title : 영화 제목 (1000개)

genres : 영화 장르

tag : 영화 태그

rating : 영화 평점

target : 영화 관람 여부 (0 = 관람 X, 1 = 관람 O)
