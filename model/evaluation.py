from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd

__all__ = ["evaluate_model", "save_predictions"]

def evaluate_model(model, test_model_input, test_targets):
    pred_ans = model.predict(test_model_input, batch_size=256)
    # 평가지표 : LogLoss, AUC 사용
    print("test LogLoss", round(log_loss(test_targets, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test_targets, pred_ans), 4))

def save_predictions(model, test, test_model_input, original_file):
    test['pred_target'] = model.predict(test_model_input, batch_size=256)

    # 영화이름 디코딩 전 
    result_df = test[['userId', 'title', 'pred_target']]

    # 영화이름 디코딩 후 파일 저장
    original_data = pd.read_csv(original_file)
    lbe = LabelEncoder()
    lbe.fit(original_data['title'])
    result_df['title'] = lbe.inverse_transform(result_df['title'])
    result_df.to_csv('./decoded_predictions.csv', index=False)

