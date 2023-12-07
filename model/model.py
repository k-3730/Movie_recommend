from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, get_feature_names
import pandas as pd

__all__ = ["train_model"]

def train_model(train, test, sparse_features):
    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=pd.concat([train, test])[feat].nunique(), embedding_dim=4)
                              for feat in sparse_features]
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
    model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'])

    history = model.fit(train_model_input, train['target'].values,
                        batch_size=256, epochs=30, verbose=2, validation_split=0.2)
    
    return model, train_model_input, test_model_input
