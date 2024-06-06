
import numpy as np  # библиотека Numpy для операций линейной алгебры и прочего

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer  # т.н. преобразователь колонок
from sklearn.base import BaseEstimator, TransformerMixin  # для создания собственных преобразователей / трансформеров данных
from sklearn.linear_model import LogisticRegression  # Логистичекая регрессия от scikit-learn

# предварительная обработка числовых признаков
from sklearn.preprocessing import MinMaxScaler  # Импортируем нормализацию от scikit-learn
from sklearn.preprocessing import StandardScaler  # Импортируем стандартизацию от scikit-learn
# предварительная обработка категориальных признаков
from sklearn.preprocessing import OneHotEncoder  # Импортируем One-Hot Encoding от scikit-learn
from sklearn.preprocessing import OrdinalEncoder  # Импортируем Порядковое кодированиеот scikit-learn

import warnings
import pickle
import os
import pandas as pd  # Библиотека Pandas для работы с табличными данными

warnings.filterwarnings('ignore')


def read_file(file_path):
    try:
        df = pd.read_csv(file_path)
        # df['day_mean_temp'] = pd.to_numeric(df['day_mean_temp'])
        df = df.fillna(0)
        logger.critical("File " + file_path + " readed successfully.")
        return df
    except IOError:
        logger.critical("Error uccured while readed file '" + file_path + "'.")


def save_model(pipeline):
    try:
        if not os.path.isdir('pipeline'):
            os.mkdir('pipeline')
        pickle.dump(pipeline, open('pipeline/pipeline.pkl', 'wb'))
        logger.critical("Pipeline pipeline/pipeline.pkl saved successfully.")
    except IOError:
        logger.critical("Error uccured while saved pipeline/pipeline.pkl.")


def preparation(train_df_path):
    df = read_file(train_df_path)

    x_train, y_train = df.drop(['cut'], axis=1), df['cut']

    num_pipe = Pipeline([
        ('QuantReplace', QuantileReplacer(threshold=0.001, )),
        ('scaler', StandardScaler()),
        ('minmaxscaler', MinMaxScaler())
    ])
    num = ['price', 'carat']

    cat_pipe = Pipeline([
        ('replace_rare', RareGrouper(threshold=0.0001, other_value='Other')),
        ('encoder', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False))
    ])

    cat = ['shape', 'color', 'clarity']

    cat_pipe_type = Pipeline([
        ('encoder', OrdinalEncoder())
    ])

    cat_type = ['type']

    # Объединяем все пайплайны в один ColumnTransformer
    preprocessors = ColumnTransformer(transformers=[
        ('num', num_pipe, num),
        ('cat', cat_pipe, cat),
        ('cat_type', cat_pipe_type, cat_type)
    ])

    pipe_all = Pipeline([
        ('preprocessors', preprocessors),
        ('model', LogisticRegression(random_state=42))
        ])
    pipe_all.fit(x_train, y_train)

    save_model(pipe_all)


class QuantileReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.05):
        self.threshold = threshold
        self.quantiles = {}

    def fit(self, X, y=None):
        for col in X.select_dtypes(include='number'):
            low_quantile = X[col].quantile(self.threshold)
            high_quantile = X[col].quantile(1 - self.threshold)
            self.quantiles[col] = (low_quantile, high_quantile)
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in X.select_dtypes(include='number'):
            low_quantile, high_quantile = self.quantiles[col]
            rare_mask = ((X[col] < low_quantile) | (X[col] > high_quantile))
            if rare_mask.any():
                rare_values = X_copy.loc[rare_mask, col]
                replace_value = np.mean([low_quantile, high_quantile])
                if rare_values.mean() > replace_value:
                    X_copy.loc[rare_mask, col] = high_quantile
                else:
                    X_copy.loc[rare_mask, col] = low_quantile
        return X_copy


class RareGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.05, other_value='Other'):
        self.threshold = threshold
        self.other_value = other_value
        self.freq_dict = {}

    def fit(self, X, y=None):
        for col in X.select_dtypes(include=['object']):
            freq = X[col].value_counts(normalize=True)
            self.freq_dict[col] = freq[freq >= self.threshold].index.tolist()
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        for col in X.select_dtypes(include=['object']):
            X_copy[col] = X_copy[col].apply(lambda x: x if x in self.freq_dict[col] else self.other_value)
        return X_copy


def mp_main():
    logger.info("<<< Start preparation >>>")
    preparation('train/df_train_0.csv')
    logger.info("<<< Finish preparation >>>\n")
