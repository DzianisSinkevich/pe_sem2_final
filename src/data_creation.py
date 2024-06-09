import os
from sklearn.datasets import fetch_openml
import pandas as pd
import warnings
import numpy as np

import configparser
import logging
import os

warnings.filterwarnings('ignore')

settings = configparser.ConfigParser()
settings.read('config/settings.ini')

stream_handler = logging.StreamHandler()
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
file_handler = logging.FileHandler(os.path.join(parent_dir, 'logs/log.log'))
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(stream_handler)
logger.addHandler(file_handler)


def delete_files(dir_path):
    try:
        files = os.listdir(dir_path)
        for file in files:
            file_path = os.path.join(dir_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        logger.info("Все файлы из '" + dir_path + "' удалены успешно.")
    except OSError:
        logger.critical("Ошибка удаления файлов.")


def save_file(df, dir_path, file_name):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    try:
        file_path = os.path.join(dir_path, file_name)
        df.to_csv(file_path, index=False)
        logger.info("Файл " + file_path + " создан успешно.")
    except IOError:
        logger.critical("Ошибка создания файла " + file_path + " .")


def get_dfs():
    openml_df = fetch_openml("Brilliant-Diamonds")

    df = pd.DataFrame(data=openml_df['data'], columns=openml_df['feature_names'])
    df.drop(df.tail(1).index, inplace=True)
    df1, df2 = np.split(df.sample(frac=1, random_state=42), 2)
    return df1, df2


def dc_main():
    logger.info("<<< Создания набора данных начато >>>")

    dir_path = 'test'
    if os.path.isdir(dir_path):
        delete_files(dir_path)
    dir_path = 'train'
    if os.path.isdir(dir_path):
        delete_files(dir_path)

    df_train, df_test = get_dfs()
    save_file(df_train, 'train', 'df_train_0.csv')
    save_file(df_test, 'test', 'df_test_0.csv')

    logger.info("<<< Создание набора данных закончено >>>\n")
