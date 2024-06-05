import os
from sklearn.datasets import fetch_openml
import pandas as pd
import warnings
import numpy as np

warnings.filterwarnings('ignore')


def delete_files(dir_path):
    try:
        files = os.listdir(dir_path)
        for file in files:
            file_path = os.path.join(dir_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print("All files from '" + dir_path + "' deleted successfully.")
    except OSError:
        print("Error occurred while deleting files.")


def save_file(df, dir_path, file_name):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    try:
        file_path = os.path.join(dir_path, file_name)
        df.to_csv(file_path, index=False)
        print("File " + file_path + " created successfully.")
    except IOError:
        print("Error uccured while creating file " + file_path + " .")


def get_dfs():
    openml_df = fetch_openml("Brilliant-Diamonds")

    df = pd.DataFrame(data=openml_df['data'], columns=openml_df['feature_names'])
    df.drop(df.tail(1).index, inplace=True)
    df1, df2 = np.split(df.sample(frac=1, random_state=42), 2)
    return df1, df2


def dc_main():
    print("<<< Start data creation >>>")

    dir_path = 'test'
    if os.path.isdir(dir_path):
        delete_files(dir_path)
    dir_path = 'train'
    if os.path.isdir(dir_path):
        delete_files(dir_path)

    df_train, df_test = get_dfs()
    save_file(df_train, 'train', 'df_train_0.csv')
    save_file(df_test, 'test', 'df_test_0.csv')

    print("<<< Finish data creation >>>\n")
