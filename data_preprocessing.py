from sklearn.preprocessing import LabelEncoder  # Импортируем LabelEncoder от scikit-learn
import pandas as pd  # Библиотека Pandas для работы с табличными данными
import warnings
import logging

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None


def read_file(file_path):
    try:
        df = pd.read_csv(file_path)
        logger.critical("File " + file_path + " readed successfully.")
        return df
    except IOError:
        logger.critical("Error uccured while readed file '" + file_path + "'.")


def save_file(df, file_path):
    try:
        df.to_csv(file_path, index=False)
        logger.critical("File " + file_path + " created successfully.")
    except IOError:
        logger.critical("Error uccured while creating file " + file_path + " .")


def df_prerpocessing(file_path):
    logger.info("<< Start preprocessing '" + file_path + "' >>")
    df = read_file(file_path)

    df = df[['shape', 'price', 'carat', 'cut', 'color', 'clarity', 'type']]
    # Сокращаем количество качества огранки до "low" и "hight"
    df['cut'] = df['cut'].replace({"Fair": "low",
                                   "'Very Good'": "low",
                                   "Very Good": "low",
                                   "'Super Ideal'": "hight",
                                   "Super Ideal": "hight"}, regex=True)
    df['cut'] = df['cut'].replace({"Good": "low", "Ideal": "hight"}, regex=True)

    df, y_data = df.drop(['cut'], axis=1), df['cut']

    Label = LabelEncoder()
    Label.fit(y_data)  # задаем столбец, который хотим преобразовать
    y_data = Label.transform(y_data)  # преобразуем и сохраняем в новую переменную

    df['cut'] = y_data
    save_file(df, file_path)
    logger.info("<< Finish preprocessing '" + file_path + "' >>\n")


def dp_main():
    logger.info("<<< Start data preprocessing >>>")
    train_path = "train/df_train_0.csv"
    df_prerpocessing(train_path)
    test_path = "test/df_test_0.csv"
    df_prerpocessing(test_path)
    logger.info("<<< Finish data preprocessing >>>\n")
