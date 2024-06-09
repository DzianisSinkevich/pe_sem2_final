from sklearn.preprocessing import LabelEncoder  # Импортируем LabelEncoder от scikit-learn
import pandas as pd  # Библиотека Pandas для работы с табличными данными
import warnings
import logging
import os

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

stream_handler = logging.StreamHandler()
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
file_handler = logging.FileHandler(os.path.join(parent_dir, 'logs/log.log'))
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(stream_handler)
logger.addHandler(file_handler)


def read_file(file_path):
    try:
        df = pd.read_csv(file_path)
        logger.info("Файл " + file_path + " прочтен успешно.")
        return df
    except IOError:
        logger.critical("Ошибка чтения файла '" + file_path + "'.")


def save_file(df, file_path):
    try:
        df.to_csv(file_path, index=False)
        logger.info("Файл " + file_path + " создан успешно.")
    except IOError:
        logger.critical("Ошибка создания файла " + file_path + " .")


def df_prerpocessing(file_path):
    logger.info("<< Обработка '" + file_path + "' начата >>")
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
    logger.info("<< Обработка '" + file_path + "' закончена >>\n")


def dp_main():
    logger.info("<<< Обработка данных начата >>>")
    train_path = "train/df_train_0.csv"
    df_prerpocessing(train_path)
    test_path = "test/df_test_0.csv"
    df_prerpocessing(test_path)
    logger.info("<<< Обработка данных закончена >>>\n")
