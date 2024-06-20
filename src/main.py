from fastapi import FastAPI
import uvicorn
from fastapi.responses import HTMLResponse

from data_creation import dc_main
from data_preprocessing import dp_main
from model_preparation import mp_main
from model_testing import mt_main

import logging
import os
import configparser
import requests

app = FastAPI()

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
logs_directory = os.path.join(parent_dir, 'logs')
file_handler = logging.FileHandler(os.path.join(parent_dir, 'logs/log.log'))
stream_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.addHandler(file_handler)

settings = configparser.ConfigParser()
settings.read('config/settings.ini')


def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


# Метод для запуска конвеера подготовки данных, модели, тестирования модели и вывода результата
@app.get("/")
def get_root():
    create_directory(logs_directory)
    logger.info("СТАРТ КОНВЕЕРА")
    dc_main()
    dp_main()
    mp_main()
    mt_main()
    logger.info("КОНВЕЕР ЗАВЕРШИЛ РАБОТУ\n")
    logger.info("START results writing")
    with open('results/results', 'r') as f:
        data = f.read()
    print(data)
    data = data.replace('\n', '<br />')
    responce_content = """
    <!DOCTYPE html>”
    <header>Model results.</header>
    <body><p>
    """ + data + """
    </p></body>
    """
    logger.info("тение результатов закончено")
    return HTMLResponse(content=responce_content, status_code=200)


def post():
    new_dct = []
    resp = requests.post('/', data=new_dct)
    file = open('config/settings.ini')
    file.write(resp.text)


def test_dummy():
    pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
