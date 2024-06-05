import requests
import configparser
from main import get_root


def get():
    request = requests.get('/')
    if request.status_code != 200:
        print("Используй метод POST + данные датасета, чтобы получить результат")


def post():
    new_dct = []
    resp = requests.post('/', data=new_dct)
    config = configparser.ConfigParser()
    config.read('config/settings.ini')
    config = resp
    with open('config/settings.ini', 'w') as configfile:
        config.write(configfile)


get()
post()
get_root()
