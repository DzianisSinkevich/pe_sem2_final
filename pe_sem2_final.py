import requests


def get():
    request = requests.get('/')
    if request.status_code != 200:
        print("Используй метод POST + данные датасета, чтобы получить результат")


def post():
    new_dct = []
    resp = requests.post('/', data=new_dct)
    file = open('config/settings.ini')
    file.write(resp.text)

post()
get()
