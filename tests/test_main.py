import unittest
import requests


class TestAPI(unittest.TestCase):

    def test_get_request(self):
        url = "/"
        params = {"": ""}
        response = requests.get(url)
        self.assertEqual(response.status_code, 200)
        response = requests.post(url, data=params)
        self.assertIn("Модель обучена", response.text)

    def test_model_preparation(self):
        url = "/"
        response = requests.get(url)
        self.assertEqual(response.status_code, 200)

        with open('../logs/log.log', 'r') as file:
            lines = file.readlines()

        self.assertIn("Модель обучена", lines)

    def test_model_result(self):
        url = "/"
        response = requests.get(url)
        self.assertEqual(response.status_code, 200)

        with open('../logs/log.log', 'r') as file:
            lines = file.readlines()

        res_f = 0

        for i in lines:
            if "f1 on test data:" in i:
                res = i[17:]
                res_f = float(res)
                break

        self.assertGreater(res_f, 0.5, 'f1 модели ниже 0.5')

