import unittest
import requests

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


class TestAPI(unittest.TestCase):

    def test_get_request(self):
        params = {"": ""}
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        response = requests.post(data=params)
        self.assertIn("Модель обучена", response.text)

    def test_model_preparation(self):
        response = client.get("/")
        self.assertEqual(response.status_code, 200)

        with open('../logs/log.log', 'r') as file:
            lines = file.readlines()

        self.assertIn("Модель обучена", lines)

    def test_model_result(self):
        response = client.get("/")
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

