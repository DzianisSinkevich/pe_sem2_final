import unittest
import requests
class TestAPI(unittest.TestCase):
  
    def test_get_request(self):


      url = "/"
      params = {""}
      response = requests.get(url)
      self.assertEqual(response.status_code, 200)
      self.assertEqual(response.json()["Message"], "Используй метод POST + данные датасета, чтобы получить результат")
      response = requests.post(url, params=params)
      self.assertIn("Модель обучена", response.text)
      


