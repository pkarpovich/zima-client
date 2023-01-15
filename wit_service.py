import requests

WIT_VERSION = "20230114"


class WitService(object):
    def __init__(self, access_token):
        self.headers = {
            "Authorization": f"Bearer {access_token}"
        }

    def predict(self, text):
        print("Predicting...")

        params = {
            "v": WIT_VERSION,
            "q": text,
        }

        x = requests.get("https://api.wit.ai/message", params, headers=self.headers)
        json = x.json()

        return json['intents']
