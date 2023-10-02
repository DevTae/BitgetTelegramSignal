import requests
import queue
import threading
import logging

class telegram_bot(threading.Thread):
    def __init__(self, msg_queue: queue.Queue, telegram_api_key: str, telegram_chat_id: str, logger):
        threading.Thread.__init__(self)
        self.msg_queue = msg_queue
        self.telegram_api_key = telegram_api_key
        self.telegram_chat_id = telegram_chat_id
        self.logger = logger
        self.running = True
        self.url = f"https://api.telegram.org/bot{self.telegram_api_key}/METHOD"

    def run(self):
        while self.running:
            if self.msg_queue.qsize() != 0:
                result = self.msg_queue.get()
                
                data = {}
                files = {}
                data["chat_id"] = self.telegram_chat_id

                if 'photo' in result.keys():
                    if "text" in data.keys():
                        del data["text"]
                    files.update({ "photo" : open(result["photo"], "rb") })
                    response = requests.post(self.url.replace("METHOD", "sendPhoto"), data=data, files=files)
                    self.logger.info("[log] telegram photo response " + str(response.status_code))

                if 'text' in result.keys():
                    data["text"] = result['text']
                    response = requests.post(self.url.replace("METHOD", "sendMessage"), data=data)
                    self.logger.info("[log] telegram text response " + str(response.status_code))

                
    def stop(self):
        self.running = False