import requests
import queue
import threading
import logging

class telegram_bot(threading.Thread):
    def __init__(self, msg_queue: queue.Queue, logger):
        threading.Thread.__init__(self)
        self.msg_queue = msg_queue
        self.logger = logger
        self.running = True

    def run(self):
        while self.running:
            pass

    def stop(self):
        self.running = False