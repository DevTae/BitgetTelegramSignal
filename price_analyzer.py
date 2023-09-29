import queue
import threading

class price_analyzer(threading.Thread):
    def __init__(self, data_queue: queue.Queue, download_format: dict):
        threading.Thread.__init__(self)
        self.data_queue = data_queue
        self.download_format = download_format
        self.running = True
    
    def run(self):
        while self.running:
            if self.data_queue.qsize() > 0:
                ticker, download_prices = self.data_queue.get()
                print(ticker)
                for period in self.download_format.keys():
                    print(period, download_prices[period][0])

    def stop(self):
        self.running = False