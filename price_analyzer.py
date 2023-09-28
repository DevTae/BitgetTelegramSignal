import queue
import threading

class price_analyzer(threading.Thread):
    def __init__(self, data_queue):
        threading.Thread.__init__(self)
        self.data_queue = data_queue
        self.running = True
    
    def run(self):
        while self.running:
            if self.data_queue.qsize() > 0:
                ticker, period, datas = self.data_queue.get()
                print("[analyzer]", ticker, period, datas)

    def stop(self):
        self.running = False