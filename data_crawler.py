import threading
import requests
import time
import queue

class data_crawler(threading.Thread):
    def __init__(self, data_queue: queue.Queue):
        threading.Thread.__init__(self)

        # { period : [ limit, milliseconds ] } (limit <= 200)
        self.supported_period = { "5m" : [ 100, 300000 ],
                                  "15m" : [ 100, 900000 ],
                                  "1H" : [ 100, 3600000 ] ,
                                  "6H" : [ 100, 21600000 ],
                                  "1D" : [ 100, 86400000 ] }
        # { period : [ datas, lasttime ] }
        self.downloaded_datas = { "5m" : [ None, None ],
                                  "15m" : [ None, None ],
                                  "1H" : [ None, None ],
                                  "6H" : [ None, None ],
                                  "1D" : [ None, None ] }
        
        self.supported_ticker = { "BTCUSDT_UMCBL" } # 더 많아지면 limit_per_sec 기반 로직 짤 것을 권고
        self.limit_per_sec = 20
        self.now = int(time.time() * 1000)
        self.running = True
        self.data_queue = data_queue

        for key in self.downloaded_datas.keys():
            self.now = int(time.time() * 1000)
            for ticker in self.supported_ticker:
                self.downloaded_datas[key][0] = self.download_datas(ticker=ticker, 
                                                                    period=key, 
                                                                    startTime=self.now - self.supported_period[key][0] * self.supported_period[key][1],
                                                                    endTime=self.now)
                self.downloaded_datas[key][1] = self.now # 시간 갱신
                self.data_queue.put((ticker, key, self.downloaded_datas[key][0])) # price_analyzer 로 넘겨줌
                
    def get_supported_period(self):
        return self.supported_period.keys()

    def download_datas(self, ticker, period, startTime, endTime):
        url = f"https://api.bitget.com/api/mix/v1/market/history-candles?symbol={ticker}&granularity={period}&startTime={startTime}&endTime={endTime}"

        while True:
            response = requests.get(url)
            if response.status_code == 200:
                print(f"Succeed to download datas [{ticker} / {period} / {startTime} / {endTime}]")
                break
            elif response.status_code == 429:
                print("Have to try downloading few later because of many requests in frequently 429")
                time.sleep(0.5)
            else:
                print("Have to try downloading few later because of unexpected result ", response.status_code)
                time.sleep(0.1)
                
        return response.content

    def run(self):
        while self.running:
            for key in self.downloaded_datas.keys():
                self.now = int(time.time() * 1000)
                for ticker in self.supported_ticker:
                    if self.now - self.downloaded_datas[key][1] > self.supported_period[key][1]:
                        self.downloaded_datas[key][0] = self.download_datas(ticker=ticker, 
                                                                            period=key, 
                                                                            startTime=self.now - self.supported_period[key][0] * self.supported_period[key][1],
                                                                            endTime=self.now)
                        self.downloaded_datas[key][1] = self.now # 시간 갱신
                        self.data_queue.put((ticker, key, self.downloaded_datas[key][0])) # price_analyzer 로 넘겨줌
            time.sleep(0.1)

    def stop(self):
        self.running = False