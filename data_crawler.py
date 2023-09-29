import threading
import requests
import time
import queue
import copy
import json

class data_crawler(threading.Thread):
    def __init__(self, data_queue: queue.Queue, supported_ticker: set, supported_period: dict, download_format: dict):
        threading.Thread.__init__(self)

        self.supported_ticker = supported_ticker
        self.supported_period = supported_period
        self.download_format = download_format
        
        # { ticker : download_format }
        self.downloaded_prices = {}
        
        self.limit_per_sec = 20
        self.running = True
        self.data_queue = data_queue

        self.now = int(time.time() * 1000)
        for idx, ticker in enumerate(self.supported_ticker):
            self.downloaded_prices.update({ ticker : copy.deepcopy(self.download_format) })

            for period in self.download_format.keys():
                self.downloaded_prices[ticker][period][0] = self.download_datas(ticker=ticker, 
                                                                                period=period, 
                                                                                startTime=self.now - self.supported_period[period][0] * self.supported_period[period][1],
                                                                                endTime=self.now)
                self.downloaded_prices[ticker][period][1] = self.now # 시간 갱신
                
                if self.data_queue is not None:
                    if self.is_all_downloaded(ticker):
                        self.data_queue.put((ticker, self.downloaded_prices.get(ticker))) # 가격 분석 업데이트가 필요할 때마다 price_analyzer 로 넘겨줌

            if idx < len(self.supported_ticker) - 1:
                time.sleep(len(self.download_format.keys()) * 1 / self.limit_per_sec)

            self.now = int(time.time() * 1000)

    def is_all_downloaded(self, ticker):
        for period in self.download_format.keys():
            if self.downloaded_prices[ticker][period][0] is None:
                return False
        return True
    
    def get_downloaded_datas(self): # for testing
        return self.downloaded_prices

    def download_datas(self, ticker, period, startTime, endTime):
        url = f"https://api.bitget.com/api/mix/v1/market/history-candles?symbol={ticker}&granularity={period}&startTime={startTime}&endTime={endTime}"

        while True:
            response = requests.get(url)
            if response.status_code == 200:
                print(f"Succeed to download datas [{ticker} / {period} / {startTime} / {endTime}]")
                break
            elif response.status_code == 429:
                print("Have to try downloading few later because of many requests in frequently 429")
                time.sleep(1 / self.limit_per_sec)
            else:
                print("Have to try downloading few later because of unexpected result ", response.status_code)
                time.sleep(1 / self.limit_per_sec)
        
        datas = json.loads(response.content)

        for idx, data in enumerate(datas):
            data = list(map(float, data))
            datas[idx] = data

        return datas

    def run(self):
        while self.running:
            self.now = int(time.time() * 1000)
            for idx, ticker in enumerate(self.supported_ticker):
                for period in self.download_format.keys():
                    if self.now - self.downloaded_prices[ticker][period][1] > self.supported_period[period][1]:
                        self.downloaded_prices[ticker][period][0] = self.download_datas(ticker=ticker, 
                                                                                        period=period, 
                                                                                        startTime=self.now - self.supported_period[period][0] * self.supported_period[period][1],
                                                                                        endTime=self.now)
                        self.downloaded_prices[ticker][period][1] = self.now # 시간 갱신

                        if self.data_queue is not None:
                            if self.is_all_downloaded(ticker):
                                self.data_queue.put((ticker, self.downloaded_prices.get(ticker))) # 가격 분석 업데이트가 필요할 때마다 price_analyzer 로 넘겨줌

                if idx < len(self.supported_ticker) - 1:
                    time.sleep(len(self.download_format.keys()) * 1 / self.limit_per_sec)

                self.now = int(time.time() * 1000)

    def stop(self):
        self.running = False