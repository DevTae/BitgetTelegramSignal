import threading
import requests
import time
import queue
import copy
import json
import logging

class data_crawler(threading.Thread):
    def __init__(self, data_queue: queue.Queue, supported_ticker: set, download_format: dict, download_candle: int, logger: logging):
        threading.Thread.__init__(self)

        self.supported_ticker = supported_ticker
        self.download_format = download_format
        self.download_candle = download_candle
        self.logger = logger

        self.limit_per_sec = 20
        self.running = True
        self.data_queue = data_queue
        self.debug = False
        
        # { ticker : download_format }
        self.downloaded_prices = {}
        
        for idx, ticker in enumerate(self.supported_ticker):
            self.downloaded_prices.update({ ticker : copy.deepcopy(self.download_format) })

            for period in self.download_format.keys():
                self.now = int(time.time() * 1000)
                downloaded_datas = self.download_datas(ticker=ticker, 
                                                       period=period, 
                                                       startTime=self.now - self.download_candle * self.downloaded_prices[ticker][period][2],
                                                       endTime=self.now)
                self.downloaded_prices[ticker][period][0] = downloaded_datas # 모든 봉 데이터에 대하여 넘겨줌
                self.downloaded_prices[ticker][period][1] = int(downloaded_datas[-1][0]) # 시간 갱신 (마지막 타임스탬프)

                if self.debug: # debug mode 일 때 다음 다운로드를 바로 다운로드 받도록 timestamp 수정
                    self.logger.info("[log] self.debug is True")
                    self.downloaded_prices[ticker][period][1] = int(downloaded_datas[-1][0]) - self.download_candle * self.downloaded_prices[ticker][period][2] * self.downloaded_prices[ticker][period][3]
                    time.sleep(2)

                if self.data_queue is not None:
                    #if self.is_all_downloaded(ticker):
                    self.data_queue.put((self.downloaded_prices.get(ticker), ticker, period)) # 가격 분석 업데이트가 필요할 때마다 price_analyzer 로 넘겨줌

            if idx < len(self.supported_ticker) - 1:
                time.sleep(len(self.download_format.keys()) * 1 / self.limit_per_sec)

    def is_all_downloaded(self, ticker):
        for period in self.download_format.keys():
            if self.downloaded_prices[ticker][period][0] is None:
                return False
        return True
    
    def get_downloaded_datas(self): # for testing
        return self.downloaded_prices

    def download_datas(self, ticker, period, startTime, endTime):
        #url = f"https://api.bitget.com/api/mix/v1/market/history-candles?symbol={ticker}&granularity={period}&startTime={startTime}&endTime={endTime}" # for backtesting
        url = f"https://api.bitget.com/api/mix/v1/market/candles?symbol={ticker}&granularity={period}&startTime={startTime}&endTime={endTime}"

        while self.running:
            response = requests.get(url)
            if response.status_code == 200:
                if self.logger != None:
                    self.logger.info("Succeed to download datas [" + str(ticker) + "/" + str(period) + "/" + str(startTime) + "/" + str(endTime) + "]")
                break
            elif response.status_code == 429:
                if self.logger != None:
                    self.logger.info("Have to try downloading few later because of many requests in frequently 429")
                time.sleep(1 / self.limit_per_sec)
            else:
                if self.logger != None:
                    self.logger.info("Have to try downloading few later because of unexpected result " + str(response.status_code))
                time.sleep(1 / self.limit_per_sec)
        
        datas = json.loads(response.content)

        for idx, data in enumerate(datas):
            data = list(map(float, data))
            datas[idx] = data

        return datas

    def run(self):
        while self.running:
            for idx, ticker in enumerate(self.supported_ticker):
                for period in self.download_format.keys():
                    self.now = int(time.time() * 1000)
                    if self.now - self.downloaded_prices[ticker][period][1] > self.downloaded_prices[ticker][period][2] * self.downloaded_prices[ticker][period][3]:
                        downloaded_datas = self.download_datas(ticker=ticker, 
                                                               period=period, 
                                                               startTime=self.now - self.download_candle * self.downloaded_prices[ticker][period][2],
                                                               endTime=self.now)
                        
                        # 다운로드 데이터의 시간에 문제가 생긴 경우
                        if self.now - downloaded_datas[-1][0] > self.downloaded_prices[ticker][period][2]:
                            self.logger.info("[log] The timestamp of last data is anomaly. it would retry. " \
                                              + str(ticker) + " " + str(period) + " " + str(self.now) + " " + str(downloaded_datas[-1][0]))
                            time.sleep(1) # 1 초 sleep
                            continue

                        self.downloaded_prices[ticker][period][0] = downloaded_datas # 모든 봉 데이터에 대하여 넘겨줌
                        self.downloaded_prices[ticker][period][1] = int(downloaded_datas[-1][0]) # 시간 갱신 (마지막 타임스탬프)

                        if self.data_queue is not None:
                            #if self.is_all_downloaded(ticker):
                            self.data_queue.put((self.downloaded_prices.get(ticker), ticker, period)) # 가격 분석 업데이트가 필요할 때마다 price_analyzer 로 넘겨줌

                if idx < len(self.supported_ticker) - 1:
                    time.sleep(len(self.download_format.keys()) * 1 / self.limit_per_sec)

    def stop(self):
        self.running = False
