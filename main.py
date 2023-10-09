import queue
import logging
import os
import time

from collections import OrderedDict

from data_crawler import data_crawler
from price_analyzer import price_analyzer
from telegram_bot import telegram_bot

if __name__ == "__main__":
    
    # { period : [ datas, lasttime (마지막 캔들), download_cycle ] }
    download_format = OrderedDict({ "1D" : [ None, None, 86400000 ],
                                    "6H" : [ None, None, 21600000 ],
                                    "1H" : [ None, None, 3600000 ],
                                    "15m" : [ None, None, 900000 ],
                                    "3m" : [ None, None, 180000 ],
                                    "1m" : [ None, None, 60000 ] })
    
    download_candle = 100 # limit <= 200
    
    # Ticker 설정
    buy_sell_ticker = "BTCUSDT_UMCBL" # 매수 및 매도 신호 대상 종목
    supported_ticker = { buy_sell_ticker } # 시장 분석 대상 종목들

    # 텔레그램 API 관련 정보 (보안 주의)
    telegram_api_key = "6436954798:AAGlQvHCA6mwYtr3AXZdY5klRCz0byNPeXU"
    telegram_chat_id = "-1001691110767"

    # imgs 폴더 설정
    imgs_folder_path = "imgs"
    if not os.path.exists(imgs_folder_path):
        os.makedirs(imgs_folder_path)

    # logs 폴더 설정
    logs_folder_path = "logs"
    if not os.path.exists(logs_folder_path):
        os.makedirs(logs_folder_path)

    # log 설정
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S")  # 원하는 포맷으로 지정
    logging.basicConfig(filename=os.path.join(logs_folder_path, f'{current_time}.log'), level=logging.INFO)
    logger = logging.getLogger(__name__)
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)
    
    # 프로그램 스레드 생성
    data_queue = queue.Queue() # data_crawler -> price_analyzer
    msg_queue = queue.Queue() # price_analyzer -> telegram_bot
    analyzer = price_analyzer(buy_sell_ticker, data_queue, msg_queue, download_format, logger)
    crawler = data_crawler(data_queue, supported_ticker, download_format, download_candle, logger)
    bot = telegram_bot(msg_queue, telegram_api_key, telegram_chat_id, logger)
    analyzer.start()
    crawler.start()
    bot.start()

    while True:
        line = str(input("Enter the 'q' if you wanna quit ..\n"))
        if line == "q":
            analyzer.stop()
            crawler.stop()
            bot.stop()
            exit()
