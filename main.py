import queue
import logging
import os

from data_crawler import data_crawler
from price_analyzer import price_analyzer
from telegram_bot import telegram_bot

if __name__ == "__main__":

    # { period : [ limit, milliseconds ] } (limit <= 200)
    supported_period = { "1H" : [ 100, 3600000 ],
                         "6H" : [ 100, 21600000 ],
                         "1D" : [ 100, 86400000 ],
                         "1W" : [ 100, 604800000 ] }
        
    # { period : [ datas, lasttime (마지막 캔들) ] }
    download_format = { "1H" : [ None, None ],
                        "6H" : [ None, None ],
                        "1D" : [ None, None ],
                        "1W" : [ None, None ] }
    
    # Ticker 설정
    buy_sell_ticker = "BTCUSDT_UMCBL" # 매수 및 매도 신호 대상 종목
    supported_ticker = { buy_sell_ticker } # 시장 분석 대상 종목들

    # 텔레그램 API 관련 정보 (보안 주의)
    telegram_api_key = "6436954798:AAGlQvHCA6mwYtr3AXZdY5klRCz0byNPeXU"
    telegram_chat_id = "-1001691110767"

    # log 설정
    logging.basicConfig(filename='main.log', level=logging.INFO)
    logger = logging.getLogger(__name__)
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)

    # imgs 폴더 설정
    folder_path = "imgs"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 프로그램 스레드 생성
    data_queue = queue.Queue() # data_crawler -> price_analyzer
    msg_queue = queue.Queue() # price_analyzer -> telegram_bot
    analyzer = price_analyzer(buy_sell_ticker, data_queue, msg_queue, download_format, logger)
    crawler = data_crawler(data_queue, supported_ticker, supported_period, download_format, logger)
    bot = telegram_bot(msg_queue, logger)
    analyzer.start()
    crawler.start()
    bot.start()

    while True:
        line = str(input("Enter the 'q' if you wanna quit .. : "))
        if line == "q":
            analyzer.stop()
            crawler.stop()
            bot.stop()
            exit()
