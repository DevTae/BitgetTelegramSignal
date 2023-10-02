import queue
import logging

from data_crawler import data_crawler
from price_analyzer import price_analyzer

if __name__ == "__main__":

    # { period : [ limit, milliseconds ] } (limit <= 200)
    supported_period = { "1H" : [ 100, 3600000 ],
                         "6H" : [ 100, 21600000 ],
                         "1D" : [ 100, 86400000 ],
                         "1W" : [ 100, 604800000 ] }
        
    # { period : [ datas, lasttime ] }
    download_format = { "1H" : [ None, None ],
                        "6H" : [ None, None ],
                        "1D" : [ None, None ],
                        "1W" : [ None, None ] }  
    
    buy_sell_ticker = "BTCUSDT_UMCBL"
    supported_ticker = { buy_sell_ticker }

    logging.basicConfig(filename='main.log', level=logging.INFO)
    logger = logging.getLogger(__name__)
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)

    data_queue = queue.Queue() # data_crawler -> price_analyzer
    analyzer = price_analyzer(buy_sell_ticker, data_queue, download_format, logger)
    crawler = data_crawler(data_queue, supported_ticker, supported_period, download_format, logger)
    analyzer.start()
    crawler.start()

    while True:
        line = str(input("Enter the 'q' if you wanna quit .. : "))
        if line == "q":
            analyzer.stop()
            crawler.stop()
            exit()
    