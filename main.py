import queue

from data_crawler import data_crawler
from price_analyzer import price_analyzer

if __name__ == "__main__":

    # { period : [ limit, milliseconds ] } (limit <= 200)
    supported_period = { "1m" : [ 100,  60000 ],
                         "5m" : [ 100, 300000 ],
                         "15m" : [ 100, 900000 ],
                         "1H" : [ 100, 3600000 ],
                         "6H" : [ 100, 21600000 ] }
        
    # { period : [ datas, lasttime ] }
    download_format = { "1m" : [ None, None ],
                        "5m" : [ None, None ],
                        "15m" : [ None, None ],
                        "1H" : [ None, None ],
                        "6H" : [ None, None ] }  
    
    supported_ticker = { "BTCUSDT_UMCBL" }

    data_queue = queue.Queue() # data_crawler -> price_analyzer
    analyzer = price_analyzer(data_queue, download_format)
    datas = data_crawler(data_queue, supported_ticker, supported_period, download_format)
    analyzer.start()
    datas.start()

    while True:
        line = str(input("Enter the 'q' if you wanna quit .. : "))
        if line == "q":
            analyzer.stop()
            datas.stop()
            exit()
    