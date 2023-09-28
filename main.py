import queue

from data_crawler import data_crawler
from price_analyzer import price_analyzer

if __name__ == "__main__":
    data_queue = queue.Queue()
    analyzer = price_analyzer(data_queue)
    datas = data_crawler(data_queue)
    analyzer.start()
    datas.start()

    while True:
        line = str(input("Enter the 'q' if you wanna quit .. : "))
        if line == "q":
            analyzer.stop()
            datas.stop()
            exit()
    