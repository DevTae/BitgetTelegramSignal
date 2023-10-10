import queue
import threading
import copy
import pandas as pd
import numpy as np
import os
import time
import logging
from scipy.stats import gaussian_kde

import matplotlib
matplotlib.use('agg') # 백엔드 사용 / 초반엔 5 분 걸리고, 이후에는 1 분 걸림. (초반 로딩이 오래 걸림)

import matplotlib.pyplot as plt


class price_analyzer(threading.Thread):
    def __init__(self, buy_sell_ticker: str, data_queue: queue.Queue, msg_queue: queue.Queue, download_format: dict, logger: logging):
        threading.Thread.__init__(self)
        self.buy_sell_ticker = buy_sell_ticker
        self.data_queue = data_queue
        self.msg_queue = msg_queue
        self.download_format = download_format # { period : [ datas, lasttime (마지막 캔들), download_cycle ] }
        self.logger = logger
        self.datas = {}

        # for caching
        self.datas_cache = {}
        self.vp_kdes_diffs_cache = {}
        self.now_kdes_diffs_cache = {}
        self.resistance_lines_cache = {}
        self.support_lines_cache = {}

        # buy_sell_ticker initializing
        self.datas_cache[buy_sell_ticker] = {}
        self.vp_kdes_diffs_cache[buy_sell_ticker] = {}
        self.now_kdes_diffs_cache[buy_sell_ticker] = {}
        self.resistance_lines_cache[buy_sell_ticker] = {}
        self.support_lines_cache[buy_sell_ticker] = {}

        self.running = True
        self.drawing = False
        self.drawing_first = True # 한 번의 draw_graph 함수 테스트 진행
        self.drawing_lock = threading.Lock()


    # 한 개의 period 에 대한 보조지표 및 매물대 지표 계산하는 함수 (datas : array -> datas_ : pandas.DataFrame)
    def calculate_indicators(self, datas: list, period: str):
        current_time = time.localtime()
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
        self.logger.info("[log] calculate_indicators function called : " + str(formatted_time) + " " + str(period))

        datas_ = pd.DataFrame(datas, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'q_volume'])
    
        """
        # 매물대 계산을 위하여 거래량 multiple 방식 구현
        volume_profiles = []

        # 매물대 계산을 위하여 period 에 따라 달라지는 거래량 크기에 따라 연산량 최소화
        if 'W' in period:
            ratio_of_removal = 0.0008
        elif 'D' in period:
            ratio_of_removal = 0.004
        elif 'H' in period:
            ratio_of_removal = 0.1
        else: # case 'm'
            ratio_of_removal = 1
        """
        
        for i in range(len(datas_['time'])):
            volume = datas_['volume'][i]
            open_price = datas_['open'][i]
            high_price = datas_['high'][i]
            low_price = datas_['low'][i]
            close_price = datas_['close'][i]
            center_price = (open_price + high_price + low_price + close_price) / 4
            
            """
            total_div = 10

            try:
                div_size = (high_price - low_price) / total_div
                index_pivot = (center_price - low_price) // div_size
                index_pivot = max(min(index_pivot, total_div - 1), 1) # 시고저종 가격을 바탕으로 중심값 선정
                multiple_of_volume = np.array(list(np.linspace(1, 0, int(index_pivot))) \
                                            + list(np.linspace(0, 1, int(total_div - index_pivot + 1))[1:]))
                multiple_of_volume /= sum(multiple_of_volume)

                for j in range(len(multiple_of_volume)):
                    for _ in range(int(volume * multiple_of_volume[j] * ratio_of_removal)):
                        volume_profiles.append(low_price + j * div_size)
            except: # 에러 발생 시 
                self.logger.info("[log] calculate_indicators function handles exception : " + str(formatted_time) + " " + str(period))
                for _ in range(int(volume * ratio_of_removal)):
                    volume_profiles.append(center_price)
            """
        
        """
        # 매물대 지표 KDE 적용
        level = 100
        x = np.linspace(min(volume_profiles) - 1, max(volume_profiles) + 1, level)
        vp_kde = gaussian_kde(volume_profiles)
        
        # 매물대 지표 KDE 정규화 적용
        vp_kdes = []
        
        for i in range(len(x)):
            vp_kdes.append(vp_kde(x[i]))
        
        vp_kdes = np.array(vp_kdes)
        vp_kdes -= np.mean(vp_kdes)
        vp_kdes /= np.std(vp_kdes)
        
        # 매물대 지표 KDE 미분 적용
        vp_kdes_diff = []
    
        for i in range(1, len(x)):
            vp_kdes_diff.append(vp_kde(x[i]) - vp_kde(x[i-1]))
        
        vp_kdes_diff = np.array(vp_kdes_diff)
        vp_kdes_diff -= np.mean(vp_kdes_diff)
        vp_kdes_diff /= np.std(vp_kdes_diff)
        """

        # RSI 상대강도지수 계산
        datas_['price_diff'] = datas_['close'].diff()
        datas_['gain'] = datas_['price_diff'].apply(lambda x: x if x > 0 else 0)
        datas_['loss'] = datas_['price_diff'].apply(lambda x: -x if x < 0 else 0)
        datas_['au'] = datas_['gain'].ewm(com=14-1, min_periods=14).mean()
        datas_['ad'] = datas_['loss'].ewm(com=14-1, min_periods=14).mean()
        datas_['rs'] = datas_['au'] / datas_['ad']
        datas_['rsi14'] = 100 - (100 / (1 + datas_['rs']))
        datas_['rsi14_ma'] = datas_['rsi14'].rolling(window=14).mean()

        # 지수 이동평균선 계산
        datas_['ema5'] = datas_['close'].ewm(span=5).mean()
        datas_['ema20'] = datas_['close'].ewm(span=20).mean()
        datas_['ema60'] = datas_['close'].ewm(span=60).mean()
        
        # MACD 지표 계산
        datas_['ema12'] = datas_['close'].ewm(span=12).mean()
        datas_['ema26'] = datas_['close'].ewm(span=26).mean()
        datas_['macd'] = datas_['ema12'] - datas_['ema26']
        datas_['macd_ma'] = datas_['macd'].rolling(window=9).mean()
        datas_['macd_oscillator'] = datas_['macd'] - datas_['macd_ma']
        
        # MACD Osillator 골든크로스 확률 계산 (대략적인)
        datas_['macd_min'] = datas_['macd_oscillator'].rolling(window=9).min()
        datas_.loc[datas_['macd_min'] > 0, 'macd_min'] = 0
        datas_['macd_max'] = datas_['macd_oscillator'].rolling(window=9).max()
        datas_.loc[datas_['macd_max'] < 0, 'macd_max'] = 0
        
        datas_['macd_gc_prob'] = (datas_['macd_oscillator'] - datas_['macd_min']) \
                               / (datas_['macd_max'] - datas_['macd_min'])
        datas_['macd_dc_prob'] = (datas_['macd_max'] - datas_['macd_oscillator']) \
                               / (datas_['macd_max'] - datas_['macd_min'])

        # 이후 calculate_indicators_last_candle 을 위하여 주석 처리 진행
        # 필요 없는 열 삭제
        #datas_.drop(['price_diff', 'gain', 'loss', 'au', 'ad', 'rs', 'macd_oscillator'], axis=1, inplace=True)
        #datas_.drop(['macd_min', 'macd_max'], axis=1, inplace=True)

        ##return datas_, x[1:], vp_kdes_diff
        return datas_, None, None

    # 마지막 캔들에 대해서만 업데이트 진행하는 함수 (datas : pd.DataFrame -> datas : pd.DataFrame)
    def calculate_indicators_last_candle(self, datas: pd.DataFrame, period: str, new_datas: pd.DataFrame, new_period: str):
        current_time = time.localtime()
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
        self.logger.info("[log] calculate_indicators_last_candle function called : " + str(formatted_time) + " " + str(period) + " " + str(new_period))

        # 마지막 타임프레임에 대한 데이터 반영
        datas['high'].iloc[-1] = max(datas['high'].iloc[-1], new_datas['high'].iloc[-1])
        datas['low'].iloc[-1] = min(datas['low'].iloc[-1], new_datas['low'].iloc[-1])
        datas['close'].iloc[-1] = new_datas['close'].iloc[-1]
        datas['volume'].iloc[-1] += new_datas['volume'].iloc[-1]

        # RSI 상대강도지수 계산
        datas['price_diff'].iloc[-1] = datas['close'].iloc[-1] - datas['close'].iloc[-2]
        datas['gain'].iloc[-1] = datas['price_diff'].iloc[-1] if datas['price_diff'].iloc[-1] > 0 else 0
        datas['loss'].iloc[-1] = datas['price_diff'].iloc[-1] if datas['price_diff'].iloc[-1] < 0 else 0

        datas['au'].iloc[-1] = datas['gain'].ewm(com=14-1, min_periods=14).mean().iloc[-1]
        datas['ad'].iloc[-1] = datas['loss'].ewm(com=14-1, min_periods=14).mean().iloc[-1]

        datas['rs'].iloc[-1] = datas['au'].iloc[-1] / datas['ad'].iloc[-1]
        datas['rsi14'].iloc[-1] = 100 - (100 / (1 + datas['rs'].iloc[-1]))
        datas['rsi14_ma'].iloc[-1] = datas['rsi14'].rolling(window=14).mean().iloc[-1]

        # 지수 이동평균선 계산
        datas['ema5'].iloc[-1] = datas['close'].ewm(span=5).mean().iloc[-1]
        datas['ema20'].iloc[-1] = datas['close'].ewm(span=20).mean().iloc[-1]
        datas['ema60'].iloc[-1] = datas['close'].ewm(span=60).mean().iloc[-1]
        
        # MACD 지표 계산
        datas['ema12'].iloc[-1] = datas['close'].ewm(span=12).mean().iloc[-1]
        datas['ema26'].iloc[-1] = datas['close'].ewm(span=26).mean().iloc[-1]
        datas['macd'].iloc[-1] = datas['ema12'].iloc[-1] - datas['ema26'].iloc[-1]
        datas['macd_ma'].iloc[-1] = datas['macd'].rolling(window=9).mean().iloc[-1]
        datas['macd_oscillator'].iloc[-1] = datas['macd'].iloc[-1] - datas['macd_ma'].iloc[-1]

        # MACD Osillator 골든크로스 확률 계산 (대략적인)
        datas['macd_min'].iloc[-1] = datas['macd_oscillator'].rolling(window=12).min().iloc[-1]
        datas['macd_min'].iloc[-1] = 0 if datas['macd_min'].iloc[-1] > 0 else datas['macd_min'].iloc[-1]
        datas['macd_max'].iloc[-1] = datas['macd_oscillator'].rolling(window=12).max().iloc[-1]
        datas['macd_max'].iloc[-1] = 0 if datas['macd_max'].iloc[-1] < 0 else datas['macd_max'].iloc[-1]
        
        datas['macd_gc_prob'] = (datas['macd_oscillator'] - datas['macd_min']) \
                               / (datas['macd_max'] - datas['macd_min'])
        datas['macd_dc_prob'] = (datas['macd_max'] - datas['macd_oscillator']) \
                               / (datas['macd_max'] - datas['macd_min'])

        return datas
        

    # 한 Ticker 에 대한 개별 period 분석을 진행하는 함수
    def fractal_analyze(self, datas: dict, ticker: str, period: str, debug: bool = False):
        current_time = time.localtime()
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
        self.logger.info("[log] fractal_analyze function called : " + str(formatted_time) + " " + str(period))
        
        """
        # 매물대 분석
        now_price = self.datas_cache[ticker][period]['close'].iloc[-1]
        now_kdes_diff_, resistance_lines_, support_lines_ = self.analyze_volume_profile(self.vp_kdes_diffs_cache[ticker], now_price, period)
        
        # cache 에 저항선 및 지지선 데이터 저장하기
        self.resistance_lines_cache[ticker].update({period : resistance_lines_}) # array
        self.support_lines_cache[ticker].update({period : support_lines_}) # array
        self.now_kdes_diffs_cache[ticker].update({period : now_kdes_diff_ }) # float
        """
        
        # 보조지표 분석, 조건 감시 및 알림 (fractal period set)
        target_periods = [("1m", "15m", "1H"), # 15분봉 매매법
                          ("1m", "1H", "6H"), # 1시간봉 매매법
                          ("1m", "6H", "1D") # 6시간봉 매매법
                          ] # (update_period, short_period, longer_period)

        for idx, (update_period, short_period, longer_period) in enumerate(target_periods):
            if update_period == period:
                # period (short_period, longer_period) 에 맞는 저항선 및 지지선 데이터 선정
                resistance_lines = []
                support_lines = []
                """
                resistance_lines += self.resistance_lines_cache[ticker][longer_period]
                support_lines += self.support_lines_cache[ticker][longer_period]
                resistance_lines += self.resistance_lines_cache[ticker][short_period]
                support_lines += self.support_lines_cache[ticker][short_period]
                """

                # 해당 target_period 에 대하여 보조지표 분석 진행
                target_period = target_periods[idx]
                self.fractal_analyze_indicator_macd(True, self.datas_cache[ticker], ticker, target_period, resistance_lines, support_lines) # long
                self.fractal_analyze_indicator_macd(False, self.datas_cache[ticker], ticker, target_period, resistance_lines, support_lines) # short

        self.logger.info("[log] fractal_analyze function done.")
            

    # it is not used because of a lot of needed time
    def analyze_volume_profile(self, kdes_diffs: dict, now_price: float, volume_profile_period: str):
        current_time = time.localtime()
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
        self.logger.info("[log] analyze_volume_profile function called : " + str(formatted_time) + " " + str(volume_profile_period))

        resistance_lines = []
        support_lines = []
        now_kdes_diff = None

        for idx, price in enumerate(kdes_diffs[volume_profile_period][0]):
            if now_price <= price:
                now_kdes_diff = np.round(kdes_diffs[volume_profile_period][1][idx], decimals=2)
                self.logger.info("--" + str(volume_profile_period) + "--")
                self.logger.info("[log] The result of analyze of volume profile : " + str(now_kdes_diff))
                if now_kdes_diff < 0:
                    # 목표가 계산
                    for i in range(len(kdes_diffs[volume_profile_period][1])):
                        if idx+i >= len(kdes_diffs[volume_profile_period][1]):
                            self.logger.info("[log] Resistance line can not be set : disable to analysis upper volume profile")
                            break
                        
                        if kdes_diffs[volume_profile_period][1][idx+i] >= 0:
                            resistance_line = int(kdes_diffs[volume_profile_period][0][idx+i])
                            self.logger.info("[log] Resistance Line : " + str(resistance_line))
                            resistance_lines.append(resistance_line)
                            break
                    # 손절가 계산
                    for i in range(len(kdes_diffs[volume_profile_period][1])):
                        if idx-i >= len(kdes_diffs[volume_profile_period][1]):
                            self.logger.info("[log] Support line can not be set : disable to analysis lower volume profile")
                            break
                        
                        if kdes_diffs[volume_profile_period][1][idx-i] >= 0:
                            support_line = int(kdes_diffs[volume_profile_period][0][idx-i])
                            self.logger.info("[log] Support line : " + str(support_line))
                            support_lines.append(support_line)
                            break
                self.logger.info("------")
                break

        return now_kdes_diff, resistance_lines, support_lines
    
    
    def fractal_analyze_indicator_macd(self, is_long: bool, datas_: pd.DataFrame, ticker: str, target_period, resistance_lines: list, support_lines: list):
        direction_value = 1 if is_long else -1
        direction_info = "long" if is_long else "short"
        signal_info_kr = "매수 신호" if is_long else "매도 신호"
        signal_info_en = "long signal" if is_long else "short signal"
        signal_info_en_simply = "LONG" if is_long else "SHORT"
        
        current_time = time.localtime()
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
        self.logger.info("[log] fractal_analyze_indicator_macd function called : " + str(formatted_time) + " " + str(direction_info))

        # target_period 해석
        _, short_period, longer_period = target_period

        # 보조지표 매수 관점 분석 (long)
        self.logger.info("--" + str(short_period) + "," + str(longer_period) + "--" + str(direction_info) + "-insight--")

        # 현재가 가져오기
        now_price = datas_[short_period]['close'].iloc[-1]

        # macd 데이터 가져오기
        macd_longer_1 = datas_[longer_period]['macd_oscillator'].iloc[-1]
        macd_short_1 = datas_[short_period]['macd'].iloc[-1]

        macd_oscillator_short_1 = datas_[short_period]['macd_oscillator'].iloc[-1]
        macd_oscillator_short_2 = datas_[short_period]['macd_oscillator'].iloc[-2]
        macd_oscillator_short_3 = datas_[short_period]['macd_oscillator'].iloc[-3]
        macd_oscillator_short_4 = datas_[short_period]['macd_oscillator'].iloc[-4]
        macd_oscillator_short_5 = datas_[short_period]['macd_oscillator'].iloc[-5]
        macd_oscillator_short_6 = datas_[short_period]['macd_oscillator'].iloc[-6]

        if macd_longer_1 * direction_value > 0 and \
           macd_short_1 * direction_value < 0 and \
           (macd_oscillator_short_2 - macd_oscillator_short_1) * direction_value < 0 and \
           (macd_oscillator_short_2 - macd_oscillator_short_3) * direction_value < 0 and \
           (macd_oscillator_short_2 - macd_oscillator_short_4) * direction_value < 0 and \
           (macd_oscillator_short_2 - macd_oscillator_short_5) * direction_value < 0 and \
           (macd_oscillator_short_2 - macd_oscillator_short_6) * direction_value < 0 or self.drawing_first:
            # 알림 메세지 준비 - 현재가 및 손절가
            msg = f"[{signal_info_en_simply} Signal] {short_period}봉 기준 MACD 지표의 {signal_info_kr}가 발생하였습니다."
            if self.drawing_first:
                self.drawing_first = False
                msg += "\n\t해당 메세지는 테스트용 메세지입니다."
                self.logger.info("[log] debug is True")
            msg += f"\n\t현재가 : {now_price}"
            msg += f"\n\t손절가 : 미정"
            msg += "\n"

            # 알림 메세지 준비 - 매물대 미분 수치 제공
            longer_now_kdes_diff = self.now_kdes_diffs_cache[ticker].get(longer_period)
            short_now_kdes_diff = self.now_kdes_diffs_cache[ticker].get(short_period)
            if longer_now_kdes_diff != None:
                msg += f"\n\t\t{longer_period} 매물대 지표 : {longer_now_kdes_diff}"
            if short_now_kdes_diff != None:
                msg += f"\n\t\t{short_period} 매물대 지표 : {short_now_kdes_diff}"
                msg += "\n"

            # 알림 메세지 준비 - 지지/저항 가격 수치 제공
            if len(support_lines) != 0:
                msg += f"\n\t\t예상 지지선 : {max(support_lines)}"
            if len(resistance_lines) != 0:
                msg += f"\n\t\t예상 저항선 : {min(resistance_lines)}"
            
            # 스레드를 바탕으로 그래프를 그려서 메세지 내용과 함께 텔레그램으로 전송
            thread = threading.Thread(target=self.send_to_telegram_bot, args=(msg, datas_, short_period, longer_period, resistance_lines, support_lines))
            thread.start()

            self.logger.info("[log] MACD " + signal_info_en + " occured.")
        else:
            self.logger.info("[log] MACD " + signal_info_en + " doesn't occured.")
        self.logger.info("------")


    def send_to_telegram_bot(self, msg: str, datas_: pd.DataFrame, short_period: str, longer_period: str, resistance_lines: list, support_lines: list):
        # 다중 스레드에 대한 matplotlib 충돌이 문제가 되어 Lock 을 바탕으로 순차적인 처리 로직을 적용하였음
        while self.running:
            with self.drawing_lock:
                if not self.drawing:
                    self.drawing = True
                    break

        png_file_path = self.draw_graph(datas_, short_period, longer_period, resistance_lines, support_lines)

        with self.drawing_lock:
            self.drawing = False

        self.msg_queue.put({ "text" : msg, "photo" : png_file_path }) # telegram bot 에 텍스트 및 메세지 전송
        
    def draw_graph(self, datas_: pd.DataFrame, period: str, longer_period: str, resistance_lines: list, support_lines: list, debug: bool = False) -> str: # return path of picture
        current_time = time.localtime()
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
        self.logger.info("[log] draw_graph function called : " + str(formatted_time))

        print_period = 60

        fig = plt.figure(figsize=(12,8))
        x = np.arange(-(print_period - 1), 1)

        # price and ema
        ax1 = plt.subplot2grid((4,2), (0,0), rowspan=2, fig=fig)
        ax2 = plt.subplot2grid((4,2), (0,1), rowspan=2, fig=fig)
        ax1.set_ylim(min(datas_[longer_period]['close'].iloc[-print_period:]) * 0.99, max(datas_[longer_period]['close'].iloc[-print_period:]) * 1.01)
        ax2.set_ylim(min(datas_[period]['close'].iloc[-print_period:]) * 0.99, max(datas_[period]['close'].iloc[-print_period:]) * 1.01)
        ax1.plot(x, np.array(datas_[longer_period]['close'].iloc[-print_period:]), color='black')
        ax1.plot(x, np.array(datas_[longer_period]['ema5'].iloc[-print_period:]), color='tomato', alpha=0.4)
        ax1.plot(x, np.array(datas_[longer_period]['ema20'].iloc[-print_period:]), color='gold', alpha=0.4)
        ax1.plot(x, np.array(datas_[longer_period]['ema60'].iloc[-print_period:]), color='green', alpha=0.4)
        ax2.plot(x, np.array(datas_[period]['close'].iloc[-print_period:]), color='black')
        ax2.plot(x, np.array(datas_[period]['ema5'].iloc[-print_period:]), color='tomato', alpha=0.4)
        ax2.plot(x, np.array(datas_[period]['ema20'].iloc[-print_period:]), color='gold', alpha=0.4)
        ax2.plot(x, np.array(datas_[period]['ema60'].iloc[-print_period:]), color='green', alpha=0.4)
        
        plt.title(period)

        # support and resistance lines
        for resistance_line  in resistance_lines:
            ax1.plot(x, [ resistance_line for _ in range(len(x)) ], color='black', alpha=0.6)
            ax2.plot(x, [ resistance_line for _ in range(len(x)) ], color='black', alpha=0.6)

        for support_line in support_lines:
            ax1.plot(x, [ support_line for _ in range(len(x)) ], color='black', alpha=0.6)
            ax2.plot(x, [ support_line for _ in range(len(x)) ], color='black', alpha=0.6)

        # rsi
        ax3 = plt.subplot2grid((4,2), (2,0), fig=fig)
        ax4 = plt.subplot2grid((4,2), (2,1), fig=fig)
        ax3.plot(x, np.array(datas_[longer_period]['rsi14'].iloc[-print_period:]), color='black')
        ax3.plot(x, np.array(datas_[longer_period]['rsi14_ma'].iloc[-print_period:]), color='orange')
        ax4.plot(x, np.array(datas_[period]['rsi14'].iloc[-print_period:]), color='black')
        ax4.plot(x, np.array(datas_[period]['rsi14_ma'].iloc[-print_period:]), color='orange')

        # macd
        ax5 = plt.subplot2grid((4,2), (3,0), fig=fig)
        ax6 = plt.subplot2grid((4,2), (3,1), fig=fig)
        ax5.plot(x, np.array(datas_[longer_period]['macd'].iloc[-print_period:]), color='black')
        ax5.plot(x, np.array(datas_[longer_period]['macd_ma'].iloc[-print_period:]), color='orange')
        ax6.plot(x, np.array(datas_[period]['macd'].iloc[-print_period:]), color='black')
        ax6.plot(x, np.array(datas_[period]['macd_ma'].iloc[-print_period:]), color='orange')

        current_time = time.strftime("%Y-%m-%d_%H-%M-%S")  # 원하는 포맷으로 지정
        png_file_path = os.path.join("imgs", f"{current_time}.png")

        plt.savefig(png_file_path, dpi=160)

        if debug is True: # jupyter notebook 전용
            plt.show()

        return png_file_path
    

    def run(self):
        while self.running:
            if self.data_queue.qsize() > 0:
                self.logger.info("[log] self.data_queue.qsize() is " + str(self.data_queue.qsize()))
                download_prices, ticker, period = self.data_queue.get()

                self.datas.update({ ticker : copy.deepcopy(download_prices) })

                # 전처리 및 보조지표 계산
                for idx, p in enumerate(list(self.download_format.keys())):
                    if p not in self.datas_cache[ticker].keys() or \
                       p == period:
                       ##p not in self.vp_kdes_diffs_cache[ticker].keys() or \
                       ##p not in self.resistance_lines_cache[ticker].keys() or \
                       ##p not in self.support_lines_cache[ticker].keys() or \
                        # 보조지표 계산
                        values, x, kdes_diff = self.calculate_indicators(self.datas[ticker][p][0], p) # datas[p][0] : 데이터, datas[p][1] : 마지막 다운로드 timestamp
                        self.datas_cache[ticker].update({p : values})
                        ##self.vp_kdes_diffs_cache[ticker].update({p : (x, kdes_diff)}) # x 값들과 그에 대한 미분값 순서대로 값을 넣음

                        # 더 큰 타임스탬프에 대한 보조지표 계산 및 반영 (매물대 지표 제외)
                        for idx_ in range(idx-1, -1, -1):
                            p_ = list(self.download_format.keys())[idx_]
                            values_ = self.calculate_indicators_last_candle(self.datas_cache[ticker][p_], p_, self.datas_cache[ticker][p], p)
                            self.datas_cache[ticker].update({p_ : values_})
                        
                        # 현재 period 까지 확인이 완료되면 fractal 분석 진행
                        if p == period:
                            break

                if ticker == self.buy_sell_ticker:
                    self.fractal_analyze(self.datas_cache[ticker], ticker, period) # 매매를 위한 분석 진행
                else:
                    pass # 시장 분석용 (전체적인 긍정 부정 의견)
                    
    def stop(self):
        self.running = False
