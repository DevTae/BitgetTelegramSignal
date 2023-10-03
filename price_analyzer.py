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
import matplotlib.pyplot as plt

class price_analyzer(threading.Thread):
    def __init__(self, buy_sell_ticker: str, data_queue: queue.Queue, msg_queue: queue.Queue, download_format: dict, logger: logging):
        threading.Thread.__init__(self)
        self.buy_sell_ticker = buy_sell_ticker
        self.data_queue = data_queue
        self.msg_queue = msg_queue
        self.download_format = download_format # { period : [ datas, lasttime (마지막 캔들) ] }
        self.logger = logger
        self.datas = {}

        # for caching
        self.datas_cache = {}
        self.vp_kdes_diffs_cache = {}
        self.resistance_lines_cache = {}
        self.support_lines_cache = {}

        # buy_sell_ticker initializing
        self.datas_cache[buy_sell_ticker] = {}
        self.vp_kdes_diffs_cache[buy_sell_ticker] = {}
        self.resistance_lines_cache[buy_sell_ticker] = {}
        self.support_lines_cache[buy_sell_ticker] = {}

        self.running = True


    # 한 개의 period 에 대한 보조지표 계산하는 함수 (datas : array -> datas_ : pandas.DataFrame)
    def calculate_indicators(self, datas, period):
        current_time = time.localtime()
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
        self.logger.info("[log] calculate_indicators function called : " + str(formatted_time) + " " + str(period))

        datas_ = pd.DataFrame(datas, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'q_volume'])
    
        # 매물대 계산을 위하여 거래량 multiple 방식 구현
        volume_profiles = []

        # 매물대 계산을 위하여 period 에 따라 달라지는 거래량 크기에 따라 연산량 최소화
        if 'W' in period:
            ratio_of_removal = 0.0008
        elif 'D' in period:
            ratio_of_removal = 0.004
        elif 'H' in period:
            ratio_of_removal = 0.1
        else:
            ratio_of_removal = 0.1
        
        for i in range(len(datas_['time'])):
            volume = datas_['volume'][i]
            open_price = datas_['open'][i]
            high_price = datas_['high'][i]
            low_price = datas_['low'][i]
            close_price = datas_['close'][i]
            center_price = (open_price + high_price + low_price + close_price) / 4
            
            total_div = 10
            div_size = (high_price - low_price) / total_div
            index_pivot = (center_price - low_price) // div_size
            index_pivot = max(min(index_pivot, total_div - 1), 1) # 시고저종 가격을 바탕으로 중심값 선정
            multiple_of_volume = np.array(list(np.linspace(1, 0, int(index_pivot))) \
                                        + list(np.linspace(0, 1, int(total_div - index_pivot + 1))[1:]))
            multiple_of_volume /= sum(multiple_of_volume)
            
            for j in range(len(multiple_of_volume)):
                for _ in range(int(volume * multiple_of_volume[j] * ratio_of_removal)):
                    volume_profiles.append(low_price + j * div_size)
        
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

        # RSI 상대강도지수 계산
        datas_['price_diff'] = datas_['close'].diff()
        datas_['gain'] = datas_['price_diff'].apply(lambda x: x if x > 0 else 0)
        datas_['loss'] = datas_['price_diff'].apply(lambda x: -x if x < 0 else 0)
        datas_['au'] = datas_['gain'].ewm(com=14-1, min_periods=14).mean()
        datas_['ad'] = datas_['loss'].ewm(com=14-1, min_periods=14).mean()
        datas_['rs'] = datas_['au'] / datas_['ad']
        datas_['rsi14'] = 100 - (100 / (1 + datas_['rs']))
        datas_['rsi14_ma'] = datas_['rsi14'].rolling(window=14).mean()
        
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
        
        # 필요 없는 열 삭제
        datas_.drop(['price_diff', 'gain', 'loss', 'au', 'ad', 'rs', 'macd_oscillator', 'macd_min', 'macd_max'], axis=1, inplace=True)

        return datas_, x[1:], vp_kdes_diff


    # 한 Ticker 에 대한 개별 period 분석을 진행하는 함수
    def fractal_analyze(self, datas: dict, ticker: str, period: str):
        current_time = time.localtime()
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
        self.logger.info("[log] fractal_analyze function called : " + str(formatted_time) + " " + str(period))

        # 전처리 및 보조지표 계산
        for p in self.download_format.keys():
            if p not in self.datas_cache[ticker].keys() or \
               p not in self.vp_kdes_diffs_cache[ticker].keys() or \
               p not in self.resistance_lines_cache[ticker].keys() or \
               p not in self.support_lines_cache[ticker].keys() or \
               p == period:
                # 보조지표 계산
                values, x, kdes_diff = self.calculate_indicators(datas[p][0], p) # datas[p][0] : 데이터, datas[p][1] : 마지막 다운로드 timestamp
                self.datas_cache[ticker].update({p : values})
                self.vp_kdes_diffs_cache[ticker].update({p : (x, kdes_diff)}) # x 값들과 그에 대한 미분값 순서대로 값을 넣음
                
                # 매물대 분석
                now_price = self.datas_cache[ticker][p]['close'].iloc[-1]
                resistance_lines_, support_lines_ = self.analyze_volume_profile(self.vp_kdes_diffs_cache[ticker], now_price, p)

                # cache 에 저항선 및 지지선 데이터 저장하기
                self.resistance_lines_cache[ticker].update({p : resistance_lines_})
                self.support_lines_cache[ticker].update({p : support_lines_})

                # 현재 period 까지 확인이 완료되면 보조지표 분석 진행
                if p == period:
                    break
        
        # period (p >= period) 에 맞는 저항선 및 지지선 데이터 선정
        resistance_lines = []
        support_lines = []

        for p in self.download_format.keys():
            resistance_lines += self.resistance_lines_cache[ticker][p]
            support_lines += self.support_lines_cache[ticker][p]
            if p == period:
                break
        
        # 보조지표 분석, 조건 감시 및 알림 (fractal period set)
        target_periods = [("1H", "6H"), ("6H", "1D"), ("1D", "1W")] # (period, longer_period)
        available = False

        for idx, (p, _) in enumerate(target_periods):
            if p == period:
                available = True
                target_period = target_periods[idx]
                break

        if available:
            self.analyze_indicator_long(self.datas_cache[ticker], target_period, resistance_lines, support_lines, debug=False)
            self.analyze_indicator_short(self.datas_cache[ticker], target_period, resistance_lines, support_lines, debug=False)
        else:
            self.logger.info("[log] not available period in analyze_indicator : " + str(period))


    def analyze_volume_profile(self, kdes_diffs, now_price, volume_profile_period):
        current_time = time.localtime()
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
        self.logger.info("[log] analyze_volume_profile function called : " + str(formatted_time) + " " + str(volume_profile_period))

        resistance_lines = []
        support_lines = []

        for idx, price in enumerate(kdes_diffs[volume_profile_period][0]):
            if now_price <= price:
                now_kdes_diff = kdes_diffs[volume_profile_period][1][idx]
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
                self.logger.info("")
                break

        return resistance_lines, support_lines


    def analyze_indicator_long(self, datas_, target_period, resistance_lines, support_lines, debug=False):
        current_time = time.localtime()
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
        self.logger.info("[log] analyze_indicator_long function called : " + str(formatted_time))

        # target_period
        period, longer_period = target_period

        now_price = datas_[period]['close'].iloc[-1]

        # 보조지표 매수 관점 분석 (long)
        self.logger.info("--" + str(period) + "," + str(longer_period) + "--long-insight--")
        macd_gc_prob = datas_[period]['macd_gc_prob'].iloc[-1]
        rsi14_1 = datas_[period]['rsi14'].iloc[-1]
        rsi14_2 = datas_[period]['rsi14'].iloc[-2]
        rsi14_ma_1 = datas_[period]['rsi14_ma'].iloc[-1]
        rsi14_ma_2 = datas_[period]['rsi14_ma'].iloc[-2]
        rsi14_longer_1 = datas_[longer_period]['rsi14'].iloc[-1]
        rsi14_longer_2 = datas_[longer_period]['rsi14'].iloc[-2]
        rsi14_longer_ma_1 = datas_[longer_period]['rsi14_ma'].iloc[-1]
        rsi14_longer_ma_2 = datas_[longer_period]['rsi14_ma'].iloc[-2]
        if rsi14_longer_1 > rsi14_longer_ma_1 and rsi14_longer_2 > rsi14_longer_ma_2 and \
           rsi14_1 > rsi14_ma_1 and rsi14_2 <= rsi14_ma_2 or debug:
            png_file_path = self.draw_graph(datas_, period, longer_period, resistance_lines, support_lines)
            msg = f"[GC Signal] {period}봉 기준 RSI 지표의 골든크로스가 발생하였습니다. 상승에 대비하세요."
            msg += f"\n\t현재가 : {now_price}"
            if len(resistance_lines) != 0:
                msg += f"\n\t근접한 예상 저항선 : {min(resistance_lines)}"
            if len(support_lines) != 0:
                msg += f"\n\t근접한 예상 지지선 : {max(support_lines)}"
            self.send_to_telegram_bot(msg, png_file_path)
            self.logger.info("[log] RSI golden cross occured. " + str(round(rsi14_1, 2)) + ", " + str(round(rsi14_ma_1, 2)) \
                             + ", " + str(round(rsi14_longer_1, 2)) + ", " + str(round(rsi14_longer_ma_1, 2)) + ", " + str(round(macd_gc_prob, 2)))
        else:
            self.logger.info("[log] RSI_1 indicator : " + str(round(rsi14_1, 2)) + ", " + str(round(rsi14_ma_1, 2)))
            self.logger.info("[log] RSI_2 indicator : " + str(round(rsi14_longer_1, 2)) + ", " + str(round(rsi14_longer_ma_1, 2)))
            self.logger.info("[log] MACD GC PROB indicator : " + str(round(macd_gc_prob, 2)))
        self.logger.info("------")
        self.logger.info("")


    def analyze_indicator_short(self, datas_, target_period, resistance_lines, support_lines, debug=False):
        current_time = time.localtime()
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
        self.logger.info("[log] analyze_indicator_short function called : " + str(formatted_time))

        # target_period
        period, longer_period = target_period

        now_price = datas_[target_period[0]]['close'].iloc[-1]

        # 보조지표 매도 관점 분석 (short)
        self.logger.info("--" + str(period) + "," + str(longer_period) + "--short-insight--")
        macd_dc_prob = datas_[period]['macd_dc_prob'].iloc[-1]
        rsi14_1 = datas_[period]['rsi14'].iloc[-1]
        rsi14_2 = datas_[period]['rsi14'].iloc[-2]
        rsi14_ma_1 = datas_[period]['rsi14_ma'].iloc[-1]
        rsi14_ma_2 = datas_[period]['rsi14_ma'].iloc[-2]
        rsi14_longer_1 = datas_[longer_period]['rsi14'].iloc[-1]
        rsi14_longer_2 = datas_[longer_period]['rsi14'].iloc[-2]
        rsi14_longer_ma_1 = datas_[longer_period]['rsi14_ma'].iloc[-1]
        rsi14_longer_ma_2 = datas_[longer_period]['rsi14_ma'].iloc[-2]
        
        if rsi14_longer_1 < rsi14_longer_ma_1 and rsi14_longer_2 < rsi14_longer_ma_2 and \
           rsi14_1 < rsi14_ma_1 and rsi14_2 >= rsi14_ma_2 or debug:
            png_file_path = self.draw_graph(datas_, period, longer_period, resistance_lines, support_lines)
            msg = f"[DC Signal] {period}봉 기준 RSI 지표의 데드크로스가 발생하였습니다. 하락에 대비하세요."
            msg += f"\n\t현재가 : {now_price}"
            if len(resistance_lines) != 0:
                msg += f"\n\t근접한 예상 저항선 : {min(resistance_lines)}"
            if len(support_lines) != 0:
                msg += f"\n\t근접한 예상 지지선 : {max(support_lines)}"
            self.send_to_telegram_bot(msg, png_file_path)
            
            self.logger.info("[log] RSI dead cross occured. " + str(round(rsi14_1, 2)) + ", " + str(round(rsi14_ma_1, 2)) \
                             + ", " + str(round(rsi14_longer_1, 2)) + ", " + str(round(rsi14_longer_ma_1, 2)) + ", " + str(round(macd_dc_prob, 2)))
        else:
            self.logger.info("[log] RSI_1 indicator : " + str(round(rsi14_1, 2)) + ", " + str(round(rsi14_ma_1, 2)))
            self.logger.info("[log] RSI_2 indicator : " + str(round(rsi14_longer_1, 2)) + ", " + str(round(rsi14_longer_ma_1, 2)))
            self.logger.info("[log] MACD DC PROB indicator : " + str(round(macd_dc_prob, 2)))
        self.logger.info("-------")
        self.logger.info("")


    def send_to_telegram_bot(self, msg, png_file_path):
        self.msg_queue.put({ "text" : msg, "photo" : png_file_path })
        

    def draw_graph(self, datas_, period, longer_period, resistance_lines, support_lines, debug=False) -> str: # return path of picture
        current_time = time.localtime()
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
        self.logger.info("[log] draw_graph function called : " + str(formatted_time))

        matplotlib.use('agg') # 백엔드 사용 / 초반엔 5 분 걸리고, 이후에는 1 분 걸림. (초반 로딩이 오래 걸림)

        fig = plt.figure(figsize=(12,8))
        x = np.arange(-59, 1)

        # price
        ax1 = plt.subplot2grid((4,2), (0,0), rowspan=2, fig=fig)
        ax2 = plt.subplot2grid((4,2), (0,1), rowspan=2, fig=fig)
        ax1.plot(x, datas_[longer_period]['close'].iloc[-60:], color='black')
        ax2.plot(x, datas_[period]['close'].iloc[-60:], color='black')
        plt.title(period)

        # support and resistance lines
        for resistance_line  in resistance_lines:
            ax1.plot(x, [ resistance_line for _ in range(len(x)) ], color='purple')
            ax2.plot(x, [ resistance_line for _ in range(len(x)) ], color='purple')

        for support_line in support_lines:
            ax1.plot(x, [ support_line for _ in range(len(x)) ], color='green')
            ax2.plot(x, [ support_line for _ in range(len(x)) ], color='green')

        # rsi
        ax3 = plt.subplot2grid((4,2), (2,0), fig=fig)
        ax4 = plt.subplot2grid((4,2), (2,1), fig=fig)
        ax3.plot(x, datas_[longer_period]['rsi14'].iloc[-60:], color='black')
        ax3.plot(x, datas_[longer_period]['rsi14_ma'].iloc[-60:], color='orange')
        ax4.plot(x, datas_[period]['rsi14'].iloc[-60:], color='black')
        ax4.plot(x, datas_[period]['rsi14_ma'].iloc[-60:], color='orange')

        # macd
        ax5 = plt.subplot2grid((4,2), (3,0), fig=fig)
        ax6 = plt.subplot2grid((4,2), (3,1), fig=fig)
        ax5.plot(x, datas_[longer_period]['macd'].iloc[-60:], color='black')
        ax5.plot(x, datas_[longer_period]['macd_ma'].iloc[-60:], color='orange')
        ax6.plot(x, datas_[period]['macd'].iloc[-60:], color='black')
        ax6.plot(x, datas_[period]['macd_ma'].iloc[-60:], color='orange')

        current_time = time.strftime("%Y-%m-%d_%H-%M-%S")  # 원하는 포맷으로 지정
        png_file_path = os.path.join("imgs", f"{current_time}.png")

        plt.savefig(png_file_path, dpi=160)

        if debug is True:
            plt.show()

        return png_file_path
    

    def run(self):
        while self.running:
            if self.data_queue.qsize() > 0:
                download_prices, ticker, period = self.data_queue.get()

                self.datas.update({ ticker : copy.deepcopy(download_prices) })

                if ticker == self.buy_sell_ticker:
                    self.fractal_analyze(self.datas[ticker], ticker, period) # 매매를 위한 분석 진행
                else:
                    pass # 시장 분석용 (전체적인 긍정 부정 의견)
                    
    def stop(self):
        self.running = False
