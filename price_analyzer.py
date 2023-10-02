import queue
import threading
import copy
import pandas as pd
import numpy as np
import time
import logging
from scipy.stats import gaussian_kde

class price_analyzer(threading.Thread):
    def __init__(self, buy_sell_ticker: str, data_queue: queue.Queue, download_format: dict, logger: logging):
        threading.Thread.__init__(self)
        self.buy_sell_ticker = buy_sell_ticker
        self.data_queue = data_queue
        self.download_format = download_format
        self.logger = logger
        self.running = True
        self.datas = {}

    # 한 개의 period 에 대한 보조지표 계산하는 함수 
    def calculate(self, datas, period):
        datas_ = pd.DataFrame(datas, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'q_volume'])
    
        # 거래량 multiple 계산 방식 구현
        volume_profiles = []

        # period 에 따라 달라지는 거래량 크기에 따라 연산량 최소화 (매물대 계산)
        if "W" in period:
            ratio_of_removal = 0.0008
        elif "D" in period:
            ratio_of_removal = 0.004
        elif "H" in period:
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
        
        # KDE 적용
        level = 100
        x = np.linspace(min(volume_profiles) - 1, max(volume_profiles) + 1, level)
        kde = gaussian_kde(volume_profiles)
        
        # KDE 정규화 적용
        kdes = []
        
        for i in range(len(x)):
            kdes.append(kde(x[i]))
        
        kdes = np.array(kdes)
        kdes -= np.mean(kdes)
        kdes /= np.std(kdes)
        
        # 미분 적용 (매물대 분석)
        kdes_diff = []
    
        for i in range(1, len(x)):
            kdes_diff.append(kde(x[i]) - kde(x[i-1]))
        
        kdes_diff = np.array(kdes_diff)
        kdes_diff -= np.mean(kdes_diff)
        kdes_diff /= np.std(kdes_diff)

        # RSI 상대강도지수 계산
        datas_['price_diff'] = datas_['close'].diff()
        datas_['gain'] = datas_['price_diff'].apply(lambda x: x if x > 0 else 0)
        datas_['loss'] = datas_['price_diff'].apply(lambda x: -x if x < 0 else 0)
        datas_['avg_gain14'] = datas_['gain'].rolling(window=14).mean()
        datas_['avg_loss14'] = datas_['loss'].rolling(window=14).mean()
        datas_['rsi14'] = 100 - (100 / (1 + (datas_['avg_gain14'] / datas_['avg_loss14'])))
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
        datas_.drop(['price_diff', 'gain', 'loss', 'avg_gain14', 'avg_loss14', 'macd_oscillator', 'macd_min', 'macd_max'], axis=1, inplace=True)

        
        # test
        print(period)
        print("close")
        print(datas_['close'].iloc[-5:])
        print("rsi14")
        print(datas_['rsi14'].iloc[-5:])
        print("rsi14_ma")
        print(datas_['rsi14_ma'].iloc[-5:])
        

        return datas_, x[1:], kdes_diff

    # 한 Ticker 에 대한 모든 period 분석을 진행하는 함수
    def analyze(self, datas: dict):
        current_time = time.localtime()
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
        self.logger.info("[log] analyze funtion called :" + str(formatted_time))

        # 초기변수 설정
        datas_ = {}
        kdes_diffs = {}

        # 보조지표 계산
        for period in self.download_format.keys():
            values, x, kdes_diff = self.calculate(datas[period][0], period) # datas[period][0] : 데이터, datas[period][1] : 마지막 다운로드 timestamp
            datas_.update({period : values})
            kdes_diffs.update({period : (x, kdes_diff)}) # x 값들과 그에 대한 미분값 순서대로 값을 넣음

        # 매물대 분석
        volume_profile_periods = ["1H", "6H", "1D", "1W"]

        resistance_lines = [] # 저항선
        support_lines = [] # 지지선
        
        now_price = datas_["1H"]['close'].iloc[-1]
        
        for volume_profile_period in volume_profile_periods:
            for idx, price in enumerate(kdes_diffs[volume_profile_period][0]):
                if now_price <= price:
                    now_kdes_diff = kdes_diffs[volume_profile_period][1][idx]

                    self.logger.info("--" + str(volume_profile_period) + "--")
                    self.logger.info("[log] 매물대 분석 결과 :" + str(now_kdes_diff))
                    if now_kdes_diff < 0:
                        # 목표가 계산
                        for i in range(len(kdes_diffs[volume_profile_period][1])):
                            if idx+i >= len(kdes_diffs[volume_profile_period][1]):
                                self.logger.info("[log] 저항선 설정 불가 : 상단 매물대 분석 불가능")
                                break
                            
                            if kdes_diffs[volume_profile_period][1][idx+i] >= 0:
                                resistance_line = int(kdes_diffs[volume_profile_period][0][idx+i])
                                self.logger.info("[log] 저항선 :" + str(resistance_line))
                                resistance_lines.append(resistance_line)
                                break

                        # 손절가 계산
                        for i in range(len(kdes_diffs[volume_profile_period][1])):
                            if idx-i >= len(kdes_diffs[volume_profile_period][1]):
                                self.logger.info("[log] 지지선 설정 불가 : 하단 매물대 분석 불가능")
                                break
                            
                            if kdes_diffs[volume_profile_period][1][idx-i] >= 0:
                                support_line = int(kdes_diffs[volume_profile_period][0][idx-i])
                                self.logger.info("[log] 지지선 :" + str(support_line))
                                support_lines.append(support_line)
                                break

                    self.logger.info("------")
                    self.logger.info("")
                    break

        # 보조지표 분석
        target_period = [("1H", "6H"), ("6H", "1D"), ("1D", "1W")] # (period, longer_period) --> fractal 개념 활용
        golden_cross_periods = []
        dead_cross_periods = []

        # 보조지표 매수 관점 분석 (long)
        for period, longer_period in target_period:
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
               rsi14_1 > rsi14_ma_1 and rsi14_2 <= rsi14_ma_2: # and macd_gc_prob > 0.5:
                self.logger.info("[log] RSI 지표의 골든크로스가 일어났습니다. " + str(round(rsi14_1, 2)) + ", " + str(round(rsi14_ma_1, 2)) \
                                 + ", " + str(round(rsi14_longer_1, 2)) + ", " + str(round(rsi14_longer_ma_1, 2)) + ", " + str(round(macd_gc_prob, 2)))
                golden_cross_periods.append(period)
            else:
                self.logger.info("[log] RSI_1 지표 : " + str(round(rsi14_1, 2)) + ", " + str(round(rsi14_ma_1, 2)))
                self.logger.info("[log] RSI_2 지표 : " + str(round(rsi14_longer_1, 2)) + ", " + str(round(rsi14_longer_ma_1, 2)))
                self.logger.info("[log] MACD GC PROB 지표 : " + str(round(macd_gc_prob, 2)))

            self.logger.info("------")
            self.logger.info("")

        # 보조지표 매도 관점 분석 (short)
        for period, longer_period in target_period:
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
               rsi14_1 < rsi14_ma_1 and rsi14_2 >= rsi14_ma_2: # and macd_dc_prob > 0.5:
                self.logger.info("[log] RSI 지표의 데드크로스가 발생하였습니다. " + str(round(rsi14_1, 2)) + ", " + str(round(rsi14_ma_1, 2)) \
                                 + ", " + str(round(rsi14_longer_1, 2)) + ", " + str(round(rsi14_longer_ma_1, 2)) + ", " + str(round(macd_dc_prob, 2)))
                dead_cross_periods.append(period)
            else:
                self.logger.info("[log] RSI_1 지표 : " + str(round(rsi14_1, 2)) + ", " + str(round(rsi14_ma_1, 2)))
                self.logger.info("[log] RSI_2 지표 : " + str(round(rsi14_longer_1, 2)) + ", " + str(round(rsi14_longer_ma_1, 2)))
                self.logger.info("[log] MACD DC PROB 지표 : " + str(round(macd_dc_prob, 2)))

            self.logger.info("-------")
            self.logger.info("")

        # 메모리 누수 방지를 위한 메모리 해제
        keys_to_delete = list(datas_.keys())
        for key in keys_to_delete:
            del datas_[key]

        return resistance_lines, support_lines, now_price, golden_cross_periods, dead_cross_periods

    def run(self):
        while self.running:
            if self.data_queue.qsize() > 0:
                ticker, download_prices = self.data_queue.get()

                self.datas.update({ ticker : copy.deepcopy(download_prices) })

                if ticker == self.buy_sell_ticker:
                    resistance_lines, support_lines, now_price, golden_cross_periods, dead_cross_periods =  self.analyze(self.datas[ticker]) # 매매를 위한 분석 진행
                    try:
                        loss_profit_ratio = (min(resistance_lines) - now_price) / (now_price - max(support_lines))
                    except:
                        loss_profit_ratio = None
                    self.logger.info("[log] 대략적인 손익비 : " + str(loss_profit_ratio))
                    if len(golden_cross_periods) == 0 and len(dead_cross_periods) == 0:
                        self.logger.info("[log] 골든크로스 및 데드크로스가 발생하지 않았습니다.")
                    else:
                        self.logger.info("[log] 골든크로스 및 데드크로스 : " + str(golden_cross_periods) + ", " + str(dead_cross_periods))
                        pass # 해당 부분에 텔레그램 알림 기능 부분 추가
                else:
                    pass # 시장 분석용 (전체적인 긍정 부정 의견)
                    
    def stop(self):
        self.running = False