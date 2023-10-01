import queue
import threading
import copy
import pandas as pd
import numpy as np
import time
from scipy.stats import gaussian_kde

class price_analyzer(threading.Thread):
    def __init__(self, buy_sell_ticker: str, data_queue: queue.Queue, download_format: dict):
        threading.Thread.__init__(self)
        self.buy_sell_ticker = buy_sell_ticker
        self.data_queue = data_queue
        self.download_format = download_format
        self.running = True
        self.datas = {}

    # 한 개의 period 에 대한 보조지표 계산하는 함수 
    def calculate(self, datas):
        datas_ = pd.DataFrame(datas, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'q_volume'])
    
        # 거래량 multiple 계산 방식 구현
        volume_profiles = []
        
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
                for k in range(int(volume * multiple_of_volume[j] / total_div)):
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

        return datas_, x[1:], kdes_diff

    # 한 Ticker 에 대한 모든 period 분석을 진행하는 함수
    def analyze(self, datas: dict):
        current_time = time.localtime()
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
        print("[log] analyze funtion called :", formatted_time)

        # 초기변수 설정
        datas_ = {}
        kdes_diffs = {}

        # 보조지표 계산
        for period in self.download_format.keys():
            values, x, kdes_diff = self.calculate(datas[period][0]) # datas[period][0] : 데이터, datas[period][1] : 마지막 다운로드 timestamp
            datas_.update({period : values})
            kdes_diffs.update({period : (x, kdes_diff)}) # x 값들과 그에 대한 미분값 순서대로 값을 넣음

        # 매물대 분석
        buy_period = ["1H"]
        sell_period = ["6H"]
        volume_profile_periods = ["1H", "6H", "1D"]

        now_price = datas_["1H"]['close'].iloc[-1]
        
        for volume_profile_period in volume_profile_periods:
            for idx, price in enumerate(kdes_diffs[volume_profile_period][0]):
                if now_price <= price:
                    now_kdes_diff = kdes_diffs[volume_profile_period][1][idx]

                    print("--" + str(volume_profile_period) + "--")
                    print("[log] 매물대 분석 결과 :", now_kdes_diff)
                    if now_kdes_diff < 0:
                        # 목표가 계산
                        for i in range(len(kdes_diffs[volume_profile_period][1])):
                            if idx+i >= len(kdes_diffs[volume_profile_period][1]):
                                print("[log] 저항선 설정 불가 : 상단 매물대 분석 불가능")
                                break
                            
                            if kdes_diffs[volume_profile_period][1][idx+i] >= 0:
                                print("[log] 저항선 :", int(kdes_diffs[volume_profile_period][0][idx+i]))
                                break

                        # 손절가 계산
                        for i in range(len(kdes_diffs[volume_profile_period][1])):
                            if idx-i >= len(kdes_diffs[volume_profile_period][1]):
                                print("[log] 지지선 설정 불가 : 하단 매물대 분석 불가능")
                                break
                            
                            if kdes_diffs[volume_profile_period][1][idx-i] >= 0:
                                print("[log] 지지선 :", int(kdes_diffs[volume_profile_period][0][idx-i]))
                                break

                    print("------")
                    print()
                    break

        # 보조지표 분석
        for period in buy_period:
            print("--" + str(period) + "-- buy insight --")
            macd_gc_prob_1 = datas_[period]['macd_gc_prob'].iloc[-1]
            macd_gc_prob_2 = datas_[period]['macd_gc_prob'].iloc[-2]
            rsi14_1 = datas_[period]['rsi14'].iloc[-1]
            rsi14_2 = datas_[period]['rsi14'].iloc[-2]
            rsi14_ma_1 = datas_[period]['rsi14_ma'].iloc[-1]
            rsi14_ma_2 = datas_[period]['rsi14_ma'].iloc[-2]
            if macd_gc_prob_1 > 0.5 and macd_gc_prob_2 <= 0.5:
                print("[log] MACD Prob 지표가 Golden-Cross 할 확률이 큽니다.", round(macd_gc_prob_1, 2))
            else:
                print("[log] MACD Prob 지표 :", round(macd_gc_prob_1, 2))

            if rsi14_1 > rsi14_ma_1 and rsi14_2 <= rsi14_ma_2:
                print("[log] RSI 지표의 골든크로스가 일어났습니다.", round(rsi14_1, 2), round(rsi14_ma_1, 2))
            else:
                print("[log] RSI 지표 :", round(rsi14_1, 2), round(rsi14_ma_1, 2))

            print("------")
            print()

        for period in sell_period:
            print("--" + str(period) + "-- sell insight --")
            macd_dc_prob_1 = datas_[period]['macd_dc_prob'].iloc[-1]
            macd_dc_prob_2 = datas_[period]['macd_dc_prob'].iloc[-2]
            rsi14_1 = datas_[period]['rsi14'].iloc[-1]
            rsi14_2 = datas_[period]['rsi14'].iloc[-2]
            rsi14_ma_1 = datas_[period]['rsi14_ma'].iloc[-1]
            rsi14_ma_2 = datas_[period]['rsi14_ma'].iloc[-2]
            if macd_dc_prob_1 > 0.5 and macd_dc_prob_2 <= 0.5:
                print("[log] MACD Prob 지표가 Dead-Cross 할 확률이 큽니다.", round(macd_dc_prob_1, 2))
            else:
                print("[log] MACD Prob 지표 :", round(macd_dc_prob_1, 2))

            if rsi14_1 < rsi14_ma_1 and rsi14_2 >= rsi14_ma_2:
                print("[log] RSI 지표의 데드크로스가 일어났습니다.", round(rsi14_1, 2), round(rsi14_ma_1, 2))
            else:
                print("[log] RSI 지표 :", round(rsi14_1, 2), round(rsi14_ma_1, 2))

            print("------")
            print()

        # 메모리 누수 방지를 위한 메모리 해제
        keys_to_delete = list(datas_.keys())
        for key in keys_to_delete:
            del datas_[key]

    def run(self):
        while self.running:
            if self.data_queue.qsize() > 0:
                ticker, download_prices = self.data_queue.get()

                self.datas.update({ ticker : copy.deepcopy(download_prices) })

                if ticker == self.buy_sell_ticker:
                    self.analyze(self.datas[ticker]) # 매매를 위한 분석 진행
                else:
                    pass # 시장 분석용 (전체적인 긍정 부정 의견)
                    
    def stop(self):
        self.running = False