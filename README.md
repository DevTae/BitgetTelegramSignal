### BitgetTelegramSignal
-----

- 해당 프로젝트는 `Bitget API`, `Telegram API` 를 활용하여 `BTCUSDT` 자산의 가격을 주기적으로 모니터링한 후 신호 포착 시 텔레그램으로 알림을 보내는 프로그램이다.

- 코드 구조는 다음과 같다.

  ```
  📦BitgetTelegramSignal
   ┣ 📂docs : API 참고 문서
   ┣ 📂imgs : 신호 포착 시 차트 사진 파일
   ┃ ┣ 📜2023-10-03_02-35-51.png
   ┃ ┗ 📜...
   ┣ 📂logs : 로그 파일
   ┃ ┣ 📜2023-10-09_22-27-31.log
   ┃ ┗ 📜...
   ┣ 📜main.py
   ┣ 📜data_crawler.py
   ┣ 📜price_analyzer.py
   ┣ 📜price_analyzer_demo.ipynb : 보조지표 분석 데모 파일
   ┣ 📜telegram_bot.py
   ┗ 📜.gitignore
  ```

- 프로그램 코드 설명
  - `main.py` : 메인 스크립트 파일. 해당 파일에서 `Telegram API Key` 및 `Telegram Chat Id` 를 입력한다.
  - `data_crawler.py` : Bitget API 를 통한 암호화폐 가격 데이터 불러오는 파일. 스레드를 바탕으로 암호화폐 가격 데이터를 주기적으로 불러온다.
  - `price_analyzer.py` : 보조지표 분석을 통한 신호 포착하는 파일. 매물대 분석 및 보조지표 프랙탈 분석 기법을 통하여 신호 포착 및 등록된 텔레그램 알림을 보낸다.
  - `telegram_bot.py` : 텔레그램 봇에게 메세지 보내는 파일.

<br/>

- 텔레그램 알림 화면
  - 아래 그림과 같다.
    
    ![24 1 1 비트겟 텔레그램 알림](https://github.com/DevTae/BitgetTelegramSignal/assets/55177359/be61a606-476d-49e6-904c-22446f3bf24a)


- 신호 포착 알고리즘
  - 가격 알림 기준
    - `MACD` 지표를 바탕으로 가격의 반전이 확인되면 신호를 주는 방식으로 구현하였다.
    
  - 매물대 분석 원리
    - 가격과 거래량을 바탕으로 매물대를 분석하고, 미분계수를 활용하여 현재 가격의 위치를 수치화하였다.

      ![24 1 1 매물대분석사례](https://github.com/DevTae/BitgetTelegramSignal/assets/55177359/dd183ab6-9147-4c54-acc9-41db9b915345)

