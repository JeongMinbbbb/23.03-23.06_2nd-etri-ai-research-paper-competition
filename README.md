# ETRI_Attachment_Place

> 제2회 ETRI 휴면이해 인공지능 논문경진대회 
- 라이프로그 데이터셋 활용 인식 및 추론 기술 분야
> 2023 한국컴퓨터종합학술대회(KCC2023)와 연계

## 주제

생체 신호 데이터를 이용한 애착 장소 분석
"Attachment Place Analysis using Biosignal Data"

## 요약
 
 본 연구는 E4 모바일 웨어러블 디바이스를 통해 수집된 생체 신호 데이터를 활용하여 사용자의 애착 장소를 분석하는 것을 목적으로 한다. 생체 신호 데이터 중 특히 Accelerometer(Acc), Blood Volume Pressure(Bvp), Electrodermal Activity(Eda), Heart Rate(Hr), Skin Temperature(Temp)를 이용하여 사용자의 장소를 예측하는 Multi-Resolution CNN 모델을 구성하였다. 이후 모델이 오분류한 시점의 gps 데이터[map.png]를 이용하여 사용자 별 애착 장소를 분석하는 방법에 대한 제안을 한다.

![map](https://user-images.githubusercontent.com/90736934/231823008-a1ca309c-b410-4cbb-844c-450f92a8d44d.png)
<map.png>
![image](https://github.com/JeongMinbbbb/23.03-23.06_ETRI_Attachment_Place/assets/130365764/2c8b7372-1d4e-44a6-aacc-935795acb3aa)
<애착장소.png>
![Uploading 애착장소.png…]()
## 모듈 구현 방법

- main.ipynb를 이용하여 세부 기능 모듈(Preprocessing.py, datasplit.py, SP_Model.py, GpsVS.py)을 사용할 수 있다.
- Data 폴더 내에 센서 데이터 업로드 후, path 변수 설정

## 사용 오픈소스 라이브러리

- tensorflow == 2.12.0
- sklearn == 1.2.2
- pandas == 1.5.3
- numpy == 1.22.4
- matplotlib == 3.7.1
- folium == 0.14.0
- tqdm == 4.65.0


## 개발 환경
Python3

## 문의
