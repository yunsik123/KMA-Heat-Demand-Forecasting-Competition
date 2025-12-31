# KMA Heat Demand Forecasting Competition

기상청(KMA) 빅데이터 콘테스트 - 열수요 예측 모델 개발 프로젝트

## 프로젝트 개요

본 프로젝트는 **역난방 열수요와 날씨 빅데이터를 융합한 열수요 예측** 경진대회를 위해 개발되었습니다.
19개 지사(A~S)의 기상 데이터를 활용하여 미래 열수요량을 예측합니다.

### 대회 정보
- **주최**: 기상청 (Korea Meteorological Administration)
- **기간**: 2025년
- **목표**: 시간별 열수요(heat_demand) 예측

## 데이터셋

### 학습 데이터
| 파일명 | 설명 | 기간 |
|--------|------|------|
| `train_heat.csv` | 열수요 + 기상 데이터 | 2021.01 ~ 2023.12 |
| `test_heat.csv` | 테스트 데이터 | 2024년 |

### 주요 변수
| 변수명 | 설명 | 단위 |
|--------|------|------|
| `tm` | 시간 | YYYYMMDDHH |
| `branch_id` | 지사명 | A~S (19개) |
| `ta` | 기온 | °C |
| `wd` | 풍향 | ° |
| `ws` | 풍속 | m/s |
| `rn_day` | 일강수량 | mm |
| `rn_hr1` | 시간강수량 | mm |
| `hm` | 상대습도 | % |
| `si` | 일사량 | MJ/m² |
| `ta_chi` | 체감온도 | °C |
| `heat_demand` | 열수요 (Target) | - |

## 프로젝트 구조

```
KMA-Heat-Demand-Forecasting-Competition/
│
├── data/                          # 데이터 파일
│   ├── train_heat.csv
│   ├── test_heat.csv
│   └── ...
│
├── notebooks/                     # Jupyter 노트북
│   ├── experiments/               # 실험용 노트북
│   │   ├── baseline_lightgbm*.ipynb
│   │   ├── baseline_xgboost*.ipynb
│   │   └── ...
│   ├── models/                    # 모델별 노트북
│   │   ├── Baseline_DeepAR.ipynb
│   │   ├── Baseline_딥러닝CNN_Transformer_버전.ipynb
│   │   └── Autogluon첫제출용.ipynb
│   └── EDA.ipynb                  # 탐색적 데이터 분석
│
├── src/                           # 소스 코드
│   ├── preprocessing.py           # 전처리 함수
│   └── baseline.py                # 베이스라인 코드
│
├── docs/                          # 문서
│   ├── Deep_AR에대한이해.txt
│   └── 다른분들아이디어.txt
│
├── submissions/                   # 제출 파일
│   └── 250464.csv
│
├── .gitignore
└── README.md
```

## 모델링 접근법

### 1. 전처리 (Preprocessing)

#### 결측치 처리
- **-99 값**: 결측치로 변환 후 SVR 보간
- **si (일사량)**: 08~18시 외 -99는 0으로 처리
- **wd (풍향)**: 9.9는 NaN으로 변환

#### 피처 엔지니어링
```python
# 시간 관련 변수
- year, month, day, hour, weekday
- is_weekend, is_holiday (한국 공휴일)
- hour_sin, hour_cos (주기성 인코딩)
- dayofyear_sin, dayofyear_cos

# 기온 관련 파생변수
- HDD18/HDD20 (Heating Degree Day)
- CDD18/CDD20 (Cooling Degree Day)
- ta_lag_1, ta_lag_2, ta_lag_3 (시차 변수)
- ta_diff_6h, ta_diff_12h, ta_diff_24h (기온 변화량)
- ta_3h_avg_* (이동 평균)
- daily_range (일교차)

# 체감온도 계산
- apparent_temp (계절별 체감온도 공식 적용)
- wind_chill (풍속 고려 체감온도)

# 기상 지수
- DCI (불쾌지수)
- wchi (풍속 냉지수)
- atemphi (실효온도)

# 시계열 분해
- STL 분해 (Trend, Seasonal, Residual)
- 푸리에 변환 특성 (FFT)
```

### 2. 모델 아키텍처

#### 최종 모델: 잔차 스태킹 앙상블
```
[LightGBM] → 예측값 + [XGBoost(잔차 예측)] → 최종 예측
```

1. **1차 모델 (LightGBM)**
   - Optuna로 하이퍼파라미터 튜닝
   - Huber Loss 사용 (이상치 로버스트)
   - Early Stopping 적용

2. **2차 모델 (XGBoost)**
   - LightGBM 잔차 학습
   - Pseudo Huber Error 사용

3. **교차 검증**
   - 시계열 기반 3-Fold (연도별 분할)
   - Fold 평균으로 최종 예측

#### 실험한 모델들
| 모델 | 설명 | 비고 |
|------|------|------|
| LightGBM | Gradient Boosting | 최종 채택 |
| XGBoost | Gradient Boosting | 잔차 보정용 |
| DeepAR | LSTM + Attention | 시계열 특화 |
| CNN-Transformer | 딥러닝 | 실험적 |
| AutoGluon | AutoML | 빠른 프로토타이핑 |

### 3. 하이퍼파라미터

```python
# LightGBM 주요 파라미터
{
    'objective': 'huber',
    'boosting_type': 'gbdt',
    'learning_rate': 0.01~0.3,
    'num_leaves': 10~300,
    'max_depth': 3~15,
    'feature_fraction': 0.4~1.0,
    'bagging_fraction': 0.4~1.0,
    'n_estimators': 1000,
}

# XGBoost 주요 파라미터
{
    'objective': 'reg:pseudohubererror',
    'max_depth': 2~10,
    'subsample': 0.5~1.0,
    'colsample_bytree': 0.5~1.0,
}
```

## 실행 방법

### 환경 설정
```bash
# 필요 라이브러리 설치
pip install pandas numpy scikit-learn lightgbm xgboost optuna
pip install holidays scipy statsmodels plotly tqdm
```

### 학습 및 예측
```bash
# 최종 모델 실행
jupyter notebook notebooks/experiments/baseline_lightgbm6_최종모델제출용.ipynb
```

## 주요 인사이트

### EDA 결과
1. **열수요 패턴**: 새벽 시간대(0~6시) 열수요가 가장 높음
2. **계절성**: 겨울철(11~3월) 열수요 급증
3. **지사별 차이**: 각 지사(A~S)마다 열수요 규모와 패턴이 상이

### 중요 변수 (Feature Importance)
1. 기온 관련 변수 (ta, ta_chi, apparent_temp)
2. 시간 관련 변수 (hour, peak_time)
3. HDD (Heating Degree Day)
4. 체감온도 및 풍속 냉지수

## 참고 자료

### 관련 논문 및 아이디어
- 2024 전력 예측 모델 최우수상/우수상 접근법 참고
- 푸리에 변환을 통한 주기성 특성 추출
- STL 분해를 통한 시계열 분해

### 체감온도 공식
- **여름철**: 복합 습구온도 기반 공식
- **겨울철**: Wind Chill 공식 (풍속 고려)

## 라이선스

이 프로젝트는 개인 연구 및 학습 목적으로 작성되었습니다.

## 저자

- GitHub: [@yunsik123](https://github.com/yunsik123)
