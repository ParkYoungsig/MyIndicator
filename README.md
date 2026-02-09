# MyIndicator (Static + Dynamic Asset Allocation System)

본 프로젝트는 **정적(Static) 자산배분 전략**과  
**동적(Dynamic) 모멘텀 기반 리밸런싱 전략**을 결합한  
**백테스트 및 라이브 리밸런싱 프레임워크**입니다.

로컬에 저장된 시계열 가격 데이터(parquet)를 기반으로  
- 과거 성과를 검증하는 **백테스트**
- 주기적으로 실행되는 **라이브 리밸런싱(실거래 연계)**  
를 동일한 전략 구조로 수행할 수 있도록 설계되었습니다.

---

## 1. 프로젝트 전체 구조

```
MyIndicator-main/
├── config/
│   ├── data.yaml
│   └── backtester.yaml
├── infra/
│   ├── config.py
│   ├── data_manager/
│   │   └── downloader.py
│   └── api/
│       └── broker/
│           └── kis.py
├── research/
│   └── backtester/
│       ├── cli.py
│       ├── runner.py
│       ├── master.py
│       ├── universe.py
│       ├── data_loader.py
│       └── strategies/
│           ├── static_engine.py
│           └── dynamic_engine.py
├── core/
│   ├── engine.py
│   ├── factors.py
│   ├── scoring.py
│   ├── signals.py
│   ├── allocators.py
│   └── rebalancing/
│       ├── triggers.py
│       └── methods.py
├── live/
│   ├── cli.py
│   ├── engine.py
│   ├── state.py
│   ├── execution/
│   │   └── order_manager.py
│   └── strategy/
│       ├── static.py
│       └── dynamic.py
└── data/
    ├── raw/
    └── processed/
```

---

## 2. 데이터 다운로드

```bash
python -m infra.data_manager.downloader --config data
```

- `config/data.yaml` 기준으로 주가 데이터를 다운로드
- parquet 형식으로 로컬 저장
- 증분 다운로드 및 스킵 기능 지원

---

## 3. 백테스트 실행

```bash
python -m research.backtester.cli --config backtester
```

- Static / Dynamic 전략 통합 백테스트
- 로컬 parquet 데이터 사용
- 성과 지표 및 결과 자동 저장

---

## 4. 동적 리밸런싱 개요 (요약)

동적 리밸런싱 전략은 다음의 흐름으로 동작합니다.

> **N일 수익률 기반 모멘텀으로 종목을 랭킹 →  
> 상관계수로 분산 필터링 →  
> 설정된 주기에 따라 목표 비중으로 리밸런싱**

### 사용 요소 요약
- 입력 데이터: 종가(close)
- 핵심 지표:
  - 모멘텀 (N일 수익률)
  - 변동성 (표준편차, 보조)
  - 종목 간 상관계수
- 의사결정 방식: 랭킹 기반 Top-N 선택
- 리밸런싱 주기: Daily / Weekly / Monthly / Quarterly

RSI, MACD, 이동평균 교차 등 전통적인 기술적 지표는 사용하지 않습니다.

---

## 5. 동적 리밸런싱 상세 설명

### 5.1 입력 데이터 및 룩백
- 종가(close price)만 사용
- 설정값:
  - `DYNAMIC.MOMENTUM_WINDOW`
  - `DYNAMIC.CORRELATION_WINDOW`
- 라이브 환경에서는 최소 약 3년치 히스토리 확보

---

### 5.2 모멘텀 지표 (핵심)

- 계산 방식: 최근 N일 수익률
```python
returns = prices.pct_change(window)
momentum = returns.iloc[-1]
```

- 단순 수익률 기반
- 가장 최근 시점 값만 사용

---

### 5.3 변동성 지표 (보조)

- 수익률 표준편차 기반
```python
vol = prices.pct_change().rolling(window).std()
```

- 고변동성 종목을 보조적으로 불리하게 평가

---

### 5.4 랭킹 및 종목 선정

1. 종목별 모멘텀 점수 계산
2. 점수 기준 내림차순 정렬
3. 상위 종목 후보군 생성

설정 연동:
- `DYNAMIC.TOP_N`
- `DYNAMIC.MIN_HOLDINGS`
- `DYNAMIC.MAX_HOLDINGS`

---

### 5.5 상관계수 기반 분산 필터

- 수익률 상관계수(Pearson) 계산
- 이미 선택된 종목과의 상관계수가 임계값 이상이면 제외
- 분산 투자 강제

---

### 5.6 리밸런싱 주기

- Daily / Weekly / Monthly / Quarterly
- pandas `resample().last()` 기반
- 해당 날짜 도래 시 리밸런싱 수행

---

### 5.7 목표 비중 → 주문 실행

1. 총 자산 평가
2. 목표 비중 기반 목표 수량 산출
3. 현재 보유 수량 대비 차이만큼 시장가 주문 실행

---

### 5.8 상태 저장

- 마지막 리밸런싱 날짜
- 고점(high water) 기준
- 마지막 목표 비중
- 현금 잔액

같은 날짜 중복 실행 시 주문을 방지합니다.

---

## 6. 라이브 리밸런싱 실행

```bash
python -m live.cli
```

- 단발 실행(`run_once`) 구조
- 외부 스케줄러(cron 등) 연계 전제
- KIS API 기반 실거래 연동

⚠️ `MarketDataProvider` 구현은 별도 보완 필요

---

## 7. 실행 요약

```bash
python -m infra.data_manager.downloader --config data
python -m research.backtester.cli --config backtester
python -m live.cli
```
