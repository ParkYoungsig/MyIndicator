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

- config/data.yaml 기준으로 주가 데이터를 다운로드
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

## 4. 동적 리밸런싱 지표

| 지표 | 설명 |
|----|----|
| 모멘텀 | N일 수익률 |
| 변동성 | 수익률 표준편차 |
| 상관계수 | 종목 간 상관 |
| 랭킹 | Top-N 종목 선정 |

RSI, MACD 등 전통적 기술적 지표는 사용하지 않습니다.

---

## 5. 라이브 리밸런싱

```bash
python -m live.cli
```

- 단발 실행(run_once)
- 외부 스케줄러(cron) 연계 전제
- KIS API 기반 주문 전송

⚠️ MarketDataProvider 구현은 별도 보완 필요

---

## 6. 실행 요약

```bash
python -m infra.data_manager.downloader --config data
python -m research.backtester.cli --config backtester
python -m live.cli
```
