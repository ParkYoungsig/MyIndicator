# MyIndicator (Static + Dynamic Asset Allocation System)

본 프로젝트는 **정적(Static) 자산배분 전략**과  
**동적(Dynamic) 모멘텀 기반 리밸런싱 전략**을 결합한  
**백테스트 및 라이브 리밸런싱 프레임워크**입니다.

---

## 1. 프로젝트 개요

- 로컬 parquet 데이터 기반 백테스트
- 설정(YAML) 중심 전략 제어
- Static / Dynamic 전략을 동일한 구조로 운용
- 실거래(KIS API) 연계 가능한 라이브 리밸런싱 구조

---

## 2. 프로젝트 구조

```
MyIndicator-main/
├── config/
├── infra/
├── research/
│   └── backtester/
├── core/
├── live/
└── data/
```

---

## 3. 데이터 다운로드

```bash
python -m infra.data_manager.downloader --config data
```

- `config/data.yaml` 기준 데이터 수집
- parquet 형식 로컬 저장
- 증분 다운로드 지원

---

## 4. 백테스트 실행

```bash
python -m research.backtester.cli --config backtester
```

- Static + Dynamic 통합 백테스트
- 로컬 parquet 데이터 사용

---

## 5. 리밸런싱 전략 개요

본 프로젝트의 리밸런싱 전략은 **정적 리밸런싱**과  
**동적 리밸런싱**으로 구분됩니다.

---

## 6. 정적(Static) 리밸런싱 – Indicator 요약

### 핵심 개념
> **정적 리밸런싱은 지표(Indicator)를 사용하지 않는 전략**

### 사용 요소
- 종가(Close price): 자산 가치 평가용
- 시간(Time): 리밸런싱 트리거

### 사용하지 않는 요소
- 모멘텀, 변동성, 상관계수
- RSI, MACD, 이동평균
- 거래량 기반 지표

### 특징
- 사전에 정의된 자산과 비중 유지
- Monthly / Quarterly 등 시간 주기 기반 리밸런싱
- 안정적인 자산군 운용 목적

---

## 7. 동적(Dynamic) 리밸런싱 – Indicator 요약

### 핵심 개념
> **가격 기반 지표로 종목을 선택하고  
> 주기적으로 포트폴리오를 재구성하는 전략**

### 사용된 Indicator 요약 표

| Indicator | 사용 여부 | 역할 |
|----|----|----|
| 모멘텀 (N일 수익률) | ✅ 핵심 | 종목 선택 기준 |
| 변동성 (표준편차) | ✅ 보조 | 리스크 조절 |
| 상관계수 | ✅ | 분산 필터 |
| 랭킹 (Top-N) | ✅ | 종목 선정 |
| RSI / MACD | ❌ | 미사용 |
| 이동평균 | ❌ | 미사용 |
| 거래량 지표 | ❌ | 미사용 |

---

## 8. 정적 vs 동적 리밸런싱 Indicator 비교

| 구분 | 정적 리밸런싱 | 동적 리밸런싱 |
|----|----|----|
| Indicator 사용 | ❌ | ✅ |
| 판단 기준 | 시간 | 가격 신호 |
| 주요 지표 | 없음 | 모멘텀, 상관계수 |
| 전략 성격 | 패시브 | 세미 액티브 |
| 목적 | 안정성 | 초과수익 |

---

## 9. 라이브 리밸런싱 실행

```bash
python -m live.cli
```

- 단발 실행(run_once)
- 외부 스케줄러 연계 전제
- KIS API 기반 주문 실행

⚠️ `MarketDataProvider` 구현은 별도 보완 필요

---

## 10. 실행 요약

```bash
python -m infra.data_manager.downloader --config data
python -m research.backtester.cli --config backtester
python -m live.cli
```

---

## 11. 한 문장 요약

> 정적 리밸런싱은 지표 없이 시간 기준으로 비중을 유지하는 전략이며,  
> 동적 리밸런싱은 모멘텀·변동성·상관계수 등 가격 기반 indicator를 활용해  
> 종목을 선택하고 주기적으로 리밸런싱하는 전략이다.
