# Stock Investment Agent

주식 투자 의견을 제공하는 AI 에이전트입니다.

## 기능

### 주식 투자
- 특정 종목 분석
- 뉴스, 재무재표, 차트 기반 투자 의견 제공

## 설치 방법

```bash
pip install -r requirements.txt
```

## 사용 방법

```python
from agents.stock_agent import StockAgent
agent = StockAgent()
analysis = agent.analyze_stock("삼성전자")
``` 