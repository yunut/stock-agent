import sys
import os
import logging
from datetime import datetime

# 상위 디렉토리의 agents 폴더를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.etf_analyzer import ETFAnalyzer

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'test/logs/test_etf_analyzer_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

def test_etf_analyzer():
    """ETF 분석기 테스트를 수행합니다."""
    try:
        logging.info("ETF 분석기 테스트 시작")
        
        # ETF 분석기 초기화
        analyzer = ETFAnalyzer()
        
        # 테스트 케이스 1: 반도체 ETF 추천
        query1 = "반도체 관련 ETF 중에서 운용보수가 낮은 것 추천해주세요"
        logging.info(f"테스트 케이스 1: {query1}")
        result1 = analyzer.analyze_query(query1)
        logging.info(f"결과 1: {result1}")
        
        # 테스트 케이스 2: 안정적인 수익 추구
        query2 = "안정적인 수익을 추구하는 ETF 추천해주세요"
        logging.info(f"테스트 케이스 2: {query2}")
        result2 = analyzer.analyze_query(query2)
        logging.info(f"결과 2: {result2}")
        
        # 테스트 케이스 3: AI 관련 ETF
        query3 = "AI 관련 ETF 중에서 순자산이 큰 것 추천해주세요"
        logging.info(f"테스트 케이스 3: {query3}")
        result3 = analyzer.analyze_query(query3)
        logging.info(f"결과 3: {result3}")
        
        logging.info("ETF 분석기 테스트 완료")
        
    except Exception as e:
        logging.error(f"ETF 분석기 테스트 실패: {str(e)}")
        raise

if __name__ == "__main__":
    test_etf_analyzer() 