import sys
import os
import logging
from datetime import datetime

# 상위 디렉토리의 scripts 폴더를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.update_etf_vectors import ETFVectorUpdater

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'test/logs/test_etf_update_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

def test_etf_update():
    """ETF 업데이트 함수를 테스트합니다."""
    try:
        logging.info("ETF 업데이트 테스트 시작")
        updater = ETFVectorUpdater()
        updater.update_vector_store()
        logging.info("ETF 업데이트 테스트 완료")
    except Exception as e:
        logging.error(f"ETF 업데이트 테스트 실패: {str(e)}")
        raise

if __name__ == "__main__":
    test_etf_update() 