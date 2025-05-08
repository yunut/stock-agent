import os
import pandas as pd
import requests
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import schedule
import time
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/etf_update.log'),
        logging.StreamHandler()
    ]
)

load_dotenv()

class ETFVectorUpdater:
    def __init__(self):
        """ETF 벡터 업데이트를 초기화합니다."""
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self._setup_directories()
        
        # KRX API 키 설정
        self.api_key = os.getenv('KRX_API_KEY')
        if not self.api_key:
            raise ValueError("KRX API 키가 설정되지 않았습니다. .env 파일을 확인해주세요.")
        
    def _setup_directories(self):
        """필요한 디렉토리를 생성합니다."""
        os.makedirs('data/etf_vectors', exist_ok=True)
        os.makedirs('data/etf_backup', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
    def _get_test_data(self) -> pd.DataFrame:
        """테스트용 ETF 데이터를 생성합니다."""
        test_data = {
            'code': ['069500', '114800', '069660'],
            'name': ['KODEX 200', 'TIGER 코스닥150', 'KODEX 미국S&P500'],
            'market': ['KODEX', 'TIGER', 'KODEX'],
            'category': ['국내주식', '국내주식', '해외주식'],
            'listing_date': ['2002-10-07', '2003-01-06', '2007-11-30'],
            'expense_ratio': [0.05, 0.05, 0.15],
            'tracking_error': [0.1, 0.1, 0.2],
            'total_assets': [1000000000000, 500000000000, 800000000000],
            'subscribers': [10000, 5000, 8000]
        }
        return pd.DataFrame(test_data)
        
    def _fetch_etf_data(self) -> pd.DataFrame:
        """KRX에서 ETF 데이터를 가져옵니다."""
        try:
            # 샘플 API 스펙의 테스트 날짜 사용
            test_date = '20200414'
            
            url = "http://data-dbg.krx.co.kr/svc/apis/etp/etf_bydd_trd"
            params = {
                'basDd': test_date
            }
            headers = {
                "User-Agent": "Mozilla/5.0",
                "AUTH_KEY": self.api_key
            }
            
            logging.info(f"API 요청 URL: {url}")
            logging.info(f"API 요청 파라미터: {params}")
            logging.info(f"API 요청 헤더: {headers}")
            
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            
            etf_data = response.json()
            logging.info(f"API 응답: {etf_data}")
            
            if 'OutBlock_1' not in etf_data:
                if 'error' in etf_data:
                    raise ValueError(f"API 오류: {etf_data['error']}")
                raise ValueError("API 응답 형식이 예상과 다릅니다.")
                
            df = pd.DataFrame(etf_data['OutBlock_1'])
            
            # 컬럼 이름 매핑
            column_mapping = {
                'ISU_CD': 'code',          # 종목코드
                'ISU_NM': 'name',          # 종목명
                'MKT_NM': 'market',        # 시장구분
                'SECUGRP_NM': 'category',  # 증권구분
                'LIST_DD': 'listing_date'  # 상장일
            }
            
            # 필요한 컬럼만 선택하고 이름 변경
            df = df.rename(columns=column_mapping)
            df = df[list(column_mapping.values())]
            
            # 추가 정보 수집
            df['expense_ratio'] = 0.0
            df['tracking_error'] = 0.0
            df['total_assets'] = 0
            df['subscribers'] = 0
            
            return df
            
        except Exception as e:
            logging.error(f"ETF 데이터 가져오기 실패: {str(e)}")
            raise
            
    def _create_documents(self, df: pd.DataFrame) -> list:
        """ETF 정보를 Document 객체로 변환합니다."""
        documents = []
        for _, row in df.iterrows():
            content = f"""
            ETF 이름: {row['name']}
            운용사: {row['market']}
            카테고리: {row['category']}
            상장일: {row['listing_date']}
            운용보수: {row['expense_ratio']}%
            추적오차: {row['tracking_error']}%
            순자산: {row['total_assets']}원
            가입자수: {row['subscribers']}명
            """
            metadata = {
                "code": row['code'],
                "name": row['name'],
                "company": row['market'],
                "category": row['category']
            }
            documents.append(Document(page_content=content, metadata=metadata))
            
        return documents
        
    def update_vector_store(self):
        """벡터 저장소를 업데이트합니다."""
        try:
            logging.info("ETF 데이터 업데이트 시작")
            
            # 1. ETF 기본 정보 가져오기
            df = self._fetch_etf_data()
            
            # 2. 데이터 백업
            backup_file = f"data/etf_backup/etf_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(backup_file, index=False)
            
            # 3. Document 객체 생성
            documents = self._create_documents(df)
            
            # 4. 벡터 저장소 업데이트
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory="data/etf_vectors"
            )
            
            logging.info(f"ETF 데이터 업데이트 완료 (총 {len(df)}개 ETF)")
            
        except Exception as e:
            logging.error(f"ETF 데이터 업데이트 실패: {str(e)}")
            raise
            
def main():
    """메인 함수"""
    updater = ETFVectorUpdater()
    
    # 매일 자정에 업데이트
    schedule.every().day.at("00:00").do(updater.update_vector_store)
    
    # 처음 실행 시 즉시 업데이트
    updater.update_vector_store()
    
    # 스케줄러 실행
    while True:
        schedule.run_pending()
        time.sleep(60)
        
if __name__ == "__main__":
    main() 