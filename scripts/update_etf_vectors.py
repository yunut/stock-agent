import os
import pandas as pd
import requests
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
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
        
    def _setup_directories(self):
        """필요한 디렉토리를 생성합니다."""
        os.makedirs('data/etf_vectors', exist_ok=True)
        os.makedirs('data/etf_backup', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
    def _fetch_etf_data(self) -> pd.DataFrame:
        """KRX에서 ETF 데이터를 가져옵니다."""
        try:
            url = "http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Referer": "http://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201020101"
            }
            data = {
                "bld": "dbms/MDC/STAT/standard/MDCSTAT01501",
                "locale": "ko_KR",
                "mktId": "STK",
                "share": "1",
                "csvxls_isNo": "false",
                "name": "fileDown",
                "url": "dbms/MDC/STAT/standard/MDCSTAT01501"
            }
            
            response = requests.post(url, headers=headers, data=data)
            response.raise_for_status()
            
            etf_data = response.json()
            if 'output' not in etf_data:
                raise ValueError("API 응답 형식이 예상과 다릅니다.")
                
            df = pd.DataFrame(etf_data['output'])
            
            # 필요한 컬럼만 선택하고 이름 변경
            df = df[['ISU_CD', 'ISU_NM', 'MKT_NM', 'SECUGRP_NM', 'LIST_DD']]
            df.columns = ['code', 'name', 'market', 'category', 'listing_date']
            
            # 추가 정보 수집
            df['expense_ratio'] = 0.0
            df['tracking_error'] = 0.0
            df['total_assets'] = 0
            df['subscribers'] = 0
            
            return df
            
        except Exception as e:
            logging.error(f"ETF 데이터 가져오기 실패: {str(e)}")
            raise
            
    def _get_etf_details(self, code: str) -> dict:
        """네이버 금융에서 ETF 상세 정보를 가져옵니다."""
        try:
            url = f"https://finance.naver.com/etf/etfDetail.nhn?code={code}"
            headers = {
                "User-Agent": "Mozilla/5.0"
            }
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            # 여기에 BeautifulSoup을 사용하여 상세 정보 파싱
            # 예시: 운용보수, 추적오차, 순자산, 가입자수 등
            
            return {
                "expense_ratio": 0.0,
                "tracking_error": 0.0,
                "total_assets": 0,
                "subscribers": 0
            }
            
        except Exception as e:
            logging.error(f"ETF 상세 정보 가져오기 실패 (코드: {code}): {str(e)}")
            return None
            
    def _update_etf_details(self, df: pd.DataFrame) -> pd.DataFrame:
        """ETF 상세 정보를 업데이트합니다."""
        for idx, row in df.iterrows():
            details = self._get_etf_details(row['code'])
            if details:
                df.at[idx, 'expense_ratio'] = details['expense_ratio']
                df.at[idx, 'tracking_error'] = details['tracking_error']
                df.at[idx, 'total_assets'] = details['total_assets']
                df.at[idx, 'subscribers'] = details['subscribers']
                
        return df
        
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
            
            # 2. ETF 상세 정보 업데이트
            df = self._update_etf_details(df)
            
            # 3. 데이터 백업
            backup_file = f"data/etf_backup/etf_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(backup_file, index=False)
            
            # 4. Document 객체 생성
            documents = self._create_documents(df)
            
            # 5. 벡터 저장소 업데이트
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