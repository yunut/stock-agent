import os
import pandas as pd
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import schedule
import time
import logging
import yfinance as yf

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
        
    def _fetch_yahoo_finance_info(self, code: str) -> dict:
        """야후 파이낸스에서 ETF 상세 정보를 가져옵니다."""
        ticker = f"{code}.KS"
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            # 1년 수익률 계산
            hist = stock.history(period="1y")
            if not hist.empty:
                start_price = hist['Close'].iloc[0]
                end_price = hist['Close'].iloc[-1]
                return_1y = ((end_price - start_price) / start_price) * 100
            else:
                return_1y = 0.0
            return {
                'market_cap': info.get('marketCap', 0),
                'current_price': info.get('regularMarketPrice', 0),
                'return_1y': return_1y
            }
        except Exception as e:
            logging.warning(f"{ticker} 야후 파이낸스 조회 실패: {str(e)}")
            return {'market_cap': 0, 'current_price': 0, 'return_1y': 0.0}

    def _fetch_etf_data(self) -> pd.DataFrame:
        """네이버 금융에서 ETF 전체 리스트와 주요 정보를 가져옵니다."""
        try:
            url = "https://finance.naver.com/api/sise/etfItemList.nhn"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            etf_list = data['result']['etfItemList']
            df = pd.DataFrame(etf_list)

            # 컬럼명 통일 및 필요한 컬럼만 추출
            df = df.rename(columns={
                'itemcode': 'code',
                'itemname': 'name',
                'market': 'market',
                'provider': 'company',
                'nowVal': 'current_price',
                'marketSum': 'market_cap',
                'nav': 'nav',
                'threeMonthEarnRate': 'return_3m',
                'quant': 'volume',
                'amount': 'amount',
                'list_shares': 'list_shares',
                'category': 'category',
                'expenseRatio': 'expense_ratio',
                'trackingError': 'tracking_error',
                'listedShare': 'listed_share',
                'listedDate': 'listing_date'
            })
            # 일부 컬럼이 없을 수 있으니, 주요 컬럼만 남기고 없는 컬럼은 기본값
            for col in ['code', 'name', 'company', 'market', 'current_price', 'market_cap', 'nav', 'return_3m', 'volume', 'amount', 'expense_ratio', 'tracking_error', 'listing_date']:
                if col not in df.columns:
                    df[col] = 0 if col not in ['code', 'name', 'company', 'market', 'listing_date'] else ''
            # 필요 컬럼만 추출
            df = df[['code', 'name', 'company', 'market', 'current_price', 'market_cap', 'nav', 'return_3m', 'volume', 'amount', 'expense_ratio', 'tracking_error', 'listing_date']]
            return df
        except Exception as e:
            logging.error(f"네이버 ETF 데이터 가져오기 실패: {str(e)}")
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
                "company": row['company'],
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
    # 한 번만 실행
    updater.update_vector_store()
        
if __name__ == "__main__":
    main() 