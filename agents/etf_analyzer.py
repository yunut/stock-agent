from typing import Dict, List, Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.vectorstores import Chroma
from langchain.schema import Document
import requests
from bs4 import BeautifulSoup
import json
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()

class ETFAnalyzer:
    def __init__(self, data_path: str = "data/etf_list.csv"):
        """ETF 분석기를 초기화합니다."""
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.7
        )
        self.embeddings = OpenAIEmbeddings()
        self._setup_prompts()
        self._load_etf_data(data_path)
        self._setup_vector_store()
        
    def _load_etf_data(self, data_path: str):
        """ETF 데이터를 로드합니다."""
        if not os.path.exists(data_path):
            self._download_etf_data(data_path)
            
        self.etf_df = pd.read_csv(data_path)
        self._setup_etf_categories()
        
    def _download_etf_data(self, data_path: str):
        """KRX에서 ETF 데이터를 다운로드합니다."""
        try:
            # KRX API를 통해 ETF 목록 가져오기
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
            
            # 응답을 DataFrame으로 변환
            etf_data = response.json()
            df = pd.DataFrame(etf_data['output'])
            
            # 필요한 컬럼만 선택
            df = df[['ISU_CD', 'ISU_NM', 'MKT_NM', 'SECUGRP_NM', 'LIST_DD']]
            df.columns = ['code', 'name', 'market', 'category', 'listing_date']
            
            # 추가 정보 수집
            df['expense_ratio'] = 0.0
            df['tracking_error'] = 0.0
            df['total_assets'] = 0
            df['subscribers'] = 0
            
            # 데이터 저장
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            df.to_csv(data_path, index=False)
            
        except Exception as e:
            print(f"ETF 데이터 다운로드 중 오류 발생: {str(e)}")
            raise
            
    def _setup_prompts(self):
        """프롬프트 템플릿을 설정합니다."""
        self.query_analysis_prompt = PromptTemplate(
            template="""사용자의 ETF 투자 관련 질문을 분석하여 추천 기준을 도출해주세요.

질문: {query}

다음 형식의 JSON으로 응답해주세요:
{{
    "categories": ["국내주식", "미국주식", "반도체", "2차전지" 등 관련 카테고리],
    "criteria": {{
        "min_return_1y": 연간 최소 수익률(%),
        "min_assets": 최소 순자산(원),
        "max_expense_ratio": 최대 운용보수(%),
        "max_tracking_error": 최대 추적오차(%)
    }},
    "preferences": ["수익률", "안정성", "성장성" 등 선호도]
}}

응답:"""
        )

        self.analysis_prompt = PromptTemplate(
            template="""다음 ETF 정보를 분석하여 상세한 투자 의견을 제시해주세요:

ETF 정보: {etf_info}

다음 항목을 포함하여 분석해주세요:
1. 각 ETF의 장단점
2. 운용사별 비교
3. 투자 위험 요소
4. 향후 전망
5. 투자 추천 의견

응답:"""
        )
        
    def _setup_etf_categories(self):
        """ETF 카테고리 정보를 설정합니다."""
        # 카테고리 매핑
        category_mapping = {
            "국내주식": ["KOSPI200", "KOSPI", "KOSDAQ"],
            "미국주식": ["S&P500", "NASDAQ", "미국"],
            "반도체": ["반도체", "반도체산업"],
            "2차전지": ["2차전지", "배터리"],
            "헬스케어": ["바이오", "헬스케어", "의료"],
            "인공지능": ["AI", "인공지능"],
            "친환경": ["친환경", "그린", "ESG"],
            "메타버스": ["메타버스", "가상현실"],
            "로봇": ["로봇", "자동화"],
            "게임": ["게임", "엔터테인먼트"]
        }
        
        # ETF 코드 -> 카테고리 매핑
        self.etf_to_category = {}
        for _, row in self.etf_df.iterrows():
            code = row['code']
            name = row['name']
            category = row['category']
            
            self.etf_to_category[code] = []
            for cat, keywords in category_mapping.items():
                if any(keyword in name or keyword in category for keyword in keywords):
                    self.etf_to_category[code].append(cat)
                    
        # 카테고리 -> ETF 코드 매핑
        self.etf_categories = {}
        for code, categories in self.etf_to_category.items():
            for category in categories:
                if category not in self.etf_categories:
                    self.etf_categories[category] = []
                self.etf_categories[category].append(code)
                
    def get_etf_info(self, code: str) -> Optional[Dict]:
        """ETF 정보를 가져옵니다."""
        try:
            etf = self.etf_df[self.etf_df['code'] == code].iloc[0]
            
            return {
                "code": code,
                "name": etf['name'],
                "company": etf['market'],
                "category": etf['category'],
                "listing_date": etf['listing_date'],
                "expense_ratio": etf['expense_ratio'],
                "tracking_error": etf['tracking_error'],
                "total_assets": etf['total_assets'],
                "subscribers": etf['subscribers']
            }
            
        except Exception as e:
            print(f"ETF 정보 조회 중 오류 발생: {str(e)}")
            return None
            
    def _setup_vector_store(self):
        """ETF 정보를 벡터 저장소에 저장합니다."""
        # ETF 정보를 Document 객체로 변환
        documents = []
        for _, row in self.etf_df.iterrows():
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
            
        # 벡터 저장소 생성
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory="data/etf_vectors"
        )
        
    def analyze_query(self, query: str) -> Dict:
        """사용자의 질문을 분석하여 추천 기준을 도출합니다."""
        try:
            # 1. 질문 분석 (카테고리와 기준 도출)
            result = self.llm.invoke(self.query_analysis_prompt.format(query=query))
            criteria = json.loads(result.content)
            
            # 2. 벡터 검색을 통한 관련 ETF 찾기
            search_results = self.vector_store.similarity_search_with_score(
                query,
                k=10  # 상위 10개 ETF 검색
            )
            
            # 3. 검색 결과를 점수 기준으로 필터링
            filtered_etfs = []
            for doc, score in search_results:
                info = self.get_etf_info(doc.metadata['code'])
                if info and self._meets_criteria(info, criteria['criteria']):
                    filtered_etfs.append({
                        'code': info['code'],
                        'name': info['name'],
                        'company': info['company'],
                        'score': score,  # 유사도 점수
                        'expense_ratio': info['expense_ratio'],
                        'tracking_error': info['tracking_error'],
                        'total_assets': info['total_assets'],
                        'subscribers': info['subscribers']
                    })
            
            if not filtered_etfs:
                return {"error": "기준에 맞는 ETF를 찾을 수 없습니다."}
            
            # 4. 상위 3개 ETF에 대한 상세 분석
            top_etfs = sorted(filtered_etfs, key=lambda x: x['score'], reverse=True)[:3]
            detailed_analysis = self._analyze_top_etfs(top_etfs, criteria)
            
            return {
                "recommendations": filtered_etfs,
                "detailed_analysis": detailed_analysis,
                "criteria": criteria
            }
            
        except Exception as e:
            print(f"질문 분석 중 오류 발생: {str(e)}")
            return {"error": str(e)}
            
    def _meets_criteria(self, etf_info: Dict, criteria: Dict) -> bool:
        """ETF가 주어진 기준을 만족하는지 확인합니다."""
        try:
            # 비용 기준
            if 'max_expense_ratio' in criteria:
                if etf_info['expense_ratio'] > criteria['max_expense_ratio']:
                    return False
                    
            # 추적오차 기준
            if 'max_tracking_error' in criteria:
                if etf_info['tracking_error'] > criteria['max_tracking_error']:
                    return False
                    
            # 순자산 기준
            if 'min_assets' in criteria:
                if etf_info['total_assets'] < criteria['min_assets']:
                    return False
                    
            # 가입자 수 기준
            if 'min_subscribers' in criteria:
                if etf_info['subscribers'] < criteria['min_subscribers']:
                    return False
                    
            return True
            
        except:
            return False
            
    def _analyze_top_etfs(self, top_etfs: List[Dict], criteria: Dict) -> str:
        """상위 ETF들에 대한 상세 분석을 수행합니다."""
        try:
            # 각 ETF의 주요 특징 추출
            etf_features = []
            for etf in top_etfs:
                features = {
                    'name': etf['name'],
                    'company': etf['company'],
                    'expense_ratio': etf['expense_ratio'],
                    'tracking_error': etf['tracking_error'],
                    'total_assets': etf['total_assets'],
                    'subscribers': etf['subscribers']
                }
                etf_features.append(features)
            
            # LLM을 사용하여 분석
            analysis = self.llm.invoke(self.analysis_prompt.format(
                etf_features=json.dumps(etf_features, ensure_ascii=False),
                criteria=json.dumps(criteria, ensure_ascii=False)
            ))
            
            return analysis.content
            
        except Exception as e:
            print(f"상세 분석 중 오류 발생: {str(e)}")
            return "상세 분석을 수행할 수 없습니다."
            
    def _calculate_score(self, etf_info: Dict, criteria: Dict) -> float:
        """ETF 정보와 기준에 따라 점수를 계산합니다."""
        try:
            score = 0.0
            
            # 비용 관련 점수 (최대 30점)
            if 'expense_ratio' in etf_info:
                expense_ratio = float(etf_info['expense_ratio'])
                max_expense = criteria.get('max_expense_ratio', 0.5)
                if expense_ratio <= max_expense:
                    score += (max_expense - expense_ratio) * 30
            
            # 성과 관련 점수 (최대 40점)
            if '1년수익률' in etf_info:
                return_1y = float(etf_info['1년수익률'].replace('%', ''))
                min_return = criteria.get('min_return_1y', 0)
                if return_1y >= min_return:
                    score += (return_1y - min_return) * 40
            
            # 규모 관련 점수 (최대 20점)
            min_assets = criteria.get('min_assets', 100000000000)
            if etf_info['total_assets'] >= min_assets:
                score += 20
            
            # 가입자 수 점수 (최대 10점)
            min_subscribers = criteria.get('min_subscribers', 1000)
            if etf_info['subscribers'] >= min_subscribers:
                score += 10
            
            return score
            
        except:
            return 0.0 