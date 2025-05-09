from typing import Dict, List, Optional, Any
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
import logging
import re

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ETFAnalyzer:
    def __init__(self):
        """ETF 분석기를 초기화합니다."""
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.7
        )
        self.embeddings = OpenAIEmbeddings()
        self._setup_prompts()
        self._load_vector_store()
        
    def _load_vector_store(self):
        """기존 벡터 저장소를 로드합니다."""
        try:
            self.vector_store = Chroma(
                persist_directory="data/etf_vectors",
                embedding_function=self.embeddings
            )
            logger.info(f"벡터 저장소 로드 완료 (총 {self.vector_store._collection.count()} 개 ETF)")
        except Exception as e:
            logging.error(f"벡터 저장소 로드 실패: {str(e)}")
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
    "preferences": ["수익률", "안정성", "성장성" 등 선호도],
    "keywords": ["반도체", "AI", "배터리" 등 관련 키워드]
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
            
    def analyze_query(self, query: str) -> Dict:
        """사용자의 질문을 분석하여 ETF를 추천합니다."""
        try:
            # 1. 질문 분석 (카테고리와 기준 도출)
            result = self.llm.invoke(self.query_analysis_prompt.format(query=query))
            try:
                criteria = json.loads(result.content)
            except json.JSONDecodeError:
                # 기본 검색 기준 설정
                criteria = {
                    "categories": [],
                    "criteria": {
                        "max_expense_ratio": 0.5,
                        "max_tracking_error": 2.0,
                        "min_assets": 100000000000  # 1000억원
                    },
                    "preferences": ["안정성"],
                    "keywords": []
                }
                # 키워드 추출
                if "반도체" in query:
                    criteria["keywords"].append("반도체")
                if "AI" in query or "인공지능" in query:
                    criteria["keywords"].append("AI")
                if "배터리" in query or "2차전지" in query:
                    criteria["keywords"].append("2차전지")
                if "안정" in query:
                    criteria["preferences"].append("안정성")
                if "수익" in query:
                    criteria["preferences"].append("수익성")
                if "성장" in query:
                    criteria["preferences"].append("성장성")
            
            # 2. 벡터 검색을 통한 관련 ETF 찾기
            search_results = self.vector_store.similarity_search_with_score(
                query,
                k=20  # 상위 20개 ETF 검색 (중복 제거 후 10개로 필터링)
            )
            
            # 3. 검색 결과를 점수 기준으로 필터링
            filtered_etfs = {}  # 중복 제거를 위해 딕셔너리 사용
            for doc, score in search_results:
                code = doc.metadata.get('code', '')
                if not code or code in filtered_etfs:
                    continue
                    
                # 메타데이터와 문서 내용 파싱
                content_lines = doc.page_content.strip().split('\n')
                content_dict = {}
                for line in content_lines:
                    line = line.strip()
                    if ':' in line:
                        key, value = line.split(':', 1)
                        content_dict[key.strip()] = value.strip()
                
                filtered_etfs[code] = {
                    'code': code,
                    'name': doc.metadata.get('name', content_dict.get('ETF 이름', '')),
                    'company': doc.metadata.get('company', content_dict.get('운용사', '')),
                    'category': doc.metadata.get('category', content_dict.get('카테고리', '')),
                    'score': float(score),
                    'expense_ratio': float(content_dict.get('운용보수', '0').replace('%', '0')),
                    'tracking_error': float(content_dict.get('추적오차', '0').replace('%', '0')),
                    'total_assets': int(content_dict.get('순자산', '0').replace('원', '0')),
                    'subscribers': int(content_dict.get('가입자수', '0').replace('명', '0')),
                    'listing_date': content_dict.get('상장일', '')
                }
            
            # 4. 점수 기준으로 정렬하고 상위 10개만 선택
            filtered_etfs = list(filtered_etfs.values())
            filtered_etfs.sort(key=lambda x: x['score'])  # 낮은 점수가 더 유사함
            filtered_etfs = filtered_etfs[:10]
            
            if not filtered_etfs:
                return {"error": "기준에 맞는 ETF를 찾을 수 없습니다."}
            
            # 5. 상위 3개 ETF에 대한 상세 분석
            top_etfs = filtered_etfs[:3]
            detailed_analysis = self._analyze_top_etfs(top_etfs, criteria)
            
            # 6. 추천 결과 정리
            recommendations = {
                "top_recommendations": top_etfs,
                "all_recommendations": filtered_etfs,
                "criteria": criteria,
                "analysis": detailed_analysis
            }
            
            return recommendations
            
        except Exception as e:
            logging.error(f"질문 분석 중 오류 발생: {str(e)}")
            return {"error": str(e)}
            
    def _analyze_top_etfs(self, top_etfs: List[Dict], criteria: Dict) -> str:
        """상위 ETF들에 대한 상세 분석을 수행합니다."""
        try:
            # 각 ETF의 주요 특징 추출
            etf_features = []
            for etf in top_etfs:
                features = {
                    'name': etf['name'],
                    'company': etf['company'],
                    'category': etf['category'],
                    'expense_ratio': f"{etf['expense_ratio']}%",
                    'tracking_error': f"{etf['tracking_error']}%",
                    'total_assets': f"{etf['total_assets']:,}원",
                    'subscribers': f"{etf['subscribers']:,}명",
                    'listing_date': etf['listing_date']
                }
                etf_features.append(features)
            
            # LLM을 사용하여 분석
            analysis = self.llm.invoke(self.analysis_prompt.format(
                etf_info=json.dumps(etf_features, ensure_ascii=False)
            ))
            
            return analysis.content
            
        except Exception as e:
            logging.error(f"상세 분석 중 오류 발생: {str(e)}")
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

    def _parse_metadata(self, doc: str) -> Dict[str, Any]:
        """문서 내용에서 ETF 메타데이터를 추출합니다."""
        metadata = {}
        
        # 숫자 데이터 추출을 위한 패턴
        patterns = {
            'expense_ratio': r'운용보수[^\d]*(\d+\.?\d*)',
            'tracking_error': r'추적오차[^\d]*(\d+\.?\d*)',
            'total_assets': r'순자산[^\d]*(\d+\.?\d*)',
            'subscribers': r'가입자수[^\d]*(\d+)',
            'listing_date': r'상장일[^\d]*(\d{4}[-/]\d{2}[-/]\d{2})',
            'company': r'운용사:\s*([^\n]+)',
            'category': r'카테고리:\s*([^\n]+)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, doc)
            if match:
                value = match.group(1).strip()
                if key in ['expense_ratio', 'tracking_error']:
                    metadata[key] = float(value)
                elif key in ['total_assets', 'subscribers']:
                    # 쉼표 제거 후 숫자로 변환
                    value = value.replace(',', '')
                    metadata[key] = int(value)
                else:
                    metadata[key] = value
            else:
                # 기본값 설정
                if key in ['expense_ratio', 'tracking_error']:
                    metadata[key] = 0.0
                elif key in ['total_assets', 'subscribers']:
                    metadata[key] = 0
                else:
                    metadata[key] = ''
        
        return metadata

    def _format_number(self, num: int) -> str:
        """숫자를 천 단위 구분자가 있는 문자열로 변환합니다."""
        return format(num, ',')

    def analyze(self, query: str) -> Dict[str, Any]:
        """사용자 쿼리에 기반하여 ETF를 분석하고 추천합니다."""
        # 쿼리 의도 파악
        intent_prompt = ChatPromptTemplate.from_messages([
            ("system", "사용자의 ETF 검색 의도를 분석하여 다음 정보를 추출하세요:\n"
                      "1. 관심 있는 ETF 카테고리 목록\n"
                      "2. 선호하는 투자 성향 (안정성/수익성)\n"
                      "3. 검색할 키워드\n"
                      "4. ETF 선별 기준 (운용보수 상한, 추적오차 상한, 최소 순자산)"),
            ("human", "{query}")
        ])
        
        intent_response = self.llm.invoke(
            intent_prompt.format_messages(query=query)
        )
        
        try:
            criteria = json.loads(intent_response.content)
        except:
            # 기본 검색 기준
            criteria = {
                "categories": [],
                "preferences": ["안정성"],
                "keywords": [],
                "criteria": {
                    "max_expense_ratio": 0.5,
                    "max_tracking_error": 2.0,
                    "min_assets": 100_000_000_000  # 1000억원
                }
            }
        
        # 벡터 검색 수행 (중복 제거를 위해 20개 검색)
        docs = self.vector_store.similarity_search(query, k=20)
        
        # 검색 결과 파싱 및 필터링
        results = []
        seen_codes = set()
        
        for doc in docs:
            metadata = self._parse_metadata(doc.page_content)
            code = doc.metadata.get('code', '')
            name = doc.metadata.get('name', '')
            
            # 중복 제거
            if code in seen_codes:
                continue
            seen_codes.add(code)
            
            # 기준에 맞는지 확인
            if (metadata['expense_ratio'] > criteria['criteria']['max_expense_ratio'] or
                metadata['tracking_error'] > criteria['criteria']['max_tracking_error'] or
                metadata['total_assets'] < criteria['criteria']['min_assets']):
                continue
                
            results.append({
                'code': code,
                'name': name,
                'company': metadata['company'],
                'category': metadata['category'],
                'score': doc.metadata.get('score', 0.0),
                'expense_ratio': metadata['expense_ratio'],
                'tracking_error': metadata['tracking_error'],
                'total_assets': metadata['total_assets'],
                'subscribers': metadata['subscribers'],
                'listing_date': metadata['listing_date']
            })
            
            if len(results) >= 10:  # 최대 10개까지만 저장
                break
        
        # 유사도 점수로 정렬
        results.sort(key=lambda x: x['score'])
        
        # 분석 생성
        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", "주어진 ETF 목록을 분석하여 다음 내용을 포함하는 종합 분석 리포트를 작성하세요:\n"
                      "1. 각 ETF의 장단점\n"
                      "2. 운용사별 비교\n"
                      "3. 투자 위험 요소\n"
                      "4. 향후 전망\n"
                      "5. 투자 추천 의견"),
            ("human", f"ETF 목록: {json.dumps(results[:3], ensure_ascii=False, indent=2)}")
        ])
        
        analysis_response = self.llm.invoke(
            analysis_prompt.format_messages()
        )
        
        return {
            'top_recommendations': results[:3],
            'all_recommendations': results,
            'criteria': criteria,
            'analysis': analysis_response.content
        } 