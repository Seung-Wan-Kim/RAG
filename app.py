# 10_streamlit_web_interface.py

import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
import faiss
import streamlit as st
from langchain.schema import Document
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch

print("10단계: Streamlit 웹 인터페이스 구현 시작")

# 하이브리드 임베딩 클래스 정의
class HybridEmbeddings:
    def __init__(self):
        print("하이브리드 임베딩 모델 초기화 중...")
        
        # 다국어 BERT 모델 (한국어 지원)
        self.bert_model_name = "klue/bert-base"
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
        self.bert_model = AutoModel.from_pretrained(self.bert_model_name)
        print(f"BERT 모델 로드 완료: {self.bert_model_name}")
        
        # SentenceTransformer 모델 (의미적 임베딩)
        self.st_model_name = "jhgan/ko-sroberta-multitask"
        self.st_model = SentenceTransformer(self.st_model_name)
        print(f"SentenceTransformer 모델 로드 완료: {self.st_model_name}")
        
        # 설정
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.bert_model.to(self.device)
        print(f"모델이 {self.device}에서 실행됩니다.")
        
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def get_bert_embeddings(self, texts):
        encoded_input = self.bert_tokenizer(texts, padding=True, truncation=True, 
                                            max_length=512, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            model_output = self.bert_model(**encoded_input)
        
        sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        
        return sentence_embeddings.cpu().numpy()
    
    def get_st_embeddings(self, texts):
        embeddings = self.st_model.encode(texts, convert_to_tensor=True)
        return embeddings.cpu().numpy()
    
    def embed_documents(self, texts):
        if not texts:
            return []
        
        batch_size = 16
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            bert_embeddings = self.get_bert_embeddings(batch_texts)
            st_embeddings = self.get_st_embeddings(batch_texts)
            
            hybrid_embeddings = np.concatenate([bert_embeddings, st_embeddings], axis=1)
            hybrid_embeddings = hybrid_embeddings / np.linalg.norm(hybrid_embeddings, axis=1, keepdims=True)
            all_embeddings.extend(hybrid_embeddings.tolist())
            
        return all_embeddings
    
    def embed_query(self, text):
        return self.embed_documents([text])[0]

# 정상범위 데이터 함수
def get_normal_ranges():
    """
    한국인 기준 혈액검사 정상범위 데이터를 반환합니다.
    """
    normal_ranges = {
        "백혈구": {"min": 4000, "max": 10000, "unit": "cells/μL"},
        "적혈구": {"min": 4.5, "max": 5.9, "unit": "million cells/μL"},
        "혈색소": {"min": 13.5, "max": 17.5, "unit": "g/dL"},
        "헤마토크릿": {"min": 41, "max": 53, "unit": "%"},
        "혈소판": {"min": 150000, "max": 450000, "unit": "cells/μL"},
        "AST": {"min": 0, "max": 40, "unit": "U/L"},
        "ALT": {"min": 0, "max": 40, "unit": "U/L"},
        "GGT": {"min": 0, "max": 60, "unit": "U/L"},
        "총 빌리루빈": {"min": 0.2, "max": 1.2, "unit": "mg/dL"},
        "알부민": {"min": 3.5, "max": 5.2, "unit": "g/dL"},
        "크레아티닌": {"min": 0.7, "max": 1.3, "unit": "mg/dL"},
        "BUN": {"min": 8, "max": 20, "unit": "mg/dL"},
        "총 콜레스테롤": {"min": 130, "max": 200, "unit": "mg/dL"},
        "HDL": {"min": 40, "max": 60, "unit": "mg/dL"},
        "LDL": {"min": 0, "max": 130, "unit": "mg/dL"},
        "중성지방": {"min": 0, "max": 150, "unit": "mg/dL"},
        "공복혈당": {"min": 70, "max": 99, "unit": "mg/dL"},
        "HbA1c": {"min": 4.0, "max": 5.6, "unit": "%"}
    }
    return normal_ranges

# RAG 검색 엔진 클래스
class BloodTestRAGSearchEngine:
    def __init__(self, vector_db=None, processed_data_path=None):
        print("RAG 검색 엔진 초기화 중...")
        self.hybrid_embeddings = HybridEmbeddings()
        self.vector_db = vector_db
        
        # 정상범위 데이터 로드
        self.normal_ranges = get_normal_ranges()
        
        # 원본 데이터 로드 (선택적)
        if processed_data_path:
            # 파일 확장자 확인
            file_extension = processed_data_path.split('.')[-1].lower()
            if file_extension == 'csv':
                self.processed_data = pd.read_csv(processed_data_path)
            elif file_extension in ['xlsx', 'xls']:
                self.processed_data = pd.read_excel(processed_data_path)
        else:
            self.processed_data = None
            
        print("RAG 검색 엔진 초기화 완료")
    
    def search(self, query, k=5):
        """
        사용자 쿼리에 가장 관련성 높은 상위 k개의 문서를 검색합니다.
        """
        print(f"쿼리 검색 중: {query}")
        
        # 쿼리 임베딩 생성
        query_embedding = self.hybrid_embeddings.embed_query(query)
        
        # 7단계에서 생성한 벡터 DB가 딕셔너리 형태인 경우 대응
        if isinstance(self.vector_db, dict):
            # 벡터 DB가 딕셔너리인 경우 직접 FAISS 검색 수행
            query_vector = np.array([query_embedding]).astype('float32')
            faiss.normalize_L2(query_vector)  # 정규화
            
            # FAISS 인덱스로 검색
            distances, indices = self.vector_db['index'].search(query_vector, k)
            
            results = []
            for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
                if idx < len(self.vector_db['documents']):
                    doc = self.vector_db['documents'][idx]
                    # 유사도 점수 계산 (거리를 유사도로 변환)
                    similarity_score = 1.0 - float(distance) / 2.0
                    
                    result = {
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'similarity_score': similarity_score
                    }
                    results.append(result)
            
            return results
        else:
            # 벡터 DB가 없는 경우 임시 결과 생성
            print("벡터 DB가 없거나 호환되지 않습니다. 임시 결과를 생성합니다.")
            # 임시 결과 생성
            results = []
            for i in range(3):
                result = {
                    'content': f"임시 검색 결과 {i+1}",
                    'metadata': {
                        '진단명': '빈혈증' if i == 0 else f'진단명{i+1}',
                        '질병코드': 'D50' if i == 0 else f'코드{i+1}'
                    },
                    'similarity_score': 0.9 - (i * 0.1)
                }
                results.append(result)
            return results
    
    def parse_query_results(self, query):
        """
        쿼리 문자열에서 혈액검사 결과값을 추출합니다.
        """
        print("쿼리에서 혈액검사 결과 추출 중...")
        
        # 결과값을 저장할 딕셔너리
        parsed_results = {}
        
        # 구분자로 분리
        items = query.split(',')
        
        for item in items:
            item = item.strip()
            
            # 검사항목과 결과값 분리
            if ':' in item:
                key, value = item.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # 숫자만 추출
                try:
                    # 단위가 포함된 경우 숫자만 추출
                    numeric_value = ''.join(c for c in value if c.isdigit() or c == '.')
                    parsed_results[key] = float(numeric_value)
                except:
                    parsed_results[key] = value
            else:
                # "검사항목 값" 형태로 입력된 경우 분리 시도
                parts = item.split()
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    
                    try:
                        parsed_results[key] = float(value)
                    except:
                        parsed_results[key] = value
        
        return parsed_results
    
    def analyze_test_results(self, test_results):
        """
        혈액검사 결과를 분석하여 정상범위와 비교합니다.
        """
        print("혈액검사 결과 분석 중...")
        
        analysis = {}
        
        for item, value in test_results.items():
            # 정상범위 데이터에 있는 항목인지 확인
            if item in self.normal_ranges:
                normal_range = self.normal_ranges[item]
                
                if isinstance(value, (int, float)):
                    if value < normal_range["min"]:
                        status = "낮음"
                        deviation = f"{((normal_range['min'] - value) / normal_range['min'] * 100):.1f}% 낮음"
                    elif value > normal_range["max"]:
                        status = "높음"
                        deviation = f"{((value - normal_range['max']) / normal_range['max'] * 100):.1f}% 높음"
                    else:
                        status = "정상"
                        deviation = "정상범위 내"
                    
                    analysis[item] = {
                        "value": value,
                        "status": status,
                        "deviation": deviation,
                        "normal_range": f"{normal_range['min']} - {normal_range['max']} {normal_range['unit']}"
                    }
            else:
                # 정상범위 데이터에 없는 항목
                if isinstance(value, (int, float)):
                    analysis[item] = {
                        "value": value,
                        "status": "정보없음",
                        "deviation": "정상범위 정보 없음",
                        "normal_range": "정보 없음"
                    }
        
        return analysis
    
    def get_possible_diagnoses(self, search_results, test_analysis):
        """
        검색 결과와 검사 분석을 기반으로 가능한 진단명을 추출합니다.
        """
        print("가능한 진단명 추출 중...")
        
        diagnoses = {}
        
        # 검색 결과에서 진단명 추출
        for result in search_results:
            metadata = result['metadata']
            if '진단명' in metadata and metadata['진단명'] != 'N/A':
                diagnosis_name = metadata['진단명']
                diagnosis_code = metadata['질병코드'] if '질병코드' in metadata else 'N/A'
                
                if diagnosis_name not in diagnoses:
                    diagnoses[diagnosis_name] = {
                        "code": diagnosis_code,
                        "count": 0,
                        "similarity_scores": [],
                        "matched_patterns": []
                    }
                
                diagnoses[diagnosis_name]["count"] += 1
                diagnoses[diagnosis_name]["similarity_scores"].append(result['similarity_score'])
        
        # 평균 유사도 점수 계산 및 정렬
        for diagnosis in diagnoses.values():
            diagnosis["avg_similarity"] = sum(diagnosis["similarity_scores"]) / len(diagnosis["similarity_scores"])
        
        # 진단명 정렬 (빈도수 및 유사도 점수 기준)
        sorted_diagnoses = sorted(
            diagnoses.items(),
            key=lambda x: (x[1]["count"], x[1]["avg_similarity"]),
            reverse=True
        )
        
        return sorted_diagnoses

# Claude 응답 생성기 클래스
class ClaudeResponseGenerator:
    def __init__(self, api_key=None):
        """
        Claude API 키로 초기화합니다.
        """
        print("Claude 응답 생성기 초기화 중...")
        if api_key:
            try:
                self.anthropic = Anthropic(api_key=api_key)
                self.use_api = True
                print("Claude API 초기화 완료")
            except Exception as e:
                print(f"Claude API 초기화 실패: {e}")
                self.use_api = False
        else:
            print("API 키가 제공되지 않아 모의 응답을 생성합니다.")
            self.use_api = False
    
    def generate_response(self, query, search_results, test_analysis, possible_diagnoses):
        """
        사용자 쿼리, 검색 결과, 검사 분석 및 가능한 진단명을 기반으로 Claude를 통해 응답을 생성합니다.
        """
        print("응답 생성 중...")
        
        # 프롬프트 템플릿 생성
        prompt = f"""
당신은 혈액검사 데이터를 기반으로 진단을 도와주는 의료 어시스턴트입니다. 아래 정보를 기반으로 환자의 혈액검사 결과를 분석하고 가능한 진단과 권장 사항을 제시해 주세요.

# 사용자 쿼리
{query}

# 혈액검사 결과 분석
```
{json.dumps(test_analysis, ensure_ascii=False, indent=2)}
```

# 검색 시스템이 찾은 유사 사례의 상위 진단명 (빈도순)
```
{json.dumps(possible_diagnoses[:5] if possible_diagnoses else [], ensure_ascii=False, indent=2)}
```

환자의 혈액검사 결과를 자세히 분석하여 다음 사항을 포함해 주세요:
1. 비정상적인 수치와 그 의미
2. 가능성 높은 진단명과 그 근거
3. 추가 검사가 필요한 항목
4. 환자에게 권장할 수 있는 치료 방법이나 생활 습관 개선 사항

전문적이고 정확한 분석을 제공하되, 환자가 이해하기 쉬운 언어로 설명해 주세요. 또한 당신은 의사가 아니므로, 최종적인 진단은 의료 전문가의 판단이 필요함을 강조해 주세요.
"""
        
        # API 키가 있는 경우 Claude API 호출
        if self.use_api:
            try:
                response = self.anthropic.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=1000,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text
            except Exception as e:
                print(f"Claude API 호출 중 오류 발생: {e}")
                return self._generate_mock_response(test_analysis, possible_diagnoses)
        else:
            # API 키가 없는 경우 모의 응답 생성
            return self._generate_mock_response(test_analysis, possible_diagnoses)
    
    def _generate_mock_response(self, test_analysis, possible_diagnoses):
        """
        API 키가 없거나 API 호출 실패 시 사용할 모의 응답을 생성합니다.
        """
        abnormal_items = []
        
        # 비정상 수치 찾기
        for item, analysis in test_analysis.items():
            if analysis["status"] != "정상":
                abnormal_items.append(f"{item} ({analysis['value']}): {analysis['status']}, {analysis['deviation']}")
        
        # 가능한 진단명
        diagnoses = []
        if possible_diagnoses:
            for i, (diagnosis, info) in enumerate(possible_diagnoses[:3]):
                diagnoses.append(f"{diagnosis} (유사도: {info['avg_similarity']:.2f})")
        
        # 모의 응답 생성
        response = """
# 혈액검사 결과 분석

## 비정상적인 수치와 의미
"""
        
        if abnormal_items:
            for item in abnormal_items:
                response += f"- {item}\n"
        else:
            response += "- 모든 검사 항목이 정상 범위 내에 있습니다.\n"
            
        response += """
## 가능성 높은 진단명
"""
        
        if diagnoses:
            for diagnosis in diagnoses:
                response += f"- {diagnosis}\n"
        else:
            response += "- 검색 결과에서 뚜렷한 진단명을 찾을 수 없습니다.\n"
            
        response += """
## 추가 검사 권장 항목
- 정확한 진단을 위해 추가 검사가 필요할 수 있습니다.
- 의사와 상담하여 적절한 추가 검사를 결정하세요.

## 생활 습관 개선 권장 사항
- 균형 잡힌 식단 유지
- 규칙적인 운동
- 충분한 수면
- 스트레스 관리

**주의: 이 분석은 참고용으로만 활용하시고, 정확한 진단과 치료를 위해 반드시 의료 전문가와 상담하셔야 합니다.**
"""
        
        return response

class BloodTestRAGPipeline:
    def __init__(self, search_engine, response_generator):
        """
        RAG 파이프라인을 초기화합니다.
        """
        print("RAG 파이프라인 초기화 중...")
        self.search_engine = search_engine
        self.response_generator = response_generator
    
    def process_query(self, query):
        """
        사용자 쿼리를 처리하여 응답을 생성합니다.
        """
        print(f"쿼리 처리 시작: {query}")
        
        # 1. 쿼리에서 혈액검사 결과 추출
        # 혈액검사 결과가 포함된 부분만 추출
        test_results_part = query
        if "혈액검사 결과:" in query:
            test_results_part = query.split("혈액검사 결과:")[1].strip()
        
        test_results = self.search_engine.parse_query_results(test_results_part)
        print(f"추출된 검사 결과: {test_results}")
        
        # 2. 검사 결과 분석
        test_analysis = self.search_engine.analyze_test_results(test_results)
        
        # 3. 벡터 검색 수행
        search_results = self.search_engine.search(query)
        
        # 4. 가능한 진단명 추출
        possible_diagnoses = self.search_engine.get_possible_diagnoses(search_results, test_analysis)
        
        # 5. 응답 생성
        response = self.response_generator.generate_response(
            query, search_results, test_analysis, possible_diagnoses
        )
        
        # 6. 결과 반환
        result = {
            "query": query,
            "test_results": test_results,
            "test_analysis": test_analysis,
            "possible_diagnoses": possible_diagnoses,
            "response": response
        }
        
        return result

# Google Drive에서 파일 다운로드 함수 정의
def download_file_from_google_drive(file_id, destination):
    import requests
    
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

# RAG 파이프라인 초기화 (전역 변수)
@st.cache_resource
def initialize_rag_pipeline():
    import tempfile
    import os
    import requests
    
    # API 키를 Streamlit secrets에서 가져옴
    try:
        claude_api_key = st.secrets["anthropic"]["api_key"]
    except Exception as e:
        st.warning("API 키를 찾을 수 없습니다. 모의 응답을 생성합니다.")
        claude_api_key = None
    
    # 클라우드 스토리지에서 파일 다운로드
    try:
        with st.spinner("벡터 DB 다운로드 중..."):
            # Google Drive 파일 ID
            vector_db_file_id = "1K0_7pDzfawEnllbtXFeZuOa5JgPaYv3h"
            
            # 임시 파일 경로
            vector_db_path = os.path.join(tempfile.gettempdir(), "vector_db.pkl")

            # GitHub 저장소의 데이터 파일 경로
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(current_dir, "final_data.csv")

            # 파일 다운로드 함수를 직접 구현
            def download_from_drive(file_id, destination):
                URL = "https://docs.google.com/uc?export=download"
                session = requests.Session()

                response = session.get(URL, params={'id': file_id}, stream=True)
                token = None
                for key, value in response.cookies.items():
                    if key.startswith('download_warning'):
                        token = value
                        break

                if token:
                    params = {'id': file_id, 'confirm': token}
                    response = session.get(URL, params=params, stream=True)

                CHUNK_SIZE = 32768
                with open(destination, "wb") as f:
                    for chunk in response.iter_content(CHUNK_SIZE):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
            
            # 파일 다운로드
            download_file_from_google_drive(vector_db_file_id, vector_db_path)
            
            st.success("파일 다운로드 완료")
            
            # 벡터 DB 로드
            with open(vector_db_path, 'rb') as f:
                vector_db = pickle.load(f)
        
        # RAG 검색 엔진 초기화
        search_engine = BloodTestRAGSearchEngine(vector_db, data_path)
        
        # Claude 응답 생성기 초기화
        response_generator = ClaudeResponseGenerator(claude_api_key)
        
        # RAG 파이프라인 초기화
        rag_pipeline = BloodTestRAGPipeline(search_engine, response_generator)
        
        return rag_pipeline
        
    except Exception as e:
        import traceback
        st.error(f"파일 다운로드 중 오류 발생: {e}")
        traceback.print_exc()
        
        # 오류 발생 시 기본 모드로 실행
        st.warning("기본 모드로 실행합니다. 일부 기능이 제한될 수 있습니다.")
        search_engine = BloodTestRAGSearchEngine(None, None)
        response_generator = ClaudeResponseGenerator(claude_api_key)
        return BloodTestRAGPipeline(search_engine, response_generator)

# Streamlit 웹 인터페이스
def main():
    st.set_page_config(
        page_title="혈액검사 분석 시스템",
        page_icon="🩸",
        layout="wide"
    )
    
    st.title("혈액검사 분석 시스템")
    
    # RAG 파이프라인 초기화
    rag_pipeline = initialize_rag_pipeline()
    
    # 정상범위 데이터 로드
    normal_ranges = get_normal_ranges()
    test_items = sorted(normal_ranges.keys())
    
    # 세션 상태 초기화
    if 'test_results' not in st.session_state:
        st.session_state.test_results = {}
    
    # 레이아웃 (2개 컬럼)
    col1, col2 = st.columns(2)
    
    # 입력 폼 (왼쪽 컬럼)
    with col1:
        st.header("혈액검사 결과 입력")
        
        with st.form("test_form"):
            selected_item = st.selectbox("검사 항목 선택", [""] + test_items)
            value = st.number_input("검사 결과 값", step=0.01)
            
            submitted = st.form_submit_button("항목 추가")
            if submitted and selected_item and value:
                st.session_state.test_results[selected_item] = float(value)
                st.success(f"{selected_item}: {value} 항목이 추가되었습니다.")
        
        # 추가된 항목 표시
        if st.session_state.test_results:
            st.subheader("추가된 검사 항목")
            
            for item, value in st.session_state.test_results.items():
                col_a, col_b = st.columns([5, 1])
                with col_a:
                    st.write(f"**{item}**: {value}")
                with col_b:
                    if st.button("삭제", key=f"del_{item}"):
                        del st.session_state.test_results[item]
                        st.experimental_rerun()
        
        # 분석 및 초기화 버튼
        col_analyze, col_reset = st.columns(2)
        
        with col_analyze:
            analyze_btn = st.button("분석하기", disabled=len(st.session_state.test_results) == 0)
        
        with col_reset:
            reset_btn = st.button("초기화")
            if reset_btn:
                st.session_state.test_results = {}
                st.experimental_rerun()
    
    # 결과 표시 (오른쪽 컬럼)
    with col2:
        st.header("분석 결과")
        
        if analyze_btn and st.session_state.test_results:
            with st.spinner("분석 중..."):
                # 쿼리 생성
                query_parts = ["혈액검사 결과:"]
                for item, val in st.session_state.test_results.items():
                    query_parts.append(f"{item}: {val}")
                
                query = ", ".join(query_parts)
                
                # RAG 파이프라인 실행
                if rag_pipeline:
                    result = rag_pipeline.process_query(query)
                    
                    # 비정상 항목 표시
                    st.subheader("비정상 수치 항목")
                    
                    analysis = result["test_analysis"]
                    abnormal_items = [(item, data) for item, data in analysis.items() if data["status"] != "정상"]
                    
                    if abnormal_items:
                        for item, data in abnormal_items:
                            if data["status"] == "높음":
                                color = "danger"
                            elif data["status"] == "낮음":
                                color = "primary"
                            else:
                                color = "secondary"
                            
                            st.markdown(f"""
                            <div style='padding: 10px; border-radius: 5px; background-color: {'#f8d7da' if color == 'danger' else '#cfe2ff' if color == 'primary' else '#e2e3e5'}'>
                                <b>{item}:</b> {data['value']} ({data['status']})<br>
                                <small>정상 범위: {data['normal_range']}</small><br>
                                <small>{data['deviation']}</small>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.success("모든 검사 항목이 정상 범위 내에 있습니다.")
                    
                    # 진단명 표시
                    st.subheader("가능성 높은 진단명")
                    
                    if result["possible_diagnoses"]:
                        for diagnosis, info in result["possible_diagnoses"][:5]:
                            st.write(f"**{diagnosis}** (코드: {info['code']})")
                            st.write(f"유사도: {info['avg_similarity']:.2f}")
                            st.write("---")
                    else:
                        st.info("유사한 진단명을 찾을 수 없습니다.")
                    
                    # 상세 분석 및 권장 사항
                    st.subheader("상세 분석 및 권장 사항")
                    st.markdown(result["response"])
                else:
                    st.error("RAG 파이프라인이 초기화되지 않았습니다.")

if __name__ == "__main__":
    main()
