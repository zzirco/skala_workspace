"""
반도체 공정 데이터 분석 RAG 에이전트 (PDF 입출력)

필수 패키지:
pip install langchain openai chromadb pypdf pandas rank_bm25 python-dotenv
pip install pdfplumber matplotlib seaborn reportlab pillow openpyxl

.env 파일 설정:
OPENAI_API_KEY=your-api-key-here

주요 기능:
- 다양한 입력 형식 지원 (CSV, Excel, PDF)
- Tool별 상한/하한 Threshold 검사
- EDA 시각화 (5종 그래프)
- RAG 기반 LLM 분석 (선택사항)
- 전문적인 PDF 보고서 생성
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import urllib.request
from datetime import datetime
from dotenv import load_dotenv
from io import BytesIO
from PIL import Image
from matplotlib.dates import DateFormatter

from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# Langchain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 한글 폰트 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# === [DEBUG] 환경변수 로드 관련 디버그 ===
print("=== [DEBUG] dotenv check start ===")

# .env 파일 위치 추적용
possible_env_paths = [
    os.path.join(os.getcwd(), ".env"),
    os.path.join(os.path.dirname(__file__), ".env"),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
]

for p in possible_env_paths:
    print(f"Checking: {p} -> exists? {os.path.exists(p)}")

# 혹시 모를 덮어쓰기 문제 방지 위해 강제로 override
load_dotenv(override=True)

api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    print(f"✅ [DEBUG] OPENAI_API_KEY loaded successfully: {api_key[:10]}********")
else:
    print("❌ [DEBUG] OPENAI_API_KEY is missing or empty!")

# === [DEBUG] 환경변수 로드 완료 ===


class SemiconductorRAGAgent:
    def __init__(self, knowledge_pdf_paths: list = None, chroma_persist_dir: str = "./chroma_db", 
                 force_reembed: bool = False):
        """
        반도체 공정 데이터 분석 RAG 에이전트 초기화
        
        Args:
            knowledge_pdf_paths: 반도체 지식 문서 PDF 경로 리스트 (None이면 RAG 비활성화)
            chroma_persist_dir: ChromaDB 저장 경로
            force_reembed: True면 기존 임베딩 무시하고 재생성, False면 기존 임베딩 재사용
        """
        api_key = os.getenv("OPENAI_API_KEY")
        print(api_key)
        self.chroma_persist_dir = chroma_persist_dir
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=api_key
        )
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0,
            openai_api_key=api_key
        )
        
        # 한글 폰트 설정
        self.setup_korean_font()
        
        # 지식 베이스 PDF 로드 및 임베딩
        self.rag_enabled = False
        if knowledge_pdf_paths and len(knowledge_pdf_paths) > 0:
            try:
                # ChromaDB가 이미 존재하는지 확인
                db_exists = os.path.exists(chroma_persist_dir) and \
                           os.path.exists(os.path.join(chroma_persist_dir, "chroma.sqlite3"))
                
                if db_exists and not force_reembed:
                    # 기존 벡터스토어 로드
                    print("ℹ️ 기존 ChromaDB 임베딩을 로드합니다...")
                    self.vectorstore = Chroma(
                        persist_directory=chroma_persist_dir,
                        embedding_function=self.embeddings
                    )
                    
                    # 기존 문서 개수 확인
                    collection = self.vectorstore._collection
                    doc_count = collection.count()
                    print(f"✓ 기존 임베딩 로드 완료: {doc_count}개 문서 청크")
                    
                else:
                    # 새로 임베딩 생성
                    if force_reembed:
                        print("ℹ️ 강제 재임베딩 모드: 새로운 임베딩을 생성합니다...")
                        # 기존 디렉토리 삭제
                        if os.path.exists(chroma_persist_dir):
                            import shutil
                            shutil.rmtree(chroma_persist_dir)
                    else:
                        print("ℹ️ ChromaDB가 없습니다. 새로운 임베딩을 생성합니다...")
                    
                    self.vectorstore = self._load_and_embed_pdfs(knowledge_pdf_paths)
                
                self.documents = self._get_documents_from_vectorstore()
                self.rag_enabled = True
                print(f"✓ RAG 지식베이스 준비 완료")
                
            except Exception as e:
                print(f"⚠️ RAG 초기화 실패: {e}")
                print("  RAG 없이 기본 분석만 수행합니다.")
        else:
            print("ℹ️ RAG 비활성화: 지식베이스 없이 분석 수행")
        
        # Tool별 Threshold 설정 (기존과 동일)
        self.threshold_by_tool = {
            "Deposition": {
                "Chamber_Temperature": {"lower": 59.9696, "upper": 90.0976},
                "Gas_Flow_Rate": {"lower": 19.3436, "upper": 80.8356},
                "RF_Power": {"lower": 156.8579, "upper": 447.4576},
                "Etch_Depth": {"lower": 196.5077, "upper": 807.2571},
                "Vacuum_Pressure": {"lower": 0.3476, "upper": 0.6486},
                "Stage_Alignment_Error": {"lower": -0.4365, "upper": 4.4373},
                "Vibration_Level": {"lower": -0.0045, "upper": 0.0247},
                "Particle_Count": {"lower": -234.1687, "upper": 1351.6870}
            },
            "Etching": {
                "Chamber_Temperature": {"lower": 60.3287, "upper": 90.2200},
                "Gas_Flow_Rate": {"lower": 20.4293, "upper": 79.6996},
                "RF_Power": {"lower": 149.0493, "upper": 453.7188},
                "Etch_Depth": {"lower": 196.4698, "upper": 798.6325},
                "Vacuum_Pressure": {"lower": 0.3543, "upper": 0.6531},
                "Stage_Alignment_Error": {"lower": -0.4266, "upper": 4.4792},
                "Vibration_Level": {"lower": -0.0050, "upper": 0.0248},
                "Particle_Count": {"lower": -233.2085, "upper": 1337.3679}
            },
            "Lithography": {
                "Chamber_Temperature": {"lower": 59.9227, "upper": 89.9170},
                "Gas_Flow_Rate": {"lower": 18.9624, "upper": 80.3334},
                "RF_Power": {"lower": 152.9034, "upper": 448.1597},
                "Etch_Depth": {"lower": 189.9225, "upper": 803.1362},
                "Vacuum_Pressure": {"lower": 0.3510, "upper": 0.6493},
                "Stage_Alignment_Error": {"lower": -0.4897, "upper": 4.4519},
                "Vibration_Level": {"lower": -0.0047, "upper": 0.0253},
                "Particle_Count": {"lower": -223.0565, "upper": 1336.5698}
            }
        }
    
    def setup_korean_font(self):
        """한글 폰트 설정"""
        font_dir = "./fonts"
        os.makedirs(font_dir, exist_ok=True)
        
        # 나눔고딕 폰트 경로
        font_paths = [
            "C:\\Windows\\Fonts\\H2MJSM.ttf"
        ]
        
        font_registered = False
        
        # 기존 폰트 찾기
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    pdfmetrics.registerFont(TTFont('Korean', font_path))
                    self.korean_font = 'Korean'
                    font_registered = True
                    print(f"✓ 한글 폰트 등록 완료: {font_path}")
                    break
                except Exception as e:
                    continue

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        다양한 형식의 데이터 로드 (자동 감지)
        
        Args:
            file_path: 데이터 파일 경로 (.pdf, .csv, .xlsx, .xls)
        
        Returns:
            pd.DataFrame: 센서 데이터
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.csv':
            return self.load_sensor_data_from_csv(file_path)
        elif file_ext in ['.xlsx', '.xls']:
            return self.load_sensor_data_from_excel(file_path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {file_ext}")
    
    def load_sensor_data_from_csv(self, csv_path: str) -> pd.DataFrame:
        """
        CSV에서 센서 데이터 로드
        
        Args:
            csv_path: CSV 파일 경로
        
        Returns:
            pd.DataFrame: 센서 데이터
        """
        try:
            df = pd.read_csv(csv_path)
            print(f"✓ CSV에서 데이터 로드 완료: {len(df)}행 x {len(df.columns)}열")
            print(f"  컬럼: {df.columns.tolist()}")
            return df
        except Exception as e:
            print(f"❌ CSV 로드 실패: {e}")
            return pd.DataFrame()
    
    def load_sensor_data_from_excel(self, excel_path: str, sheet_name: int = 0) -> pd.DataFrame:
        """
        Excel에서 센서 데이터 로드
        
        Args:
            excel_path: Excel 파일 경로
            sheet_name: 시트 인덱스 또는 이름
        
        Returns:
            pd.DataFrame: 센서 데이터
        """
        try:
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            print(f"✓ Excel에서 데이터 로드 완료: {len(df)}행 x {len(df.columns)}열")
            print(f"  컬럼: {df.columns.tolist()}")
            return df
        except Exception as e:
            print(f"❌ Excel 로드 실패: {e}")
            return pd.DataFrame()
    
    
    def _load_and_embed_pdfs(self, pdf_paths: list):
        """지식 베이스 PDF 로드 및 ChromaDB 임베딩 (새로 생성)"""
        print(f"  PDF 로드 중: {len(pdf_paths)}개 파일")
        all_docs = []
        
        for pdf_path in pdf_paths:
            try:
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()
                all_docs.extend(documents)
                print(f"    ✓ {os.path.basename(pdf_path)}: {len(documents)}페이지")
            except Exception as e:
                print(f"    ✗ {os.path.basename(pdf_path)} 로드 실패: {e}")
        
        print(f"  텍스트 분할 중...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(all_docs)
        print(f"  총 {len(splits)}개 청크 생성")
        
        print(f"  임베딩 생성 및 ChromaDB 저장 중...")
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.chroma_persist_dir
        )
        vectorstore.persist()
        print(f"  ✓ 임베딩 저장 완료: {self.chroma_persist_dir}")
        
        return vectorstore
    
    def _get_documents_from_vectorstore(self):
        """벡터스토어에서 문서 추출"""
        return self.vectorstore.get()['documents']
    
    def check_embedding_status(self):
        """현재 임베딩 상태 확인"""
        if not self.rag_enabled:
            print("❌ RAG가 비활성화되어 있습니다.")
            return
        
        try:
            collection = self.vectorstore._collection
            doc_count = collection.count()
            print(f"\n=== ChromaDB 임베딩 상태 ===")
            print(f"저장 위치: {self.chroma_persist_dir}")
            print(f"문서 청크 수: {doc_count}개")
            print(f"임베딩 모델: text-embedding-ada-002")
            print("=" * 40)
        except Exception as e:
            print(f"❌ 임베딩 상태 확인 실패: {e}")
    
    def _create_hybrid_retriever(self):
        """BM25 + Semantic Search 하이브리드 리트리버"""
        bm25_retriever = BM25Retriever.from_texts(self.documents)
        bm25_retriever.k = 3
        
        semantic_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, semantic_retriever],
            weights=[0.5, 0.5]
        )
        
        return ensemble_retriever
    
    def add_new_documents(self, new_pdf_paths: list):
        """기존 임베딩에 새 문서 추가"""
        if not self.rag_enabled:
            print("❌ RAG가 비활성화되어 있습니다. 초기화 시 knowledge_pdf_paths를 지정하세요.")
            return
        
        try:
            print(f"\n새 문서 추가 중: {len(new_pdf_paths)}개")
            all_docs = []
            
            for pdf_path in new_pdf_paths:
                try:
                    loader = PyPDFLoader(pdf_path)
                    documents = loader.load()
                    all_docs.extend(documents)
                    print(f"  ✓ {os.path.basename(pdf_path)}: {len(documents)}페이지")
                except Exception as e:
                    print(f"  ✗ {os.path.basename(pdf_path)} 로드 실패: {e}")
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(all_docs)
            
            # 기존 vectorstore에 추가
            self.vectorstore.add_documents(splits)
            self.vectorstore.persist()
            
            # documents 리스트 업데이트
            self.documents = self._get_documents_from_vectorstore()
            
            print(f"✓ {len(splits)}개 청크 추가 완료")
            self.check_embedding_status()
            
        except Exception as e:
            print(f"❌ 문서 추가 실패: {e}")
    
    def create_eda_visualizations(self, df: pd.DataFrame, output_dir: str = "./eda_plots") -> list:
        """
        EDA 시각화 생성 (한글 지원)
        
        Args:
            df: 센서 데이터
            output_dir: 그래프 저장 디렉토리
        
        Returns:
            생성된 이미지 파일 경로 리스트
        """
        os.makedirs(output_dir, exist_ok=True)
        image_paths = []
        # Timestamp를 datetime으로 변환
        # 1. 공정 파라미터 분포 (히스토그램)
        # Figure 생성 (4개 서브플롯)
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        fig.suptitle('반도체 제조공정 시계열 분석', fontsize=16, fontweight='bold')


        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        # 시간순 정렬
        df = df.sort_values('Timestamp')

        # Threshold 정의
        thresholds = {
            "Deposition": {
                "Chamber_Temperature": {"lower": 59.9696, "upper": 90.0976},
                "RF_Power": {"lower": 156.8579, "upper": 447.4576}
            },
            "Etching": {
                "Chamber_Temperature": {"lower": 60.3287, "upper": 90.2200},
                "RF_Power": {"lower": 149.0493, "upper": 453.7188}
            },
            "Lithography": {
                "Chamber_Temperature": {"lower": 59.9227, "upper": 89.9170},
                "RF_Power": {"lower": 152.9034, "upper": 448.1597}
            }
        }

        # Tool_Type 목록
        tool_types = df['Tool_Type'].unique()

        # Figure 생성 (Tool_Type 개수 × 4개 지표)
        fig, axes = plt.subplots(len(tool_types), 4, figsize=(20, 4*len(tool_types)))
        fig.suptitle('Tool Type별 반도체 제조공정 분석', fontsize=18, fontweight='bold')

        # 각 Tool_Type별로 행 생성
        for i, tool in enumerate(tool_types):
            data = df[df['Tool_Type'] == tool]
            
            # 1. 온도
            axes[i, 0].plot(data['Timestamp'], data['Chamber_Temperature'], 
                            color='red', alpha=0.7, linewidth=1.2)
            # Threshold 선 추가
            axes[i, 0].axhline(y=thresholds[tool]['Chamber_Temperature']['upper'], 
                            color='darkred', linestyle='--', linewidth=2, label='Upper Limit')
            axes[i, 0].axhline(y=thresholds[tool]['Chamber_Temperature']['lower'], 
                            color='darkred', linestyle='--', linewidth=2, label='Lower Limit')
            axes[i, 0].set_ylabel('챔버 온도 (°C)', fontsize=10)
            axes[i, 0].set_title(f'{tool} - 챔버 온도', fontsize=11, fontweight='bold')
            axes[i, 0].legend(loc='best', fontsize=8)
            axes[i, 0].grid(True, alpha=0.3)
            
            # 2. RF 파워
            axes[i, 1].plot(data['Timestamp'], data['RF_Power'], 
                            color='green', alpha=0.7, linewidth=1.2)
            # Threshold 선 추가
            axes[i, 1].axhline(y=thresholds[tool]['RF_Power']['upper'], 
                            color='darkgreen', linestyle='--', linewidth=2, label='Upper Limit')
            axes[i, 1].axhline(y=thresholds[tool]['RF_Power']['lower'], 
                            color='darkgreen', linestyle='--', linewidth=2, label='Lower Limit')
            axes[i, 1].set_ylabel('RF 파워', fontsize=10)
            axes[i, 1].set_title(f'{tool} - RF 파워', fontsize=11, fontweight='bold')
            axes[i, 1].legend(loc='best', fontsize=8)
            axes[i, 1].grid(True, alpha=0.3)
            
            # 3. 입자 수
            axes[i, 2].scatter(data['Timestamp'], data['Particle_Count'], 
                            color='blue', alpha=0.5, s=15)
            axes[i, 2].set_ylabel('입자 수', fontsize=10)
            axes[i, 2].set_title(f'{tool} - 입자 수', fontsize=11, fontweight='bold')
            axes[i, 2].grid(True, alpha=0.3)
            
            # 4. 결함률
            hourly = data.set_index('Timestamp').resample('1H').agg({
                'Defect': ['sum', 'count']
            })
            hourly.columns = ['defect_count', 'total_count']
            hourly['defect_rate'] = (hourly['defect_count'] / hourly['total_count']) * 100
            axes[i, 3].plot(hourly.index, hourly['defect_rate'], 
                            color='darkred', linewidth=1.5)
            axes[i, 3].fill_between(hourly.index, hourly['defect_rate'], 
                                    alpha=0.3, color='red')
            axes[i, 3].set_ylabel('결함률 (%)', fontsize=10)
            axes[i, 3].set_title(f'{tool} - 결함률', fontsize=11, fontweight='bold')
            axes[i, 3].grid(True, alpha=0.3)
            
            # 날짜 형식 설정
            for ax in axes[i]:
                ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H:%M'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

        # 마지막 행에만 x축 레이블 표시
        for j in range(4):
            axes[-1, j].set_xlabel('시간', fontsize=10)

        plt.tight_layout()
        path1 = os.path.join(output_dir, "01_parameter_distribution.png")
        plt.savefig(path1, dpi=200, bbox_inches='tight')
        plt.close()
        image_paths.append(path1)
        
        # 2. Tool Type별 박스플롯
        if 'Tool_Type' in df.columns:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('장비 타입별 파라미터 비교 (Parameters by Tool Type)', 
                        fontsize=16, fontweight='bold')
            
            key_params = ['Chamber_Temperature', 'RF_Power', 'Particle_Count', 'Etch_Depth']
            
            for idx, param in enumerate(key_params):
                if param in df.columns:
                    ax = axes[idx // 2, idx % 2]
                    df.boxplot(column=param, by='Tool_Type', ax=ax)
                    ax.set_title(f'{param} 분포 by 장비 타입', fontsize=12)
                    ax.set_xlabel('장비 타입 (Tool Type)')
                    ax.set_ylabel(param)
                    plt.sca(ax)
                    plt.xticks(rotation=45)
            
            plt.tight_layout()
            path2 = os.path.join(output_dir, "02_tool_type_comparison.png")
            plt.savefig(path2, dpi=200, bbox_inches='tight')
            plt.close()
            image_paths.append(path2)
        
        # 4. Defect 분석
        if 'Defect' in df.columns and 'Tool_Type' in df.columns:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle('결함 분석 (Defect Analysis)', fontsize=16, fontweight='bold')
            
            # Defect 비율
            defect_counts = df['Defect'].value_counts()
            labels = ['정상 (No Defect)', '결함 (Defect)']
            axes[0].pie(defect_counts, labels=labels, autopct='%1.1f%%',
                       colors=['lightgreen', 'lightcoral'], startangle=90)
            axes[0].set_title('전체 결함 발생률')
            
            # Tool Type별 Defect 비율
            tool_defect = df.groupby('Tool_Type')['Defect'].mean() * 100
            colors = ['skyblue', 'gold', 'violet']
            tool_defect.plot(kind='bar', ax=axes[1], color=colors)
            axes[1].set_title('장비 타입별 결함률')
            axes[1].set_ylabel('결함률 (%)')
            axes[1].set_xlabel('장비 타입')
            axes[1].grid(True, alpha=0.3)
            plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=0)
            
            plt.tight_layout()
            path3 = os.path.join(output_dir, "04_defect_analysis.png")
            plt.savefig(path3, dpi=200, bbox_inches='tight')
            plt.close()
            image_paths.append(path3)
        
        print(f"✓ {len(image_paths)}개 EDA 그래프 생성 완료")
        return image_paths
    
    def _check_threshold_violations_visual(self, df: pd.DataFrame) -> dict:
        """시각화를 위한 Threshold 위반 카운트"""
        if 'Tool_Type' not in df.columns:
            return {}
        
        violation_counts = {}
        
        for tool_type in df['Tool_Type'].unique():
            tool_df = df[df['Tool_Type'] == tool_type]
            
            if tool_type not in self.threshold_by_tool:
                continue
            
            thresholds = self.threshold_by_tool[tool_type]
            count = 0
            
            for param, limits in thresholds.items():
                if param in tool_df.columns:
                    lower = limits['lower']
                    upper = limits['upper']
                    violations = ((tool_df[param] < lower) | (tool_df[param] > upper)).sum()
                    count += violations
            
            violation_counts[tool_type] = count
        
        return violation_counts
    
    def _generate_stats_summary(self, df: pd.DataFrame) -> str:
        """통계 요약 생성"""
        summary = "=== 반도체 공정 센서 데이터 분석 ===\n"
        summary += f"\n총 데이터 건수: {len(df)}\n"
        
        if 'Wafer_ID' in df.columns:
            summary += f"분석 웨이퍼 수: {df['Wafer_ID'].nunique()}\n"
        if 'Tool_Type' in df.columns:
            summary += f"장비 타입: {df['Tool_Type'].unique().tolist()}\n"
        
        # 공정 파라미터 통계
        process_params = ['Chamber_Temperature', 'Gas_Flow_Rate', 'RF_Power', 
                         'Vacuum_Pressure', 'Etch_Depth']
        summary += "\n### 공정 파라미터 통계 ###\n"
        for col in process_params:
            if col in df.columns:
                summary += f"\n[{col}]\n"
                summary += f"  평균: {df[col].mean():.2f} | 표준편차: {df[col].std():.2f}\n"
                summary += f"  범위: {df[col].min():.2f} ~ {df[col].max():.2f}\n"
        
        # 품질 지표
        quality_params = ['Stage_Alignment_Error', 'Vibration_Level', 'Particle_Count']
        summary += "\n### 품질 지표 통계 ###\n"
        for col in quality_params:
            if col in df.columns:
                summary += f"\n[{col}]\n"
                summary += f"  평균: {df[col].mean():.2f} | 표준편차: {df[col].std():.2f}\n"
        
        # 결함 분석
        if 'Defect' in df.columns:
            defect_rate = (df['Defect'].sum() / len(df) * 100)
            summary += f"\n### 결함 분석 ###\n"
            summary += f"결함 발생률: {defect_rate:.2f}%\n"
        
        return summary
    
    def _check_threshold_violations(self, df: pd.DataFrame) -> str:
        """Tool별 상한/하한 Threshold 검사"""
        violations = "\n=== Tool Type별 Threshold 위반 분석 ===\n"
        
        if 'Tool_Type' not in df.columns:
            return violations + "Tool_Type 컬럼이 없습니다.\n"
        
        total_violations = 0
        
        for tool_type in df['Tool_Type'].unique():
            tool_df = df[df['Tool_Type'] == tool_type]
            
            if tool_type not in self.threshold_by_tool:
                violations += f"\n⚠️ [{tool_type}] Threshold 기준 미설정\n"
                continue
            
            violations += f"\n### {tool_type} ###\n"
            thresholds = self.threshold_by_tool[tool_type]
            tool_violation_count = 0
            
            for param, limits in thresholds.items():
                if param not in tool_df.columns:
                    continue
                
                lower = limits['lower']
                upper = limits['upper']
                
                below_lower = tool_df[tool_df[param] < lower]
                above_upper = tool_df[tool_df[param] > upper]
                
                total_param_violations = len(below_lower) + len(above_upper)
                
                if total_param_violations > 0:
                    violations += f"\n  ⚠️ [{param}]\n"
                    violations += f"    허용범위: {lower:.2f} ~ {upper:.2f}\n"
                    
                    if len(below_lower) > 0:
                        violations += f"    하한 위반: {len(below_lower)}건 (최소값: {below_lower[param].min():.2f})\n"
                    
                    if len(above_upper) > 0:
                        violations += f"    상한 위반: {len(above_upper)}건 (최대값: {above_upper[param].max():.2f})\n"
                    
                    tool_violation_count += total_param_violations
            
            if tool_violation_count == 0:
                violations += "  ✓ 모든 파라미터 정상 범위 내\n"
            else:
                violations += f"\n  총 위반 건수: {tool_violation_count}건\n"
            
            total_violations += tool_violation_count
        
        violations += f"\n### 전체 요약 ###\n"
        violations += f"총 Threshold 위반: {total_violations}건\n"
        
        return violations
    
    def _analyze_correlations(self, df: pd.DataFrame) -> str:
        """상관관계 분석"""
        corr_summary = "\n=== 파라미터 상관관계 분석 ===\n"
        
        if 'Defect' in df.columns:
            corr_summary += "\n결함과 높은 상관관계를 보이는 파라미터:\n"
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != 'Defect':
                    corr = df[col].corr(df['Defect'])
                    if abs(corr) > 0.3:
                        corr_summary += f"  {col}: {corr:.3f}\n"
        
        return corr_summary
    
    def generate_llm_report(self, df: pd.DataFrame) -> str:
        """LLM 기반 상세 분석 보고서 생성"""
        stats = self._generate_stats_summary(df)
        violations = self._check_threshold_violations(df)
        correlations = self._analyze_correlations(df)
        
        prompt = f"""
다음은 반도체 제조 공정에서 수집된 센서 데이터입니다. 
상세한 분석 보고서를 작성해주세요.

{stats}

{violations}

{correlations}

# 분석 요구사항(각 문항별로 한줄요약과 함께 내용 작성)

1. 자세한 데이터 분포에 대한 해석 
2. 공정 파라미터 안정성 평가
3. Tool Type별 성능 차이 분석
4. Threshold 위반 원인 및 개선방안
5. 결함 발생 패턴 분석
6. 종합 평가 및 권고사항

전문적인 엔지니어링 보고서 형식으로 작성하되, 명확하고 실행 가능한 개선안을 제시해주세요.
"""
        
        if self.rag_enabled:
            # RAG 활성화: 지식베이스 참조
            retriever = self._create_hybrid_retriever()
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": self._get_prompt_template()}
            )
            
            result = qa_chain({"query": prompt})
            return result['result']
        else:
            # RAG 비활성화: 직접 LLM 호출
            from langchain.schema import HumanMessage
            
            simple_prompt = f"""
당신은 반도체 제조 공정 전문가입니다.
다음 데이터를 분석하여 상세한 보고서를 작성해주세요.

{prompt}

명확하고 실용적인 분석을 제공하되, 기술적 근거를 명시하세요.
"""
            response = self.llm.predict(simple_prompt)
            return response
    
    def _get_prompt_template(self) -> PromptTemplate:
        """RAG 프롬프트 템플릿"""
        template = """
당신은 반도체 제조 공정 전문가입니다.
제공된 기술 문서를 참조하여 정확하고 실용적인 분석을 제공하세요.

## 참조 문서:
{context}

## 분석 요청:
{question}

## 작성 가이드:
- 기술적 근거 명확히 제시
- 정량적 데이터 기반 판단
- 실행 가능한 구체적 개선안
- 명확하고 간결한 문장

답변:
"""
        return PromptTemplate(template=template, input_variables=["context", "question"])
    def create_pdf_report(self, df: pd.DataFrame, image_paths: list, 
                         llm_report: str, output_path: str = "./semiconductor_report.pdf"):
        """
        개선된 PDF 보고서 생성 (한글 지원, 향상된 디자인)
        """
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.platypus import PageBreak, Table, TableStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
        
        doc = SimpleDocTemplate(output_path, pagesize=letter,
                               topMargin=0.5*inch, bottomMargin=0.75*inch,
                               leftMargin=0.6*inch, rightMargin=0.6*inch)
        
        story = []
        styles = getSampleStyleSheet()
        
        # 색상 팔레트
        primary_color = colors.HexColor('#1f4788')
        secondary_color = colors.HexColor('#2c5aa0')
        accent_color = colors.HexColor('#4a90e2')
        light_bg = colors.HexColor('#f0f4f8')
        
        # 스타일 정의
        korean_title = ParagraphStyle(
            'KoreanTitle',
            parent=styles['Heading1'],
            fontName=self.korean_font,
            fontSize=28,
            textColor=primary_color,
            spaceAfter=8,
            alignment=TA_CENTER,
            leading=34
        )
        
        korean_subtitle = ParagraphStyle(
            'KoreanSubtitle',
            parent=styles['Heading2'],
            fontName=self.korean_font,
            fontSize=18,
            textColor=secondary_color,
            spaceAfter=20,
            alignment=TA_CENTER,
            leading=22
        )
        
        korean_heading2 = ParagraphStyle(
            'KoreanHeading2',
            parent=styles['Heading2'],
            fontName=self.korean_font,
            fontSize=18,
            textColor=colors.white,
            spaceAfter=14,
            spaceBefore=20,
            leftIndent=12,
            leading=22
        )
        
        korean_heading3 = ParagraphStyle(
            'KoreanHeading3',
            parent=styles['Heading3'],
            fontName=self.korean_font,
            fontSize=14,
            textColor=secondary_color,
            spaceAfter=6,
            spaceBefore=8,
            leftIndent=6,
            leading=18
        )
        
        korean_normal = ParagraphStyle(
            'KoreanNormal',
            parent=styles['Normal'],
            fontName=self.korean_font,
            fontSize=10,
            leading=14,
            spaceAfter=3,
            leftIndent=10,
            rightIndent=10
        )
        
        info_box_style = ParagraphStyle(
            'InfoBox',
            parent=korean_normal,
            fontSize=11,
            textColor=colors.HexColor('#2c3e50'),
            leftIndent=15,
            rightIndent=15
        )
        
        # ===== 표지 =====
        story.append(Spacer(1, 0.8*inch))
        story.append(Paragraph("반도체 공정 데이터 분석 보고서", korean_title))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph("Semiconductor Process Analysis Report", korean_subtitle))
        
        # 구분선
        story.append(Spacer(1, 0.3*inch))
        line = Table([['']], colWidths=[6.5*inch])
        line.setStyle(TableStyle([
            ('LINEABOVE', (0,0), (-1,0), 2, accent_color),
            ('LINEBELOW', (0,0), (-1,0), 2, accent_color),
        ]))
        story.append(line)
        story.append(Spacer(1, 0.3*inch))
        
        # 분석 정보 박스
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        info_data = [
            ['항목', '내용'],
            ['분석 일시', date_str],
            ['총 데이터 건수', f'{len(df):,}'],
        ]
        
        if 'Tool_Type' in df.columns:
            tools = ', '.join(df['Tool_Type'].unique())
            info_data.append(['장비 타입', tools])
        
        if 'Defect' in df.columns:
            defect_rate = (df['Defect'].sum() / len(df) * 100)
            info_data.append(['결함률', f'{defect_rate:.2f}%'])
        
        info_table = Table(info_data, colWidths=[2*inch, 4.5*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), secondary_color),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (-1,0), self.korean_font),
            ('FONTSIZE', (0,0), (-1,0), 12),
            ('FONTNAME', (0,1), (-1,-1), self.korean_font),
            ('FONTSIZE', (0,1), (-1,-1), 10),
            ('BACKGROUND', (0,1), (-1,-1), light_bg),
            ('GRID', (0,0), (-1,-1), 1, colors.HexColor('#d0d0d0')),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [light_bg, colors.white]),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('LEFTPADDING', (0,0), (-1,-1), 12),
            ('RIGHTPADDING', (0,0), (-1,-1), 12),
            ('TOPPADDING', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(info_table)
        
        story.append(PageBreak())
        
        # ===== EDA 섹션 =====
        # 섹션 헤더
        section_header = Table([['  1. 탐색적 데이터 분석 (EDA)']], colWidths=[6.5*inch])
        section_header.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), primary_color),
            ('TEXTCOLOR', (0,0), (-1,-1), colors.white),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (-1,-1), self.korean_font),
            ('FONTSIZE', (0,0), (-1,-1), 18),
            ('TOPPADDING', (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 10),
            ('LEFTPADDING', (0,0), (-1,-1), 15),
        ]))
        story.append(section_header)
        story.append(Spacer(1, 0.25*inch))
        
        # 그래프 추가
        for idx, img_path in enumerate(image_paths, 1):
            try:
                img_name = os.path.basename(img_path).replace('.png', '').replace('_', ' ').title()
                
                # 그래프 타이틀 박스
                graph_title = Table([[f'그래프 {idx}: {img_name}']], colWidths=[6.5*inch])
                graph_title.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,-1), accent_color),
                    ('TEXTCOLOR', (0,0), (-1,-1), colors.white),
                    ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                    ('FONTNAME', (0,0), (-1,-1), self.korean_font),
                    ('FONTSIZE', (0,0), (-1,-1), 13),
                    ('TOPPADDING', (0,0), (-1,-1), 6),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                    ('LEFTPADDING', (0,0), (-1,-1), 12),
                ]))
                story.append(graph_title)
                story.append(Spacer(1, 0.1*inch))
                
                img = RLImage(img_path, width=6.5*inch, height=3.8*inch)
                story.append(img)
                story.append(Spacer(1, 0.3*inch))
                
            except Exception as e:
                print(f"이미지 추가 실패 ({img_path}): {e}")
        
        story.append(PageBreak())
        
        # ===== AI 분석 보고서 섹션 =====
        section_header2 = Table([['  2. AI 기반 상세 분석 보고서']], colWidths=[6.5*inch])
        section_header2.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), primary_color),
            ('TEXTCOLOR', (0,0), (-1,-1), colors.white),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (-1,-1), self.korean_font),
            ('FONTSIZE', (0,0), (-1,-1), 18),
            ('TOPPADDING', (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 15),
            ('LEFTPADDING', (0,0), (-1,-1), 15),
        ]))
        story.append(section_header2)
        story.append(Spacer(0.05, 0.05*inch))
        
        # 보고서 내용 파싱
        report_lines = llm_report.split('\n')
        for line in report_lines:
            line = line.strip()
            if not line:
                continue
            
            # 헤더 처리
            if line.startswith('###'):
                line_clean = line.replace('#', '').strip()
                # 서브섹션 박스
                subsection = Table([[f'  {line_clean}']], colWidths=[6.5*inch])
                subsection.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,-1), light_bg),
                    ('TEXTCOLOR', (0,0), (-1,-1), secondary_color),
                    ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                    ('FONTNAME', (0,0), (-1,-1), self.korean_font),
                    ('FONTSIZE', (0,0), (-1,-1), 13),
                    ('TOPPADDING', (0,0), (-1,-1), 8),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 8),
                    ('LEFTPADDING', (0,0), (-1,-1), 12),
                    ('LINEBELOW', (0,0), (-1,-1), 2, accent_color),
                ]))
                story.append(Spacer(1, 0.15*inch))
                story.append(subsection)
                story.append(Spacer(1, 0.08*inch))
                
            elif line.startswith('##'):
                line_clean = line.replace('#', '').strip()
                story.append(Paragraph(f"<b>{line_clean}</b>", korean_heading2))
                
            elif line.startswith('#'):
                line_clean = line.replace('#', '').strip()
                story.append(Paragraph(f"<b>{line_clean}</b>", korean_heading2))
                
            # 리스트 항목
            elif line.startswith('-') or line.startswith('•'):
                line_clean = line[1:].strip()
                bullet_text = f'<bullet>•</bullet> {line_clean}'
                story.append(Paragraph(bullet_text, korean_normal))
                
            elif line.startswith(tuple([f'{i}.' for i in range(1,10)])):
                story.append(Paragraph(f"  {line}", korean_normal))
                
            else:
                # 일반 텍스트
                line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                story.append(Paragraph(line, korean_normal))
            
            story.append(Spacer(1, 0.02*inch))
        
        # ===== 푸터 =====
        story.append(PageBreak())
        story.append(Spacer(1, 2.5*inch))
        
        footer_table = Table([
            ['보고서 끝'],
            [f'생성 일시: {date_str}'],
            ['Powered by AI Analysis System']
        ], colWidths=[6.5*inch])
        footer_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (0,0), self.korean_font),
            ('FONTSIZE', (0,0), (0,0), 16),
            ('TEXTCOLOR', (0,0), (0,0), primary_color),
            ('FONTNAME', (0,1), (-1,-1), self.korean_font),
            ('FONTSIZE', (0,1), (-1,-1), 10),
            ('TEXTCOLOR', (0,1), (-1,-1), colors.grey),
            ('LINEABOVE', (0,0), (-1,0), 2, accent_color),
            ('TOPPADDING', (0,0), (-1,-1), 10),
        ]))
        story.append(footer_table)
        
        # PDF 생성
        try:
            doc.build(story)
            print(f"✓ 개선된 PDF 보고서 생성 완료: {output_path}")
        except Exception as e:
            print(f"❌ PDF 생성 실패: {e}")
            import traceback
            traceback.print_exc()
    
    
    
    def run_full_analysis(self, data_file_path: str, output_pdf_path: str = "./report.pdf"):
        """
        전체 분석 파이프라인 실행
        
        Args:
            data_file_path: 센서 데이터 파일 경로 (.pdf, .csv, .xlsx)
            output_pdf_path: 출력 보고서 PDF 경로
        """
        print("\n" + "="*80)
        print("반도체 공정 데이터 분석 시작")
        print("="*80)
        
        # 1. 데이터 로드 (자동 형식 감지)
        print(f"\n[1/4] 데이터 로드 중... ({os.path.basename(data_file_path)})")
        df = self.load_data(r"C:\skala_workspace\MLOps\model_serving_win\server_model\semiconductor_quality_control.csv")
        
        if df.empty:
            print("❌ 데이터 로드 실패. 분석을 중단합니다.")
            return
        
        # 2. EDA 시각화
        print("\n[2/4] EDA 그래프 생성 중...")
        image_paths = self.create_eda_visualizations(df)
        
        # 3. LLM 보고서 생성
        print("\n[3/4] AI 분석 보고서 생성 중...")
        llm_report = self.generate_llm_report(df)
        
        # 4. PDF 보고서 생성
        print("\n[4/4] 최종 PDF 보고서 생성 중...")
        self.create_pdf_report(df, image_paths, llm_report, output_pdf_path)
        
        print("\n" + "="*80)
        print(f"✓ 분석 완료! 보고서: {output_pdf_path}")
        print("="*80)

if __name__ == "__main__":
    print("="*80)
    print("반도체 공정 데이터 분석 RAG 에이전트")
    print("Semiconductor Process Analysis RAG Agent")
    print("="*80)
    
    # 에이전트 초기화 및 분석 실행
    try:
        print("\n에이전트 초기화 중...")
        # RAG 없이 실행 (지식베이스 PDF 없어도 작동)
        agent = SemiconductorRAGAgent(knowledge_pdf_paths=["1.pdf", "2.pdf","3.pdf","4.pdf","5.pdf","6.pdf","7.pdf"],
                                      force_reembed=False)
        
        print("\n전체 분석 파이프라인 실행 중...")
        agent.run_full_analysis(
            data_file_path="semiconductor_quality_control.csv",
            output_pdf_path="반도체_분석_보고서4.pdf"
        )
        
        print("\n" + "="*80)
        print("✅ 분석 완료!")
        print("생성된 파일:")
        print("  - sample_sensor_data.csv (샘플 데이터)")
        print("  - 반도체_분석_보고서.pdf (최종 보고서)")
        print("  - ./eda_plots/ (EDA 그래프)")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ 분석 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
