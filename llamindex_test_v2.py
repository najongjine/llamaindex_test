import os
import sys
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini

# 1. 환경변수 로드
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    print("❌ 에러: .env 파일에 GOOGLE_API_KEY가 없습니다.")
    sys.exit(1)

def init_rag_system():
    print("🚀 [System] RAG 시스템 초기화 중...")

    # ---------------------------------------------------------
    # [수정됨] BAAI/bge-m3 (현재 가장 안정적이고 성능 좋은 모델)
    # ---------------------------------------------------------
    print("📥 [Embedding] BAAI/bge-m3 모델 로딩 중...")
    
    # 이 모델은 표준 아키텍처라 trust_remote_code 필요 없고, 에러 안 남
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-m3",
        device="cpu"  # GPU 있으면 "cuda"
    )

    # ---------------------------------------------------------
    # LLM 설정 (Google Gemini 1.5 Flash)
    # ---------------------------------------------------------
    print("🤖 [LLM] Google Gemini 1.5 Flash 연결 중...")
    
    llm = Gemini(
        model="models/gemini-2.5-flash", 
        api_key=google_api_key,
        temperature=0.1
    )

    # ---------------------------------------------------------
    # 전역 설정
    # ---------------------------------------------------------
    Settings.embed_model = embed_model
    Settings.llm = llm
    
    print("✅ [System] 설정 완료!")

def main():
    init_rag_system()

    # 데이터 로드
    if not os.path.exists("./data"):
        os.makedirs("./data")
        with open("./data/manual.txt", "w", encoding="utf-8") as f:
            f.write("LlamaIndex는 데이터 프레임워크입니다. 강사님은 서버 개발자 출신입니다.")

    print("📚 [Data] 문서 인덱싱 시작...")
    documents = SimpleDirectoryReader("./data").load_data()
    
    # 인덱싱
    index = VectorStoreIndex.from_documents(documents)
    
    # 쿼리 엔진
    query_engine = index.as_query_engine()

    # 질문하기
    print("\n" + "="*30)
    user_question = "회사에서 키우는 개 이름이 뭐임?"
    print(f"❓ 질문: {user_question}")
    
    response = query_engine.query(user_question)
    
    print(f"💡 답변: {response}")
    print("="*30)

if __name__ == "__main__":
    main()