import os
import sys
from dotenv import load_dotenv

# LlamaIndex 관련 임포트
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini

# 1. 환경변수 로드 (.env 파일에서 GOOGLE_API_KEY 읽기)
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    print("❌ 에러: .env 파일에 GOOGLE_API_KEY가 없습니다.")
    sys.exit(1)

def init_rag_system():
    print("🚀 [System] RAG 시스템 초기화 중...")

    # ---------------------------------------------------------
    # A. 임베딩 모델 설정 (encord-team/ebind-full)
    # ---------------------------------------------------------
    print("📥 [Embedding] ebind-full 모델 로딩 중... (최초 실행 시 다운로드 오래 걸림)")
    
    # trust_remote_code=True : 이 모델은 커스텀 코드가 있어서 반드시 켜야 함
    embed_model = HuggingFaceEmbedding(
        model_name="encord-team/ebind-full",
        trust_remote_code=True,
        device="cpu"  # GPU 있으면 "cuda"로 변경하세요 (속도 차이 큼)
    )

    # ---------------------------------------------------------
    # B. LLM 설정 (Google Gemini 1.5 Flash)
    # ---------------------------------------------------------
    print("🤖 [LLM] Google Gemini 1.5 Flash 연결 중...")
    
    # 나중에 2.5 나오면 model_name="models/gemini-2.5-flash"로 바꾸면 됨
    llm = Gemini(
        model="models/gemini-2.5-flash", 
        api_key=google_api_key,
        temperature=0.1 # 0에 가까울수록 사실 기반 답변 (RAG에 유리)
    )

    # ---------------------------------------------------------
    # C. 전역 설정 (LlamaIndex에게 "이제부터 이거 써" 라고 등록)
    # ---------------------------------------------------------
    Settings.embed_model = embed_model
    Settings.llm = llm
    
    print("✅ [System] 설정 완료!")

def main():
    # 설정 초기화
    init_rag_system()

    # 1. 데이터 로드 (./data 폴더에 있는 모든 파일 읽기)
    if not os.path.exists("./data"):
        os.makedirs("./data")
        with open("./data/sample.txt", "w", encoding="utf-8") as f:
            f.write("LlamaIndex는 데이터 프레임워크입니다. 강사님은 서버 개발자 출신입니다.")
        print("⚠️ ./data 폴더가 없어서 샘플 파일을 생성했습니다.")

    print("📚 [Data] 문서 인덱싱(Vectorizing) 시작...")
    documents = SimpleDirectoryReader("./data").load_data()
    
    # 2. 인덱스 생성 (여기서 ebind-full이 열심히 돕니다)
    index = VectorStoreIndex.from_documents(documents)
    
    # 3. 쿼리 엔진 생성
    query_engine = index.as_query_engine()

    # 4. 질문하기
    print("\n" + "="*30)
    user_question = "회사에서 키우는 개 이름이 뭐임?"
    print(f"❓ 질문: {user_question}")
    
    response = query_engine.query(user_question)
    
    print(f"💡 답변: {response}")
    print("="*30)

if __name__ == "__main__":
    main()