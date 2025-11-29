import os
import sys
from dotenv import load_dotenv

# PromptTemplate 임포트 필수
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, PromptTemplate
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
    # [Embedding] BAAI/bge-m3
    # ---------------------------------------------------------
    print("📥 [Embedding] BAAI/bge-m3 모델 로딩 중...")
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-m3",
        device="cpu"  # GPU 있으면 "cuda"
    )

    # ---------------------------------------------------------
    # [LLM] Google Gemini
    # ---------------------------------------------------------
    print("🤖 [LLM] Google Gemini 연결 중...")
    llm = Gemini(
        model="models/gemini-2.5-flash", 
        api_key=google_api_key,
        temperature=0.1
    )

    Settings.embed_model = embed_model
    Settings.llm = llm
    print("✅ [System] 설정 완료!")

def main():
    init_rag_system()

    # 데이터 로드 (샘플 데이터 생성)
    if not os.path.exists("./data"):
        os.makedirs("./data")
    
    # [테스트를 위해 내용을 좀 더 풍성하게 넣었습니다]
    with open("./data/manual.txt", "w", encoding="utf-8") as f:
        f.write("""
        1. 트랜스포머(Transformer)는 구글이 2017년에 발표한 딥러닝 모델이다.
        2. 이 모델은 'Attention' 메커니즘을 사용하여 문맥을 파악한다.
        3. 강사님은 10년 차 서버 개발자 출신이며, Spring Boot와 DB 튜닝 전문가다.
        4. 강사님은 국비지원 과정에서 학생들에게 실무 위주의 교육을 강조한다.
        """)

    print("📚 [Data] 문서 인덱싱 시작...")
    documents = SimpleDirectoryReader("./data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    
    # =================================================================
    # [핵심 수정] 커스텀 프롬프트 (욕쟁이 할머니 페르소나)
    # =================================================================
    my_prompt_str = """
    너는 2d 미소녀 캐릭 메이드 비서다. 
    
    규칙:
    - [정보]에 없는 내용은 니가 알고있는 한도 내에서 답변하라.
    -- 설명은 아주 쉽게

    [정보]
    ---------------------
    {context_str}
    ---------------------

    [손님 질문]: {query_str}

    [AI 답변]:
    """
    
    # 템플릿 객체 생성
    my_template = PromptTemplate(my_prompt_str)

    # 쿼리 엔진에 템플릿 주입 (Dependency Injection)
    # text_qa_template에 우리가 만든 템플릿을 넣어줍니다.
    query_engine = index.as_query_engine(
        text_qa_template=my_template,
        similarity_top_k=3 # 3개 정도만 참고하게 설정
    )

    # =================================================================

    print("\n" + "="*30)
    # 질문을 데이터에 있는 내용으로 바꿔봤습니다.
    user_question = "회사에서 뭐 키워?"
    print(f"❓ 질문: {user_question}")
    
    response = query_engine.query(user_question)
    
    print(f"💡 답변:\n{response}")
    print("="*30)

    # [추가] 서버 개발자용 디버깅: 실제로 뭘 참고했는지 찍어보기
    print("\n🔍 [참고한 문서 조각(Chunk)]")
    for node in response.source_nodes:
        print(f"- (유사도: {node.score:.3f}): {node.node.get_content().strip()[:50]}...")

if __name__ == "__main__":
    main()