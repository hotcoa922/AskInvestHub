from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate, PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain.chains import LLMChain, llm

from prompts.prompts import LEGAL_COMPLIANCE_PROMPT, FEE_SERVICE_PROMPT #미리 정의한 프롬프트 가져오기

from core.settings import AOAI_ENDPOINT, AOAI_API_KEY, AOAI_DEPLOY_GPT4O, AOAI_DEPLOY_EMBED_3_LARGE, \
    AOAI_DEPLOY_EMBED_3_SMALL

import logging

# 로그 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

import os
def init_langsmith():
    # LangSmith 설정 활성화
    # 환경변수 LANGSMITH_TRACING, LANGSMITH_ENDPOINT, LANGSMITH_API_KEY 등이 자동으로 반영됨
    if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
        print("[LangSmith] Tracing is enabled. All LLM calls will be tracked.")


called_tools = []

def log_tool_call(func):
    def wrapper(*args, **kwargs):
        logging.info(f"Tool '{func.__name__}' 호출됨. 인자: {args}, {kwargs}")
        result = func(*args, **kwargs)
        logging.info(f"Tool '{func.__name__}' 응답: {result}")
        # 전역 리스트에 호출된 도구 이름을 중복 없이 추가
        if func.__name__ not in called_tools:
            called_tools.append(func.__name__)
        return result
    return wrapper

# PDF 전처리 및 벡터 DB 구축 함수
import os, re
from langchain_community.document_loaders import PyPDFLoader                     # pdf에서 텍스트 추출
from langchain_community.vectorstores import FAISS                            # 고속 벡터 검색을 위한 라이브러리
from langchain_community.embeddings import AzureOpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from langchain.schema import Document

vector_store = None

def preprocess_legal_text(text: str) -> str:
    """
    '제...조' 또는 '제...항' 패턴 앞뒤에 줄바꿈을 추가하고,
    '제1조\n(목적)' 같은 경우를 '제1조(목적)'로 바꿔주는 전처리.
    """
    text = re.sub(r'(제\s*\d+(?:의\d+)?\s*(?:조|항))', r'\n\1\n', text)
    text = re.sub(r'(조|항)\n\(', r'\1(', text)
    text = re.sub(r'\n+', '\n', text)
    return text


def split_by_article(text: str) -> list:
    """
    '제N조(...)'를 기준으로 청크를 분할.
    """
    text = preprocess_legal_text(text)
    pattern = r'(제\s*\d+조(?:의\d+)?\([^)]*\).+?)(?=제\s*\d+조(?:의\d+)?\(|$)'
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if not matches:
        return [text.strip()]
    return [m.strip() for m in matches]


def chunk_by_chars(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """
    글자수 기준으로 text를 chunk_size 단위로 쪼개되,
    각 청크 사이에 overlap 만큼 겹치도록 분할.
    """
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        # 다음 청크 시작을 overlap만큼 겹쳐서
        start = end - overlap
        if start < 0:
            start = 0

    return chunks


def initialize_vector_store():
    """
    서버 시작 전에 벡터 DB를 초기화(로드 또는 생성)하는 함수.
    """
    global vector_store
    pdf_directory = "/data/legal_docs"  # 실제 PDF 문서 경로
    vector_store_dir = "/data/vector_store/faiss_index"

    if os.path.exists(vector_store_dir):
        logger.info("저장된 벡터 DB 로드 중...")
        embeddings = AzureOpenAIEmbeddings(
            model=AOAI_DEPLOY_EMBED_3_LARGE,
            openai_api_version="2024-02-01",
            api_key=AOAI_API_KEY,
            azure_endpoint=AOAI_ENDPOINT
        )
        try:
            vector_store = FAISS.load_local(vector_store_dir, embeddings, allow_dangerous_deserialization=True)
            logger.info(f"벡터 DB 로드 완료 (차원: {vector_store.index.d})")
        except Exception as e:
            logger.error(f"벡터 DB 로드 실패: {e}, 새로 생성 중...")
    else:
        logger.info("벡터 DB 구축 중...")
        documents = []
        if os.path.exists(pdf_directory):
            for filename in os.listdir(pdf_directory):
                if filename.lower().endswith(".pdf"):
                    file_path = os.path.join(pdf_directory, filename)
                    if os.stat(file_path).st_size == 0:
                        logger.warning("빈 PDF 파일 건너뛰기: %s", file_path)
                        continue

                    try:
                        loader = PyPDFLoader(file_path)
                        docs = loader.load()  # 페이지별 Document 리스트
                    except Exception as e:
                        logger.warning("PDF 로드 중 오류 발생, 파일 무시: %s, 오류: %s", file_path, e)
                        continue

                    # 1) 여러 페이지를 하나로 합침
                    merged_text = "\n".join(doc.page_content for doc in docs)

                    # 2) '제N조(...)' 단위로 분할
                    article_chunks = split_by_article(merged_text)

                    for article_chunk in article_chunks:
                        # 3) 각 article_chunk를 다시 500자 단위로 재분할 (오버랩 50자)
                        small_chunks = chunk_by_chars(article_chunk, chunk_size=500, overlap=50)

                        # 기사(조항) 제목을 추출
                        match = re.search(r'(제\s*\d+조(?:의\d+)?\([^)]*\))', article_chunk)
                        if match:
                            article_name = match.group(1).strip()
                        else:
                            article_name = "UNKNOWN_ARTICLE"

                        # 4) 최종 Document 생성
                        for i, small_chunk in enumerate(small_chunks, start=1):
                            documents.append(
                                Document(
                                    page_content=small_chunk,
                                    metadata={
                                        "source": f"{filename} - {article_name}",
                                        "chunk_idx": i
                                    }
                                )
                            )

            logger.info("문서 전처리 완료. 총 %d개 청크", len(documents))
        else:
            logger.warning("PDF 디렉토리가 존재하지 않습니다: %s", pdf_directory)

        embeddings = AzureOpenAIEmbeddings(
            model=AOAI_DEPLOY_EMBED_3_LARGE,
            openai_api_version="2024-02-01",
            api_key=AOAI_API_KEY,
            azure_endpoint=AOAI_ENDPOINT,
        )
        vector_store = FAISS.from_documents(documents, embeddings)
        vector_store.save_local(vector_store_dir)
        logger.info("벡터 DB 구축 완료.")

    logger.info("최종: 벡터 DB 구축/로드 완료.")


# AzureChatOpenAI 인스턴스 생성
chat_llm = AzureChatOpenAI(
    azure_endpoint=AOAI_ENDPOINT,
    api_key=AOAI_API_KEY,
    azure_deployment = AOAI_DEPLOY_GPT4O,
    api_version="2024-08-01-preview",
)


# 직접호출방식, tool데코레이터 방식중 tool데코레이터 방식 채택

from langchain.tools import tool


@tool
def legal_compliance(query: str) -> str:
    """
    주어진 질문에 대해 법률 적합성을 평가하는 응답을 생성합니다.
    최종 응답은 LEGAL_COMPLIANCE_PROMPT에 따라 작성되어야 하며,
    rag_argumentation의 증강 내용을 반영하여 답변하도록 합니다.
    """

    augmented_data = rag_argumentation(query)
    modified_prompt = f"""
    {LEGAL_COMPLIANCE_PROMPT}
    [증강 내용]
    {augmented_data}
    """

    prompt_template = PromptTemplate(
        template=modified_prompt,
        input_variables=["question"]
    )

    formatted_prompt = prompt_template.format(question=query)
    response = chat_llm.invoke(formatted_prompt)
    return response.content.strip()



@tool
def info_service(query: str) -> str:
    """
    주어진 질문에 기반하여 수수료 및 서비스 정보를 분석한 응답을 생성합니다.
    """
    prompt_template = PromptTemplate(
        template=FEE_SERVICE_PROMPT,
        input_variables=["question"]
    )
    logger.info("info service 작동")
    logger.info(f"{prompt_template}")
    formatted_prompt = prompt_template.format(question=query)
    response = chat_llm.invoke(formatted_prompt)
    return response.content.strip()

# RAG방식으로 법률 문서 정보를 요약하는 것
def rag_argumentation(query: str) -> str:
    """
    RAG 방식으로, 전역 vector_store에서 query와 유사한 문서를 검색하여
    추가 정보를 생성한 뒤 반환합니다.
    """
    logger.info("rag_argumentation 호출됨, query: %s", query)

    global vector_store
    if not vector_store:
        logger.warning("vector_store가 초기화되지 않았습니다. RAG를 건너뜁니다.")
        return "No augmented data (vector_store not initialized)."

    try:
        # ✅ 쿼리 벡터 차원 확인
        query_vector = vector_store.embedding_function.embed_query(query)
        if len(query_vector) != vector_store.index.d:
            logger.error(f"벡터 차원 불일치! FAISS={vector_store.index.d}, Query={len(query_vector)}")
            return "Vector dimension mismatch error."

        # 1. 유사 문서 검색
        similar_docs = vector_store.similarity_search(query, k=5)
    except AssertionError as e:
        logger.error(f"FAISS 벡터 검색 오류 발생: {e}")
        return "Vector search error."
    except Exception as e:
        logger.error(f"RAG 검색 중 예외 발생: {e}")
        return "RAG processing error."


    if not similar_docs:
        logger.info("유사 문서가 없습니다.")
        return "No relevant documents found."

    # 2. 검색된 문서 내용 간단 요약/결합
    # 간단히 본문만 합치는 예시 (실제론 LLM Summarization Chain 등 사용 가능)
    combined_text = ""
    for i, doc in enumerate(similar_docs, start=1):
        snippet = doc.page_content[:3000]  # 일부만 잘라내거나 원하는 만큼

        # 로그에 간략히 출력 (메타데이터 + 스니펫 앞부분)
        logger.info(f"[RAG] 문서 {i} (metadata={doc.metadata}): {snippet[:30000]}...")

        combined_text += f"\n--- 문서 {i} ---\n{snippet}\n"

    # 최종 결과
    result = f"Augmented data (Top {len(similar_docs)} docs):\n{combined_text}"
    logger.info("rag_argumentation 반환 내용:\n%s", result)
    return result


def process_brokerage_agent(query: str) -> str:
    """
     LLM에게 적절한 도구를 선택하도록 요청
     선택된 도구를 직접 실행
    """

    #  도구 선택을 위한 LLM 프롬프트 (JSON 없이)
    tool_selection_prompt = f"""
    아래의 질문을 분석하여, 적절한 도구를 선택하세요.

    질문: "{query}"

    ### 사용 가능한 도구:
    - info_service: 금융 서비스 및 수수료 정보를 제공합니다.
    - legal_compliance: 법률 및 규제 적합성을 분석합니다.

    **다음과 같은 형식으로만 응답하세요 (예시):**
    - 선택: info_service
    - 선택: legal_compliance
    """

    # LLM에게 도구 선택 요청
    response = chat_llm.invoke(tool_selection_prompt)
    selected_tool = response.content.strip().lower()  # 소문자로 변환하여 비교

    logger.info(f"도구 선택 응답: {selected_tool}")

    # LLM이 선택한 도구 실행 (JSON 파싱 없이 간단한 조건문 활용)
    if "info_service" in selected_tool:
        return info_service.invoke(query)
    elif "legal_compliance" in selected_tool:
        return legal_compliance.invoke(query)
    else:
        return "유효한 도구가 선택되지 않았습니다."