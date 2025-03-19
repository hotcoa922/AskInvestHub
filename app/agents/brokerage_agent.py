from langchain_openai import AzureChatOpenAI
from langchain.chains import LLMChain

from prompts.prompts import LEGAL_COMPLIANCE_PROMPT, FEE_SERVICE_PROMPT #미리 정의한 프롬프트 가져오기

from core.settings import AOAI_ENDPOINT, AOAI_API_KEY, AOAI_DEPLOY_GPT4O, AOAI_DEPLOY_EMBED_3_LARGE

import logging

# 로그 설정
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 로깅 기본 설정 (필요에 따라 포맷 및 출력 대상 조정 가능)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

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
    법률 문서 전처리 함수:
    '제...조' 또는 '제...항' 패턴 앞뒤에 줄바꿈을 추가하여 구조를 분리합니다.
    """
    text = re.sub(r'(제\s*\d+\s*(조|항))', r'\n\1\n', text)
    text = re.sub(r'\n+', '\n', text)  # 과도한 줄바꿈 정리
    return text


def split_by_articles(text: str) -> list:
    """
    법률 문서를 '제...조' 또는 '제...항' 단위로 분할하는 함수.
    각 청크는 하나의 조항 또는 항으로 구성됩니다.
    """
    # 전처리: 법률 텍스트 구조 강조
    text = preprocess_legal_text(text)
    # '제' 다음에 숫자와 '조' 또는 '항'이 나오는 부분을 기준으로 분할 (lookahead 사용)
    pattern = r'(?=(제\s*\d+\s*(조|항)))'
    parts = re.split(pattern, text)

    # 캡처 그룹을 재조합하여 청크 생성
    chunks = []
    current = ""
    for part in parts:
        if re.match(r'제\s*\d+\s*(조|항)', part):
            if current:
                chunks.append(current.strip())
            current = part  # 새 청크 시작
        else:
            current += part
    if current:
        chunks.append(current.strip())
    return chunks



def initialize_vector_store():
    """
    서버 시작 전에 벡터 DB를 초기화(로드 또는 생성)하는 함수.
    """
    pdf_directory = "/data/legal_docs"  # 실제 PDF 문서 경로
    vector_store_dir = ".data/vector_store/faiss_index"

    if os.path.exists(vector_store_dir):
        logger.info("저장된 벡터 DB 로드 중...")
        embeddings = AzureOpenAIEmbeddings(
            model=AOAI_DEPLOY_EMBED_3_LARGE,
            openai_api_version="2024-02-01",
            api_key=AOAI_API_KEY,
            azure_endpoint=AOAI_ENDPOINT
        )
        vector_store = FAISS.load_local(vector_store_dir, embeddings)
        logger.info("벡터 DB 로드 완료.")
    else:
        logger.info("벡터 DB 구축 중...")
        documents = []
        if os.path.exists(pdf_directory):
            for filename in os.listdir(pdf_directory):
                if filename.lower().endswith(".pdf"):
                    file_path = os.path.join(pdf_directory, filename)
                    # 파일 크기가 0바이트이면 건너뜁니다.
                    if os.stat(file_path).st_size == 0:
                        logger.warning("빈 PDF 파일 건너뛰기: %s", file_path)
                        continue
                    try:
                        loader = PyPDFLoader(file_path)
                        docs = loader.load()
                    except Exception as e:
                        logger.warning("PDF 로드 중 오류 발생, 파일 무시: %s, 오류: %s", file_path, e)
                        continue
                    for doc in docs:
                        chunks = split_by_articles(doc.page_content)
                        for chunk in chunks:
                            documents.append(Document(page_content=chunk, metadata=doc.metadata))
            logger.info("문서 전처리 완료. 총 %d개 청크", len(documents))
        else:
            logger.warning("PDF 디렉토리가 존재하지 않습니다: %s", pdf_directory)
        embeddings = AzureOpenAIEmbeddings(
            model=AOAI_DEPLOY_EMBED_3_LARGE,
            openai_api_version="2024-02-01",
            api_key=AOAI_API_KEY,
            azure_endpoint=AOAI_ENDPOINT
        )
        vector_store = FAISS.from_documents(documents, embeddings)
        vector_store.save_local(vector_store_dir)
        logger.info("벡터 DB 구축 완료.")
    logger.info("최종: 벡터 DB 구축/로드 완료.")
    return  # 실제로 vector_store를 전역 변수에 저장하거나 반환할 수 있음.



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

    logger.info("legal_compliance 작동, 질문: %s", query)
    formatted_prompt = prompt_template.format(question=query)

    response = chat_llm.invoke(formatted_prompt)
    logger.info("LLM 응답: %s", response.content.strip())
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

    # 1. 유사 문서 검색
    # k=3 정도로 상위 3개 문서를 검색해 예시로 사용
    similar_docs = vector_store.similarity_search(query, k=3)
    if not similar_docs:
        logger.info("유사 문서가 없습니다.")
        return "No relevant documents found."

    # 2. 검색된 문서 내용 간단 요약/결합
    # 간단히 본문만 합치는 예시 (실제론 LLM Summarization Chain 등 사용 가능)
    combined_text = ""
    for i, doc in enumerate(similar_docs, start=1):
        snippet = doc.page_content[:300]  # 일부만 잘라내거나 원하는 만큼
        combined_text += f"\n--- 문서 {i} ---\n{snippet}\n"

    # 최종 결과
    result = f"Augmented data (Top {len(similar_docs)} docs):\n{combined_text}"
    logger.info("rag_argumentation 반환 내용:\n%s", result)
    return result


from langchain.agents import initialize_agent, AgentType
from langchain.prompts import PromptTemplate


def process_brokerage_agent(query: str) -> str:
    tools = [legal_compliance, info_service]  # 두 도구만 사용
    custom_prefix = """
    You have access to the following tools:
    - legal_compliance(query: str) -> str: Generates a detailed legal analysis using the provided legal compliance prompt.
      The final answer MUST strictly follow the format below:

      [문제상황분석]
      질문에서 제기된 문제 상황을 명확하게 분석하고, 주요 이슈와 이해관계자, 상황의 배경을 구체적으로 서술합니다.

      [관련 법]
      1. [법률명]: 해당 법률의 주요 내용과 질문과의 관련성을 간단히 설명
      2. [법률명]: 관련 조항 및 적용 가능성 등
      3. [법률명]: 추가로 고려해야 할 법률적 측면

      [메인]
      문제 상황에 대한 전반적인 논리 전개 및 분석을 수행합니다. 관련 법률의 적용, 판례, 해석 등을 종합하여 주장의 근거를 체계적으로 서술합니다.
      최대한 상세하고 많은 내용을 담습니다.

      [결론]
      전체 분석을 토대로 내릴 수 있는 최종 판단과 권고사항을 명확하게 제시합니다.

      [추가고려점]
      추가적으로 고려해야 할 법률적, 실무적, 정책적 사항이나 주의사항이 있다면 기재합니다.

      [요약]
      전체 응답의 핵심 내용을 한두 문장으로 간략하게 요약합니다.

    - info_service(query: str) -> str: Generates a detailed analysis of fee and service information.

    Your task is to automatically choose the appropriate tool based solely on the user's input. If legal_compliance is chosen, the final answer MUST strictly adhere to the above format.
    """

    agent = initialize_agent(
        tools,
        chat_llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        agent_kwargs={"prefix": custom_prefix}
    )

    result = agent.invoke({"input": query})


    if isinstance(result, dict):
        if "content" in result:
            return result["content"].strip()
        elif "output" in result:
            return result["output"].strip()
        else:
            raise ValueError("결과 딕셔너리에 'content' 또는 'output' 키가 없습니다.")
    elif isinstance(result, str):
        return result.strip()
    else:
        raise ValueError("agent.invoke의 반환값을 처리할 수 없습니다.")