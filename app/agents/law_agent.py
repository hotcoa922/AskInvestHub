from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from prompts.prompts import LEGAL_COMPLIANCE_PROMPT, FEE_SERVICE_PROMPT #미리 정의한 프롬프트 가져오기

# from langgraph import graph

from core.settings import AOAI_ENDPOINT, AOAI_API_KEY, AOAI_DEPLOY_GPT4O


# PDF 전처리 및 벡터 DB 구축 함수
import os
from langchain.document_loaders import PyPDFLoader                  # pdf에서 텍스트 추출
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 텍스트 분할 방식
from langchain.vectorstores import FAISS                            # 고속 벡터 검색을 위한 라이브러리
from langchain.embeddings import AzureOpenAIEmbeddings
from core.settings import AOAI_DEPLOY_EMBED_ADA

def build_vector_store(pdf_directory: str) -> FAISS:
    documents = [] # 모든 PDF 파일에서 추출된 문서를 저장할 리스트

    # 1. PDF 파일 로드 및 텍스트 추출
    for filename in os.listdir(pdf_directory):  # 내부의 모든 파일 목록을 가져옴
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(pdf_directory, filename)   # 파일 전체경로 생성
            loader = PyPDFLoader(file_path)     # PDF 파일 로드할 수 있는 PyPDFLoader객체 생성
            docs = loader.load()        # PDF에서 텍스트 추출
            documents.extend(docs)      # 추출된 문서를 documents 리스트에 추가

    # 2. 텍스트 분할: RecursiveCharacterTextSplitter 사용
    # chunk_size와 chunk_overlap은 필요에 따라 조정 가능
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    # 3. 임베딩 생성: 각 텍스트 청크를 벡터화
    embeddings = AzureOpenAIEmbeddings(
        endpoint=AOAI_ENDPOINT,
        api_key=AOAI_API_KEY,
        deployment=AOAI_DEPLOY_EMBED_ADA,
        api_version="2024-08-01-preview",       # 추후 변경
    )

    # 4. FAISS 벡터 스토어 생성
    vector_store = FAISS.from_documents(docs, embeddings)   # 임베딩 벡터를 FAISS DB로 변환
    return vector_store     # FAISS 객체 반환


def save_vector_store(vector_store: FAISS, save_path: str):
    vector_store.save(save_path)

def load_vector_store(save_path: str, embeddings) -> FAISS:
    return FAISS.load(save_path, embeddings)

# 직접 실행될 때만 해당 코드 블록이 실행
if __name__ == "__main__":
    pdf_directory = "data/legal_docs"
    vector_store_dir = "data/vector_store/faiss_index"

    if os.path.exists(vector_store_dir):
        print("저장된 벡터 DB 로드 중...")
        embeddings = AzureOpenAIEmbeddings(
            endpoint=AOAI_ENDPOINT,
            api_key=AOAI_API_KEY,
            deployment=AOAI_DEPLOY_EMBED_ADA,
            api_version="2024-08-01-preview",
        )
        vector_store = load_vector_store(vector_store_dir, embeddings)
    else:
        print("벡터 DB 구축 중...")
        vector_store = build_vector_store(pdf_directory)
        save_vector_store(vector_store, vector_store_dir)

    print("벡터 DB 구축/로드 완료.")




# AzureChatOpenAI 인스턴스 생성
chat_llm = AzureChatOpenAI(
    endpoint=AOAI_ENDPOINT,
    api_key=AOAI_API_KEY,
    deployment = AOAI_DEPLOY_GPT4O,
    api_version="2024-08-01-preview",
)


# 직접호출방식, tool데코레이터 방식중 tool데코레이터 방식 채택

from langchain.tools import tool

@tool
def legal_compliance(query: str) -> str:
    prompt_template = PromptTemplate(
        template=LEGAL_COMPLIANCE_PROMPT,
        input_variables=["question"]
    )
    chain = LLMChain(llm=chat_llm, prompt= prompt_template)     # 체인 구성
    return chain.run(question=query).strip()

@tool
def info_service(query: str) -> str:
    prompt_template = PromptTemplate(
        template=FEE_SERVICE_PROMPT,
        input_variables=["question"]
    )
    chain = LLMChain(llm=chat_llm, prompt = prompt_template)     # 체인 구성
    return chain.run(question=query).strip()

# RAG방식으로 법률 문서 정보를 요약하는 것
@tool
def rag_argumentation(query: str) -> str:
    return f"Augmented data: Based on the query '{query}', additional legal documents were summarized."


# 에이전트 초기화 및 실행
from langchain.agents import initialize_agent, AgentType

def process_law_agent(query: str) -> dict:
    #데코레이터 사용 할 때의 방식 코딩
    tools = [
        legal_compliance,
        info_service,
        rag_argumentation
    ]

    agent = initialize_agent(
        tools,
        chat_llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    result = agent.run(query)
    return result
