from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from prompts.prompts import LEGAL_COMPLIANCE_PROMPT, FEE_SERVICE_PROMPT #미리 정의한 프롬프트 가져오기

from langgraph import graph

from core.settings import AOAI_ENDPOINT, AOAI_API_KEY, AOAI_DEPLOY_GPT4O

# AzureChatOpenAI 인스턴스 생성
chat_llm = AzureChatOpenAI(
    endpoint=AOAI_ENDPOINT,
    api_key=AOAI_API_KEY,
    deployment = AOAI_DEPLOY_GPT4O,
    api_version="2024-08-01-preview",
)

"""
    직접호출방식, tool데코레이터 방식중 tool데코레이터 방식 채택
"""
from langchain_core.tools import tool

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
    return f"Augmented data: Based on the query '{query}', additional legal documents werw summarized."


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

    # 입력된 포트폴리오 데이터와 시황 관련 질문을 하나의 쿼리로 통합하여 전달
    combined_query = f"포트폴리오 데이터: {request.portfolio_data}. 질문: {request.query}"
    result = agent.run(combined_query)
    return result