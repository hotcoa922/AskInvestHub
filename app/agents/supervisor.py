import logging

from pydantic import BaseModel
from langchain_openai import AzureChatOpenAI
from core.settings import AOAI_ENDPOINT, AOAI_API_KEY, AOAI_DEPLOY_GPT4O
from langchain.chains import LLMChain

logger = logging.getLogger(__name__)

class SupervisorRequest(BaseModel):
    query: str

# 시스템 프롬프트: 에이전트 결정 기준을 시스템 메시지로 설정
SYSTEM_PROMPT = (
    """
    당신은 종합 증권 서비스의 에이전트 결정자입니다.
    아래 기준에 따라 사용자의 질문을 분석하여 반드시 아래 두 가지 중 하나만을 단독으로 출력하세요.

    기준:
    - "brokerage": 질문이 증권관련 법률, 증권사의 수수료 등과 같은 정보, 투자법, 수수료, 규제 내용일 경우.
    - "portfolio": 질문이 기업 재무, 시장 시황, 포트폴리오 구성, 개별 종목 정보, 종목 비교, 포트폴리오 분석 관련 내용일 경우.

    반드시 출력은 추가 문장 없이 단 하나의 단어, "brokerage" 또는 "portfolio"만을 반환해야 합니다.
    """
)

from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

# 시스템 메시지와 인간 메시지를 래핑
system_message = SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT)   # LangChain 내부에서 자동으로 SystemMessage 객체로 변환 -> GPT가 이해할 수 있는 LangChain 시스템 메시지 형식으로 변환
human_message = HumanMessagePromptTemplate.from_template("{query}") # 위와 동일
# json 구조를 직접 만들 필요가 없어진 것

# 채팅 프롬프트 템플릿 생성 (시스템 메시지 + 인간 메시지)
chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

chat_llm = AzureChatOpenAI(
    azure_endpoint=AOAI_ENDPOINT,
    api_key=AOAI_API_KEY,
    azure_deployment=AOAI_DEPLOY_GPT4O,
    api_version="2024-08-01-preview",
)

from agents.brokerage_agent import process_brokerage_agent
from agents.portfolio_agent import process_portfolio_agent

def determine_agent(query: str) -> str:
    # 메시지 목록을 생성
    messages = chat_prompt.format_messages(query=query)
    # AzureChatOpenAI 인스턴스를 호출하여 응답을 받음
    response = chat_llm.invoke(messages)        #deprecated 되었으며 대신 invoke 메서드를 사용
    # -> AIMessage 객체에는 .run() 메서드가 없으므로 .content 속성을 사용해야함
    classification = response.content.strip().lower()
    # 후처리 없이 첫 번째 단어만 사용하되, 가능하면 출력이 확실하도록 함.
    return classification.split()[0]


def supervisor_agent(request: SupervisorRequest) -> dict:
    logger.info(f"📥 [INFO] supervisor_agent 실행: {request.query}")
    agent_type = determine_agent(request.query)     # request.query 통해 사용자 질문 가져옴

    if agent_type not in ["brokerage", "portfolio"]:
        print(f"[ERROR] 잘못된 에이전트 분류: {agent_type}")  # 로그 추가
        return {"error": f"잘못된 에이전트 분류: {agent_type}"}

    if agent_type == "brokerage":
        result = process_brokerage_agent(request.query)
        selected_agent = "brokerage_agent"
    elif agent_type == "portfolio":
        result = process_portfolio_agent(request.query)
        selected_agent = "portfolio_agent"
    else:
        return {
            "error": f"입력 질문에 대해 적절한 에이전트를 결정할 수 없습니다. (분류 결과: {agent_type})"
        }

    return {
        "selected_agent": selected_agent,
        "result": result
    }