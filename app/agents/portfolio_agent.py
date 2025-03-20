from langchain.agents import initialize_agent, AgentType
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from prompts.prompts import FINANCIAL_STATEMENT_PROMPT, MARKET_INFO_PROMPT

from core.settings import AOAI_ENDPOINT, AOAI_API_KEY, AOAI_DEPLOY_GPT4O


import logging
# 로그 설정\
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


chat_llm = AzureChatOpenAI(
    azure_endpoint=AOAI_ENDPOINT,
    api_key = AOAI_API_KEY,
    azure_deployment = AOAI_DEPLOY_GPT4O,
    api_version = "2024-08-01-preview",
)

from langchain_core.tools import tool

@tool
def financial_statement(query: str) -> str:
    """
    주어진 질문에 대해 재무제표 분석을 제공합니다.
    """

    modified_prompt = f"""
    {FINANCIAL_STATEMENT_PROMPT}
    """
    prompt_template = PromptTemplate(
        template=modified_prompt,
        input_variables=["question"]
    )
    logger.info("financial_statement 작동")
    formatted_prompt = prompt_template.format(question=query)
    response = chat_llm.invoke(formatted_prompt)
    return response.content.strip()


# 오늘의 시황은 추후 상세 구현 예정
@tool
def market_info(query: str) -> str:
    """
    주어진 질문에 대해 일일 시황을 반홥합니다.
    """
    prompt_template = PromptTemplate(
        template=MARKET_INFO_PROMPT,
        input_variables=["question"]
    )
    logger.info("market_info")
    formatted_prompt = prompt_template.format(question=query)
    response = chat_llm.invoke(formatted_prompt)
    return response.content.strip()



def process_portfolio_agent(query: str) -> str:
    tools = [financial_statement, market_info]  # 두 도구만 사용
    custom_prefix = """
    You have access to the following tools:
        - financial_statement(query: str) -> str: 기업에 대한 상세한 재무적인 내용을 포함하여 답변하며 제공, .
          당신은 최대한 상세하게 답변을 해야합니다. The final answer MUST strictly follow the format below:
            
          [주요 재무정보]
            매출: 최신 매출정보를 가져옴, 미 발견시 데이터 없음 표시
            영업이익: 최신 정보를 가져옴, 미 발견시 데이터 없음 표시
            순이익: 최신 정보를 가져옴, 미 발견시 데이터 없음 표시
            부채와 자본 비율: 최신 정보를 가져옴, 미 발견시 데이터 없음 표시
            배당금 지급 현황: 최신 정보를 가져옴, 미 발견시 데이터 없음 표시
            연구 및 개발(R&D) 지출: 새최신 정보를 가져옴, 미 발견시 데이터 없음 표시
            
            [주요 비용 항목]
            운영 비용: 최신 정보를 가져옴, 미 발견시 데이터 없음 표시 + 주요 항목들 표시
            물류 및 유통비: 최신 정보를 가져옴, 미 발견시 데이터 없음 표시 + 주요 항목들 표시
            규제 준수 비용: 최신 정보를 가져옴, 미 발견시 데이터 없음 표시 + 주요 항목들 표시
            에너지 비용: 최신 정보를 가져옴, 미 발견시 데이터 없음 표시 + 주요 항목들 표시

            [기타 서비스 및 전략]
            위험 관리 및 금융 전략: 최신 정보를 가져옴
            연구 개발(R&D): 최신 정보를 가져옴
            지속 가능성 경영: 최신 정보를 가져옴

            [관련 정보 확인 위치]


            [요약]
            
        - market_info(query: str) -> str: 오늘의 시황 정보는 현재 미구현 상태라고 사용자에게 알림.
            
        Your task is to automatically choose the appropriate tool based solely on the user's input.
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