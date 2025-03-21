from langchain.agents import initialize_agent, AgentType
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from prompts.prompts import FINANCIAL_STATEMENT_PROMPT, MARKET_INFO_PROMPT

from core.settings import AOAI_ENDPOINT, AOAI_API_KEY, AOAI_DEPLOY_GPT4O


import logging
# 로그 설정\
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
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
    """
    LLM에게 적절한 도구를 선택하도록 요청
    선택된 도구를 직접 실행
    """

    tool_selection_prompt = f"""
    아래의 질문을 분석하여, 적절한 도구를 선택하세요.

    질문: "{query}"

    ### 사용 가능한 도구:
    - financial_statement: 여러 상장 기업의 정보나 재무 정보를 제공합니다. 
    - market_info: 오늘이나 과거, 미래 등의 시장 상황에 대해 정보를 제공합니다.

    **다음과 같은 형식으로만 응답하세요 (예시):**
    - 선택: financial_statement
    - 선택: market_info
    """

    # LLM에게 도구 선택 요청
    response = chat_llm.invoke(tool_selection_prompt)
    selected_tool = response.content.strip().lower()  # 소문자로 변환하여 비교

    logger.info(f"도구 선택 응답: {selected_tool}")

    # LLM이 선택한 도구 실행 (JSON 파싱 없이 간단한 조건문 활용)
    if "financial_statement" in selected_tool:
        return financial_statement.invoke(query)
    elif "market_info" in selected_tool:
        return market_info.invoke(query)
    else:
        return "유효한 도구가 선택되지 않았습니다."
