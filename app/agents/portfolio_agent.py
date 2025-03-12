from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from prompts.prompts import FINANCIAL_STATEMENT_PROMPT, MARKET_INFO_PROMPT

from core.settings import AOAI_ENDPOINT, AOAI_API_KEY, AOAI_DEPLOY_GPT4O



chat_llm = AzureChatOpenAI(
    endpoint = AOAI_ENDPOINT,
    api_key = AOAI_API_KEY,
    deployment = AOAI_DEPLOY_GPT4O,
    api_version = "2024-08-01-preview",
)

from langchain_core.tools import tool

@tool
def financial_statement(portfolio_data: str) -> str:
    prompt_template = PromptTemplate(
        template = FINANCIAL_STATEMENT_PROMPT,
        input_variables=["portfolio_data"]
    )
    chain = LLMChain(llm = chat_llm, prompt = prompt_template)
    return chain.run(portfolio_data=portfolio_data).strip()


# 오늘의 시황은 추후 상세 구현 예정
@tool
def market_info(query: str) -> str:

    prompt_template = PromptTemplate(
        template=MARKET_INFO_PROMPT,
        input_variables=["query"]
    )
    chain = LLMChain(llm=chat_llm, prompt=prompt_template)
    return chain.run(query=query).strip()


from pydantic import BaseModel
from langchain.agents import initialize_agent, AgentType

# Pydantic 모델 정의: 구조화된 입력 데이터를 검증
class PortfolioRequest(BaseModel):
    query: str
    portfolio_data: str  # 실제 사용시 dict, list 등 더 복잡한 구조로 확장 가능

def process_portfolio_agent(request: PortfolioRequest) -> str:
    tools = [
        financial_statement,
        market_info
    ]

    agent = initialize_agent(
        tools,
        chat_llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )