from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command
from langchain.schema import HumanMessage, AIMessage
from agents.supervisor import supervisor_agent, SupervisorRequest
from agents.brokerage_agent import process_brokerage_agent
from agents.portfolio_agent import process_portfolio_agent
from typing import List

import logging
logger = logging.getLogger(__name__)

class State(MessagesState):
    query: str
    selected_agent: str
    messages: List

# Supervisor 노드 함수: 사용자의 최신 질문을 분석하여 적합한 에이전트를 결정
def supervisor_node(state) -> Command:

    query = state["messages"][0][-1] # request.query: str

    # SupervisorRequest 모델로 변환하여 supervisor_agent 함수를 호출
    supervisor_req = SupervisorRequest(query=query)
    result = supervisor_agent(supervisor_req)    # result에는 선택된 에이전트와 그 결과가 포함되어 있음
    selected = result.get("selected_agent", "")


    # 대화 메시지 업데이트: 대화 메시지에 supervisor_agent의 응답(예: result["result"])를 추가
    state.setdefault("messages", []).append(
        AIMessage(content=str(result.get("result", "")))
    )

    # 다음 노드 결정: 선택된 에이전트에 따라 "brokerage" 또는 "portfolio"로 전이
    if selected == "brokerage_agent":
        next_node = "brokerage"
    elif selected == "portfolio_agent":
        next_node = "portfolio"
    else:
        next_node = END  # 잘못된 분류 결과면 종료 처리

    state["selected_agent"] = selected

    # Command 객체 반환: 다음 노드로 전이하며 업데이트된 상태 전달
    return Command(goto=next_node, update=state)    # goto는 다음으로 이동할 노드 / update는 현재의 상태


# Brokerage 노드 함수: 증권 관련 질문을 처리
def brokerage_node(state: dict) -> Command:
    query = state["messages"][0][-1]
    result = process_brokerage_agent(query)

    # 응답을 메시지로 업데이트 (여기서는 문자열로 변환)
    state.setdefault("messages", []).append(
        AIMessage(content=str(result))
    )
    logger.info(f"brokerage: {state}")

    # 처리 후 supervisor로 전이 (후속 질문을 위해)
    return Command(goto=END, update=state)


# Portfolio 노드 함수: 포트폴리오 관련 질문을 처리
def portfolio_node(state: dict) -> Command:
    query = state.get("latest_query", "")
    # portfolio_agent는 구조화된 입력을 요구하므로 state에서 portfolio_data를 읽습니다.
    portfolio_data = state.get("portfolio_data", "")
    result = process_portfolio_agent({"query": query, "portfolio_data": portfolio_data})

    state.setdefault("messages", []).append(
        AIMessage(content=str(result))
    )

    return Command(goto=END, update=state)


# 그래프 빌더 함수: 초기 상태와 노드를 구성합니다.
def build_graph():
    # 초기 상태: 대화 기록 및 최신 질문을 포함하는 사전
    # initial_state = {"latest_query": "", "messages": []}
    builder = StateGraph(State)

    # START에서 supervisor로 전이
    builder.add_edge(START, "supervisor")   # add_edge를 통해 start->supervisor 노드로 바로 이동

    # 각 노드 등록 (노드 이름은 문자열로 지정)
    # 노드란 대화의 특정 단계나 역할(예: 질문 분류, 증권 정보 처리, 포트폴리오 정보 처리)을 나타냄
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("brokerage", brokerage_node)
    builder.add_node("portfolio", portfolio_node)

    # 필요 시 supervisor에서 END로의 에지 추가 (종료 조건)
    builder.add_edge("supervisor", END)

    return builder.compile()


# 최종 그래프 객체 (전체 대화 흐름)
graph = build_graph()

