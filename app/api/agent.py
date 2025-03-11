# 프로젝트 내에서 에이전트 관련 API 엔드포인트들을 모아둔 파일
# 사실상 endpoints 모음소
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


router = APIRouter()

# 요청 데이터 모델 정의 (예시: 질문)
class AgentRequest(BaseModel):
    question: str


# 임시 에이전트 결정 함수 (추후 실제 로직으로 대체)
def decide_agent(question: str) -> str:
    # 간단 예시: 질문에 "법률"이라는 단어가 포함되면 Agent1, 그렇지 않으면 Agent2
    if "법률" in question:
        return "agent1"  # 법률 관련 처리
    else:
        return "agent2"  # 포트폴리오 분석 관련 처리

@router.post("/agent")
async def select_agent(request: AgentRequest):
    """
    요청된 질문을 분석하여 호출할 에이전트를 결정하고, 결과를 반환합니다.
    """
    agent = decide_agent(request.question)
    if not agent:
        raise HTTPException(status_code=400, detail="적절한 에이전트를 결정할 수 없습니다.")
    return {"selected_agent": agent, "question": request.question}