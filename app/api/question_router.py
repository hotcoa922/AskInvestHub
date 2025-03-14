# 프로젝트 내에서 에이전트 관련 API 엔드포인트들을 모아둔 파일
# 사실상 endpoints 모음소
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agents.supervisor import supervisor_agent


router = APIRouter()


# 요청 데이터 모델: 사용자 질문은 단일 문자열로 구성
class UserQuestion(BaseModel):
    query: str

# 아래 코드 폐기
# # 임시 에이전트 결정 함수 (추후 실제 로직으로 대체)
# def decide_agent(question: str) -> str:
#     # 간단 예시: 질문에 "법률"이라는 단어가 포함되면 Agent1, 그렇지 않으면 Agent2
#     if "법률" in question:
#         return "agent1"  # 법률 관련 처리
#     else:
#         return "agent2"  # 포트폴리오 분석 관련 처리

@router.post("/ask")
async def ask_question(request: UserQuestion):
    try:
        result = supervisor_agent(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

