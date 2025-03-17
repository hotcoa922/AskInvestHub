# 프로젝트 내에서 에이전트 관련 API 엔드포인트들을 모아둔 파일
# 사실상 endpoints 모음소
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agents.graph_builder import graph


logger = logging.getLogger(__name__)
router = APIRouter()


# 요청 데이터 모델: 사용자 질문은 단일 문자열로 구성
class UserQuestion(BaseModel):
    query: str


@router.post("/ask")
async def ask_question(request: UserQuestion):
    try:
        state = {"messages": [], "latest_query": request.query} # 초기 state 생성
        logger.info(f"📨 [INFO] /ask 엔드포인트 호출됨: {request.query}")
        logger.debug(f"🔍 [DEBUG] graph 실행 전 state: {state}")
        final_state = graph.invoke(state)
        logger.debug(f"✅ [DEBUG] graph 실행 후 state: {final_state}")
        result = final_state
        return result
    except Exception as e:
        logger.error(f"❌ [ERROR] /ask 엔드포인트 실행 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

