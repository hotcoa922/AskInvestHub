from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware

import logging

# 로그 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logger.info("FastAPI 서버 시작됨")

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI(
    title="Ask Invest Hub 서비스(AIH)",
    description="에이전트 기반 AI 서비스 (법률 및 포트폴리오 분석)",
    version="0.7.0"
)
import os
def init_langsmith():
    # LangSmith 설정 활성화
    # 환경변수 LANGSMITH_TRACING, LANGSMITH_ENDPOINT, LANGSMITH_API_KEY 등이 자동으로 반영됨
    if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
        print("[LangSmith] Tracing is enabled. All LLM calls will be tracked.")


# 벡터 DB 초기화 함수 임포트
from agents.brokerage_agent import initialize_vector_store

# 서버 시작 전 벡터 DB 초기화 (startup 이벤트)
@app.on_event("startup")
async def startup_event():
    logger.info("서버 시작 전 벡터 DB 초기화 시작")
    initialize_vector_store()
    logger.info("서버 시작 전 벡터 DB 초기화 완료")

# 기본 엔드포인트: API 테스트용
@app.get("/")
async def root():
    return {"message": "증권 AI 서비스에 오신 것을 환영합니다!"}


if __name__ == "__main__":
    import uvicorn
    # uvicorn을 이용해 서버 실행 (포트: 8000)
    uvicorn.run(app, host="0.0.0.0", port=8000)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # 모든 도메인에서 오는 요청을 허용
    allow_credentials=True,     # 인증 정보(쿠키, 인증 헤더 등)를 포함한 요청을 허용
    allow_methods=["*"],        # 모든 HTTP 메서드(GET, POST, PUT, DELETE 등)의 요청을 허용
    allow_headers=["*"],        # 모든 HTTP 헤더를 허용
)


# 라우터 등록
from api.question_router import router as question_router
app.include_router(question_router, prefix="/agent", tags=["agent"])