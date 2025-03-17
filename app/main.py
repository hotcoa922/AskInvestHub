from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware

import logging

# 로그 설정
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logger.info("FastAPI 서버 시작됨")

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI(
    title="Ask Invest Hub 서비스(AIH)",
    description="에이전트 기반 AI 서비스 (법률 및 포트폴리오 분석)",
    version="0.7.0"
)

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