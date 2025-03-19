FROM python:3.11-slim


# 필요한 빌드 도구 설치 (예: gcc, build-essential 등)
RUN apt-get update && apt-get install -y gcc build-essential


# 컨테이너 내 작업 디렉터리를 /app으로 설정
WORKDIR /app

# 의존성 파일 복사 및 설치
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# app 폴더 내부의 코드만 /app 디렉터리에 복사
COPY app/ .
COPY data/ ../data

# 컨테이너 시작 시 uvicorn으로 FastAPI 애플리케이션 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
