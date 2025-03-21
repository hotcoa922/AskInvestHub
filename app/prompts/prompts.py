# 법률 적합성 검토 프롬프트 템플릿
LEGAL_COMPLIANCE_PROMPT = """
아래 질문에 기반하여 법률 관련 답변을 해주세요.
질문: {question}

### 응답 형식 (이 형식을 반드시 유지)
### [문제상황분석]
- 질문에서 제기된 문제 상황을 명확하게 분석합니다.
- 주요 이슈와 이해관계자, 상황의 배경을 구체적으로 서술합니다.

### [관련 법]
1. **[법률명]**: 해당 법률의 주요 내용과 질문과의 관련성을 간단히 설명합니다.
2. **[법률명]**: 관련 조항 및 적용 가능성 등을 서술합니다.
3. **[법률명]**: 추가로 고려해야 할 법률적 측면을 기재합니다.
   - 관련법의 갯수는 상황에 따라 조정될 수 있습니다.

### [메인]
- 문제 상황에 대한 전반적인 논리 전개 및 분석을 수행합니다.
- 관련 법률의 적용, 판례, 해석 등을 종합하여 주장의 근거를 체계적으로 서술합니다.
- **최대한 상세하고 많은 내용을 담아야 합니다.**

### [결론]
- 전체 분석을 토대로 내릴 수 있는 최종 판단과 권고사항을 명확하게 제시합니다.

### [추가고려점]
- 추가적으로 고려해야 할 법률적, 실무적, 정책적 사항이나 주의사항을 기재합니다.

### [요약]
- 전체 응답의 핵심 내용을 한두 문장으로 간략하게 요약합니다.
      
중요: 위의 형식을 절대로 변경하지 마세요.


"""

# 수수료 및 기타 서비스 정보 프롬프트 템플릿
FEE_SERVICE_PROMPT = """
아래 질문에 대한 수수료 및 기타 서비스 정보를 분석하여 정리된 형식으로 답변해 주세요.

질문: {question}

## 응답 형식 (이 형식을 반드시 유지)
### [확실한 정보]
- 조사된 확실한 정보를 담고, 기준 날짜를 포함합니다.
- 확정된 정보로 사용자에게 신뢰를 줄 수 있도록 확답 형태로 작성합니다.

### [불확실한 정보]
- 확실하지 않지만 어느 정도 신뢰할 수 있는 정보를 담습니다.
- 불확실한 부분은 추상적으로 서술합니다.

### [종합]
- 확실한 정보와 불확실한 정보를 종합하여 부수적인 정보까지 제공하는 종합적인 답변을 작성합니다.

중요: 위의 형식을 절대로 변경하지 마세요.
"""

# 종목에 대한 재무제표 분석 프롬프트 템플릿
FINANCIAL_STATEMENT_PROMPT = """
아래 질문에 기반하여 관련 답변을 해주세요.
질문: {question}

### 응답 형식 (이 형식을 반드시 유지)
1. 기업 개요
기업명: (예: 삼성전자)
영문명: (예: Samsung Electronics Co., Ltd.)
설립연도: (예: 1969년)
본사 위치: (예: 대한민국 서울특별시 서초구)
대표이사: (예: 한종희, 경계현)
사업 분야: (예: 반도체, 스마트폰, 디스플레이, 가전제품)
주요 제품 및 서비스: (예: 갤럭시 스마트폰, QLED TV, 반도체 메모리)
홈페이지: [공식 웹사이트 링크]
기업 로고: (이미지 링크 또는 첨부)
2. 주식 및 시장 정보
상장 여부: (예: 코스피 상장)
티커(symbol): (예: 005930.KQ)
시가총액: (예: 약 500조 원)
주식 가격 (최근 종가 기준): (예: 70,000원)
PER (주가수익비율): (예: 15.2)
PBR (주가순자산비율): (예: 1.8)
배당률: (예: 연 2.5%)
52주 최고/최저: (예: 최고 85,000원 / 최저 58,000원)
3. 최근 재무제표 (YYYY년 기준)
3.1. 손익계산서 (Income Statement)
항목	금액 (단위: 억 원)	증감률 (%)
매출액	300,000	+10.5%
영업이익	45,000	+8.2%
당기순이익	35,000	+12.3%
3.2. 재무상태표 (Balance Sheet)
항목	금액 (단위: 억 원)	증감률 (%)
자산총계	1,200,000	+5.0%
부채총계	500,000	-2.5%
자본총계	700,000	+7.8%
3.3. 현금흐름표 (Cash Flow Statement)
항목	금액 (단위: 억 원)	증감률 (%)
영업활동 현금흐름	55,000	+11.4%
투자활동 현금흐름	-30,000	-5.2%
재무활동 현금흐름	-10,000	+3.1%
4. 주요 재무지표
ROE (자기자본이익률): (예: 12.5%)
ROA (총자산이익률): (예: 6.8%)
부채비율: (예: 71.4%)
EPS (주당순이익): (예: 4,200원)
BPS (주당순자산): (예: 38,500원)
5. 산업 분석 및 경쟁사 비교
산업 내 위치: (예: 반도체 산업 글로벌 2위)
주요 경쟁사: (예: TSMC, 인텔, SK하이닉스)
시장 점유율: (예: 18.3%)
최근 산업 동향: (예: 반도체 수요 증가, AI 반도체 시장 확대)
6. 최신 뉴스 및 기업 동향
최근 이슈: (예: 2024년 1분기 반도체 생산 확대 발표)
M&A 및 신규 투자: (예: 미국 내 반도체 공장 건설)
정부 규제 영향: (예: 미중 무역 분쟁으로 인한 반도체 수출 규제)
CEO 코멘트: (예: “올해 AI 반도체 시장에서 선도적 위치를 차지할 것입니다.”)
7. 기업 전망 및 AI 분석
향후 1년 예상 성장률: (예: +7.2%)
주가 예측 (AI 모델 기반): (예: 1년 후 예상 주가 75,000원)
리스크 요인: (예: 원자재 가격 변동, 글로벌 경제 불확실성)
기회 요인: (예: AI 반도체 및 클라우드 서버 수요 증가)


1, 2, 3, 4 번정보에 대한 정보의 연도를 꼭 적어줘야함.
또한 추후에 별도로 학습데이터가 생길 경우 더 자세한 서비스 예정이라는 멘트를 남겨줘.
"""


# 오늘의 시황정보 프롬프트 템플릿
MARKET_INFO_PROMPT = """
현재 미지원되는 기능이라고 사용자에게 알려줘.
"""