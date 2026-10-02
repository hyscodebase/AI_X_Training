# AI+X 응용 AI 실습 포트폴리오

스마트폰 행동 인식, 미세먼지 분석, RAG 기초를 프로젝트별로 정리한 학습 저장소입니다. 데이터와 학생용 실습 노트북이 함께 있으며, 일부 셀은 학습자가 직접 완성하는 템플릿입니다.

## 프로젝트 지도

| 프로젝트 | 경로 | 수록 자료 |
|---|---|---|
| 스마트폰 행동 인식 | [human-activity-analysis](AI-X-Portfolio/human-activity-analysis) | 센서 EDA·기본 분류·단계별 분류 노트북, 학습·테스트 CSV와 특징 설명 |
| 미세먼지 예측 | [fine-dust-analysis](AI-X-Portfolio/fine-dust-analysis) | EDA·전처리·회귀 모델링 노트북, 2021·2022 대기·기상 데이터 |
| RAG 기초 | [file-management-rag-agent](AI-X-Portfolio/file-management-rag-agent) | 코사인 유사도·청킹 노트북과 PDF 예제 |
| 자율 프로젝트 | [autonomous-project](AI-X-Portfolio/autonomous-project) | 프로젝트 공간과 임시 파일; 분석 노트북은 아직 빈 파일 |

## 실행

Python 3와 독립된 가상환경에서 시작합니다. 저장소 루트에서 실행하세요.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install jupyterlab numpy pandas matplotlib seaborn scikit-learn joblib openpyxl
jupyter lab
```

Windows에서는 가상환경 활성화 명령을 `.venv\Scripts\activate`로 바꿉니다.

프로젝트의 `notebooks/` 폴더에서 EDA → 전처리 또는 기본 모델링 → 확장 모델링 순서로 공부합니다. 노트북의 데이터 경로를 같은 프로젝트의 `data/` 폴더에 맞추고, 빈 셀에 분석 코드를 작성하세요.

## RAG 실습

RAG 노트북은 `file-management-rag-agent/documents/`에 있습니다. 청킹 실습에서 사용하는 `langchain-text-splitters`, `langchain-community`, `pypdf`, `tiktoken` 등은 해당 실습 환경에 추가 설치하세요. PDF 로딩 셀의 경로를 포함된 `src/MIT.pdf` 위치에 맞춥니다.

## 현재 범위

이 저장소는 학습 자료와 실습 작업 공간을 제공합니다. 노트북의 미션 설명, 학생용 빈 셀, 완성된 분석 코드를 구분해서 읽으세요. 배포된 웹 서비스나 완성된 모든 모델·논문·PRD가 포함된 것으로 해석하지 않습니다.

수업 및 제공 실습 자료의 출처는 각 노트북에 표기된 정보를 따릅니다.
