# Ragas 학습 노트

`test/ch05_ragas.ipynb`에서 실습한 내용 정리. PDF 문서로부터 RAG 평가용 합성 테스트셋을 만드는 흐름.

## 전체 흐름

1. PDF 로드 → LangChain `Document` 리스트
2. `TestsetGenerator` 생성 (생성용 LLM + 임베딩 모델)
3. `transforms` 정의 — 각 문서에서 요약/임베딩/개체명 추출
4. `query_distribution` 정의 — 어떤 종류의 질문을 어떤 비율로 만들지
5. `generate_with_langchain_docs()`로 테스트셋 생성
6. `to_pandas()` → CSV 저장

## 1. 문서 준비

```python
from langchain_community.document_loaders import PDFPlumberLoader

loader = PDFPlumberLoader("../data/SPRI_AI_Brief_2023년12월호_F.pdf")
docs = loader.load()
docs = docs[3:-1]   # 표지·목차·마지막 페이지 제외 → 19페이지

for doc in docs:
    doc.metadata["filename"] = doc.metadata["source"]
```

- 페이지 단위로 `Document`가 만들어지고, 이 단위가 그대로 테스트셋의 문맥(`reference_contexts`)이 된다.

## 2. TestsetGenerator

```python
from ragas.testset import TestsetGenerator
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

generator_llm = ChatOpenAI(model="gpt-4o-mini")                      # 데이터셋 생성 모델
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")        # 임베딩 모델

generator = TestsetGenerator.from_langchain(
    llm=generator_llm,
    embedding_model=embeddings,
)
```

- Ragas 0.4.x에서는 예전 버전처럼 `KeyphraseExtractor`와 `critic_llm`을 직접 구성하지 않는다.
- `generator.llm`, `generator.embedding_model`로 Ragas용으로 감싼 모델을 꺼내 transform/synthesizer에 재사용한다.

## 3. Transforms — 문서에서 정보 추출

페이지를 분할하지 않고, 각 `Document`에 필요한 정보만 추출한다.

```python
from ragas.testset.transforms import SummaryExtractor, EmbeddingExtractor
from ragas.testset.transforms.extractors import NERExtractor

transforms = [
    SummaryExtractor(llm=generator.llm),
    EmbeddingExtractor(
        embedding_model=generator.embedding_model,
        property_name="summary_embedding",
        embed_property_name="summary",
    ),
    NERExtractor(llm=generator.llm),
]
```

| Transform | 역할 |
|---|---|
| `SummaryExtractor` | LLM으로 문서 내용 요약 → `summary` 속성 |
| `EmbeddingExtractor` | 지정한 텍스트 속성(`summary`)을 벡터로 변환 → `summary_embedding` 속성 |
| `NERExtractor` | 개체명 인식(Named Entity Recognition)으로 사람·기관·장소 등 주요 개체 추출 |

## 4. Query Distribution — 질문 종류와 비율

```python
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer

query_distribution = [
    (SingleHopSpecificQuerySynthesizer(llm=generator.llm), 1.0),
]
```

- `(synthesizer, 비율)` 튜플의 리스트. 비율 합은 1.0.
- `SingleHopSpecificQuerySynthesizer`: 한 페이지(하나의 문맥)의 개체 정보를 바탕으로, 그 문맥만으로 답할 수 있는 구체적인 질문과 참조 정답을 생성한다.

## 5. 테스트셋 생성

```python
testset = generator.generate_with_langchain_docs(
    documents=docs,                         # PDF 페이지 Document 리스트
    testset_size=10,                        # 생성할 질문 수
    transforms=transforms,                  # 요약, 임베딩, 개체명 인식
    query_distribution=query_distribution,  # 질문 생성 방식
    raise_exceptions=True,                  # 생성 중 오류 시 예외 발생
)
```

## 6. 결과 확인 및 저장

```python
test_df = testset.to_pandas()

if test_df.empty:
    raise RuntimeError("샘플이 0개입니다. 개체 추출 결과와 페르소나 매칭을 확인하세요.")

test_df.to_csv("../data/ragas_synthetic_dataset.csv", index=False, encoding="utf-8-sig")
```

- 샘플이 0개면 개체 추출 결과나 페르소나 매칭이 실패한 경우가 많다.
- `utf-8-sig`로 저장해야 엑셀에서 한글이 깨지지 않는다.

### 생성된 컬럼

| 컬럼 | 의미 |
|---|---|
| `user_input` | 생성된 질문 |
| `reference_contexts` | 질문의 근거가 된 문맥(페이지 본문) |
| `reference` | 참조 정답 |
| `persona_name` | 질문자 페르소나 (예: AI Policy Analyst, AI Safety Researcher, AI Ethics Officer) |
| `query_style` | 질문 문체 (`PERFECT_GRAMMAR`, `POOR_GRAMMAR`, `WEB_SEARCH_LIKE`, `MISSPELLED`) |
| `query_length` | 질문 길이 (`SHORT`, `MEDIUM`, `LONG`) |
| `synthesizer_name` | 사용된 synthesizer |

- Ragas가 페르소나·문체·길이를 섞어서 실제 사용자처럼 다양한 질문을 만든다.
- 문서는 한국어지만 질문·정답이 영어로 생성되거나 한영 혼용되는 경우가 있다.

## 겪은 문제와 해결

### `No module named 'langchain_community.chat_models.vertexai'`

- 원인: ragas 0.4.3이 `langchain_community.chat_models.vertexai`를 import하는데, `langchain-community` 0.4.x에서 이 모듈이 제거됨.
- 해결: `pyproject.toml`에서 langchain 계열을 0.3.x로 고정 후 `uv lock --upgrade` → `uv sync`.
  ```toml
  "langchain>=0.3,<1",
  "langchain-chroma>=0.2,<1",
  "langchain-community>=0.3,<0.4",
  "langchain-core>=0.3,<1",
  "langchain-openai>=0.3,<1",
  "langchain-text-splitters>=0.3,<1",
  ```
- `uv sync` 중 "액세스가 거부되었습니다"가 나면 노트북 커널을 종료하고 다시 실행.
- uv로 관리하는 `.venv`에 `pip install`을 섞으면 버전이 꼬인다. `uv add`만 사용.

### `'temperature' does not support 0.01 with this model`

- 원인: Ragas는 요청마다 temperature를 0.01로 덮어쓰는데, 추론 모델(gpt-5 계열 등)은 기본값 1만 허용.
- 해결 1: 추론 모델이 아닌 모델 사용 (`gpt-4o-mini`) — 노트북에서 채택한 방법.
- 해결 2: `from_langchain()` 대신 직접 감싸고 temperature 덮어쓰기를 끈다.
  ```python
  from ragas.llms import LangchainLLMWrapper
  from ragas.embeddings import LangchainEmbeddingsWrapper

  generator = TestsetGenerator(
      llm=LangchainLLMWrapper(generator_llm, bypass_temperature=True),
      embedding_model=LangchainEmbeddingsWrapper(embeddings),
  )
  ```
