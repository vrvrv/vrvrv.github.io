---
name: daily-post
description: vLLM/SGLang daily post 작성. 당일 새로 올라온 PR/Issue/RFC를 수집하여 블로그 포스트를 생성한다.
disable-model-invocation: true
user-invocable: true
argument-hint: "[YYYY-MM-DD] (기본값: 오늘)"
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, WebFetch, Agent
---

# Daily vLLM/SGLang Post

오늘 날짜 또는 `$ARGUMENTS`로 지정된 날짜 기준으로 vLLM/SGLang daily post를 작성한다.

## 날짜 결정

- `$ARGUMENTS`가 있으면 해당 날짜 사용
- 없으면 오늘 날짜 사용
- 형식: `YYYY-MM-DD`

---

## Phase 1: 수집 & 필터링

### Step 1: PR/Issue 수집

아래 명령으로 해당 날짜에 **새로 생성된** PR/Issue와 **해당 날짜에 머지된** PR을 수집:

```bash
# 새로 생성된 PR
gh search prs --repo vllm-project/vllm --created {DATE} --limit 50 --json number,title,state
gh search prs --repo sgl-project/sglang --created {DATE} --limit 50 --json number,title,state

# 새로 생성된 Issue
gh search issues --repo vllm-project/vllm --created {DATE} --limit 50 --json number,title,state
gh search issues --repo sgl-project/sglang --created {DATE} --limit 50 --json number,title,state

# 해당 날짜에 머지된 PR
gh search prs --repo vllm-project/vllm --merged {DATE} --limit 50 --json number,title,state
gh search prs --repo sgl-project/sglang --merged {DATE} --limit 50 --json number,title,state
```

### Step 2: 필터링

CLAUDE.md의 **User Interests**와 **PR/Issue 선정 기준**에 따라 필터링:

**포함:**
- Speculative Decoding (Eagle, MTP, DFlash, dynamic speculation)
- vLLM IR / Model Runner V2
- Distributed KV Cache / KV Connector (NIXL, Mooncake, FlexKV, prefix caching)
- P/D Disaggregation, PD-Multiplexing
- Pipeline Parallel, Tensor Parallel, Expert Parallelism
- Large-scale MoE Serving (DeepSeek, EPLB, DeepEP)
- Performance optimization (TTFT, ITL, throughput, scheduling)
- Kernel fusion, attention optimization
- OOT/NPU/하드웨어 중립적 아키텍처 변경

**제외:**
- GPU/특정 vendor에 specific한 버그 수정 (ROCm, 특정 GPU 모델 이슈 등)
- Routine cleanup, dead code 제거
- 단순 feature request (기술적 내용 없음)
- Dependency bump, doc fix, CI fix
- Diffusion model 관련 (NPU diffusion은 예외)

### Step 3: 상세 내용 확인

필터링된 PR/Issue에 대해 `gh api`로 **본문 전체**를 확인한다. 분석 품질의 핵심은 원문 이해에 있으므로 충분히 읽는다.

```bash
# PR 본문 + 변경 파일
gh api repos/{owner}/{repo}/pulls/{number} --jq '.body'
gh api repos/{owner}/{repo}/pulls/{number}/files --jq '.[].filename'

# Issue 본문
gh api repos/{owner}/{repo}/issues/{number} --jq '.body'

# 핵심 설계 논의 코멘트 (필요 시)
gh api repos/{owner}/{repo}/pulls/{number}/comments --jq '.[].body' | head -200
```

**분석 시 반드시 확인할 것:**
- PR description의 motivation/background 섹션
- 변경된 파일 목록에서 어떤 모듈/레이어가 영향받는지
- 벤치마크/성능 수치 (있으면 정확히 기록)
- 관련 이전 PR/RFC 참조 (시리즈물 추적)
- 리뷰어 코멘트의 핵심 설계 논의 포인트

---

## Phase 2: 초안 작성

### Step 4: 포스트 작성

파일 경로: `_posts/{DATE}-vllm-sglang-daily.md`

**Front matter:**
```yaml
---
layout: post
category: llm-serving
title: "vLLM/SGLang Daily - {DATE}"
---
```

**구조:**

1. **Highlights** — 2~3줄로 당일 핵심 요약
2. **Speculative Decoding** — 해당 날짜 항목이 있을 때만
3. **vLLM IR / Model Runner V2** — 해당 날짜 항목이 있을 때만
4. **Distributed KV Cache / KV Connector** — 해당 날짜 항목이 있을 때만
5. **Pipeline Parallel / Large Scale Serving** — 해당 날짜 항목이 있을 때만
6. **Notable Issues & Bugs** — 중요 버그만
7. **Community & Ecosystem** — 생태계 관련

각 섹션이 비어 있으면 해당 섹션은 생략한다.

**각 섹션 헤더에 최신 월간 overview 링크 포함:**
```markdown
## Speculative Decoding [→ 동향](/speculative-decoding-overview.html)
```
(해당 overview 포스트가 존재할 때만)

### ⭐ 섹션별 심층 분석 작성 규칙 (CRITICAL)

**각 섹션은 (1) PR 테이블 + (2) 섹션 분석으로 구성한다.**

#### PR 테이블

```markdown
| PR | Repo | Status | 요약 |
|----|------|--------|------|
| [#36767](link) | vLLM | Open | DFlash spec decode 통합 — batch=32 기준 SGLang 대비 ~8% 처리량 향상 |
```

- 각 PR 요약은 1~2줄. (1) 무엇을 하는 PR인지 (2) 벤치마크 수치 포함
- Korean 설명, English 기술 용어

#### 섹션 분석 (8~12줄) — 핵심

**테이블 아래에 섹션 전체를 관통하는 분석을 8~12줄로 작성한다.**

이 분석의 목적: 블로그를 읽는 것만으로 해당 영역의 vLLM/SGLang 내부 동작과 개발 추세를 이해할 수 있게 하는 것.

**분석에 포함할 내용:**

1. **내부 구조 설명** — 이 섹션의 PR들이 vLLM/SGLang 아키텍처에서 어디에 위치하는지. 관련 핵심 모듈/클래스/파이프라인을 자연스럽게 설명.
2. **기술적 맥락** — 왜 이런 변경들이 지금 일어나고 있는지. 어떤 아키텍처적 제약이나 기술적 부채가 이 작업들을 촉발했는지.
3. **PR 간 연결** — 당일 PR들 사이의 관계, 또는 이전 daily post에서 추적하던 시리즈와의 연결.
4. **개발 방향성** — 이 변경들이 가리키는 전체적인 개발 트렌드.

**예시:**

```markdown
## Speculative Decoding

| PR | Repo | Status | 요약 |
|----|------|--------|------|
| [#36767](link) | vLLM | Open | DFlash spec decode 통합 — Qwen3-8B batch=32 기준 2741 tok/s |
| [#20355](link) | SGLang | Open | MTP CUDA graph 빌드 시 레이어 부재 오류 수정 |

vLLM의 speculative decoding 파이프라인은 `SpecDecodeWorker`가 draft 모델과 target 모델의
실행을 조율하는 구조다. 기존에는 attention backend가 PagedAttention으로 하드코딩되어 있어
DFlash 같은 대안 backend를 spec decode 경로에서 사용하려면 metadata 빌더부터 slot mapping까지
전면 추상화가 필요했다. #36767은 이 시리즈(#32206 → #34014)의 최종 통합 단계로, DFlash
attention metadata를 spec decode 경로에 연결한다. batch=32에서 SGLang 대비 8% 높은 처리량은
DFlash의 prefill/decode attention 분리 최적화가 verification 단계에서도 유효함을 보여준다.

한편 SGLang 쪽 #20355는 MTP(Multi-Token Prediction) 모델에서 piecewise CUDA graph를 빌드할 때
draft 레이어 누락으로 크래시가 발생하는 문제를 수정한다. MTP는 draft 모델 없이 target 모델
자체의 추가 prediction head를 사용하는 방식이라, CUDA graph 캡처 시 레이어 구조가 표준
Transformer와 달라 이런 edge case가 발생한다. 양쪽 프레임워크 모두 spec decode의 attention
backend 다양화와 CUDA graph 호환성 확보에 집중하는 흐름이다.
```

**금지:**
- 테이블만 있고 섹션 분석이 없는 경우
- 3줄 이하의 피상적 분석
- 각 PR을 개별로 나열만 하고 섹션 단위 맥락을 연결하지 않는 경우
- 벤치마크 수치 누락 (PR에 있는 경우)

### Step 5: Blog Post Sources 확인

CLAUDE.md의 Blog Post Sources에 나열된 블로그에서 당일 게시된 글이 있는지 확인.
당일 게시된 글이 있으면 `## Blog & Article Highlights` 섹션에 추가.

### Step 6: 이전 포스트 참조

이전 daily post가 존재하면 읽어서 Issue Tracking 연속성을 유지한다.
```bash
ls _posts/*-vllm-sglang-daily.md | tail -3
```

---

## Phase 3: 반복 검토 루프 (최소 5회)

**포스트 작성 후 반드시 아래 루프를 최소 5회 반복한다. 각 iteration마다 commit을 남긴다.**

### Iteration 구조

```
for i in 1..5+ :
    1. 작성/수정 → 파일에 반영
    2. git add && git commit -m "daily-post {DATE}: iteration {i} - {변경 요약}"
    3. 자체 검토 (아래 체크리스트)
    4. 문제 발견 → 수정 후 다음 iteration
    5. 5회차 이후: 모든 체크리스트 통과 시 종료
```

### 검토 체크리스트

각 iteration에서 아래 항목을 **모두** 확인한다:

#### A. 섹션 분석 품질 (Section Analysis Quality)
- [ ] 모든 섹션에 8~12줄의 심층 분석이 있는가?
- [ ] 분석이 vLLM/SGLang 내부 구조(모듈, 클래스, 파이프라인)를 설명하는가?
- [ ] 당일 PR들 간의 관계, 이전 시리즈와의 연결이 서술되어 있는가?
- [ ] 개발 방향성/트렌드가 드러나는가?
- [ ] 블로그만 읽어도 해당 영역의 내부 동작을 이해할 수 있는가?

#### B. 기술적 정확성 (Technical Accuracy)
- [ ] PR 본문과 대조했을 때 사실 오류가 없는가?
- [ ] 벤치마크 수치가 정확하게 인용되었는가?
- [ ] 모듈/클래스/함수 이름이 정확한가?
- [ ] 관련 PR 시리즈가 올바르게 추적되었는가?

#### C. 형식 & 완성도 (Format & Completeness)
- [ ] 각 섹션이 테이블 + 분석 형식을 따르는가?
- [ ] Front matter, 섹션 구조가 올바른가?
- [ ] Highlights가 당일 핵심을 정확히 요약하는가?
- [ ] 이전 포스트와의 연속성이 유지되는가?

### Iteration별 집중 포인트

| Iteration | 집중 영역 | 설명 |
|-----------|----------|------|
| 1 | 초안 작성 | 수집된 PR/Issue 기반 전체 포스트 초안. 테이블 + 섹션 분석 초안. |
| 2 | 분석 깊이 강화 | 모든 섹션 분석이 8~12줄인지 확인. 내부 구조 설명 보강. PR 간 맥락 연결. |
| 3 | 기술적 정확성 | PR 본문 재확인. 수치 대조. 모듈명 검증. 사실 오류 수정. |
| 4 | 트렌드 & 맥락 | 섹션 간 개발 방향 연결. "왜 지금 이 변경이 일어나는가" 보강. |
| 5 | 최종 다듬기 | 문장 다듬기. 중복 제거. 흐름 개선. 체크리스트 최종 통과 확인. |

**5회 이후에도 체크리스트 미통과 항목이 있으면 통과할 때까지 추가 iteration을 진행한다.**

### Commit Convention

```
daily-post {DATE}: iteration {N} - {변경 요약}

예시:
daily-post 2026-03-12: iteration 1 - 초안 작성 (PR 8건, Issue 3건)
daily-post 2026-03-12: iteration 2 - 섹션별 심층 분석 보강
daily-post 2026-03-12: iteration 3 - 벤치마크 수치 대조 및 모듈명 수정
daily-post 2026-03-12: iteration 4 - 섹션 간 트렌드 연결 강화
daily-post 2026-03-12: iteration 5 - 최종 교정 및 문장 다듬기
```
