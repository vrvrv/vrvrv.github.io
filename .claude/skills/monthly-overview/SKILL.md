---
name: monthly-overview
description: vLLM/SGLang 월간 섹션별 동향 포스트 작성. 지난 1개월간의 주요 변화를 섹션별로 정리한다.
disable-model-invocation: true
user-invocable: true
argument-hint: "[section-name] (speculative-decoding | vllm-ir | kv-cache | pipeline-parallel | all)"
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, WebFetch, Agent
---

# Monthly Section Overview Post

`$ARGUMENTS`로 지정된 섹션 또는 `all`로 전체 섹션의 월간 동향 포스트를 작성한다.

## 섹션 목록

| Section Slug | 섹션 이름 | 키워드 |
|-------------|----------|--------|
| `speculative-decoding` | Speculative Decoding | Eagle, MTP, DFlash, draft model, dynamic speculation, FSM |
| `vllm-ir` | vLLM IR / Model Runner V2 | IR, batch-invariant, fused op, kernel fusion, Model Runner |
| `kv-cache` | Distributed KV Cache / KV Connector | NIXL, Mooncake, FlexKV, LMCache, KVCacheSpec, P/D disagg, prefix caching, cache eviction |
| `pipeline-parallel` | Pipeline Parallel / Large Scale Serving | PP, TP, EP, MoE, DeepEP, EPLB, scheduling, Pod Attention |

---

## Phase 1: 수집 & 필터링

### Step 1: 기간 결정

- 지난 1개월 (오늘 기준 30일 전 ~ 오늘)
- 이전 monthly overview 포스트가 존재하면 그 날짜 이후부터

```bash
# 이전 overview 확인
ls _posts/*-{section-slug}-overview.md 2>/dev/null | tail -1
```

### Step 2: PR/Issue 수집

해당 기간에 생성 또는 머지된 PR/Issue를 수집:

```bash
# 기간 범위로 검색
gh search prs --repo vllm-project/vllm --created {START}..{END} --limit 100 --json number,title,state
gh search prs --repo vllm-project/vllm --merged {START}..{END} --limit 100 --json number,title,state
gh search prs --repo sgl-project/sglang --created {START}..{END} --limit 100 --json number,title,state
gh search prs --repo sgl-project/sglang --merged {START}..{END} --limit 100 --json number,title,state
```

### Step 3: 섹션별 필터링

CLAUDE.md의 User Interests와 각 섹션의 키워드에 따라 필터링.
제외 기준은 daily-post와 동일 (GPU-specific 버그, routine cleanup 등).

### Step 4: 상세 내용 확인

필터링된 PR/Issue에 대해 `gh api`로 **본문 전체**를 확인한다:

```bash
gh api repos/{owner}/{repo}/pulls/{number} --jq '.body'
gh api repos/{owner}/{repo}/pulls/{number}/files --jq '.[].filename'
gh api repos/{owner}/{repo}/issues/{number} --jq '.body'
```

---

## Phase 2: 포스트 작성

### Step 5: 포스트 작성

파일 경로: `_posts/{TODAY}-{section-slug}-overview.md`

**Front matter:**
```yaml
---
layout: post
category: llm-serving
title: "{섹션 이름} Overview - {YYYY-MM}"
---
```

**구조:**

### 1. Summary (3~5줄)
이번 달의 핵심 변화를 요약. 방향성과 트렌드 위주.

### 2. Key Merged PRs
머지된 PR 중 중요한 것들을 표로 정리. 날짜순.

```markdown
| Date | PR | Repo | 요약 |
|------|-----|------|------|
```

**테이블 하단에 8~12줄의 통합 분석** — 월간 머지된 PR들을 관통하는 기술적 맥락을 서술.
단순 PR 나열이 아니라, 이 PR들이 전체적으로 어떤 아키텍처 변화를 반영하는지,
vLLM/SGLang 내부 구조의 어떤 부분이 어떻게 진화하고 있는지를 설명한다.

### 3. Active RFCs & Proposals
아직 open인 RFC, 설계 제안 등. 향후 방향을 보여주는 항목들.

**테이블 하단에 8~12줄의 통합 분석** — RFC들이 가리키는 설계 방향, 아키텍처적 의미,
기존 구조와의 관계를 서술한다.

### 4. Open Issues & Known Problems
주요 버그, 미해결 이슈. 프로덕션 영향도 높은 것 위주.

### 5. Trend & Outlook (8~12줄)
이번 달 트렌드를 종합적으로 정리하고 다음 달 예상되는 방향을 서술한다.
단순 2~3문장이 아니라, 월간 변화의 큰 흐름을 읽을 수 있는 실질적인 분석을 제공한다.

**작성 톤:**
- 팀원 수준 (LLM serving infra 엔지니어)
- Korean 설명, English 기술 용어
- 벤치마크 수치 반드시 포함
- GPU-specific 내용보다 아키텍처 레벨 변화에 집중
- 블로그만 읽어도 vLLM/SGLang 내부 동작을 이해할 수 있어야 함

### Step 6: Daily post 링크 업데이트

새 overview 포스트 작성 후, 가장 최근 daily post의 해당 섹션 헤더에 링크를 추가한다:

```markdown
## Speculative Decoding [→ 동향](/{section-slug}-overview.html)
```

---

## Phase 3: 반복 검토 루프 (최소 5회)

**포스트 작성 후 반드시 아래 루프를 최소 5회 반복한다. 각 iteration마다 commit을 남긴다.**

### Iteration 구조

```
for i in 1..5+ :
    1. 작성/수정 → 파일에 반영
    2. git add && git commit -m "monthly-overview {SECTION}: iteration {i} - {변경 요약}"
    3. 자체 검토 (아래 체크리스트)
    4. 문제 발견 → 수정 후 다음 iteration
    5. 5회차 이후: 모든 체크리스트 통과 시 종료
```

### 검토 체크리스트

#### A. 섹션 분석 품질
- [ ] Key Merged PRs 하단에 8~12줄 통합 분석이 있는가?
- [ ] Active RFCs 하단에 8~12줄 통합 분석이 있는가?
- [ ] Trend & Outlook이 8~12줄의 실질적 분석인가?
- [ ] vLLM/SGLang 내부 구조 설명이 포함되어 있는가?
- [ ] 월간 변화의 큰 흐름이 읽히는가?

#### B. 기술적 정확성
- [ ] PR 본문과 대조 시 사실 오류가 없는가?
- [ ] 벤치마크 수치가 정확하게 인용되었는가?
- [ ] 모듈/클래스/함수 이름이 정확한가?

#### C. 형식 & 완성도
- [ ] Summary가 3~5줄로 핵심 변화를 정확히 요약하는가?
- [ ] 테이블 형식이 일관적인가?
- [ ] Daily post 링크가 업데이트되었는가?

### Iteration별 집중 포인트

| Iteration | 집중 영역 | 설명 |
|-----------|----------|------|
| 1 | 초안 작성 | 전체 구조 + 테이블 + 분석 초안 |
| 2 | 분석 깊이 강화 | 각 섹션 분석이 8~12줄인지 확인. 내부 구조 설명 보강. |
| 3 | 기술적 정확성 | PR 본문 재확인. 수치 대조. 모듈명 검증. |
| 4 | 트렌드 연결 | 섹션 내 PR들 간 맥락 연결. Trend & Outlook 보강. |
| 5 | 최종 다듬기 | 문장 다듬기. 중복 제거. 체크리스트 최종 통과. |

**5회 이후에도 체크리스트 미통과 항목이 있으면 통과할 때까지 추가 iteration을 진행한다.**

### Commit Convention

```
monthly-overview {SECTION}: iteration {N} - {변경 요약}
```

---

## `all` 모드

`$ARGUMENTS`가 `all`이면 모든 섹션에 대해 개별 포스트를 작성한다.
각 섹션별로 별도 파일을 생성하되, Agent를 활용하여 병렬로 처리한다.
