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

## Step 1: 기간 결정

- 지난 1개월 (오늘 기준 30일 전 ~ 오늘)
- 이전 monthly overview 포스트가 존재하면 그 날짜 이후부터

```bash
# 이전 overview 확인
ls _posts/*-{section-slug}-overview.md 2>/dev/null | tail -1
```

## Step 2: PR/Issue 수집

해당 기간에 생성 또는 머지된 PR/Issue를 수집:

```bash
# 기간 범위로 검색
gh search prs --repo vllm-project/vllm --created {START}..{END} --limit 100 --json number,title,state
gh search prs --repo vllm-project/vllm --merged {START}..{END} --limit 100 --json number,title,state
gh search prs --repo sgl-project/sglang --created {START}..{END} --limit 100 --json number,title,state
gh search prs --repo sgl-project/sglang --merged {START}..{END} --limit 100 --json number,title,state
```

## Step 3: 섹션별 필터링

CLAUDE.md의 User Interests와 각 섹션의 키워드에 따라 필터링.
제외 기준은 daily-post와 동일 (GPU-specific 버그, routine cleanup 등).

## Step 4: 상세 내용 확인

필터링된 PR/Issue에 대해 `gh api`로 본문 확인:
```bash
gh api repos/{owner}/{repo}/pulls/{number} --jq '.body' | head -100
```

## Step 5: 포스트 작성

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
| Date | PR | Repo | Summary |
|------|-----|------|---------|
```

### 3. Active RFCs & Proposals
아직 open인 RFC, 설계 제안 등. 향후 방향을 보여주는 항목들.

### 4. Open Issues & Known Problems
주요 버그, 미해결 이슈. 프로덕션 영향도 높은 것 위주.

### 5. Trend & Outlook
이번 달 트렌드를 2~3문장으로 정리. 다음 달 예상되는 방향.

**작성 톤:**
- 팀원 수준 (LLM serving infra 엔지니어)
- Korean 설명, English 기술 용어
- 벤치마크 수치 반드시 포함
- GPU-specific 내용보다 아키텍처 레벨 변화에 집중

## Step 6: Daily post 링크 업데이트

새 overview 포스트 작성 후, 가장 최근 daily post의 해당 섹션 헤더에 링크를 추가한다:

```markdown
## Speculative Decoding [→ 동향](/{section-slug}-overview.html)
```

## `all` 모드

`$ARGUMENTS`가 `all`이면 모든 섹션에 대해 개별 포스트를 작성한다.
각 섹션별로 별도 파일을 생성하되, Agent를 활용하여 병렬로 처리한다.
