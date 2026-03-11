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

## Step 1: PR/Issue 수집

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

## Step 2: 필터링

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

## Step 3: 상세 내용 확인

필터링된 PR/Issue에 대해 `gh api`로 본문을 확인하고 요약:

```bash
gh api repos/{owner}/{repo}/pulls/{number} --jq '.body' | head -80
gh api repos/{owner}/{repo}/issues/{number} --jq '.body' | head -80
```

## Step 4: 포스트 작성

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

**테이블 형식:**
```markdown
| PR | Repo | Status | Summary |
|----|------|--------|---------|
```

**Summary 작성 규칙:**
- (1) 문제 배경 (2) 변경 내용 (3) 효과/벤치마크
- Korean 설명, English 기술 용어
- 간결하고 기술적으로 (팀원 수준, 기본 개념 설명 불필요)
- 벤치마크 수치가 있으면 반드시 포함

## Step 5: Blog Post Sources 확인

CLAUDE.md의 Blog Post Sources에 나열된 블로그에서 당일 게시된 글이 있는지 확인.
당일 게시된 글이 있으면 `## Blog & Article Highlights` 섹션에 추가.

## Step 6: 이전 포스트 참조

이전 daily post가 존재하면 읽어서 Issue Tracking 연속성을 유지한다.
```bash
ls _posts/*-vllm-sglang-daily.md | tail -3
```
