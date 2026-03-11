# Project Instructions

## Blog Structure
- Jekyll blog at vrvrv.github.io, theme: no-style-please
- Posts in `_posts/` with format `YYYY-MM-DD-slug.md` (no underscore prefix for published posts; underscore-prefixed are examples/drafts)
- Front matter: `layout: post`, optional `category`, `title`
- Config: `_config.yml`, permalink: `/:slug.html`, lowercase_titles: true

## Monthly Section Overview Posts
- 각 섹션(Speculative Decoding, vLLM IR, KV Cache/Connector, PP/Large Scale Serving 등)별 주요 동향을 1달 주기로 별도 포스트로 작성
- Filename: `YYYY-MM-DD-{section-slug}-overview.md` (예: `2026-03-10-speculative-decoding-overview.md`)
- Category: `llm-serving`
- Daily post의 각 섹션 헤더에 해당 섹션의 최신 월간 동향 포스트 링크를 포함

## Daily vLLM/SGLang Posts
- Category: `llm-serving`
- Filename: `YYYY-MM-DD-vllm-sglang-daily.md`
- Topics tracked: Speculative Decoding, vLLM IR, Distributed KV Cache/KV Connector, Pipeline Parallel, Large Scale Serving
- 각 섹션 헤더에 해당 섹션의 최신 월간 overview 포스트 링크 포함 (예: `[→ 동향 보기](/speculative-decoding-overview.html)`)
- GitHub repos: vllm-project/vllm, sgl-project/sglang
- **PR/Issue 선정 기준**: 해당 날짜에 새로 올라온(opened) PR, 새로 머지된 PR, 새 RFC/Issue만 포함. 오래된 open PR은 제외.
- 관심사(User Interests)와 일치하고 중요한 내용만 선별 — routine cleanup, 단순 feature request, dependency bump 등은 제외
- GPU/특정 vendor에 specific한 버그 수정은 중요도 낮음 (NPU 회사 재직 중이므로 하드웨어 중립적/아키텍처 레벨 내용 우선)
- Issue tracking: reference previous daily posts for PR/issue continuity
- Language: Korean descriptions, English technical terms
- Summary style: 각 섹션(테이블 아래)에 8~12줄의 심층 분석을 포함. 해당 영역의 vLLM/SGLang 내부 구조, PR 간 맥락, 개발 트렌드를 서술하여 블로그만 읽어도 내부 동작을 이해할 수 있게 함. PR 개별 요약은 테이블 내 1~2줄.
- No blockquote concept explanations per section (too verbose)
- Blog & Article Highlights: only include posts published on the same day
- No GitHub Actions workflow

## User Interests (= Team Focus)
LLM 추론/서빙 인프라 엔지니어링 팀 소속. 블로그 독자도 같은 팀원들.
- **LLM Serving Frameworks**: vLLM, SGLang, llm-d 내부 구조 및 최적화
- **Speculative Decoding**: Eagle, MTP, DFlash, draft model 최적화, dynamic speculation
- **KV Cache Management**: distributed KV cache, KV connector (NIXL, Mooncake, FlexKV), prefix caching, cache eviction
- **P/D Disaggregation**: prefill/decode 분리 아키텍처, PD-Multiplexing
- **Distributed Inference**: tensor parallelism, pipeline parallelism, expert parallelism, multi-node serving
- **Large-scale MoE Serving**: DeepSeek, expert load balancing (EPLB), DeepEP
- **Performance Optimization**: TTFT, ITL, throughput, tail latency, memory utilization, PCIe/RDMA overhead
- **Kernel & Compiler**: FlashAttention, FlashInfer, Triton, CUDA graph, kernel fusion
- **Hardware Platforms**: GPU (NVIDIA Hopper/Blackwell), NPU 지원

## Blog Post Sources (for Blog & Article Highlights)
Daily post 작성 시 아래 블로그들에서 당일 게시된 글이 있는지 확인:
- https://pytorch.org/blog/ — PyTorch 공식 블로그
- https://lmsys.org/blog/ — LMSYS/SGLang 블로그
- http://blog.ezyang.com/ — ezyang's blog (PyTorch internals, distributed systems)
- https://vllm.ai/blog — vLLM 공식 블로그 (간헐적 404, 접속 가능 시 확인)
- https://developer.nvidia.com/blog — NVIDIA Developer Blog (CUDA, TensorRT-LLM 관련)
- https://www.anyscale.com/blog — Anyscale/Ray blog (vLLM 관련 포스트 종종 게재)

## RSS Feeds
- https://lmsys.org/rss.xml — LMSYS/SGLang blog
- https://feeds.feedburner.com/ezyang — ezyang's blog
- https://pytorch.org/feed/ — PyTorch official blog
- https://vllm.ai/feed.xml — currently 404, skip
