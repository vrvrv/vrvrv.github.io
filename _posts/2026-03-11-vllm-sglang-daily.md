---
layout: post
category: llm-serving
title: "vLLM/SGLang Daily - 2026-03-11"
---

## Highlights

**vLLM 0.17.0 안정화** — DFlash spec decode 통합, NIXL Docker 배포 문제 해결, hybrid SSM-FA 모델의 KV connector 지원 RFC가 올라옴.

**SGLang — DeepSeek v3.2 멀티노드 불안정** — 2노드 H200 환경에서 DeepEP internode dispatch 실패 및 런타임 stall이 동시 보고. Pod Attention, HiSparse 등 성능 최적화 PR도 활발.

---

## Speculative Decoding

| PR | Repo | Status | Summary |
|----|------|--------|---------|
| [#36767](https://github.com/vllm-project/vllm/pull/36767) | vLLM | Open | **DFlash 통합** — spec decode 파이프라인에 DFlash attention metadata/slot mapping 처리 추가. Qwen3-8B batch=32 기준 SGLang 대비 ~8% 높은 처리량 (2741 vs 2540 tok/s). DFlash 시리즈: [#32206](https://github.com/vllm-project/vllm/pull/32206) → [#34014](https://github.com/vllm-project/vllm/pull/34014) → 본 PR |
| [#20355](https://github.com/sgl-project/sglang/pull/20355) | SGLang | Open | MTP 모델에서 piecewise CUDA graph 빌드 시 레이어 부재 오류 수정 |

---

## Distributed KV Cache / KV Connector

| PR | Status | Summary |
|----|--------|---------|
| [#36735](https://github.com/vllm-project/vllm/pull/36735) | Open | NIXL runtime wheel Docker 빌드 문제 해결 — 프로덕션 배포 blocker |
| [#36780](https://github.com/vllm-project/vllm/issues/36780) | RFC | **NixlConnector hybrid SSM-FA 지원** — Transformer(K/V)와 Mamba(Conv/SSM state) 레이어의 서로 다른 cache layout을 동일 메모리 영역 위 dual descriptor view로 처리하는 설계 제안 |

---

## Pipeline Parallel / Large Scale Serving

| PR/Issue | Repo | Status | Summary |
|----------|------|--------|---------|
| [#20336](https://github.com/sgl-project/sglang/issues/20336) | SGLang | Bug | **DeepSeek v3.2 멀티노드 실패** — 2노드 8xH200에서 DeepEP internode dispatch 시 illegal memory access. 멀티노드 MoE 서빙의 근본적 안정성 문제 |
| [#20315](https://github.com/sgl-project/sglang/issues/20315) | SGLang | Bug | **DeepSeek v3.2 런타임 stall** — TP=16, DP=16에서 수 시간 정상 운영 후 전 rank 동시 멈춤. NSA 백엔드, py-spy dump → watchdog timeout(300s) |
| [#20346](https://github.com/sgl-project/sglang/pull/20346) | SGLang | Open | **Pod Attention** — Mixed Chunk Prefill에서 prefill/decode 어텐션 커널을 fuse하여 compute + memory bandwidth 동시 활용. [논문](https://arxiv.org/pdf/2410.18038) |
| [#20343](https://github.com/sgl-project/sglang/pull/20343) | SGLang | Open | **HiSparse** — NSA 모델(DeepSeek-V3.2, GLM-5)에서 디코딩 중 미사용 KV cache를 CPU offload. GPU 메모리 확보 → 배치↑ → throughput↑ |
| [#20349](https://github.com/sgl-project/sglang/pull/20349) | SGLang | Open | PD Multiplexing에서 attn/MoE 레이어 통신 그룹 초기화 미분리 버그 수정 |
| [#20332](https://github.com/sgl-project/sglang/issues/20332) | SGLang | RFC | MoE 모델 TP 기반 병렬 로딩으로 시작 시간 단축 제안 |

### Note
DeepSeek v3.2 멀티노드 이슈([#20336](https://github.com/sgl-project/sglang/issues/20336), [#20315](https://github.com/sgl-project/sglang/issues/20315))가 핵심. EP + P/D disagg 조합의 edge case가 드러나는 단계. 특히 #20315는 장시간 후 발생하여 재현/디버깅 난이도 높음.

---

## Notable Issues & Bugs

- **vLLM [#36771](https://github.com/vllm-project/vllm/issues/36771)**: LMCache가 vLLM 0.17.0과 호환 불가. PyTorch 2.10 C++ 심볼 변경 → import 크래시
- **vLLM [#36755](https://github.com/vllm-project/vllm/issues/36755)**: preemption 시 캐시 히트 카운터 미리셋 → Prometheus counter 음수 크래시. async scheduling + KV connector 환경
- **SGLang [#20334](https://github.com/sgl-project/sglang/issues/20334)**: CUDA Graph + MTP + page_size=64에서 MiMo-V2-Flash precision 저하

## Community & Ecosystem

- **vLLM [#36786](https://github.com/vllm-project/vllm/pull/36786)**: EPLB 지원 확장 — 인기 expert를 여러 GPU에 복제하여 부하 분산
- **SGLang [#20307](https://github.com/sgl-project/sglang/issues/20307)**: MoE Routing Simulator — expert load imbalance 사전 시뮬레이션/시각화 도구 제안

---

*Sources: [vLLM GitHub](https://github.com/vllm-project/vllm), [SGLang GitHub](https://github.com/sgl-project/sglang)*
