---
layout: post
category: llm-serving
title: "vLLM/SGLang Daily - 2026-03-10"
---

## Highlights

**vLLM — KVCacheSpec Registry RFC** — OOT(out-of-tree) 플랫폼이 KV cache 동작을 확장할 수 있도록 pluggable registry 메커니즘 제안. NPU 등 비-CUDA 하드웨어에서 alignment/padding을 커스터마이징할 수 있는 구조.

**SGLang — Decode Offload로 장문 성능 2배 향상** — DeepSeek-V3.2 장문 추론에서 디코딩 중 KV cache를 즉시 offload하는 전략으로 동시 처리량을 2배 이상 끌어올린 PR이 머지됨.

---

## Speculative Decoding

| PR | Repo | Status | Summary |
|----|------|--------|---------|
| [#36723](https://github.com/vllm-project/vllm/pull/36723) | vLLM | Merged | **DSV3.2 MTP Indexer 최적화** — DeepSeek-V3.2의 Multi-Token Prediction indexer 핸들링 개선 |
| [#36657](https://github.com/vllm-project/vllm/issues/36657) | vLLM | RFC | **Dynamic Speculation Length** — draft 모델 confidence threshold 기반 early exit. 불필요한 draft 토큰 연산을 줄여 spec decode 효율 향상 |
| [#20255](https://github.com/sgl-project/sglang/pull/20255) | SGLang | Open | **Eagle3 for Qwen3.5** — dense/MoE 모두 지원. mini-specforge로 학습한 draft model로 H100에서 평균 ~2x, 코드 생성에서 ~2.4x speedup 확인 |
| [#20266](https://github.com/sgl-project/sglang/pull/20266) | SGLang | Open | Eagle info 수집 시 GPU-CPU sync로 인한 deadlock 방지 — async 방식으로 변경 |
| [#19536](https://github.com/sgl-project/sglang/pull/19536) | SGLang | Merged | **NSA metadata 최적화 under MTP** — NSA(Sparse Attention) 백엔드의 metadata 처리를 MTP 환경에서 최적화 |

---

## vLLM IR / Model Runner V2

| PR | Status | Summary |
|----|--------|---------|
| [#36722](https://github.com/vllm-project/vllm/pull/36722) | Open | **IR 2/N** — batch-invariant op 자동 인식 및 dispatching. 배치 차원과 무관한 연산을 분리하여 중복 계산 제거 가능 |

---

## Distributed KV Cache / KV Connector

| PR | Repo | Status | Summary |
|----|------|--------|---------|
| [#36668](https://github.com/vllm-project/vllm/issues/36668) | vLLM | RFC | **KVCacheSpec Registry** — KVCacheSpec/Manager 매핑이 하드코딩되어 OOT 플랫폼이 확장 불가능한 문제. pluggable registry + decorator 패턴으로 NPU/TPU 등이 alignment, padding, manager를 vLLM 코드 수정 없이 등록 가능하도록 제안 |
| [#36687](https://github.com/vllm-project/vllm/pull/36687) | vLLM | Open | **NixlConnector hybrid SSM-FA 지원** — Transformer(K/V)와 Mamba(Conv/SSM state) 레이어의 이기종 cache layout을 NixlConnector에서 처리 |
| [#17220](https://github.com/sgl-project/sglang/pull/17220) | SGLang | Merged | **SWA KV cache 디코딩 중 eviction** — 기존에는 요청 완료 시에만 eviction했으나, 디코딩 중 sliding window 밖 토큰을 즉시 제거하도록 변경. 장문 생성 시 메모리 낭비 대폭 감소 |
| [#20252](https://github.com/sgl-project/sglang/issues/20252) | SGLang | Bug | **대규모 P/D cascading failure** — 90 prefill + 30 decode(H20) 환경에서 prefill 서버 재시작 시, decode 서버가 dead prefill에 무한 재연결 시도 → health check timeout → router가 정상 decode도 제거 → 남은 decode에 트래픽 집중 → 전체 장애. 프로덕션 P/D에서 심각한 resilience 이슈 |

---

## Pipeline Parallel / Large Scale Serving

| PR/Issue | Repo | Status | Summary |
|----------|------|--------|---------|
| [#19670](https://github.com/sgl-project/sglang/pull/19670) | SGLang | Merged | **Qwen3.5 Pipeline Parallelism 지원** — PP 시 embed_tokens에 PPMissingLayer placeholder 누락으로 크래시 → 수정. GSM8K accuracy 유지 확인 |
| [#17216](https://github.com/sgl-project/sglang/pull/17216) | SGLang | Merged | **Decode Offload** — 요청 종료 시 일괄 offload → 디코딩 중 page-size 단위 즉시 offload로 변경. DeepSeek-V3.2-W4AFP8 장문(131K) 서빙에서 H20 기준 **동시 처리량 2배 이상, E2E 성능 2배 향상**. P/D 분리 환경(1P PP8 + 1D EP8)에서 검증 |
| [#36643](https://github.com/vllm-project/vllm/issues/36643) | vLLM | Bug | Qwen3.5 PP 미동작 — vLLM에서도 동일한 PP 이슈 보고 |
| [#36594](https://github.com/vllm-project/vllm/issues/36594) | vLLM | Bug | **DP wave 재시작 버그** — DPEngineCoreProc가 pause 상태에서 START_DP_WAVE를 무시하지 않아 collective timeout 발생. pause_generation + collective_rpc 시나리오 |
| [#20273](https://github.com/sgl-project/sglang/pull/20273) | SGLang | Open | `pause_generation` in_place 모드에서 prefill worker 크래시 수정 |
| [#20287](https://github.com/sgl-project/sglang/pull/20287) | SGLang | Open | 비-현재 rank 죽을 때 scheduler launch hang 수정 — 분산 환경 resilience |

---

## Notable

- **vLLM [#36602](https://github.com/vllm-project/vllm/issues/36602)** (RFC): **커널/op 테스트 device-agnostic화** — CUDA_DEVICES 하드코딩, opcheck(), torch.cuda.* 직접 호출 등을 `current_platform` 추상화로 교체 제안. OOT plugin이 vLLM 업스트림 테스트 스위트를 그대로 재사용 가능
- **vLLM [#36701](https://github.com/vllm-project/vllm/pull/36701)**: FlashAttention block size 제한을 hybrid 모델에서 제거 — Qwen3.5 등 SSM-FA 혼합 모델의 유연성 향상
- **vLLM [#35794](https://github.com/vllm-project/vllm/pull/35794)** (Merged): FusedMoEModularKernel output을 torch.empty로 최적화 — 불필요한 zero initialization 제거
- **vLLM [#34206](https://github.com/vllm-project/vllm/pull/34206)** (Merged): grouped topk kernel 최적화 — MoE expert selection 속도 향상
- **SGLang [#19775](https://github.com/sgl-project/sglang/pull/19775)** (Merged): Qwen3.5 GDN gating + recurrent 커널 fuse — verify_target 단계 성능 개선
- **SGLang [#19148](https://github.com/sgl-project/sglang/pull/19148)** (Merged): DeepSeek-V3.2 NSA fuse store indexer k cache JIT 커널

---

*Sources: [vLLM GitHub](https://github.com/vllm-project/vllm), [SGLang GitHub](https://github.com/sgl-project/sglang)*
