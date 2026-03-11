---
layout: post
category: llm-serving
title: "vLLM/SGLang Daily - 2026-03-11"
---

## Highlights

**vLLM 0.17.0 안정화 진행 중** — Qwen3.5, Kimi K2.5 등 최신 모델 지원 관련 버그 리포트가 다수 올라오고 있으며, FlashAttention 4 통합과 Model Runner V2가 핵심 마일스톤을 달성했습니다.

**SGLang — DeepSeek v3.2 대규모 서빙 이슈** — 멀티노드 H200 환경에서 DeepEP internode dispatch 실패 및 런타임 stall 이슈가 보고되고 있어 large-scale MoE 서빙의 안정성 개선이 활발히 진행 중입니다.

---

## Speculative Decoding

| PR | Repo | Status | Summary |
|----|------|--------|---------|
| [#36767](https://github.com/vllm-project/vllm/pull/36767) | vLLM | Open | **Dflash integration** — speculative decoding 파이프라인에 DFlash 지원 추가. DFlash-specific attention metadata 및 slot mapping 처리 |
| [#36527](https://github.com/vllm-project/vllm/pull/36527) | vLLM | Open | Eagle3 speculative decoding Qwen3Next 기반 모델 버그픽스 |
| [#36545](https://github.com/vllm-project/vllm/pull/36545) | vLLM | Open | GPT-OSS draft model용 `norm_before_fc` 추가 |
| [#36391](https://github.com/vllm-project/vllm/pull/36391) | vLLM | Open | Draft-token hook을 spec decode에서 분리하여 모듈성 개선 |
| [#35311](https://github.com/vllm-project/vllm/pull/35311) | vLLM | Open | **FSM(Finite State Machine) Speculative Decoding** — constrained decoding과 spec decode 결합 |
| [#35301](https://github.com/vllm-project/vllm/pull/35301) | vLLM | Open | **Dynamic speculation length** — confidence-threshold 기반 early exit |
| [#36777](https://github.com/vllm-project/vllm/issues/36777) | vLLM | Issue | Kimi K2.5 Speculative Decoding 기능 요청 |
| [#20355](https://github.com/sgl-project/sglang/pull/20355) | SGLang | Open | MTP 모델에서 piecewise CUDA graph 비활성화 픽스 |
| [#20115](https://github.com/sgl-project/sglang/pull/20115) | SGLang | Open | `nextn=2` DeepGEMM fp8 paged MQA logits 성능 최적화 (target_verify) |

### Issue Tracking
- **DFlash**: vLLM [#32206](https://github.com/vllm-project/vllm/pull/32206) → [#34014](https://github.com/vllm-project/vllm/pull/34014) → [#36767](https://github.com/vllm-project/vllm/pull/36767)로 이어지는 통합 작업. Ascend NPU에서도 DFlash 지원 PR [#36764](https://github.com/vllm-project/vllm/pull/36764)가 동시에 올라옴
- **MTP + NVFP4**: Qwen3.5 weight shape mismatch 이슈 ([#35041](https://github.com/vllm-project/vllm/pull/35041), [#35675](https://github.com/vllm-project/vllm/pull/35675)) 아직 open 상태

---

## vLLM IR / Model Runner V2

| PR | Status | Summary |
|----|--------|---------|
| [#33825](https://github.com/vllm-project/vllm/pull/33825) | Open | **vLLM IR 1/N** — IR skeleton 및 `rms_norm` op 구현 |
| [#36722](https://github.com/vllm-project/vllm/pull/36722) | Open | **vLLM IR 2/N** — batch-invariant-aware dispatching 및 rms_norm |
| [#34068](https://github.com/vllm-project/vllm/pull/34068) | Open | **vLLM IR 3/N** — `fused_add_rms_norm` 및 `maybe_inplace` |
| [#36762](https://github.com/vllm-project/vllm/pull/36762) | Open | Model Runner V2 — 미사용 `warmup_for_prefill` 메서드 제거 |
| [#35520](https://github.com/vllm-project/vllm/pull/35520) | Open | Model Runner V2 — Qwen3.5 / Mamba hybrid 모델 지원 |

### Note
vLLM IR 시리즈가 본격적으로 진행되고 있습니다. IR 레이어에서 batch-invariant operation을 인식하고 dispatching하는 구조를 잡아가고 있으며, 이는 향후 커널 fusion 및 그래프 최적화의 기반이 될 전망입니다.

---

## Distributed KV Cache / KV Connector

| PR | Status | Summary |
|----|--------|---------|
| [#36735](https://github.com/vllm-project/vllm/pull/36735) | Open | NIXL runtime wheel Docker 빌드 해결 |
| [#36549](https://github.com/vllm-project/vllm/pull/36549) | Open | MultiConnector에서 HMA sub-connector 버그픽스 |
| [#36424](https://github.com/vllm-project/vllm/pull/36424) | Open | KV connector dead code 제거 리팩토링 |
| [#35876](https://github.com/vllm-project/vllm/pull/35876) | Open | NIXL throughput 계산을 timestamp 기반으로 수정 |
| [#34328](https://github.com/vllm-project/vllm/pull/34328) | Open | **FlexKV** — KV Cache Offloading 옵션으로 FlexKV 지원 |
| [#34312](https://github.com/vllm-project/vllm/pull/34312) | Open | **Mooncake Store Connector** 추가 |
| [#36780](https://github.com/vllm-project/vllm/issues/36780) | RFC | **NixlConnector에 hybrid SSM-FA 모델 지원** — FA/Mamba 레이어의 서로 다른 state layout(K/V vs Conv/SSM) 처리 |

### Issue Tracking
- **NIXL Connector**: Docker 빌드([#36735](https://github.com/vllm-project/vllm/pull/36735)), throughput 측정([#35876](https://github.com/vllm-project/vllm/pull/35876)), hybrid 모델 지원([#36780](https://github.com/vllm-project/vllm/issues/36780))이 동시에 진행 중. P/D disaggregation의 핵심 인프라로 자리잡고 있음

---

## Pipeline Parallel / Large Scale Serving

| PR/Issue | Repo | Status | Summary |
|----------|------|--------|---------|
| [#24403](https://github.com/vllm-project/vllm/pull/24403) | vLLM | Open | NIXL 기반 P/D에서 Pipeline Parallel 지원 (장기 진행) |
| [#20336](https://github.com/sgl-project/sglang/issues/20336) | SGLang | Bug | **DeepSeek v3.2** — 2노드 8xH200 DeepEP internode dispatch 실패 |
| [#20315](https://github.com/sgl-project/sglang/issues/20315) | SGLang | Bug | **DeepSeek v3.2** — 정상 서빙 후 런타임 stall, py-spy dump 트리거 후 watchdog timeout |
| [#20332](https://github.com/sgl-project/sglang/issues/20332) | SGLang | RFC | **MoE 모델 TP 기반 빠른 로딩** |
| [#20346](https://github.com/sgl-project/sglang/pull/20346) | SGLang | Open | **Pod Attention** — Mixed Chunk Prefill 통합 |
| [#20343](https://github.com/sgl-project/sglang/pull/20343) | SGLang | Open | **HiSparse** — DeepSeek용 Sparse Attention |
| [#20286](https://github.com/sgl-project/sglang/issues/20286) | SGLang | Bug | Qwen3-32B TP=2 + PD-Multiplexing에서 H100 PCIe illegal memory access |
| [#20349](https://github.com/sgl-project/sglang/pull/20349) | SGLang | Open | PD Mux prefill groups — attn/moe 분리 초기화 수정 |

### Note
SGLang에서 DeepSeek v3.2 대규모 서빙(멀티노드) 안정성 이슈가 연속으로 보고되고 있습니다. Expert Parallelism과 PD Disaggregation 조합에서의 edge case들이 드러나는 단계로 보입니다.

---

## Notable Issues & Bugs (2026-03-11)

- **vLLM [#36771](https://github.com/vllm-project/vllm/issues/36771)**: LMCache가 vLLM 0.17.0 (Qwen3Next)과 호환되지 않음
- **vLLM [#36776](https://github.com/vllm-project/vllm/issues/36776)**: Qwen3.5 DP=8 환경에서 크래시
- **vLLM [#36755](https://github.com/vllm-project/vllm/issues/36755)**: Preemption + async scheduling race로 Prometheus counter 크래시
- **vLLM [#36789](https://github.com/vllm-project/vllm/pull/36789)**: input prompt가 너무 긴 경우 negative max_tokens 버그 수정
- **SGLang [#20334](https://github.com/sgl-project/sglang/issues/20334)**: CUDA Graph + MTP + page_size=64에서 MiMo-V2-Flash precision 이슈
- **SGLang [#20363](https://github.com/sgl-project/sglang/issues/20363)**: NVIDIA DGX sm121a에서 FA4 지원 불가

## Community & Ecosystem

- **SGLang [#20358](https://github.com/sgl-project/sglang/issues/20358)**: Anthropic Compatible API 지원 요청 — SGLang Model Gateway에 Anthropic API 호환 추가
- **SGLang [#20307](https://github.com/sgl-project/sglang/issues/20307)**: MoE Routing Simulator — Expert Load Imbalance 프로파일링 도구
- **vLLM [#36786](https://github.com/vllm-project/vllm/pull/36786)**: EPLB(Expert-level load balancing) 지원 확장
- **vLLM [#36768](https://github.com/vllm-project/vllm/pull/36768)**: FlashInfer 0.6.6 업데이트
- **SGLang [#20348](https://github.com/sgl-project/sglang/pull/20348)**: Blackwell NVFP4 dense GEMM torch backend 추가

---

*Sources: [vLLM GitHub](https://github.com/vllm-project/vllm), [SGLang GitHub](https://github.com/sgl-project/sglang), [SGLang Q1 Roadmap](https://github.com/sgl-project/sglang/issues/12780)*
