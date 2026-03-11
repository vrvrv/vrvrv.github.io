---
layout: post
category: llm-serving
title: "vLLM/SGLang Daily - 2026-03-11"
---

## Highlights

오늘은 speculative path 안정화와 대규모 MoE 운영 안정성 이슈가 동시에 부각된 날이다.
vLLM은 DFlash spec decode 통합과 NIXL hybrid SSM-FA RFC로 "attention backend/descriptor 다변화"를 밀었고,
SGLang은 DeepSeek-V3.2 멀티노드 stall/dispatch 실패 리포트와 Pod Attention/HiSparse로 "고부하 운영 병목"을 정면으로 다뤘다.

## Speculative Decoding

| PR | Repo | Status | 요약 |
|----|------|--------|------|
| [#36767](https://github.com/vllm-project/vllm/pull/36767) | vLLM | Open | DFlash를 speculative proposer 경로에 통합. Qwen3-8B, draft=16, max_tokens=2048에서 batch=32 기준 2741 tok/s로 SGLang 2540 tok/s 대비 약 8% 높음. |
| [#20355](https://github.com/sgl-project/sglang/pull/20355) | SGLang | Open | MTP 모델에 `layers` 속성이 없는 경우 piecewise CUDA graph를 비활성화해 크래시를 방지. SpecForge 계열 Eagle3 모델의 graph 안정성 보강. |
| [#19775](https://github.com/sgl-project/sglang/pull/19775) | SGLang | Merged | verify_target의 GDN 경로에서 gating+recurrent update 커널을 재구성. Qwen3-Next 8k/1.5k 벤치에서 TTFT 약 11%, output tok/s 약 13% 개선 보고. |

vLLM의 #36767은 `vllm/v1/spec_decode/eagle.py`와 `gpu_model_runner.py`를 건드리며,
spec decode를 "PagedAttention 전제"에서 "backend-특화 metadata 선택" 구조로 이동시키는 단계다.
핵심은 first drafting pass에서 DFlash query input과 packed slot mapping을 보존하는 것으로,
이 경로가 깨지면 acceptance length가 유지돼도 verifier throughput이 급락한다.
동시에 SGLang #20355는 model runner 레벨에서 PCG 적용 대상을 좁혀,
MTP 계열의 비정형 layer topology가 graph capture를 깨뜨리는 문제를 막는다.
#19775가 verify_target kernel 내부 state 제어를 다시 설계한 점까지 묶어 보면,
양쪽 모두 "speculative 알고리즘 자체"보다 "speculative를 유지하는 런타임 경계조건"에 집중하고 있다.
3/10 포스트에서 보인 Eagle/MTP 확장 흐름이, 3/11에는 안정성/커널 통합 국면으로 넘어온 셈이다.

## Distributed KV Cache / KV Connector

| PR/Issue | Repo | Status | 요약 |
|----------|------|--------|------|
| [#36780](https://github.com/vllm-project/vllm/issues/36780) | vLLM | Open (RFC) | NixlConnector에 hybrid SSM-FA 지원 제안. 동일 메모리 region 위에 FA descriptor view와 Mamba descriptor view를 이중 구성하는 설계. |
| [#36735](https://github.com/vllm-project/vllm/pull/36735) | vLLM | Open | Docker 빌드에서 `nixl` meta package 대신 CUDA major별 `nixl-cu12/cu13`으로 해석하도록 수정. 테스트 4건 통과로 배포 경로 안정화. |
| [#20343](https://github.com/sgl-project/sglang/pull/20343) | SGLang | Open | HiSparse 도입으로 NSA 모델 디코딩 중 idle KV를 CPU로 offload해 GPU batch 여유 확보를 노림. DeepSeek-V3.2/GLM-5를 주요 타깃으로 명시. |
| [#17216](https://github.com/sgl-project/sglang/pull/17216) | SGLang | Merged | decode 종료 시점 offload를 page 단위 즉시 offload로 변경. 16k/1.3k 장문 벤치에서 output tok/s 139.26→296.37로 약 2.1x 개선 보고. |

이 섹션의 공통점은 "KV를 어디에, 어떤 단위로, 어떤 descriptor로 본다"는 저장 계층 설계다.
vLLM RFC #36780은 HMA 전제 하에 FA(K/V)와 Mamba(Conv/SSM)가 같은 tensor를 공유하되,
block id를 서로 다른 descriptor 집합으로 해석하는 dual-view 접근을 제안한다.
이는 connector가 단순 전송 계층이 아니라, logical block과 physical block의 변환 계층이 됨을 의미한다.
#36735는 이 복잡한 connector 경로를 실제 배포에서 깨뜨리던 wheel 해석 문제를 잡는 운영 패치다.
SGLang #17216/#20343은 같은 문제를 runtime policy 측면에서 다룬다.
즉시 offload와 HiSparse는 모두 "활성 토큰 집합만 GPU에 남긴다"는 전략이며,
PD disaggregation에서 decode 병목을 메모리 상주량으로 직접 완화한다.
3/10의 SWA eviction 흐름이 3/11에는 계층형 KV 운영(온디바이스-호스트)으로 확장된 모습이다.

## Pipeline Parallel / Large Scale Serving

| PR/Issue | Repo | Status | 요약 |
|----------|------|--------|------|
| [#20346](https://github.com/sgl-project/sglang/pull/20346) | SGLang | Open | Mixed Chunk Prefill에서 Pod Attention 커널을 선택 적용. 고동시·장입력 구간에서 throughput +4~48%, TTFT/TPOT -5~30%, 피크 throughput +47.7% 보고. |
| [#20349](https://github.com/sgl-project/sglang/pull/20349) | SGLang | Open | PD-Multiplexing prefill stream용 attn/MoE 통신 그룹 누락으로 illegal memory access가 나던 문제 수정. 분리 stream에서 group 초기화를 명시화. |
| [#20336](https://github.com/sgl-project/sglang/issues/20336) | SGLang | Open (Bug) | 2노드(각 8xH200) DeepSeek-V3.2에서 DeepEP internode dispatch timeout 및 illegal memory access 보고. |
| [#20315](https://github.com/sgl-project/sglang/issues/20315) | SGLang | Open (Bug) | TP16/DP16 장시간 서비스 후 전 rank 동시 stall, py-spy dump 이후 watchdog timeout(300s) 발생. |
| [#20332](https://github.com/sgl-project/sglang/issues/20332) | SGLang | Open (RFC) | TP 로딩의 디스크 접근 패턴 병목을 지적하며, EP 방식 로딩 후 D2D 재배치로 DeepSeek-R1 로딩 시간을 ~1h→~2min 줄였다는 제안. |

대규모 서빙 관점에서 3/11 이슈는 "커널 최적화 속도"보다 "분산 상태 일관성"이 병목임을 보여준다.
#20346은 attention kernel을 더 빠르게 만드는 전형적 성능 PR이지만,
실운영에서는 #20336/#20315처럼 node 간 dispatch와 watchdog 경로가 전체 가용성을 좌우한다.
#20349가 중요한 이유도 동일하다: PD-Multiplexing은 stream 분리만으로 끝나지 않고,
prefill/decode 각각의 attn/EP/DP group 생명주기를 분리하지 않으면 통신 레이어에서 터진다.
#20349 변경 파일이 `distributed/parallel_state.py` 한 곳에 집중된 것도,
문제가 kernel이 아니라 process-group 구성 단계의 초기화 순서였음을 보여준다.
#20332 RFC가 지적한 TP 로딩 병목은 초기화 단계의 tail latency를 줄여
복구 시간(RTO)과 autoscaling 효율까지 건드리는 주제라, 단순 startup 개선으로 보기 어렵다.
결국 SGLang 대규모 서빙 트랙은 "kernel + scheduler + comm group + bootstrap I/O"를
하나의 파이프라인으로 최적화하는 방향으로 이동 중이며,
3/10에서 확인된 PD/PP 확장 흐름이 3/11에는 장애 모드와 복구 경로 검증으로 이어진다.

## Notable Issues & Bugs

| PR/Issue | Repo | Status | 요약 |
|----------|------|--------|------|
| [#36755](https://github.com/vllm-project/vllm/issues/36755) | vLLM | Open (Bug) | preemption 시 `num_cached_tokens` 리셋 누락과 async scheduling race가 겹치며 `local_cache_hit` counter 음수 증가로 엔진 크래시. |
| [#20334](https://github.com/sgl-project/sglang/issues/20334) | SGLang | Open (Bug) | CUDA Graph + MTP + page_size=64 조합에서 MiMo-V2-Flash precision 저하 보고. graph 최적화와 품질 보존 간 충돌 사례. |

vLLM #36755는 scheduler 출력 물질화가 live mutable `Request`를 읽는 구조적 위험을 찌른다.
issue 본문은 `_preempt_request()`에서 `num_cached_tokens=-1` 리셋 필요성과,
schedule(N+1) / update_from_output(N) 겹침 시 snapshot 기반 accounting이 필요하다고 명확히 제시한다.
이 문제는 KV connector + async scheduling + preemption이 동시에 켜질 때 증폭되므로,
단일 버그라기보다 V1 엔진 동시성 모델의 경계조건 테스트로 보는 편이 정확하다.
즉 `scheduler.py`의 상태 전이와 metrics logger의 관측 타이밍이 분리되지 않은 것이 핵심이며,
fix가 들어가더라도 동일 계열 race는 output materialization 경로 전반을 다시 보게 만들 가능성이 높다.
SGLang #20334도 같은 결의 사례다.
MTP와 CUDA graph는 각각 성능 기능이지만, 결합 시 numerical behavior가 달라질 수 있다.
두 이슈 모두 "고성능 기능 조합"이 관측 가능성/정확성 계층을 먼저 압박한다는 점에서,
앞으로 메트릭 방어 로직과 안전한 fallback path가 핵심 유지보수 포인트가 될 가능성이 높다.

## Community & Ecosystem

| PR/Issue | Repo | Status | 요약 |
|----------|------|--------|------|
| [#36786](https://github.com/vllm-project/vllm/pull/36786) | vLLM | Open (Draft) | EPLB 적용 범위를 확장하는 대형 변경(20 files). routing simulator/router/quantized MoE 경로까지 연쇄 수정 중. |
| [#20307](https://github.com/sgl-project/sglang/issues/20307) | SGLang | Open (Feature) | MoE Routing Simulator 제안. TensorRT-LLM의 perfect router, vLLM simulator 사례를 참조해 expert imbalance 상한선 계측 목적. |

두 항목은 당장 성능 수치보다 "운영 의사결정 도구"를 강화한다는 공통점이 있다.
vLLM #36786은 draft 단계지만 router 계층 전반을 건드려 EPLB를 기본 기능으로 밀어 넣는 흐름이다.
특히 `routing_simulator_router.py`까지 함께 수정된 점은
실제 배치에서의 imbalance 관측과 보정 전략을 같은 프레임에서 다루려는 신호로 읽힌다.
SGLang #20307 역시 동일하게, imbalance 완화 기법을 넣기 전에
"현재 손실이 얼마인지"를 계측하는 시뮬레이터가 먼저 필요하다는 문제의식이다.
대규모 MoE 서빙이 tuning 단계에서 운영 단계로 넘어가면서,
프레임워크가 단순 추론 엔진을 넘어 capacity planning 도구를 내장하는 방향이 강화되고 있다.

---

*Sources: [vLLM GitHub](https://github.com/vllm-project/vllm), [SGLang GitHub](https://github.com/sgl-project/sglang)*
