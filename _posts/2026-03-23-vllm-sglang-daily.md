---
layout: post
category: llm-serving
title: "vLLM/SGLang Daily - 2026-03-23"
---

## Highlights

3/12~3/23 기간 동안 양 프레임워크 모두 **Hybrid Model(GDN/Mamba+Attention) 지원 고도화**와 **KV Cache 계층화(offload/eviction/HiCache)**, 그리고 **P/D Disaggregation 운영 안정성**에 집중했다.
vLLM은 HMA 모델을 위한 contiguous KV cache allocation, T-LRU eviction policy, MooncakeConnector health check 등 인프라 계층을 확장했고,
SGLang은 GDN packed decode 커널(머지), HybridRadixTree V2, HiCache+spec decode 연동, Ngram spec decode C++ 리팩토링 등 hybrid 모델 런타임과 캐시 계층을 동시에 재설계하고 있다.

## Speculative Decoding

| PR | Repo | Status | 요약 |
|----|------|--------|------|
| [#37880](https://github.com/vllm-project/vllm/pull/37880) | vLLM | Open | `--speculative-config`에 `moe_backend` 필드 추가 — draft 모델과 target 모델에 서로 다른 MoE backend(marlin/cutlass 등) 지정 가능 |
| [#37812](https://github.com/vllm-project/vllm/pull/37812) | vLLM | Open | MRV2 warmup에서 spec decode 경로를 고려하도록 수정 |
| [#37813](https://github.com/vllm-project/vllm/pull/37813) | vLLM | Open | GDN prefill 경로에서 conv output split을 Q/K/V 분리 없이 fuse — Qwen3.5-35B TTFT ~1% 개선 |
| [#21207](https://github.com/sgl-project/sglang/pull/21207) | SGLang | Open | DP attention 하에서 EAGLE `draft_forward()`의 multi-step 루프가 padded state를 다음 step으로 누출하는 문제 수정 — 800번째 요청 부근 크래시 방지 |
| [#21186](https://github.com/sgl-project/sglang/pull/21186) | SGLang | Merged | Ngram spec decode 3/N: C++ trie의 insert queue race 수정 — condition variable 기반 동기화로 전환 |
| [#21181](https://github.com/sgl-project/sglang/pull/21181) | SGLang | Merged | Ngram spec decode 2/N: branch length → max trie depth 리네이밍, 코드 정리 |
| [#21094](https://github.com/sgl-project/sglang/pull/21094) | SGLang | Open | DLLM(Discrete LLM) Joint Threshold/Low Confidence 경로를 vectorized tensor ops로 전환 — Python 루프 제거로 output throughput 275→299 tok/s (+8.7%) |
| [#21165](https://github.com/sgl-project/sglang/pull/21165) | SGLang | Open | Piecewise CUDA Graph에서 런타임 torch guard 제거 — shape별 fx graph가 고정이므로 guard 오버헤드 불필요 |

이 기간 speculative decoding 트랙은 "알고리즘 확장"보다 **"운영 안정성 + 런타임 효율"**에 무게가 실린다.
vLLM #37880은 FP8 target + BF16 draft 같은 heterogeneous quantization 조합에서 MoE backend가 일치하지 않아 성능이 저하되던 문제를 해결한다. 기존에는 `--moe-backend` 하나로 target과 draft 모두에 적용됐는데, draft 모델은 unquantized인 경우가 많아 `cutlass`가 더 적합할 수 있다. 이 변경은 `SpeculativeConfig` → `create_vllm_config_for_draft_model` 경로에서 drafter의 `FusedMoEConfig`만 선택적으로 override한다.

SGLang 쪽에서 가장 주목할 변경은 #21207이다. EAGLE의 multi-step draft 루프는 동일한 `forward_batch`에 대해 여러 번 decode forward를 호출하는데, DP attention이 켜지면 `prepare_mlp_sync_batch()`가 padding을 추가한다. 문제는 `post_forward_mlp_sync_batch()`가 모든 상태를 완벽히 복원하지 못해, `spec_info.accept_length`와 `global_num_tokens_cpu`가 다음 draft step으로 누출된다는 점이다. 이 stale padded state가 `_pad_tensor_to_size()`에서 음수 차원을 만들어 ~800번째 요청에서 크래시를 유발한다. Fix는 draft loop 내에서 save/restore를 명시적으로 수행하는 최소 범위 패치다.

Ngram 시리즈(#21181, #21186)는 SGLang의 C++ 기반 N-gram trie를 체계적으로 리팩토링하는 흐름이다. 3/N에서 수정된 동기화 race는 insert queue가 drain된 후에도 worker thread가 아직 trie를 갱신 중일 때 `synchronize()`가 조기 리턴하는 문제로, busy-wait를 condition variable로 교체했다. 이는 N-gram spec decode의 프로덕션 안정성을 높이는 기반 작업이다.

#21094의 DLLM vectorized decoding은 LLaDA 같은 discrete diffusion 모델의 병렬 디코딩 경로에서, Python `for batch_size` 루프와 `.item()` 동기화를 제거하고 batched tensor ops로 전환한다. 동시에 decode loop 내 반복적인 attention backend 초기화를 `skip_attn_backend_init` 플래그로 건너뛴다. 이 최적화는 DLLM뿐 아니라 multi-step draft 경로 전반에 적용 가능한 패턴이다.

## Distributed KV Cache / KV Connector

| PR/Issue | Repo | Status | 요약 |
|----------|------|--------|------|
| [#37885](https://github.com/vllm-project/vllm/pull/37885) | vLLM | Open | HMA 모델(Gemma3 등)에 contiguous KV cache allocation 확장 — 동일 page size의 multi-group KV cache를 연속 메모리에 배치해 RDMA 전송 효율 개선 |
| [#37874](https://github.com/vllm-project/vllm/pull/37874) | vLLM | Open | `LRUManager`/`ARCManager`를 `CPUOffloadingManager` + pluggable `CachePolicy` 패턴으로 통합 — eviction 알고리즘 교체를 policy 등록만으로 가능하게 함 |
| [#37825](https://github.com/vllm-project/vllm/pull/37825) | vLLM | Open | T-LRU(Tail-Optimized LRU) eviction policy 구현 — NeurIPS 2025 논문 기반, TEL-safe cap으로 대화형 워크로드의 P95/P99 TTFT 최적화 |
| [#37853](https://github.com/vllm-project/vllm/pull/37853) | vLLM | Open | kv_offload+HMA 시리즈 7/N: hybrid 모델의 `register_kv_caches`에서 canonical tensor set + group별 page size 매핑 정의 |
| [#37859](https://github.com/vllm-project/vllm/pull/37859) | vLLM | Open | P/D 환경에서 request abort 후 도착하는 stale KV transfer callback을 skip 처리 — assert 실패로 decode 노드 크래시 방지 |
| [#37894](https://github.com/vllm-project/vllm/pull/37894) | vLLM | Open | MooncakeConnector Proxy에 prefill 노드 health check + 자동 failover 기능 추가 — unhealthy 노드 자동 제외 및 복구 시 재포함 |
| [#21206](https://github.com/sgl-project/sglang/pull/21206) | SGLang | Open | MambaRadixCache/SWARadixCache를 `HybridRadixTree V2`로 통합 — TreeComponent 플러그인 방식으로 새 attention 변종 지원 시 core tree 로직 수정 불필요 |
| [#21125](https://github.com/sgl-project/sglang/pull/21125) | SGLang | Open | HiCache에 draft KV cache L2/L3 backing 추가 — target KV load_back 시 draft KV도 동기화하여 accept length 퇴행 방지 (3.07→6.94) |

이 기간 KV cache 트랙의 핵심 주제는 **"다양한 모델 아키텍처와 배포 토폴로지에서 KV cache를 통합적으로 관리하는 추상화 계층 구축"**이다.

vLLM에서 가장 구조적인 변화는 #37885와 #37874의 조합이다. #37885는 기존에 single-group(uniform) 모델만 지원하던 contiguous KV cache allocation을 HMA 모델로 확장한다. Gemma3 같은 모델은 full attention과 sliding window가 서로 다른 eviction policy를 갖지만, page size가 동일하면 물리적으로 연속된 메모리 영역에 배치할 수 있다. 이는 NIXL/Mooncake 같은 RDMA 기반 KV connector가 단일 전송으로 전체 KV를 옮길 수 있게 하는 전제 조건이다. #37874는 eviction 알고리즘 자체를 pluggable policy로 분리하여, LRU/ARC 외에 T-LRU(#37825) 같은 새로운 eviction 전략을 `_CACHE_POLICIES` 레지스트리에 등록만으로 추가할 수 있게 만든다. T-LRU는 대화형 워크로드에서 conversation history 길이와 예상 다음 쿼리 길이를 기반으로 "안전하게 evict할 수 있는 tail 블록"을 별도 큐로 우선 배출하는 전략으로, P95/P99 TTFT를 직접 타겟한다.

P/D disaggregation 운영 안정성 측면에서 #37859와 #37894가 중요하다. #37859는 decode 측에서 request를 abort한 뒤에도 비동기로 도착하는 `finished_recving`/`finished_sending` 콜백이 이미 제거된 request를 참조해 assert 실패를 일으키던 문제를 잡는다. #37894는 더 큰 그림으로, MooncakeConnector proxy에 주기적 health check를 넣어 prefill 노드 장애 시 자동으로 라우팅에서 제외하고 복구 시 재포함한다. 이는 3/11 포스트에서 다룬 #37745(prefill 재시작 후 decode 노드가 old engine_id를 계속 사용하는 문제)에 대한 직접적 대응이다.

SGLang #21206의 HybridRadixTree V2는 아키텍처 수준의 리팩토링이다. 기존에는 MambaRadixCache와 SWARadixCache가 각각 독립적으로 구현되어, 새로운 attention 변종(full+SWA+SSM 혼합 등)을 추가하려면 또 다른 cache 구현이 필요했다. V2는 core tree 연산(match/insert/evict/LRU)을 component-agnostic으로 만들고, `MambaComponent`/`SWAComponent` 등을 플러그인으로 붙이는 구조다. AIME25 벤치에서 기존 MambaRadixTree 대비 동등한 정확도를 확인했다. #21125의 HiCache draft KV backing은 spec decode + HiCache 조합에서 발생하는 근본적 문제를 해결한다: target KV가 L2/L3에서 load_back될 때 draft model의 KV는 갱신되지 않아 accept length가 급락하는 현상을, draft KV도 함께 offload/load_back하는 방식으로 해결한다.

## Pipeline Parallel / Large Scale Serving

| PR/Issue | Repo | Status | 요약 |
|----------|------|--------|------|
| [#37824](https://github.com/vllm-project/vllm/pull/37824) | vLLM | Open | RIY(Reap It Yourself): 런타임 expert pruning — JSON 프로파일로 미사용 expert를 로딩 시 제거, VRAM 절감. Qwen3.5-397B INT4에서 20% pruning 시 0% 오버헤드 |
| [#37830](https://github.com/vllm-project/vllm/pull/37830) | vLLM | Merged | MRV2에서 PP + CUDA graph 테스트 활성화 |
| [#37818](https://github.com/vllm-project/vllm/pull/37818) | vLLM | Merged | MRV2 piecewise CUDA graph에서 hidden states allocation 생략 — 불필요한 메모리 할당 제거 |
| [#37799](https://github.com/vllm-project/vllm/pull/37799) | vLLM | Open | MLA decode output buffer를 layer 간 공유 — DeepSeek-R1 61개 레이어에서 ~15 GiB → ~256 MiB 메모리 절감, OOM 해결 |
| [#37776](https://github.com/vllm-project/vllm/pull/37776) | vLLM | Open | MoE oracle들을 class 구조로 통합 — oracle 선택을 data-driven으로 전환 |
| [#21098](https://github.com/sgl-project/sglang/pull/21098) | SGLang | Open | parallel state 리팩토링 3/N: `BaseCommunicator` 추상화 도입 — `can_all_reduce()`, `all_reduce(inplace=...)` 등 통합 인터페이스, backend-name 분기 제거 |
| [#21096](https://github.com/sgl-project/sglang/pull/21096) | SGLang | Open | PP+TP 환경에서 `DecodeInputBuffers.create()`의 tensor device 배치 오류 수정 — `torch.device` context manager 대신 명시적 device 파라미터 |
| [#21205](https://github.com/sgl-project/sglang/pull/21205) | SGLang | Open | MiniMax M2.5에서 EPLB 실행 시 `routed_experts_weights_of_layer` 속성 누락 크래시 수정 |
| [#18858](https://github.com/sgl-project/sglang/pull/18858) | SGLang | Merged | Blackwell MXFP4 MoE weight loading ~9.5x 가속 (265s→28s) — per-expert shuffle을 cached FlashInfer permute-index로 교체 |

대규모 서빙 트랙에서 이 기간의 핵심 주제는 **"메모리 효율 + 초기화 속도 + 분산 통신 추상화"** 세 축이다.

vLLM #37799는 DeepSeek-R1 같은 대규모 MoE 모델의 실제 운영에서 발견된 OOM을 해결한다. PR #37442에서 CUDA graph padding slot의 NaN 방지를 위해 추가한 per-layer `_decode_out` 버퍼가 61개 레이어에서 ~15 GiB를 먹는 문제다. MLA forward는 sequential하므로 layer 간 동시 사용이 없어, module-level singleton으로 공유하면 ~256 MiB로 줄어든다. 이 패턴은 FlashInferMLAImpl과 CutlassMLAImpl 모두에 적용된다.

#37824 RIY는 MoE 모델의 expert utilization이 불균일한 실운영 환경을 타겟한다. JSON 프로파일에 pruned expert 목록을 명시하면, 로딩 시 해당 expert를 할당하지 않고 router의 logit mask로 라우팅에서 제외한다. 모니터 모드(`VLLM_RIY_MONITOR=1`)에서는 per-expert activation stats를 수집해 프로파일을 생성하고, 프로덕션 모드에서는 0% 오버헤드로 동작한다. Qwen3.5-397B INT4 TP=2에서 20% pruning이 정확도 손실 없이 VRAM을 절감한 벤치마크가 제시되었다.

SGLang #21098의 `BaseCommunicator` 추상화는 분산 통신 스택의 근본적 리팩토링이다. 기존에는 backend별로 `custom_all_reduce`, `should_*` 같은 메서드가 산발적으로 구현되어 있었는데, `can_all_reduce()` → `all_reduce(inplace=...)` → `should_use_custom_op()` → `graph_capture_context()` 라는 통합 계약으로 정리한다. 이는 새로운 통신 backend(예: NPU 전용) 추가 시 기존 코드 수정 없이 `CommunicatorImpl`에 등록만 하면 되게 만든다.

#18858의 MXFP4 weight loading 9.5x 가속은 Blackwell에서 대규모 MoE 모델의 cold start 시간을 직접 단축한다. 기존 per-expert shuffle이 265초 걸리던 것을 cached FlashInfer permute-index 기반 전처리로 28초로 줄였다. 이는 autoscaling 환경에서 scale-up 시간에 직접 영향을 주는 개선이다.

## Kernel & Performance

| PR | Repo | Status | 요약 |
|----|------|--------|------|
| [#20627](https://github.com/sgl-project/sglang/pull/20627) | SGLang | Merged | GDN packed decode 커널 — 6-step decode를 3-step으로 축소, zero-copy Q/K/V 읽기 + gate/beta 인라인 계산으로 커널 호출 50% 감소 |
| [#21203](https://github.com/sgl-project/sglang/pull/21203) | SGLang | Open | CuTeDSL KDA decode 커널 — Kimi-Linear/Kimi-2.5용, batch=1에서 Triton 대비 1.05x speedup |
| [#21104](https://github.com/sgl-project/sglang/pull/21104) | SGLang | Open | FA3 `scheduler_metadata` 사전 계산 — 64-layer 모델에서 per-layer `prepare_varlen_num_blocks` 호출 63회 제거, decode throughput +1.4~2.6% |
| [#37789](https://github.com/vllm-project/vllm/pull/37789) | vLLM | Open | GDN prefill conv output의 Q/K/V split → fused rearrange로 전환, 메모리 복사 감소 |
| [#35777](https://github.com/vllm-project/vllm/pull/35777) | vLLM | Merged | Qwen3 Next용 `fused_sigmoid_gating_delta_rule_update` 커널 추가 |
| [#34206](https://github.com/vllm-project/vllm/pull/34206) | vLLM | Merged | grouped topk 커널 최적화 |
| [#19536](https://github.com/sgl-project/sglang/pull/19536) | SGLang | Merged | NSA backend metadata를 MTP 하에서 최적화 — spec decode 시 불필요한 metadata 재계산 제거 |

GDN(Gated Delta Network)/KDA(Key-Dimension Attention) 계열 커널 최적화가 이 기간의 중심 테마다.

SGLang #20627의 GDN packed decode는 기존 6-step decode 경로(conv1d → split → contiguous copy → gate 계산 → beta 계산 → recurrence)를 3-step(conv1d → packed kernel → transpose)으로 압축한다. 핵심은 `fused_recurrent_gated_delta_rule_packed_decode()` 단일 커널이 packed QKV 레이아웃에서 pointer arithmetic으로 Q/K/V를 직접 읽고, gate(g)와 beta를 레지스터에서 계산한 뒤, recurrence update와 output write까지 한 번에 처리하는 것이다. GDN의 비대칭 head 설계(Q/K head 수 ≠ V head 수)를 packed layout 내에서 `H`와 `HV`를 동시에 인덱싱하는 방식으로 처리한다.

#21203의 CuTeDSL KDA decode 커널은 같은 계열의 최적화를 KDA(Key-Dimension Attention) 아키텍처로 확장한다. KDA는 GDN과 유사하지만 per-head-per-key gating(shape `[N, 1, HV, K]`)을 사용해 더 복잡한 forget pattern을 학습할 수 있는 대신 compute complexity가 높다. batch=1에서 1.05x speedup으로 아직 극적이지 않지만, prefill 커널까지 완성되면 end-to-end 통합이 가능해진다.

#21104의 FA3 scheduler_metadata 사전 계산은 "per-layer 중복 호출 제거"라는 단순한 아이디어지만, 64-layer 모델에서 63회의 GPU 커널 호출을 아끼는 효과가 있다. BS=1에서 TPOT 7.6ms → 7.4ms(-2.6%), TP=8에서는 ~5% 개선이 예상된다. vLLM에서는 이미 동일한 최적화가 적용되어 있어, SGLang이 이를 따라가는 형태다.

## Notable Issues & Bugs

| Issue | Repo | Status | 요약 |
|-------|------|--------|------|
| [#37745](https://github.com/vllm-project/vllm/issues/37745) | vLLM | Open | MooncakeConnector: prefill 노드 재시작 후 decode 노드가 old engine_id로 요청 → WAITING_FOR_REMOTE_KVS 영구 stall |
| [#37703](https://github.com/vllm-project/vllm/issues/37703) | vLLM | Open | TRITON_ATTN이 `VLLM_KV_CACHE_LAYOUT=HND`를 무시 → heterogeneous TP + NIXL P/D에서 KV layout 불일치 |
| [#37638](https://github.com/vllm-project/vllm/issues/37638) | vLLM | Open | Tracking: Mamba Heterogeneous TP for NIXL P/D — conv_states의 비균일 차원(x_dim ≠ B_dim) 때문에 flat split 불가, 3가지 접근법(staging buffer / chunk-interleaved permutation / 3-read) 비교 실험 중 |
| [#37796](https://github.com/vllm-project/vllm/pull/37796) | vLLM | Open | Hybrid 모델(Qwen3.5) preemption 후 stale `prompt_logprobs` 잔류 → mamba state 소실과 결합해 livelock 발생 |
| [#21210](https://github.com/sgl-project/sglang/issues/21210) | SGLang | Open | EAGLE draft_forward에서 DP attention padding state 누출 → 음수 차원 크래시 (#21207로 수정) |
| [#21138](https://github.com/sgl-project/sglang/issues/21138) | SGLang | Open | NemotronH에서 MTP spec decode accept rate가 0.33에 고정 — draft token을 항상 reject |

P/D disaggregation 관련 이슈들이 두드러진다. #37745는 MooncakeConnector의 engine_id가 prefill 재시작 시 변경되는데, decode 측이 이를 감지하지 못해 영구 stall에 빠지는 운영 이슈다. #37894(MooncakeConnector health check)가 이 문제의 직접적 대응이다. #37638은 Mamba 모델의 heterogeneous TP를 NIXL P/D에서 구현하려는 tracking issue로, conv_states의 `[x | B | C]` 레이아웃에서 x_dim ≠ B_dim이라 단순 flat split이 불가능한 근본적 제약을 다룬다. chunk-interleaved permutation(P측 재배치 + D측 역변환)과 3-read(x/B/C 각각 RDMA) 두 접근이 Nemotron-Nano-30B에서 검증되었다.

#37796은 3/11 포스트의 #36755(preemption 시 `num_cached_tokens` 리셋 누락)와 같은 계열이다. Hybrid 모델에서는 preemption 시 mamba state가 소실되어 `num_computed_tokens=0`으로 재시작하는데, 이전 실행의 `in_progress_prompt_logprobs_cpu`가 남아 있으면 새 계산과 충돌해 livelock이 발생한다. V1 엔진의 async scheduling + hybrid model + preemption 조합이 만드는 경계조건 패밀리가 계속 확장되고 있다.

---

*Sources: [vLLM GitHub](https://github.com/vllm-project/vllm), [SGLang GitHub](https://github.com/sgl-project/sglang)*
