---
layout: post
category: llm-serving
title: "vLLM/SGLang Daily - 2026-03-23"
---

## Highlights

3/12~3/23 기간 동안 vLLM은 세 가지 방향에서 대규모 구조적 변화를 보였다:
**(1) KV Cache 추상화 계층 재설계** — HMA 모델 contiguous allocation, pluggable eviction policy(T-LRU), kv_offload+HMA 시리즈로 connector→eviction→allocation을 일관된 아키텍처로 통합 중이다.
**(2) P/D Disaggregation 운영 안정성** — MooncakeConnector health check, stale callback 처리, Mamba heterogeneous TP 실험 등 장애 복구와 이종 모델 P/D를 정면으로 다루고 있다.
**(3) MoE 메모리 효율** — MLA decode buffer 공유(15 GiB→256 MiB), 런타임 expert pruning(RIY), MoE oracle 통합 등 대규모 MoE 서빙의 VRAM 병목을 운영 레벨에서 해결하고 있다.

## Speculative Decoding

| PR | Repo | Status | 요약 |
|----|------|--------|------|
| [#37880](https://github.com/vllm-project/vllm/pull/37880) | vLLM | Open | `--speculative-config`에 `moe_backend` 필드 추가 — draft/target에 각각 다른 MoE backend(marlin/cutlass 등) 지정 가능 |
| [#37812](https://github.com/vllm-project/vllm/pull/37812) | vLLM | Open | MRV2 warmup에서 spec decode 경로를 고려하도록 수정 |
| [#21207](https://github.com/sgl-project/sglang/pull/21207) | SGLang | Open | EAGLE multi-step draft에서 DP attention padding state 누출 수정 — ~800req 후 크래시 방지 |
| [#21186](https://github.com/sgl-project/sglang/pull/21186) | SGLang | Merged | Ngram spec decode 3/N: C++ trie insert queue race → condition variable 기반 동기화 |

vLLM #37880은 FP8 target + BF16 draft 같은 heterogeneous quantization 조합에서 MoE backend 불일치 문제를 해결한다.
기존에는 `--moe-backend` 하나로 target과 draft 모두에 적용됐는데, draft 모델이 unquantized인 경우 `cutlass`가 더 적합할 수 있다.
이 변경은 `SpeculativeConfig`에 `moe_backend` 필드를 추가하고, `create_vllm_config_for_draft_model` 경로에서 drafter의 `FusedMoEConfig`만 선택적으로 override한다.
vLLM의 speculative decoding 파이프라인이 "target과 동일한 설정을 상속"하는 모델에서, "draft 고유의 최적화 설정을 독립적으로 제어"하는 모델로 진화하는 신호다.

#37812는 MRV2(Model Runner V2)의 warmup 단계에서 spec decode가 활성화되었을 때 warmup shape에 draft 토큰까지 반영하는 변경이다.
warmup에서 실제 서빙 시 shape을 정확히 커버하지 않으면 CUDA graph cache miss로 성능이 급락하는데,
spec decode의 verification 단계에서 draft_len+1개 토큰이 한 번에 들어오는 패턴이 warmup에 포함되지 않던 것이 원인이다.
SGLang #21207은 EAGLE draft의 multi-step 루프에서 DP attention padding이 `spec_info.accept_length`와
`global_num_tokens_cpu`를 오염시키는 문제를 잡는다. `_forward_raw()` 내 `post_forward_mlp_sync_batch()`가
모든 상태를 복원하지 못하는 것이 원인으로, draft loop 내 명시적 save/restore로 해결한다.
Ngram 시리즈 #21186은 C++ trie의 insert queue race(drain 후 worker가 아직 trie 갱신 중일 때 `synchronize()` 조기 리턴)를
condition variable로 수정하여 프로덕션 안정성을 높인다.

## Distributed KV Cache / KV Connector

| PR/Issue | Repo | Status | 요약 |
|----------|------|--------|------|
| [#37885](https://github.com/vllm-project/vllm/pull/37885) | vLLM | Open | HMA 모델(Gemma3 등)에 contiguous KV cache allocation 확장 — 동일 page size의 multi-group KV를 연속 메모리에 배치, RDMA 단일 전송 가능 |
| [#37874](https://github.com/vllm-project/vllm/pull/37874) | vLLM | Open | `LRUManager`/`ARCManager` → `CPUOffloadingManager` + pluggable `CachePolicy` 통합. eviction 전략을 `_CACHE_POLICIES` 레지스트리 등록만으로 교체 가능 |
| [#37825](https://github.com/vllm-project/vllm/pull/37825) | vLLM | Open | T-LRU(Tail-Optimized LRU) eviction policy — NeurIPS 2025 논문 기반, 대화형 워크로드의 P95/P99 TTFT 최적화. TEL-safe queue로 "evict해도 SLA를 침해하지 않는 tail 블록"을 우선 배출 |
| [#37853](https://github.com/vllm-project/vllm/pull/37853) | vLLM | Open | kv_offload+HMA 시리즈 7/N: hybrid 모델의 `register_kv_caches`에서 `CanonicalKVCaches` 클래스로 고유 tensor set + group별 page size 매핑 정의 |
| [#37859](https://github.com/vllm-project/vllm/pull/37859) | vLLM | Open | P/D 환경에서 request abort 후 도착하는 stale KV transfer callback(`finished_recving`/`finished_sending`)을 skip 처리 — decode 노드 assert 크래시 방지 |
| [#37894](https://github.com/vllm-project/vllm/pull/37894) | vLLM | Open | MooncakeConnector Proxy에 prefill 노드 health check + 자동 failover. `VLLM_HEALTH_CHECK_INTERVAL`로 주기 설정, unhealthy 노드 자동 제외 및 복구 시 재포함 |
| [#37827](https://github.com/vllm-project/vllm/pull/37827) | vLLM | Open | 여러 request가 동일 prefix를 공유할 때 async load에서 race condition 발생 수정 |
| [#21206](https://github.com/sgl-project/sglang/pull/21206) | SGLang | Open | MambaRadixCache/SWARadixCache → `HybridRadixTree V2` 통합. TreeComponent 플러그인으로 새 attention 변종 추가 시 core tree 로직 수정 불필요 |
| [#21125](https://github.com/sgl-project/sglang/pull/21125) | SGLang | Open | HiCache에 draft KV cache L2/L3 backing — target KV load_back 시 draft KV 동기화로 accept length 퇴행 방지 (3.07→6.94) |

이 기간 KV cache 트랙은 vLLM에서 **allocation → eviction → offload → connector를 관통하는 추상화 재설계**가 동시다발적으로 진행되고 있다.

#37885는 `allocate_uniform_kv_caches` 경로를 HMA 모델로 확장하는 multi-phase effort의 첫 단계다.
기존에는 single-group(모든 레이어가 동일한 KV 구조) 모델만 contiguous 할당을 지원해, Gemma3 같은 full attention + sliding window 혼합 모델은 per-layer 할당으로 fallback되어 메모리가 산발적으로 흩어졌다.
이는 NIXL/Mooncake 같은 RDMA connector가 하나의 전송 요청으로 전체 KV를 옮길 수 없게 만든다.
#37885는 page size가 동일한 multi-group KV를 하나의 연속 메모리 영역에 배치해, 향후 #37853의 `CanonicalKVCaches` 추상화와 결합하면 connector가 "그룹별 descriptor + 단일 물리 전송"으로 동작할 수 있게 된다.

#37874와 #37825는 eviction 계층의 재설계다. #37874는 `LRUManager`와 `ARCManager`가 ~40줄의 동일한 boilerplate(take_events, ref-count, allocate_blocks 등)를 중복하고 있던 것을 `CPUOffloadingManager` + `CachePolicy` ABC로 정리한다.
핵심 설계 결정은 eviction을 **atomic**으로 만든 것이다: 기존 코드는 n개를 evict하다 중간에 실패하면 부분 eviction이 발생했는데, 이제 candidate를 먼저 수집하고 모두 충족 시에만 state를 변경한다.
#37825의 T-LRU는 이 pluggable policy 위에 구현된 첫 번째 사례다. 대화형 워크로드에서 request의 conversation history `H` 블록과 예상 다음 쿼리 `Q_hat` 블록을 기반으로 TEL-safe cap `B = max(0, H + Q_hat - xi)`를 계산하고, tail 위치 블록을 `tel_safe_queue`로 우선 배출하여 P95/P99 TTFT를 직접 최적화한다.

P/D disaggregation 운영 측면에서 #37859와 #37894는 3/11 포스트에서 다룬 #37745(prefill 재시작 시 decode stall)의 연장선이다.
#37859는 request가 이미 abort된 후 비동기로 도착하는 KV transfer 완료 콜백이 `self.requests`에서 request를 찾지 못해 assert로 죽던 문제를 "stale callback으로 간주하고 skip"하는 방식으로 수정한다.
#37894는 더 근본적으로, proxy 레벨에서 prefill 노드의 health를 주기적으로 체크하고 장애 시 round-robin에서 자동 제외, 복구 시 재포함하는 메커니즘이다.
두 PR 모두 "P/D 분리 아키텍처에서 노드 간 상태 불일치가 전체 파이프라인을 멈추는 문제"를 서로 다른 레이어에서 방어한다.

## MoE / Large Scale Serving

| PR/Issue | Repo | Status | 요약 |
|----------|------|--------|------|
| [#37799](https://github.com/vllm-project/vllm/pull/37799) | vLLM | Open | MLA decode output buffer를 layer 간 공유 — DeepSeek-R1 61 레이어에서 ~15 GiB → ~256 MiB, DP4 OOM 해결 |
| [#37824](https://github.com/vllm-project/vllm/pull/37824) | vLLM | Open | RIY(Reap It Yourself): JSON 프로파일 기반 런타임 expert pruning. 20% pruning 시 0% 오버헤드, 모니터 모드에서 ~5% |
| [#37879](https://github.com/vllm-project/vllm/pull/37879) | vLLM | Open | `RoutedExpertsCapturer` DP>1 + MK(Modular Kernel) 경로에서 assertion 실패 수정 — local vs total token count 불일치 처리 |
| [#37776](https://github.com/vllm-project/vllm/pull/37776) | vLLM | Open | MoE oracle들을 class 구조로 통합 — `_MOE_ORACLES` 레지스트리로 data-driven oracle 선택 |
| [#37830](https://github.com/vllm-project/vllm/pull/37830) | vLLM | Merged | MRV2에서 PP + CUDA graph 테스트 활성화 |
| [#37818](https://github.com/vllm-project/vllm/pull/37818) | vLLM | Merged | MRV2 piecewise CUDA graph에서 hidden states allocation 생략 |
| [#37865](https://github.com/vllm-project/vllm/pull/37865) | vLLM | Open | cudagraph capture size 상한이 설정 값과 맞지 않던 문제 수정 (257→256 truncation 등) |
| [#21205](https://github.com/sgl-project/sglang/pull/21205) | SGLang | Open | MiniMax M2.5 EPLB 실행 시 `routed_experts_weights_of_layer` 속성 누락 크래시 수정 |
| [#18858](https://github.com/sgl-project/sglang/pull/18858) | SGLang | Merged | Blackwell MXFP4 MoE weight loading ~9.5x 가속 (265s→28s) |

이 기간 대규모 MoE 서빙 트랙의 핵심은 **"VRAM 절감 + 운영 도구 + MRV2 PP 안정화"**다.

#37799는 DeepSeek-R1 같은 61-layer MoE에서 발견된 실전 OOM을 해결한다.
PR #37442에서 CUDA graph padding slot의 NaN 방지를 위해 추가된 per-layer `_decode_out` 버퍼가 DeepSeek-R1의 61 레이어 × ~250 MiB = ~15 GiB를 소비했다.
MLA의 `forward_mqa`는 sequential이므로 layer 간 동시 사용이 없어, module-level singleton으로 공유하면 ~256 MiB로 줄어든다.
이 문제가 `profile_run(is_profile=True)` 경로에서 `forward_mqa`가 호출되지 않아 메모리 프로파일링에도 잡히지 않았다는 점이 운영 관점에서 중요하다.
FlashInferMLAImpl과 CutlassMLAImpl 모두에 동일 패턴을 적용한다.

#37824 RIY(Reap It Yourself)는 MoE의 expert utilization 불균형을 **배포 시점에 제거**하는 접근이다.
모니터 모드(`VLLM_RIY_MONITOR=1`)로 per-expert activation 통계를 수집 → JSON 프로파일을 생성 → 프로덕션에서 `--riy-expert-profile profile.json`로 미사용 expert를 로딩 시 제거한다.
router의 logit mask로 제외하므로 hot-path 오버헤드가 0%이고, `local_num_experts` 자체가 줄어 VRAM을 절감한다.
EP, TP, CUDA Graph, torch.compile, quantized 모델(GPTQ/NVFP4/FP8), MTP/EAGLE spec decode와 모두 호환된다.
Qwen3.5-397B-A17B INT4 TP=2(DGX Spark GB10)에서 20% pruning이 정확도 손실 없이 동작함을 확인했다.

#37879는 DP>1 환경에서 MoE CUDA graph capture의 assertion 실패를 수정한다. `DefaultMoERunner.forward()`에는 두 가지 DP dispatch 경로가 있다:
(1) naive dispatch(전체 DP rank 토큰 concat 후 routing)와 (2) MK(Modular Kernel) path(DP combine이 `quant_method.apply` 내부에서 발생).
MK 경로에서는 `select_experts()`가 local 토큰만 보므로 `cumsum[-1] == topk_ids.shape[0]` assertion이 실패한다.

MRV2의 PP 지원도 꾸준히 진행 중이다. #37830이 PP + CUDA graph 테스트를 활성화하고, #37818이 piecewise CUDA graph에서 불필요한 hidden states allocation을 제거한다.
vLLM의 PP 지원이 "실험적 기능"에서 "CI 테스트가 붙은 검증 완료 기능"으로 넘어가는 과정이다.

## Kernel & Performance

| PR | Repo | Status | 요약 |
|----|------|--------|------|
| [#37813](https://github.com/vllm-project/vllm/pull/37813) | vLLM | Open | GDN의 gating+recurrent+beta 커널을 `causal_conv1d_fn` 내부로 fuse — Qwen3.5-35B-A3B prefill에서 별도 커널 호출 제거 |
| [#37789](https://github.com/vllm-project/vllm/pull/37789) | vLLM | Open | GDN prefill conv output Q/K/V split을 fused rearrange로 전환 — split+contiguous 메모리 복사 제거 |
| [#35777](https://github.com/vllm-project/vllm/pull/35777) | vLLM | Merged | Qwen3 Next용 `fused_sigmoid_gating_delta_rule_update` 커널 |
| [#34206](https://github.com/vllm-project/vllm/pull/34206) | vLLM | Merged | grouped topk 커널 최적화 |
| [#20627](https://github.com/sgl-project/sglang/pull/20627) | SGLang | Merged | GDN packed decode — 6-step을 3-step으로 압축, `fused_recurrent_gated_delta_rule_packed_decode()` 단일 커널 |
| [#21203](https://github.com/sgl-project/sglang/pull/21203) | SGLang | Open | CuTeDSL KDA decode 커널 — Kimi-Linear/Kimi-2.5용 |
| [#21104](https://github.com/sgl-project/sglang/pull/21104) | SGLang | Open | FA3 `scheduler_metadata` 사전 계산 — decode throughput +1.4~2.6%, TPOT -2.6% (BS=1) |

vLLM과 SGLang 모두 **GDN/KDA 계열 hybrid attention의 커널 최적화**에 집중하고 있다.

vLLM에서 머지된 #35777은 Qwen3 Next의 GDN 레이어를 위한 `fused_sigmoid_gating_delta_rule_update` 커널로,
기존에 별도 커널로 수행하던 gate 계산(`g = -exp(A_log)*softplus(a+dt_bias)`)과 beta 계산(`sigmoid(b)`)을
recurrence update와 함께 하나의 커널로 합친다. #37813과 #37789는 같은 GDN의 prefill 경로를 서로 다른 레벨에서 최적화한다:
#37813은 gating/recurrent/beta 계산을 `causal_conv1d_fn` 안으로 fuse하여 별도 커널 호출을 제거하고,
#37789는 conv 출력의 `torch.split()` → `.view()` → implicit `.contiguous()` Q/K/V 분리를 fused rearrange로 전환해 메모리 복사를 없앤다.

SGLang #20627(머지)은 decode 경로에서 더 극적인 최적화를 보여준다. 기존 6-step 경로를
`conv1d → packed decode kernel → transpose` 3-step으로 축소하며, 핵심은 packed QKV 레이아웃(`[B, 2*H*K + HV*V]`)에서
pointer arithmetic으로 Q/K/V를 zero-copy로 읽고, gate와 beta를 레지스터에서 인라인 계산한 뒤
recurrence update와 output write까지 단일 커널에서 처리하는 것이다.
GDN의 비대칭 head 설계(Q/K head 수 `H` ≠ V head 수 `HV`)를 packed layout 내에서 동시 인덱싱하는 방식이 핵심 기술이다.
#21203의 CuTeDSL KDA 커널은 같은 계열을 KDA(Key-Dimension Attention)로 확장한다.

#21104의 FA3 scheduler_metadata 사전 계산은 64-layer 모델에서 63회의 `prepare_varlen_num_blocks` GPU 호출을 제거해
TPOT를 일관되게 ~2% 개선한다. vLLM에서는 이미 적용된 최적화이며, SGLang이 이를 따라가는 형태다.

## Notable Issues & Bugs

| Issue | Repo | Status | 요약 |
|-------|------|--------|------|
| [#37745](https://github.com/vllm-project/vllm/issues/37745) | vLLM | Open | MooncakeConnector: prefill 재시작 후 decode가 old engine_id로 요청 → WAITING_FOR_REMOTE_KVS 영구 stall |
| [#37703](https://github.com/vllm-project/vllm/issues/37703) | vLLM | Open | `TRITON_ATTN`이 `VLLM_KV_CACHE_LAYOUT=HND` 무시 → heterogeneous TP + NIXL P/D에서 KV layout 불일치 |
| [#37638](https://github.com/vllm-project/vllm/issues/37638) | vLLM | Open | Tracking: Mamba Heterogeneous TP for NIXL P/D — conv_states `[x\|B\|C]`에서 x_dim ≠ B_dim이라 flat split 불가, 3가지 접근 비교 실험 |
| [#37729](https://github.com/vllm-project/vllm/issues/37729) | vLLM | Open | V1 engine core가 concurrent load 하에서 deadlock — /health 200이지만 0 token 생성, prefix caching + fp8 + Qwen3.5 조합에서 재현 |
| [#37832](https://github.com/vllm-project/vllm/issues/37832) | vLLM | Open | DeepSeek-R1 FP8에서 `fuse_norm_quant=true`(기본값)가 오히려 느림 — Inductor의 opaque extern이 no-op bf16→f32→bf16 커널을 생성하여 ~10us/layer 낭비 |
| [#37856](https://github.com/vllm-project/vllm/issues/37856) | vLLM | Open | Qwen3.5 MoE에서 Sequence Parallel + EP + TP > 1 + DP > 1 시 shared expert 출력 오류 |
| [#37796](https://github.com/vllm-project/vllm/pull/37796) | vLLM | Open | Hybrid 모델 preemption 후 stale `prompt_logprobs` → mamba state 소실과 결합해 livelock |

이 기간의 vLLM 이슈들은 **"고성능 기능 조합이 만드는 경계조건"**이 공통 패턴이다.

#37745/#37859/#37894는 P/D disaggregation의 장애 복구 경로에서 engine_id 불일치, stale callback, health 미감지라는 세 가지 문제가 동시에 드러난 사례다. MooncakeConnector가 stateless proxy에서 stateful health manager로 진화하는 계기가 되었다.

#37638은 Mamba 모델의 heterogeneous TP를 NIXL P/D에서 구현하려는 tracking issue다.
Attention KV는 모든 head가 동일 크기라 flat split이 가능하지만, Mamba conv_states는 `[x | B | C]`에서 x_dim ≠ B_dim이므로 각 D rank가 필요한 데이터가 비연속적이다. 3가지 접근이 Nemotron-Nano-30B-A3B-FP8에서 검증되었다:
(1) full read + local staging buffer(추가 GPU 메모리 필요), (2) chunk-interleaved permutation(P측 재배치 → 1 RDMA → D측 역변환), (3) 3-read(x/B/C 각각 별도 RDMA). Approach 2와 3이 유력하며, 아직 P_TP > D_TP 케이스는 미구현이다.

#37729는 V1 엔진의 silent deadlock으로, fp8 + prefix caching + Qwen3.5 조합에서 concurrent load 시 /health는 200을 반환하지만 token 생성이 완전히 멈추는 심각한 이슈다.
#37832는 DeepSeek-R1의 Inductor 컴파일 경로에서 norm fusion이 오히려 성능을 저하시키는 문제를 보고한다.
`RMSNormQuantFusionPass`가 `rms_norm_per_block_quant`를 opaque extern으로 삽입하면서,
Inductor가 앞선 `_to_copy`(bf16→f32→bf16 round-trip)를 DCE하지 못해 per-layer ~10us의 no-op 커널이 추가된다.
`fuse_norm_quant=false`로 Inductor가 norm을 직접 decompose하게 하면 28us로 줄어든다 (기본 38us 대비).

#37796은 3/11의 #36755와 같은 "V1 엔진 동시성 모델의 경계조건" 계열이다.
Hybrid 모델에서 preemption 시 mamba state가 소실되어 `num_computed_tokens=0`으로 재시작하는데,
이전 실행의 `in_progress_prompt_logprobs_cpu`가 잔류하면 새 계산과 충돌해 livelock이 발생한다.
이 문제들은 async scheduling + hybrid model + preemption + prefix caching이 동시에 활성화될 때 증폭되므로,
향후 V1 엔진의 상태 전이 모델 자체를 재검토할 필요성을 시사한다.

---

*Sources: [vLLM GitHub](https://github.com/vllm-project/vllm), [SGLang GitHub](https://github.com/sgl-project/sglang)*
