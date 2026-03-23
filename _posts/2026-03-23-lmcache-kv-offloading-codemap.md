---
layout: post
title: "LMCache KV Cache Offloading (1) - Overview"
category: llm-serving
---

> **Note**: 이 포스트는 [vLLM v0.17.1](https://github.com/vllm-project/vllm/tree/v0.17.1) 기준으로 작성되었습니다. 이후 버전에서 API나 내부 구조가 변경되었을 수 있습니다.

70B 모델을 A100 80GB에서 서빙한다고 하자. FP16 파라미터에 ~35GB, activation과 CUDA context에 ~5GB. KV cache에 쓸 수 있는 GPU 메모리는 ~40GB인데, 긴 context가 몇 개만 들어와도 이 40GB는 순식간에 찬다. 이때 선택지는 세 가지다 — preemption으로 요청을 쫓아내거나(TTFT 폭증), 더 비싼 GPU를 쓰거나(비용 폭증), KV cache를 GPU 밖으로 빼거나.

세 번째 선택지가 KV cache offloading이다. 개념은 직관적이다. 당장 attention에 참여하지 않는 KV를 CPU DRAM이나 disk로 내려놓고, 다시 필요할 때 올린다. 그런데 이걸 serving engine에 실제로 구현하려면 즉시 부딪히는 문제가 있다: **granularity mismatch**. vLLM의 paged KV buffer는 16-token 블록 단위로 GPU 메모리를 관리하는 반면, LMCache는 256-token chunk 단위로 외부 storage를 관리한다. Scheduler가 "이 prefix는 외부 cache에 있다"고 판단해서 블록을 할당해도, worker가 실제로 KV를 꺼내오려면 chunk 경계에 맞춰 token mask를 만들어야 한다.

더 흥미로운 문제는 timing이다. KV cache를 CPU에서 GPU로 올리는 건 PCIe bandwidth에 묶인 느린 작업이다. Naive하게 forward pass 전에 모든 레이어의 KV를 한번에 올리면, 그 시간 동안 GPU compute는 놀게 된다. 하지만 LLM은 layer 0, 1, 2, ... 순서대로 KV에 접근한다는 걸 우리는 **사전에** 안다. 이 예측 가능성을 이용하면 "layer N의 attention 연산"과 "layer N+1의 KV transfer"를 overlap시킬 수 있다 — LMCache의 layerwise pipelining이 정확히 이걸 한다.

이 포스트는 vLLM 0.17.1의 LMCache connector 코드를 multi-tiered KV cache offloading 관점에서 추적한다. Scheduler의 2-phase cache lookup에서 시작해서, worker의 레이어별 KV load/save까지, 실행 흐름을 코드 레벨에서 따라간다.

* toc
{:toc}

---

## 1. vLLM KV Cache 관리: Baseline

LMCache가 어디에 끼어드는지를 이해하려면, 먼저 vLLM이 KV cache를 어떻게 관리하는지 알아야 한다.

### 1.1 Paged KV Buffer

vLLM은 서버 시작 시 GPU에 고정 크기의 paged KV buffer를 pre-allocate한다. 이 buffer는 `block_size`(기본 16) 토큰 단위의 physical block 배열이다. 각 physical block은 모든 레이어의 K, V 텐서를 담는다.

요청이 들어오면 `KVCacheManager`가 필요한 physical block을 할당하고, **block_table**(logical block → physical block 매핑)을 관리한다. 같은 prefix를 공유하는 요청들은 physical block을 공유할 수 있다(prefix caching).

<div class="mermaid-wide">
flowchart LR
    subgraph A["Req A: 48 tok"]
        direction LR
        LA0["L0"]
        LA1["L1"]
        LA2["L2"]
    end
    subgraph B["Req B: 32 tok"]
        direction LR
        LB0["L0"]
        LB1["L1"]
    end

    LA0 -->|block_table| P1
    LA1 --> P3
    LA2 --> P5
    LB0 -.-> P1
    LB1 -.-> P3

    subgraph GPU["GPU Paged KV Buffer"]
        direction LR
        P0["#0 free"]
        P1["#1"]
        P2["#2 free"]
        P3["#3"]
        P4["#4 free"]
        P5["#5"]
    end

    classDef shared fill:#4a90d9,stroke:#333,color:#fff
    classDef exclusive fill:#e67e22,stroke:#333,color:#fff
    classDef free fill:#eee,stroke:#999,color:#999
    classDef req fill:#fff,stroke:#333

    class P1,P3 shared
    class P5 exclusive
    class P0,P2,P4 free
    class LA0,LA1,LA2,LB0,LB1 req
</div>

> 파랑 = prefix 공유 block (ref_count=2), 주황 = Req A 전용, 회색 = free. 점선 = 공유 참조.
> 각 physical block shape: `[num_layers, 2(K,V), block_size, num_kv_heads, head_dim]`

### 1.2 Scheduler의 스케줄링 사이클

Scheduler의 핵심 결정은 **"이 요청의 토큰 중 이미 KV cache가 존재하는 부분은 어디까지인가"**이다. 이미 KV가 있는 토큰은 forward pass에서 다시 계산할 필요가 없으므로, `num_computed_tokens`가 스케줄링의 핵심 변수가 된다.

<div class="mermaid-wide">
sequenceDiagram
    participant S as Scheduler
    participant KVM as KVCacheManager
    participant HT as Prefix Cache<br/>(hash tree)
    participant W as Worker

    Note over S: 새 요청 도착: 1000 tokens

    S->>KVM: get_computed_blocks(request)
    KVM->>HT: hash(token_ids) → prefix lookup
    HT-->>KVM: 앞 512 tokens의 블록이 GPU에 존재
    KVM-->>S: num_computed_tokens = 512

    Note over S: 512개 토큰은 KV가 이미 GPU에 있음 → skip<br/>488개 토큰만 forward pass 필요

    S->>KVM: allocate_slots(request, num_computed=512)
    Note over KVM: 기존 32개 block 재사용 (ref_count++)<br/>488 토큰 → 새 physical block 31개 할당
    KVM-->>S: block_table, slot_mapping

    S->>S: SchedulerOutput 빌드
    Note over S: input_tokens = tokens[512:] (새 488개만)<br/>slot_mapping = 새 K,V를 쓸 GPU 위치<br/>block_table = 전체 K,V를 읽을 블록 목록

    S->>W: SchedulerOutput
</div>

핵심은 **`slot_mapping`과 `block_table`의 역할 분리**다:
- **`slot_mapping`**: 새로 계산한 K, V를 **어디에 쓸지** (write 경로)
- **`block_table`**: attention 연산 시 전체 K, V를 **어디서 읽을지** (read 경로)

### 1.3 Worker: 레이어별 KV Read/Write

Worker는 SchedulerOutput을 받아 forward pass를 실행한다. 새 토큰에 대해서만 Q, K, V를 계산하고, 각 attention 레이어에서 다음이 일어난다:

<div class="mermaid-wide">
flowchart LR
    A["hidden_states<br/>(새 토큰만)"] --> B["Q, K, V =<br/>linear(hidden)"]
    B --> C["Write: slot_mapping<br/>위치에 새 K,V 저장"]
    C --> D["Read: block_table로<br/>전체 K,V 읽기"]
    D --> E["FlashAttention<br/>(Q, K_all, V_all)"]
    E --> F["→ 다음 레이어"]

    style C fill:#e67e22,stroke:#333,color:#fff
    style D fill:#4a90d9,stroke:#333,color:#fff
    style E fill:#27ae60,stroke:#333,color:#fff
</div>

이 과정이 layer 0부터 마지막 레이어까지 순차적으로 반복된다. 모든 레이어가 같은 paged KV buffer를 공유하지만, 레이어별로 다른 offset에 접근한다.

### 1.4 Preemption: Baseline의 한계

GPU 블록은 유한하다. 새 요청에 블록을 할당하려는데 여유 블록이 없으면, scheduler는 **preemption**을 수행한다 — 낮은 우선순위 요청의 GPU 블록을 강제로 해제하여 새 요청에게 넘긴다.

<div class="mermaid-wide">
sequenceDiagram
    participant S as Scheduler
    participant KVM as KVCacheManager
    participant W as Worker

    Note over S: 요청 B 도착, GPU 블록 부족

    S->>KVM: can_allocate(request_B)?
    KVM-->>S: False (여유 블록 없음)

    rect rgba(255, 150, 150, 0.2)
    Note over S,KVM: Preemption
    S->>S: 낮은 우선순위 요청 A 선택
    S->>KVM: free_blocks(request_A)
    Note over KVM: 요청 A의 GPU 블록 해제<br/>KV 데이터 유실
    S->>S: 요청 A → waiting queue로 이동
    end

    S->>KVM: allocate_slots(request_B)
    KVM-->>S: 성공
    S->>W: SchedulerOutput (요청 B)

    Note over S: ... 시간 경과 ...

    rect rgba(255, 200, 150, 0.2)
    Note over S,W: 요청 A 재스케줄
    S->>KVM: get_computed_blocks(request_A)
    KVM-->>S: num_computed = 0 (KV 전부 유실)
    Note over S: 1000 토큰 전체를 처음부터 recompute
    S->>W: SchedulerOutput (요청 A, 전체 재계산)
    end
</div>

이것이 baseline의 근본적 한계다. 32k context 요청이 preempt되면 수만 토큰의 KV를 **처음부터** 다시 계산해야 하고, 그동안 GPU는 이미 한 번 했던 연산을 반복하느라 다른 요청을 처리하지 못한다. Preemption이 잦은 환경에서는 유효 throughput이 크게 떨어진다.

### 1.5 LMCache가 확장하는 지점

LMCache는 위의 baseline 흐름에 두 가지를 추가하여 preemption의 한계를 해결한다.

**확장 ①: Scheduler에 외부 cache 조회 추가**

Baseline에서 scheduler는 GPU prefix cache만 확인했다 (1-phase lookup). LMCache는 여기에 **외부 cache 조회**를 추가한다 (2-phase lookup). GPU에 없더라도 CPU/disk에 KV가 있으면 해당 토큰의 recompute를 건너뛸 수 있다.

단, 외부에서 찾은 KV도 attention 연산을 위해 결국 GPU에 올라와야 하므로, `allocate_slots()`에서 해당 토큰분의 **빈 GPU physical block을 미리 할당**해둬야 한다. 이 빈 block은 이후 worker의 `start_load_kv()`가 CPU/disk에서 KV를 load하여 채운다.

**확장 ②: Worker에서 매 forward pass마다 KV를 외부로 저장**

이것이 핵심 설계 결정이다. LMCache가 있으면 worker는 **매번** 새로 계산한 KV를 CPU/disk에 저장한다. Preemption이 발생하든 안 하든 관계없이, 모든 forward pass 결과가 외부에 남는다 — **write-through cache** 모델.

이 두 확장이 합쳐지면 preemption 문제가 해결된다:

<div class="mermaid-wide">
sequenceDiagram
    participant S as Scheduler
    participant KVM as KVCacheManager
    participant LC as LMCache Connector
    participant W as Worker
    participant EXT as CPU/Disk

    rect rgba(200, 255, 200, 0.15)
    Note over S,EXT: Step 1: 요청 A의 첫 번째 실행
    S->>KVM: allocate_slots(request_A)
    S->>W: SchedulerOutput
    Note over W: forward pass — 1000 토큰의 KV 계산
    W->>LC: save_kv_layer() (매 레이어)
    LC->>EXT: KV → CPU 저장 (write-through)
    Note over EXT: 요청 A의 KV가 CPU에 보존됨
    end

    rect rgba(255, 150, 150, 0.2)
    Note over S,KVM: Step 2: Preemption 발생
    S->>KVM: free_blocks(request_A)
    Note over KVM: GPU 블록 해제, 하지만...
    Note over EXT: CPU에 KV가 이미 존재!
    end

    rect rgba(200, 220, 255, 0.15)
    Note over S,EXT: Step 3: 요청 A 재스케줄
    S->>KVM: get_computed_blocks(request_A)
    KVM-->>S: num_local = 0 (GPU에는 없음)

    S->>LC: get_num_new_matched_tokens(req_A, 0)
    LC-->>S: ext_tokens = 1000 (CPU에 전부 있음!)

    Note over S: total_computed = 1000<br/>forward pass 불필요 — CPU → GPU 복원만 하면 됨

    S->>KVM: allocate_slots(request_A, 1000)
    Note over KVM: 1000 토큰분 빈 블록 할당
    S->>W: SchedulerOutput + LoadSpec

    W->>LC: start_load_kv()
    LC->>EXT: retrieve()
    EXT-->>W: CPU → GPU 빈 블록에 KV 복원
    Note over W: recompute 없이 바로 decode 재개
    end
</div>

Baseline에서는 **preemption = KV 유실 = recompute**였다. LMCache에서는 **preemption = GPU 블록만 해제 = CPU에서 복원**이다. Recompute 대신 CPU → GPU memcpy만 하면 되므로, 복구 비용이 연산 비용에서 전송 비용으로 바뀐다.

### 1.6 흔한 오해: "GPU에서 evict할 때 CPU로 내리는 것 아닌가?"

KV cache offloading이라는 이름을 들으면, OS의 swap처럼 "GPU 메모리가 부족할 때 block을 CPU로 내리고, 다시 필요하면 올린다"는 그림을 떠올리기 쉽다. 하지만 **LMCache는 그렇게 동작하지 않는다.**

<div class="mermaid-wide">
flowchart LR
    subgraph wrong["일반적인 상상 (OS swap 모델)"]
        direction LR
        W1["GPU block 부족"] --> W2["evict 대상 선택"] --> W3["GPU → CPU 복사"] --> W4["GPU block 해제"]
    end

    subgraph right["LMCache 실제 동작 (write-through 모델)"]
        direction LR
        R1["forward pass 완료"] --> R2["save_kv_layer():<br/>GPU → CPU 복사<br/>(매번, 무조건)"] --> R3["... 시간 경과 ..."] --> R4["GPU block 부족<br/>→ LRU evict<br/>→ GPU block 해제<br/>(CPU 복사 없음)"]
    end

    style W3 fill:#c0392b,stroke:#333,color:#fff
    style R2 fill:#27ae60,stroke:#333,color:#fff
    style R4 fill:#e67e22,stroke:#333,color:#fff
</div>

차이를 정리하면:

| | OS swap / OffloadingConnector | LMCache |
|---|---|---|
| **저장 시점** | GPU block을 evict할 때 (reactive) | 매 forward pass에서 (proactive) |
| **evict 시 동작** | GPU → CPU 복사 후 GPU 해제 | GPU 해제만 (이미 CPU에 있음) |
| **CPU에도 없는 경우** | 해당 없음 (evict 시 항상 복사) | recompute fallback |

**그런데 CPU 메모리도 유한하다.** `save_kv_layer()`가 매번 KV를 CPU에 쓰면, 언젠가 LMCache가 할당받은 CPU 메모리(`max_local_cpu_size`)도 가득 찬다. 이때 LMCache 자체의 LRU eviction이 발생한다 — 오래된 chunk를 버리고 새 chunk를 저장한다.

<div class="mermaid-wide">
sequenceDiagram
    participant W as Worker
    participant LMC as LMCacheEngine
    participant CPU as CPU Buffer<br/>(max_local_cpu_size)

    Note over W: forward pass 완료, 새 KV 저장

    W->>LMC: save_kv_layer(layer_name, kv)

    alt CPU 여유 공간 있음
        LMC->>CPU: store(new_chunk)
        Note over CPU: [A][B][C][new] ← 추가
    else CPU 가득 참
        Note over CPU: [A][B][C][D] ← 꽉 참
        LMC->>CPU: evict(A) — LRU chunk 제거
        LMC->>CPU: store(new_chunk)
        Note over CPU: [B][C][D][new] ← A 유실
    end

    Note over LMC: chunk A는 GPU에서도 이미 evict된 상태일 수 있음<br/>→ 양쪽 다 없으면 다음 lookup 시 miss → recompute
</div>

이것이 **두 계층의 독립적 eviction** 문제다. GPU prefix cache와 LMCache CPU cache는 서로의 상태를 모른 채 각자 LRU eviction을 수행한다:

1. **GPU의 eviction 대상은 prefix cache의 재사용 대기 block**이다 — 현재 실행 중인 요청의 block은 active reference가 있어 절대 evict되지 않는다.
2. **CPU의 eviction 대상은 LMCache의 가장 오래된 chunk**이다 — `save_kv_layer()`로 새 chunk를 저장할 때 공간이 부족하면 LRU chunk를 버린다.
3. **양쪽 다 없으면 recompute로 fallback한다** — GPU miss + LMCache miss이면 `num_computed_tokens = 0`이 되어 baseline과 동일하게 전체 recompute한다. [`kv_load_failure_policy`](https://github.com/vllm-project/vllm/blob/v0.17.1/vllm/config/kv_transfer.py)가 이 경로를 제어한다 ([PR #26813](https://github.com/vllm-project/vllm/pull/26813)).

즉 LMCache는 **best-effort cache**다. 두 계층(GPU prefix cache, LMCache CPU/disk) 간에 inclusive/exclusive 보장 같은 건 없고, 각자 독립적으로 eviction을 수행한다. `max_local_cpu_size`를 충분히 크게 잡으면 CPU eviction 빈도는 낮아지지만, 완전히 없앨 수는 없다.

### 1.7 Automatic Prefix Caching과 LMCache: 두 개의 독립적 해싱

섹션 1.5의 2-phase lookup에서 "local prefix cache 조회 → LMCache 외부 조회" 흐름을 봤다. 여기서 자연스러운 질문이 생긴다: 두 cache가 같은 해싱을 쓰는가? 겹치면 어떻게 되는가?

**답: 완전히 독립적인 해싱이다.**

| | vLLM APC (GPU local) | LMCache (CPU/disk external) |
|---|---|---|
| **해싱 단위** | `block_size` (16 tokens) | `chunk_size` (256 tokens) |
| **해시 알고리즘** | SHA-256/xxhash (CBOR 직렬화) | 자체 prefix-hash chain |
| **해시 체인** | `hash(block_i) = hash(parent_hash, token_ids, extra_keys)` | `hash(chunk_i) = hash(parent_hash, chunk_tokens)` |
| **해시 시드** | `NONE_HASH` (프로세스별 랜덤) | deterministic (cross-instance 공유 가능) |
| **저장 위치** | `BlockPool.cached_block_hash_to_block` (in-memory dict) | `StorageManager` backends (CPU/disk/remote) |

vLLM의 APC는 16-token block 단위로 해시를 체이닝하고, LMCache는 256-token chunk 단위로 독립적인 해시를 체이닝한다. 같은 prefix라도 해시값 자체는 다르다.

**2-phase lookup에서 겹침 방지**

Local APC가 token 0-200을 hit하고, LMCache가 token 0-700을 hit하는 상황을 생각하자:

```python
# vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/vllm_v1_adapter.py
def get_num_new_matched_tokens(self, request, num_computed_tokens):
    # num_computed_tokens = 200 (local APC hit)
    num_external_hit_tokens = self.lookup_client.lookup(token_ids)  # = 700
    need_to_allocate = num_external_hit_tokens - num_computed_tokens  # = 500
    return need_to_allocate  # 500 (local과 겹치는 200은 제외)
```

LMCache가 반환하는 700에서 local hit 200을 **빼서** 추가분 500만 반환한다. 이렇게 double-counting을 방지한다.

**Load 시 token masking**

실제로 KV를 load할 때도, local APC에 이미 있는 부분은 skip한다:

```python
# start_load_kv() 내부
token_mask = torch.ones(len(tokens), dtype=torch.bool)
masked_token_count = load_spec.vllm_cached_tokens // chunk_size * chunk_size
token_mask[:masked_token_count] = False  # local에 있는 부분은 LMCache에서 안 가져옴
lmcache_engine.retrieve(tokens, mask=token_mask, ...)
```

**Save 후 local APC 등록**

LMCache가 외부에서 load한 KV가 GPU block에 채워지면, 그 block들은 다음 스케줄링 사이클에서 vLLM의 정상적인 `cache_blocks()` 호출을 통해 **local APC에도 등록**된다. LMCache가 직접 APC를 업데이트하는 게 아니라, vLLM의 기존 메커니즘이 자동으로 처리한다.

결과적으로 한번 LMCache에서 load된 KV는 이후 **local APC hit**로 승격되어, 같은 prefix가 다시 오면 LMCache 조회 없이 GPU에서 바로 서빙된다.

---

## 2. KVConnectorBase_V1: 추상 인터페이스

[`KVConnectorBase_V1`](https://github.com/vllm-project/vllm/blob/v0.17.1/vllm/distributed/kv_transfer/kv_connector/v1/base.py)은 vLLM의 **KV cache 이동**을 위한 통합 추상 클래스다 ([PR #15960](https://github.com/vllm-project/vllm/pull/15960)). 이 인터페이스는 두 가지 용도를 동시에 지원한다:

- **KV Cache Offloading** — GPU ↔ CPU/Disk 간 KV 이동 (LMCache, OffloadingConnector)
- **P/D Disaggregation** — Prefill instance ↔ Decode instance 간 KV 전송 (NIXL, P2P NCCL, Mooncake)

같은 추상 클래스를 공유하기 때문에, offloading connector와 disaggregation connector 모두 동일한 scheduler/worker hook에 끼어든다. `KVConnectorRole` enum으로 역할이 구분되며, 같은 클래스의 인스턴스가 scheduler와 worker에 각각 하나씩 생성된다.

아래 다이어그램에서 **주황 점선 박스가 이 포스트에서 다루는 multi-tiered KV cache offloading 관련 API**다. 박스 밖의 메서드(`request_finished`, `take_events`, `register_kv_caches`, `get_finished`)는 offloading과 disaggregation 양쪽에서 공통으로 사용된다.

<div class="mermaid-wide">
flowchart LR
    subgraph Scheduler["Scheduler-side"]
        direction TB
        subgraph sched_offload[" KV Cache Offloading 핵심 API "]
            S1["get_num_new_matched_tokens()"]
            S2["update_state_after_alloc()"]
            S3["build_connector_meta()"]
        end
        S4["request_finished()"]
        S5["take_events()"]
    end

    subgraph Worker["Worker-side"]
        direction TB
        subgraph worker_offload[" KV Cache Offloading 핵심 API "]
            W2["start_load_kv()"]
            W3["wait_for_layer_load()"]
            W4["save_kv_layer()"]
            W5["wait_for_save()"]
        end
        W1["register_kv_caches()"]
        W6["get_finished()"]
    end

    Scheduler ====> Worker

    style sched_offload stroke-dasharray:5 5,stroke:#e67e22,fill:transparent
    style worker_offload stroke-dasharray:5 5,stroke:#e67e22,fill:transparent
    style Scheduler fill:#eaf0f9,stroke:#4a90d9
    style Worker fill:#fdf2e9,stroke:#e67e22
</div>

### 2.1 Scheduler-side 메서드

```python
# 외부 KV cache에서 hit되는 토큰 수 조회
def get_num_new_matched_tokens(
    self, request: Request, num_computed_tokens: int
) -> tuple[int | None, bool]:
    # Returns: (매칭 토큰 수 or None, async 로딩 여부)

# 블록 할당 후 connector 상태 업데이트
def update_state_after_alloc(
    self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
):

# 현재 step의 connector metadata 생성 (worker에게 전달)
def build_connector_meta(
    self, scheduler_output: SchedulerOutput
) -> KVConnectorMetadata:

# 요청 완료 시 호출 — 블록 해제 또는 비동기 전송 결정
def request_finished(
    self, request: Request, block_ids: list[int]
) -> tuple[bool, dict | None]:

# KV cache event 발행 (llm-d 등 외부 시스템용)
def take_events(self) -> Iterable[KVCacheEvent]:
```

### 2.2 Worker-side 메서드

```python
# KV cache 텐서 등록 (1회)
def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):

# 비동기 KV 로딩 시작 (forward pass 전)
def start_load_kv(self, forward_context: ForwardContext, **kwargs):

# 특정 레이어 로딩 완료 대기 (attention layer 내부)
def wait_for_layer_load(self, layer_name: str):

# 특정 레이어 KV 저장 (attention layer 내부)
def save_kv_layer(self, layer_name: str, kv_layer, attn_metadata, **kwargs):

# 모든 저장 완료 대기 (forward pass 후)
def wait_for_save(self):

# 비동기 전송 완료된 요청 ID 반환
def get_finished(self, finished_req_ids: set[str]) -> tuple[set | None, set | None]:
```

---

## 3. LMCacheConnectorV1Impl: Core 구현 ([source](https://github.com/vllm-project/vllm/blob/v0.17.1/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/vllm_v1_adapter.py), [PR #25542](https://github.com/vllm-project/vllm/pull/25542))

### 3.1 주요 데이터 구조

```python
# vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/vllm_v1_adapter.py

@dataclass
class LoadSpec:
    vllm_cached_tokens: int        # vLLM local prefix cache에서 hit된 토큰 수
    lmcache_cached_tokens: int     # LMCache에서 hit된 토큰 수 (local 포함)
    can_load: bool = False         # 블록 할당 완료 후 True

@dataclass
class SaveSpec:
    skip_leading_tokens: int       # 이미 저장된 토큰 수 (skip할 prefix)
    can_save: bool = True          # 저장 실행 여부

@dataclass
class RequestTracker:
    req_id: str
    prompt_len: int
    token_ids: list[int]           # 현재까지의 토큰 ID
    allocated_block_ids: list[int] # 할당된 vLLM 블록 ID
    num_saved_tokens: int          # LMCache에 저장 완료된 토큰 수
    mm_hashes: list[str] | None    # multimodal hash
    mm_positions: list | None
    skip_save: bool = False
    is_decode_phase: bool = False

@dataclass
class ReqMeta:
    req_id: str
    token_ids: list[int]
    slot_mapping: torch.Tensor     # vLLM paged KV buffer의 slot mapping
    is_last_prefill: bool = False
    save_spec: SaveSpec | None = None
    load_spec: LoadSpec | None = None
```

### 3.2 초기화 (Scheduler vs Worker)

```python
class LMCacheConnectorV1Impl:
    def __init__(self, vllm_config, role, parent):
        config = lmcache_get_or_create_config()  # LMCACHE_CONFIG_FILE or default
        # extra_config의 "lmcache.*" 키를 LMCache config에 반영
        for key, value in kv_connector_extra_config.items():
            if key.startswith("lmcache."):
                _validate_and_set_config_value(config, key[8:], value)

        if role == KVConnectorRole.SCHEDULER:
            # LookupClient 생성 — token hash 기반 외부 캐시 조회
            self.lookup_client = LookupClientFactory.create_lookup_client(...)
        else:
            # Worker-side: LMCacheEngine + LookupServer + ZMQOffloadServer
            self.lmcache_engine = _init_lmcache_engine(config, vllm_config)
            self.lookup_server = LookupClientFactory.create_lookup_server(...)
            self.offload_server = ZMQOffloadServer(...)
```

---

## 4. 실행 흐름: Scheduler Side ([`scheduler.py`](https://github.com/vllm-project/vllm/blob/v0.17.1/vllm/v1/core/sched/scheduler.py))

<div class="mermaid-wide">
flowchart LR
    A["Request"] --> B["① get_computed_blocks<br/>prefix cache 조회"]
    B --> D["② get_num_new_matched_tokens<br/>LMCache 외부 조회"]
    D --> F["③ allocate_slots<br/>(total_computed)"]
    F --> G["④ update_state_after_alloc<br/>can_load = True"]
    G --> H["⑤ build_connector_meta<br/>LoadSpec + SaveSpec"]
    H --> J["→ Worker"]

    style B fill:#4a90d9,stroke:#333,color:#fff
    style D fill:#e67e22,stroke:#333,color:#fff
    style H fill:#27ae60,stroke:#333,color:#fff
</div>

### 4.1 Connector 생성

```python
# vllm/v1/core/sched/scheduler.py:119-131 (source)
self.connector = KVConnectorFactory.create_connector(
    config=self.vllm_config,
    role=KVConnectorRole.SCHEDULER,
    kv_cache_config=self.kv_cache_config,
)
```

### 4.2 요청 스케줄링 — 2단계 prefix cache 조회

```python
# vllm/v1/core/sched/scheduler.py:596-640

# Step 1: vLLM local prefix cache 조회
new_computed_blocks, num_new_local_computed_tokens = (
    self.kv_cache_manager.get_computed_blocks(request)
)

# Step 2: LMCache 외부 캐시 조회
ext_tokens, load_kv_async = (
    self.connector.get_num_new_matched_tokens(
        request, num_new_local_computed_tokens
    )
)

# 합산
num_computed_tokens = num_new_local_computed_tokens + ext_tokens
```

`get_num_new_matched_tokens` 내부에서는:
1. token ID에 multimodal hash 적용
2. `self.lookup_client.lookup(token_ids, ...)` 호출 — LMCache backend(CPU/disk/remote)에서 연속 prefix 매칭
3. `LoadSpec(vllm_cached_tokens, lmcache_cached_tokens, can_load=False)` 저장

### 4.3 블록 할당 후 상태 업데이트

```python
# vllm/v1/core/sched/scheduler.py:746-749
self.connector.update_state_after_alloc(
    request,
    self.kv_cache_manager.get_blocks(request_id),
    num_external_computed_tokens,
)
```

`update_state_after_alloc` 내부에서 `LoadSpec.can_load = True`로 전환.

### 4.4 Connector Metadata 빌드

```python
# vllm/v1/core/sched/scheduler.py:898-901
meta = self.connector.build_connector_meta(scheduler_output)
scheduler_output.kv_connector_metadata = meta
```

`build_connector_meta` 내부에서:
1. 새 요청 → `RequestTracker` 생성 → `ReqMeta` 생성 (LoadSpec/SaveSpec 포함)
2. 기존 요청 → `RequestTracker.update()` → 필요시 `ReqMeta` 생성
3. `LMCacheConnectorMetadata(requests=[...])` 반환

`ReqMeta.from_request_tracker`에서 SaveSpec 결정 로직:

```python
skip_save = (
    tracker.skip_save
    or (num_saved_tokens > 0 and input_token_len < chunk_boundary)  # chunk 경계 미달
    or (is_decode_phase and not save_decode_cache)                   # decode phase
    or request_skip                                                  # per-request skip
)
```

---

## 5. 실행 흐름: Worker Side

### 5.1 Connector 초기화 ([`kv_transfer_state.py`](https://github.com/vllm-project/vllm/blob/v0.17.1/vllm/distributed/kv_transfer/kv_transfer_state.py))

```python
# kv_transfer_state.py:59-73
def ensure_kv_transfer_initialized(vllm_config, kv_cache_config=None):
    global _KV_CONNECTOR_AGENT
    if vllm_config.kv_transfer_config.is_kv_transfer_instance \
       and _KV_CONNECTOR_AGENT is None:
        _KV_CONNECTOR_AGENT = KVConnectorFactory.create_connector(
            config=vllm_config,
            role=KVConnectorRole.WORKER,
            kv_cache_config=kv_cache_config,
        )
```

### 5.2 Forward Pass Lifecycle ([PR #21980](https://github.com/vllm-project/vllm/pull/21980))

[`KVConnectorModelRunnerMixin._get_kv_connector_output`](https://github.com/vllm-project/vllm/blob/v0.17.1/vllm/v1/worker/kv_connector_model_runner_mixin.py)가 전체 lifecycle을 관리한다:

<div class="mermaid-wide">
flowchart LR
    A["bind_connector<br/>_metadata"] --> B["start_load_kv<br/>(비동기 로딩)"]
    B --> C["yield:<br/>Forward Pass"]
    C --> D["wait_for_save<br/>(저장 완료 대기)"]
    D --> E["get_finished"]
    E --> F["clear_connector<br/>_metadata"]

    style B fill:#e67e22,stroke:#333,color:#fff
    style C fill:#27ae60,stroke:#333,color:#fff
    style D fill:#4a90d9,stroke:#333,color:#fff
</div>

```python
# vllm/v1/worker/kv_connector_model_runner_mixin.py:87-116
@contextmanager
def _get_kv_connector_output(scheduler_output, wait_for_save=True):
    output = KVConnectorOutput()
    kv_connector = get_kv_transfer_group()

    # 1. Scheduler가 만든 metadata를 bind
    kv_connector.bind_connector_metadata(scheduler_output.kv_connector_metadata)

    # 2. 비동기 KV 로딩 시작
    kv_connector.start_load_kv(get_forward_context())

    try:
        yield output                    # 3. model forward pass 실행
    finally:
        if wait_for_save:
            kv_connector.wait_for_save()  # 4. 모든 저장 완료 대기

        output.finished_sending, output.finished_recving = (
            kv_connector.get_finished(scheduler_output.finished_req_ids)
        )
        output.invalid_block_ids = kv_connector.get_block_ids_with_load_errors()
        kv_connector.clear_connector_metadata()
```

### 5.3 Attention Layer 내부: @maybe_transfer_kv_layer ([PR #27816](https://github.com/vllm-project/vllm/pull/27816))

<div class="mermaid-wide">
flowchart LR
    A["unified_attention()"] --> B{"kv_transfer_group?"}
    B -- No --> C["attention만 실행"]
    B -- Yes --> D["wait_for_layer_load"]
    D --> E["attention 연산"]
    E --> F["save_kv_layer"]

    style D fill:#e67e22,stroke:#333,color:#fff
    style E fill:#27ae60,stroke:#333,color:#fff
    style F fill:#4a90d9,stroke:#333,color:#fff
</div>

```python
# kv_transfer_utils.py (source)
def maybe_transfer_kv_layer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
            return func(*args, **kwargs)

        layer_name = args[layer_name_index]
        attn_metadata, _, kv_cache, _ = get_attention_context(layer_name)
        connector = get_kv_transfer_group()

        connector.wait_for_layer_load(layer_name)   # 이 레이어 KV 로딩 완료 대기
        result = func(*args, **kwargs)               # attention 연산
        connector.save_kv_layer(layer_name, kv_cache, attn_metadata)  # KV 저장
        return result
    return wrapper
```

이 decorator가 적용된 함수들:
- `unified_attention()` — attention.py:615
- `unified_attention_with_output()` — attention.py:685
- MLA attention의 `mla_unified_attention()` 등 — mla_attention.py:834, 921

### 5.4 start_load_kv 상세

```python
def start_load_kv(self, forward_context, **kwargs):
    for request in metadata.requests:
        if request.load_spec is None:
            continue

        token_mask = torch.ones(len(tokens), dtype=torch.bool)
        masked_token_count = (
            load_spec.vllm_cached_tokens
            // self._lmcache_chunk_size * self._lmcache_chunk_size
        )
        token_mask[:masked_token_count] = False  # local hit 부분 마스킹

        if self.use_layerwise:
            # 레이어별 파이프라이닝: retrieve_layer() → generator
            layerwise_retriever = self.lmcache_engine.retrieve_layer(
                tokens[:lmcache_cached_tokens],
                token_mask[:lmcache_cached_tokens],
                kvcaches=kvcaches,
                slot_mapping=slot_mapping[:lmcache_cached_tokens],
            )
            next(layerwise_retriever)  # 첫 2개 레이어 prefetch
            next(layerwise_retriever)
        else:
            # 전체 레이어 한번에 retrieve
            self.lmcache_engine.retrieve(
                tokens[:lmcache_cached_tokens],
                token_mask[:lmcache_cached_tokens],
                kvcaches=kvcaches,
                slot_mapping=slot_mapping[:lmcache_cached_tokens],
            )
```

### 5.5 wait_for_save 상세 (non-layerwise)

```python
def wait_for_save(self):
    for request in connector_metadata.requests:
        if save_spec is None or not save_spec.can_save:
            continue

        # chunk 경계에 정렬
        skip_leading_tokens = (
            save_spec.skip_leading_tokens
            // self._lmcache_chunk_size * self._lmcache_chunk_size
        )

        store_mask = torch.ones(len(token_ids), dtype=torch.bool)
        store_mask[:skip_leading_tokens] = False

        self.lmcache_engine.store(
            token_ids, mask=store_mask,
            kvcaches=kvcaches, slot_mapping=slot_mapping,
            offset=skip_leading_tokens,
        )
```

---

## 6. LMCache 내부: store/retrieve는 어떻게 동작하는가

섹션 5에서 vLLM connector가 `start_load_kv()`, `save_kv_layer()`, `wait_for_save()`를 호출하는 것까지 봤다. 이 호출들은 결국 `LMCacheEngine`의 5개 API로 이어진다:

| LMCacheEngine API | vLLM Connector 호출자 | 용도 |
|---|---|---|
| `lookup()` | LookupServer ← `get_num_new_matched_tokens()` | cache hit 토큰 수 조회. 데이터 이동 없이 `batched_contains()`만 실행 |
| `retrieve()` | `start_load_kv()` (non-layerwise) | 외부 storage → GPU paged KV buffer로 KV 로드 |
| `retrieve_layer()` | `start_load_kv()` (layerwise) | 위와 동일하나 generator 기반 — 레이어별 파이프라이닝 |
| `store()` | `wait_for_save()` (non-layerwise) | GPU paged KV buffer → 외부 storage로 KV 저장 |
| `store_layer()` | `save_kv_layer()` (layerwise) | 위와 동일하나 generator 기반 — 레이어별 파이프라이닝 |

`lookup()`은 Scheduler 프로세스에서 LookupClient → ZMQ → LookupServer → `lmcache_engine.lookup()` 경로로 호출된다. 나머지 4개는 Worker 프로세스에서 직접 호출한다.

### 6.1 Component 구조

<div class="mermaid-wide">
flowchart LR
    subgraph vllm["vLLM (Connector Layer)"]
        direction TB
        SCHED["Scheduler<br/>(LMCacheConnectorV1Impl)"]
        WORKER["Worker<br/>(LMCacheConnectorV1Impl)"]
    end

    subgraph lmcache["LMCache"]
        direction TB

        LC["LookupClient<br/>(scheduler-side)"]

        subgraph worker_components["Worker-side"]
            LS["LookupServer"]
            ZMQ_OFF["ZMQOffloadServer"]
            ENGINE["LMCacheEngine"]
            TDB["TokenDatabase<br/>(prefix-hash → chunk key)"]
            GPU_CONN["GPUConnector<br/>(CUDA kernel)"]
        end

        subgraph storage["StorageManager"]
            direction TB
            subgraph cpu_path[" CPU 경유 (기본) "]
                direction LR
                CPU_BE["LocalCPUBackend<br/>(pinned memory pool)"]
                REMOTE["RemoteBackend<br/>(Redis/Valkey)"]
                DISK["LocalDiskBackend"]
            end
            subgraph direct_path[" CPU bypass (GPU Direct) "]
                direction LR
                GDS["GdsBackend<br/>(GPU ↔ NVMe)"]
                NIXL["NixlStorageBackend<br/>(GPU ↔ GPU, RDMA)"]
            end
        end

        BMS["BatchedMessageSender<br/>(admit/evict 통지)"]
    end

    subgraph external["External"]
        CC["Cache Controller<br/>(cross-node 관리)"]
    end

    SCHED -->|"lookup()"| LC
    LC -.->|ZMQ| LS
    LS --> ENGINE
    CC -.->|"P2P lookup 응답"| LS

    WORKER -.->|"ZMQ IPC"| ZMQ_OFF
    ZMQ_OFF -->|"store()"| ENGINE
    ENGINE --> TDB
    ENGINE --> GPU_CONN
    ENGINE --> storage

    CPU_BE -->|"evict/admit"| BMS
    BMS -->|ZMQ| CC

    style ENGINE fill:#4a90d9,stroke:#333,color:#fff
    style CPU_BE fill:#e67e22,stroke:#333,color:#fff
    style GPU_CONN fill:#27ae60,stroke:#333,color:#fff
    style cpu_path fill:#fff8e1,stroke:#e67e22,stroke-dasharray:5 5
    style direct_path fill:#e8f5e9,stroke:#27ae60,stroke-dasharray:5 5
    style GDS fill:#27ae60,stroke:#333,color:#fff
    style NIXL fill:#27ae60,stroke:#333,color:#fff
    style CC fill:#9b59b6,stroke:#333,color:#fff
</div>

핵심 구성 요소:

| Component | 역할 |
|-----------|------|
| **`LookupClient`** | Scheduler-side. `get_num_new_matched_tokens()` 호출 시 ZMQ로 LookupServer에 cache hit 조회 |
| **`LookupServer`** | Worker-side. LookupClient의 요청을 받아 `LMCacheEngine.lookup()` 실행 후 결과 반환 |
| **`ZMQOffloadServer`** | Worker-side. vLLM의 multi-process worker 아키텍처에서 LMCacheEngine이 있는 프로세스와 다른 프로세스의 offload 요청을 ZMQ IPC로 수신하여 `store()` 실행. LMCacheEngine에 직접 접근할 수 없는 프로세스가 KV 저장을 위임할 때 사용 |
| **`LMCacheEngine`** | 중앙 허브. store/retrieve/lookup의 진입점 |
| **`TokenDatabase`** | token ID 시퀀스 → 256-token chunk 단위의 `CacheEngineKey`(prefix hash chain) 변환 |
| **`GPUConnector`** | GPU paged KV buffer ↔ CPU/Storage 간 데이터 이동. CUDA kernel(`lmc_ops`) 사용 |
| **`StorageManager`** | 여러 storage backend를 관리. put/get을 모든 backend에 분배 |
| **`LocalCPUBackend`** | 항상 생성되는 primary store + allocator. pinned CPU memory pool에서 `MemoryObj` 할당 |
| **`GdsBackend`** | GPU Direct Storage — GPU ↔ NVMe SSD를 CPU bypass로 직접 DMA |
| **`NixlStorageBackend`** | RDMA 기반 — GPU ↔ 원격 GPU/Storage를 CPU bypass로 직접 전송 |
| **`BatchedMessageSender`** | CPU backend의 admit/evict 이벤트를 batching하여 Cache Controller에 전달 |
| **`Cache Controller`** | 별도 프로세스. 어떤 worker가 어떤 chunk를 갖고 있는지 전역 인덱스를 관리. P2P lookup 시 "이 chunk는 worker #3에 있다"는 정보를 제공하고, worker 등록/해제와 cross-node 데이터 이동을 조율 |

### 6.2 Store 흐름

`save_kv_layer()` → `LMCacheEngine.store()`가 호출되면, 설정된 backend에 따라 데이터 경로가 달라진다:

<div class="mermaid-wide">
sequenceDiagram
    participant V as vLLM Worker
    participant E as LMCacheEngine
    participant T as TokenDatabase
    participant G as GPUConnector
    participant SM as StorageManager
    participant CPU as LocalCPUBackend
    participant DRAM as CPU DRAM<br/>(pinned memory)
    participant SSD as NVMe SSD
    participant GDS as GdsBackend
    participant BMS as BatchedMsg<br/>Sender

    V->>E: store(token_ids, slot_mapping, kvcaches)

    E->>T: process_tokens(token_ids)
    Note over T: 256-token chunk로 분할<br/>각 chunk에 prefix hash 계산<br/>→ CacheEngineKey 목록
    T-->>E: keys[]

    E->>SM: allocate(keys)
    SM->>CPU: allocate from pool
    alt pool 가득 참
        CPU->>DRAM: LRU evict → 메모리 반환
        CPU->>BMS: add_kv_op(EVICT, old_key)
    end
    CPU-->>SM: MemoryObj[] (빈 텐서)

    E->>G: batched_from_gpu(slot_mapping, MemoryObj[])
    Note over G: CUDA kernel (D2H):<br/>GPU paged KV → CPU pinned memory
    G->>DRAM: KV 데이터 복사
    G-->>E: MemoryObj[] (채워짐)

    E->>SM: batched_put(keys, MemoryObj[])

    rect rgba(255, 200, 100, 0.15)
    Note over CPU,DRAM: 🟠 LocalCPUBackend (기본)
    SM->>CPU: submit_put(key, MemoryObj)
    Note over DRAM: CPU DRAM에 보존
    CPU->>BMS: add_kv_op(ADMIT, key)
    end

    rect rgba(100, 150, 255, 0.15)
    Note over CPU,SSD: 🔵 LocalDiskBackend (활성 시)
    SM->>SSD: async write(key, data)
    Note over SSD: CPU DRAM → NVMe SSD (비동기)
    end

    rect rgba(100, 200, 100, 0.15)
    Note over GDS,SSD: 🟢 GdsBackend (활성 시, CPU bypass)
    SM->>GDS: submit_put(key, data)
    GDS->>SSD: GPU → NVMe 직접 DMA<br/>(cuFile, CPU 미경유)
    end
</div>

주목할 점:
- 🟠 **LocalCPUBackend** (기본, 항상 활성) — GPU → CPU DRAM. `LocalCPUBackend`가 allocator 역할을 겸하므로 항상 CPU를 거침
- 🔵 **LocalDiskBackend** (선택) — CPU DRAM → NVMe SSD. CPU를 거쳐서 SSD에 저장
- 🟢 **GdsBackend** (선택, CPU bypass) — GPU → NVMe SSD 직접 DMA. `cuFile` API로 CPU bounce buffer 없이 전송
- **CPU 할당과 eviction이 store 시점에 발생**한다 — 섹션 1.6에서 설명한 CPU eviction이 바로 여기서 일어남

### 6.3 Retrieve 흐름

`start_load_kv()` → `LMCacheEngine.retrieve()`가 호출되면, 데이터가 어느 backend에 있느냐에 따라 CPU를 거칠 수도, 건너뛸 수도 있다:

<div class="mermaid-wide">
sequenceDiagram
    participant V as vLLM Worker
    participant E as LMCacheEngine
    participant T as TokenDatabase
    participant SM as StorageManager
    participant DRAM as CPU DRAM<br/>(pinned memory)
    participant CPU as LocalCPUBackend
    participant SSD as NVMe SSD
    participant GDS as GdsBackend
    participant NIXL as NixlBackend
    participant G as GPUConnector

    V->>E: retrieve(token_ids, token_mask, slot_mapping, kvcaches)

    E->>T: process_tokens(token_ids)
    T-->>E: keys[]

    E->>SM: batched_contains(keys)
    Note over SM: 각 backend에 순서대로 확인:<br/>LocalCPU → Disk → GDS → NIXL
    SM-->>E: block_mapping (key → backend)

    E->>SM: batched_get(keys, block_mapping)

    rect rgba(255, 200, 100, 0.15)
    Note over CPU,DRAM: 🟠 LocalCPU hit
    SM->>CPU: get_blocking(key)
    CPU->>DRAM: read
    DRAM-->>SM: MemoryObj
    SM-->>E: MemoryObj[]
    E->>G: batched_to_gpu(MemoryObj[], slot_mapping)
    Note over G: CUDA kernel (H2D):<br/>CPU DRAM → GPU paged KV
    end

    rect rgba(100, 150, 255, 0.15)
    Note over SSD: 🔵 LocalDisk hit (CPU 경유)
    SM->>SSD: read(key)
    SSD->>DRAM: NVMe → CPU DRAM
    DRAM-->>SM: MemoryObj
    SM->>CPU: write-back (다음 접근 시 local hit)
    SM-->>E: MemoryObj[]
    E->>G: batched_to_gpu → GPU
    end

    rect rgba(100, 200, 100, 0.15)
    Note over GDS,SSD: 🟢 GDS hit (CPU bypass)
    SM->>GDS: get(key)
    GDS->>SSD: NVMe → GPU 직접 DMA<br/>(cuFile, CPU 미경유)
    GDS-->>SM: GPU에 직접 로드 완료
    end

    rect rgba(200, 150, 255, 0.15)
    Note over NIXL: 🟣 NIXL hit (CPU bypass)
    SM->>NIXL: get(key)
    Note over NIXL: RDMA: 원격 GPU → local GPU<br/>(CPU 미경유)
    NIXL-->>SM: GPU에 직접 로드 완료
    end

    G-->>V: GPU KV buffer에 KV 복원 완료
</div>

주목할 점:
- 🟠 **LocalCPU hit** — CPU DRAM → CUDA kernel(H2D) → GPU. 가장 빠른 경로 (PCIe bandwidth)
- 🔵 **LocalDisk hit** — NVMe SSD → CPU DRAM → GPU. CPU를 거침. write-back으로 다음 접근 시 🟠 경로로 승격
- 🟢 **GDS hit** — NVMe SSD → GPU 직접 DMA. `cuFile` API로 CPU bypass
- 🟣 **NIXL hit** — 원격 GPU → local GPU RDMA. 네트워크를 타지만 CPU bypass
- **계층적 lookup 우선순위**: LocalCPU → Disk → GDS → NIXL. 가까운 tier에서 hit되면 느린 tier는 skip

### 6.4 GPU-Native Primitives

위의 store/retrieve 흐름에서 `GPUConnector`가 "CUDA kernel로 복사"한다고만 설명했는데, 실제 구현은 상당히 정교하다. vLLM의 paged KV buffer와 LMCache의 contiguous chunk 사이의 **format 변환**, **async stream 관리**, **pinned memory 할당**, **layerwise double-buffering**이 모두 GPU-native primitive 위에 구축되어 있다.

#### KV Format 변환: paged ↔ contiguous

vLLM은 KV cache를 block 단위로 흩어서 저장하지만(paged), LMCache는 chunk 단위로 연속 저장한다(contiguous). 이 변환이 store/retrieve의 핵심이다.

<div class="mermaid-wide">
flowchart LR
    subgraph gpu["GPU — vLLM Paged KV Buffer"]
        direction TB
        subgraph layer0["Layer 0: [2, num_blocks, block_size, num_kv_heads, head_dim]"]
            direction LR
            B0["Block #1<br/>tok 0-15"]
            B1["Block #3<br/>tok 16-31"]
            B2["Block #6<br/>tok 32-47"]
            B3["..."]
        end
        subgraph layer1["Layer 1: 같은 구조, 다른 physical block"]
            direction LR
            B4["Block #2"]
            B5["Block #5"]
            B6["Block #8"]
            B7["..."]
        end
    end

    subgraph kernel["CUDA Kernel<br/>multi_layer_kv_transfer"]
        direction TB
        SM["slot_mapping<br/>scatter / gather"]
    end

    subgraph cpu["CPU — LMCache Contiguous Chunk"]
        direction TB
        subgraph chunk["[2, num_layers, chunk_size, hidden_dim]"]
            direction LR
            C0["K layer0<br/>tok 0-255"]
            C1["K layer1<br/>tok 0-255"]
            C2["V layer0<br/>tok 0-255"]
            C3["V layer1<br/>tok 0-255"]
        end
    end

    gpu -->|"D2H: store"| kernel
    kernel -->|"contiguous 변환"| cpu
    cpu -->|"H2D: retrieve"| kernel
    kernel -->|"paged scatter"| gpu

    style kernel fill:#27ae60,stroke:#333,color:#fff
    style SM fill:#27ae60,stroke:#333,color:#fff
</div>

> 왼쪽: block이 물리적으로 흩어져 있음 (block #1, #3, #6...). 오른쪽: 256 토큰이 연속 배치.
> `slot_mapping`이 "token N → physical slot M" 매핑을 제공하여, kernel이 흩어진 block에서 올바른 위치를 scatter/gather한다.

이 format 변환은 별도 단계가 아니라, **GPU ↔ CPU 데이터 이동과 동시에** 한번의 kernel launch로 수행된다. 호출 경로:

```
Store (D2H):
  save_kv_layer() / wait_for_save()
    → LMCacheEngine.store()        → GPUConnector.batched_from_gpu()
    → LMCacheEngine.store_layer()  → GPUConnector.batched_from_gpu()  (per layer)
      → lmc_ops.multi_layer_kv_transfer(D2H)   # non-layerwise: 전체 레이어 한번에
      → lmc_ops.single_layer_kv_transfer(D2H)  # layerwise: 레이어별

Retrieve (H2D):
  start_load_kv()
    → LMCacheEngine.retrieve()        → GPUConnector.batched_to_gpu()
    → LMCacheEngine.retrieve_layer()  → GPUConnector.batched_to_gpu()  (per layer)
      → lmc_ops.multi_layer_kv_transfer(H2D)   # non-layerwise
      → lmc_ops.single_layer_kv_transfer(H2D)  # layerwise
```

이 CUDA kernel의 핵심 로직:

```cpp
// Grid: (num_tokens, num_layers, k_or_v_size) — token × layer × K/V 3차원 병렬화
__global__ void load_and_reshape_multi_layer_kernel(
    scalar_t* key_value,           // LMCache contiguous buffer
    scalar_t** key_value_ptrs,     // vLLM per-layer KV cache pointers
    int* slot_mapping,             // token → physical slot 매핑
    ...
) {
    int slot_idx = slot_mapping[token_id];
    if (slot_idx < 0) return;  // prefix-cached token → skip

    int block_idx = slot_idx / block_size;
    int block_offset = slot_idx % block_size;

    // direction에 따라 복사 방향 결정:
    // D2H (store): paged[block_idx][block_offset] → contiguous[token_id]
    // H2D (retrieve): contiguous[token_id] → paged[block_idx][block_offset]
}
```

핵심 최적화:
- **Widening copy** — element 크기에 따라 `int64_t`, `int32_t`, `int16_t`, `int8_t` 중 가장 넓은 타입으로 복사하여 memory throughput 극대화
- **Negative slot skip** — `slot_mapping < 0`인 토큰(이미 GPU prefix cache에 있는 토큰)은 kernel 내부에서 즉시 return
- **6종 GPUKVFormat 지원** — vLLM Flash Attention, Flash Infer, MLA, SGLang MHA/MLA 등 각각의 메모리 layout을 compile-time template으로 분기

Layerwise 경로에서는 `lmc_ops.single_layer_kv_transfer()`가 한 레이어씩 동일한 작업을 수행한다.

#### CUDA Streams: store와 retrieve의 비동기 분리

GPUConnector는 **두 개의 전용 CUDA stream**을 생성하여 store/retrieve를 default compute stream과 분리한다:

```python
self.store_stream = torch.cuda.Stream()  # GPU → CPU (D2H)
self.load_stream = torch.cuda.Stream()   # CPU → GPU (H2D)
```

<div class="mermaid-wide">
sequenceDiagram
    participant CS as Default Stream<br/>(attention 연산)
    participant LS as load_stream<br/>(H2D)
    participant SS as store_stream<br/>(D2H)

    Note over LS: retrieve 시:
    LS->>LS: multi_layer_kv_transfer(H2D)
    LS->>CS: load_stream.synchronize()<br/>또는 current_stream.wait_stream(load_stream)
    Note over CS: attention 연산 시작

    Note over SS: store 시:
    CS->>SS: store_stream.wait_stream(current_stream)<br/>(attention 완료 대기)
    SS->>SS: multi_layer_kv_transfer(D2H)
    SS->>SS: store_stream.synchronize()
    Note over SS: CPU에서 MemoryObj 사용 가능
</div>

이렇게 분리하면 **attention 연산과 KV transfer가 서로 다른 CUDA stream에서 overlap**될 수 있다. 특히 layerwise 모드에서는 "layer N의 attention"과 "layer N+1의 KV load"가 동시에 진행된다.

#### Pinned Memory: Lazy Allocation

CPU-side 버퍼는 `cudaHostAlloc`으로 할당한 **pinned memory**를 사용한다. Pinned memory는 OS가 page-out하지 않으므로 DMA 전송 시 bounce buffer 없이 GPU와 직접 통신할 수 있다.

문제는 수십 GB의 pinned memory를 서버 시작 시 한번에 할당하면 **startup latency가 수 초~수십 초**에 달한다는 점이다. LMCache는 이를 `LazyMemoryAllocator`로 해결한다:

1. 전체 크기(`max_local_cpu_size`)만큼 unpinned memory를 `mmap`으로 즉시 예약
2. 초기에는 일부(`init_size`)만 `cudaHostRegister`로 pinning
3. **Background daemon thread**가 64MB 단위로 점진적으로 추가 pinning
4. 1GB마다 `AddressManager.sbrk()`를 호출하여 allocator에 새 주소 공간 공개

이 방식으로 서버는 즉시 시작하고, pinning은 백그라운드에서 점진적으로 완료된다.

NUMA-aware 할당도 지원한다: `mbind(MPOL_BIND)` + first-touch로 GPU에 가까운 NUMA 노드에 메모리를 배치하여 PCIe latency를 최소화한다.

#### Layerwise Double-Buffering (Ping-Pong)

`VLLMBufferLayerwiseGPUConnector`의 retrieve 경로는 **GPU 버퍼 2개를 ping-pong**하며 3-stage 파이프라인을 구성한다:

<div class="mermaid-wide">
sequenceDiagram
    participant CPU as CPU DRAM
    participant LB as load_buffer<br/>(GPU)
    participant CB as compute_buffer<br/>(GPU)
    participant PG as Paged KV<br/>(GPU)

    Note over CPU,PG: Layer 0
    CPU->>LB: H2D copy (load_stream)
    Note over LB: load_buffer에 layer 0 데이터

    Note over CPU,PG: Layer 1
    LB->>CB: swap (load ↔ compute)
    CPU->>LB: H2D copy layer 1 (load_stream)
    Note over CB: RoPE re-encoding on layer 0

    Note over CPU,PG: Layer 2
    CB->>PG: scatter to paged KV (layer 0)
    LB->>CB: swap
    CPU->>LB: H2D copy layer 2
    Note over CB: RoPE on layer 1

    Note over CPU,PG: Layer N — 정상 상태
    Note over LB,CB: 3-stage pipeline:<br/>load[N] + RoPE[N-1] + scatter[N-2]
</div>

정상 상태에서 매 layer마다 세 작업이 동시에 진행된다:
- **load_buffer**: CPU → GPU H2D copy (layer N)
- **compute_buffer**: RoPE re-encoding (layer N-1)
- **paged KV**: scatter write (layer N-2)

`torch.cuda.synchronize()`로 swap point를 동기화하고, 버퍼를 교대(`compute_buffer, load_buffer = load_buffer, compute_buffer`)한다.

#### cuFile (GDS) & NIXL (RDMA)

**GDS** (`GdsBackend`): GPU 메모리를 `cuFileBufRegister()`로 등록한 뒤, `cuFile.write(gpu_addr, nbytes)` / `cuFile.read(gpu_addr, nbytes)`로 GPU ↔ NVMe SSD 간 직접 DMA를 수행한다. CPU의 page cache를 거치지 않으므로 CPU memory bandwidth를 소비하지 않는다. `tmpfs`/`overlayfs`에서는 자동으로 disable되고, `wekafs`에서는 thread pool을 통한 parallel I/O를 사용한다.

**NIXL** (`NixlStorageBackend`): `NixlAgent`를 생성하고 메모리를 `register_memory(mem_type="VRAM")`로 RDMA에 등록한다. Transfer descriptor를 미리 준비(`prep_xfer_dlist`)한 뒤, `transfer(handle)` → `check_xfer_state()` polling으로 비동기 전송을 수행한다. GPU VRAM을 직접 등록하므로 CPU를 거치지 않고 원격 노드와 통신한다.

#### CUDA 의존 지점 요약

LMCache가 사용하는 CUDA-specific primitive를 맥락별로 정리하면:

**1. KV Transfer Kernels** — store/retrieve의 핵심. paged ↔ contiguous 변환 + D2H/H2D 복사를 한 kernel에서 수행.

```cpp
// csrc/mem_kernels.cu — non-layerwise: 전체 레이어를 한번에 전송
lmc_ops.multi_layer_kv_transfer(
    key_value,         // LMCache contiguous buffer [2, L, T, D]
    key_value_ptrs,    // vLLM per-layer KV cache pointer 배열
    slot_mapping,      // [num_tokens] token → physical slot
    paged_memory_device,
    page_buffer_size,
    direction,         // TransferDirection::H2D or D2H
    gpu_kv_format,     // 6종 format 중 하나 (compile-time template)
    block_size
);

// csrc/mem_kernels.cu — layerwise: 한 레이어씩 전송 (generator 내부)
lmc_ops.single_layer_kv_transfer(
    lmc_key_value_cache,   // LMCache per-layer buffer
    vllm_key_value_cache,  // vLLM 해당 레이어 KV cache
    slot_mapping,
    direction,
    gpu_kv_format
);
```

**2. Pinned Memory 할당** — GPU DMA가 bounce buffer 없이 직접 접근하기 위한 host memory.

```cpp
// csrc/mem_alloc.cpp — 표준 pinned 할당
void* alloc_pinned_ptr(size_t size, unsigned int flags) {
    void* ptr;
    cudaHostAlloc(&ptr, size, flags);  // cudaHostAllocDefault
    return ptr;
}

// csrc/mem_alloc.cpp — NUMA-aware: GPU에 가까운 노드에 배치
void* alloc_pinned_numa_ptr(size_t size, int numa_node) {
    void* ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, ...);
    mbind(ptr, size, MPOL_BIND, &numa_mask, ...);  // NUMA 바인딩
    first_touch(ptr, size);                          // 물리 페이지 확보
    cudaHostRegister(ptr, size, cudaHostRegisterDefault);  // pinned 등록
    return ptr;
}
```

```python
# lmcache/v1/lazy_memory_allocator.py — Lazy pinning: background에서 점진적 등록
class LazyMemoryAllocator:
    def _expand_worker(self):
        while self._committed < self._final_size:
            chunk = PIN_CHUNK_SIZE  # 64 MB
            cudaHostRegister(ptr + offset, chunk, cudaHostRegisterMapped)
            self._committed += chunk
            if self._committed % (1 << 30) == 0:  # 매 1GB
                self._commit_expansion()  # allocator에 새 주소 공간 공개
```

**3. CUDA Streams** — store/retrieve를 attention 연산과 비동기로 분리.

```python
# lmcache/v1/gpu_connector/gpu_connectors.py
class VLLMPagedMemGPUConnectorV2:
    def __init__(self):
        self.store_stream = torch.cuda.Stream()  # D2H 전용
        self.load_stream = torch.cuda.Stream()   # H2D 전용

    def from_gpu(self, ...):  # store path
        current_stream = torch.cuda.current_stream()
        self.store_stream.wait_stream(current_stream)  # attention 완료 대기
        with torch.cuda.stream(self.store_stream):
            lmc_ops.multi_layer_kv_transfer(..., direction=D2H)
        self.store_stream.synchronize()  # CPU에서 MemoryObj 접근 전 완료 보장

    def to_gpu(self, ...):  # retrieve path
        with torch.cuda.stream(self.load_stream):
            lmc_ops.multi_layer_kv_transfer(..., direction=H2D)
        self.load_stream.synchronize()
```

```python
# layerwise connector — cross-stream 의존성
class VLLMPagedMemLayerwiseGPUConnector:
    def batched_to_gpu(self, ...):  # generator
        for layer in range(num_layers):
            with torch.cuda.stream(self.load_stream):
                lmc_ops.single_layer_kv_transfer(...)
            current_stream = torch.cuda.current_stream()
            current_stream.wait_stream(self.load_stream)  # load 완료 후 attention 시작
            yield  # caller가 attention 수행
```

**4. GPU Memory** — intermediate buffer, raw pointer 추출.

```python
# GPU intermediate buffer (double-buffering용)
gpu_buffer = torch.empty(
    [2, num_tokens, hidden_dim],
    dtype=kv_dtype, device="cuda"
)

# CUDA kernel에 raw pointer 전달
key_value_ptrs = torch.tensor(
    [kv_cache[layer].data_ptr() for layer in range(num_layers)],
    dtype=torch.uint64, device="cuda"
)
```

**5. cuFile (GDS)** — GPU ↔ NVMe 직접 DMA.

```python
# lmcache/v1/storage_backend/gds_backend.py
class CuFileMemoryAllocator(GPUMemoryAllocator):
    def __init__(self, ...):
        super().__init__(...)  # torch.empty(device="cuda")
        cuFileBufRegister(self.base_pointer, self.total_size)  # GPU buffer를 GDS에 등록

class GdsBackend:
    def _save_gds(self, gpu_addr, path, offset, nbytes):
        f = cufile.CuFile(path, "r+", use_direct_io=True)
        f.write(gpu_addr, nbytes, file_offset=offset)  # GPU → NVMe 직접 DMA

    def _load_gds(self, gpu_addr, path, offset, nbytes):
        f = cufile.CuFile(path, "r", use_direct_io=True)
        f.read(gpu_addr, nbytes, file_offset=offset)   # NVMe → GPU 직접 DMA
```

**6. NIXL (RDMA)** — GPU VRAM 직접 등록 후 원격 전송.

```python
# lmcache/v1/storage_backend/nixl_storage_backend.py
class NixlStorageAgent:
    def __init__(self, ...):
        self.nixl_agent = NixlAgent(...)
        self.nixl_agent.create_backend(backend_type, params)

    def init_mem_handlers(self, buffer):
        # GPU VRAM을 RDMA에 직접 등록
        reg_list = [(buffer.data_ptr(), buffer.nbytes)]
        self.nixl_agent.register_memory(reg_list, mem_type="VRAM")
        self.xfer_descs = self.nixl_agent.get_xfer_descs(...)
        self.handle = self.nixl_agent.prep_xfer_dlist(...)

    def post_blocking(self, handle):
        self.nixl_agent.transfer(handle)
        while self.nixl_agent.check_xfer_state(handle) != "DONE":
            pass  # polling
```

---

## 7. 전체 실행 흐름

<div class="mermaid-wide">
sequenceDiagram
    participant S as Scheduler
    participant W as Worker
    participant LMC as LMCacheEngine
    participant KV as GPU KV Buffer
    participant EXT as CPU/Disk/Remote

    rect rgba(200, 220, 255, 0.15)
    Note over S: Scheduler Process

    S->>S: ① get_computed_blocks(request)<br/>→ num_local_computed_tokens
    S->>S: ② get_num_new_matched_tokens()<br/>→ lookup_client.lookup(token_ids)<br/>→ LoadSpec(vllm_cached, lmcache_cached)
    S->>S: ③ allocate_slots(total_computed)
    S->>S: ④ update_state_after_alloc()<br/>→ LoadSpec.can_load = True
    S->>S: ⑤ build_connector_meta()<br/>→ LMCacheConnectorMetadata
    S->>W: scheduler_output (KVConnectorMetadata 포함)
    end

    rect rgba(255, 220, 200, 0.15)
    Note over W,EXT: Worker Process

    W->>W: ⑥ bind_connector_metadata(meta)
    W->>LMC: ⑦ start_load_kv()
    LMC->>EXT: retrieve()
    EXT-->>KV: CPU/Disk/Remote → GPU paged KV buffer

    rect rgba(255, 255, 200, 0.15)
    Note over W,KV: ⑧ Model Forward (per layer)
    loop @maybe_transfer_kv_layer
        W->>LMC: wait_for_layer_load(layer_name)
        Note over W: attention_forward(Q, K, V)
        W->>LMC: save_kv_layer(layer_name)
    end
    end

    W->>LMC: ⑨ wait_for_save()
    LMC->>KV: store()
    KV-->>EXT: GPU → CPU/Disk/Remote
    W->>W: ⑩ get_finished(finished_req_ids)
    W->>W: ⑪ clear_connector_metadata()
    end
</div>

---

## 8. LMCache vs OffloadingConnector 비교

[`OffloadingConnector`](https://github.com/vllm-project/vllm/blob/v0.17.1/vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py)는 vLLM에 내장된 CPU offloading 구현이다 ([PR #22595](https://github.com/vllm-project/vllm/pull/22595)). LMCache와의 주요 차이:

| 항목 | LMCacheConnectorV1 | OffloadingConnector |
|------|-------------------|---------------------|
| 의존성 | 외부 `lmcache` 패키지 (또는 번들 impl) | vLLM 내장 |
| Storage 계층 | GPU → CPU → Disk → Remote | GPU → CPU only |
| 블록 크기 | 256 토큰 (LMCache chunk) | vLLM 블록 크기 (16 토큰), cross-layer layout |
| 전송 방식 | LMCache engine (configurable GPU connector) | `cudaMemcpyAsync` DMA |
| Multimodal | 지원 (mm_hashes, mm_positions) | 미지원 |
| CUDA Graph | PIECEWISE (layerwise=True일 때) | PIECEWISE |
| Eviction 정책 | LMCache 자체 관리 (LRU 등) | LRU 또는 ARC |
| 적합한 사용처 | 멀티 계층 캐싱, 인스턴스 간 KV 공유 | 단순 CPU overflow, preemption recovery |

