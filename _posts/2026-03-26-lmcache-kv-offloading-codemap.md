---
layout: post
title: "LMCache KV Cache Offloading (1) - Overview"
category: llm-serving
---

> **Note**: 이 포스트는 [vLLM v0.17.1](https://github.com/vllm-project/vllm/tree/v0.17.1) 기준으로 작성되었습니다. 이후 버전에서 API나 내부 구조가 변경되었을 수 있습니다.

> **Scope**: 이 포스트에서는 **CPU backend (`LocalCPUBackend`)를 사용하는 경우**만 다룹니다. GPU Direct Storage(GDS)를 활용한 SSD offloading 경로는 차후 별도 포스트에서 다룰 예정입니다.

Llama 3 70B로 128K context를 서빙하면, 요청 **하나**의 KV cache가 [~40GB를 차지한다](https://developer.nvidia.com/blog/accelerate-large-scale-llm-inference-and-kv-cache-offload-with-cpu-gpu-memory-sharing/). H100 80GB에서 모델 파라미터를 빼면 요청 1개만으로 GPU가 거의 찬다. 요청이 조금만 쌓여도 preemption이 발생하고, preempt된 128K 요청은 처음부터 KV를 recompute해야 한다.

해결책은 직관적이다 — KV cache를 CPU DRAM이나 disk로 내려놓고, 다시 필요하면 올린다. vLLM의 [KV offloading connector 벤치마크](https://vllm.ai/blog/kv-offloading-connector)에서는 이 방식으로 **TTFT 2~22x 감소, throughput 최대 9x 향상**을 달성했다.

그런데 코드를 들여다보면 생각보다 단순하지 않다. Scheduler가 "이 요청의 prefix가 CPU에 있다"는 걸 어떻게 아는가? vLLM은 16-token block 단위로 GPU를 관리하는데 LMCache는 256-token chunk 단위다 — 이 **granularity mismatch**는 어떻게 처리하는가? Forward pass 중에 레이어별로 KV를 load하면 GPU는 PCIe 전송을 기다리느라 놀지 않는가? `save_kv_layer()`는 정말 매번 실행되는가, 아니면 skip 조건이 있는가?

이 포스트는 vLLM의 LMCache connector 코드를 multi-tiered KV cache offloading 관점에서 추적하며, 이런 질문들에 답한다. Scheduler의 2-phase cache lookup에서 시작해서, LMCacheEngine 내부의 CUDA kernel까지 따라간다.

---

* toc
{:toc}

---

## 1. vLLM KV Cache 관리

LMCache가 어디에 끼어드는지를 이해하려면, 먼저 vLLM이 KV cache를 어떻게 관리하는지 알아야 한다.

### 1.1 Paged KV Buffer

vLLM은 서버 시작 시 GPU에 고정 크기의 paged KV buffer를 pre-allocate한다. 이 buffer는 `block_size`(기본 16) 토큰 단위의 physical block 배열이다. **레이어마다 별도의 KV cache 텐서**가 할당되며, 같은 `block_id`가 모든 레이어에서 공유된다 (block table은 하나, 물리 메모리는 레이어별).

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
> 각 레이어의 KV cache shape: `[2(K,V), num_blocks, block_size, num_kv_heads, head_dim]` — 레이어마다 별도 텐서, block_id는 공유.

### 1.2 Scheduler의 스케줄링 사이클

매 스케줄링 사이클마다 scheduler는 요청의 토큰을 순회하며 KV cache가 이미 존재하는 구간을 확인하고, 그 끝 지점을 `num_computed_tokens`로 기록한다. 이 값 이후의 토큰만 forward pass에서 계산하면 된다.

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

여기서 `slot_mapping`과 `block_table`은 서로 다른 역할을 한다:
- **`slot_mapping`** (`int32[num_new_tokens]`): 새로 계산한 K, V를 **어디에 쓸지** — 각 토큰이 저장될 physical slot index (write 경로)
- **`block_table`** (`int32[max_seq_len // block_size]`): attention 연산 시 전체 K, V를 **어디서 읽을지** — logical block → physical block 매핑 (read 경로)

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

### 1.4 Preemption: Offloading 없이 겪는 문제

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

Offloading이 없으면 preemption은 곧 KV 유실이다. 32k context 요청이 preempt되면 수만 토큰의 KV를 **처음부터** 다시 계산해야 하고, 그동안 GPU는 이미 한 번 했던 연산을 반복하느라 다른 요청을 처리하지 못한다. Preemption이 잦은 환경에서는 유효 throughput이 크게 떨어진다.

### 1.5 LMCache가 확장하는 지점

LMCache는 위 흐름에 두 가지를 추가하여 이 문제를 해결한다.

**확장 ①: Scheduler에 외부 cache 조회 추가**

Offloading이 없으면 scheduler는 GPU prefix cache만 확인한다 (1-phase lookup). LMCache는 여기에 **외부 cache 조회**를 추가한다 (2-phase lookup). GPU에 없더라도 CPU/disk에 KV가 있으면 해당 토큰의 recompute를 건너뛸 수 있다.

단, 외부에서 찾은 KV도 attention 연산을 위해 결국 GPU에 올라와야 하므로, `allocate_slots()`에서 해당 토큰분의 **빈 GPU physical block을 미리 할당**해둬야 한다. 이 빈 block은 이후 worker의 `start_load_kv()`가 CPU/disk에서 KV를 load하여 채운다.

**확장 ②: Worker에서 매 forward pass마다 KV를 외부로 저장**

이것이 핵심 설계 결정이다. LMCache가 있으면 worker는 새로 계산한 KV를 CPU/disk에 저장한다 — **proactive save** 모델. 단, 모든 forward pass에서 무조건 실행되는 것은 아니다. `SaveSpec.can_save`가 다음 조건 중 하나라도 해당하면 skip된다: chunk 경계에 도달하지 못한 경우, decode phase에서 `save_decode_cache=False`인 경우, per-request `skip_save` 플래그가 설정된 경우. 그럼에도 prefill phase에서 chunk 경계를 넘는 대부분의 경우에는 저장이 실행되므로, preemption 복구에 충분한 KV가 외부에 남는다.

이 두 확장이 합쳐지면 preemption 문제가 해결된다:

<div class="mermaid-wide">
sequenceDiagram
    participant S as Scheduler
    participant KVM as KVCacheManager
    participant LC as LMCache Connector
    participant W as Worker
    participant EXT as CPU/Disk

    rect rgba(200, 255, 200, 0.15)
    Note over S,EXT: Step 1: 요청 A의 첫 번째 실행 (1000 토큰)
    S->>KVM: allocate_slots(request_A)
    S->>W: SchedulerOutput + SaveSpec
    Note over W: forward pass — 1000 토큰의 KV 계산

    alt chunk 경계(256배수) 도달
        W->>LC: save_kv_layer() (매 레이어)
        LC->>EXT: KV → CPU 저장
        Note over EXT: 256토큰 단위로 chunk 저장<br/>1000 tok → 3 chunk(768 tok) 저장, 나머지 232 tok은 skip
    else chunk 경계 미달
        Note over W: save skip (저장할 완성된 chunk 없음)
    end
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
    LC-->>S: ext_tokens = 768 (3 chunk분만 CPU에 있음)

    Note over S: total_computed = 768<br/>나머지 232 토큰만 recompute

    S->>KVM: allocate_slots(request_A, 768)
    Note over KVM: 768 토큰분 빈 블록 할당
    S->>W: SchedulerOutput + LoadSpec

    W->>LC: start_load_kv()
    LC->>EXT: retrieve()
    EXT-->>W: CPU → GPU 빈 블록에 768 tok KV 복원
    Note over W: 232 토큰만 recompute 후 decode 재개
    end
</div>

Offloading 없이는 **preemption = KV 유실 = recompute**였다. LMCache에서는 **preemption = GPU 블록만 해제 = CPU에서 복원**이다. Recompute 대신 CPU → GPU memcpy만 하면 되므로, 복구 비용이 연산 비용에서 전송 비용으로 바뀐다.

### 1.6 흔한 오해: "GPU에서 evict할 때 CPU로 내리는 것 아닌가?"

KV cache offloading이라는 이름을 들으면, OS의 swap처럼 "GPU 메모리가 부족할 때 block을 CPU로 내리고, 다시 필요하면 올린다"는 그림을 떠올리기 쉽다. 하지만 **LMCache는 그렇게 동작하지 않는다** — 메모리 압박과 무관하게, 매 forward pass에서 chunk 경계(256 토큰 배수)에 도달하면 proactive하게 CPU에 저장한다.

| | 일반적인 상상 (OS swap) | LMCache 실제 동작 (proactive) |
|---|---|---|
| **저장 시점** | GPU 메모리 부족 시 | 매 forward pass에서 chunk 경계 도달 시 |
| **evict 시 동작** | GPU → CPU 복사 후 GPU 해제 | CPU에 이미 있으므로 GPU만 해제 |
| **CPU에도 없는 경우** | — | LRU eviction으로 유실 가능 → [`kv_load_failure_policy`](https://github.com/vllm-project/vllm/blob/v0.17.1/vllm/config/kv_transfer.py)로 제어 (기본값: `"fail"`) |

**그런데 CPU 메모리도 유한하다.** `save_kv_layer()`가 매번 KV를 CPU에 쓰면, 언젠가 LMCache가 할당받은 CPU 메모리(`max_local_cpu_size`)도 가득 찬다. 이때 LMCache 자체의 LRU eviction이 발생한다 — 오래된 chunk를 버리고 새 chunk를 저장한다.

아래 다이어그램은 이 과정을 보여준다. Worker가 `save_kv_layer()`를 호출하면, LMCacheEngine이 CPU buffer에 저장을 시도한다. 여유 공간이 있으면 그냥 추가하지만, 가득 찬 경우에는 가장 오래된 chunk A를 LRU evict한 뒤 새 chunk를 넣는다. 이때 evict된 chunk A가 GPU prefix cache에서도 이미 evict된 상태라면, 해당 KV는 어디에도 남아있지 않게 된다.

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

    Note over LMC: chunk A가 GPU에서도 evict된 상태라면<br/>→ 양쪽 다 없음 → 다음 lookup 시 miss → recompute
</div>

다이어그램의 `else` 분기에서 chunk A가 CPU에서 evict되었다. 그런데 GPU prefix cache도 별도의 LRU 정책으로 독립적으로 eviction을 수행하고 있고, chunk A에 해당하는 GPU block이 이미 evict된 상태일 수 있다. 두 계층은 서로의 상태를 모르기 때문에 이런 상황을 방지할 수 없다:

- **GPU prefix cache**: 재사용 대기 중인 block 중 LRU를 evict한다. 현재 실행 중인 요청의 block은 active reference가 있어 대상이 되지 않는다.
- **LMCache CPU cache**: `save_kv_layer()` 시점에 공간이 부족하면 가장 오래된 chunk를 evict한다.
- **양쪽 다 miss**: GPU miss + LMCache miss이면 [`kv_load_failure_policy`](https://github.com/vllm-project/vllm/blob/v0.17.1/vllm/config/kv_transfer.py) ([PR #26813](https://github.com/vllm-project/vllm/pull/26813))에 따라 동작이 결정된다. 기본값은 `"fail"`(요청 즉시 실패)이며, `"recompute"`로 설정하면 처음부터 다시 계산한다.

결국 LMCache는 **best-effort cache**다. 두 계층 간에 inclusive/exclusive 보장은 없으며, `max_local_cpu_size`를 충분히 크게 잡으면 CPU eviction 빈도는 낮아지지만 완전히 없앨 수는 없다.

---

## 2. LMCache KV Connector: 인터페이스와 구현

vLLM의 scheduler와 worker는 서로 다른 프로세스에서 돌아간다. Scheduler는 "이 요청의 KV가 CPU에 있다"는 사실을 알고 있지만, 실제로 CPU → GPU memcpy를 실행할 수 있는 건 worker뿐이다. 그러면 scheduler가 내린 결정은 worker에 어떻게 전달되는가? `save_kv_layer()`는 매 forward pass마다 호출될 수 있는데, "지금 이 요청은 저장해야 하고 저 요청은 skip해야 한다"는 판단은 누가 내리는가? LMCache의 256-token chunk와 vLLM의 16-token block 사이의 granularity mismatch는 어느 계층에서 흡수하는가?

이 질문들의 공통점은, 답이 LMCache 코드 안에만 있지 않다는 것이다. vLLM이 외부 KV connector를 끼워넣기 위해 열어둔 **추상 인터페이스** — 어떤 메서드가 어떤 시점에 호출되고, 어떤 정보가 scheduler에서 worker로 흘러가는지 — 를 먼저 이해해야 LMCache의 구현이 읽힌다.

[`KVConnectorBase_V1`](https://github.com/vllm-project/vllm/blob/v0.17.1/vllm/distributed/kv_transfer/kv_connector/v1/base.py)은 vLLM의 **KV cache 이동**을 위한 통합 추상 클래스다 ([PR #15960](https://github.com/vllm-project/vllm/pull/15960)). 이 인터페이스는 두 가지 용도를 동시에 지원한다:

- **KV Cache Offloading** — GPU ↔ CPU/Disk 간 KV 이동 (LMCache, OffloadingConnector)
- **P/D Disaggregation** — Prefill instance ↔ Decode instance 간 KV 전송 (P2P NCCL, Mooncake 등)

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

이 추상 인터페이스를 LMCache가 구현한 것이 [`LMCacheConnectorV1Impl`](https://github.com/vllm-project/vllm/blob/v0.17.1/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/vllm_v1_adapter.py) ([PR #25542](https://github.com/vllm-project/vllm/pull/25542))이다. 하나의 클래스가 `KVConnectorRole`에 따라 scheduler와 worker에서 **완전히 다른 컴포넌트**를 초기화한다:

```python
class LMCacheConnectorV1Impl:
    def __init__(self, vllm_config, role, parent):
        config = lmcache_get_or_create_config()  # LMCACHE_CONFIG_FILE or default
        # extra_config의 "lmcache.*" 키를 LMCache config에 반영
        for key, value in kv_connector_extra_config.items():
            if key.startswith("lmcache."):
                _validate_and_set_config_value(config, key[8:], value)

        if role == KVConnectorRole.SCHEDULER:
            # LookupClient만 생성 — 데이터를 옮기지 않고 hit 수만 조회
            self.lookup_client = LookupClientFactory.create_lookup_client(...)
        else:
            # Worker-side: 실제 KV 데이터를 다루는 heavy 컴포넌트들
            self.lmcache_engine = _init_lmcache_engine(config, vllm_config)
            self.lookup_server = LookupClientFactory.create_lookup_server(...)
            self.offload_server = ZMQOffloadServer(...)
```

**Scheduler 프로세스**에서는 `LookupClient`만 생성한다. 이 client는 ZMQ를 통해 worker의 `LookupServer`에 "이 token sequence의 KV가 외부에 있는가?"를 물어보고, hit된 토큰 수만 반환받는다. 실제 KV 데이터는 건드리지 않으므로 가벼운 RPC 호출이다.

**Worker 프로세스**에서는 세 가지를 생성한다:
- **`LMCacheEngine`** — store/retrieve의 핵심. GPU ↔ CPU/disk 간 실제 KV 데이터를 이동시키는 엔진 (섹션 3에서 상세 설명)
- **`LookupServer`** — scheduler의 LookupClient로부터 ZMQ 요청을 받아 `lmcache_engine.lookup()`을 호출하는 daemon
- **`ZMQOffloadServer`** — vLLM의 multi-process worker 아키텍처에서, LMCacheEngine에 직접 접근할 수 없는 다른 프로세스의 store 요청을 중계

이 분리 덕분에 scheduler는 외부 cache hit 정보만 빠르게 얻어 스케줄링 결정을 내리고, 실제 데이터 이동은 worker에서 비동기로 처리된다.

---

### 2.1 실행 흐름: Scheduler Side

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

#### 2.1.1 Connector 생성

```python
# vllm/v1/core/sched/scheduler.py:119-131 (source)
self.connector = KVConnectorFactory.create_connector(
    config=self.vllm_config,
    role=KVConnectorRole.SCHEDULER,
    kv_cache_config=self.kv_cache_config,
)
```

#### 2.1.2 요청 스케줄링 — 2단계 prefix cache 조회

Scheduler는 매 스케줄링 사이클에서 "이 요청의 토큰 중 KV가 이미 어딘가에 존재하는 부분은 어디까지인가?"를 파악해야 한다. KV connector가 없으면 GPU local prefix cache만 확인하지만, LMCache가 있으면 **2단계로 조회**한다.

**Step 1: GPU local prefix cache 조회**

```python
# vllm/v1/core/sched/scheduler.py:602-604
new_computed_blocks, num_new_local_computed_tokens = (
    self.kv_cache_manager.get_computed_blocks(request)
)
```

`KVCacheManager`가 request의 token hash chain을 순차 탐색하여, GPU에 이미 KV가 존재하는 physical block들을 찾는다. 예를 들어 1000 토큰 요청에서 앞 200 토큰의 block이 GPU prefix cache에 있으면 `num_new_local_computed_tokens = 200`을 반환한다. 이 block들은 이후 `allocate_slots()`에서 ref_count를 올려 재사용된다.

**Step 2: LMCache 외부 cache 조회**

```python
# vllm/v1/core/sched/scheduler.py:607-612
ext_tokens, load_kv_async = (
    self.connector.get_num_new_matched_tokens(
        request, num_new_local_computed_tokens
    )
)
```

두 번째 반환값 `load_kv_async`는 **P/D disaggregation**을 위한 플래그다 ([`base.py` docstring](https://github.com/vllm-project/vllm/blob/v0.17.1/vllm/distributed/kv_transfer/kv_connector/v1/base.py#L417-L450)). P/D 구조에서는 prefill instance가 계산한 KV를 네트워크를 통해 decode instance로 전송하는데, 이 전송이 완료되기까지 여러 step이 걸릴 수 있다. **Decode instance의 scheduler**가 이 플래그를 보고 "KV가 아직 도착하지 않았으니 forward pass를 하지 말라"고 판단한다 — `num_new_tokens = 0`으로 설정하여 블록만 할당해두고, KV 수신이 완료된 다음 사이클에서 decode를 시작한다. LMCache의 로컬 offloading에서는 CPU → GPU load가 같은 step의 `start_load_kv()`에서 완료되므로 항상 `False`를 반환한다.

Connector는 local hit 수를 받아서, LMCache에 **추가로** 얼마나 더 있는지를 조회한다. 내부적으로 `LookupClient` → ZMQ → `LookupServer` → `LMCacheEngine.lookup()` 경로로 CPU/disk/remote backend의 `batched_contains()`를 호출한다. LMCache가 token 0-700까지 보유하고 있으면 `num_external_hit = 700`이지만, local에 이미 200이 있으므로 `ext_tokens = 700 - 200 = 500`만 반환한다 (섹션 1.7의 겹침 방지 로직).

**합산: 무엇을 계산하고 무엇을 건너뛸 것인가**

```python
# vllm/v1/core/sched/scheduler.py:631-639
num_computed_tokens = num_new_local_computed_tokens + ext_tokens
# 200 (GPU) + 500 (LMCache) = 700
# → 나머지 300 토큰만 forward pass에서 계산
```

| 구간 | 토큰 수 | KV 출처 | GPU block |
|------|---------|---------|-----------|
| 0 - 199 | 200 | GPU prefix cache (이미 있음) | 기존 block 재사용 |
| 200 - 699 | 500 | LMCache (CPU/disk) | 빈 block 새로 할당 → `start_load_kv()`가 채움 |
| 700 - 999 | 300 | 없음 → forward pass 계산 | 빈 block 새로 할당 → attention에서 write |

#### 2.1.3 블록 할당 후 상태 업데이트

```python
# vllm/v1/core/sched/scheduler.py:746-749
self.connector.update_state_after_alloc(
    request,
    self.kv_cache_manager.get_blocks(request_id),
    num_external_computed_tokens,
)
```

`allocate_slots()` 이후에 호출된다. 이 시점에서 LMCache가 채울 500개 토큰분의 빈 GPU block이 확보되었으므로, connector는 `LoadSpec.can_load = True`로 전환한다. 이 플래그가 True여야 worker가 실제로 `start_load_kv()`를 실행한다 — GPU에 넣을 자리가 생기기 전에는 load를 시작할 수 없기 때문이다.

#### 2.1.4 Connector Metadata 빌드

```python
# vllm/v1/core/sched/scheduler.py:898-901
meta = self.connector.build_connector_meta(scheduler_output)
scheduler_output.kv_connector_metadata = meta
```

Scheduler가 결정한 load/save 정보를 `KVConnectorMetadata`로 패키징하여 `SchedulerOutput`에 담는다.

이 metadata가 필요한 이유는, 하나의 forward pass에서 토큰 구간마다 KV의 출처가 다르기 때문이다. 섹션 2.1.2의 예시를 다시 보면:

| 구간 | KV 출처 | load | save |
|------|---------|------|------|
| 0 - 199 | GPU prefix cache (이미 있음) | 불필요 | 불필요 (이미 CPU에도 있을 수 있음) |
| 200 - 699 | LMCache CPU | CPU → GPU 로드 필요 | 불필요 (이미 CPU에 있음) |
| 700 - 999 | 없음 → forward pass에서 새로 계산 | 불필요 | CPU에 저장 필요 |

Worker가 이 구분 없이 동작하면, 이미 GPU에 있는 KV를 다시 CPU에서 로드하거나, 이미 CPU에 있는 KV를 다시 저장하는 등 불필요한 PCIe 전송이 발생한다. 이를 방지하기 위해 `build_connector_meta()`는 요청별로 `LoadSpec`과 `SaveSpec`을 생성하여 `KVConnectorMetadata`에 담는다:

- **`LoadSpec`**: CPU에서 GPU로 가져올 토큰 범위와 `can_load` 플래그. 빈 GPU block이 할당된 후에만 `can_load = True`가 된다.
- **`SaveSpec`**: GPU에서 CPU로 저장할 토큰 범위, `skip_leading_tokens`(이미 CPU에 있는 토큰 수), `can_save` 플래그.

이 두 spec이 `SchedulerOutput`에 실려 worker로 전달되면, worker는 요청별로 정확히 어떤 구간을 load/save할지 알 수 있다.

`LoadSpec`은 단순하다 — 빈 GPU block이 할당되었으면 `can_load = True`, 아니면 `False`. 반면 `SaveSpec`의 `can_save`를 결정하는 로직은 좀 더 복잡한데, CPU cache 공간이 유한하고 LRU eviction이 있기 때문이다. 재사용 가치가 낮은 KV를 저장하면 가치가 높은 KV를 밀어내므로, `build_connector_meta()`는 요청별로 "이 KV를 저장할 가치가 있는가"를 평가한다. 네 가지 조건 중 하나라도 해당하면 save를 skip한다:

```python
skip_save = (
    tracker.skip_save
    or (num_saved_tokens > 0 and input_token_len < chunk_boundary)
    or (is_decode_phase and not save_decode_cache)
    or request_skip
)
```

각 조건의 의미:

1. **`tracker.skip_save`** — 요청 생성 시 한번 결정되면 이후 바뀌지 않는 플래그. `priority_limit` 설정 시 우선순위가 낮은 요청은 저장을 건너뛴다 — 제한된 CPU cache를 우선순위 높은 요청의 KV에 확보하기 위해서다 ([LMCache PR #1368](https://github.com/LMCache/LMCache/pull/1368)).

2. **`num_saved_tokens > 0 and input_token_len < chunk_boundary`** — chunk 정렬 조건. LMCache는 256-token chunk 단위로 저장하므로, 이전에 이미 저장한 토큰이 있고(`num_saved_tokens > 0`) 아직 다음 chunk 경계(256의 배수)에 도달하지 못했으면 저장할 게 없다. 예: 256개를 이미 저장했으면, 총 512개가 쌓일 때까지 save를 skip한다. 첫 번째 save(`num_saved_tokens == 0`)는 이 조건을 bypass하여 항상 실행된다.

3. **`is_decode_phase and not save_decode_cache`** — decode phase(한 번에 1토큰씩 생성하는 단계)에서는 기본적으로 save를 skip한다 (`save_decode_cache` 기본값: `False`). Prefill KV는 system prompt나 공통 prefix를 담고 있어 다른 요청이 재사용할 수 있지만, decode KV는 모델이 생성한 고유한 토큰을 포함하므로 다른 요청의 prefix와 일치할 가능성이 거의 없다. 재사용되지 않을 KV를 저장하면 CPU 공간만 차지하고, 정작 재사용 가치가 높은 prefill KV가 LRU eviction 당할 수 있다 ([LMCache PR #973](https://github.com/LMCache/LMCache/pull/973)).

4. **`request_skip`** — API 호출 시 `kv_transfer_params: {"lmcache.skip_save": true}`로 개별 요청 단위의 저장 skip을 지정할 수 있다. 호출자가 "이 prompt는 일회성이므로 다시 hit될 일이 없다"고 판단할 때 사용한다 ([LMCache PR #1574](https://github.com/LMCache/LMCache/pull/1574)).

이 평가 결과는 `SaveSpec(skip_leading_tokens, can_save=not skip_save)`로 패키징되어 worker에 전달된다.

---

### 2.2 실행 흐름: Worker Side

#### 2.2.1 Connector 초기화

Worker 시작 시 [`kv_transfer_state.py`](https://github.com/vllm-project/vllm/blob/v0.17.1/vllm/distributed/kv_transfer/kv_transfer_state.py)에서 global singleton으로 connector를 초기화한다:

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

#### 2.2.2 Forward Pass Lifecycle

[`KVConnectorModelRunnerMixin._get_kv_connector_output`](https://github.com/vllm-project/vllm/blob/v0.17.1/vllm/v1/worker/kv_connector_model_runner_mixin.py) ([PR #21980](https://github.com/vllm-project/vllm/pull/21980))가 전체 lifecycle을 관리한다:

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

#### 2.2.3 start_load_kv 상세

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

#### 2.2.4 wait_for_save 상세 (non-layerwise)

`wait_for_save()`는 섹션 2.2.2의 `_get_kv_connector_output` context manager의 `finally` 블록에서 호출되므로, layerwise든 non-layerwise든 **forward pass가 끝나면 항상 실행**된다. 내부에서 `use_layerwise`를 확인하여 모드에 따라 다른 작업을 수행한다:

```python
# wait_for_save() 내부 분기
def wait_for_save(self):
    if self.use_layerwise:
        # layerwise: generator의 마지막 iteration 실행 → 비동기 저장 완료 보장
        for layerwise_storer in self.layerwise_storers:
            next(layerwise_storer)
        return

    # non-layerwise: 여기서 전체 레이어의 KV를 한번에 store()
    for request in connector_metadata.requests:
        ...
        self.lmcache_engine.store(token_ids, mask=store_mask, ...)
```

**Non-layerwise 모드** (기본값): forward pass 중에는 KV save를 하지 않는다. `wait_for_save()` 내부에서 `lmcache_engine.store()`를 호출하여 전체 레이어의 KV를 **한번에** 저장한다. 구현이 단순하지만, store 시간 동안 GPU가 다음 작업을 시작하지 못한다.

**Layerwise 모드**: 각 attention layer에 `@maybe_transfer_kv_layer` decorator ([PR #27816](https://github.com/vllm-project/vllm/pull/27816))가 걸려 있어서, forward pass 도중에 레이어별로 `store_layer()` generator가 비동기로 save를 진행한다. `wait_for_save()`에서는 이 generator의 **마지막 iteration**(`next(storer)`)을 실행하여 아직 완료되지 않은 저장을 마무리한다. GPU compute와 CPU 전송이 overlap되어 latency가 숨겨진다.

아래 코드는 non-layerwise 경로에서 `wait_for_save()`가 실행하는 store 로직이다:

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

#### 2.2.5 vLLM Connector → LMCacheEngine: API 경계 정리

섹션 2.1~2.2를 통해 scheduler와 worker의 실행 흐름을 추적했다. 여기서 한 발 물러서 보면, **KV cache offloading 경로에서** LMCache connector가 `LMCacheEngine`으로 호출하는 API는 5개로 수렴한다 (LMCache connector는 P/D disaggregation에서도 사용될 수 있지만, 그 경우 원격 전송 관련 추가 경로가 활성화된다 — 이 포스트에서는 offloading 경로만 다룬다):

| LMCacheEngine API | vLLM Connector 호출자 | 용도 |
|---|---|---|
| `lookup()` | LookupServer ← `get_num_new_matched_tokens()` | cache hit 토큰 수 조회. 데이터 이동 없이 `batched_contains()`만 실행 |
| `retrieve()` | `start_load_kv()` (non-layerwise) | 외부 storage → GPU paged KV buffer로 KV 로드 |
| `retrieve_layer()` | `start_load_kv()` (layerwise) | 위와 동일하나 generator 기반 — 레이어별 파이프라이닝 |
| `store()` | `wait_for_save()` (non-layerwise) | GPU paged KV buffer → 외부 storage로 KV 저장 |
| `store_layer()` | `save_kv_layer()` (layerwise) | 위와 동일하나 generator 기반 — 레이어별 파이프라이닝 |

`lookup()`은 scheduler 프로세스에서 LookupClient → ZMQ → LookupServer → `lmcache_engine.lookup()` 경로로 호출된다. 나머지 4개는 worker 프로세스에서 직접 호출한다. 각 API에서 `token_ids`, `slot_mapping`, `kvcaches`(GPU paged buffer 참조), `token_mask`가 공통적으로 전달되며, LMCacheEngine은 이 정보를 바탕으로 GPU paged buffer의 올바른 위치에서 KV를 읽거나 쓴다.

**P/D disaggregation에서도 같은 5개 API를 사용**하지만, 동작이 달라지는 부분이 있다:

| | Offloading (`kv_both`) | Disagg Producer (`kv_producer`) | Disagg Consumer (`kv_consumer`) |
|---|---|---|---|
| `lookup` / `retrieve` | 활성 | 비활성 (KV를 받지 않음) | 활성 |
| `store` | CPU/disk에 저장 | `transfer_spec=DisaggSpec`으로 **원격 전송** | 비활성 (KV를 보내지 않음) |
| `save_kv_layer` skip 조건 | chunk 경계 미달 시 skip | **skip 불가** — 원격 전송이므로 반드시 실행 | 항상 skip |

핵심 차이는 `store()`의 `transfer_spec` 파라미터다. Offloading에서는 `None`이라 LocalCPUBackend에 저장하지만, disagg에서는 `DisaggSpec`(receiver host/port 정보)이 전달되어 LMCacheEngine이 원격 노드로 KV를 전송한다.

다음 섹션에서는 이 API 경계를 넘어, LMCacheEngine 내부에서 실제로 무슨 일이 일어나는지를 추적한다.

---

## 3. LMCache 내부: LMCacheEngine은 어떻게 동작하는가

### 3.1 Component 구조

LMCache는 vLLM의 두 프로세스(Scheduler, Worker)에 걸쳐 동작하며, 각 프로세스에서 역할이 완전히 다르다.

**Scheduler 프로세스**에는 `LookupClient` 하나만 존재한다. 이 client는 "이 token sequence의 KV가 외부에 있는가?"라는 질문만 던지고, 실제 KV 데이터는 건드리지 않는다. ZMQ를 통해 worker의 `LookupServer`에 RPC를 보내고 hit 토큰 수만 받아온다.

**Worker 프로세스**에 핵심 로직이 집중된다. 중심에 `LMCacheEngine`이 있고, 이 엔진이 세 컴포넌트를 조율한다:
- **`TokenDatabase`**: token ID 시퀀스를 256-token chunk 단위의 `CacheEngineKey`(prefix hash chain)로 변환한다. 이 key가 store/retrieve/lookup 모두에서 주소 역할을 한다.
- **`GPUConnector`**: vLLM의 paged KV buffer와 LMCache의 contiguous chunk 사이에서 CUDA kernel 기반 scatter/gather + D2H/H2D 복사를 수행한다.
- **`StorageManager` → `LocalCPUBackend`**: pinned CPU memory에 KV chunk를 보관한다. CPU 메모리가 가득 차면 LRU eviction이 여기서 발생한다.

데이터 흐름은 두 갈래다:
- **Store** (forward pass 후): GPU paged KV → `GPUConnector`가 gather+D2H → `StorageManager`가 CPU에 보관
- **Retrieve** (preemption 복구 시): `StorageManager`가 CPU에서 꺼냄 → `GPUConnector`가 H2D+scatter → GPU paged KV에 복원

<div class="mermaid-wide">
flowchart LR
    subgraph sched["Scheduler Process"]
        direction TB
        S["vLLM Scheduler"] -->|"lookup"| LC["LookupClient"]
    end

    LC -.->|"ZMQ"| LS

    subgraph worker["Worker Process"]
        direction TB
        W["vLLM Worker"]
        LS["LookupServer"]

        W -->|"store / retrieve"| ENG
        LS -->|"lookup"| ENG

        ENG["LMCacheEngine"]

        ENG --> TDB["TokenDatabase<br/>chunk key 생성"]
        ENG --> GPU["GPUConnector<br/>GPU ↔ CPU DMA"]
        ENG --> SM["StorageManager"]

        subgraph backends["Storage Backends"]
            CPU["LocalCPUBackend<br/>pinned memory"]
            DSK["LocalDiskBackend"]
            RMT["RemoteBackend"]
        end

        SM --> CPU
        SM --> DSK
        SM --> RMT
    end

    style sched fill:#f0f4f8,stroke:#4a90d9
    style worker fill:#fdf8f0,stroke:#e67e22
    style backends fill:transparent,stroke:#999,stroke-dasharray:5 5
    style ENG fill:#4a90d9,stroke:#333,color:#fff
    style TDB fill:#2ecc71,stroke:#333,color:#fff
    style GPU fill:#27ae60,stroke:#333,color:#fff
    style CPU fill:#e67e22,stroke:#333,color:#fff
</div>

핵심 구성 요소:

**Lookup (Scheduler ↔ Worker RPC)**

| Component | Side | 역할 |
|-----------|------|------|
| **`LookupClient`** | Scheduler | `get_num_new_matched_tokens()` 호출 시 ZMQ로 cache hit 수 조회 |
| **`LookupServer`** | Worker | LookupClient 요청을 받아 `LMCacheEngine.lookup()` 실행 후 결과 반환 |

**Core Engine (Worker)**

| Component | 역할 |
|-----------|------|
| **`LMCacheEngine`** | 중앙 허브. store/retrieve/lookup의 진입점 |
| **`TokenDatabase`** | token ID 시퀀스 → 256-token chunk 단위의 `CacheEngineKey`(prefix hash chain) 변환 |
| **`GPUConnector`** | GPU paged KV buffer ↔ CPU/Storage 간 데이터 이동. CUDA kernel(`lmc_ops`) 사용 |
| **`ZMQOffloadServer`** | LMCacheEngine에 직접 접근할 수 없는 프로세스의 store 요청을 ZMQ IPC로 수신하여 중계 |

**Storage Backends (Worker)**

| Component | CPU 경유 | 역할 |
|-----------|---------|------|
| **`StorageManager`** | — | 여러 backend를 관리. put/get을 모든 backend에 분배 |
| **`LocalCPUBackend`** | Yes | 항상 생성되는 primary store + allocator. pinned CPU memory pool에서 `MemoryObj` 할당 |
| **`LocalDiskBackend`** | Yes | CPU DRAM → NVMe SSD. 설정 시 활성화 |
| **`RemoteBackend`** | Yes | CPU DRAM → Redis/Valkey 등 네트워크 전송 |

**Cross-node 관리 (별도 프로세스)**

| Component | 역할 |
|-----------|------|
| **`BatchedMessageSender`** | LocalCPUBackend의 admit/evict 이벤트를 batching하여 Cache Controller에 전달 |
| **`Cache Controller`** | 어떤 worker가 어떤 chunk를 보유하는지 전역 인덱스 관리. P2P lookup 응답, worker 등록/해제, cross-node 데이터 이동 조율 |

### 3.2 Store 흐름: GPU → CPU

아래는 `LocalCPUBackend` 경로를 기준으로 한 store 흐름이다.

<div class="mermaid-wide">
sequenceDiagram
    participant V as vLLM Worker
    participant E as LMCacheEngine
    participant T as TokenDatabase
    participant G as GPUConnector
    participant SM as StorageManager
    participant CPU as LocalCPUBackend

    V->>E: store(token_ids, slot_mapping, kvcaches)

    E->>T: process_tokens(token_ids)
    Note over T: 256-token chunk로 분할<br/>prefix hash chain → CacheEngineKey[]
    T-->>E: keys[]

    E->>SM: allocate(keys)
    SM->>CPU: allocate from pinned memory pool
    alt pool 가득 참
        CPU->>CPU: LRU evict → 메모리 반환
    end
    CPU-->>SM: MemoryObj[] — 빈 CPU 텐서

    E->>G: batched_from_gpu(slot_mapping, MemoryObj[])
    Note over G: CUDA kernel D2H:<br/>GPU paged KV → CPU pinned memory
    G-->>E: MemoryObj[] — KV 데이터 채워짐

    E->>SM: batched_put(keys, MemoryObj[])
    SM->>CPU: submit_put(key, MemoryObj)
    Note over CPU: CPU DRAM에 보존 완료
</div>

> CPU 할당과 LRU eviction이 store 시점에 발생한다 — 섹션 1.6에서 설명한 CPU eviction이 바로 이 `allocate()` 호출에서 일어난다.

### 3.3 Retrieve 흐름: CPU → GPU

아래는 `LocalCPUBackend`에서 hit된 경우의 retrieve 흐름이다.

<div class="mermaid-wide">
sequenceDiagram
    participant V as vLLM Worker
    participant E as LMCacheEngine
    participant T as TokenDatabase
    participant SM as StorageManager
    participant CPU as LocalCPUBackend
    participant G as GPUConnector

    V->>E: retrieve(token_ids, token_mask, slot_mapping, kvcaches)

    E->>T: process_tokens(token_ids)
    T-->>E: keys[]

    E->>SM: batched_contains(keys)
    Note over SM: backend 순서대로 확인
    SM-->>E: block_mapping — key별 hit backend

    E->>SM: batched_get(keys, block_mapping)
    SM->>CPU: get_blocking(key)
    CPU-->>SM: MemoryObj
    SM-->>E: MemoryObj[]

    E->>G: batched_to_gpu(MemoryObj[], slot_mapping)
    Note over G: CUDA kernel H2D:<br/>CPU pinned memory → GPU paged KV
    G-->>V: GPU KV buffer에 KV 복원 완료
</div>

### 3.4 GPUConnector: 하드웨어 추상화와 구현

위의 store/retrieve 흐름에서 `GPUConnector`가 "CUDA kernel로 복사"한다고만 설명했는데, 실제 구현은 상당히 정교하다. vLLM의 paged KV buffer와 LMCache의 contiguous chunk 사이의 **format 변환**, **async stream 관리**, **pinned memory 할당**, **layerwise double-buffering**이 모두 하드웨어 종속적인 primitive 위에 구축되어 있다.

#### GPUConnector 인터페이스와 디바이스별 구현

이 포스트에서 다루는 CPU backend(`LocalCPUBackend`) 경로에서는, `GPUConnector`가 **하드웨어 종속 로직이 집중되는 유일한 계층**이다. `LMCacheEngine`, `StorageManager`, `TokenDatabase` 등 나머지 컴포넌트는 모두 디바이스에 무관하게 동작한다. (NIXL이나 GDS 같은 storage backend를 사용하면 디바이스 메모리 등록 등 추가적인 하드웨어 종속 지점이 생기지만, 이는 후속 포스트에서 다룬다.)

[`CreateGPUConnector()`](https://github.com/LMCache/LMCache/blob/dev/lmcache/v1/gpu_connector/__init__.py)가 vLLM의 `current_platform`을 조회하여 디바이스에 맞는 connector를 생성한다:

| Connector | 디바이스 | scatter/gather 구현 | 비고 |
|---|---|---|---|
| `VLLMPagedMemGPUConnectorV2` | CUDA | `lmc_ops.multi_layer_kv_transfer` (CUDA kernel) | 기본 connector |
| `VLLMPagedMemLayerwiseGPUConnector` | CUDA | `lmc_ops.single_layer_kv_transfer` (per-layer) | layerwise 파이프라이닝 |
| `VLLMBufferLayerwiseGPUConnector` | CUDA | 위 + RoPE re-encoding | CacheBlend 지원 |
| `VLLMPagedMemXPUConnectorV2` | Intel XPU | `torch.Tensor.index_copy_` / `index_select` | CUDA kernel 미사용 |
| `VLLMPagedMemHPUConnectorV2` | Gaudi HPU | `torch.Tensor` ops + `htcore.mark_step()` | 독자적 동기화 모델 |

주목할 점은 **XPU/HPU connector가 CUDA kernel 없이 순수 PyTorch tensor 연산만으로 scatter/gather를 구현**한다는 것이다. `index_copy_`(scatter)와 `index_select`(gather)로 `slot_mapping` 기반의 paged ↔ contiguous 변환을 수행하며, 이는 custom kernel 대비 성능은 떨어지지만 어떤 PyTorch backend에서든 동작한다. CUDA connector는 이를 전용 kernel + stream 분리 + widening copy 등의 최적화로 가속한 것이다.

아래에서는 CUDA connector(`VLLMPagedMemGPUConnectorV2`)의 구현을 따라가며 각 하드웨어 종속 primitive를 설명한다.

#### KV Format 변환: paged ↔ contiguous

vLLM은 KV cache를 block 단위로 GPU 메모리 여기저기에 흩어서 저장한다(paged) — token 0-15는 physical block #1에, token 16-31은 block #3에, token 32-47은 block #6에 있을 수 있다. 반면 LMCache는 256개 토큰의 KV를 **하나의 연속된 메모리**에 담아야 한다 (chunk 단위로 해싱하고 저장하기 때문이다).

이 두 layout 사이를 변환하는 것이 scatter/gather이며, `slot_mapping`("token N의 KV는 physical slot M에 있다")을 기반으로 동작한다:
- **Gather** (store, D2H): GPU의 흩어진 block들에서 토큰별 KV를 **모아서** CPU의 연속 버퍼에 쓴다
- **Scatter** (retrieve, H2D): CPU의 연속 버퍼에서 토큰별 KV를 읽어 GPU의 **흩어진 block 위치에 뿌린다**

이 변환은 별도 단계가 아니라, **GPU ↔ CPU 데이터 이동과 동시에** `GPUConnector` 내부에서 수행된다:

```python
# lmcache/v1/gpu_connector/gpu_connectors.py — VLLMPagedMemGPUConnectorV2

def batched_from_gpu(self, ...):
    """Store path (D2H): GPU paged KV → CPU contiguous chunk"""
    current_stream = torch.cuda.current_stream()
    self.store_stream.wait_stream(current_stream)  # attention 완료 대기
    with torch.cuda.stream(self.store_stream):
        lmc_ops.multi_layer_kv_transfer(
            key_value,         # 목적지: LMCache contiguous buffer [2, L, T, D]
            key_value_ptrs,    # 소스: vLLM per-layer KV cache pointer 배열
            slot_mapping,      # [num_tokens] token → physical slot 매핑
            direction=D2H,     # GPU → CPU
            gpu_kv_format=..., # vLLM KV layout format
            block_size=16,
        )
    self.store_stream.synchronize()

def batched_to_gpu(self, ...):
    """Retrieve path (H2D): CPU contiguous chunk → GPU paged KV"""
    with torch.cuda.stream(self.load_stream):
        lmc_ops.multi_layer_kv_transfer(
            key_value,         # 소스: LMCache contiguous buffer
            key_value_ptrs,    # 목적지: vLLM per-layer KV cache
            slot_mapping,
            direction=H2D,     # CPU → GPU
            gpu_kv_format=...,
            block_size=16,
        )
    self.load_stream.synchronize()
```

Layerwise 경로에서는 `lmc_ops.single_layer_kv_transfer()`가 한 레이어씩 동일한 작업을 수행한다.

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

위의 `batched_from_gpu` / `batched_to_gpu` 코드에서 이미 보았듯이, GPUConnector는 **두 개의 전용 CUDA stream**(`store_stream`, `load_stream`)을 생성하여 KV transfer를 default compute stream과 분리한다. 핵심은 stream 간 의존성 관리다:

- **Store** (D2H): `store_stream.wait_stream(current_stream)`으로 attention 완료를 기다린 뒤 D2H를 시작하고, `store_stream.synchronize()`로 CPU에서 `MemoryObj`에 접근하기 전에 전송 완료를 보장한다.
- **Retrieve** (H2D): `load_stream`에서 H2D를 수행한 뒤 `load_stream.synchronize()`로 완료를 보장하고, 이후 attention이 해당 KV를 사용한다.

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

#### 하드웨어 종속 지점 요약

위에서 설명한 내용을 종합하면, LMCache에서 하드웨어에 종속적인 코드는 크게 세 가지 계층에 집중되어 있다. 나머지 컴포넌트(LMCacheEngine, TokenDatabase, StorageManager, Scheduler-side LookupClient 등)는 모두 디바이스에 무관하게 동작한다.

| 계층 | CUDA 구현 | 소스 위치 | 역할 | 대안 경로 |
|------|----------|-----------|------|----------|
| **KV Transfer Kernel** | `lmc_ops.multi_layer_kv_transfer` | `csrc/mem_kernels.cu` | paged ↔ contiguous 변환 + D2H/H2D를 한 kernel에서 수행 | `torch.Tensor.index_copy_` / `index_select` (XPU connector가 사용) |
| **Pinned Memory** | `cudaHostAlloc` / `cudaHostRegister` | `csrc/mem_alloc.cpp`, `lazy_memory_allocator.py` | DMA 전송 시 bounce buffer 없이 직접 접근 가능한 host memory | `non_cuda_equivalents.py` (unpinned fallback, DMA 불가) |
| **Async Streams** | `torch.cuda.Stream()` | `gpu_connectors.py` | store/retrieve를 attention 연산과 비동기 분리 | `torch.{device}.synchronize()` (XPU/HPU가 사용하는 동기 모델) |

**KV Transfer Kernel이 가장 핵심적인 종속 지점**이다. 이 kernel은 단순 memcpy가 아니라 scatter/gather + format 변환 + D2H/H2D 복사를 **하나의 kernel launch**로 수행하며, 6종의 `GPUKVFormat`(vLLM Flash Attention, FlashInfer, MLA, SGLang MHA/MLA 등)을 compile-time template으로 분기한다.

---

## Appendix: CPU Offloading + P/D Disaggregation 동시 사용

이 포스트에서는 CPU backend를 사용한 로컬 offloading만 다뤘다. 그런데 실제 프로덕션 환경에서는 "prefill 노드가 KV를 로컬 CPU에도 보관하면서, 동시에 NIXL로 decode 노드에 전송"하는 구성이 필요할 수 있다. LMCache는 이 조합을 지원한다.

### 시나리오: Multi-turn Coding Assistant

Coding assistant agent처럼 multi-turn 대화를 서빙하는 경우를 생각하자:

| Turn | 요청 내용 | KV 상태 |
|------|----------|---------|
| 1 | `[system] + [user: "이 코드 리팩토링해줘"]` | prefill KV → CPU 저장 ✓ |
| 1 | agent 응답: "다음과 같이 수정합니다..." (500 tok) | decode KV → CPU 저장 여부? |
| 2 | `[system] + [turn 1 전체] + [user: "테스트도 추가해줘"]` | turn 1의 agent 응답이 prefix에 포함됨 |

Turn 2에서 turn 1의 agent 응답(500 tok)이 prefix로 들어온다. 이 구간의 KV가 CPU에 없으면 500 토큰을 처음부터 recompute해야 한다. Multi-turn 대화에서는 이전 턴의 decode 출력이 다음 턴의 prefix가 되므로, **prefill뿐 아니라 decode phase의 KV도 offloading해야** recompute를 최소화할 수 있다.

이를 위해 `save_decode_cache=True`를 설정한다 ([LMCache PR #973](https://github.com/LMCache/LMCache/pull/973)). 기본값 `False`는 단발성 요청이 주인 워크로드에 적합하고, multi-turn agent 시나리오에서는 `True`로 변경해야 한다.

동시에 P/D disaggregation도 사용한다면 — prefill 노드가 decode 노드에 KV를 NIXL로 전송하면서, 로컬 CPU에도 보관하는 구성이 된다.

### 설정 방법

**vLLM 측**: `kv_role`을 `"kv_both"`로 설정한다.

```python
# vllm/config/kv_transfer.py
KVProducer = Literal["kv_producer", "kv_both"]
KVConsumer = Literal["kv_consumer", "kv_both"]
KVRole = Literal[KVProducer, KVConsumer]
```

[`kv_role`](https://github.com/vllm-project/vllm/blob/v0.17.1/vllm/config/kv_transfer.py)의 유효값은 `"kv_producer"`, `"kv_consumer"`, `"kv_both"` 세 가지다. `"kv_both"`는 `is_kv_producer`와 `is_kv_consumer` 모두 `True`를 반환하므로, 한 인스턴스가 offloading과 disaggregation을 동시에 수행할 수 있다.

**LMCache 측**: `enable_pd=True` + `local_cpu=True` + `max_local_cpu_size > 0` + `save_decode_cache=True`를 설정한다.

```yaml
# lmcache-config.yaml 예시
chunk_size: 256
max_local_cpu_size: 40g
local_cpu: true
enable_pd: true
transfer_channel: "nixl"
save_decode_cache: true          # decode phase KV도 저장
save_unfull_chunk: false

# NIXL buffer 설정
# buffer size는 chunk size로 나누어떨어져야 함
nixl_buffer_size: 3019898880     # ~2880MiB (Qwen3-4B/8B/32B 기준)
nixl_buffer_device: cpu

# NIXL storage backend는 이 시나리오에서 사용하지 않음
# (SSD offloading은 후속 포스트에서 다룸)
extra_config:
  enable_nixl_storage: false
```

### 왜 동시에 동작하는가: StorageManager의 multi-backend 구조

핵심은 [`CreateStorageBackends()`](https://github.com/LMCache/LMCache/blob/dev/lmcache/v1/storage_backend/__init__.py)가 설정에 따라 **여러 backend를 동시에 생성**한다는 것이다:

```python
# lmcache/v1/storage_backend/__init__.py — CreateStorageBackends()
storage_backends = {}

if config.enable_pd and "PDBackend" not in _skip:
    storage_backends["PDBackend"] = PDBackend(config, metadata)

# ...

elif not config.enable_pd or config.local_cpu:
    if config.max_local_cpu_size > 0:
        storage_backends["LocalCPUBackend"] = LocalCPUBackend(...)
```

`enable_pd=True`이고 `local_cpu=True`이면, `not config.enable_pd or config.local_cpu`가 `True`로 평가되어 **`PDBackend`와 `LocalCPUBackend` 모두** `storage_backends` dict에 들어간다.

### store() 호출 시 두 backend에 동시 분배

[`StorageManager.batched_put()`](https://github.com/LMCache/LMCache/blob/dev/lmcache/v1/storage_backend/storage_manager.py)은 `store_location`이 `None`(기본값)이면 **모든 등록된 backend를 순회**한다:

```python
# lmcache/v1/storage_backend/storage_manager.py — batched_put()
def batched_put(self, keys, memory_objs, transfer_spec=None, location=None):
    obj_dict[get_backend_cname(self.allocator_backend)] = (keys, memory_objs)

    for backend_name, backend in self.storage_backends.items():
        if location and backend_name != location:
            continue  # location 지정 시 해당 backend만 실행
        # 다른 allocator의 backend면 메모리 복사본 생성
        backend.batched_submit_put_task(ks, objs, transfer_spec=transfer_spec)
```

같은 `batched_put()` 호출에서:
- **`PDBackend`**: `transfer_spec`(receiver host/port 정보)을 받아 NIXL로 decode 노드에 KV를 전송한다.
- **`LocalCPUBackend`**: `transfer_spec`을 받지만 **무시**하고, 로컬 pinned CPU memory에 KV를 저장한다.

각 backend가 서로 다른 allocator를 사용하면, `batched_put()` 내부에서 메모리 복사본이 독립적으로 생성된다 (line 421-424). 따라서 두 경로가 서로의 메모리를 간섭하지 않는다.

### 데이터 흐름 요약

`save_decode_cache=True`이면 prefill과 decode 모두에서 save가 실행된다. Coding assistant의 Turn 1을 예로 들면:

```
[Prefill phase — "이 코드 리팩토링해줘" 처리]
  → forward pass (prompt 토큰 KV 계산)
  → LMCacheEngine.store(transfer_spec=DisaggSpec)
    → GPUConnector.batched_from_gpu()
    → StorageManager.batched_put()
      ├── PDBackend.submit_put()       # NIXL → decode 노드 전송
      └── LocalCPUBackend.submit_put() # 로컬 CPU 저장

[Decode phase — agent 응답 생성, save_decode_cache=True]
  → 토큰 생성 (매 step 1 tok)
  → 256 tok 누적 시 chunk 경계 도달
  → LMCacheEngine.store(transfer_spec=None)  # decode는 로컬 저장만
    → StorageManager.batched_put()
      └── LocalCPUBackend.submit_put() # agent 응답 KV → CPU 저장

[Turn 2 — "테스트도 추가해줘"]
  → scheduler: get_num_new_matched_tokens()
  → LMCache lookup: turn 1 전체(prefill + decode 응답)가 CPU에 있음!
  → CPU → GPU 복원, recompute 최소화
```

### 제약 사항

- **Layerwise store + disagg 미지원**: layerwise 저장 경로(`save_kv_layer`)와 disagg spec의 조합은 아직 구현되어 있지 않다 ([`vllm_v1_adapter.py` L1015-1017](https://github.com/vllm-project/vllm/blob/v0.17.1/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/vllm_v1_adapter.py)). Non-layerwise 경로(`wait_for_save`)에서만 동시 사용이 가능하다.

- **`kv_producer` vs `kv_both`의 skip 로직 차이**: `wait_for_save()` 내부에서 `kv_role == "kv_producer"`인 경우 disagg 전용 skip 로직을 타고 ([`vllm_v1_adapter.py` L1077](https://github.com/vllm-project/vllm/blob/v0.17.1/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/vllm_v1_adapter.py)), `kv_both`인 경우 이 분기를 타지 않고 일반 로컬 save 경로의 skip 로직을 따른다 ([`vllm_v1_adapter.py` L1063](https://github.com/vllm-project/vllm/blob/v0.17.1/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/vllm_v1_adapter.py)). `kv_both`에서 disagg transfer가 의도대로 동작하는지는 설정에 따라 검증이 필요하다.

- **`save_decode_cache` 버그 (vLLM ≥ 0.9.2)**: `save_decode_cache=True` 설정이 vLLM 0.9.2 이후 정상 동작하지 않는 버그가 보고되어 있다. vLLM의 상태 업데이트 순서 변경으로 decode phase에서 `skip_save`가 항상 `True`로 평가되는 문제다 ([LMCache PR #2821](https://github.com/LMCache/LMCache/pull/2821), open).
