---
layout: post
title: "LMCache KV Cache Offloading (2) - NIXL Connector vs LMCache + NIXL"
category: llm-serving
---

> **Note**: 이 포스트는 vLLM `main` 브랜치 (2026-03-27 기준)와 [LMCache `dev` 브랜치](https://github.com/LMCache/LMCache/tree/dev)를 기준으로 작성되었습니다.

> **이전 포스트**: [LMCache KV Cache Offloading (1) - Overview](/2026-03-25-lmcache-kv-offloading-codemap.html)에서 LMCache의 CPU offloading 경로 (LocalCPUBackend)를 다뤘습니다. 이번 포스트는 **cross-node P/D disaggregation** — prefill 노드와 decode 노드 사이의 KV 전송을 다룹니다.

[이전 포스트](/2026-03-25-lmcache-kv-offloading-codemap.html)에서 다룬 KV offloading은 단일 노드 내에서 GPU ↔ CPU 사이의 수직 이동이었다. 그런데 프로덕션 환경에서는 다른 차원의 문제가 있다: **prefill과 decode를 서로 다른 GPU에서 실행**하는 P/D disaggregation이다.

Prefill은 긴 prompt를 한 번에 처리하므로 compute-bound이고, decode는 한 토큰씩 생성하므로 memory-bound다. 같은 GPU에서 둘 다 하면 하드웨어 활용률이 떨어진다. 이를 분리하면 각 워크로드에 최적화된 GPU 구성을 사용할 수 있지만, 핵심 문제가 생긴다: **prefill GPU에서 계산한 KV cache를 decode GPU로 어떻게 빠르게 옮기는가?**

vLLM에는 이 문제를 해결하는 **두 가지 경로**가 있다:

1. **NIXL Connector** — vLLM에 내장된 first-party 구현. RDMA를 통해 prefill GPU 메모리에서 decode GPU 메모리로 **직접** 읽어간다.
2. **LMCache Connector + NIXL backend** — LMCache 라이브러리의 PDBackend가 NIXL을 transfer channel로 사용하여 KV를 전송한다. 동시에 CPU offloading도 병행 가능.

두 경로 모두 NIXL 라이브러리를 사용하지만, 아키텍처가 근본적으로 다르다. 이 포스트는 각 경로의 내부 구현을 코드 레벨에서 추적하며, 어떤 상황에서 어떤 경로가 적합한지를 분석한다.

---

* toc
{:toc}

---

## 1. 두 경로의 아키텍처 개관

코드를 들어가기 전에 전체 그림을 먼저 잡자.

### 1.1 NIXL Connector: vLLM 내장, 직접 RDMA

<div class="mermaid-wide">
flowchart LR
    subgraph prefill["Prefill Node"]
        direction TB
        PS["vLLM Scheduler"] --> PW["vLLM Worker"]
        PW --> PKV["GPU KV Cache<br/>(NIXL 등록됨)"]
    end

    subgraph decode["Decode Node"]
        direction TB
        DS["vLLM Scheduler"] --> DW["vLLM Worker"]
        DW --> DKV["GPU KV Cache<br/>(NIXL 등록됨)"]
    end

    DKV -->|"NIXL READ<br/>(one-sided RDMA)"| PKV

    style prefill fill:#fdf8f0,stroke:#e67e22
    style decode fill:#e8f4fd,stroke:#4a90d9
    style PKV fill:#e74c3c,stroke:#333,color:#fff
    style DKV fill:#4a90d9,stroke:#333,color:#fff
</div>

NIXL Connector는 vLLM의 **기존 KV cache GPU 텐서를 NIXL에 직접 등록**한다. Prefill 노드의 GPU 메모리가 곧 RDMA source이고, decode 노드의 GPU 메모리가 곧 RDMA destination이다. **중간 버퍼가 없다.** Decode 워커가 `NIXL READ` 연산으로 prefill 워커의 GPU 메모리를 one-sided RDMA로 읽어가며, 이 과정에서 prefill 워커의 CPU는 개입하지 않는다.

### 1.2 LMCache Connector + NIXL: 중간 버퍼 경유

<div class="mermaid-wide">
flowchart LR
    subgraph prefill["Prefill Node"]
        direction TB
        PS2["vLLM Scheduler"] --> PW2["vLLM Worker"]
        PW2 --> PKV2["GPU KV Cache<br/>(paged)"]
        PKV2 -->|"GPUConnector<br/>gather + D2H"| PB["PDBackend Buffer<br/>(CPU 또는 GPU)"]
        PB -->|"NIXL WRITE"| NET((" "))
    end

    subgraph decode["Decode Node"]
        direction TB
        DS2["vLLM Scheduler"] --> DW2["vLLM Worker"]
        DW2 --> DKV2["GPU KV Cache<br/>(paged)"]
        DB["PDBackend Buffer<br/>(CPU 또는 GPU)"] -->|"GPUConnector<br/>H2D + scatter"| DKV2
        NET2((" ")) -->|"RDMA 도착"| DB
    end

    NET -.->|"RDMA"| NET2

    style prefill fill:#fdf8f0,stroke:#e67e22
    style decode fill:#e8f4fd,stroke:#4a90d9
    style PKV2 fill:#e74c3c,stroke:#333,color:#fff
    style DKV2 fill:#4a90d9,stroke:#333,color:#fff
    style PB fill:#e67e22,stroke:#333,color:#fff
    style DB fill:#e67e22,stroke:#333,color:#fff
</div>

LMCache 경로는 **중간 버퍼를 거친다**. Prefill 쪽에서 GPUConnector가 paged KV를 contiguous 버퍼로 gather + 복사한 뒤, PDBackend가 NIXL WRITE로 decode 노드에 전송한다. Decode 쪽에서는 버퍼에 도착한 데이터를 다시 GPUConnector가 GPU paged KV에 scatter한다.

**왜 이런 차이가 생기는가?** NIXL Connector는 vLLM의 paged KV 블록을 **그대로** RDMA 단위로 사용한다. 반면 LMCache는 자체적인 256-token chunk 기반 content-addressed storage 체계를 가지고 있어, vLLM의 paged layout과 LMCache의 contiguous layout 사이를 GPUConnector가 변환해야 한다. 이 변환 과정이 중간 버퍼를 필요로 한다.

### 1.3 비교 요약

| | NIXL Connector | LMCache + NIXL |
|---|---|---|
| **코드 위치** | vLLM 내장 (135KB, 3100+ lines) | vLLM 12KB wrapper + LMCache 라이브러리 |
| **KV 전송 단위** | vLLM block (16 tokens) | LMCache chunk (256 tokens) |
| **GPU 메모리 등록** | vLLM KV cache 텐서 직접 등록 | PDBackend의 별도 버퍼 등록 |
| **중간 버퍼** | 없음 | PDBackend buffer (CPU/GPU) |
| **RDMA 방향** | Decode가 READ (pull) | Prefill이 WRITE (push) |
| **CPU offloading** | 미지원 | 동시 지원 (LocalCPUBackend 병행) |
| **Heterogeneous TP** | 지원 (P/D 서로 다른 TP 크기) | 미지원 |
| **Layer pipelining** | 미지원 (bulk transfer) | 지원 (layerwise GPU connector) |

---

## 2. NIXL Connector 상세

### 2.1 초기화: GPU 메모리를 NIXL에 등록

NIXL Connector의 핵심은 **vLLM의 기존 KV cache 텐서를 RDMA에 직접 노출**한다는 것이다. 별도의 staging buffer가 없다.

[`register_kv_caches()`](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py#L1546)가 이 등록을 수행한다. vLLM이 KV cache 텐서를 할당한 직후 한 번 호출된다:

```python
# nixl_connector.py — NixlConnectorWorker.register_kv_caches()
def register_kv_caches(self, kv_caches):
    for layer_name, kv_cache in kv_caches.items():
        # (base_addr, size_bytes, device_id) 튜플 수집
        caches_data.append((kv_cache.data_ptr(), kv_cache.nbytes, device_id))

    # 1. NIXL에 GPU 메모리 영역 등록
    reg_descs = self.nixl_wrapper.get_reg_descs(caches_data, "VRAM")
    self.nixl_wrapper.register_memory(reg_descs)

    # 2. block 단위 transfer descriptor 생성
    #    num_regions(레이어 수 × K/V) × num_blocks 개의 descriptor
    for region_idx in range(num_regions):
        for block_idx in range(num_blocks):
            desc = (base_addr + block_idx * block_len, block_len, device_id)
            descs.append(desc)

    # 3. 로컬 transfer handler 준비
    self.xfer_handler = self.nixl_wrapper.prep_xfer_dlist("NIXL_INIT_AGENT", descs)
```

이 과정의 결과:
- 각 KV cache 블록이 **하나의 NIXL transfer descriptor**에 매핑된다
- descriptor index는 `region_index × num_blocks + block_id`로 계산된다
- Remote 노드에서 이 descriptor index만 지정하면 해당 블록의 GPU 메모리를 직접 읽을 수 있다

**핵심**: vLLM의 paged KV 블록 = NIXL의 RDMA 전송 단위. 별도 변환 없이 블록 ID가 곧 RDMA 주소다.

### 2.2 Handshake: P/D 엔진 간 연결

Decode 노드가 prefill 노드의 KV를 읽으려면, 먼저 상대방의 NIXL agent metadata(메모리 주소, descriptor 정보)를 알아야 한다. 이 교환이 **ZMQ 기반 handshake**로 이루어진다.

<div class="mermaid-wide">
sequenceDiagram
    participant P as Prefill Worker<br/>(ZMQ ROUTER listener)
    participant D as Decode Worker<br/>(ThreadPoolExecutor)

    Note over P: 시작 시 ZMQ listener 스레드 실행<br/>NixlHandshakePayload 준비

    D->>P: GET_META_MSG (target_tp_rank)
    P-->>D: NixlHandshakePayload

    Note over D: 1. compatibility hash 검증<br/>(vLLM version, model, dtype, backend)
    Note over D: 2. add_remote_agent(remote_meta)
    Note over D: 3. remote descriptor 구성<br/>(block_id → RDMA 주소 매핑)

    Note over D: 이후 RDMA READ 가능
</div>

Handshake는 **lazy하고 비동기적**이다. Decode 워커가 특정 prefill 엔진의 KV를 처음 요청할 때 `ThreadPoolExecutor`에서 백그라운드로 수행된다. Handshake가 완료되기 전에 도착한 요청들은 큐에 쌓였다가, 완료 후 일괄 처리된다.

`NixlHandshakePayload`에는 **compatibility hash** (vLLM 버전, 모델 아키텍처, dtype, attention backend 등의 SHA-256)가 포함되어, P/D 엔진 간 설정 불일치를 조기에 감지한다.

### 2.3 KV 전송: One-sided RDMA READ

Handshake가 완료되면, decode 워커는 prefill 워커의 GPU 메모리를 **one-sided READ**로 직접 읽어온다. 이 과정에서 prefill 워커의 CPU는 전혀 개입하지 않는다.

[`start_load_kv()`](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py#L2448)가 이 전송을 시작한다:

```python
# nixl_connector.py — NixlConnectorWorker.start_load_kv()
def start_load_kv(self, metadata):
    for req_id, meta in metadata.reqs_to_recv.items():
        remote_engine_id = meta.remote.engine_id
        if remote_engine_id not in self._remote_agents:
            # handshake 미완료 → 백그라운드 handshake 시작, 요청 큐잉
            self._background_nixl_handshake(req_id, remote_engine_id, meta)
            continue
        # handshake 완료 → 즉시 RDMA READ 시작
        self._read_blocks_for_req(req_id, meta)
```

실제 RDMA 전송은 [`_read_blocks()`](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py#L2565)에서 수행된다:

```python
# nixl_connector.py — NixlConnectorWorker._read_blocks()
def _read_blocks(self, local_block_ids, remote_block_ids, dst_engine_id, ...):
    # block ID → NIXL descriptor index 매핑
    remote_descs = self._get_block_descs_ids(dst_engine_id, remote_block_ids)
    local_descs = self._get_block_descs_ids(self.engine_id, local_block_ids)

    # RDMA READ: remote(P) GPU → local(D) GPU
    handle = self.nixl_wrapper.make_prepped_xfer(
        "READ",
        local_xfer_handle,     # decode 쪽 descriptor
        local_descs,           # decode 쪽 block들
        remote_xfer_handle,    # prefill 쪽 descriptor
        remote_descs,          # prefill 쪽 block들
        notif_msg=notif_id,    # 완료 시 prefill에게 notification
    )
    self.nixl_wrapper.transfer(handle)  # 비동기 전송 시작
    self._recving_transfers[req_id].append(handle)
```

**데이터 흐름 요약:**

```
Prefill GPU KV block #3  ──[RDMA READ]──>  Decode GPU KV block #7
Prefill GPU KV block #5  ──[RDMA READ]──>  Decode GPU KV block #2
...
(block 단위, GPU-to-GPU 직접 전송, CPU 미경유)
```

전송 완료는 `get_finished()`에서 `check_xfer_state()`를 polling하여 확인하고, 완료 시 prefill 쪽에 notification을 보내 블록 해제를 허용한다.

### 2.4 블록 해제 프로토콜

P/D disaggregation에서 미묘한 문제가 하나 있다: **prefill 노드가 블록을 너무 일찍 해제하면, decode 노드가 RDMA READ를 하기 전에 데이터가 덮어써질 수 있다.**

NIXL Connector는 이를 **delayed block freeing + notification** 프로토콜로 해결한다:

1. Prefill의 요청이 완료되면 (forward pass 끝), scheduler는 블록을 바로 해제하지 않고 **지연**시킨다 (`request_finished()` → `delay_free=True`)
2. Decode가 RDMA READ를 완료하면, `notif_msg`가 prefill에게 전달된다
3. Prefill이 notification을 받으면 비로소 블록을 해제한다
4. Timeout 내에 notification이 오지 않으면 abort하고 블록을 해제한다 (`VLLM_NIXL_ABORT_REQUEST_TIMEOUT`)

### 2.5 `save_kv_layer()` / `wait_for_layer_load()`는 no-op

NIXL Connector에서 layer-by-layer pipelining 관련 메서드들은 **no-op**이다:

```python
def save_kv_layer(self, layer_name, kv_layer, attn_metadata):
    pass  # no-op — NIXL은 forward pass 중 개별 레이어 저장을 하지 않음

def wait_for_layer_load(self, layer_name):
    pass  # no-op — start_load_kv()에서 bulk transfer를 시작하고,
          #          forward pass 전에 완료되기를 기대
```

NIXL은 **bulk transfer** 모델이다: `start_load_kv()`에서 모든 블록의 RDMA READ를 한 번에 시작하고, forward pass가 시작되기 전에 완료되기를 기대한다. 레이어별 세밀한 파이프라이닝은 지원하지 않는다.

---

## 3. LMCache + NIXL (PDBackend) 상세

### 3.1 아키텍처: Transfer Channel 추상화

LMCache에서 NIXL은 **transfer channel**이라는 추상화 뒤에 숨어 있다. [`PDBackend`](https://github.com/LMCache/LMCache/blob/dev/lmcache/v1/storage_backend/pd_backend.py)가 `NixlChannel`을 사용하여 KV를 전송한다:

```
PDBackend
  └── NixlChannel (transfer_channel/nixl_channel.py)
        └── NixlAgentWrapper (NIXL agent + registered memory)
```

PDBackend는 **sender**(prefill)와 **receiver**(decode) 두 역할로 나뉜다. 이전 포스트에서 다룬 `LocalCPUBackend`와 같은 `StorageBackendInterface`를 구현하지만, 로컬 저장이 아니라 **원격 전송**을 수행한다.

### 3.2 메모리 등록: 별도 버퍼

NIXL Connector와의 **가장 근본적인 차이**가 여기에 있다. LMCache는 vLLM의 KV cache 텐서를 직접 등록하지 않는다. 대신 PDBackend가 **자체 버퍼를 할당하고 NIXL에 등록**한다:

```python
# pd_backend.py — PDBackend.__init__()
# PagedCpuGpuMemoryAllocator가 별도 버퍼 할당 (CPU 또는 GPU)
self.memory_allocator = PagedCpuGpuMemoryAllocator(
    buffer_size=config.pd_buffer_size,      # nixl_buffer_size에서 alias
    device=self.corrected_device,           # "cpu" 또는 "cuda:X"
)

# 이 버퍼를 NIXL에 등록
self.transfer_channel = CreateTransferChannel(
    channel_type=config.transfer_channel,   # "nixl"
    buffer_ptr=self.memory_allocator.buffer_ptr,
    buffer_size=self.memory_allocator.buffer_size,
    device=self.corrected_device,
)
```

```python
# nixl_channel.py — NixlAgentWrapper.__init__()
# 버퍼를 NIXL agent에 등록
memory_desc = [(buffer_ptr, buffer_size, tp_rank, "")]
reg_descs = nixl_agent.get_reg_descs(memory_desc, mem_type=mem_type)
nixl_agent.register_memory(reg_descs)

# 버퍼를 page 단위로 분할하여 transfer descriptor 생성
for base_addr in range(buffer_ptr, buffer_ptr + buffer_size, page_size):
    xfer_desc.append((base_addr, page_size, tp_rank))
xfer_handler = nixl_agent.prep_xfer_dlist("", xfer_descs, mem_type=mem_type)
```

**왜 별도 버퍼가 필요한가?**

vLLM의 KV cache는 **paged layout** (16-token block 단위, 물리적으로 흩어져 있음)이고, LMCache는 **contiguous chunk layout** (256-token chunk, 연속된 메모리)을 사용한다. NIXL RDMA 전송은 contiguous memory 단위로 동작하므로, vLLM의 흩어진 block들을 먼저 contiguous 버퍼에 모아야 한다. 이것이 [이전 포스트](/2026-03-25-lmcache-kv-offloading-codemap.html)에서 다룬 GPUConnector의 scatter/gather 역할이다.

### 3.3 P/D 전송 흐름: Sender (Prefill) 쪽

Prefill 노드에서 KV를 전송하는 전체 흐름을 단계별로 추적하자.

<div class="mermaid-wide">
sequenceDiagram
    participant W as vLLM Worker
    participant E as LMCacheEngine
    participant G as GPUConnector
    participant PD as PDBackend (sender)
    participant NC as NixlChannel
    participant R as Decode Node

    W->>E: store(token_ids, slot_mapping, transfer_spec)

    E->>G: batched_from_gpu(memory_objs, slot_mapping)
    Note over G: GPU paged KV → CPU/GPU contiguous buffer<br/>(CUDA kernel gather + D2H)
    G-->>E: MemoryObj[] 채워짐

    E->>PD: batched_submit_put_task(keys, memory_objs, transfer_spec)

    PD->>R: ZMQ REQ: AllocRequest(keys, shape, dtype)
    R-->>PD: ZMQ REP: AllocResponse(remote_indexes)
    Note over R: Decode 노드가 자기 버퍼에<br/>공간 할당 후 주소 반환

    PD->>NC: batched_write(memory_objs, remote_indexes)
    NC->>R: NIXL WRITE (one-sided RDMA)
    Note over NC: sender 버퍼 → receiver 버퍼<br/>직접 RDMA 전송

    PD->>R: ProxyNotif(req_id) — decode에게 전송 완료 알림
</div>

각 단계를 코드로 따라가면:

#### Step 1: GPUConnector — paged → contiguous 변환

이전 포스트에서 상세히 다룬 부분이다. `batched_from_gpu()`가 vLLM의 paged KV를 PDBackend 버퍼(CPU 또는 GPU)의 MemoryObj로 gather + 복사한다.

**이 단계가 NIXL Connector에는 없다.** NIXL Connector는 vLLM의 paged block을 직접 RDMA 단위로 사용하므로 format 변환이 불필요하다.

#### Step 2: Remote 메모리 할당 요청

PDBackend의 sender는 실제 데이터를 보내기 전에, decode 노드에게 **"이 크기의 공간을 할당해줘"**라고 요청한다:

```python
# pd_backend.py — _remote_allocate()
alloc_request = AllocRequest(
    keys=keys,
    shape=kv_shapes,
    dtype=kv_dtypes,
    last_chunk_toks=last_chunk_toks,
)
# ZMQ REQ/REP로 decode 노드에 전송
response = self._send_alloc_request(receiver_host, receiver_alloc_port, alloc_request)
# response.remote_indexes: decode 버퍼 내의 page index들
# response.already_sent_indexes: 이미 보낸 chunk (중복 제거)
```

Decode 노드의 receiver는 `_mem_alloc_loop()` 스레드에서 이 요청을 받아:
1. 각 key에 대해 중복 확인 (`self.contains(key)`)
2. 자신의 PDBackend 버퍼에서 메모리 할당
3. 할당된 page index를 `remote_indexes`로 반환

**이 프로토콜이 NIXL Connector와 다른 핵심 지점이다.** NIXL Connector는 decode가 prefill의 메모리를 READ하는 **pull 모델**이지만, LMCache PDBackend는 prefill이 decode에게 먼저 공간을 확보시킨 뒤 WRITE하는 **push 모델**이다.

#### Step 3: NIXL WRITE

할당된 remote index를 받으면, NixlChannel이 RDMA WRITE를 수행한다:

```python
# nixl_channel.py — NixlChannel.batched_write()
handle = self.nixl_agent.make_prepped_xfer(
    "WRITE",
    self.nixl_wrapper.xfer_handler,            # sender 쪽 descriptor
    self.get_local_mem_indices(memory_objs),    # sender 버퍼의 page index
    self.remote_xfer_handlers_dict[receiver_id], # receiver 쪽 descriptor
    transfer_spec["remote_indexes"],            # receiver 버퍼의 page index
)
self.nixl_agent.transfer(handle)
# poll until DONE
```

전송이 완료되면 ProxyNotif을 보내 decode 노드에 알린다.

### 3.4 P/D 전송 흐름: Receiver (Decode) 쪽

Decode 노드에서 KV를 수신하고 GPU에 적재하는 흐름:

<div class="mermaid-wide">
sequenceDiagram
    participant S as Prefill Node
    participant PD as PDBackend (receiver)
    participant E as LMCacheEngine
    participant G as GPUConnector
    participant W as vLLM Worker

    S->>PD: ZMQ REQ: AllocRequest
    PD-->>S: ZMQ REP: AllocResponse(remote_indexes)
    Note over PD: 버퍼에 공간 할당,<br/>key 등록 (데이터는 아직 없음)

    S->>PD: NIXL WRITE — RDMA로 데이터 도착
    Note over PD: RDMA가 직접 receiver 버퍼에 쓰기<br/>CPU 개입 없이 데이터 도착

    Note over PD: ProxyNotif 수신 → 데이터 준비 완료

    W->>E: retrieve(token_ids, slot_mapping)
    E->>PD: batched_get(keys)
    PD-->>E: MemoryObj[] (이미 데이터가 채워져 있음)

    E->>G: batched_to_gpu(memory_objs, slot_mapping)
    Note over G: contiguous buffer → GPU paged KV<br/>(CUDA kernel H2D + scatter)
    G-->>W: GPU KV cache 복원 완료
</div>

Receiver 쪽에서 주목할 점:

1. **AllocRequest 처리 시점에 데이터는 없다** — 공간만 할당하고 key를 등록한다
2. **RDMA WRITE가 도착하면** receiver 버퍼에 직접 데이터가 쓰여진다 — receiver CPU의 개입 없이
3. **`batched_get()`은 참조만 반환** — [이전 포스트](/2026-03-25-lmcache-kv-offloading-codemap.html)의 LocalCPUBackend와 동일한 패턴
4. **GPUConnector가 contiguous → paged 변환**을 수행 — 이전 포스트의 `to_gpu()` 경로

### 3.5 Handshake: 2-phase 메타데이터 교환

LMCache의 NIXL handshake도 ZMQ를 사용하지만, NIXL Connector보다 **한 단계가 더 있다**:

```python
# nixl_channel.py — lazy_init_peer_connection()

# Phase 1: NIXL agent 메타데이터 교환
send(NixlInitRequest(local_meta_bytes=agent.get_agent_metadata()))
recv(NixlInitResponse(remote_agent_name, remote_meta_bytes))
agent.add_remote_agent(remote_meta_bytes)

# Phase 2: Transfer descriptor 교환
send(NixlMemRegRequest(local_xfer_dlist_bytes))
recv(NixlMemRegResponse(remote_xfer_dlist_bytes))
# 양쪽이 상대방의 descriptor를 등록
```

Phase 1에서 NIXL agent가 서로를 인식하고, Phase 2에서 실제 메모리 descriptor를 교환한다. 이 과정이 완료되면 양방향 RDMA가 가능해진다.

---

## 4. 데이터 흐름 비교: 실제 메모리 복사 횟수

두 경로에서 prefill GPU의 KV가 decode GPU에 도달하기까지 **물리적인 메모리 복사가 몇 번 발생하는지** 비교하자.

### 4.1 NIXL Connector

```
Prefill GPU KV block ──[RDMA READ]──> Decode GPU KV block
```

| 복사 | 소스 → 목적지 | 비고 |
|------|-------------|------|
| 1회 | Prefill GPU → Decode GPU | NIXL RDMA READ (NIC DMA) |

**총 1회.** GPU 메모리 간 직접 전송. 중간 버퍼 없음.

### 4.2 LMCache + NIXL (CPU 버퍼)

`nixl_buffer_device: cpu` 설정 시:

```
P GPU paged KV ──[gather+D2H]──> P CPU buffer ──[RDMA WRITE]──> D CPU buffer ──[H2D+scatter]──> D GPU paged KV
```

| 복사 | 소스 → 목적지 | 비고 |
|------|-------------|------|
| 1회 | Prefill GPU → Prefill CPU | GPUConnector D2H (CUDA kernel) |
| 2회 | Prefill CPU → Decode CPU | NIXL RDMA WRITE |
| 3회 | Decode CPU → Decode GPU | GPUConnector H2D (CUDA kernel) |

**총 3회.** PCIe D2H + RDMA + PCIe H2D.

### 4.3 LMCache + NIXL (GPU 버퍼)

`nixl_buffer_device: cuda` 설정 시:

```
P GPU paged KV ──[gather]──> P GPU buffer ──[RDMA WRITE]──> D GPU buffer ──[scatter]──> D GPU paged KV
```

| 복사 | 소스 → 목적지 | 비고 |
|------|-------------|------|
| 1회 | Prefill GPU paged → Prefill GPU buffer | GPUConnector gather (GPU 내부) |
| 2회 | Prefill GPU buffer → Decode GPU buffer | NIXL RDMA WRITE (GPUDirect RDMA) |
| 3회 | Decode GPU buffer → Decode GPU paged | GPUConnector scatter (GPU 내부) |

**총 3회이지만**, 1회와 3회는 GPU 내부 메모리 복사 (PCIe 미경유)이므로 latency 영향이 작다. 실질적인 병목은 2회(RDMA)뿐이다.

**그렇다면 왜 NIXL Connector처럼 직접 전송하지 않는가?** LMCache의 content-addressed storage 체계 때문이다. vLLM의 block ID와 LMCache의 chunk hash는 다른 주소 체계이며, vLLM의 16-token paged block들을 LMCache의 256-token contiguous chunk로 변환하려면 GPU 내에서 scatter/gather가 필수다. 이 변환 과정이 곧 GPUConnector의 역할이고, NIXL Connector가 이 비용을 피할 수 있는 이유는 vLLM의 block을 변환 없이 그대로 전송하기 때문이다.

---

## 5. NIXL의 두 가지 용도: Transfer Channel vs Storage Backend

LMCache에서 NIXL은 **두 가지 완전히 다른 역할**로 사용될 수 있다. 이를 혼동하면 안 된다.

### 5.1 Transfer Channel (PDBackend)

위에서 다룬 것. **노드 간 실시간 KV 전송**에 사용된다.

```yaml
# LMCache config
transfer_channel: "nixl"    # PDBackend가 NIXL을 전송 채널로 사용
pd_buffer_device: cpu       # 또는 cuda
pd_buffer_size: 3019898880
```

UCX backend를 사용하여 RDMA (InfiniBand/RoCE) 위에서 동작한다.

### 5.2 Storage Backend (NixlStorageBackend)

**NIXL의 storage plugin을 사용하여 KV를 파일/오브젝트 스토어에 저장**한다. P/D 전송과는 무관하다.

```yaml
# LMCache config
extra_config:
  enable_nixl_storage: true
  nixl_backend: "GDS"        # 또는 "POSIX", "OBJ", "HF3FS"
  nixl_path: "/mnt/nvme/kv"
  nixl_pool_size: 1024
```

이 모드에서 NIXL은 `cuFile.write()`(GDS)나 POSIX `write()`를 래핑하는 **storage I/O 엔진**으로 동작한다. 네트워크 전송과는 무관하며, 이전 포스트에서 다룬 `LocalDiskBackend`이나 `GdsBackend`의 대안이다.

| | Transfer Channel | Storage Backend |
|---|---|---|
| **NIXL 역할** | 네트워크 전송 (RDMA) | Storage I/O (파일/오브젝트) |
| **NIXL backend** | UCX | GDS, POSIX, OBJ, HF3FS |
| **메모리 타입** | DRAM/VRAM ↔ DRAM/VRAM | DRAM/VRAM ↔ FILE/OBJ |
| **사용 backend** | `PDBackend`, `P2PBackend` | `NixlStorageBackend` |
| **대상** | 원격 노드 | 로컬/원격 스토리지 |

두 역할은 독립적이며 **동시에 사용**할 수 있다: PDBackend로 P/D 전송을 하면서, NixlStorageBackend로 SSD에도 저장하는 구성이 가능하다.

---

## 6. LMCache의 차별점: Offloading + P/D를 하나의 인터페이스에서

NIXL Connector는 **P/D disaggregation 전용**이다. CPU offloading, disk 저장, prefix caching은 별도 메커니즘이다.

LMCache는 이 모든 것을 **하나의 `LMCacheEngine.store()` / `retrieve()` 인터페이스**에서 처리한다. [이전 포스트](/2026-03-25-lmcache-kv-offloading-codemap.html)의 Appendix에서 다룬 것처럼, `StorageManager.batched_put()`이 모든 활성 backend에 데이터를 분배한다:

```python
# storage_manager.py — batched_put()
for backend in self.storage_backends:
    backend.batched_submit_put_task(keys, memory_objs, transfer_spec)
    # PDBackend: NIXL WRITE → decode 노드 전송
    # LocalCPUBackend: hot_cache[key] = memobj (참조 저장)
    # NixlStorageBackend: NIXL WRITE → SSD 저장
```

하나의 `store()` 호출에서:
- **PDBackend**: decode 노드에 NIXL WRITE로 KV 전송
- **LocalCPUBackend**: 로컬 CPU에 참조 저장 (multi-turn 재사용)
- **NixlStorageBackend**: SSD에 영구 저장 (eviction 후 복구용)

이 구조 덕분에, prefill 노드가 decode 노드에 KV를 보내면서 동시에 로컬 CPU에도 보관하여, 같은 prefix를 가진 후속 요청이 왔을 때 recompute 없이 CPU에서 바로 올릴 수 있다.

---

## 7. 정리: 어떤 경로를 선택할 것인가?

### NIXL Connector가 적합한 경우

- **최소 latency P/D 전송**이 최우선일 때 — GPU-to-GPU 직접 RDMA, 복사 1회
- **Heterogeneous TP**가 필요할 때 — P와 D의 TP 크기가 다른 환경
- **단순한 P/D 전용 파이프라인**일 때 — offloading 불필요, prefix caching은 vLLM 내장으로 충분

### LMCache + NIXL이 적합한 경우

- **Offloading과 P/D를 동시에** 사용할 때 — multi-turn agent 시나리오에서 CPU 캐싱 + P/D 전송 병행
- **다양한 storage backend**가 필요할 때 — CPU, SSD, Redis, P2P 등 플러그인 구성
- **Layer-wise pipelining**이 필요할 때 — 레이어별 전송으로 GPU idle 최소화
- **Content-addressed prefix caching**이 중요할 때 — token sequence 기반 hash로 cross-request prefix 공유

### 아키텍처 트레이드오프

| | NIXL Connector | LMCache + NIXL |
|---|---|---|
| **P/D 전송 latency** | 최소 (1회 RDMA) | 높음 (gather + RDMA + scatter) |
| **기능 범위** | P/D 전용 | Offloading + P/D + Storage |
| **코드 복잡도** | 높음 (3100 lines, self-contained) | 분산 (vLLM wrapper + LMCache 라이브러리) |
| **확장성** | vLLM 내부 수정 필요 | 플러그인 backend 추가 |
| **GPU 메모리 추가 사용** | 없음 (기존 KV cache 재사용) | PDBackend 버퍼 별도 할당 |

결국 **"전송 속도 vs 기능 범위"의 트레이드오프**다. 순수 P/D disaggregation만 필요하면 NIXL Connector가 최적이고, multi-tiered caching이 필요한 복잡한 시나리오에서는 LMCache가 그 오버헤드만큼의 가치를 제공한다.

---

## 8. 제3의 선택지: MultiConnector와 다른 Connector들

NIXL Connector 단독 vs LMCache + NIXL만이 선택지는 아니다. vLLM에는 **MultiConnector**로 여러 connector를 조합하는 방식과, Mooncake/FlexKV 같은 대안 connector도 있다.

### 8.1 MultiConnector: NIXL + OffloadingConnector 조합

vLLM의 [`MultiConnector`](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py)는 여러 `KVConnectorBase_V1`을 하나로 합성한다. NIXL(P/D 전송) + OffloadingConnector(CPU offloading)를 조합하면 LMCache 없이도 두 기능을 동시에 사용할 수 있다:

```json
{
  "kv_connector": "MultiConnector",
  "kv_connector_extra_config": {
    "connectors": [
      {"kv_connector": "NixlConnector", "kv_role": "kv_both"},
      {"kv_connector": "OffloadingConnector", "kv_role": "kv_both",
       "kv_connector_extra_config": {"cpu_bytes_to_use": 1073741824}}
    ]
  }
}
```

핵심 동작 원칙: **Load는 first-wins, Save는 all-connectors**.

- **Load**: 설정 순서대로 connector를 순회하며 `get_num_new_matched_tokens()` 호출. 먼저 `toks > 0`을 반환하는 connector가 load를 담당한다. NIXL이 첫 번째면 remote KV 우선, 없으면 OffloadingConnector가 CPU cache에서 load.
- **Save**: `save_kv_layer()` / `wait_for_save()`가 **모든 connector에** 호출된다. NIXL은 decode 노드에 전송, OffloadingConnector는 동시에 CPU에 offload.

이 조합의 P/D 전송은 NIXL의 직접 RDMA (1회 복사)를 유지하므로, LMCache의 3회 복사 오버헤드를 피한다.

#### MultiConnector의 한계

**1. Garbage KV 문제 ([#34526](https://github.com/vllm-project/vllm/issues/34526))**

가장 심각한 known issue였다. NIXL이 RDMA로 KV를 아직 load 중인데, OffloadingConnector가 같은 GPU 블록을 CPU에 offload하면 **초기화되지 않은 쓰레기 값**이 CPU에 저장된다. 후속 요청이 이 garbage KV를 CPU에서 load하면 accuracy corruption이 발생한다.

[PR #35092](https://github.com/vllm-project/vllm/pull/35092)에서 수정되었지만, 근본적으로 **connector 간 동기화 메커니즘이 부재**하다는 구조적 문제를 보여준다. MultiConnector는 각 connector를 독립적으로 호출할 뿐, connector 간 상태를 조율하는 계층이 없다.

**2. CPU-only offloading — SSD/disk 미지원**

`OffloadingConnector`의 유일한 구현체인 `CPUOffloadingSpec`은 **CPU 메모리만** 지원한다. SSD offloading은 구현되어 있지 않다. `OffloadingSpec`이 추상 클래스라 확장 가능하지만, 현재 코드에는 없다.

**3. 단일 노드 한정 — 분산 lookup 없음**

OffloadingConnector는 **로컬 CPU cache만** 관리한다. 다른 노드의 CPU cache를 조회하거나, 크로스-노드 KV 공유는 불가능하다. LMCache의 Cache Controller + P2P backend 같은 분산 lookup 메커니즘이 없다.

**4. Content-addressed caching 없음**

OffloadingConnector는 vLLM의 `BlockHash`(블록 단위 토큰 해시)를 사용한다. LMCache의 prefix hash chain(256-token chunk, 이전 chunk의 hash에 의존)보다 **granularity가 굵고**, cross-request prefix dedup 능력이 제한적이다. 정확한 block boundary에서의 exact match만 가능하다.

**5. Layer pipelining 없음**

OffloadingConnector의 `wait_for_layer_load()`와 `save_kv_layer()`는 **no-op**이다. 모든 전송이 bulk로 처리되어, forward pass 중 GPU idle이 발생할 수 있다.

**6. HMA 호환성 문제 ([#36547](https://github.com/vllm-project/vllm/issues/36547))**

Hybrid Memory Allocator(HMA) 마이그레이션 이후, MultiConnector가 `SupportsHMA` 인터페이스를 제대로 위임하지 않아 NIXL 전송이 **조용히 실패**하는 문제가 있었다. [PR #36549](https://github.com/vllm-project/vllm/pull/36549)에서 수정됨.

#### MultiConnector vs LMCache 비교

| | MultiConnector (NIXL + Offloading) | LMCache + NIXL |
|---|---|---|
| **P/D latency** | 최소 (1회 RDMA) | 높음 (3회 복사) |
| **CPU offloading** | 지원 (CPU만) | 지원 (CPU + SSD + Redis + ...) |
| **SSD offloading** | 미지원 | 지원 (GDS, POSIX, NIXL storage) |
| **분산 KV lookup** | 미지원 (로컬만) | Cache Controller + P2P |
| **Content-addressed caching** | block hash (vLLM 내장) | token hash chain (256-token chunk) |
| **Layer pipelining** | 미지원 (bulk) | 지원 (layerwise connector) |
| **Multi-turn 최적화** | 기본 (block hash dedup) | save_decode_cache + 다중 backend |
| **Heterogeneous TP** | 지원 (NIXL) | 미지원 |
| **안정성** | 알려진 race condition (#34526) | 단일 엔진 내 동기적 처리 |
| **외부 의존성** | 없음 (vLLM 내장) | LMCache 라이브러리 필요 |

### 8.2 Mooncake Connector

[MooncakeConnector](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/)는 [Mooncake TransferEngine](https://github.com/kvcache-ai/Mooncake)을 사용하는 P/D disaggregation connector다.

NIXL Connector와 비교하면:

| | NIXL | Mooncake |
|---|---|---|
| **전송 모델** | Pull (decode가 READ) | Push (prefill이 WRITE) |
| **Transport** | UCX/RDMA (NIXL) | Mooncake TransferEngine (RDMA/TCP) |
| **Discovery** | ZMQ handshake | HTTP bootstrap server |
| **Het. TP** | 지원 | 지원 |
| **PP (Pipeline Parallel)** | 지원 | 미지원 |
| **Offloading** | 미지원 (MultiConnector로 조합 가능) | 미지원 |

Mooncake도 **P/D 전용**이며 offloading은 지원하지 않는다. NIXL과의 핵심 차이는 **push vs pull 모델**: Mooncake는 prefiller가 decoder에게 KV를 보내는 push 방식이고, NIXL은 decoder가 prefiller의 GPU 메모리를 읽어가는 pull 방식이다.

알려진 이슈로 HBM leak ([#36014](https://github.com/vllm-project/vllm/issues/36014))이 있다.

### 8.3 FlexKV Connector

[FlexKVConnector](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/v1/flexkv_connector.py)는 [taco-project/FlexKV](https://github.com/taco-project/FlexKV) 라이브러리를 사용하는 connector다. LMCache와 유사하게 **multi-level caching** (CPU + SSD + remote storage)을 지원한다.

| | LMCache | FlexKV |
|---|---|---|
| **Storage 계층** | CPU, SSD (GDS), Redis, P2P, NIXL storage | CPU, SSD, remote (외부 라이브러리) |
| **Transfer 방식** | Worker에서 GPUConnector 경유 | Scheduler에서 관리 (worker는 no-op) |
| **Layer pipelining** | 지원 | 미지원 |
| **P/D disaggregation** | NIXL PDBackend | 미확인 |
| **vLLM 통합** | 12KB wrapper + 외부 라이브러리 | ~200 lines wrapper + 외부 라이브러리 |

FlexKV의 특이점은 **scheduler-side에서 모든 KV 전송을 관리**한다는 것이다. `start_load_kv()`, `save_kv_layer()`, `wait_for_save()` 모두 no-op이고, `request_finished()` 시점에 비동기로 offloading이 수행된다.

### 8.4 전체 Connector 지형도

<div class="mermaid-wide">
flowchart TD
    subgraph pd["P/D Disaggregation"]
        NIXL["NIXL Connector<br/>직접 RDMA, 1회 복사<br/>Het. TP 지원"]
        MOON["Mooncake Connector<br/>Push 모델, RDMA/TCP"]
    end

    subgraph offload["KV Offloading"]
        OFF["OffloadingConnector<br/>CPU only, block hash"]
    end

    subgraph hybrid["P/D + Offloading + Storage"]
        LMC["LMCache Connector<br/>NIXL PDBackend + LocalCPU<br/>+ SSD + Redis + P2P"]
        FKV["FlexKV Connector<br/>CPU + SSD + Remote"]
    end

    subgraph combo["조합"]
        MULTI["MultiConnector<br/>NIXL + Offloading 조합<br/>⚠️ race condition 이력"]
    end

    NIXL --> MULTI
    OFF --> MULTI

    style pd fill:#fce4e4,stroke:#e74c3c
    style offload fill:#fdf8f0,stroke:#e67e22
    style hybrid fill:#e8f4fd,stroke:#4a90d9
    style combo fill:#f5f0ff,stroke:#8e44ad
    style NIXL fill:#e74c3c,stroke:#333,color:#fff
    style MOON fill:#c0392b,stroke:#333,color:#fff
    style OFF fill:#e67e22,stroke:#333,color:#fff
    style LMC fill:#4a90d9,stroke:#333,color:#fff
    style FKV fill:#2980b9,stroke:#333,color:#fff
    style MULTI fill:#8e44ad,stroke:#333,color:#fff
</div>

정리하면:
- **P/D만 필요**: NIXL 또는 Mooncake
- **Offloading만 필요**: OffloadingConnector 또는 FlexKV
- **P/D + Offloading**: MultiConnector(NIXL + Offloading), LMCache, 또는 FlexKV
- **P/D + Offloading + 분산 caching + SSD**: LMCache가 현재 유일한 통합 솔루션
