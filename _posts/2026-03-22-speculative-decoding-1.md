---
layout: post
category: llm-serving
title: "Speculative Decoding in vLLM v1 (1): Overview"
---

Speculative decoding의 논문은 간결하다. Draft model이 token을 제안하고, target model이 검증하고, rejection sampling으로 틀린 token을 교정한다 — 알고리즘 자체는 한 페이지면 끝난다.

그런데 이걸 실제 serving engine에 넣으려고 하면 생각지 못한 질문들이 쏟아진다. Draft와 bookkeeping의 순서가 drafter 종류마다 왜 다른가? GPU drafter인 Medusa가 왜 CPU sync를 기다리는가? `draft_probs`를 왜 버렸고, 그 대가는 무엇인가? Rejection sampling에서 greedy와 random request가 한 batch에 섞이면 어떻게 처리하는가? 논문 어디에도 없는 greedy rejection 커널은 왜 존재하는가?

이 포스트는 vLLM (v1 engine)의 speculative decoding 코드를 한 step의 시작부터 끝까지 따라가며, 이런 질문들에 답한다.

**목차**
- [1. 전체 사이클](#1-전체-사이클)
- [2. execute_model() — Preprocess + Forward](#2-execute_model--preprocess--forward)
  - [2.1 _prepare_inputs(): Spec Decode Metadata 생성](#21-_prepare_inputs-spec-decode-metadata-생성)
  - [2.2 Forward Pass](#22-forward-pass)
- [3. sample_tokens() — Sample + Rejection + Draft + Bookkeeping](#3-sample_tokens--sample--rejection--draft--bookkeeping)
  - [3.1 Sampling: _sample()](#31-sampling-_sample)
  - [3.2 Rejection Sampling](#32-rejection-sampling)
  - [3.3 Draft Proposal: propose_draft_token_ids()](#33-draft-proposal-propose_draft_token_ids)
  - [3.4 Bookkeeping: _bookkeeping_sync()](#34-bookkeeping-_bookkeeping_sync)
- [4. Scheduler의 Spec Decode 관리](#4-scheduler의-spec-decode-관리)
- [5. 핵심 Data Structures](#5-핵심-data-structures)

---

## 1. 전체 사이클

vLLM v1 `GPUModelRunner`의 model execution은 두 메서드로 나뉜다:
- **`execute_model()`**: Preprocess + Target Model Forward → `execute_model_state`에 저장 후 `None` 반환
- **`sample_tokens()`**: Sampling + Rejection + Draft Proposal + Bookkeeping

`EngineCore`가 이 두 메서드를 순차적으로 호출하며, 그 사이에 `get_grammar_bitmask()` 등의 CPU 작업을 끼워넣어 GPU와 오버랩할 수 있다.

`EngineCore`는 **매 step마다 이전 draft를 검증하고, 동시에 다음 draft를 생성**한다.

<div class="mermaid-wide">
sequenceDiagram
    participant S as Scheduler
    participant EC as EngineCore
    participant EX as ModelExecutor
    participant MR as ModelRunner

    rect rgba(200, 220, 255, 0.15)
    Note over S,MR: Step N (정상 상태: draft 있음)

    S->>EC: schedule()<br/>spec_decode_tokens = {req: [d1,d2,d3]}

    rect rgba(100, 160, 255, 0.15)
    Note over EX,MR: execute_model (non_block=True)
    EC->>EX: execute_model(sched_output, non_block=True)
    EX->>MR: execute_model(sched_output)
    Note over MR: Preprocess: _update_states, _prepare_inputs<br/>→ spec_decode_metadata 생성
    Note over MR: Target Model Forward<br/>→ logits 계산
    Note over MR: execute_model_state에 저장, None 반환
    MR-->>EX: None (future)
    end

    EC->>S: get_grammar_bitmask(sched_output)
    Note over EC: GPU forward와 병렬로 grammar bitmask 계산

    EC->>EC: future.result() — forward 완료 대기

    rect rgba(255, 180, 120, 0.15)
    Note over EX,MR: sample_tokens
    EC->>EX: sample_tokens(grammar_output)
    EX->>MR: sample_tokens(grammar_output)
    Note over MR: Sample: bonus + rejection sampling
    Note over MR: Draft: propose_draft_token_ids()
    Note over MR: Bookkeeping: input_batch 업데이트
    MR-->>EX: ModelRunnerOutput
    EX-->>EC: ModelRunnerOutput
    end

    EC->>EC: post_step()<br/>draft_token_ids → scheduler에 전달
    EC->>S: update_from_output()<br/>reject 반영 + 새 spec_token_ids 저장
    end

    rect rgba(255, 220, 200, 0.15)
    Note over S,MR: Step N+1 (동일 흐름)
    S->>EC: schedule() — 새 drafts 포함
    Note over EX,MR: execute_model → sample_tokens
    EC->>S: update_from_output()
    end
</div>

**핵심 사이클** — 매 step에서 일어나는 일:
1. `EngineCore.step()`이 `execute_model(non_block=True)` 호출 → GPU forward 시작
2. Forward 대기 중 `get_grammar_bitmask()` 실행 (CPU-GPU 오버랩)
3. `sample_tokens(grammar_output)` 호출 → Sampling + Rejection + Draft Proposal
4. `post_step()`에서 draft token을 scheduler에 전달
5. `update_from_output()`에서 reject된 token 수만큼 `num_computed_tokens` 되돌림

```python
# vllm/v1/engine/core.py — step()
def step(self):
    scheduler_output = self.scheduler.schedule()
    future = self.model_executor.execute_model(scheduler_output, non_block=True)

    grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
 
    model_output = future.result()
    if model_output is None:
        model_output = self.model_executor.sample_tokens(grammar_output)

    engine_core_outputs = self.scheduler.update_from_output(
        scheduler_output, model_output)
```

---

## 2. `execute_model()` — Preprocess + Forward

`gpu_model_runner.py`의 `execute_model()`은 target model의 forward pass를 실행한 뒤, 결과를 `execute_model_state`에 저장하고 **`None`을 반환**한다. 실제 sampling은 하지 않는다.

<div class="mermaid">
flowchart TD
    A["execute_model(scheduler_output)"] --> B["_update_states()"]
    B --> C["_prepare_inputs()"]
    C --> D{"spec_decode_tokens?"}
    D -- No --> E1["logits_indices = last token만\nmetadata = None"]
    D -- Yes --> E2["_calc_spec_decode_metadata()\nlogits/target/bonus indices"]
    E1 --> F["_build_attention_metadata()"]
    E2 --> F
    F --> G["_preprocess() → input_ids, positions"]
    G --> H["Target Model Forward\n_model_forward()"]
    H --> I["hidden_states → logits 계산"]
    I --> J["execute_model_state에 저장\n(logits, metadata, hidden_states, ...)"]
    J --> K["return None"]
</div>

### 2.1 `_prepare_inputs()`: Spec Decode Metadata 생성

**일반 Decoding** (spec decode 없음): 각 request의 마지막 token 위치에서만 logit 추출.

```python
logits_indices = query_start_loc[1:] - 1
spec_decode_metadata = None
```

**Speculative Decoding 활성화**: 각 request는 1개의 real token(이전 step에서 accept된 token) + K개의 draft token = K+1개의 새 token을 target model에 넘긴다. Target model의 forward pass는 모든 입력 위치에서 next-token logit을 생성하므로, K+1개의 logit이 나온다. 이 중 앞 K개는 draft token 검증용(`target_logits_indices`), 마지막 1개는 bonus token sampling용(`bonus_logits_indices`)이다. Bonus logit은 마지막 draft token d[K-1] 위치에서 "그 다음 token"을 예측한 것으로, forward pass의 자연스러운 부산물이다.

`_calc_spec_decode_metadata()`가 세 종류의 인덱스를 계산.

모든 request가 decode 중인 단순한 예시로 보면 (각 request는 1개 real token + K개 draft token을 스케줄링):

```
num_draft_tokens:         [  3,   0,   2,   0,   1]
num_scheduled_tokens:     [  4,   1,   3,   1,   2]   ← 1 real + K draft
cu_num_scheduled_tokens:  [  4,   5,   8,   9,  11]

전체 scheduled tokens (11개):
  req0: [real, d0, d1, d2]  req1: [real]  req2: [real, d0, d1]  req3: [real]  req4: [real, d0]
  pos:    0    1   2   3             4             5    6   7            8             9   10
```

이 11개 위치에서 logit을 추출할 인덱스:

```
logits_indices:         [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10]  ← 11개 전부
                          ↓logits 배열 내 인덱스 (0~10)↓
target_logits_indices:  [ 0,  1,  2,  5,  6,  9]   ← draft 검증용 (K개씩)
bonus_logits_indices:   [ 3,  4,  7,  8, 10]        ← 각 request의 마지막 (bonus)
```

<div class="mermaid">
block-beta
    columns 11
    block:req0:4
        T0["0: draft"] T1["1: draft"] T2["2: draft"] B0["3: bonus"]
    end
    block:req1:1
        B1["4: bonus"]
    end
    block:req2:3
        T4["5: draft"] T5["6: draft"] B2["7: bonus"]
    end
    block:req3:1
        B3["8: bonus"]
    end
    block:req4:2
        T7["9: draft"] B4["10: bonus"]
    end

    classDef draft fill:#f96,stroke:#333
    classDef bonus fill:#69f,stroke:#333
    class T0,T1,T2,T4,T5,T7 draft
    class B0,B1,B2,B3,B4 bonus
</div>

> 주황 = `target_logits_indices` (draft 검증용), 파랑 = `bonus_logits_indices` (bonus token sampling용).
> Request 1, 3은 draft가 없으므로 bonus만 있다 — 일반 decode와 동일하게 마지막 위치의 logit으로 다음 token을 sampling한다.

### 2.2 Forward Pass

Target model forward 후 `logits_indices` 위치에서만 `compute_logits`를 수행한다.

```python
sample_hidden_states = hidden_states[logits_indices]
logits = self.model.compute_logits(sample_hidden_states)
```

| 경우 | `logits_indices` |
|------|-----------------|
| Spec decode 없음 | `query_start_loc[1:] - 1` — 각 request의 마지막 scheduled token |
| Spec decode 있음 | draft 검증 위치 + bonus 위치 (위 예제의 11개) |

결과는 `ExecuteModelState`에 저장된다: `logits`, `spec_decode_metadata`, `hidden_states`, `sample_hidden_states`, `aux_hidden_states`, `slot_mappings` 등.

---

## 3. `sample_tokens()` — Sample + Rejection + Draft + Bookkeeping

`sample_tokens(grammar_output)`은 `execute_model_state`를 꺼내서 실제 token 생성과 draft proposal을 수행한다. Spec decode 관점에서 이 state에 담긴 핵심 데이터는 세 가지다:

- **`logits`**: target model이 draft 위치 + bonus 위치에서 계산한 logits. Rejection sampling에서 draft token의 accept/reject를 판정하는 기준이 된다.
- **`spec_decode_metadata`**: `target_logits_indices`, `bonus_logits_indices`, `draft_token_ids` 등 — logits 텐서에서 어디가 draft 검증용이고 어디가 bonus용인지를 알려주는 인덱스 집합. 이게 `None`이면 일반 decoding이고, 있으면 rejection sampling 경로로 분기한다.
- **`spec_decode_common_attn_metadata`**: draft model이 forward할 때 사용할 attention metadata. Target model forward 시 생성되며, draft model은 target과 같은 KV cache를 공유하므로 동일한 block table, sequence length 정보를 재활용한다. Eagle/MTP 등 GPU drafter의 `propose()` 호출 시 전달된다.
- **`hidden_states`** / **`aux_hidden_states`**: Eagle/Eagle3처럼 target model의 hidden states를 입력으로 받는 drafter가 draft proposal 시 사용한다.

이 데이터를 가지고 `sample_tokens()`는 크게 네 단계를 밟는다: **(1) sampling** → **(2) rejection sampling** → **(3) draft proposal** → **(4) bookkeeping**. 그런데 코드를 읽어보면 (3)과 (4)의 순서가 drafter 종류에 따라 뒤바뀐다.

**Bookkeeping(`_bookkeeping_sync()`)이 하는 일**부터 보자. Rejection sampling이 끝나면 "어떤 token이 최종 accept되었는가"를 request별 상태에 반영해야 한다 — GPU에서 sampled token을 CPU로 가져온 뒤(`output_token_ids.cpu().numpy()`), reject된 placeholder를 제거하고, `num_computed_tokens`, `all_token_ids` 갱신, output 집계 등 CPU 측 상태를 최신으로 만든다. 이 과정의 산출물이 `valid_sampled_token_ids` (`list[list[int]]`)다.

**왜 두 경로가 필요한가?** 문제는 이 `valid_sampled_token_ids`의 **생산 경로**에 있다. N-gram proposer는 CPU 메모리의 `token_ids_cpu` 배열에서 패턴 매칭을 하므로, bookkeeping이 GPU→CPU sync를 완료하고 CPU 측 이력을 갱신한 **뒤에야** draft를 생성할 수 있다. 즉 draft의 입력이 bookkeeping의 출력에 의존한다.

반면 Eagle 같은 GPU 기반 drafter는 `sampler_output.sampled_token_ids` (GPU 텐서)를 그대로 입력으로 받고, rejection 처리도 GPU에서 자체적으로 수행한다 (`prepare_next_token_ids_padded()`). **GPU→CPU sync를 거치지 않는 별도의 데이터 경로**가 있기 때문에 bookkeeping을 기다릴 필요가 없다. 그래서 draft를 먼저 실행한 뒤 bookkeeping을 나중에 처리한다.

**Medusa는 모델 기반인데 왜 bookkeeping 이후에 실행되는가?** Medusa의 `MedusaProposer.propose()` 자체는 GPU에서 hidden states를 입력받아 동작한다. 문제는 그 **이전 단계**에 있다. Verification round에서 target model은 기존 token과 draft token을 함께 처리하므로, `sample_hidden_states`에는 모든 위치의 hidden states가 섞여 있다. Medusa에 넘겨줄 hidden state를 추출하려면 **각 request에서 마지막으로 accept된 token의 위치**를 알아야 하는데, 이는 rejection sampling 결과인 `valid_sampled_token_ids` (가변 길이 `list[list[int]]`)의 `len(tokens)`으로 결정된다. Eagle은 `prepare_next_token_ids_padded()`를 통해 이 인덱싱을 GPU에서 padded 텐서로 처리하는 경로가 있지만, Medusa에는 이에 상응하는 GPU 경로가 구현되어 있지 않다. 아키텍처 상의 한계가 아니라 **코드 성숙도 차이**로 보인다.

"그래도 항상 bookkeeping을 먼저 하면 코드가 단순하지 않은가?" — 맞다. 동작 자체는 문제없다. 하지만 `_bookkeeping_sync()`의 `.cpu().numpy()`는 **blocking CUDA sync**로, pending GPU 연산이 모두 끝날 때까지 CPU가 멈춘다. 이 sync 이전에 draft forward를 먼저 실행하면 sampled token이 GPU→CPU를 거치지 않고 바로 drafter로 들어가므로, draft가 이 sync의 critical path에서 빠진다. 실제로 이 설계를 도입한 [PR #24539](https://github.com/vllm-project/vllm/pull/24539)에서는 이를 "efficient padded speculation"이라 부르며, draft batch를 GPU에서 padded 형태로 직접 처리하도록 변경했다.

이 차이가 정말 유의미한지는 이후 PR이 증명한다. [PR #29184](https://github.com/vllm-project/vllm/pull/29184)에서는 N-gram proposer를 GPU로 재구현(`NgramProposerGPU`)하여 bookkeeping 전에 실행되도록 옮겼고, async scheduling과 결합해 **20-39%의 성능 향상**을 달성했다. 같은 알고리즘이 데이터 경로(CPU vs GPU)만 바꿔도 이 정도 차이가 나는 것이다.

**정리하면: (3) draft와 (4) bookkeeping의 순서는 drafter가 CPU 데이터에 의존하는지 여부로 결정된다. CPU drafter는 bookkeeping 후 필수, GPU drafter는 bookkeeping 전 실행으로 CUDA sync를 회피한다.**

<div style="display: flex; gap: 1.5rem; align-items: flex-start;">
<div style="flex: 1; min-width: 0;">
<p style="margin:0 0 0.3rem"><strong>Draft → Bookkeeping</strong></p>
<p style="margin:0 0 0.5rem; font-size:0.85em; color:#666;">
<code>use_gpu_toks = True</code><br/>
Eagle, Eagle3, MTP, Draft Model<br/>
GPU sampled ids를 바로 사용 — CPU sync 불필요
</p>
<div class="mermaid">
flowchart TD
    A["sample_tokens()"] --> B["state 언팩 + grammar bitmask"]
    B --> C["_sample()\nrejection sampling"]
    C --> D["propose_draft_token_ids\n(gpu_sampled_ids)"]
    D --> E["_bookkeeping_sync()"]
    E --> F["ModelRunnerOutput"]
    style D fill:#f96,stroke:#333,color:#000
    style E fill:#69f,stroke:#333,color:#000
</div>
</div>
<div style="flex: 1; min-width: 0;">
<p style="margin:0 0 0.3rem"><strong>Bookkeeping → Draft</strong></p>
<p style="margin:0 0 0.5rem; font-size:0.85em; color:#666;">
<code>use_gpu_toks = False</code><br/>
N-gram, Suffix Decoding, Medusa<br/>
CPU의 valid_sampled_token_ids가 필요
</p>
<div class="mermaid">
flowchart TD
    A["sample_tokens()"] --> B["state 언팩 + grammar bitmask"]
    B --> C["_sample()\nrejection sampling"]
    C --> D["_bookkeeping_sync()"]
    D --> E["propose_draft_token_ids\n(cpu_sampled_ids)"]
    E --> F["ModelRunnerOutput"]
    style D fill:#69f,stroke:#333,color:#000
    style E fill:#f96,stroke:#333,color:#000
</div>
</div>
</div>

> 주황 = Draft Proposal, 파랑 = Bookkeeping — 순서가 반전됨에 주목.
> GPU 기반 방법은 sampled token이 이미 GPU에 있으므로 bookkeeping(CPU sync) 전에 draft를 생성할 수 있다.
> CPU 기반 방법은 bookkeeping에서 확정된 valid token이 있어야 draft를 생성할 수 있다.

### 3.1 Sampling: `_sample()`

<div class="mermaid">
flowchart TD
    L["logits"] --> Q{"spec_decode_metadata?"}
    Q -- None --> S1["Sampler(logits) → [batch, 1]"]
    Q -- 있음 --> SPLIT["logits 분리"]
    SPLIT --> BL["bonus_logits"]
    SPLIT --> TL["target_logits"]
    BL --> S2["Sampler(bonus_logits) → bonus_token_ids"]
    TL --> RS["RejectionSampler(target_logits, drafts, bonus)"]
    S2 --> RS
    RS --> R2["output_token_ids: [batch, max_spec_len+1]"]
</div>

**Bonus token**: Target model이 마지막 draft 위치 **다음** 위치에서 생성하는 token. 모든 draft가 accept되었을 때 추가로 출력된다. Speculative decoding은 항상 **"accepted tokens + 1"**개를 출력하는데, 이 "+1"의 출처가 상황에 따라 다르다: 전부 accept되면 bonus token, 중간에 reject되면 해당 위치의 recovered token이 "+1"에 해당한다. 최악의 경우(첫 position에서 reject)에도 1개는 보장되므로, 일반 autoregressive decoding보다 나빠지지 않는다.

### 3.2 Rejection Sampling

[Leviathan et al. (2023)](https://arxiv.org/abs/2211.17192)의 알고리즘을 Triton 커널로 구현한다 ([PR #14930](https://github.com/vllm-project/vllm/pull/14930)). vLLM은 serving engine이므로 `temperature`는 per-request 파라미터다 — 한 batch 안에 `temperature=0`(greedy)과 `temperature>0`(random) request가 섞일 수 있다. 이를 처리하기 위해 **두 개의 Triton 커널을 순차 실행**하되 같은 `output_token_ids` 버퍼에 쓴다. 두 커널이 캐스케이드로 동작하는 것이 아니라, `is_greedy = (temperature == 0)` 마스크로 **request별로 담당 커널이 나뉜다** — 각 커널은 자기 담당이 아닌 request를 skip한다. (전부 greedy이거나 전부 random이면 `all_greedy`/`all_random` 플래그로 커널 하나만 실행한다.)

`RejectionSampler.forward()`부터 전체 흐름을 pseudocode로 보면:

```python
def forward(
    self,
    metadata: SpecDecodeMetadata,
    draft_probs,            # [N, V] or None
    logits,                 # [N + B, V] — target logits (draft 위치 N개 + bonus 위치 B개)
    sampling_metadata,
    # N = sum(num_draft_tokens), B = batch_size, V = vocab_size
):
    # ── logits 분리 ──
    target_logits = logits[metadata.target_logits_indices]  # [N, V]
    bonus_logits  = logits[metadata.bonus_logits_indices]   # [B, V]

    # ── target_logits에 sampling constraints 적용 ──
    target_logits = apply_sampling_constraints(target_logits)  # temperature, top-k, top-p

    # ── bonus token 사전 생성 ──
    bonus_token_ids = self.sampler(bonus_logits, ...)       # [B, 1]

    # ── rejection sampling ──
    output_token_ids = rejection_sample(
        metadata.draft_token_ids,      # [N]
        metadata.num_draft_tokens,     # list[int], len=B
        metadata.cu_num_draft_tokens,  # [B]
        draft_probs,                   # [N, V] or None
        target_logits,                 # [N, V]
        bonus_token_ids,               # [B, 1]
        sampling_metadata,
    )
    return parse_output(output_token_ids)  # placeholder 제거 → 가변 길이 list
```

`rejection_sample()` 내부:

```python
def rejection_sample(...):
    output_token_ids = full([B, max_spec_len + 1], PLACEHOLDER)  # -1로 초기화

    # ── Greedy rejection (is_greedy인 request만 처리) ──
    if not all_random:
        target_argmax = target_logits.argmax(dim=-1)                # [N]
        for req in range(B):
            if not is_greedy[req]: continue
            for i in range(num_draft_tokens[req]):
                output_token_ids[req][i] = target_argmax[offset + i]
                if draft_token_ids[offset + i] != target_argmax[offset + i]:
                    break                                           # reject → 이후 버림
            else:
                output_token_ids[req][K] = bonus_token_ids[req]     # 전부 accept → bonus
        if all_greedy:
            return output_token_ids                                 # softmax 불필요

    # ── Random rejection 준비 ──
    target_probs = target_logits.softmax(dim=-1)                    # [N, V]
    uniform_probs = generate_uniform_probs(...)                      # [N], float64
    recovered_token_ids = sample_recovered_tokens(...)               # [N]

    # ── Random rejection (!is_greedy인 request만 처리) ──
    for req in range(B):
        if is_greedy[req]: continue
        for i in range(num_draft_tokens[req]):
            if target_probs[offset+i][d[i]] / draft_probs[offset+i][d[i]] >= uniform_probs[offset+i]:
                output_token_ids[req][i] = draft_token_ids[offset+i]  # accept
            else:
                output_token_ids[req][i] = recovered_token_ids[offset+i]  # reject → recovery
                break                                                     # 이후 버림
        else:
            output_token_ids[req][K] = bonus_token_ids[req]               # 전부 accept → bonus

    return output_token_ids  # [B, max_spec_len + 1]
```

#### Greedy Rejection

`rejection_greedy_sample_kernel` (grid `(B,)`): 각 request에 대해 draft position을 순회하며 매 위치에서 `target_argmax`를 출력한다. `draft_token != target_argmax`이면 그 위치의 target argmax를 마지막으로 출력하고 이후 draft를 모두 버린다. 전부 accept된 경우에만 bonus token을 append한다.

**Greedy rejection은 논문에 별도 알고리즘으로 기술되어 있지 않다.** Leviathan et al.은 greedy decoding을 확률적 framework의 special case로 다룬다: `temperature=0`이면 target과 draft 분포가 모두 one-hot이 되어, acceptance 기준 `min(1, p_target(x) / p_draft(x)) ≥ r`이 단순한 equality check로 축약된다 — `draft == target_argmax`이면 `1/1 = 1 ≥ r`으로 항상 accept, 다르면 `0/1 = 0`으로 항상 reject. 수학적으로 동일하므로 softmax, uniform random, recovery 계산을 모두 건너뛸 수 있고, vLLM은 이를 별도의 Triton 커널로 분리하여 최적화했다 ([PR #14930](https://github.com/vllm-project/vllm/pull/14930)). `all_greedy`일 때 softmax를 아예 건너뛰는 추가 최적화도 적용되어 있다 ([PR #32852](https://github.com/vllm-project/vllm/pull/32852)).

**정리하면: greedy rejection은 확률적 rejection sampling의 degenerate case이며, vLLM은 이를 별도 커널로 분리하여 softmax와 recovery 계산을 생략한다.**

#### Random Rejection

**`generate_uniform_probs`**: acceptance 판정에 사용할 uniform random 값을 `[N]` shape으로 생성한다. **float64**를 사용하는데, 이는 PyTorch의 `torch.rand`가 float32에서 정확히 0.0을 반환할 수 있는 [known issue](https://github.com/pytorch/pytorch/issues/16706) 때문이다 — 0.0이 나오면 `p_target / p_draft ≥ 0`이 항상 성립하여 reject가 불가능해진다.

**`sample_recovered_tokens`**: reject 시 출력할 **recovered token을 사전에 계산**한다. Triton 커널을 grid `(B, max_spec_len)`으로 launch하여, 각 (request, position)에 대해 잔차 분포에서 Gumbel-max trick으로 sampling한다 (`argmax(prob * inv_q)`, inv_q = 1/Exp(1)). vocabulary를 `BLOCK_SIZE=8192` 단위로 tiling하며 running max를 유지한다. Rejection 루프 안에서 vocab 전체를 스캔하지 않도록, **미리 계산해두는 것이 핵심 최적화**다.

**`rejection_random_sample_kernel`** (grid `(B,)`): 각 request에 대해 draft position을 순회하며 `p_target[d[i]] / p_draft[d[i]] ≥ uniform_prob`이면 accept(draft token 출력). Reject 시 사전에 계산된 `recovered_token_ids[i]`를 출력하고 이후 draft를 버린다. Greedy와 마찬가지로 전부 accept된 경우에만 bonus token을 append한다.

#### Recovery

Draft가 reject되면 그 위치에서 어떤 token을 출력해야 할까. 단순히 target 분포에서 다시 sampling하면, accept 확률이 높았던 token(= draft와 target이 비슷한 token)이 과대 대표된다 — accept 경로에서도 출력되고, reject 후 재sampling에서도 뽑힐 수 있기 때문이다. 잔차 분포 `max(p_target - p_draft, 0)`은 "target이 원하지만 draft가 충분히 커버하지 못한 token"에 확률을 몰아주어, accept 경로와 합쳤을 때 정확히 target 분포를 복원한다.

**정리하면: reject 시 target에서 바로 재sampling하면 분포가 왜곡된다. 잔차 분포 `max(p_target - p_draft, 0)`에서 sampling해야 accept 경로와 합산했을 때 target 분포를 정확히 복원할 수 있다.**

#### `draft_probs`는 항상 `None` — Lossless하지 않다

원래 알고리즘의 acceptance 기준과 recovery 분포에는 draft model이 **각 speculative 위치에서 전체 vocabulary에 대해 계산한 확률 분포**가 필요하다. 즉 `[N, V]` 크기의 draft logits 텐서를 저장해야 한다. vocab_size가 128k인 모델에서 spec token 5개, batch 256이면 이것만으로 ~320MB(fp16)다. v0.17.1에서는 이 비용을 피하기 위해 **모든 drafter 종류에 대해** `draft_probs=None`을 하드코딩한다 ([PR #16899](https://github.com/vllm-project/vllm/pull/16899)). 이 경우 Triton 커널은 `draft_prob=1`로 간주하여, acceptance 기준이 `p_target(x) ≥ U(0,1)`로 단순화되고, recovery 시에도 draft token을 제외한 target 분포에서 바로 sampling한다.

**이 방식은 `temperature > 0`에서 lossless하지 않다.** Speculative decoding에서 "lossless"란 최종 출력 분포가 target model 단독 실행과 정확히 일치함을 의미하며, [Chen et al. (2023)](https://arxiv.org/abs/2302.01318)은 정확한 `p_target`과 `p_draft`를 사용한 rejection sampling이 이를 수학적으로 보장함을 증명했다. `draft_probs=None`은 draft 분포를 one-hot(argmax)으로 가정하는 것과 같은데, `temperature > 0`에서 target 분포는 여러 token에 걸쳐 퍼져 있으므로 draft의 argmax token과 불일치할 확률이 높아진다. 결과적으로 acceptance rate가 급락한다:

| Temperature | Acceptance Length (w/o probs) | Acceptance Length (w/ probs) |
|---|---|---|
| 0 | 2.37 | 2.37 |
| 0.7 | 2.30 | 2.31 |
| 1.0 | 2.09 | 2.21 |
| 1.5 | 1.13 | 2.07 |
| 2.0 | 1.01 | 2.66 |

(출처: [PR #20459](https://github.com/vllm-project/vllm/pull/20459), EAGLE on Llama-3.1-8B-Instruct)

`temperature=0`에서는 target도 one-hot이므로 차이가 없지만, `temperature=2.0`에서는 acceptance length가 **2.66 → 1.01**로 떨어진다 — 사실상 매 draft가 reject되어 speculative decoding의 이점이 사라진다. 이는 `p_target(x) / p_draft(x)` 비율을 정확히 계산하지 못해 rejection sampling의 수학적 보장이 깨지기 때문이다.

이 문제는 Model Runner V2에서 해결되었다: [PR #35461](https://github.com/vllm-project/vllm/pull/35461)에서 `rejection_sample_method="probabilistic"` 옵션이 추가되어, draft logits를 캐싱하고 정확한 확률 비율 기반 rejection sampling을 수행한다. Model Runner V2의 speculative decoding 구현에 대해서는 다음 포스트에서 다룬다.

**정리하면: v1 engine은 메모리 절약을 위해 `draft_probs`를 버렸고, 그 대가로 `temperature > 0`에서 lossless 보장이 깨진다. Model Runner V2가 도입되기 전까지 v1이 갖고 있는 구조적 한계다.**

**Output**: `[B, max_spec_len + 1]` — reject된 slot은 `PLACEHOLDER_TOKEN_ID = -1`. `parse_output()`에서 placeholder 제거하여 가변 길이 list로 변환.

### 3.3 Draft Proposal: `propose_draft_token_ids()`

Draft proposal의 타이밍은 **draft model 종류**와 **입력 크기**에 따라 달라진다:

```python
use_gpu_toks = (
    spec_config.use_eagle() 
    or spec_config.uses_draft_model()
    or spec_config.uses_extract_hidden_states()
) and not spec_config.disable_padded_drafter_batch

input_fits_in_drafter = (max_seq_len + num_spec_tokens <= effective_drafter_max_model_len)
```

`input_fits_in_drafter`는 `max_seq_len + num_spec_tokens <= effective_drafter_max_model_len`으로 계산된다. N-gram, suffix decoding은 `effective_drafter_max_model_len`이 target model의 `max_model_len`과 같으므로, 시퀀스가 거의 끝에 도달한 경우가 아니면 **사실상 항상 True**다. Eagle 등은 별도의 draft model이 있어 그 model의 `max_model_len`이 적용된다.

| 조건 | Draft 타이밍 | 이유 |
|------|-------------|------|
| `use_gpu_toks` + `input_fits_in_drafter` | **Bookkeeping 전** | GPU에 있는 sampled_token_ids를 바로 사용 |
| `use_gpu_toks` + `!input_fits_in_drafter` | **스킵** | Drafter의 max_model_len 초과. 빈 draft 반환 |
| `!use_gpu_toks` + `input_fits_in_drafter` | **Bookkeeping 후** | CPU에 있는 valid_sampled_token_ids 필요 |
| `!use_gpu_toks` + `!input_fits_in_drafter` | **스킵** | Draft 불필요 (시퀀스가 max_model_len에 근접) |

`propose_draft_token_ids()`(class method)의 흐름을 pseudocode로 보면:

```python
def propose_draft_token_ids(
    self,
    scheduler_output,
    sampled_token_ids,        # GPU tensor (use_gpu_toks) 또는 list[list[int]] (CPU drafter)
    sampling_metadata,
    hidden_states,            # target model hidden states (전체)
    sample_hidden_states,     # logits_indices 위치의 hidden states
    aux_hidden_states,        # EAGLE3용 보조 hidden states
    spec_decode_metadata,     # 이전 step의 draft 검증 정보 (첫 step이면 None)
    common_attn_metadata,     # draft model이 사용할 attention metadata
    slot_mappings,            # attention layer별 KV cache slot mapping
):
    if method == "eagle" or method == "draft_model":
        # accepted 위치의 hidden states 추출 + draft model forward
        # slot_mappings → draft model의 attention layer가 KV cache에 접근할 때 사용
        return self.drafter.propose(
            sampled_token_ids, hidden_states,
            common_attn_metadata, slot_mappings, ...)

    elif method == "medusa":
        # sample_hidden_states에서 마지막 accept 위치의 hidden state 추출
        # → Medusa head(MLP) forward. KV cache 없음 → slot_mappings 미사용
        return self.drafter.propose(
            target_hidden_states, sampling_metadata, slot_mappings=None)

    elif method == "ngram":
        # CPU token 이력에서 n-gram 패턴 매칭. 모델 forward 없음
        return self.drafter.propose(
            input_batch, sampled_token_ids, slot_mappings=None)
```

Eagle/DraftModel은 자체 attention layer를 가진 별도 모델이므로, draft forward 시 target model과 동일한 KV cache slot에 접근해야 한다. 그래서 `slot_mappings`(attention layer별로 KV cache의 어느 slot에 write할지를 지정하는 텐서)를 전달받아 사용한다. N-gram, suffix decoding은 모델 forward가 없고, Medusa는 MLP head만 있어 KV cache를 쓰지 않으므로 사용하지 않는다.

#### `_copy_draft_token_ids_to_cpu()`

Draft proposal이 끝나면 draft token ID를 scheduler에 전달해야 한다. 중요한 점은 **draft token은 `ModelRunnerOutput`에 포함되지 않는다**는 것이다. `ModelRunnerOutput.sampled_token_ids`는 이번 step에서 accept된 token(bookkeeping 산출물)이고, 다음 step을 위한 draft token은 `take_draft_token_ids()`라는 **별도 경로**로 전달된다.

GPU drafter(Eagle, NgramGPU 등)는 결과가 GPU 텐서(`[batch, K]`)이므로 CPU로 복사해야 하는데, `_copy_draft_token_ids_to_cpu()`는 이를 **전용 CUDA stream에서 async D2H copy**로 수행한다. Copy 완료를 CUDA event로 기록해두고, scheduler가 실제로 값을 읽을 때(`take_draft_token_ids()`) 비로소 event를 synchronize한다 — 복사를 일찍 시작하고 동기화는 최대한 늦추는 패턴이다. `ModelRunnerOutput`과 독립된 경로이므로 bookkeeping 전에 호출해도 문제가 없고, 오히려 그래야 bookkeeping과 D2H copy가 overlap된다.

CPU drafter(N-gram, Medusa 등)는 결과가 이미 `list[list[int]]`이므로, 이 함수가 호출되더라도 `not torch.is_tensor(draft_token_ids)` 체크에서 early return한다.

```
sample_tokens()의 두 가지 출력 경로:

[경로 1: accepted tokens]                    [경로 2: draft tokens]
_bookkeeping_sync()                          _copy_draft_token_ids_to_cpu()
  → valid_sampled_token_ids (CPU list)         → async D2H copy (side stream)
  → ModelRunnerOutput에 포함                    → ModelRunnerOutput에 포함되지 않음
  → IPC로 scheduler에 전송                      → take_draft_token_ids()로 별도 전달
  → scheduler.update_from_output()             → scheduler.update_draft_token_ids()
    → num_computed_tokens 보정                    → request.spec_token_ids 설정
```

### 3.4 Bookkeeping: `_bookkeeping_sync()`

`ModelRunnerOutput`은 scheduler process로 IPC를 통해 전송되므로, GPU 텐서가 아닌 **CPU의 list/dict** 형태여야 한다. `_bookkeeping_sync()`는 GPU 텐서인 `SamplerOutput.sampled_token_ids`를 CPU로 가져와서 `ModelRunnerOutput`에 넣을 수 있는 형태로 변환하는 **GPU↔CPU 동기화 경계**다.

구체적으로 하는 일:

**1. GPU→CPU 전송 + 정리** — Spec decode가 없으면 `_to_list()`로 단순 변환. Spec decode가 있으면 `RejectionSampler.parse_output()`이 `PLACEHOLDER_TOKEN_ID`(-1)를 제거하고, request별 **가변 길이 list**인 `valid_sampled_token_ids`를 생성한다. 예: 3개 draft 중 2개 accept → `[42, 87, 13]`, prefill 중인 request → `[]`.

**2. `input_batch` 캐싱** — `valid_sampled_token_ids`를 `input_batch.token_ids_cpu`에 기록하고 `num_tokens_no_spec`를 갱신한다. 이렇게 하면 다음 step의 `_prepare_inputs()`가 scheduler로부터 token을 다시 받지 않아도 된다 — scheduler가 sampled token을 echo할 필요가 없으므로 async scheduling이 가능해진다.

**3. `req_state.output_token_ids` 갱신** — 각 request의 `CachedRequestState.output_token_ids`에 새로 생성된 token을 append한다. 이 이력은 repetition penalty, logit processor 등이 이전 출력을 참조할 때 사용된다.

**4. `ModelRunnerOutput` 구성** — bookkeeping의 7개 반환값(`valid_sampled_token_ids`, `logprobs_lists`, `req_ids` 등)이 그대로 `ModelRunnerOutput`의 필드가 된다. 즉 `ModelRunnerOutput.sampled_token_ids`는 raw GPU 텐서가 아니라 bookkeeping이 정리한 **CPU list of list**이다.

```python
# _bookkeeping_sync() 반환 후 ModelRunnerOutput 구성
output = ModelRunnerOutput(
    req_ids=req_ids_output_copy,
    req_id_to_index=req_id_to_index_output_copy,
    sampled_token_ids=valid_sampled_token_ids,  # GPU tensor가 아님!
    logprobs=logprobs_lists,
    prompt_logprobs_dict=prompt_logprobs_dict,
)
```

Spec decode에서는 `valid_sampled_token_ids[i]`의 길이가 request마다 다르다 (1~K+1개). Scheduler는 이 길이를 보고 몇 개가 accept되었는지 파악하여 `num_computed_tokens`를 보정한다 (Section 4.2).

---

## 4. Scheduler의 Spec Decode 관리

Scheduler는 spec decode를 **unified scheduling** 관점에서 처리한다:

```python
# num_tokens_with_spec =
#   len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids)
num_new_tokens = request.num_tokens_with_spec - request.num_computed_tokens
```

Spec token도 "아직 계산되지 않은 token"으로 취급되어 일반 prefill/decode와 동일한 예산 관리 체계로 스케줄링된다.

### 4.1 Schedule 시: draft token 포함

```python
if request.spec_token_ids:
    num_scheduled_spec_tokens = (num_new_tokens + request.num_computed_tokens
                                 - request.num_tokens)
    if num_scheduled_spec_tokens > 0:
        scheduled_spec_decode_tokens[request.request_id] = request.spec_token_ids
```

### 4.2 Update 시: rejection 반영

```python
if scheduled_spec_token_ids:
    num_tokens_rejected = len(scheduled_spec_token_ids) + 1 - len(generated_token_ids)
    request.num_computed_tokens -= num_tokens_rejected
```

Draft 3개를 보냈는데 2개만 accept(+ bonus 1)되어 `generated_token_ids`가 3개면 → reject 1개 → `num_computed_tokens`를 1 감소시켜 다음 step에서 해당 위치부터 다시 계산하게 한다.

---

## 5. 핵심 Data Structures
### ExecuteModelState

`execute_model()`이 `sample_tokens()`에 전달하는 중간 상태:

| 필드 | 설명 |
|------|------|
| `scheduler_output` | 현재 step의 scheduling 결과 |
| `logits` | Target model의 logits |
| `spec_decode_metadata` | Draft 검증용 인덱스들 |
| `spec_decode_common_attn_metadata` | Draft model이 사용할 attention metadata |
| `hidden_states` | Target model의 전체 hidden states |
| `sample_hidden_states` | logits_indices 위치의 hidden states |
| `aux_hidden_states` | EAGLE3용 보조 hidden states |
| `slot_mappings` | KV cache slot mapping |

### SpecDecodeMetadata

| 필드 | Shape | 설명 |
|------|-------|------|
| `draft_token_ids` | `[num_draft_tokens_total]` | 검증 대상 draft token ID (flattened) |
| `num_draft_tokens` | `list[int]`, len=batch | 각 request별 draft token 수 |
| `cu_num_draft_tokens` | `[batch_size]` | draft token 수의 cumulative sum |
| `logits_indices` | `[num_draft + batch_size]` | logit 추출 위치 (draft + bonus) |
| `target_logits_indices` | `[num_draft_tokens_total]` | logits 내 draft 검증 위치 |
| `bonus_logits_indices` | `[batch_size]` | logits 내 bonus token 위치 |

### ModelRunnerOutput (spec decode 관련)

| 필드 | 설명 |
|------|------|
| `sampled_token_ids` | Accept된 token들 (가변 길이 list of list) |
| `spec_token_ids` | 다음 step을 위한 draft token (list of list) |

---

## 정리

vLLM v0.17.1의 Speculative Decoding은 **`execute_model` + `sample_tokens` 두 단계**로 나뉜다:

1. **`execute_model`**: Preprocess + Target Forward → 결과를 `execute_model_state`에 저장
2. **그 사이**: `get_grammar_bitmask()` 등 CPU 작업을 GPU forward와 오버랩
3. **`sample_tokens`**: Sampling → Rejection → Draft Proposal → Bookkeeping
   - Draft 타이밍은 `use_gpu_toks` × `input_fits_in_drafter`에 따라 bookkeeping 전/후로 결정
4. **Async scheduling**: `step_with_batch_queue`에서 Step N의 sample_tokens와 Step N+1의 execute_model을 오버랩. Structured output 사용 시 deferred sampling

다음 포스트에서는 EAGLE/EAGLE3의 draft model 내부 구조와 KV cache 관리를 더 깊이 살펴본다.
