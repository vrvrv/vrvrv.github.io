---
layout: post
category: llm-serving
title: "Speculative Decoding in vLLM v1 (1): Workflow"
---

vLLM v0.17.1 (v1 engine) 기준으로 Speculative Decoding의 전체 코드 흐름을 정리한다.
v0.17.1에서 `GPUModelRunner`의 model execution은 두 메서드로 분리되어 있다:
- **`execute_model()`**: Preprocess + Target Model Forward → `execute_model_state`에 저장 후 `None` 반환
- **`sample_tokens()`**: Sampling + Rejection + Draft Proposal + Bookkeeping

`EngineCore`가 이 두 메서드를 순차적으로 호출하며, 그 사이에 `get_grammar_bitmask()` 등의 CPU 작업을 끼워넣어 GPU와 오버랩할 수 있다.

**목차**
- [1. Step N → Step N+1: 전체 사이클](#1-step-n--step-n1-전체-사이클)
- [2. execute_model() — Preprocess + Forward](#2-execute_model--preprocess--forward)
  - [2.1 _prepare_inputs(): Spec Decode Metadata 생성](#21-_prepare_inputs-spec-decode-metadata-생성)
  - [2.2 Forward Pass](#22-forward-pass)
- [3. sample_tokens() — Sample + Rejection + Draft + Bookkeeping](#3-sample_tokens--sample--rejection--draft--bookkeeping)
  - [3.1 Sampling: _sample()](#31-sampling-_sample)
  - [3.2 Rejection Sampling 상세](#32-rejection-sampling-상세)
  - [3.3 Draft Proposal: propose_draft_token_ids()](#33-draft-proposal-propose_draft_token_ids)
  - [3.4 Bookkeeping: _bookkeeping_sync()](#34-bookkeeping-_bookkeeping_sync)
- [4. Async Scheduling: step_with_batch_queue()](#4-async-scheduling-step_with_batch_queue)
- [5. Scheduler의 Spec Decode 관리](#5-scheduler의-spec-decode-관리)
- [6. 핵심 Data Structures](#6-핵심-data-structures)

---

## 1. Step N → Step N+1: 전체 사이클

정상 상태(steady state)에서는 **매 step마다 이전 draft를 검증하고, 동시에 다음 draft를 생성**한다.

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

**Speculative Decoding 활성화**: `_calc_spec_decode_metadata()`가 세 종류의 인덱스를 계산.

```
cu_num_scheduled_tokens:  [  4, 104, 107, 207, 209]
num_draft_tokens:         [  3,   0,   2,   0,   1]

logits_indices:         [ 0,  1,  2,  3, 103, 104, 105, 106, 206, 207, 208]
target_logits_indices:  [ 0,  1,  2,  5,  6,  9]   ← draft 위치만
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

> 주황 = `target_logits_indices` (draft 검증용), 파랑 = `bonus_logits_indices` (bonus token sampling용)

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
- **`hidden_states`** / **`aux_hidden_states`**: Eagle/Eagle3처럼 target model의 hidden states를 입력으로 받는 drafter가 draft proposal 시 사용한다.

이 데이터를 가지고 `sample_tokens()`는 크게 네 단계를 밟는다: **(1) sampling** → **(2) rejection sampling** → **(3) draft proposal** → **(4) bookkeeping**. 그런데 코드를 읽어보면 (3)과 (4)의 순서가 drafter 종류에 따라 뒤바뀐다.

**Bookkeeping(`_bookkeeping_sync()`)이 하는 일**부터 보자. Rejection sampling이 끝나면 "어떤 token이 최종 accept되었는가"를 request별 상태에 반영해야 한다 — GPU에서 sampled token을 CPU로 가져온 뒤(`output_token_ids.cpu().numpy()`), reject된 placeholder를 제거하고, `num_computed_tokens`, `all_token_ids` 갱신, output 집계 등 CPU 측 상태를 최신으로 만든다. 이 과정의 산출물이 `valid_sampled_token_ids` (`list[list[int]]`)다.

**왜 두 경로가 필요한가?** 문제는 이 `valid_sampled_token_ids`의 **생산 경로**에 있다. N-gram proposer는 CPU 메모리의 `token_ids_cpu` 배열에서 패턴 매칭을 하므로, bookkeeping이 GPU→CPU sync를 완료하고 CPU 측 이력을 갱신한 **뒤에야** draft를 생성할 수 있다. 즉 draft의 입력이 bookkeeping의 출력에 의존한다.

반면 Eagle이나 Medusa 같은 GPU 기반 drafter는 `sampler_output.sampled_token_ids` (GPU 텐서)를 그대로 입력으로 받고, rejection 처리도 GPU에서 자체적으로 수행한다 (`prepare_next_token_ids_padded()`). **GPU→CPU sync를 거치지 않는 별도의 데이터 경로**가 있기 때문에 bookkeeping을 기다릴 필요가 없다. 그래서 draft를 먼저 실행한 뒤 bookkeeping을 나중에 처리한다.

"그래도 항상 bookkeeping을 먼저 하면 코드가 단순하지 않은가?" — 맞다. 동작 자체는 문제없다. 하지만 `_bookkeeping_sync()`의 `.cpu().numpy()`는 **blocking CUDA sync**로, pending GPU 연산이 모두 끝날 때까지 CPU가 멈춘다. 이 sync 이전에 draft forward를 먼저 실행하면 sampled token이 GPU→CPU를 거치지 않고 바로 drafter로 들어가므로, draft가 이 sync의 critical path에서 빠진다. 실제로 이 설계를 도입한 [PR #24539](https://github.com/vllm-project/vllm/pull/24539)에서는 이를 "efficient padded speculation"이라 부르며, draft batch를 GPU에서 padded 형태로 직접 처리하도록 변경했다.

이 차이가 정말 유의미한지는 이후 PR이 증명한다. [PR #29184](https://github.com/vllm-project/vllm/pull/29184)에서는 N-gram proposer를 GPU로 재구현(`NgramProposerGPU`)하여 bookkeeping 전에 실행되도록 옮겼고, async scheduling과 결합해 **20-39%의 성능 향상**을 달성했다. 같은 알고리즘이 데이터 경로(CPU vs GPU)만 바꿔도 이 정도 차이가 나는 것이다.

<div style="display: flex; gap: 1.5rem; align-items: flex-start;">
<div style="flex: 1; min-width: 0;">
<p style="margin:0 0 0.3rem"><strong>Draft → Bookkeeping</strong></p>
<p style="margin:0 0 0.5rem; font-size:0.85em; color:#666;">
<code>use_gpu_toks = True</code><br/>
Eagle, Eagle3, MTP, Medusa, Draft Model<br/>
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
N-gram, Suffix Decoding<br/>
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

[Leviathan et al. (2023)](https://arxiv.org/abs/2211.17192)의 알고리즘을 Triton 커널로 구현한다. vLLM은 serving engine이므로 `temperature`는 per-request 파라미터다 — 한 batch 안에 `temperature=0`(greedy)과 `temperature>0`(random) request가 섞일 수 있다. 이를 처리하기 위해 **두 개의 Triton 커널을 순차 실행**하되 같은 `output_token_ids` 버퍼에 쓴다. 두 커널이 캐스케이드로 동작하는 것이 아니라, `is_greedy = (temperature == 0)` 마스크로 **request별로 담당 커널이 나뉜다** — 각 커널은 자기 담당이 아닌 request를 skip한다. (전부 greedy이거나 전부 random이면 `all_greedy`/`all_random` 플래그로 커널 하나만 실행한다.)

`rejection_sample()` 함수의 입력 shape과 전체 흐름:

```python
def rejection_sample(
    draft_token_ids,       # [N] — 전체 draft token (flattened)
    num_draft_tokens,      # list[int], len=B — request별 draft 수
    cu_num_draft_tokens,   # [B] — draft 수의 cumulative sum
    draft_probs,           # [N, V] or None — draft model 확률 분포
    target_logits,         # [N, V] — target model logits (draft 위치)
    bonus_token_ids,       # [B, 1] — bonus token (Sampler에서 사전 생성)
    # N = sum(num_draft_tokens), B = batch_size, V = vocab_size
):
    # output 버퍼: [B, max_spec_len + 1], PLACEHOLDER(-1)로 초기화
    output_token_ids = full(PLACEHOLDER_TOKEN_ID)

    # ── Greedy rejection ──
    if not all_random:
        target_argmax = target_logits.argmax(dim=-1)  # [N]
        rejection_greedy_sample_kernel(               # grid=(B,)
            output_token_ids, draft_token_ids,
            target_argmax, bonus_token_ids, is_greedy)
        if all_greedy:
            return output_token_ids

    # ── Random rejection 준비 ──
    target_probs = target_logits.softmax(dim=-1)      # [N, V]
    uniform_probs = generate_uniform_probs(...)        # [N], float64
    recovered_token_ids = sample_recovered_tokens(     # [N]
        target_probs, draft_probs, draft_token_ids)

    # ── Random rejection 실행 ──
    rejection_random_sample_kernel(                    # grid=(B,)
        output_token_ids, draft_token_ids,
        target_probs, draft_probs, bonus_token_ids,
        uniform_probs, recovered_token_ids, is_greedy)

    return output_token_ids  # [B, max_spec_len + 1]
```

각 함수를 좀 더 자세히 보면:

**`rejection_greedy_sample_kernel`**: grid `(B,)`. 각 request에 대해 draft position을 순회하며 매 위치에서 `target_argmax`를 출력한다. `draft_token != target_argmax`이면 그 위치의 target argmax를 마지막으로 출력하고 이후 draft를 모두 버린다 (이 경우 **bonus token은 추가되지 않는다**). 전부 accept된 경우에만 bonus token을 append한다.

**`generate_uniform_probs`**: acceptance 판정에 사용할 uniform random 값을 `[N]` shape으로 생성한다. **float64**를 사용하는데, 이는 PyTorch의 `torch.rand`가 float32에서 정확히 0.0을 반환할 수 있는 [known issue](https://github.com/pytorch/pytorch/issues/16706) 때문이다 — 0.0이 나오면 `p_target / p_draft ≥ 0`이 항상 성립하여 reject가 불가능해진다.

**`sample_recovered_tokens`**: reject 시 출력할 **recovered token을 사전에 계산**한다. Triton 커널을 grid `(B, max_spec_len)`으로 launch하여, 각 (request, position)에 대해 잔차 분포에서 Gumbel-max trick으로 sampling한다 (`argmax(prob * inv_q)`, inv_q = 1/Exp(1)). vocabulary를 `BLOCK_SIZE=8192` 단위로 tiling하며 running max를 유지한다. Rejection 루프 안에서 vocab 전체를 스캔하지 않도록, **미리 계산해두는 것이 핵심 최적화**다.

**`rejection_random_sample_kernel`**: grid `(B,)`. 각 request에 대해 draft position을 순회하며 `p_target[d[i]] / p_draft[d[i]] ≥ uniform_prob`이면 accept(draft token 출력). Reject 시 사전에 계산된 `recovered_token_ids[i]`를 출력하고 이후 draft를 버린다. Greedy와 마찬가지로 전부 accept된 경우에만 bonus token을 append한다.

**Reject 시 왜 bonus token을 추가할 수 없는가?** Target model은 forward pass에서 모든 draft token `d[0..K-1]`이 맞다고 가정하고 한번에 logits을 계산한다. Position i의 logit은 prefix + `d[0..i-1]`을 기반으로 계산되었으므로, `d[0..i-1]`이 모두 accept된 상태에서 position i의 reject/recovery는 유효하다. 하지만 position i+1 이후의 logit은 `d[i]`가 입력이라고 가정하고 계산된 것인데, 실제로는 recovered token(또는 target argmax)이 출력되어 **prefix가 달라졌으므로 후속 logit이 모두 무효**가 된다. Bonus logit 역시 `d[0..K-1]` 전체를 가정하고 계산되었으므로, 하나라도 reject되면 사용할 수 없다.

**Recovery란?** Draft가 reject되면 그 위치에서 어떤 token을 출력해야 할까. 단순히 target 분포에서 다시 sampling하면, accept 확률이 높았던 token(= draft와 target이 비슷한 token)이 과대 대표된다 — accept 경로에서도 출력되고, reject 후 재sampling에서도 뽑힐 수 있기 때문이다. 잔차 분포 `max(p_target - p_draft, 0)`은 "target이 원하지만 draft가 충분히 커버하지 못한 token"에 확률을 몰아주어, accept 경로와 합쳤을 때 정확히 target 분포를 복원한다.

**v0.17.1에서 `draft_probs`는 항상 `None`**: 원래 알고리즘의 acceptance 기준과 recovery 분포에는 draft model이 **각 speculative 위치에서 전체 vocabulary에 대해 계산한 확률 분포**가 필요하다. 즉 `[num_reqs, num_spec_tokens, vocab_size]` 크기의 draft logits 텐서를 저장해야 한다. vocab_size가 128k인 모델에서 spec token 5개, batch 256이면 이것만으로 ~320MB(fp16)다. v0.17.1에서는 이 비용을 피하기 위해 **모든 drafter 종류에 대해** `draft_probs=None`을 하드코딩한다. 이 경우 Triton 커널은 `draft_prob=1`로 간주하여, acceptance 기준이 `p_target(x) ≥ U(0,1)`로 단순화되고, recovery 시에도 draft token을 제외한 target 분포에서 바로 sampling한다. Target 분포를 엄밀하게 보존하지는 않는 근사적 방식이지만, draft logits 캐싱이 불필요하여 메모리 효율적이다.

Rejection sampling의 커널 구현과 확률적 보장에 대해서는 [3편](/2026-03-22-speculative-decoding-3.html)에서 자세히 다룬다.

**Output**: `[batch_size, max_spec_len + 1]` — reject된 slot은 `PLACEHOLDER_TOKEN_ID = -1`. `parse_output()`에서 placeholder 제거.

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

| 조건 | Draft 타이밍 | 이유 |
|------|-------------|------|
| `use_gpu_toks` + `input_fits_in_drafter` | **Bookkeeping 전** | GPU에 있는 sampled_token_ids를 바로 사용. CPU sync 불필요 |
| `use_gpu_toks` + `!input_fits_in_drafter` | **스킵** | Drafter의 max_model_len 초과. 빈 draft 반환 |
| `!use_gpu_toks` (ngram 등) | **Bookkeeping 후** | CPU에 있는 valid_sampled_token_ids 필요 |

<div class="mermaid">
flowchart TD
    A["propose_draft_token_ids()"] --> B{"method?"}
    B -- ngram --> C1["NgramProposer\nn-gram 매칭 (CPU)"]
    B -- medusa --> C2["MedusaProposer\nMultiple head 병렬 예측"]
    B -- "eagle/eagle3/mtp" --> C3["EagleProposer"]
    C3 --> D{"spec_decode_metadata?"}
    D -- "None (첫 step)" --> E1["target 전체 hidden states"]
    D -- "있음 (이후 step)" --> E2["accepted 위치만\n(rejected 제외)"]
    E1 --> F["Draft model forward\n× num_speculative_tokens"]
    E2 --> F
    F --> G["draft_token_ids: [batch, K]"]
</div>

### 3.4 Bookkeeping: `_bookkeeping_sync()`

Sampling이 끝난 후 그 결과를 request state에 반영하는 후처리:
- `num_computed_tokens` 업데이트
- `last_sampled_tokens`, `all_token_ids` 갱신
- `output_bin_counts`, `num_computed_prefill_tokens` 갱신
- `ModelRunnerOutput` 생성에 필요한 데이터 준비

---

## 4. Async Scheduling: `step_with_batch_queue()`

`--async-scheduling` 또는 Pipeline Parallel(`pp_size > 1`) 시 batch queue가 활성화된다. **Step N의 sample_tokens와 Step N+1의 execute_model을 오버랩**할 수 있다.

<div class="mermaid">
sequenceDiagram
    participant S as Scheduler
    participant EC as EngineCore
    participant EX as Executor

    rect rgba(200, 220, 255, 0.15)
    Note over S,EX: Step N
    EC->>EX: execute_model(N, non_block)
    EC->>S: get_grammar_bitmask(N)
    Note over EC: pending_structured_output 없으면 즉시 sample
    EC->>EX: sample_tokens(N, non_block)
    Note over EC: future를 batch_queue에 넣기
    end

    Note over EC: queue 안 찼으면 바로 다음 step!

    rect rgba(255, 220, 200, 0.15)
    Note over S,EX: Step N+1 (N과 병렬 가능!)
    EC->>EX: execute_model(N+1, non_block)
    Note over EC: deferred_scheduler_output 발생 가능<br/>(structured output 대기)
    end

    EX-->>EC: Step N sample_tokens 결과
    EC->>S: update_from_output(N)

    Note over EC: deferred가 있으면 여기서 sample
    EC->>EX: sample_tokens(N+1)
    Note over EC: batch_queue에 넣기

    EX-->>EC: Step N+1 결과
    EC->>S: update_from_output(N+1)
</div>

**Deferred sampling**: Structured output 사용 시, 이전 step의 token이 확정되어야 grammar bitmask를 계산할 수 있다. 이 경우 `deferred_scheduler_output`에 저장해두고, 이전 step의 `update_from_output()` 후에 `sample_tokens()`를 호출한다.

**Spec decode + async**: `post_step()`에서 async일 때는 draft token을 worker process에서 직접 업데이트하므로 scheduler에 별도로 전달하지 않는다. `step_with_batch_queue` 내부에서 deferred sampling 시에만 `take_draft_token_ids()`로 가져온다.

---

## 5. Scheduler의 Spec Decode 관리

Scheduler는 spec decode를 **unified scheduling** 관점에서 처리한다:

```python
# num_tokens_with_spec =
#   len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids)
num_new_tokens = request.num_tokens_with_spec - request.num_computed_tokens
```

Spec token도 "아직 계산되지 않은 token"으로 취급되어 일반 prefill/decode와 동일한 예산 관리 체계로 스케줄링된다.

### 5.1 Schedule 시: draft token 포함

```python
if request.spec_token_ids:
    num_scheduled_spec_tokens = (num_new_tokens + request.num_computed_tokens
                                 - request.num_tokens)
    if num_scheduled_spec_tokens > 0:
        scheduled_spec_decode_tokens[request.request_id] = request.spec_token_ids
```

### 5.2 Update 시: rejection 반영

```python
if scheduled_spec_token_ids:
    num_tokens_rejected = len(scheduled_spec_token_ids) + 1 - len(generated_token_ids)
    request.num_computed_tokens -= num_tokens_rejected
```

Draft 3개를 보냈는데 2개만 accept(+ bonus 1)되어 `generated_token_ids`가 3개면 → reject 1개 → `num_computed_tokens`를 1 감소시켜 다음 step에서 해당 위치부터 다시 계산하게 한다.

---

## 6. 핵심 Data Structures

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
