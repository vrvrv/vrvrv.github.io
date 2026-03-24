---
layout: post
category: llm-serving
title: "Speculative Decoding in vLLM v1 (2): EAGLE / EAGLE3"
---

[이전 포스트](/speculative-decoding-1.html)에서는 vLLM v1의 speculative decoding 사이클을 처음부터 끝까지 따라가며, `execute_model()`에서 target logits를 계산하고 `sample_tokens()`에서 rejection sampling과 draft proposal을 수행하는 전체 흐름을 살펴보았다. 그 과정에서 "GPU drafter는 bookkeeping 전에 실행된다", "hidden_states를 draft model에 전달한다" 같은 사실을 확인했지만, **draft model 내부에서 실제로 무슨 일이 일어나는지**는 블랙박스로 남겨두었다.

이 포스트는 그 블랙박스를 연다. EAGLE의 draft model은 정확히 어떤 구조인가? 단 1개의 Transformer layer가 어떻게 target model의 수십 개 layer를 흉내 내는가? "KV cache를 공유한다"는 것이 구체적으로 어떤 메커니즘인가 — draft model의 attention layer가 target model의 KV를 어떻게 읽는가? EAGLE3는 무엇을 바꿨고, 왜 그것이 acceptance rate를 올리는가?

* toc
{:toc}

---

## 1. EAGLE 모델 구조: 전체 그림 먼저

코드를 따라가기 전에, EAGLE draft model이 **정확히 어떤 형태인지** 먼저 그림으로 보자.

<div class="mermaid">
flowchart TD
    TGT["Target Model\n(Llama-3.1-8B, 32 layers)"]

    TGT -->|"last layer hs"| HS["target last HS\n[N, H]"]
    TGT -->|"Layer 2, 16, 29 hs"| AUX["aux hidden states\n[N, 3×H]"]:::e3

    EMB["embed_tokens\n(target weight 공유)"]

    HS --> FC["FC: cat·embed, hs·\n[2H → H]"]
    EMB -->|"EAGLE"| FC
    FC --> TFL

    AUX --> FC3["FC: [3H → H]\naux hs 압축"]:::e3
    FC3 --> NORM["hidden_norm"]:::e3
    EMB -->|"EAGLE3"| CAT["첫 layer QKV:\ncat·embed, hs· → [2H]"]:::e3
    NORM --> CAT
    CAT --> TFL

    TFL["Layer 32 · Transformer ·\nAttention: target KV cache 공유\n· 같은 block table, 같은 메모리 풀 ·"]
    TFL --> LMH["lm_head\n(target weight 공유)"]
    LMH --> OUT["draft logits → argmax\n→ draft_token_ids"]

    classDef e3 fill:#fff0f0,stroke:#e33,stroke-width:2.5px,color:#c00
    linkStyle 1,5,6,7,8,9 stroke:#e33,stroke-width:2.5px
</div>

> 검은 선 = **EAGLE** 경로 (target last hidden states + embed → `FC[2H→H]` → Transformer).
> <span style="color:#e33">붉은 선</span> = **EAGLE3**에서 변경/추가된 부분: target **3개 중간 layer**의 aux hidden states → <span style="color:#e33">`FC[3H→H]`</span> 압축 후, embed 통합이 FC가 아닌 <span style="color:#e33">첫 layer의 QKV input에서 cat</span>으로 수행.
> 두 경로 모두 같은 Layer 32(Transformer)와 lm_head를 거쳐 draft token을 생성한다.

**핵심을 먼저 말하면**: EAGLE draft model은 **자체 attention layer를 가진 독립적인 nn.Module**이다. 이 attention layer가 target model과 **같은 paged KV cache 메모리 풀**에서 read/write한다. "KV를 전달받는다"가 아니라, **같은 메모리 주소를 본다.**

이것이 가능한 구조를 이해하려면 세 가지를 알아야 한다:

1. **Draft model의 Transformer layer가 왜 `Layer 32`인가** — target model이 Layer 0-31이면 draft는 Layer 32로 이름을 붙인다
2. **이 naming이 KV cache 구조에서 어떤 의미인가** — vLLM은 layer name으로 KV cache를 인덱싱한다
3. **Draft model의 attention이 target의 historical KV를 어떻게 읽는가** — 같은 block table + seq_lens

하나씩 풀어보자.

---

## 2. 왜 Hidden States인가 — EAGLE의 핵심 아이디어

Speculative decoding의 원래 구상(Leviathan et al., 2023)은 단순하다: target model보다 훨씬 작은 독립된 draft model을 학습시키고, 이 모델이 제안한 token을 target model이 검증한다. 문제는 **draft model이 target model과 완전히 별개의 모델**이라는 점이다.

독립적인 작은 모델은 같은 데이터로 학습해도 target model과 분포가 괴리된다. 7B target에 68M draft를 붙이면 acceptance rate가 낮아지고, 1B draft를 붙이면 acceptance rate는 올라가지만 draft model 자체의 forward 비용이 커져서 latency 이득이 줄어든다. **모델 크기와 acceptance rate 사이의 trade-off**가 근본적인 병목이다.

EAGLE(Li et al., 2024)은 이 trade-off를 우회한다: next-token prediction은 **Transformer forward** (어려움)와 **LM head** (단순한 linear projection) 두 단계로 나뉜다. 어려운 것은 forward이고, LM head는 hidden state에 행렬 하나를 곱하는 것뿐이다.

> Target model이 이미 forward를 수행하고 hidden states를 내놓았다면, draft model은 forward를 처음부터 다시 할 필요가 없다. Hidden state를 입력으로 받아 **다음 step의 hidden state만 예측**하고, LM head는 target과 공유하면 된다 — **feature-level autoregression**.

<figure style="text-align:center; margin:1.5rem 0;">
<img src="https://arxiv.org/html/2401.15077v3/x3.png" alt="EAGLE feature-level autoregression" style="max-width:420px; width:100%;">
<figcaption style="font-size:0.8em; color:#666; margin-top:0.3rem;">Hidden state <code>f</code>에서 확률 분포 <code>p</code>로의 mapping은 단순한 LM head projection. EAGLE은 <code>f</code> 수준에서 autoregression한다.</figcaption>
</figure>

### 2.1 Draft Model의 구조: Target에 Layer를 추가한 것이 아니다

EAGLE의 draft model은 **FC layer + single Transformer layer**라는 극도로 작은 구조다. 여기서 흔한 오해가 하나 있다: "target model의 32개 layer 위에 33번째 layer를 얹은 것이니, target의 KV cache에 attend하는 것 아닌가?"

**아니다.** EAGLE의 draft model은 target model의 Layer 0~31과 완전히 독립된 별도의 모델이다.

> Draft model은 target의 KV cache를 읽지 않는다. 자기만의 KV 텐서를 가지고 있고, 매 forward마다 자기 hidden states에서 K/V를 계산하고, 자기 전용 KV 텐서에 저장하고, 이전에 자신이 저장해둔 KV를 읽어서 attention한다.

Target model에서 가져오는 것은 오직 **hidden states**(FC layer의 입력)뿐이다. 그 이후의 attention은 draft model 자체의 KV로 수행한다.

> Target model과 공유하는 것은 KV 데이터가 아니라 **block table과 slot mapping** — 즉 주소 체계다.

이 구조는 Section 3에서 자세히 설명한다.

아래 그림이 EAGLE의 전체 구조를 보여준다. 왼쪽의 target LLM이 hidden states `f`와 token embedding `e`를 생성하면, 오른쪽의 draft model이 이들을 입력으로 받아 autoregressive하게 draft token을 생성한다.

<figure style="text-align:center; margin:1.5rem 0;">
<img src="https://arxiv.org/html/2401.15077v3/x6.png" alt="EAGLE architecture" style="max-width:540px; width:100%;">
<figcaption style="font-size:0.85em; color:#666; margin-top:0.3rem;">EAGLE 전체 아키텍처. Target LLM(왼쪽)이 hidden states <code>f</code>를 생성하면, Draft Model(가운데)이 <code>(f, e)</code> 쌍을 입력으로 "One Auto-regression Head" (FC + Transformer layer)를 반복 실행하여 draft token을 생성한다. ★ = target과 weight 공유.</figcaption>
</figure>

---

## 3. KV Cache 공유: 무엇이 공유되고, 무엇은 각자의 것인가

"EAGLE은 target model과 KV cache를 공유한다"는 말은 자주 나오는데, 정확히 무엇이 공유되는 건지 혼동하기 쉽다. KV 데이터 자체를 공유하는 것인지, 아니면 다른 무언가를 공유하는 것인지. 이 섹션에서 정확히 풀어본다.

### 3.1 KV Cache의 물리적 구조부터 이해하기

먼저 vLLM의 paged KV cache가 GPU 메모리에 어떻게 존재하는지 보자. 33-layer model(target 32 + draft 1)이라면:

```
GPU 메모리
├── Layer 0  K_cache: [num_blocks × block_size × num_heads × head_dim]
│            V_cache: [num_blocks × block_size × num_heads × head_dim]
├── Layer 1  K_cache: [...]  V_cache: [...]
├── ...
├── Layer 31 K_cache: [...]  V_cache: [...]
└── Layer 32 K_cache: [...]  V_cache: [...]   ← EAGLE draft model 전용
```

**각 layer는 자기만의 K 텐서와 V 텐서를 가진다.** 코드에서 이것이 어떻게 분리되는지 보면 바로 이해된다. vLLM의 `unified_attention()`은 `layer_name`으로 KV cache를 lookup한다:

```python
# attention.py — get_attention_context()
def get_attention_context(layer_name: str):
    forward_context = get_forward_context()
    attn_layer = forward_context.no_compile_layers[layer_name]
    kv_cache = attn_layer.kv_cache[forward_context.virtual_engine]  # ← layer별 별도 텐서
    layer_slot_mapping = forward_context.slot_mapping[layer_name]   # ← layer별 slot mapping
    return attn_metadata, attn_layer, kv_cache, layer_slot_mapping

# 이 kv_cache가 attention kernel에 전달된다:
def unified_attention(query, key, value, layer_name):
    attn_metadata, self, kv_cache, _ = get_attention_context(layer_name)
    return self.impl.forward(self, query, key, value, kv_cache, attn_metadata)
```

`"model.layers.0.self_attn"`으로 호출하면 Layer 0의 KV 텐서를, `"model.layers.32.self_attn"`으로 호출하면 Layer 32의 KV 텐서를 가져온다. 같은 slot 번호(예: slot 90)라도 **다른 텐서의 slot 90**이므로 물리적 메모리 주소가 다르다.

그렇다면 "KV cache를 공유한다"는 것은 도대체 무슨 뜻인가?

### 3.2 공유되는 것: Block Table과 Slot Mapping (주소 체계)

Paged attention에서 KV cache는 **block** 단위로 관리된다. 각 request가 어떤 physical block을 사용하는지를 `block_table`이 기록한다:

```
block_table (request 0): [block 5, block 12, block 7, ...]
→ position 0~15는 block 5에, position 16~31은 block 12에, ...
```

**이 block_table은 모든 layer가 공유한다.** Request 0의 position 10에 대해:
- Layer 0은 block 5의 offset 10에 자기 K/V를 저장
- Layer 31은 block 5의 offset 10에 자기 K/V를 저장
- Layer 32(draft)도 block 5의 offset 10에 자기 K/V를 저장

"같은 주소"지만 "같은 메모리"가 아니다. 아파트에 비유하면: 101호, 201호, 3201호는 모두 "1동 01호 라인"이라는 같은 주소 체계를 공유하지만, 물리적으로 다른 공간이다. Layer 번호가 "층"이고, slot이 "호수"인 셈이다.

```
slot_mapping[i] = block_number × block_size + block_offset

예: position 10 → block 5, offset 10 → slot = 5 × 16 + 10 = 90

Layer 0:  K_cache[slot 90] = Layer 0이 계산한 position 10의 key
Layer 32: K_cache[slot 90] = Layer 32(draft)가 계산한 position 10의 key
                              ↑ 다른 텐서, 같은 인덱스
```

**공유되는 것**: block_table, slot_mapping, seq_lens — 즉 "어떤 position이 어떤 slot에 대응하는지"의 **주소 변환 규칙**

**공유되지 않는 것**: 실제 K/V 데이터 — 각 layer는 자기 전용 텐서에 read/write

### 3.3 코드에서 이것이 어떻게 구현되는가

3단계로 나뉜다: **(1) KV 텐서 할당** → **(2) 같은 group에 등록** → **(3) slot mapping 전달**.

**Step 1: KV 텐서 할당 — "옥탑방 만들기"**

vLLM은 초기화 시 `initialize_kv_cache_tensors()`에서 모든 attention layer에 대해 KV 텐서를 할당한다. 결과는 `kv_caches: dict[str, torch.Tensor]` — **layer_name을 key로 하는 dict**다:

```python
# gpu_model_runner.py — initialize_kv_cache_tensors()
# 결과:
kv_caches = {
    "model.layers.0.self_attn":  tensor([2, num_blocks, block_size, ...]),  # Layer 0
    "model.layers.1.self_attn":  tensor([2, num_blocks, block_size, ...]),  # Layer 1
    ...
    "model.layers.31.self_attn": tensor([2, num_blocks, block_size, ...]),  # Layer 31
    "model.layers.32.self_attn": tensor([2, num_blocks, block_size, ...]),  # ← EAGLE!
}
```

Target model만 있으면 Layer 0~31까지 32개의 KV 텐서가 할당된다. EAGLE이 추가되면 `model.layers.32.self_attn`이라는 이름의 **33번째 KV 텐서가 할당**된다. 이 이름은 `start_layer_id=target_layer_num`에서 온다:

```python
# llama_eagle.py — EAGLE draft model의 layer 이름
prefix = f"model.layers.{i + start_layer_id}"  # "model.layers.32"
```

이름만 32번이지 draft model의 첫 번째(유일한) layer다. 이 이름 덕분에 vLLM의 KV cache 시스템이 Layer 32 전용 텐서를 할당하는 것이다. 비유하면 32층 아파트에 옥탑방 1개를 추가한 것과 같다.

**Step 2: 같은 KV Cache Group — "같은 동, 같은 주소 체계"**

Draft layer는 target layer와 **같은 KV cache group**에 배치된다:

```python
# eagle.py — initialize_attn_backend()
for gid, group in enumerate(kv_cache_config.kv_cache_groups):
    if self._draft_attn_layer_names & set(group.layer_names):
        self.kv_cache_gid = gid    # target과 같은 group
```

같은 group = 같은 block pool = 같은 block_table. 따라서 모든 33개 layer가 "position X → slot Y"라는 **동일한 주소 변환 규칙**을 따른다. (Hybrid attention 모델에서는 drafter layer가 여러 group에 분산될 수 있어 [PR #35062](https://github.com/vllm-project/vllm/pull/35062)에서 drafter layer를 전용 group으로 분리하는 수정이 이루어졌다.)

**Step 3: Slot Mapping 전달 — "옥탑방에 주소 알려주기"**

Target model forward 시, `slot_mappings_by_layer`가 생성된다. 이것은 같은 group의 모든 layer에 **동일한 slot_mapping**을 매핑하는 dict다:

```python
# gpu_model_runner.py — target forward 시
slot_mappings_by_layer = {}
for gid, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
    slot_mapping = slot_mappings_by_gid[gid]           # position → slot 주소 배열
    for layer_name in kv_cache_group.layer_names:      # Layer 0~31
        slot_mappings_by_layer[layer_name] = slot_mapping
# → {"model.layers.0.self_attn": [90,91,92,...], ..., "model.layers.31.self_attn": [90,91,92,...]}
```

여기서 **Layer 32는 빠져있다** — target forward에서는 draft layer가 실행되지 않으므로. Draft model이 실행될 때 `_get_slot_mapping()`이 이 빈자리를 채운다:

```python
# eagle.py — _get_slot_mapping()
def _get_slot_mapping(self, num_tokens, slot_mapping=None):
    if slot_mapping is not None:
        self._slot_mapping_buffer[:num_actual].copy_(slot_mapping)  # target의 주소 복사
    view = self._slot_mapping_buffer[:num_tokens]
    return {name: view for name in self._draft_attn_layer_names}
    #      → {"model.layers.32.self_attn": [90, 91, 92, ...]}
```

> **target이 만든 slot_mapping(Layer 0~31용)을, draft layer 이름(Layer 32)으로 래핑**하는 것이다. 같은 주소 배열을 공유하되, Layer 32 전용 KV 텐서에 접근하게 된다.

이 dict를 `set_forward_context(slot_mapping=...)`에 전달하면, `get_attention_context("model.layers.32.self_attn")`가 Layer 32의 KV 텐서와 이 slot_mapping을 꺼내서 attention kernel에 넘긴다:

> **"KV cache 공유"의 정확한 의미**: K/V 데이터가 아니라, block table·slot mapping·block pool을 공유한다. 각 layer는 자기 전용 K/V 텐서에 read/write하되, "position X → slot Y" 규칙만 동일하다.

---

## 4. vLLM 구현: `EagleLlamaForCausalLM` 모델 코드

### 4.1 Forward Path

```python
# llama_eagle.py — LlamaModel.forward()
def forward(self, input_ids, positions, hidden_states):
    input_embeds = self.embed_tokens(input_ids)                  # [N, H]
    hidden_states = self.fc(
        torch.cat((input_embeds, hidden_states), dim=-1)         # [N, 2H] → [N, H]
    )
    residual = None
    for layer in self.layers:                                    # 보통 1회만 반복
        hidden_states, residual = layer(positions, hidden_states, residual)
    hidden_states = hidden_states + residual
    return hidden_states, hidden_states                          # (last, all)
```

세 단계다:

1. **`embed_tokens(input_ids)`**: sampled token을 embedding으로 변환. 이 embedding layer는 **target model과 weight를 공유**하므로 추가 메모리 0.
2. **`fc(cat(embed, hidden_states))`**: token embedding `[H]`과 target hidden state `[H]`를 concat하여 `[2H]`, 이를 다시 `[H]`로 projection. 이 FC layer가 "새 token이 context에 추가된 효과"를 하나의 representation으로 통합한다.
3. **단일 Transformer layer**: `LlamaDecoderLayer`를 통과. **첫 번째 layer의 `input_layernorm`이 `nn.Identity()`로 대체**되어 있다 — FC layer 출력이 이미 적절한 scale이므로 추가 normalization이 불필요하다. 이 layer의 attention이 Layer 32의 KV cache를 read/write한다.

출력은 `(hidden_states, hidden_states)` — 동일한 텐서를 두 번 반환한다. 첫 번째는 `lm_head`에 넘겨서 logits를 계산하는 데 쓰이고, 두 번째는 다음 draft step에서 다시 입력 `hidden_states`로 사용된다.

### 4.2 Weight Sharing

EAGLE draft model의 실제 **고유 파라미터**는 세 가지뿐이다:

| 파라미터 | Shape (Llama-3.1-8B 기준) | 설명 |
|---------|-------|------|
| `fc.weight` | `[4096, 8192]` | Embedding + Hidden state fusion |
| `layers[0].self_attn.*` | QKV + O proj | 단일 attention layer |
| `layers[0].mlp.*` | gate + up + down proj | 단일 MLP |

`embed_tokens`와 `lm_head`는 target model에서 가져온다:

```python
# eagle.py — load_model()
self._maybe_share_embeddings(target_language_model)
self._maybe_share_lm_head(target_language_model)
```

`_maybe_share_embeddings()`는 checkpoint에 embedding이 포함되어 있더라도 target과 동일하면(`torch.equal`) 공유로 전환한다 — EAGLE checkpoint 배포 방식의 다양성을 처리하기 위함이다.

**주의: 일부 EAGLE3 checkpoint는 자체 lm_head/embed_tokens가 필요하다.** 이전에는 vLLM이 항상 target의 weight로 덮어썼는데, [PR #28549](https://github.com/vllm-project/vllm/pull/28549)에서 이를 수정하여 acceptance length가 **1.32 → 2.50** (acceptance rate: 23% → 70%)으로 크게 향상되었다. 자체 vocabulary를 가진 EAGLE3 checkpoint에서는 target의 lm_head를 공유하면 오히려 성능이 떨어진다.

**메모리 관점**: Llama-3.1-8B 기준으로 fc layer ≈ 64MB, 단일 Transformer layer ≈ 200MB. **Target model 대비 ~3%의 추가 파라미터**만으로 draft model이 동작한다. Draft model 로딩 시 peak memory를 줄이는 최적화도 적용되어 있다 ([PR #24585](https://github.com/vllm-project/vllm/pull/24585)).

---

## 5. Draft Loop: `propose()`의 Autoregressive K-step

### 5.1 전체 흐름

`EagleProposer.propose()`는 K개의 draft token을 생성하기 위해 **K번의 autoregressive forward**를 수행한다. 먼저 `set_inputs_first_pass()`로 target의 hidden states와 token ids를 draft model 입력 형식으로 변환한 뒤, 첫 번째 forward로 draft token 1개를 생성한다 (argmax). 이후 K-1회의 loop에서 매번 `positions += 1`, `seq_lens += 1`, slot_mapping 재계산을 수행하고, 이전 draft token + 이전 hidden state를 입력으로 다음 draft token을 생성한다. 최종적으로 `torch.stack()`으로 `[batch, K]` 텐서를 반환한다.

첫 번째 forward와 이후 forward의 차이가 중요하다:

| | 첫 번째 Forward | 이후 Forward (K-1회) |
|--|----------------|---------------------|
| **input_ids** | target의 전체 scheduled tokens (shifted) | 이전 draft token 1개 |
| **hidden_states** | target model의 hidden states | 이전 draft forward의 hidden states |
| **num_tokens** | `num_scheduled_tokens` (가변) | `batch_size` (고정) |
| **attention** | prefill-like (여러 token) | decode-like (1 token/request) |
| **KV cache** | Layer 32에 다수 position write | Layer 32에 1 position write |

### 5.2 첫 번째 Forward: `set_inputs_first_pass()`

`propose()`가 호출되기 전에, `gpu_model_runner.py`의 `propose_draft_token_ids()`가 target model의 출력에서 draft에 필요한 데이터를 잘라낸다. 이때 `spec_decode_metadata`의 유무에 따라 두 경로로 나뉜다:

- **첫 draft** (`spec_decode_metadata=None`): 이전 step에 draft가 없었으므로, hidden states를 `[:num_scheduled_tokens]`로 그대로 잘라서 전달한다.
- **이후 draft** (padded): `prepare_inputs_padded()`가 reject된 token을 포함한 padded 입력을 구성하고, `token_indices_to_sample`(유효한 위치)과 `num_rejected_tokens_gpu`(각 request의 reject 수)를 반환한다. Hidden states는 reject된 position의 것까지 포함한 `[:total_num_tokens]`를 전달하지만, reject된 position의 결과는 `token_indices_to_sample`에 의해 sampling에서 무시된다.

이 데이터를 받은 `propose()` 내부의 `set_inputs_first_pass()`가 draft model의 입력 형식으로 변환한다. EAGLE의 입력은 target model의 입력과 **1 position 어긋나** 있다: target이 position `t`에서 hidden state `h[t]`를 출력하면, EAGLE은 `h[t]`와 position `t+1`에 올 token의 embedding을 받아 `h[t+1]`을 예측한다.

```python
# eagle.py — set_inputs_first_pass()
# 예: target input이 [a1, b1, b2, c1, c2, c3], next_token_ids가 [a2, b3, c4]

# Step 1: 왼쪽으로 shift
self.input_ids[:num_tokens - 1] = target_token_ids[1:]
# → [b1, b2, c1, c2, c3, c3]  (마지막은 아직 old)

# Step 2: 각 request의 마지막 위치에 next_token 삽입
self.input_ids[token_indices_to_sample] = next_token_ids
# → [a2, b2, b3, c2, c3, c4]

# Hidden states는 shift 없이 그대로 복사
self.hidden_states[:num_tokens] = target_hidden_states
```

Hidden states는 shift하지 않는다. `h[0]`은 position 0까지의 context를 반영한 벡터이고, 이것이 position 1의 token embedding(`target_token_ids[1]`)과 결합되어야 하므로, **input_ids만 shift하면 올바른 정렬이 된다.**

이 첫 번째 forward가 prefill-like인 이유는, 이번 step에서 target이 처리한 모든 token(1 real + K draft from previous step)에 대해 한 번에 draft model의 KV를 생성해야 하기 때문이다.

### 5.3 Autoregressive Loop

첫 번째 forward에서 `last_hidden_states[token_indices_to_sample]`을 sampling하여 첫 번째 draft token을 얻은 뒤, K-1회의 추가 forward가 반복된다:

```python
for token_index in range(self.num_speculative_tokens - 1):
    input_ids = draft_token_ids_list[-1].int()   # 이전 draft token
    positions += 1                                # position 1 증가
    common_attn_metadata.seq_lens += 1            # attention 범위 확장

    # slot_mapping 재계산 — 새 position의 KV cache slot
    block_numbers = clamped_positions // block_size
    block_ids = block_table_tensor.gather(dim=1, index=block_numbers.view(-1, 1)).view(-1)
    common_attn_metadata.slot_mapping = block_ids * block_size + clamped_positions % block_size

    # Draft model forward
    self.input_ids[:batch_size] = input_ids
    self.hidden_states[:batch_size] = hidden_states   # 이전 draft의 output

    with set_forward_context(..., slot_mapping=self._get_slot_mapping(...)):
        last_hidden_states, hidden_states = self.model(
            input_ids=..., positions=..., hidden_states=...)

    draft_token_ids = self._greedy_sample(last_hidden_states[:batch_size])
    draft_token_ids_list.append(draft_token_ids)
```

**주목할 점들:**

**Greedy sampling만 사용한다.** `_greedy_sample()`은 `compute_logits(hs).argmax(dim=-1)`이다. Temperature sampling을 지원하는 코드가 존재하지만 주석에 "Currently not used"라고 명시되어 있다. TP 환경에서는 이 argmax 과정이 all-gather 비용을 발생시키는데, [PR #34049](https://github.com/vllm-project/vllm/pull/34049)에서 local argmax + gather reduction으로 변경하여 통신량을 `O(batch × vocab_size)` → `O(batch × 2 × tp_size)`로 줄였다 (Llama4 Maverick TP=8 기준 TPOT **-1.3%**).

**`max_model_len` 초과 처리.** `positions >= max_model_len`인 request는 position을 0으로 clamp하고 slot_mapping을 `PADDING_SLOT_ID(-1)`로 설정한다. 이 request의 draft token은 의미 없지만, batch에서 제거하는 것보다 padding으로 유지하는 것이 CUDA kernel 효율상 유리하다.

**`seq_lens += 1`의 의미.** 매 iteration마다 attention 범위가 1 증가한다. 이전 draft forward에서 Layer 32에 write한 KV가 다음 draft forward의 attention 범위에 포함되므로, **draft model은 자기 자신의 이전 출력을 참조**할 수 있다.

**Autoregressive loop의 오버헤드.** 매 iteration마다 position 증가, slot_mapping 재계산, seq_lens 업데이트 등 여러 작은 PyTorch op이 실행되어 GPU kernel launch 오버헤드가 누적된다. [PR #33503](https://github.com/vllm-project/vllm/pull/33503)에서 이들을 하나의 Triton 커널로 fuse했다. CUDA graph 지원도 진행 중이다: [PR #34880](https://github.com/vllm-project/vllm/pull/34880)에서 drafter를 `CUDAGraphWrapper`로 감싸는 방식이 제안되었고, Model Runner V2에서는 [PR #35040](https://github.com/vllm-project/vllm/pull/35040)으로 Eagle3에 CUDA graph가 적용되었다.

---

## 6. GPU에서 CPU Sync 없이: `prepare_next_token_ids_padded()`

이전 포스트에서 "GPU drafter는 bookkeeping 전에 실행된다"는 사실을 확인했다. 즉, rejection sampling 결과를 CPU로 가져와서 "어떤 token이 accept되었는지" 정리하는 `_bookkeeping_sync()` — 이 안에 `.cpu().numpy()`라는 **blocking CUDA sync**가 있다 — 를 기다리지 않고, draft model forward를 먼저 실행한다는 뜻이다. Draft forward가 이 sync의 critical path에서 빠지므로 GPU utilization이 높아진다. 이것을 가능하게 하는 핵심 함수가 `prepare_next_token_ids_padded()`다.

### 6.1 문제

Rejection sampling의 출력 `sampled_token_ids`는 `[batch_size, max_spec_len + 1]` shape의 GPU 텐서다. Draft model의 다음 입력을 구성하려면 각 request에서 마지막으로 accept된 token이 무엇인지 알아야 한다. CPU로 가져와서 처리하면 CUDA sync가 발생하여 GPU가 idle 상태가 된다.

### 6.2 Triton 커널로 GPU에서 직접 처리

`eagle_prepare_next_token_padded_kernel`이 이 작업을 GPU에서 수행한다:

```python
def prepare_next_token_ids_padded(self, ...):
    eagle_prepare_next_token_padded_kernel[grid=(batch_size,)](
        sampled_token_ids,          # [B, max_spec+1] — rejection 출력
        discard_request_mask,       # [B] — prefill 중인 request 마스크
        backup_tokens_gpu,          # [B] — fallback token ids
        next_token_ids,             # [B] — 출력: 다음 draft의 시작 token
        valid_sampled_tokens_count, # [B] — 출력: accept된 token 수
        ...
    )
    return next_token_ids, valid_sampled_tokens_count
```

각 request에 대해: `sampled_token_ids[req]`를 스캔하여 유효한 token의 마지막 값(`next_token_ids`)과 개수(`valid_sampled_tokens_count`)를 구한다. 전체 과정에서 **GPU→CPU transfer가 발생하지 않는다.** [PR #28597](https://github.com/vllm-project/vllm/pull/28597)에서 이 함수들을 Triton 커널로 fuse하여 오버헤드를 **next_token_ids: 50μs → 5μs (10x)**, **prepare_inputs: 40μs → 2μs (20x)** 로 줄였다.

### 6.3 Padded Input 처리

Draft model은 reject된 token을 포함한 채로 입력을 받는다. 이전 step에서 draft 3개 중 1개가 reject되었어도, target이 처리한 4개 token 전부를 draft의 첫 forward에 넘긴다. `token_indices_to_sample`로 유효한 위치만 sampling하고 나머지는 버린다. 이것은 낭비처럼 보이지만, GPU에서 가변 길이 batch를 처리하는 것보다 고정 크기 padded batch가 CUDA kernel 효율이 높다.

**정리하면: `prepare_next_token_ids_padded()`가 rejection 결과를 GPU에서 직접 처리하여, draft가 bookkeeping(CPU sync) 전에 실행될 수 있게 한다.** 이 padded speculation은 [PR #24539](https://github.com/vllm-project/vllm/pull/24539)에서 도입되었으며 (Llama 3.1 8B + EAGLE3, 1xB200 기준 **TTFT/TPOT -2%**), 이후 unpadded 경로는 완전히 제거되었다 ([PR #35629](https://github.com/vllm-project/vllm/pull/35629)). Padded acceptance rate 계산의 보정은 [PR #29845](https://github.com/vllm-project/vllm/pull/29845)에서 이루어졌고, 이것만으로 H100+FA3 환경에서 **K=10 throughput +4.5% (5637→5890 tok/s), acceptance length +4.6% (3.25→3.40)** 향상이 있었다.

### 6.4 Draft Token의 CPU 전달: Async D2H Copy

`propose()`가 반환한 draft token IDs는 GPU 텐서 `[batch, K]`다. 이것을 scheduler에 전달하려면 CPU로 옮겨야 하는데, `_copy_draft_token_ids_to_cpu()`가 **전용 CUDA stream에서 async copy**를 수행한다:

```python
def _copy_draft_token_ids_to_cpu(self, scheduler_output, zeros_only=False):
    with torch.cuda.stream(self.draft_token_ids_copy_stream):
        self.draft_token_ids_copy_stream.wait_stream(default_stream)
        self.draft_token_ids_cpu[:num_reqs].copy_(draft_token_ids, non_blocking=True)
        self.draft_token_ids_event.record()
```

Copy 완료는 CUDA event로 기록하고, scheduler가 `take_draft_token_ids()`에서 실제로 값을 읽을 때 비로소 event를 synchronize한다 — **복사를 일찍 시작하고 동기화는 최대한 늦추는 패턴**이다. 이 async copy가 bookkeeping과 overlap된다.

---

## 7. EAGLE3: Auxiliary Hidden States

### 7.1 EAGLE의 정보 병목

EAGLE은 target model의 **마지막 layer hidden state만** 사용한다. 이것은 충분한 정보를 담고 있을까?

Transformer의 각 layer는 서로 다른 수준의 abstraction을 학습한다. 초기 layer는 local pattern(문법, 토큰 간 인접 관계)을, 후반 layer는 global semantics(문맥, 추론)를 처리한다. 마지막 layer의 hidden state는 이 모든 정보가 하나의 벡터로 압축된 결과이지만, 압축 과정에서 불가피하게 정보 손실이 발생한다 — 특히 초기 layer에서 포착된 low-level feature는 residual connection을 통해 전파되지만, 점진적으로 희석된다.

### 7.2 Multi-Layer Feature Aggregation

EAGLE3(Li et al., 2025)는 **target model의 중간 layer에서 추가 hidden states를 추출**한다. 마지막 layer만 보는 대신, 서로 다른 추상화 수준의 layer에서 feature를 가져와 draft model에 함께 제공한다. 아래 그림은 EAGLE3의 inference pipeline을 보여준다:

<figure style="text-align:center; margin:1.5rem 0;">
<img src="https://arxiv.org/html/2503.01840v1/x7.png" alt="EAGLE3 architecture" style="max-width:520px; width:100%;">
<figcaption style="font-size:0.85em; color:#666; margin-top:0.3rem;">EAGLE3 inference pipeline. Target model(왼쪽)의 여러 중간 layer에서 hidden states를 추출하고, FC Layer로 압축한 뒤 Decoder Layer에 전달한다. 각 step(①②③)에서 draft model은 이 multi-layer feature와 token embedding을 결합하여 autoregressive하게 draft token을 생성한다.</figcaption>
</figure>

vLLM에서 기본 추출 layer:

```python
# llama.py — LlamaForCausalLM.get_eagle3_aux_hidden_state_layers()
def get_eagle3_aux_hidden_state_layers(self):
    num_layers = len(self.model.layers)
    return (2, num_layers // 2, num_layers - 3)
```

Llama-3.1-8B (32 layers)에서는 layer **2, 16, 29**:
- **Layer 2**: 초기 — local pattern, token-level feature
- **Layer 16**: 중간 — intermediate representation
- **Layer 29**: 후반 — high-level semantics (마지막에서 3번째)

Target model forward 시, 지정된 layer에서 `hidden_states + residual` (pre-layer activation)을 추출하고, 이것들이 feature dimension으로 concat되어 `[N, 3*H]`가 된다. Draft model의 `combine_hidden_states()`가 FC layer `[3H → H]`로 압축한다:

```python
# llama.py — target model forward에서 추출
aux_hidden_states = []
for idx, layer in enumerate(self.layers):
    if idx in self.aux_hidden_state_layers:
        aux_hidden_states.append(hidden_states + residual)
    hidden_states, residual = layer(...)

# gpu_model_runner.py — concat
target_hidden_states = torch.cat(
    [h[:num_tokens] for h in aux_hidden_states], dim=-1)  # [N, 3*H]

# eagle.py — propose() 진입 시 압축
target_hidden_states = self.model.combine_hidden_states(
    target_hidden_states)   # FC: [N, 3*H] → [N, H]
```

### 7.3 EAGLE3의 Transformer Layer: Embedding 통합 방식 변경

EAGLE3의 draft model 구조는 EAGLE과 미묘하게 다르다. EAGLE에서는 `fc(cat(embed, hs))`로 embedding과 hidden state를 먼저 통합한 뒤 Transformer layer에 넣었지만, EAGLE3에서는 이 통합이 **첫 번째 layer의 attention 입력에서** 직접 일어난다:

```python
# llama_eagle3.py — LlamaDecoderLayer.forward()
def forward(self, positions, embeds, hidden_states, residual):
    if self.layer_idx == 0:
        # 첫 layer: embed와 hidden_states를 cat하여 QKV 입력으로
        embeds = self.input_layernorm(embeds)
        hidden_states, residual = self._residual_norm(hidden_states)
        hidden_states = torch.cat([embeds, hidden_states], dim=-1)  # [N, 2H]
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)

    hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
    ...
```

이를 위해 첫 번째 layer의 QKV projection 입력 크기가 `2*hidden_size`로 설정된다:

```python
qkv_input_size = 2 * self.hidden_size if layer_idx == 0 else self.hidden_size
```

### 7.4 EAGLE vs EAGLE3 아키텍처 비교

<div style="display: flex; gap: 1.5rem; align-items: flex-start;">
<div style="flex: 1; min-width: 0;">
<p style="margin:0 0 0.3rem"><strong>EAGLE</strong></p>
<div class="mermaid">
flowchart TD
    E["embed(token)"] --> FC["fc: [2H→H]"]
    H["target last HS"] --> FC
    FC --> TF["Transformer Layer\n(layernorm=Identity)"]
    TF --> LM["LM Head → logits"]
    style FC fill:#f96,stroke:#333,color:#000
</div>
</div>
<div style="flex: 1; min-width: 0;">
<p style="margin:0 0 0.3rem"><strong>EAGLE3</strong></p>
<div class="mermaid">
flowchart TD
    AUX["aux HS\n(3 layers)"] --> FC["fc: [3H→H]"]
    FC --> NORM["hidden_norm"]
    E["embed(token)"] --> LN["input_layernorm"]
    LN --> CAT["cat → [2H]"]
    NORM --> CAT
    CAT --> QKV["QKV Proj (2H→...)"]
    QKV --> ATT["Attention + MLP"]
    ATT --> LM["LM Head → logits"]
    style FC fill:#69f,stroke:#333,color:#000
    style CAT fill:#f96,stroke:#333,color:#000
</div>
</div>
</div>

| | EAGLE | EAGLE3 |
|--|-------|--------|
| **Target에서 가져오는 것** | 마지막 layer hidden state | 3개 layer의 hidden states |
| **FC layer** | `[2H→H]`: embed + hs fusion | `[3H→H]`: 3개 hs 압축 |
| **Embed 통합** | FC layer (Transformer 이전) | 첫 layer의 QKV projection |
| **추가 오버헤드** | 없음 | Target forward 시 3개 layer에서 hs 추출 |


---

## 8. 정리

<figure style="text-align:center; margin:1.5rem 0;">
<img src="https://arxiv.org/html/2503.01840v1/x3.png" alt="EAGLE family speedup comparison" style="max-width:640px; width:100%;">
<figcaption style="font-size:0.85em; color:#666; margin-top:0.3rem;">Speedup 비교. EAGLE3는 다양한 모델에서 기존 방법 대비 최대 5.6x speedup을 달성한다.</figcaption>
</figure>

EAGLE은 speculative decoding의 draft model 설계를 근본적으로 바꿨다. Target model과 독립된 작은 모델을 학습하는 대신, **target model의 hidden states 위에 기생하는 lightweight head**로 설계함으로써:

1. **Hidden states를 재활용**: `fc(cat(embed, hs))`로 새 token 정보를 통합하고, single Transformer layer로 다음 hidden state를 예측한다. Target의 32-layer stack이 이미 해놓은 작업 위에서 incremental update만 수행하므로, ~3%의 추가 파라미터로 충분하다.

2. **KV cache를 공유 — 같은 메모리, 다른 layer**: Draft model의 attention layer는 `model.layers.32`라는 이름으로 target과 같은 KV cache group에 속한다. 같은 paged block pool, 같은 block table을 사용하되, **자신만의 layer 슬롯에 KV를 read/write**한다. 별도의 KV 전달 메커니즘이 아니라, 주소 공간을 공유하는 구조다.

3. **GPU 경로로 CPU sync를 회피**: `prepare_next_token_ids_padded()`의 Triton 커널이 rejection 결과를 GPU에서 직접 처리하여, draft가 bookkeeping 전에 실행될 수 있게 한다.

EAGLE3은 여기에 **multi-layer feature aggregation**을 추가했다. Target model의 초기/중간/후반 layer에서 hidden states를 추출하여 `[3H → H]` FC로 압축함으로써, 마지막 layer만으로는 포착할 수 없는 multi-scale feature를 draft model에 제공한다.

다음 포스트에서는 또 다른 GPU drafter인 MTP(Multi-Token Prediction)와, Model Runner V2에서의 speculative decoding 구현 변경을 살펴본다.

---

## References

**논문**
- **EAGLE**: Li et al., "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty", ICML 2024. [arXiv:2401.15077](https://arxiv.org/abs/2401.15077)
- **EAGLE-2**: Li et al., "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees", EMNLP 2024. [arXiv:2406.16858](https://arxiv.org/abs/2406.16858)
- **EAGLE-3**: Li et al., "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test", 2025. [arXiv:2503.01840](https://arxiv.org/abs/2503.01840)
- **Speculative Decoding**: Leviathan et al., "Fast Inference from Transformers via Speculative Decoding", ICML 2023. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)

**vLLM PRs** ([vllm-project/vllm](https://github.com/vllm-project/vllm))
- [#15729](https://github.com/vllm-project/vllm/pull/15729) — EAGLE Proposer 최초 구현 (v1)
- [#16035](https://github.com/vllm-project/vllm/pull/16035) — EAGLE model loading
- [#18030](https://github.com/vllm-project/vllm/pull/18030) — Multi-layer eagle draft model 지원
- [#24539](https://github.com/vllm-project/vllm/pull/24539) — Efficient padded speculation (TTFT/TPOT -2%)
- [#24585](https://github.com/vllm-project/vllm/pull/24585) — EAGLE model loading peak memory 최적화
- [#26164](https://github.com/vllm-project/vllm/pull/26164) — EAGLE3 multi-layer decoder 지원 (첫 layer QKV input = 2H)
- [#28435](https://github.com/vllm-project/vllm/pull/28435) — EAGLE/EAGLE3 draft model quantization 지원
- [#28549](https://github.com/vllm-project/vllm/pull/28549) — 자체 lm_head/embed_tokens 지원 (AL 1.32→2.50)
- [#28597](https://github.com/vllm-project/vllm/pull/28597) — prepare_inputs Triton 커널 fuse (10-20x 오버헤드 감소)
- [#29845](https://github.com/vllm-project/vllm/pull/29845) — Padded speculation acceptance rate 보정 (throughput +4.5%)
- [#32887](https://github.com/vllm-project/vllm/pull/32887) — Unified Parallel Drafting (P-EAGLE, GPT-OSS 120B 1.52x)
- [#33503](https://github.com/vllm-project/vllm/pull/33503) — EAGLE step slot mapping/metadata Triton 커널 fuse
- [#34049](https://github.com/vllm-project/vllm/pull/34049) — TP communication 최적화 (local argmax reduction, TPOT -1.3%)
- [#34880](https://github.com/vllm-project/vllm/pull/34880) — EAGLE FULL CUDA Graph 지원
- [#35029](https://github.com/vllm-project/vllm/pull/35029) — Model Runner V2 Eagle3 지원
- [#35040](https://github.com/vllm-project/vllm/pull/35040) — Model Runner V2 Eagle3 CUDA graph
- [#35062](https://github.com/vllm-project/vllm/pull/35062) — Hybrid attention model drafter KV cache group 분리
- [#35629](https://github.com/vllm-project/vllm/pull/35629) — Unpadded drafter batch mode 제거 (padded only)
