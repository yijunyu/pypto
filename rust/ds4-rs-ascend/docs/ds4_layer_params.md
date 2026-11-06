# DS4.1Flash layer parameters, from the checkpoint's own config

Every value below is read from `/data0/DeepSeek-V4-Flash-0731-w8a8/config.json` (measured 2026-11-06),
not from memory. The right-hand column names the `LayerParams` field the engine's real layer path
(`decode_attn_layer_with`) expects, so the mapping can be made without guessing.

| `LayerParams` field | value | source in `config.json` |
|---|---|---|
| `layer_idx` | 0, 1, ... 42 | position |
| `d_embd` | 4096 | `hidden_size` |
| `n_hc` | 4 | `hc_mult` |
| `n_head` | 64 | `num_attention_heads` |
| `head_dim` | 512 | `head_dim` |
| `n_rot` | 64 | `qk_rope_head_dim` |
| `n_lora_q` | 1024 | `q_lora_rank` |
| `n_lora_kv` | 512 | `head_dim` (KV-down emits `n_lora_kv` floats; the rope tail is its last `n_rot`) |
| `n_lora_o` | 1024 | `o_lora_rank` |
| `n_out_group` | 8 | `o_groups` |
| `hc_sinkhorn_iter` | 20 | `hc_sinkhorn_iters` |
| `hc_eps` | 1e-6 | `hc_eps` |
| `rms_eps` | 1e-6 | `rms_norm_eps` |
| `compress_ratio` | 0, 0, 4, 128, 4, 128, ... | `compress_ratios[layer]` |
| `rope_orig_ctx` | 65536 | `rope_scaling.original_max_position_embeddings` |
| `rope_freq_base` | 10000 | `rope_theta` |
| `rope_freq_scale` | 1/16 | `rope_scaling.factor` = 16 |
| YaRN shape | `beta_fast` 32, `beta_slow` 1 | `rope_scaling` |
| compressed layers | `compress_rope_theta` 160000 | `compress_rope_theta` |

Other facts that change the arithmetic rather than just the sizes:

* **`num_key_value_heads = 1`** -- MQA. ONE 512-wide kv is shared by all 64 heads. A kernel sized for 64
  KV heads is wrong by 64x on the KV side.
* **`o_groups = 8`** with `group_dim = 64/8 * 512 = 4096` -- the output projection is GROUPED, not one
  32768-wide matmul. Independently corroborated by `LayerParams`'s own doc comment: "`attn_output_a` is
  row-major `[out_low_dim, group_dim]` where `out_low_dim = n_out_group * n_lora_o`. For V4 Flash,
  n_out_group=8."
* **`num_experts_per_tok = 6`**, `n_routed_experts = 256`, `n_shared_experts = 1`.
* **`num_hash_layers = 3`** -- layers 0-2 route by `ffn.gate.tid2eid` indexed by TOKEN ID, not by a
  top-k over router scores.
* **`expert_dtype = fp4`** in the config, but the shipped tensors are **I8 with per-output-channel
  scale/offset**, which is the w8a8 layout the kernels implement. The config names the training dtype,
  not the file's.

## What a faithful one-token forward still needs

The `hc_*` mechanism is NOT optional and NOT implemented in the hand-rolled example: layer 0 carries
`hc_attn_fn`/`hc_ffn_fn` [24, 16384], `hc_attn_scale`/`hc_ffn_scale` [3] and `hc_attn_base`/`hc_ffn_base`
[24]. The engine already implements it (`hc_collapse_norm`, `hc_expand_add_only`,
`hc_mean_collapse`), so the correct move is to drive `decode_attn_layer_with` with these parameters
rather than to reimplement the mechanism in an example.
