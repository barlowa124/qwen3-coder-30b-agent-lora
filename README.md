# qwen3-coder-30b-agent-lora

Attention-LoRA fine-tune of Qwen3-Coder-30B-A3B-Instruct for local
coding-agent tool use. Trained on 189 real agent trajectories from a local
agent setup. Evaluated by a behavioral battery that executes real
read-only tool calls through an OpenAI-compatible chat shim. The battery
records each tool call plus the final answer per case.

## Contents

- `adapter/` - trained adapter (about 52 MB in safetensors) with tokenizer files
  and chat template. LoRA rank 16 with alpha 32 on the attention
  projections against an NF4-quantized base. Merged to a q4_k_m GGUF and
  served under Ollama as `qwen3-coder:30b-ft3`.
- `train_qwen3_lora.py` - TRL SFTTrainer + PEFT training script.
- `train_qwen3_lora_unsloth.py` - unsloth variant. It derives
  max_seq_length from the tokenized dataset so completion tokens are never
  truncated.
- `ailm_lora_colab.ipynb` - Colab notebook for the same adapter on A100 or
  L4. It merges the GGUF and publishes artifacts to Drive or a tunnel.
  A 4-bit 30B does not fit a free T4.
- `build_lora_dataset.py` - builds `lora_dataset.jsonl`, one row per
  assistant turn with loss masked to completion tokens. The dataset itself
  stays private since rows are real local trajectories containing personal
  data.
- `merge_lora_gguf.py` - adapter merge into base and GGUF export.
- `test_qwen_battery.py` - behavioral eval script. Replays agent turns
  against an Ollama tag through `:8015`. Executes whitelisted read-only
  tool calls for real and feeds tool results back into the turn. Records
  calls plus final answers per case.
- `evals/battery-qwen3-coder-30b-ft3.log` - battery run on the fine-tuned
  tag, run on October 26. Personal identifiers in recorded answers are redacted.

## Reproduce

```bash
python3 test_qwen_battery.py qwen3-coder:30b-ft3        # full battery
python3 test_qwen_battery.py qwen3-coder:30b-ft3 emails # one case
```

Requires a local Ollama with the merged tag and a shim on :8015 exposing
`/ollama/v1/chat/completions`.
