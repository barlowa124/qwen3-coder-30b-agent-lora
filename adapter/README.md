# qwen3-coder-30b-agent-lora adapter

LoRA adapter for Qwen/Qwen3-Coder-30B-A3B-Instruct, trained on 189 local
coding-agent trajectories.

- rank 16 with alpha 32, dropout 0.0, on the attention projections
- trained against an NF4-quantized base with TRL SFTTrainer + PEFT on one
  Colab A100/L4 run
- loss masked to completion tokens only
- deployed merged to q4_k_m GGUF under Ollama
- eval details in `../evals/battery-qwen3-coder-30b-ft3.log`
