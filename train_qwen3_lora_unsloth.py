"""Unsloth 4-bit LoRA fine-tune of Qwen3-Coder-30B-A3B on local-agent
trajectories. Attention adapters only; behavioral tuning, not knowledge.
"""
import gc
import json
import os
import sys

sys.path = [p for p in sys.path if os.path.abspath(p or ".") != "/home/asdf/Work"]

from unsloth import FastLanguageModel
from unsloth_zoo.block_swap import estimate_training_reserve_bytes, find_decoder_layers
from unsloth_zoo.device_map_planner import build_meta_model
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL_DIR = sys.argv[1]
DATA = sys.argv[2] if len(sys.argv) > 2 else \
    "/home/asdf/Work/ailm-stack-scripts/lora_dataset.jsonl"
OUT = "/home/asdf/Work/ailm-stack-scripts/q3c_lora_out"
LORA_RANK = 16
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Derive max_seq_length empirically from the actual tokenized dataset so 100% of
# completion tokens are preserved without arbitrary magic-number truncation.
_tok_probe = AutoTokenizer.from_pretrained(MODEL_DIR)
_obs_max_tokens = max(
    len(_tok_probe.encode(row["prompt"] + row["completion"]))
    for row in (json.loads(line) for line in open(DATA, encoding="utf-8") if line.strip())
)
MAX_SEQ_LENGTH = ((_obs_max_tokens + 63) // 64) * 64

# Derive training reserve_bytes empirically for the actual attention-only LoRA
# target_modules so Unsloth's block-swap planner does not charge 11.97 GiB for
# unused MoE expert LoRA adapters.
_meta_model, _hf_q, _meta_cfg = build_meta_model(MODEL_DIR, load_in_4bit=True)
_meta_layers = find_decoder_layers(_meta_model)
_attn_linears = [
    m
    for layer in _meta_layers
    for name, m in layer.named_modules()
    if name.endswith(tuple(TARGET_MODULES))
]
_lora_attn_bytes = sum((m.in_features + m.out_features) * LORA_RANK for m in _attn_linears) * 20
RESERVE_BYTES = int(
    estimate_training_reserve_bytes(_meta_cfg, MAX_SEQ_LENGTH, 1, extra_bytes=_lora_attn_bytes)
)
del _meta_model, _meta_layers, _attn_linears
print(
    f"empirical dataset max_tokens={_obs_max_tokens} -> max_seq_length={MAX_SEQ_LENGTH} "
    f"attention_lora_reserve={RESERVE_BYTES / (1024**3):.2f} GiB"
)

kwargs = {
    "model_name": MODEL_DIR,
    "max_seq_length": MAX_SEQ_LENGTH,
    "load_in_4bit": True,
    "offload_layers": "auto",
    "device_map_planner_kwargs": {"reserve_bytes": RESERVE_BYTES},
}
try:
    model, tok = FastLanguageModel.from_pretrained(**kwargs)
except ValueError:
    gc.collect()
    torch.cuda.empty_cache()
    model, tok = FastLanguageModel.from_pretrained(**kwargs)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=TARGET_MODULES,
    lora_alpha=32,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

ds = load_dataset("json", data_files=DATA, split="train")
print("rows:", len(ds))

cfg = SFTConfig(
    output_dir=OUT,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=3,
    logging_steps=1,
    save_strategy="no",
    bf16=True,
    max_length=MAX_SEQ_LENGTH,
    completion_only_loss=True,
    optim="adamw_8bit",
    report_to=[],
    seed=42,
)

trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                     processing_class=tok)
trainer.train()
model.save_pretrained(OUT)
tok.save_pretrained(OUT)
print("saved adapter ->", OUT)
