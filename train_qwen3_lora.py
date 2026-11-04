"""QLoRA-style LoRA fine-tune of Qwen3-Coder-30B-A3B (AWQ-4bit base) on
local-agent trajectories. Attention-module adapters only: the goal is
tool-call format reliability + agentic habits (discover before write, act
on errors, never defer), not new knowledge.
"""
import os
import sys

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL_DIR = sys.argv[1]
DATA = sys.argv[2] if len(sys.argv) > 2 else \
    "/home/asdf/Work/ailm-stack-scripts/lora_dataset.jsonl"
OUT = "/home/asdf/Work/ailm-stack-scripts/q3c_lora_out"

tok = AutoTokenizer.from_pretrained(MODEL_DIR)

from transformers import BitsAndBytesConfig

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True,
)
device_map = {
    "model.embed_tokens": "cpu",
    "model.layers": 0,
    "model.norm": "cpu",
    "model.rotary_emb": 0,
    "lm_head": "cpu",
}
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    quantization_config=bnb,
    dtype=torch.bfloat16,
    device_map=device_map,
)
model.config.use_cache = False

peft_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model = get_peft_model(model, peft_cfg)
model.print_trainable_parameters()

ds = load_dataset("json", data_files=DATA, split="train")
print("rows:", len(ds))

cfg = SFTConfig(
    output_dir=OUT,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=1,
    save_strategy="no",
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_length=3072,
    completion_only_loss=True,
    optim="adamw_torch_fused",
    report_to=[],
    seed=42,
)

trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                     processing_class=tok)
trainer.train()
model.save_pretrained(OUT)
tok.save_pretrained(OUT)
print("saved adapter ->", OUT)
