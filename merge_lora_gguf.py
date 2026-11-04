"""Merge the QLoRA adapter into Qwen3-Coder-30B and export a q4_k_m GGUF."""
import os
import sys

sys.path = [p for p in sys.path if os.path.abspath(p or ".") != "/home/asdf/Work"]

from unsloth import FastLanguageModel

ADAPTER = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/asdf/Work/ailm-stack-scripts/q3c_lora_out"
OUT = sys.argv[2] if len(sys.argv) > 2 else \
    "/home/asdf/Work/ailm-stack-scripts/q3c_merged_gguf"

try:
    model, tok = FastLanguageModel.from_pretrained(
        model_name=ADAPTER,
        max_seq_length=1024,
        load_in_4bit=True,
        offload_layers="auto",
        device_map_planner_kwargs={"reserve_bytes": 1024 * 1024 * 1024},
    )
except Exception:
    # VL bases (qwen3.6 is Qwen3_5MoeForConditionalGeneration) only load
    # through the text-only extraction path.
    model, tok = FastLanguageModel.from_pretrained(
        model_name=ADAPTER,
        max_seq_length=1024,
        load_in_4bit=True,
        offload_layers="auto",
        text_only=True,
        device_map_planner_kwargs={"reserve_bytes": 1024 * 1024 * 1024},
    )
model.save_pretrained_gguf(OUT, tok, quantization_method="q4_k_m")
print("merged gguf ->", OUT)
