"""Export the fine-tuned model to GGUF for Ollama/llama.cpp.

Recommended usage on Colab (GPU):
    python scripts/export_gguf.py --quant q4_k_m
"""

import argparse
from pathlib import Path

from unsloth import FastLanguageModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="./checkpoints/qwen25-7b-evidence/final",
    )
    parser.add_argument("--out-dir", default="./export")
    parser.add_argument(
        "--quant",
        default="q4_k_m",
        choices=["q4_k_m", "q5_k_m", "q8_0", "f16"],
    )
    args = parser.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=False,
    )

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Unsloth writes GGUF (requires llama.cpp cloned in the environment)
    model.save_pretrained_gguf(
        args.out_dir,
        tokenizer,
        quantization_method=args.quant,
    )
    print(f"GGUF exported to {args.out_dir}")


if __name__ == "__main__":
    main()
