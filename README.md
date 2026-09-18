# Alejandro Intelligent AI - Evidence Layer Core (0.5B)

An ultra-low latency local AI workflow engine designed for high-throughput prop trading data audit and financial compliance tracking.

## Tech Stack & Architecture
- **Core Engine:** Hybrid execution setup (SLM Extraction + Deterministic Validation Rules).
- **Model Backbone:** `Qwen2.5-0.5B-Instruct` distilled via supervised fine-tuning (SFT) with Unsloth and PEFT QLoRA.
- **Inference Runtime:** Local deployment optimization with customized 4bit GGUF quantization over Ollama.

## Repository Structure
- `configs/`: Hyperparameter LoRA matrix allocation.
- `data/`: Automated balanced synthetic trading logs split (Train/Val/Test).
- `engine/`: Deterministic math validation rules.
- `eval/`: Strict evaluation suites for JSON validity and numeric hallucination checks.
- `training/`: Core SFT training scripts.

