"""
Script de entrenamiento con Unsloth + TRL (SFTTrainer)
Compatible con Google Colab (T4) y entornos con CUDA.
"""

import os
import sys
import yaml

# --- Fix 1: Desactivar vLLM para evitar el bug de pickle al guardar ---
os.environ["VLLM_USE_V1"] = "0"

# --- Fix 2: Forzar pickle estándar si dill da problemas en Python 3.13 ---
# (solo se aplica si dill está causando el TypeError de ConfigModuleInstance)
try:
    import dill  # noqa
except ImportError:
    pass

import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel


def load_config(path: str = "configs/train_config.yaml") -> dict:
    """Carga la configuración YAML del entrenamiento."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()

    # ------------------------------------------------------------------
    # 1. Cargar modelo base con Unsloth (4-bit por defecto)
    # ------------------------------------------------------------------
    max_seq_length = cfg["model"].get("max_seq_length", 2048)
    dtype = None  # autodetect
    load_in_4bit = cfg["model"].get("load_in_4bit", True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"]["name"],
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    # ------------------------------------------------------------------
    # 2. Aplicar LoRA
    # ------------------------------------------------------------------
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora"].get("r", 16),
        target_modules=cfg["lora"].get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj",
             "gate_proj", "up_proj", "down_proj"],
        ),
        lora_alpha=cfg["lora"].get("alpha", 16),
        lora_dropout=cfg["lora"].get("dropout", 0),
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg["training"].get("seed", 3407),
        use_rslora=False,
        loftq_config=None,
    )

    # ------------------------------------------------------------------
    # 3. Cargar dataset
    # ------------------------------------------------------------------
    dataset = load_dataset(
        cfg["data"]["path"],
        split=cfg["data"].get("split", "train"),
    )

    # ------------------------------------------------------------------
    # 4. Argumentos de entrenamiento
    # ------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=cfg["training"]["output_dir"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        warmup_steps=cfg["training"]["warmup_steps"],
        num_train_epochs=cfg["training"]["num_train_epochs"],
        learning_rate=cfg["training"]["learning_rate"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        optim=cfg["training"]["optim"],
        weight_decay=cfg["training"]["weight_decay"],
        fp16=True,
        bf16=False,
        logging_steps=cfg["training"]["logging_steps"],
        save_steps=cfg["training"]["save_steps"],
        eval_steps=cfg["training"]["eval_steps"],
        eval_strategy="steps",
        save_total_limit=cfg["training"]["save_total_limit"],
        seed=cfg["training"]["seed"],
        report_to=cfg["training"]["report_to"],
    )

    # ------------------------------------------------------------------
    # 5. Entrenador (TRL moderno)
    # ------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=training_args,
    )

    # ------------------------------------------------------------------
    # 6. Entrenar
    # ------------------------------------------------------------------
    trainer.train()

    # ------------------------------------------------------------------
    # 7. Guardar SOLO el adaptador LoRA (evita el bug de pickle)
    # ------------------------------------------------------------------
    out_dir = f"{cfg['training']['output_dir']}/final"
    os.makedirs(out_dir, exist_ok=True)

    try:
        model.save_pretrained(f"{out_dir}_lora")
        tokenizer.save_pretrained(f"{out_dir}_lora")
        print(f"✅ Adaptador LoRA guardado en {out_dir}_lora")
    except Exception as e:
        print(f"⚠️ Falló save_pretrained estándar: {e}")
        print("Intentando guardado alternativo con safetensors...")
        model.save_pretrained(f"{out_dir}_lora", safe_serialization=True)
        tokenizer.save_pretrained(f"{out_dir}_lora")
        print(f"✅ Adaptador LoRA guardado (fallback) en {out_dir}_lora")


if __name__ == "__main__":
    main() 
    print 
