

import json
import yaml
from pathlib import Path

import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_messages(record: dict, system_prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": record["input"]},
        {"role": "assistant", "content": json.dumps(record["output"], ensure_ascii=False)},
    ]


def format_chat_template(tokenizer, record: dict, system_prompt: str) -> dict:
    messages = build_messages(record, system_prompt)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def main():
    cfg = load_config("configs/lora_config.yaml")

    system_prompt = Path(cfg["dataset"]["system_prompt_path"]).read_text(encoding="utf-8")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"]["base_model"],
        max_seq_length=cfg["model"]["max_seq_length"],
        dtype=cfg["model"]["dtype"],
        load_in_4bit=cfg["model"]["load_in_4bit"],
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias=cfg["lora"]["bias"],
        use_gradient_checkpointing=cfg["lora"]["use_gradient_checkpointing"],
        random_state=cfg["training"]["seed"],
    )

    if tokenizer.chat_template is None:
        raise RuntimeError("Tokenizer sin chat_template. Verifica el modelo base.")

    train_records = load_jsonl(cfg["dataset"]["train_path"])
    val_records = load_jsonl(cfg["dataset"]["val_path"])

    train_ds = Dataset.from_list(
        [format_chat_template(tokenizer, r, system_prompt) for r in train_records]
    )
    val_ds = Dataset.from_list(
        [format_chat_template(tokenizer, r, system_prompt) for r in val_records]
    )

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
        fp16=cfg["training"]["fp16"] and not is_bfloat16_supported(),
        bf16=cfg["training"]["bf16"] and is_bfloat16_supported(),
        logging_steps=cfg["training"]["logging_steps"],
        save_steps=cfg["training"]["save_steps"],
        eval_steps=cfg["training"]["eval_steps"],
        eval_strategy="steps",
        save_total_limit=cfg["training"]["save_total_limit"],
        seed=cfg["training"]["seed"],
        report_to=cfg["training"]["report_to"],
    )

    # Configuración limpia y minimalista compatible con TRL moderno
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=training_args,
    )

    trainer.train()

    out_dir = f"{cfg['training']['output_dir']}/final"
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Modelo guardado en {out_dir}")


if __name__ == "__main__":
    main()
