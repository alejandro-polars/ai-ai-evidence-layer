import json
from pathlib import Path

import torch
from jsonschema import ValidationError, validate
from transformers import AutoModelForCausalLM, AutoTokenizer

from metrics import compute_all


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_schema(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_output(raw: str, schema: dict) -> dict | None:
    raw = raw.strip()
    # Limpia fences por si el modelo los añade
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        obj = json.loads(raw)
        validate(instance=obj, schema=schema)
        return obj
    except (json.JSONDecodeError, ValidationError):
        return None


def main(
    model_path: str = "./checkpoints/qwen25-7b-evidence/final",
    test_path: str = "data/test.jsonl",
    schema_path: str = "data/schema.json",
    system_prompt_path: str = "deployment/system_prompt.md",
    output_path: str = "eval/regression_baseline.json",
):
    schema = load_schema(schema_path)
    system_prompt = Path(system_prompt_path).read_text(encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    records = load_jsonl(test_path)
    predictions: list[dict | None] = []
    references: list[dict] = []

    for rec in records:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": rec["input"]},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.0,
            )
        generated = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        predictions.append(parse_output(generated, schema))
        references.append(rec["output"])

    metrics = compute_all(predictions, references)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
