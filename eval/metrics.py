from collections import defaultdict
from typing import Any


def json_validity_rate(predictions: list[dict | None]) -> float:
    valid = sum(1 for p in predictions if p is not None)
    return valid / len(predictions) if predictions else 0.0


def field_level_f1(
    predictions: list[dict | None],
    references: list[dict],
    field_path: str,
) -> dict[str, float]:
    """F1 por campo extraído. field_path usa notación 'a.b'."""
    tp = fp = fn = 0
    for pred, ref in zip(predictions, references):
        p_val = _get_path(pred, field_path) if pred else None
        r_val = _get_path(ref, field_path)
        if p_val is None and r_val is None:
            continue
        if p_val == r_val:
            tp += 1
        else:
            if p_val is not None:
                fp += 1
            if r_val is not None:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def violation_false_negative_rate(
    predictions: list[dict | None],
    references: list[dict],
) -> float:
    """Peor error en propfirm: decir 'cumple' cuando hay violación."""
    total_violations = 0
    missed = 0
    for pred, ref in zip(predictions, references):
        ref_rules = {v["rule_id"] for v in ref.get("violations", [])}
        pred_rules = {v["rule_id"] for v in (pred or {}).get("violations", [])}
        total_violations += len(ref_rules)
        missed += len(ref_rules - pred_rules)
    return missed / total_violations if total_violations else 0.0


def numeric_hallucination_rate(
    predictions: list[dict | None],
    references: list[dict],
    numeric_fields: list[str],
) -> float:
    """Detecta números inventados donde la referencia es null."""
    total = 0
    hallucinated = 0
    f or pred, ref in zip(predictions, references):
        for field in numeric_fields:
            r_val = _get_path(ref, f"extracted_facts.{field}")
            if r_val is None:
                total += 1
                p_val = _get_path(pred, f"extracted_facts.{field}") if pred else None
                if p_val is not None:
                    hallucinated += 1
    return hallucinated / total if total else 0.0


def _get_path(obj: Any, path: str) -> Any:
    if obj is None:
        return None
    for part in path.split("."):
        if not isinstance(obj, dict):
            return None
        obj = obj.get(part)
    return obj


def compute_all(
    predictions: list[dict | None],
    references: list[dict],
) -> dict[str, Any]:
    return {
        "json_validity_rate": json_validity_rate(predictions),
        "field_f1_extracted_facts.max_drawdown_pct": field_level_f1(
            predictions, references, "extracted_facts.max_drawdown_pct"
        ),
        "field_f1_extracted_facts.daily_loss_pct": field_level_f1(
            predictions, references, "extracted_facts.daily_loss_pct"
        ),
        "violation_false_negative_rate": violation_false_negative_rate(
            predictions, references
        ),
        "numeric_hallucination_rate": numeric_hallucination_rate(
            predictions,
            references,
            ["max_drawdown_pct", "daily_loss_pct", "largest_win_ratio"],
        ),
    }
