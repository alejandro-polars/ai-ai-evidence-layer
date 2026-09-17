from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Trade:
    open_ts: datetime
    close_ts: datetime
    symbol: str
    lot: float
    pnl: float


@dataclass
class AccountConfig:
    account_size: float = 100_000
    daily_loss_limit: float = 1000
    max_drawdown_limit: float = 5000
    consistency_limit_pct: float = 40
    position_size_limit_lot: float = 5.0


def compute_daily_loss(trades: list[Trade], cfg: AccountConfig) -> dict[str, float]:
    """Devuelve la peor pérdida diaria agregada (negativa si pérdida)."""
    by_day: dict = {}
    for t in trades:
        day = t.close_ts.date()
        by_day[day] = by_day.get(day, 0.0) + t.pnl
    if not by_day:
        return {"worst_day_pnl": 0.0, "worst_day": None}
    worst_day = min(by_day, key=by_day.get)
    return {"worst_day_pnl": by_day[worst_day], "worst_day": worst_day}


def compute_max_drawdown(trades: list[Trade], cfg: AccountConfig) -> dict[str, float]:
    """Drawdown sobre equity curve en USD y %."""
    equity = cfg.account_size
    peak = equity
    max_dd = 0.0
    max_dd_pct = 0.0
    for t in sorted(trades, key=lambda x: x.close_ts):
        equity += t.pnl
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = dd / peak * 100
    return {"max_drawdown_usd": max_dd, "max_drawdown_pct": max_dd_pct}


def compute_consistency(trades: list[Trade], cfg: AccountConfig) -> dict[str, float]:
    """Día más rentable como % del profit total."""
    by_day: dict = {}
    for t in trades:
        day = t.close_ts.date()
        by_day[day] = by_day.get(day, 0.0) + t.pnl
    total_profit = sum(v for v in by_day.values() if v > 0)
    if total_profit <= 0:
        return {"largest_day_ratio_pct": 0.0}
    largest = max(by_day.values()) if by_day else 0.0
    return {"largest_day_ratio_pct": largest / total_profit * 100}


def detect_martingale(trades: list[Trade], cfg: AccountConfig, window: int = 4) -> bool:
    """Detecta incremento de lot tras pérdidas consecutivas."""
    sorted_trades = sorted(trades, key=lambda x: x.open_ts)
    for i in range(len(sorted_trades) - window + 1):
        seq = sorted_trades[i : i + window]
        lots = [t.lot for t in seq]
        losses = [t.pnl < 0 for t in seq[:-1]]
        increasing = all(lots[j] < lots[j + 1] for j in range(len(lots) - 1))
        if increasing and all(losses):
            return True
    return False


def detect_weekend_holding(trades: list[Trade]) -> int:
    """Cuenta posiciones abiertas cruzando fin de semana (vie 22:00 -> lun 00:00)."""
    count = 0
    for t in trades:
        if t.open_ts.weekday() == 4 and t.close_ts.weekday() in (0, 5, 6):
            count += 1
    return count


def evaluate(trades: list[Trade], cfg: AccountConfig) -> dict:
    """Evalúa todas las reglas deterministas. El LLM NO hace esto."""
    daily = compute_daily_loss(trades, cfg)
    dd = compute_max_drawdown(trades, cfg)
    consistency = compute_consistency(trades, cfg)
    martingale = detect_martingale(trades, cfg)
    weekend = detect_weekend_holding(trades)

    violations = []

    if daily["worst_day_pnl"] < -cfg.daily_loss_limit:
        violations.append(
            {
                "rule_id": "DAILY_LOSS_LIMIT",
                "severity": "breach",
                "evidence_ref": str(daily["worst_day"]),
                "notes": f"Pérdida diaria {daily['worst_day_pnl']:.2f} USD",
            }
        )
    if dd["max_drawdown_usd"] > cfg.max_drawdown_limit:
        violations.append(
            {
                "rule_id": "MAX_DRAWDOWN",
                "severity": "breach",
                "evidence_ref": "equity_curve",
                "notes": f"Drawdown {dd['max_drawdown_usd']:.2f} USD ({dd['max_drawdown_pct']:.2f}%)",
            }
        )
    if consistency["largest_day_ratio_pct"] > cfg.consistency_limit_pct:
        violations.append(
            {
                "rule_id": "CONSISTENCY_RULE",
                "severity": "warning",
                "evidence_ref": "daily_pnl_distribution",
                "notes": f"Día más rentable = {consistency['largest_day_ratio_pct']:.1f}% del profit",
            }
        )
    if martingale:
        violations.append(
            {
                "rule_id": "MARTINGALE_PATTERN",
                "severity": "warning",
                "evidence_ref": "lot_sequence",
                "notes": "Incremento de lot tras pérdidas consecutivas",
            }
        )
    if weekend > 0:
        violations.append(
            {
                "rule_id": "WEEKEND_HOLDING",
                "severity": "breach",
                "evidence_ref": "position_lifecycle",
                "notes": f"{weekend} posición(es) cruzando fin de semana",
            }
        )

    return {
        "violations": violations,
        "extracted_facts": {
            "max_drawdown_pct": round(dd["max_drawdown_pct"], 4),
            "daily_loss_pct": round(abs(daily["worst_day_pnl"]) / cfg.account_size * 100, 4)
            if daily["worst_day_pnl"] < 0
            else 0.0,
            "largest_win_ratio": round(consistency["largest_day_ratio_pct"] / 100, 4),
            "trades_during_news": 0,
            "weekend_positions": weekend,
        },
    }
