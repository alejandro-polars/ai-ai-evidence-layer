import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import yaml

RULES = [
    "DAILY_LOSS_LIMIT",
    "MAX_DRAWDOWN",
    "CONSISTENCY_RULE",
    "NEWS_TRADING",
    "WEEKEND_HOLDING",
    "POSITION_SIZE_LIMIT",
    "MARTINGALE_PATTERN",
]

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "US500", "NAS100"]


class TraceGenerator:
    def __init__(self, rng: random.Random, config: dict):
        self.rng = rng
        self.cfg = config

    def _random_timestamp(self, base: datetime) -> datetime:
        minutes = self.rng.randint(0, 60 * 24 * 20)
        return base + timedelta(minutes=minutes)

    def _gen_trade(self, ts: datetime, symbol: str, outcome: str) -> dict:
        if outcome == "win":
            pnl = self.rng.uniform(50, 800)
        elif outcome == "loss":
            pnl = self.rng.uniform(-900, -50)
        else:
            pnl = self.rng.uniform(-50, 50)
        return {
            "open_ts": ts,
            "close_ts": ts + timedelta(minutes=self.rng.randint(5, 480)),
            "symbol": symbol,
            "lot": round(self.rng.uniform(0.1, 2.0), 2),
            "pnl": round(pnl, 2),
        }

    def _render_trace(self, trades: list[dict], meta: dict) -> str:
        lines = []
        for t in trades:
            lines.append(
                f"{t['open_ts'].strftime('%Y-%m-%d %H:%M')} OPEN {t['symbol']} {t['lot']} lot"
            )
            lines.append(
                f"{t['close_ts'].strftime('%Y-%m-%d %H:%M')} CLOSE {t['symbol']} {t['pnl']} USD"
            )
        for k, v in meta.items():
            lines.append(f"{k}: {v}")
        return "\n".join(lines)

    def _base_trace(self, n_trades: int) -> list[dict]:
        base = datetime(2024, 1, 2, 8, 0)
        trades = []
        for _ in range(n_trades):
            ts = self._random_timestamp(base)
            sym = self.rng.choice(SYMBOLS)
            outcome = self.rng.choices(["win", "loss", "flat"], weights=[0.45, 0.45, 0.10])[0]
            trades.append(self._gen_trade(ts, sym, outcome))
        trades.sort(key=lambda t: t["open_ts"])
        return trades

    def generate_clean(self) -> dict:
        trades = self._base_trace(self.rng.randint(3, 12))
        meta = {
            "daily_loss_limit": 1000,
            "max_drawdown_limit": 5000,
            "consistency_limit_pct": 40,
            "account_size": 100000,
        }
        return self._package(trades, meta, violations=[], facts_override={})

    def generate_with_daily_loss(self) -> dict:
        trades = self._base_trace(self.rng.randint(3, 8))
        daily_limit = self.rng.choice([300, 500, 800])
        # Fuerza una pérdida diaria que supere el límite
        day = trades[0]["open_ts"].date()
        loss_total = -daily_limit - self.rng.uniform(50, 400)
        trades.append(
            {
                "open_ts": datetime.combine(day, datetime.min.time()) + timedelta(hours=10),
                "close_ts": datetime.combine(day, datetime.min.time()) + timedelta(hours=14),
                "symbol": self.rng.choice(SYMBOLS),
                "lot": round(self.rng.uniform(0.5, 2.0), 2),
                "pnl": round(loss_total, 2),
            }
        )
        trades.sort(key=lambda t: t["open_ts"])
        meta = {
            "daily_loss_limit": daily_limit,
            "max_drawdown_limit": 5000,
            "consistency_limit_pct": 40,
            "account_size": 100000,
        }
        v = {
            "rule_id": "DAILY_LOSS_LIMIT",
            "severity": "breach",
            "evidence_ref": f"{trades[-1]['close_ts'].strftime('%Y-%m-%d %H:%M')} CLOSE {trades[-1]['symbol']} {trades[-1]['pnl']} USD",
            "notes": f"Pérdida diaria de {abs(loss_total):.0f} USD supera límite de {daily_limit} USD",
        }
        return self._package(trades, meta, violations=[v], facts_override={})

    def generate_with_weekend_holding(self) -> dict:
        # Viernes abre, lunes cierra
        friday = datetime(2024, 1, 5, 16, 0)
        monday = datetime(2024, 1, 8, 9, 0)
        trades = [
            {
                "open_ts": friday,
                "close_ts": monday,
                "symbol": self.rng.choice(SYMBOLS),
                "lot": round(self.rng.uniform(0.1, 1.0), 2),
                "pnl": round(self.rng.uniform(-300, 300), 2),
            }
        ]
        meta = {
            "daily_loss_limit": 1000,
            "max_drawdown_limit": 5000,
            "consistency_limit_pct": 40,
            "account_size": 100000,
        }
        v = {
            "rule_id": "WEEKEND_HOLDING",
            "severity": "breach",
            "evidence_ref": f"{friday.strftime('%Y-%m-%d %H:%M')} OPEN -> {monday.strftime('%Y-%m-%d %H:%M')} CLOSE",
            "notes": "Posición mantenida durante el fin de semana",
        }
        return self._package(trades, meta, violations=[v], facts_override={"weekend_positions": 1})

    def generate_with_martingale(self) -> dict:
        base = datetime(2024, 1, 10, 9, 0)
        lots = [0.1, 0.2, 0.4, 0.8]
        trades = []
        t = base
        for lot in lots:
            trades.append(
                {
                    "open_ts": t,
                    "close_ts": t + timedelta(hours=1),
                    "symbol": "EURUSD",
                    "lot": lot,
                    "pnl": round(-100 * lot * 10, 2),
                }
            )
            t += timedelta(hours=2)
        meta = {
            "daily_loss_limit": 1000,
            "max_drawdown_limit": 5000,
            "consistency_limit_pct": 40,
            "account_size": 100000,
        }
        v = {
            "rule_id": "MARTINGALE_PATTERN",
            "severity": "warning",
            "evidence_ref": "lots: 0.1 -> 0.2 -> 0.4 -> 0.8",
            "notes": "Incremento progresivo de tamaño tras pérdidas",
        }
        return self._package(trades, meta, violations=[v], facts_override={})

    def _package(
        self,
        trades: list[dict],
        meta: dict,
        violations: list[dict],
        facts_override: dict,
    ) -> dict:
        facts = {
            "max_drawdown_pct": None,
            "daily_loss_pct": None,
            "largest_win_ratio": None,
            "trades_during_news": 0,
            "weekend_positions": 0,
        }
        facts.update(facts_override)
        output = {
            "trace_id": "auto",
            "violations": violations,
            "extracted_facts": facts,
            "confidence": round(self.rng.uniform(0.85, 0.98), 2),
        }
        return {"input": self._render_trace(trades, meta), "output": output}


def main():
    with open("configs/lora_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["training"]["seed"]
    rng = random.Random(seed)
    gen = TraceGenerator(rng, cfg)

    # Distribución: 50% clean, 50% violaciones (balanceado)
    generators = [
        (gen.generate_clean, 50),
        (gen.generate_with_daily_loss, 15),
        (gen.generate_with_weekend_holding, 10),
        (gen.generate_with_martingale, 10),
    ]
    # Rellena el resto con combinaciones limpias/violaciones variadas
    generators.append((gen.generate_with_daily_loss, 5))
    generators.append((gen.generate_clean, 10))

    n_total = 2000
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    n_test = n_total - n_train - n_val

    pool = []
    for fn, weight in generators:
        for _ in range(weight):
            pool.append(fn())

    # Repite/mezcla hasta llenar
    full = []
    while len(full) < n_total:
        full.append(rng.choice(pool))
    rng.shuffle(full)

    Path("data").mkdir(exist_ok=True)
    for name, chunk in [
        ("train", full[:n_train]),
        ("val", full[n_train : n_train + n_val]),
        ("test", full[n_train + n_val :]),
    ]:
        with open(f"data/{name}.jsonl", "w", encoding="utf-8") as f:
            for rec in chunk:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"data/{name}.jsonl -> {len(chunk)} ejemplos")


if __name__ == "__main__":
    main()
