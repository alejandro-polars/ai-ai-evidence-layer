# Violation Taxonomy — Evidence Layer

Version: 0.1.0

## Supported rules

### DAILY_LOSS_LIMIT
- **Type**: breach
- **Description**: aggregated daily loss exceeds `daily_loss_limit`.
- **Computed by**: deterministic engine (`engine/rules.py::compute_daily_loss`).
- **LLM extracts**: trades, timestamps, PnL. Does NOT compute the total.

### MAX_DRAWDOWN
- **Type**: breach
- **Description**: drawdown on the equity curve exceeds `max_drawdown_limit`.
- **Computed by**: `compute_max_drawdown`.
- **LLM extracts**: trade sequence. Does NOT compute peak or DD.

### CONSISTENCY_RULE
- **Type**: warning
- **Description**: the most profitable day accounts for >X% of total profit.
- **Computed by**: `compute_consistency`.
- **Only applies when total profit is positive.**

### NEWS_TRADING
- **Type**: breach
- **Description**: trades opened during high-impact news windows.
- **Requires**: external news calendar (pending integration).
- **LLM extracts**: trade timestamps.

### WEEKEND_HOLDING
- **Type**: breach
- **Description**: positions held from Friday to Monday.
- **Computed by**: `detect_weekend_holding`.

### POSITION_SIZE_LIMIT
- **Type**: breach
- **Description**: position size exceeds `position_size_limit_lot`.
- **Computed by**: direct comparison.

### MARTINGALE_PATTERN
- **Type**: warning
- **Description**: progressive lot increase after consecutive losses.
- **Computed by**: `detect_martingale`, window of 4 trades.

## Principles

1. **The LLM extracts, the engine decides.** Never the other way around.
2. **False negatives > false positives for breach severity.** If the model is unsure, it does not report.
3. **Hard numeric values always come from the engine.** The LLM returns `null` when not explicitly present in the input.
