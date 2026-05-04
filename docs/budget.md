# Budget

Each project has a token budget specified at creation time. The budget
governs `/run` termination and is logged append-only to
`<project>/budget.jsonl`.

## Components

- **`token_cap`** — the hard cap. `/run` halts at `cumulative_total >= token_cap`.
- **`iteration_cap`** — the iteration ceiling (default 100).
- **`stagnation_window`** — iterations with no improvement before halting (default 8).
- **`catastrophic_failure_window`** — same FAIL skeptic key in N consecutive iterations halts (default 3).

## Tuning

For a manufacturing-defect project on a 100-column / 100k-row dataset, a
30k-token budget is enough for a baseline pass + a couple of
synthesis cycles. Bump to 100k for the full vertical slice of universal
seeds + 10 generated hypotheses.

## Reading the ledger

`budget.jsonl` rows are `BudgetLedgerEntry` objects with
`cumulative_total`, `cap`, `fraction_consumed`. The `eda status <project>`
CLI reads the latest row.
