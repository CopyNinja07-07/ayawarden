---
tags:
- curatorkit
configs:
- config_name: sft_alpaca
  default: true
  description: curatorkit sft_alpaca export, sha256 4b8f06737ac17fbc2fb447d678449764ed11308d835d30417b788ca735ee416d
  data_files:
  - split: train
    path: sft_alpaca.jsonl
- config_name: sft_sharegpt
  description: curatorkit sft_sharegpt export, sha256 316818a792c2802ecdd1aef9e89e07a9bf23da1f6e7fa5c2af5be56b7b176de7
  data_files:
  - split: train
    path: sft_sharegpt.jsonl
- config_name: sft_messages
  description: curatorkit sft_messages export, sha256 fcb5c9ba27602b56d5b13bc93df9000340e5f5f9a100735a9df937860f687000
  data_files:
  - split: train
    path: sft_messages.jsonl
---

# Dataset Card: ayawarden_train

## Summary

- **Run timestamp**: 2026-11-14T19:30:30.080144Z
- **Wall clock time**: 0.2s
- **Pipeline config hash**: `bd4a635ef81790cf`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 2400 | 2400 | — | 0 | 2400 |
| SchemaGate | 2400 | 2400 | 0 | 0 | 2400 |
| ExactDeduplicator | 2400 | 2400 | — | — | 2400 |
| TextCleaner | 2400 | 2400 | — | — | 2400 |
| MessagesExporter | — | 2400 | — | — | 2400 |
| ShareGPTExporter | — | 2400 | — | — | 2400 |
| AlpacaExporter | — | 2400 | — | — | 2400 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 2400/0 passed (N/A), 0 rejected
- **SchemaGate**: 2400/2400 passed (100.0%), 0 rejected


**Rejection reasons:**
  - None

## Deduplication

  - exact_duplicates_removed: 0

## Known Limitations

- No MinHash deduplication applied.
- Table extraction requires `extract_tables: true` on the PDF reader and MinerU with GPU support. Disabled by default.
- Token counting uses a whitespace tokenizer by default. For exact LLM context window
  enforcement, enable `use_tiktoken: true` in the SchemaGate config.
- Both within-run (MinHash/Diversity) and cross-run (EmbeddingDeduplicator) deduplication
  are available. Cross-run dedup persists an embedding index across runs; use
  `embedding_reset_index: true` to start fresh.

## Reproduction

```bash
pip install curatorkit
curatorkit run pipeline.yaml --output-dir ./out/
```

## Tool Versions

- **curatorkit**: 0.1.28
- **python**: 3.12.15

## Configs

Load one with `datasets.load_dataset("<this folder or Hub repo>", "<config>")`.

| Config | Splits (rows) | Columns |
|--------|---------------|---------|
| `sft_alpaca` | train (2400) | `instruction`: string<br>`input`: string (empty when there is no input)<br>`output`: string |
| `sft_sharegpt` | train (2400) | `conversations`: list of {from, value}; from is system, human or gpt |
| `sft_messages` | train (2400) | `messages`: list of {role, content}; role is system, user or assistant |
