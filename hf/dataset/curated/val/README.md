---
tags:
- curatorkit
configs:
- config_name: sft_alpaca
  default: true
  description: curatorkit sft_alpaca export, sha256 d782737265db6e48059422349ec67d0146b76a04eebd6da520ce15c59e42a437
  data_files:
  - split: train
    path: sft_alpaca.jsonl
- config_name: sft_sharegpt
  description: curatorkit sft_sharegpt export, sha256 3c5348d5f725af665235b1bb4d7926c2d29d766ea62248f7c6f73bcadc4bdb16
  data_files:
  - split: train
    path: sft_sharegpt.jsonl
- config_name: sft_messages
  description: curatorkit sft_messages export, sha256 8703cbbef4b91897516faf17698aaea639ffac641ffc22a75eb3261917849843
  data_files:
  - split: train
    path: sft_messages.jsonl
---

# Dataset Card: ayawarden_val

## Summary

- **Run timestamp**: 2026-11-14T19:30:30.128975Z
- **Wall clock time**: 0.0s
- **Pipeline config hash**: `e1c1af9e5d857054`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 240 | 240 | — | 0 | 240 |
| SchemaGate | 240 | 240 | 0 | 0 | 240 |
| ExactDeduplicator | 240 | 240 | — | — | 240 |
| TextCleaner | 240 | 240 | — | — | 240 |
| MessagesExporter | — | 240 | — | — | 240 |
| ShareGPTExporter | — | 240 | — | — | 240 |
| AlpacaExporter | — | 240 | — | — | 240 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 240/0 passed (N/A), 0 rejected
- **SchemaGate**: 240/240 passed (100.0%), 0 rejected


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
| `sft_alpaca` | train (240) | `instruction`: string<br>`input`: string (empty when there is no input)<br>`output`: string |
| `sft_sharegpt` | train (240) | `conversations`: list of {from, value}; from is system, human or gpt |
| `sft_messages` | train (240) | `messages`: list of {role, content}; role is system, user or assistant |
