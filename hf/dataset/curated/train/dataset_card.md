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
