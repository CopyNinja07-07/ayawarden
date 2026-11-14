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
