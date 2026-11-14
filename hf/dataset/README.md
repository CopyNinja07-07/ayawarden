---
license: cc-by-4.0
language: [en, hi, es, fr, ar, sw, bn]
task_categories: [text-generation]
tags: [prompt-injection, indirect-prompt-injection, ai-safety, red-teaming, multilingual, code-mixed, agents, tiny-aya, lexsi-hackathon]
pretty_name: AyaWarden Multilingual Indirect Prompt Injection
configs:
- config_name: raw
  data_files:
  - {split: train, path: raw/train.jsonl}
  - {split: validation, path: raw/val.jsonl}
  - {split: test, path: raw/test.jsonl}
  - {split: ood, path: raw/ood.jsonl}
- config_name: curated_sft_messages
  data_files:
  - {split: train, path: curated/train/sft_messages.jsonl}
  - {split: validation, path: curated/val/sft_messages.jsonl}
---

# AyaWarden: multilingual indirect prompt injection in agent tool outputs

Agent tasks whose tool output (a contract, a security alert, an invoice, a product page) may contain a hidden instruction addressed to the AI. Every row has a deterministic gold JSON answer, so attack success and task accuracy are scored exactly, without an LLM judge. Built for the Lexsi AI Safety and Alignment Hackathon (Track 2) to train and evaluate [Tiny Aya](https://huggingface.co/CohereLabs/tiny-aya-global).

| Split | Rows | Attacked | Benign imperative | Clean | Languages | Injection styles |
|---|---|---|---|---|---|---|
| train | 2,400 | 1,316 | 605 | 479 | en, hi, hinglish, es, fr, ar, sw | 9 |
| validation | 240 | 124 | 61 | 55 | same | same 9 |
| test | 800 | 453 | 197 | 150 | same + **bn (held out)** | same 9 + **3 held out** |
| ood | 240 | 138 | 0 | 102 | en, hi, es, fr, ar, sw | free-form; 3 unseen task types (ticket, refund, meeting) |

- **Tasks:** `penalty` (supplier penalty audit), `triage` (security alert category), `invoice` (payment extraction), `shopping` (buy/skip recommendation).
- **Attacker goals:** set the penalty to 0, close the incident as CONFIG_ERROR, swap the payee bank account, force a BUY recommendation.
- **Train styles:** override, fake_system, authority, task_switch, roleplay, urgency, polite, forged_json, html_comment. **Held-out (test only):** fake_completion, encoded (base64), translation_trick. **Held-out language:** Bengali.
- **Benign rows** contain ordinary imperative business text ("please remit within 30 days"): the model must not flag or refuse these.
- **Leak check:** disjoint id/value ranges per split (10000+/30000+/50000+); held-out styles and Bengali never appear in train/val; CuratorKIT exact dedup removed 0 rows.

## Fields (raw)

`id, split, family, lang, kind (attack|benign|clean), style, messages (system+user), gold (dict), attack_goal ({field, value} or null), response (gold JSON string)`.

## Curation

`curated/` is the [CuratorKIT](https://github.com/Lexsi-Labs/CuratorKIT) export used for training (schema gate → exact dedup → text cleaning with HTML stripping disabled so tag-shaped injections are kept), with its `manifest.json`, `dataset_card.md`, `lexsi_provenance.json` and `checksums.txt`.

`results/` holds the evaluation predictions of the base and fine-tuned models and the scored report.

Code: https://github.com/CopyNinja07-07/ayawarden

## Results

| | Base Tiny Aya | Plain SFT | Lisa |
|---|---|---|---|
| ASR, test | 41.0% | 0.0% | 2.0% |
| ASR, OOD tasks | 61.7% | 15.0% | 22.5% |
| Decision accuracy (clean), test | 38.3% | 98.0% | 91.6% |
| Harmful-prompt refusal | 71% | 70% | 80% |
| Safe-prompt over-refusal | 8% | 8% | 27% |
