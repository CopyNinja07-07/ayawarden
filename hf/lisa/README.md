---
base_model: CohereLabs/tiny-aya-global
library_name: peft
license: cc-by-nc-4.0
language: [en, hi, es, fr, ar, sw, bn]
tags: [lora, prompt-injection, ai-safety, safety-preserving-fine-tuning, lisa, multilingual, agents, safetune, lexsi-hackathon]
datasets: [SuryanshHere/ayawarden-multilingual-injection]
---

# AyaWarden, SafeTune Lisa adapter (LoRA on Tiny Aya Global)

Tiny Aya Global fine-tuned with SafeTune `LisaTrainer` (Lazy Safety Alignment, NeurIPS 2024): task batches from the
[AyaWarden dataset](https://huggingface.co/datasets/SuryanshHere/ayawarden-multilingual-injection) (CuratorKIT export) alternate with
SafeTune's built-in refusal safety set (20 steps each, proximal penalty rho=0.1). LoRA r=16 on all attention + MLP projections,
1 epoch, lr 1e-4, fp16 on one T4. Code: https://github.com/CopyNinja07-07/ayawarden

| Test (800 rows) | Base | Plain SFT | **Lisa (this)** |
|---|---|---|---|
| Attack success rate | 41.0% | 0.0% | 2.0% |
| Decision accuracy, attacked | 39.1% | 97.1% | 91.6% |
| Decision accuracy, clean | 38.3% | 98.0% | 91.6% |
| Refusal, held-out harmful prompts (higher = safer) | 71% | 70% | **80%** |
| Refusal, safe-but-edgy prompts (lower = better) | 8% | 8% | 27% |

Lisa raised refusal of harmful prompts by 9 points but over-refused safe prompts and lost some task accuracy. SafeTune's default
safety set has only 6 rows, replayed every alignment phase; see the repo for the follow-up with a larger safety set.

Built during the Lexsi AI Safety and Alignment Hackathon (team RBoysXAi). Non-commercial license inherited from Tiny Aya.
