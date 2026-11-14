---
base_model: CohereLabs/tiny-aya-global
library_name: peft
license: cc-by-nc-4.0
language: [en, hi, es, fr, ar, sw, bn]
tags: [lora, prompt-injection, ai-safety, multilingual, agents, safetune, lexsi-hackathon]
datasets: [SuryanshHere/ayawarden-multilingual-injection]
---

# AyaWarden, plain-SFT ablation (LoRA on Tiny Aya Global)

The **ablation** for [AyaWarden](https://github.com/CopyNinja07-07/ayawarden): Tiny Aya Global fine-tuned with SafeTune `PlainSFTTrainer`
(LoRA r=16 on all attention + MLP projections, 1 epoch, lr 1e-4) on the CuratorKIT export of the
[AyaWarden dataset](https://huggingface.co/datasets/SuryanshHere/ayawarden-multilingual-injection). The main model is the SafeTune
`LisaTrainer` adapter in the same collection.

Task: do the user's task from tool outputs and ignore instructions hidden inside them (8 languages), replying in JSON with a
`suspicious_instructions` flag.

| Test (800 rows) | Base Tiny Aya | This adapter |
|---|---|---|
| Attack success rate | 41.0% | 0.0% |
| ASR, held-out attack styles | 35.5% | 0.0% |
| ASR, held-out language (Bengali) | 34.3% | 0.0% |
| Decision accuracy, attacked | 39.1% | 97.1% |
| Decision accuracy, clean | 38.3% | 98.0% |
| Refusal on held-out harmful prompts | 71% | 70% |
| Refusal on safe-but-edgy prompts | 8% | 8% |

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("CohereLabs/tiny-aya-global", torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(base, "SuryanshHere/ayawarden-tiny-aya-plain-sft-lora")
tok = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global")
```

Built during the Lexsi AI Safety and Alignment Hackathon (team RBoysXAi). Non-commercial license inherited from Tiny Aya.
