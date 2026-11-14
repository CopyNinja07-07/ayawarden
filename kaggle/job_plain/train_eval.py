# AyaWarden — Kaggle job: baseline eval -> SafeTune SafeGrad fine-tune -> tuned eval.
# Inputs : Tiny Aya Global (Kaggle Models) + the AyaWarden data (CuratorKIT export + test set).
# Outputs: /kaggle/working/{preds_base,preds_tuned}.jsonl, safety_{base,tuned}.jsonl, adapter/, train_log.json
import glob, json, os, subprocess, sys, time

METHOD = os.environ.get("AW_METHOD", "plain")  # "safegrad" (main) or "plain" (ablation)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "safetune", "bitsandbytes", "peft"], check=True)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

T0 = time.time()
W = "/kaggle/working"
MODEL = os.path.dirname(sorted(glob.glob("/kaggle/input/**/config.json", recursive=True),
                               key=lambda p: ("tiny-aya" not in p, len(p)))[0])
DATA = os.path.dirname(glob.glob("/kaggle/input/**/test.jsonl", recursive=True)[0])
TRAIN_DIR = os.path.dirname(glob.glob("/kaggle/input/**/train/sft_messages.jsonl", recursive=True)[0])
print("model:", MODEL, "\ndata:", DATA, "\ntrain:", TRAIN_DIR, flush=True)
log = {"method": METHOD, "model_path": MODEL, "gpu": torch.cuda.get_device_name(0), "n_gpu": torch.cuda.device_count()}


def prompt_of(row):  # same flattening as CuratorKIT stage: one user turn
    system, user = row["messages"][0]["content"], row["messages"][1]["content"]
    task, tool = user.split("\n\n<tool_output>", 1)
    return f"{system}\n\n{task}\n\n<tool_output>{tool}"


def generate(model, tok, prompts, max_new=160, bs=16):
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    outs = []
    for i in range(0, len(prompts), bs):
        chats = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                         add_generation_prompt=True) for p in prompts[i:i + bs]]
        enc = tok(chats, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"  gen {i + len(chats)}/{len(prompts)}  {time.time() - T0:.0f}s", flush=True)
    return outs


def safety_prompts():
    """Held-out general-safety probes: harmful prompts (should refuse) + safe-but-edgy prompts (should answer)."""
    from datasets import load_dataset
    out = []
    try:
        bt = load_dataset("PKU-Alignment/BeaverTails", split="30k_test")
        bt = bt.filter(lambda r: not r["is_safe"]).shuffle(seed=7).select(range(100))
        out += [{"id": f"harm-{i}", "type": "harmful", "prompt": r["prompt"]} for i, r in enumerate(bt)]
    except Exception as e:
        print("BeaverTails unavailable:", e)
    for name in ("natolambert/xstest-v2-copy", "walledai/XSTest", "Paul/XSTest"):
        try:
            xs = load_dataset(name, split="train" if "natolambert" not in name else "prompts")
            col = "prompt" if "prompt" in xs.column_names else xs.column_names[0]
            typ = "type" if "type" in xs.column_names else None
            rows = [r for r in xs if (typ is None or not str(r[typ]).startswith("contrast"))]
            out += [{"id": f"xs-{i}", "type": "safe_edgy", "prompt": r[col]} for i, r in enumerate(rows[:100])]
            break
        except Exception as e:
            print(name, "unavailable:", e)
    return out


def run_eval(model, tok, tag):
    test = [json.loads(l) for l in open(f"{DATA}/test.jsonl")]
    preds = generate(model, tok, [prompt_of(r) for r in test])
    with open(f"{W}/preds_{tag}.jsonl", "w") as f:
        for r, p in zip(test, preds):
            f.write(json.dumps({"id": r["id"], "output": p}, ensure_ascii=False) + "\n")
    sp = safety_prompts()
    sout = generate(model, tok, [s["prompt"] for s in sp], max_new=80)
    with open(f"{W}/safety_{tag}.jsonl", "w") as f:
        for s, o in zip(sp, sout):
            f.write(json.dumps({**s, "output": o}, ensure_ascii=False) + "\n")
    log[f"eval_{tag}_s"] = round(time.time() - T0)


# ---------------- 1) baseline
tok = AutoTokenizer.from_pretrained(MODEL)
if METHOD == "safegrad":
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map={"": 0})
    run_eval(base, tok, "base")
    del base
    torch.cuda.empty_cache()

# ---------------- 2) fine-tune through SafeTune
import safetune
from safetune.runner import harden
import safetune.runner.harden._gradient_surgery as gs
import safetune.runner.harden._base as hb

safetune.configure(device="cuda")

# SafeGrad keeps a frozen reference copy for its KL term; load it in 4-bit so both fit on one 16 GB GPU.
def _ref_4bit(path, torch_dtype=None, **kw):
    m = AutoModelForCausalLM.from_pretrained(
        path, device_map={"": 0},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                               bnb_4bit_quant_type="nf4"))
    m.to = lambda *a, **k: m
    return m
gs._load_pretrained_lm = _ref_4bit

# Also keep the LoRA adapter (small, publishable) before SafeTune merges it.
_orig_save = hb.HardenTrainerBase._save_merged if hasattr(hb, "HardenTrainerBase") else None
def _save_with_adapter(self, model, out_dir):
    try:
        model.save_pretrained(f"{W}/adapter")
        tok.save_pretrained(f"{W}/adapter")
        print("adapter saved", flush=True)
    except Exception as e:
        print("adapter save failed:", e)
    return _orig_save_any(self, model, out_dir)

cls = harden.SafeGradTrainer if METHOD == "safegrad" else harden.PlainSFTTrainer
_orig_save_any = cls._save_merged
cls._save_merged = _save_with_adapter

kw = dict(model_id=MODEL, epochs=1, batch_size=2, lr=1e-4, logging_steps=10, fp16=True)
if METHOD == "safegrad":
    kw["gradient_accumulation_steps"] = 8
trainer = cls(**kw)
ckpt = trainer.train(TRAIN_DIR, dataset_config="sft_messages", out_dir=f"{W}/ckpt_{METHOD}")
log["train_end_s"] = round(time.time() - T0)
log["checkpoint"] = ckpt
try:
    log["train_history"] = getattr(getattr(trainer, "_last_trainer", None), "state", None).log_history
except Exception:
    pass

# ---------------- 3) tuned eval
del trainer
torch.cuda.empty_cache()
tuned = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.float16, device_map={"": 0})
run_eval(tuned, tok, f"tuned_{METHOD}")

# merged checkpoint is large; keep only adapter + small files in the output
import shutil
for f in glob.glob(f"{ckpt}/*.safetensors"):
    os.remove(f)
json.dump(log, open(f"{W}/train_log_{METHOD}.json", "w"), indent=1, default=str)
print("DONE", time.time() - T0)
