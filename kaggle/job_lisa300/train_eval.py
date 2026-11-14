# AyaWarden — Kaggle job. Each phase runs in its own process so GPU memory is clean between them.
#   python script.py            -> orchestrates: [baseline eval] -> train -> tuned eval
#   python script.py train      -> SafeTune fine-tune (SafeGrad or PlainSFT), saves LoRA adapter
#   python script.py eval PATH TAG [ADAPTER]  -> predictions on the test set + general-safety probes
# Outputs in /kaggle/working: preds_<tag>.jsonl, safety_<tag>.jsonl, adapter_<method>/, train_log_<method>.json
import glob, json, os, subprocess, sys, time

METHOD = "lisa"            # "lisa" (main), "plain" (ablation); "safegrad" needs >16 GB for Tiny Aya
SKIP_BASE = "1"             # "1" = baseline predictions already exist
SAFETY_N = 300             # >0: use SafeTune load_harden_data(n) as the safety set
EVAL_ONLY_ADAPTER = ""     # dir name of an adapter from an earlier run: skip training, only evaluate it
W = "/kaggle/working"
T0 = time.time()


def paths():
    model = os.path.dirname(sorted([p for p in glob.glob("/kaggle/input/**/config.json", recursive=True)
                                    if "adapter" not in p],
                                   key=lambda p: ("tiny-aya" not in p, len(p)))[0])
    data = os.path.dirname(glob.glob("/kaggle/input/**/test.jsonl", recursive=True)[0])
    train = os.path.dirname(glob.glob("/kaggle/input/**/train/sft_messages.jsonl", recursive=True)[0])
    return model, data, train


def prompt_of(row):  # same flattening as the CuratorKIT stage: one user turn
    system, user = row["messages"][0]["content"], row["messages"][1]["content"]
    task, tool = user.split("\n\n<tool_output>", 1)
    return f"{system}\n\n{task}\n\n<tool_output>{tool}"


def generate(model, tok, prompts, max_new=160, bs=16):
    import torch
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


def do_eval(path, tag, adapter=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_dir, data, _ = paths()
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map={"": 0})
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()
    test = [json.loads(l) for l in open(f"{data}/test.jsonl")]
    preds = generate(model, tok, [prompt_of(r) for r in test])
    with open(f"{W}/preds_{tag}.jsonl", "w") as f:
        for r, p in zip(test, preds):
            f.write(json.dumps({"id": r["id"], "output": p}, ensure_ascii=False) + "\n")
    sp = safety_prompts()
    sout = generate(model, tok, [s["prompt"] for s in sp], max_new=80)
    with open(f"{W}/safety_{tag}.jsonl", "w") as f:
        for s, o in zip(sp, sout):
            f.write(json.dumps({**s, "output": o}, ensure_ascii=False) + "\n")


def do_train():
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    import safetune
    from safetune.runner import harden
    import safetune.runner.harden._gradient_surgery as gs

    model_dir, _, train_dir = paths()
    log = {"method": METHOD, "gpu": torch.cuda.get_device_name(0)}
    safetune.configure(device="cuda")
    if METHOD == "safegrad":
        try:
            safetune.configure(max_len=1280)
        except TypeError:
            print("max_len not configurable", flush=True)

    # SafeGrad keeps a frozen reference copy for its KL term; load it in 4-bit so both fit on one 16 GB GPU.
    def _ref_4bit(path, torch_dtype=None, **kw):
        m = AutoModelForCausalLM.from_pretrained(
            path, device_map={"": 0},
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                                   bnb_4bit_quant_type="nf4"))
        m.to = lambda *a, **k: m
        return m
    gs._load_pretrained_lm = _ref_4bit

    # QLoRA: the trained (policy) model is also loaded in 4-bit NF4. Tiny Aya's 262k-token vocabulary makes the
    # logits of policy + reference + safety batch too big for 16 GB otherwise. LoRA weights stay fp32.
    import safetune.runner.utils.model_utils as mu
    def _policy_4bit(name, **kw):
        return AutoModelForCausalLM.from_pretrained(
            name, device_map={"": 0}, torch_dtype=torch.float16,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                                   bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True))
    if METHOD == "safegrad":
        mu.load_model = _policy_4bit
        log["policy_quantization"] = "nf4 (QLoRA)"

    cls = {"safegrad": harden.SafeGradTrainer, "lisa": harden.LisaTrainer, "plain": harden.PlainSFTTrainer}[METHOD]
    # SafeTune 0.x workaround: _fp32_master_weights reads self.model after _lora_base cleared it, which loads a
    # SECOND full model and upcasts all its weights (OOM on 16 GB). Cast the LoRA weights here instead.
    _orig_lora = cls._lora_base
    def _lora_base_fixed(self):
        m = _orig_lora(self)
        for p in m.parameters():
            if p.requires_grad and p.dtype == torch.float16:
                p.data = p.data.float()
        try:
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            m.enable_input_require_grads()
        except Exception as e:
            print("grad ckpt not enabled:", e)
        log["trainable_params"] = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print("trainable params:", log["trainable_params"], flush=True)
        return m
    cls._lora_base = _lora_base_fixed
    cls._fp32_master_weights = lambda self, args: args

    # Keep the small LoRA adapter (what we publish) and skip writing the 6.7 GB merged checkpoint.
    def _save_adapter(self, model, out_dir):
        model.save_pretrained(f"{W}/adapter_{METHOD}300")
        self.tok.save_pretrained(f"{W}/adapter_{METHOD}300")
        return f"{W}/adapter_{METHOD}300"
    cls._save_merged = _save_adapter

    kw = dict(model_id=model_dir, epochs=1, batch_size=2, lr=1e-4, logging_steps=10, fp16=True)
    if METHOD == "safegrad":
        kw.update(batch_size=1, gradient_accumulation_steps=8, surgery_per_optimizer_step=True)
    if METHOD == "lisa":
        kw.update(lisa_rho=0.1, lisa_warmup_steps=10, lisa_alignment_step=20, lisa_finetune_step=20)
    trainer = cls(**kw)
    extra = {}
    if METHOD in ("safegrad", "lisa"):
        # SafeTune's built-in refusal safety set (the alignment signal kept during fine-tuning)
        if SAFETY_N:
            # larger refusal set from SafeTune's BeaverTails loader (default set is only a handful of rows)
            _, extra["safety_dataset"] = harden.load_harden_data(model_dir, n=SAFETY_N)
        else:
            from safetune.runner.utils.data_utils import build_safety_dataset
            extra["safety_dataset"] = build_safety_dataset(trainer.tok)
        log["safety_rows"] = len(extra["safety_dataset"])
    out = trainer.train(train_dir, dataset_config="sft_messages", out_dir=f"{W}/ckpt_{METHOD}", **extra)
    log["adapter"] = out
    log["train_seconds"] = round(time.time() - T0)
    json.dump(log, open(f"{W}/train_log_{METHOD}300.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        do_train()
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        do_eval(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
        sys.exit(0)

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "safetune", "bitsandbytes", "peft"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])  # old torchao breaks peft
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # one GPU: on T4x2 machines the HF Trainer would use DataParallel and OOM
    me = os.path.abspath(__file__)
    model_dir, _, _ = paths()
    print("model:", model_dir, flush=True)
    if METHOD != "plain" and not SKIP_BASE and not EVAL_ONLY_ADAPTER:
        subprocess.run([sys.executable, me, "eval", model_dir, "base"], check=True)
    if EVAL_ONLY_ADAPTER:
        cands = glob.glob("/kaggle/input/**/adapter_config.json", recursive=True)
        if not cands:  # dataset folders may arrive as archives
            import tarfile, zipfile
            for a in glob.glob("/kaggle/input/**/*.tar", recursive=True):
                tarfile.open(a).extractall(f"{W}/extracted")
            for a in glob.glob("/kaggle/input/**/*.zip", recursive=True):
                zipfile.ZipFile(a).extractall(f"{W}/extracted")
            cands = glob.glob(f"{W}/extracted/**/adapter_config.json", recursive=True)
        print("input files:", glob.glob("/kaggle/input/**", recursive=True)[:40], flush=True)
        adapter = os.path.dirname(cands[0])
    else:
        subprocess.run([sys.executable, me, "train"], check=True)
        adapter = f"{W}/adapter_{METHOD}300"
    subprocess.run([sys.executable, me, "eval", model_dir, f"tuned_{METHOD}300", adapter], check=True)
    print("DONE", round(time.time() - T0))
