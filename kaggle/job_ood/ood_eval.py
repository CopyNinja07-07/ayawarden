# AyaWarden: evaluate base / plain / lisa on the out-of-distribution test (new task families + free-form attacks).
import glob, json, os, subprocess, sys, time
W = "/kaggle/working"; T0 = time.time()
if len(sys.argv) == 1:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ads = {}
    for cfg in glob.glob("/kaggle/input/**/adapter_config.json", recursive=True):
        d = os.path.dirname(cfg)
        ads["tuned_lisa" if "lisa" in d else "tuned_plain"] = d
    print("adapters:", ads, flush=True)
    for tag in ["base"] + sorted(ads):
        subprocess.run([sys.executable, os.path.abspath(__file__), tag, ads.get(tag, "")], check=True)
    sys.exit(0)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tag, adapter = sys.argv[1], sys.argv[2]
model_dir = os.path.dirname(sorted([p for p in glob.glob("/kaggle/input/**/config.json", recursive=True)],
                                   key=lambda p: ("tiny-aya" not in p, len(p)))[0])
rows = [json.loads(l) for l in open(glob.glob("/kaggle/input/**/ood.jsonl", recursive=True)[0])]
tok = AutoTokenizer.from_pretrained(model_dir); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16, device_map={"": 0})
if adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
def prompt_of(r):
    s, u = r["messages"][0]["content"], r["messages"][1]["content"]
    t, x = u.split("\n\n<tool_output>", 1)
    return f"{s}\n\n{t}\n\n<tool_output>{x}"
outs = []
for i in range(0, len(rows), 16):
    chats = [tok.apply_chat_template([{"role": "user", "content": prompt_of(r)}], tokenize=False, add_generation_prompt=True) for r in rows[i:i+16]]
    enc = tok(chats, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=120, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    outs += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
with open(f"{W}/preds_{tag}_ood.jsonl", "w") as f:
    for r, o in zip(rows, outs):
        f.write(json.dumps({"id": r["id"], "output": o}, ensure_ascii=False) + "\n")
print(tag, "done", round(time.time() - T0), flush=True)
