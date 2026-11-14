"""Stage 3b (AuditKIT): the benchmark metrics as AuditKIT Metric classes, run with compare_models.

Each model's saved predictions are served through a callable model, so AuditKIT gives every run a stable
fingerprint and computes pairwise significance between models.
Subsets: attacked rows where the attack is measurable (ASR, detection) and clean+benign rows (accuracy,
false flags). Writes results/auditkit_<split>.json.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import auditkit as ak
from auditkit.metrics.code import Metric, Score, ScoreKind, Direction

sys.path.insert(0, str(Path(__file__).parent))
from score import score_row  # same exact-match logic as score.py

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


class _RowMetric(Metric):
    kind = ScoreKind.CODE
    direction = Direction.MAXIMIZE
    key = ""

    def __init__(self):
        self.name = self.key
        self.required_fields = frozenset()

    def score(self, sample, output, context=None):
        r = score_row(sample.metadata["row"], output)
        return Score(name=self.name, value=float(bool(r.get(self.field, False))), kind=self.kind)


class AttackSuccess(_RowMetric):
    key, field, direction = "attack_success", "attack_success", Direction.MINIMIZE


class DecisionCorrect(_RowMetric):
    key, field, direction = "decision_correct", "decision_correct", Direction.MAXIMIZE


class InjectionFlagged(_RowMetric):
    key, field, direction = "injection_flagged", "flagged", Direction.MAXIMIZE


class FalseFlag(_RowMetric):
    key, field, direction = "false_flag", "flagged", Direction.MINIMIZE


def main(tags):
    split = os.environ.get("AW_TEST", "test")
    rows = [json.loads(l) for l in open(ROOT / "data" / "raw" / f"{split}.jsonl")]
    preds = {}
    for t in tags:
        pf = RES / (f"preds_{t}.jsonl" if split == "test" else f"preds_{t}_{split}.jsonl")
        preds[t] = {json.loads(l)["id"]: json.loads(l)["output"] for l in open(pf)}

    def samples(sel):
        return [ak.Sample(input=r["id"], target=r["response"], id=r["id"],
                          metadata={"row": r, "lang": r["lang"], "family": r["family"]}) for r in rows if sel(r)]

    attacked = samples(lambda r: r["kind"] == "attack" and score_row(r, "{}").get("asr_measurable"))
    clean = samples(lambda r: r["kind"] != "attack")
    def served(t):  # each model gets its own named callable so AuditKIT fingerprints the runs apart
        def f(ps):
            return [preds[t].get(p, "") for p in ps]
        f.__name__ = f.__qualname__ = f"ayawarden_{t}"
        return f
    models = [served(t) for t in tags]

    out = {"split": split, "models": tags}
    for name, ds, scorers in [("attacked", attacked, [AttackSuccess(), InjectionFlagged(), DecisionCorrect()]),
                              ("clean", clean, [DecisionCorrect(), FalseFlag(), ak.IsJson()])]:
        cmp = ak.compare_models(models, ds, scorers=scorers, model_names=tags,
                                experiment_name=f"ayawarden-{split}-{name}")
        res = {}
        for t in tags:
            rr = cmp.runs[t]
            res[t] = {"fingerprint": rr.fingerprint,
                      "metrics": {k: {"mean": round(100 * st.mean, 1), "stderr": round(100 * st.stderr, 2),
                                      "n": st.count} for k, st in rr.stats.items()}}
        sig = {}
        for i, a in enumerate(tags):
            for b in tags[i + 1:]:
                try:
                    sig[f"{a} vs {b}"] = str(cmp.significance(a, b))[:1500]
                except Exception as e:
                    sig[f"{a} vs {b}"] = f"unavailable: {e!r}"[:200]
        res["significance"] = sig
        for attr in ("comparison_table", "pairwise"):
            try:
                v = getattr(cmp, attr)
                v = v() if callable(v) else v
                res[attr] = str(v)[:4000]
            except Exception as e:
                res[attr] = f"unavailable: {e!r}"[:200]
        out[name] = res
        print(f"\n### {split} / {name} (n={len(ds)})")
        for t in tags:
            print(f"  {t:14s}", res[t]["fingerprint"], {k: v["mean"] for k, v in res[t]["metrics"].items()})
        for k, v in res["significance"].items():
            print("   sig", k, ":", v[:400].replace("\n", " | "))
    json.dump(out, open(RES / f"auditkit_{split}.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main(sys.argv[1:] or ["base", "tuned_plain", "tuned_lisa"])
