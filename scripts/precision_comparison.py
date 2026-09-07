"""Alternate-provider comparison for the two open model families."""
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "outputs"
# Historical directory names use ``unquantized``. The saved run metadata do not
# independently record numerical precision, so this script makes only a
# cross-provider comparison.
ALT = OUT / "unquantized_qwen_llama__20260717_120926"
ORIGINAL = OUT / "paper_llm_4way_isotonic_retry__20260522_000630"
print(f"alternate-provider run: {ALT}")

cf = pd.read_csv(ALT / "paper_bundle/tables/table_2_classification.csv")
cq = pd.read_csv(ORIGINAL / "paper_bundle/tables/table_2_classification.csv")

def get(classification, name):
    row = classification[classification.model == name].drop_duplicates("model")
    if len(row) != 1:
        raise RuntimeError(f"expected one classification row for {name}, found {len(row)}")
    return row.accuracy.iloc[0], row.macro_f1.iloc[0], row.parse_error_rate.iloc[0]

pairs=[("Qwen zero","qwen_full_zero_shot","qwen_zero_shot"),
       ("Qwen few ","qwen_full_few_shot_k3","qwen_few_shot_k3"),
       ("Llama zero","llama_full_zero_shot","llama_zero_shot"),
       ("Llama few ","llama_full_few_shot_k3","llama_few_shot_k3")]
print(f"\n{'cell':11s} | {'acc_alt':>8s} {'acc_orig':>9s} {'Δacc':>6s} | {'pErr_alt':>9s} {'pErr_orig':>10s} | {'>0.477?':>7s}")
for lbl,fn,qn in pairs:
    af,f1f,pef=get(cf,fn)
    aq,f1q,peq=get(cq,qn)
    flag="YES" if af>0.477 else ""
    print(f"{lbl:11s} | {af:8.3f} {aq:9.3f} {af-aq:+6.3f} | {pef:9.3f} {peq:10.3f} | {flag:>7s}")

# Best alternate-provider open-LLM accuracy and current headline (Haiku few 0.477).
best_alt=max(get(cf,fn)[0] for _,fn,_ in pairs)
print(f"\nbest alternate-provider open-LLM acc = {best_alt:.3f}  (current headline max across ALL LLMs = 0.477 Haiku few-shot)")
print("=> If best_alt > 0.477, the 'no prompted LLM exceeds 0.477' headline needs updating.")
