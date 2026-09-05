"""Full-precision vs quantized comparison for the two open models (same 1000 subsample)."""
import pandas as pd, glob, os
OUT="outputs"
# newest unquantized full run
FULL=sorted(glob.glob(f"{OUT}/unquantized_qwen_llama__*"))[-1]
QUANT=f"{OUT}/paper_llm_4way_isotonic_retry__20260522_000630"   # the quantized subsample run
print(f"full-precision run: {FULL}")

cf=pd.read_csv(f"{FULL}/paper_bundle/tables/table_2_classification.csv")
kf=pd.read_csv(f"{FULL}/paper_bundle/tables/table_3_calibration.csv")
cq=pd.read_csv(f"{QUANT}/paper_bundle/tables/table_2_classification.csv")
kq=pd.read_csv(f"{QUANT}/paper_bundle/tables/table_3_calibration.csv")

def get(c2,c3,name):
    a=c2[c2.model==name].drop_duplicates("model")
    acc=a.accuracy.iloc[0]; f1=a.macro_f1.iloc[0]; pe=a.parse_error_rate.iloc[0]
    e=c3[(c3.model==name)&(c3.confidence_method=="model_reported")]
    ece=e.ece.iloc[0] if len(e) else float('nan')
    mc=e.mean_confidence.iloc[0] if len(e) else float('nan')
    return acc,f1,ece,pe,mc

pairs=[("Qwen zero","qwen_full_zero_shot","qwen_zero_shot"),
       ("Qwen few ","qwen_full_few_shot_k3","qwen_few_shot_k3"),
       ("Llama zero","llama_full_zero_shot","llama_zero_shot"),
       ("Llama few ","llama_full_few_shot_k3","llama_few_shot_k3")]
print(f"\n{'cell':11s} | {'acc_full':>8s} {'acc_quant':>9s} {'Δacc':>6s} | {'ece_full':>8s} {'ece_quant':>9s} | {'pErr_full':>9s} | {'>0.477?':>7s}")
for lbl,fn,qn in pairs:
    af,f1f,ef,pef,mcf=get(cf,kf,fn)
    aq,f1q,eq,peq,mcq=get(cq,kq,qn)
    flag="YES" if af>0.477 else ""
    print(f"{lbl:11s} | {af:8.3f} {aq:9.3f} {af-aq:+6.3f} | {ef:8.3f} {eq:9.3f} | {pef:9.3f} | {flag:>7s}")

# best full-precision open-LLM accuracy, and current headline (Haiku few 0.477)
best_full=max(get(cf,kf,fn)[0] for _,fn,_ in pairs)
print(f"\nbest full-precision open-LLM acc = {best_full:.3f}  (current headline max across ALL LLMs = 0.477 Haiku few-shot)")
print("=> If best_full > 0.477, the 'no prompted LLM exceeds 0.477' headline needs updating.")
