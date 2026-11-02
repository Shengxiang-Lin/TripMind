from __future__ import annotations
import argparse,json,math,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tripmind.config import get_config,resolve_project_path
from tripmind.preference_model import PreferenceScoringModel,resolve_device
from tripmind.preference_prompt import build_record_prompt
def rows(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():yield json.loads(line)
def parse_args():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",default=None);p.add_argument("--data-root",default=None);p.add_argument("--split",choices=["validation","test"],default="test");p.add_argument("--device",default="auto");p.add_argument("--max-contexts",type=int,default=None);p.add_argument("--no-history",action="store_true");p.add_argument("--no-profile",action="store_true");p.add_argument("--output",default=None);return p.parse_args()
@torch.inference_mode()
def eval_file(model,tok,path,device,ml,hc,limit,nohist,noprofile):
    c=pairs=wins=hits=0;rr=ndcg=rank_sum=margin=0.
    for r in rows(path):
        if limit is not None and c>=limit:break
        cand=[r["positive_candidate"],*r["negative_candidates"]];prompts=[build_record_prompt(r,x,no_history=nohist,no_profile=noprofile,history_char_limit=hc) for x in cand];b=tok(prompts,padding=True,truncation=True,max_length=ml,return_tensors="pt");b={k:v.to(device) for k,v in b.items()};s=model(b["input_ids"],b["attention_mask"]).float().cpu().tolist();pos=s[0];neg=s[1:];pairs+=len(neg);wins+=sum(pos>x for x in neg);rank=1+sum(x>=pos for x in neg);hits+=rank==1;rr+=1/rank;ndcg+=1/math.log2(rank+1);rank_sum+=rank;margin+=pos-max(neg);c+=1
    return {"contexts":c,"pairs":pairs,"pairwise_accuracy":wins/pairs if pairs else 0.,"hit_at_1":hits/c if c else 0.,"mrr":rr/c if c else 0.,"ndcg":ndcg/c if c else 0.,"average_positive_rank":rank_sum/c if c else 0.,"average_top_margin":margin/c if c else 0.}
def aggregate(rs):
    c=sum(x["contexts"] for x in rs);p=sum(x["pairs"] for x in rs)
    if not c:return {"contexts":0,"pairs":0}
    return {"contexts":c,"pairs":p,"pairwise_accuracy":sum(x["pairwise_accuracy"]*x["pairs"] for x in rs)/p if p else 0.,"hit_at_1":sum(x["hit_at_1"]*x["contexts"] for x in rs)/c,"mrr":sum(x["mrr"]*x["contexts"] for x in rs)/c,"ndcg":sum(x["ndcg"]*x["contexts"] for x in rs)/c,"average_positive_rank":sum(x["average_positive_rank"]*x["contexts"] for x in rs)/c,"average_top_margin":sum(x["average_top_margin"]*x["contexts"] for x in rs)/c}
def main():
    a=parse_args();cfg=get_config();tc=cfg.get("training",{}).get("preference",{});rc=cfg.get("preference_model",{});cp=resolve_project_path(a.checkpoint or rc.get("checkpoint") or Path(tc.get("checkpoint_dir","checkpoints/tripmind-preference"))/"best");root=resolve_project_path(a.data_root or tc.get("output_dir","data/processed/genup_preference"));paths=sorted(root.glob(f"*-{a.split}.jsonl"));
    if not paths:raise FileNotFoundError(f"No *-{a.split}.jsonl files under {root}")
    device=resolve_device(a.device);model,tok,meta=PreferenceScoringModel.load_checkpoint(cp,device=device);model.eval();ml=int(meta.get("max_length",1024));hc=int(meta.get("history_char_limit",6000));per={};allm=[]
    for path in paths:
        m=eval_file(model,tok,path,device,ml,hc,a.max_contexts,a.no_history,a.no_profile);per[path.name.split("-",1)[0]]=m;allm.append(m)
    result={"checkpoint":str(cp),"split":a.split,"no_history":a.no_history,"no_profile":a.no_profile,"overall":aggregate(allm),"per_source":per};print(json.dumps(result,indent=2,sort_keys=True))
    if a.output:
        out=resolve_project_path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
