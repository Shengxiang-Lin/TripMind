from __future__ import annotations
import argparse, json, math, random, sys
from pathlib import Path
from typing import Any, Iterable
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tripmind.config import get_config, resolve_project_path
from tripmind.preference_model import PreferenceScoringModel, load_tokenizer, resolve_device
from tripmind.preference_prompt import build_record_prompt
from tripmind.processing.common import stable_id

def parse_args():
    p=argparse.ArgumentParser(description="Train TripMind's GenUP preference scorer.")
    for name,kwargs in [("--data-root",{}),("--model",{}),("--output",{}),("--device",{"default":"auto"})]: p.add_argument(name,**kwargs)
    for name in ["--epochs","--batch-size","--gradient-accumulation-steps","--max-length","--history-char-limit","--seed","--max-train-contexts","--max-validation-contexts","--log-every","--shuffle-buffer","--num-workers"]: p.add_argument(name,type=int,default=None)
    for name in ["--learning-rate","--weight-decay","--warmup-ratio","--max-grad-norm","--history-dropout"]: p.add_argument(name,type=float,default=None)
    p.add_argument("--gradient-checkpointing",action="store_true"); p.add_argument("--no-gradient-checkpointing",action="store_true")
    return p.parse_args()
def iter_jsonl(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip(): yield json.loads(line)
def data_files(root,split):
    paths=sorted(root.glob(f"*-{split}.jsonl"))
    if not paths: raise FileNotFoundError(f"No *-{split}.jsonl files found under {root}")
    return paths
def count_contexts(paths,limit=None):
    n=0
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n+=1
                    if limit and n>=limit:return n
    return n
def drop_history(cid,nid,fraction,seed):
    if fraction<=0:return False
    return int(stable_id("history_dropout",seed,cid,nid)[:12],16)/16**12 < fraction
class PairwiseJsonlDataset(IterableDataset):
    def __init__(self,paths,history_dropout,history_char_limit,seed,shuffle_buffer,max_contexts):
        self.paths=paths; self.history_dropout=history_dropout; self.history_char_limit=history_char_limit; self.seed=seed; self.shuffle_buffer=max(1,shuffle_buffer); self.max_contexts=max_contexts; self.epoch=0
    def set_epoch(self,e):self.epoch=e
    def _pairs(self):
        worker=get_worker_info(); nw=worker.num_workers if worker else 1; wid=worker.id if worker else 0; seen=idx=0; paths=list(self.paths); random.Random(self.seed+self.epoch).shuffle(paths)
        for path in paths:
            for record in iter_jsonl(path):
                if self.max_contexts is not None and seen>=self.max_contexts:return
                seen+=1; positive=record["positive_candidate"]; cid=str(record["context_id"])
                for negative in record["negative_candidates"]:
                    if idx%nw!=wid: idx+=1; continue
                    nohist=drop_history(cid,str(negative.get("poi_id","")),self.history_dropout,self.seed)
                    kw={"no_history":nohist,"history_char_limit":self.history_char_limit}
                    yield {"positive":build_record_prompt(record,positive,**kw),"negative":build_record_prompt(record,negative,**kw)}
                    idx+=1
    def __iter__(self):
        worker=get_worker_info(); rng=random.Random(self.seed+1009*self.epoch+(worker.id if worker else 0)); buf=[]
        for pair in self._pairs():
            buf.append(pair)
            if len(buf)>=self.shuffle_buffer: yield buf.pop(rng.randrange(len(buf)))
        while buf: yield buf.pop(rng.randrange(len(buf)))
class PairCollator:
    def __init__(self,tok,max_length): self.tok=tok; self.max_length=max_length
    def __call__(self,rows):
        texts=[r["positive"] for r in rows]+[r["negative"] for r in rows]
        return self.tok(texts,padding=True,truncation=True,max_length=self.max_length,return_tensors="pt")
@torch.inference_mode()
def evaluate(model,tok,paths,device,max_length,history_char_limit,max_contexts):
    model.eval(); contexts=pairs=wins=hits=0; rr=ndcg=rank_sum=0.0
    for path in paths:
        for record in iter_jsonl(path):
            if max_contexts is not None and contexts>=max_contexts:break
            candidates=[record["positive_candidate"],*record["negative_candidates"]]
            prompts=[build_record_prompt(record,c,history_char_limit=history_char_limit) for c in candidates]
            b=tok(prompts,padding=True,truncation=True,max_length=max_length,return_tensors="pt"); b={k:v.to(device) for k,v in b.items()}; scores=model(b["input_ids"],b["attention_mask"]).float().cpu().tolist(); pos=scores[0]; neg=scores[1:]
            pairs+=len(neg); wins+=sum(pos>x for x in neg); rank=1+sum(x>=pos for x in neg); hits+=rank==1; rr+=1/rank; ndcg+=1/math.log2(rank+1); rank_sum+=rank; contexts+=1
        if max_contexts is not None and contexts>=max_contexts:break
    model.train(); return {"contexts":contexts,"pairs":pairs,"pairwise_accuracy":wins/pairs if pairs else 0.0,"hit_at_1":hits/contexts if contexts else 0.0,"mrr":rr/contexts if contexts else 0.0,"ndcg":ndcg/contexts if contexts else 0.0,"average_positive_rank":rank_sum/contexts if contexts else 0.0}
def main():
    a=parse_args(); cfg=get_config(); c=cfg.get("training",{}).get("preference",{}); pick=lambda arg,key,default: arg if arg is not None else c.get(key,default)
    seed=int(pick(a.seed,"seed",42)); random.seed(seed); torch.manual_seed(seed); [torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None]
    root=resolve_project_path(a.data_root or c.get("output_dir","data/processed/genup_preference")); out=resolve_project_path(a.output or c.get("checkpoint_dir","checkpoints/tripmind-preference")); base=a.model or c.get("base_model","Qwen/Qwen2.5-0.5B-Instruct")
    epochs=int(pick(a.epochs,"epochs",2)); bs=int(pick(a.batch_size,"batch_size",2)); ga=int(pick(a.gradient_accumulation_steps,"gradient_accumulation_steps",8)); lr=float(pick(a.learning_rate,"learning_rate",2e-5)); wd=float(pick(a.weight_decay,"weight_decay",.01)); wr=float(pick(a.warmup_ratio,"warmup_ratio",.05)); gn=float(pick(a.max_grad_norm,"max_grad_norm",1.0)); ml=int(pick(a.max_length,"max_length",1024)); hc=int(pick(a.history_char_limit,"history_char_limit",6000)); hd=float(pick(a.history_dropout,"history_dropout",.25)); gc=False if a.no_gradient_checkpointing else True if a.gradient_checkpointing else bool(c.get("gradient_checkpointing",True)); sb=int(a.shuffle_buffer or 256); nw=int(a.num_workers or 0); log_every=int(a.log_every or 20)
    train=data_files(root,"train"); val=data_files(root,"validation"); n=count_contexts(train,a.max_train_contexts); manifest=json.loads((root/"manifest.json").read_text()) if (root/"manifest.json").is_file() else {}; negs=int(manifest.get("negatives_per_positive",4)); total_steps=max(1,math.ceil(n*negs/bs/ga))*epochs; device=resolve_device(a.device); tok=load_tokenizer(base); model=PreferenceScoringModel.from_base_model(base,device=device,gradient_checkpointing=gc); opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd); sched=get_linear_schedule_with_warmup(opt,int(total_steps*wr),total_steps)
    ds=PairwiseJsonlDataset(train,hd,hc,seed,sb,a.max_train_contexts); loader=DataLoader(ds,batch_size=bs,collate_fn=PairCollator(tok,ml),num_workers=nw,pin_memory=device.type=="cuda"); out.mkdir(parents=True,exist_ok=True); best=-1.; hist=[]; step=0
    print(json.dumps({"device":str(device),"base_model":base,"train_contexts":n,"epochs":epochs,"batch_size":bs,"gradient_accumulation_steps":ga,"estimated_optimizer_steps":total_steps},indent=2)); opt.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        ds.set_epoch(epoch); model.train(); loss_sum=acc_sum=micro=0.; progress=tqdm(loader,desc=f"epoch {epoch+1}/{epochs}")
        for batch in progress:
            batch={k:v.to(device,non_blocking=True) for k,v in batch.items()}; b=batch["input_ids"].size(0)//2; scores=model(batch["input_ids"],batch["attention_mask"]); pos,neg=scores[:b],scores[b:]; loss=F.softplus(neg-pos).mean(); (loss/ga).backward(); loss_sum+=float(loss.detach().cpu()); acc_sum+=float((pos>neg).float().mean().detach().cpu()); micro+=1
            if micro%ga==0: clip_grad_norm_(model.parameters(),gn); opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step+=1
            if micro%log_every==0: progress.set_postfix(loss=f"{loss_sum/micro:.4f}",pair_acc=f"{acc_sum/micro:.3f}",step=step)
        if micro%ga: clip_grad_norm_(model.parameters(),gn); opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step+=1
        metrics=evaluate(model,tok,val,device,ml,hc,a.max_validation_contexts); row={"epoch":epoch+1,"train_loss":loss_sum/max(1,micro),"train_pairwise_accuracy":acc_sum/max(1,micro),"validation":metrics,"global_step":step}; hist.append(row); print(json.dumps(row,indent=2)); extra={"training_epoch":epoch+1,"history_dropout":hd,"validation_metrics":metrics}; model.save_checkpoint(out/"last",tok,base_model=base,max_length=ml,history_char_limit=hc,extra=extra)
        if metrics["pairwise_accuracy"]>best: best=metrics["pairwise_accuracy"]; model.save_checkpoint(out/"best",tok,base_model=base,max_length=ml,history_char_limit=hc,extra=extra)
    (out/"train_summary.json").write_text(json.dumps({"base_model":base,"best_validation_pairwise_accuracy":best,"history":hist,"best_checkpoint":str(out/"best")},indent=2,sort_keys=True)+"\n",encoding="utf-8"); print("Training complete:",out/"best")
if __name__=="__main__":main()
