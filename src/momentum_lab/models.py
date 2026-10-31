"""Paper 67 stateless LSTM, a CPU/Metal port of the preserved author Keras model.

Same 63-step sequences, tanh positions, input/output dropout, Sharpe loss,
Adam, gradient norm clipping, output max-norm and validation-only selection.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from .paths import ROOT, configuration
from .dataset import FEATURE_COLUMNS
from .training_events import emit

def resolve_device(requested='auto'):
    if requested=='auto':
        requested='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    return torch.device(requested)

class DeepMomentumNetwork(nn.Module):
    def __init__(self, inputs, hidden, dropout):
        super().__init__();self.dropout=dropout
        self.lstm=nn.LSTM(inputs,hidden,batch_first=True)
        self.output=nn.Linear(hidden,1)
        nn.init.xavier_uniform_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)
        nn.init.zeros_(self.lstm.bias_ih_l0);nn.init.zeros_(self.lstm.bias_hh_l0)
        # Keras has one trainable bias vector. Freeze PyTorch's redundant second
        # vector so that Adam cannot update the effective bias twice per step.
        self.lstm.bias_hh_l0.requires_grad_(False)
        with torch.no_grad():self.lstm.bias_ih_l0[hidden:2*hidden].fill_(1.)
        nn.init.xavier_uniform_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,x):
        if self.training and self.dropout:
            # Keras LSTM uses one input dropout mask for all time steps.
            mask=nn.functional.dropout(torch.ones_like(x[:,:1,:]),self.dropout,True)
            x=x*mask
        h,_=self.lstm(x)
        return torch.tanh(self.output(nn.functional.dropout(h,self.dropout,self.training))).squeeze(-1)

def sharpe_loss(position,target):
    captured=position*target;mu=captured.mean()
    return -np.sqrt(252)*mu/torch.sqrt((captured*captured).mean()-mu*mu+1e-9)

def parameter_grid(cfg):
    rng=np.random.default_rng(cfg["seed"])
    params=[]
    seen=set()
    while len(params)<cfg["random_search_trials"]:
        hp={k:rng.choice(cfg[v]).item() for k,v in {
            "hidden":"hidden_sizes","dropout":"dropouts","batch":"batch_sizes",
            "lr":"learning_rates","clip":"gradient_norms"}.items()}
        key=tuple(hp.items())
        if key not in seen:params.append(hp);seen.add(key)
    return params

def training_signature(feature_hash,year,lbw,hp,cfg,seed):
    values={"features":feature_hash,"year":year,"lbw":lbw,"hp":hp,"seed":seed,"core":"paper67_lstm_v1",
        **{k:cfg[k] for k in ["sequence_length","train_fraction","max_epochs","patience","validation_objective"]}}
    return hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()

def load_panel(lbw=None):
    f=pd.read_parquet(ROOT/"data/processed/features.parquet").reset_index(drop=True)
    if lbw is not None:
        paths=sorted((ROOT/f"data/processed/cpd/{lbw}").glob("*.parquet"))
        if len(paths)!=f.ticker.nunique():raise FileNotFoundError(f"CPD {lbw} not yet complete for all instruments")
        cols=["ticker","date",f"cp_score_{lbw}",f"cp_rl_{lbw}"]
        cp=pd.concat([pd.read_parquet(p,columns=cols) for p in paths],ignore_index=True)
        f=f.merge(cp,on=["ticker","date"],how="left",validate="one_to_one")
        if f[cols[2:]].isna().any().any():raise ValueError(f"Incomplete daily CPD {lbw} panel")
    return f.sort_values(["ticker","date"]).reset_index(drop=True)

def sequence_data(f,year,cfg,columns):
    """Per-instrument chronological 90/10 splits; purge labels at both boundaries."""
    length=cfg["sequence_length"]; boundary=pd.Timestamp(year,1,1)
    train=[];valid=[];eligible=[];split_rows=[]
    for ticker,g in f.groupby("ticker",sort=True):
        if g.published_first_test.iloc[0]>year:continue
        history=g[(g.date<boundary)&(g.label_date<boundary)].sort_values("date")
        cut=int(len(history)*cfg["train_fraction"])
        if len(history)-cut<length or cut<length:continue
        val_start=history.date.iloc[cut]
        a=history.iloc[:cut];a=a[a.label_date<val_start]
        b=history.iloc[cut:]
        for source,out in [(a,train),(b,valid)]:
            count=len(source)//length
            if not count:continue
            source=source.iloc[-count*length:]
            x=source[columns].to_numpy(dtype=np.float32).reshape(count,length,len(columns))
            y=source.target_returns.to_numpy(dtype=np.float32).reshape(count,length)
            out.append((x,y))
        eligible.append(ticker)
        split_rows.append({"ticker":ticker,"fold":year,"training_last_feature":str(a.date.max().date()),
            "training_last_label":str(a.label_date.max().date()),"validation_first":str(b.date.min().date()),
            "validation_last_label":str(b.label_date.max().date()),"test_start":str(boundary.date())})
    if not train or not valid:raise ValueError("No eligible training/validation sequences")
    arrays=[torch.from_numpy(np.concatenate([pair[j] for pair in group])) for group in [train,valid] for j in [0,1]]
    return arrays,eligible,split_rows

def train_trial(arrays,hp,cfg,seed,checkpoint):
    torch.manual_seed(seed);np.random.seed(seed);torch.set_num_threads(2)
    device=resolve_device(cfg.get("device","auto"))
    if device.type=="mps":
        if not torch.backends.mps.is_available():raise RuntimeError("Configured Metal GPU is unavailable; run with GPU access or set device=cpu")
        torch.mps.manual_seed(seed)
    x,y,vx,vy=[a.to(device) for a in arrays]
    model=DeepMomentumNetwork(x.shape[-1],hp["hidden"],hp["dropout"]).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=hp["lr"],eps=1e-7)
    generator=torch.Generator().manual_seed(seed)
    best=float('inf');wait=0;history=[];start=time.monotonic()
    for epoch in range(cfg["max_epochs"]):
        model.train();order=torch.randperm(len(x),generator=generator);losses=[]
        for ids in order.split(hp["batch"]):
            optimizer.zero_grad(set_to_none=True);loss=sharpe_loss(model(x[ids]),y[ids])
            if not torch.isfinite(loss):raise FloatingPointError("Nonfinite training Sharpe loss")
            loss.backward()
            # tf.keras Adam(clipnorm=...) clips each parameter gradient separately.
            for parameter in model.parameters():
                if parameter.grad is not None:torch.nn.utils.clip_grad_norm_([parameter],hp["clip"])
            optimizer.step()
            with torch.no_grad():
                norm=model.output.weight.norm();model.output.weight.mul_(torch.clamp(3/norm,max=1.))
            losses.append(loss.detach())
        model.eval()
        with torch.no_grad():pred=torch.cat([model(part) for part in vx.split(512)])
        val=sharpe_loss(pred,vy).item()
        history.append({"epoch":epoch+1,"training_loss":torch.stack(losses).mean().item(),"validation_loss":val})
        if val<best-1e-4:
            best=val;wait=0
            temporary=checkpoint.with_suffix(".pt.tmp")
            torch.save({"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"hp":hp,"epoch":epoch+1,
                "validation_loss":val,"seed":seed,"device_at_fit":str(device)},temporary)
            temporary.replace(checkpoint)
        else:wait+=1
        emit('epoch',checkpoint,**history[-1],best_validation_loss=best,device=str(device))
        if (epoch+1)%25==0:print("epoch",epoch+1,"val",round(val,3),"seconds",round(time.monotonic()-start),flush=True)
        if wait>=cfg["patience"]:break
    saved=torch.load(checkpoint,weights_only=False,map_location="cpu");model.load_state_dict(saved["state_dict"])
    return model,best,history,round(time.monotonic()-start,3)

def predict(model,f,tickers,year,end,columns,length):
    records=[];model.eval();device=next(model.parameters()).device
    for ticker,g in f[f.ticker.isin(tickers)].groupby("ticker",sort=True):
        g=g.sort_values("date").reset_index(drop=True)
        ix=np.where((g.date>=pd.Timestamp(year,1,1))&(g.label_date<pd.Timestamp(end,1,1)))[0]
        ix=ix[ix>=length-1]
        if not len(ix):continue
        x=g[columns].to_numpy(dtype=np.float32)
        positions=[]
        with torch.no_grad():
            for batch in np.array_split(ix,max(1,int(np.ceil(len(ix)/512)))):
                seq=np.stack([x[i-length+1:i+1] for i in batch])
                positions.extend(model(torch.from_numpy(seq).to(device))[:,-1].cpu().numpy())
        out=g.iloc[ix][["ticker","date","label_date","target_returns","daily_vol","raw_target_return",
            "decision_contract","asset_class","norm_monthly_return","norm_annual_return"]].copy()
        out["position"]=positions;out["fold"]=year;records.append(out)
    return pd.concat(records,ignore_index=True)

def run_model(year,lbw=None,config_path="configs/config.yaml"):
    cfg=configuration(config_path);name="LSTM" if lbw is None else f"LSTM-CPD-{lbw}"
    feature_hash=hashlib.sha256((ROOT/"data/processed/features.parquet").read_bytes()).hexdigest()
    source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    dest=ROOT/f"results/replication/checkpoints/{name}/{year}";dest.mkdir(parents=True,exist_ok=True)
    result=dest/"predictions.parquet";receipt=dest/"receipt.json"
    if result.exists() and receipt.exists():
        previous=json.loads(receipt.read_text())
        if previous.get("config")==cfg and previous.get("features_sha256")==feature_hash and previous.get("model_source_sha256")==source_hash:
            return name,year,"cached"
    f=load_panel(lbw);columns=FEATURE_COLUMNS+([] if lbw is None else [f"cp_score_{lbw}",f"cp_rl_{lbw}"])
    arrays,eligible,splits=sequence_data(f,year,cfg,columns)
    pd.DataFrame(splits).to_csv(dest/"splits.csv",index=False)
    ends=cfg["test_boundaries"];index=ends.index(year);end=ends[index+1] if index+1<len(ends) else cfg["last_test_end"]
    trials=[];best=float('inf');best_model=None
    for trial,hp in enumerate(parameter_grid(cfg)):
        cp=dest/f"trial_{trial}.pt";hist=dest/f"trial_{trial}_history.csv";summary=dest/f"trial_{trial}.json"
        signature=training_signature(feature_hash,year,lbw,hp,cfg,cfg["seed"]+trial)
        cached_info=json.loads(summary.read_text()) if summary.exists() else {}
        if cp.exists() and cached_info.get("training_signature")==signature:
            info=cached_info;saved=torch.load(cp,weights_only=False,map_location="cpu")
            model=DeepMomentumNetwork(len(columns),hp["hidden"],hp["dropout"]);model.load_state_dict(saved["state_dict"])
            val=info["validation_loss"]
        else:
            print("Fit",name,year,"trial",trial,hp,"sequences",len(arrays[0]),len(arrays[2]),flush=True)
            model,val,history,seconds=train_trial(arrays,hp,cfg,cfg["seed"]+trial,cp)
            pd.DataFrame(history).to_csv(hist,index=False)
            info={"trial":trial,"parameters":hp,"validation_loss":val,"epochs":len(history),"seconds":seconds,
                "seed":cfg["seed"]+trial,"device":str(next(model.parameters()).device),"training_signature":signature}
            summary.write_text(json.dumps(info,indent=2))
            emit('candidate_completed',cp,validation_loss=val,epochs=len(history))
        trials.append(info)
        if val<best:best=val;best_model=model;best_trial=trial
    predictions=predict(best_model,f,eligible,year,end,columns,cfg["sequence_length"])
    predictions["strategy"]=name;predictions.to_parquet(result,index=False)
    receipt.write_text(json.dumps({"strategy":name,"fold":year,"test_end_exclusive":end,
        "eligible_instruments":eligible,"features":columns,"best_trial":best_trial,"validation_loss":best,
        "trials":trials,"rows":len(predictions),"config":cfg,
        "features_sha256":feature_hash,"model_source_sha256":source_hash},indent=2))
    print("Finished",name,year,len(predictions),flush=True)
    return name,year,len(predictions)

def run(config_path="configs/config.yaml",windows=None,years=None):
    cfg=configuration(config_path)
    for lbw in ([None]+cfg["cpd_windows"] if windows is None else windows):
        for year in (years or cfg["test_boundaries"]):run_model(year,lbw,config_path)
