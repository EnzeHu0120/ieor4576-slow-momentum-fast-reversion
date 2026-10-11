"""Copy/import a local course cache, rebuild causal returns, record all missing dates."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from momentum_lab.contracts import continuous_contract_returns

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root',type=Path,required=True,help='Course-style raw/contracts.parquet parent')
    p.add_argument('--series',type=Path,required=True,help='JSON mapping ticker -> contrcode, pilot only')
    p.add_argument('--start',default='2010-01-01');p.add_argument('--end',default='2025-12-31')
    a=p.parse_args();mapping=json.loads(a.series.read_text());r={};ids={};manifest=[]
    meta=pd.read_parquet(a.raw_root/'contracts.parquet')
    for ticker,code in mapping.items():
        path=a.raw_root/'prices'/f'{int(code)}.parquet';bars=pd.read_parquet(path)
        ret,held=continuous_contract_returns(bars,meta[meta.contrcode.eq(int(code))])
        r[ticker]=ret;ids[ticker]=held
        manifest.append({'ticker':ticker,'contrcode':code,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                         'rows':len(bars),'price_start':str(bars.date.min()),'price_end':str(bars.date.max())})
    panel=pd.DataFrame(r).loc[a.start:a.end];contracts=pd.DataFrame(ids).reindex_like(panel)
    # Restriction is explicit and saved for inspection, never silently converted to zero.
    missing=panel[panel.isna().any(axis=1)]
    clean=panel.dropna(how='any');contracts=contracts.reindex(clean.index)
    out=ROOT/'Data/processed';out.mkdir(parents=True,exist_ok=True)
    panel.to_parquet(out/'pilot_returns_with_missing.parquet');missing.to_csv(out/'missing_dates.csv')
    # Internal missing dates would turn a row-based rolling horizon into an irregular one.
    # Fail on internal gaps; initial warm-up dates can be removed safely.
    if len(clean) and missing.index.to_series().between(clean.index.min(),clean.index.max()).any():
        raise ValueError(f'Internal incomplete dates: {len(missing)}; inspect series before choosing universe')
    clean.to_parquet(out/'pilot_returns.parquet');contracts.to_parquet(out/'pilot_contracts.parquet')
    (out/'course_manifest.json').write_text(json.dumps({'status':'substitute_data_pilot_only',
        'source':'existing WRDS trdstrm course cache; not a new WRDS download',
        'mapping':mapping,'rows_after_audit':len(clean),'missing_dates':len(missing),
        'rule':'nearest vendor expiry/notice, exit 5 calendar days early, lagged OI, same-contract return',
        'limitations':['vendor metadata not timestamped','nonpositive settlements excluded',
                      'settlement is valuation proxy; no executable OHLC data',
                      'common panel restriction; inspect missing_dates.csv'],
        'files':manifest},indent=2))
    print(f'{len(clean)} complete sessions; {len(missing)} incomplete initial sessions recorded. {out}')
