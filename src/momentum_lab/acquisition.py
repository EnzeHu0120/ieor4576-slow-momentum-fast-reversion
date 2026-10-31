"""Authenticated WRDS snapshots. Passwords are obtained by libpq, never logged."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
import pandas as pd
from .paths import ROOT, configuration, prepare_directories

def connect():
    import psycopg2
    username=os.environ.get('WRDS_USERNAME') or os.environ.get('PGUSER')
    if not username:raise RuntimeError('Set WRDS_USERNAME to your own authorized WRDS account; use secure libpq authentication.')
    c = psycopg2.connect(host="wrds-pgdata.wharton.upenn.edu", port=9737,
        dbname="wrds", user=username,
        sslmode="require", connect_timeout=30,
        **({"password": os.environ["WRDS_PASSWORD"]} if os.environ.get("WRDS_PASSWORD") else {}))
    c.set_session(readonly=True, autocommit=True)
    return c

def frame(c, sql, params=()):
    with c.cursor() as q:
        q.execute(sql, params)
        return pd.DataFrame(q.fetchall(), columns=[x[0] for x in q.description])

def download_one(row, cfg):
    dest = ROOT / "data/raw/wrds" / row.ticker
    dest.mkdir(parents=True, exist_ok=True)
    receipt = dest / "manifest.json"
    if receipt.exists() and json.loads(receipt.read_text()).get("status") == "complete":
        return row.ticker, "cached"
    codes = [int(x) for x in str(row.contract_codes).split("|")]
    c = connect()
    try:
        meta = frame(c, "SELECT DISTINCT * FROM trdstrm.wrds_contract_info WHERE contrcode = ANY(%s)", (codes,))
        meta.to_parquet(dest / "contracts.parquet", index=False)
        futcodes = sorted(set(meta.futcode.map(int)))
        bars = frame(c, """SELECT DISTINCT futcode, date_ AS date, settlement,
                  volume, openinterest AS oi FROM trdstrm.wrds_fut_contract
                  WHERE futcode = ANY(%s) AND date_ BETWEEN %s AND %s
                  ORDER BY date_, futcode""", (futcodes, cfg["start"], cfg["end"]))
        bars.to_parquet(dest / "bars.parquet", index=False)
        files = [{"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                 for p in [dest / "contracts.parquet", dest / "bars.parquet"]]
        receipt.write_text(json.dumps({"status":"complete", "source":"WRDS trdstrm",
            "downloaded_utc":datetime.now(timezone.utc).isoformat(), "ticker":row.ticker,
            "contract_codes":codes, "start_requested":cfg["start"], "end_requested":cfg["end"],
            "rows":len(bars), "files":files}, indent=2))
        return row.ticker, len(bars)
    finally:
        c.close()

def download(config_path="configs/config.yaml"):
    cfg = configuration(config_path); prepare_directories()
    universe = pd.read_csv(ROOT / "configs/universe.csv")
    errors = []
    # Composite historical contracts sometimes omit scheduled expiry fields.
    # Retrieve the vendor's corresponding pit-contract calendars, not inferred
    # final-observation dates, for the exact same contract month and underlying.
    for ticker,code in {"GI":342,"ZW":430,"ZZ":364}.items():
        dest=ROOT/f"data/raw/wrds/{ticker}/calendar_reference.parquet"
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():
            c=connect()
            try:
                meta=frame(c,"SELECT DISTINCT * FROM trdstrm.wrds_contract_info WHERE contrcode=%s",(code,))
                meta.to_parquet(dest,index=False)
                dest.with_suffix('.json').write_text(json.dumps({"source":"WRDS trdstrm.wrds_contract_info",
                    "reference_contract_code":code,"underlying":ticker,
                    "sha256":hashlib.sha256(dest.read_bytes()).hexdigest()},indent=2))
            finally:c.close()
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(download_one, row, cfg): row.ticker for row in universe.itertuples()}
        for job in as_completed(jobs):
            try:
                print("WRDS", *job.result(), flush=True)
            except Exception as e:
                # Avoid displaying raw connection strings or password-bearing errors.
                errors.append({"ticker": jobs[job], "error_type":type(e).__name__})
                print("WRDS", jobs[job], type(e).__name__, flush=True)
    (ROOT / "data/raw/wrds/download_status.json").write_text(json.dumps({"errors":errors},indent=2))
    if errors:
        raise RuntimeError(f"{len(errors)} instrument downloads failed; successful snapshots are preserved")
