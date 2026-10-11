"""Preview first; password is requested only for explicit WRDS inspect/download."""
import argparse
from datetime import date
from getpass import getpass
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def selection(path):
    c=json.loads(Path(path).read_text())
    if c['decision_status']!='agreed_by_team':
        raise ValueError('Data choice is pending. Agree universe/dates and set decision_status=agreed_by_team.')
    start,end=date.fromisoformat(c['start']),date.fromisoformat(c['end'])
    if start>end or not c['series']:
        raise ValueError('Select a nonempty series list and valid dates')
    codes=[int(x['contrcode']) for x in c['series']]
    if len(codes)!=len(set(codes)) or any(x<=0 for x in codes):
        raise ValueError('Use unique positive contrcode IDs')
    allowed={'settlement':'settlement','open':'open_','high':'high','low':'low',
             'volume':'volume','oi':'openinterest'}
    fields=c['price_fields']
    if 'settlement' not in fields or 'oi' not in fields or not set(fields)<=allowed.keys():
        raise ValueError('Include settlement and oi; use only supported fields')
    return c,codes,allowed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(ROOT/'configs/wrds_selection.example.json'))
    p.add_argument('--username')
    p.add_argument('--inspect',action='store_true',help='Connect; list accessible table columns only')
    p.add_argument('--download',action='store_true',help='Download the agreed config into a NEW timestamped raw folder')
    a=p.parse_args()
    if a.inspect and a.download: p.error('Choose inspect or download')
    if not (a.inspect or a.download):
        print(Path(a.config).read_text());print('Preview only; no connection. Final data selection remains a team decision.');return
    parsed=selection(a.config) if a.download else None
    identity=ROOT/'configs/identity.local.json'
    default=json.loads(identity.read_text()).get('wrds_username','') if identity.exists() else ''
    username=a.username or input(f'WRDS username [{default}]: ').strip() or default
    if not username: raise ValueError('WRDS username required')
    # Direct PostgreSQL + getpass: no .pgpass creation, no password in args/files/logs.
    import psycopg2
    import pandas as pd
    conn=psycopg2.connect(host='wrds-pgdata.wharton.upenn.edu',port=9737,dbname='wrds',
                          user=username,password=getpass('WRDS password (not saved): '),
                          sslmode='require',connect_timeout=30)
    conn.set_session(readonly=True,autocommit=True)
    try:
        def query(sql,params=()):
            with conn.cursor() as cur:
                cur.execute(sql,params)
                return pd.DataFrame(cur.fetchall(),columns=[d[0] for d in cur.description])
        if a.inspect:
            print(query("""SELECT table_name,column_name,data_type FROM information_schema.columns
                WHERE table_schema=%s AND table_name IN (%s,%s) ORDER BY table_name,ordinal_position""",
                ('trdstrm','wrds_contract_info','wrds_fut_contract')).to_string(index=False))
            # A LIMIT 0 checks SELECT rights without transferring price history.
            query('SELECT * FROM trdstrm.wrds_fut_contract LIMIT 0')
            print('Table SELECT permission confirmed; no price history downloaded.');return
        from datetime import datetime,timezone
        c,codes,allowed=parsed
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        out=ROOT/'data/raw/wrds'/stamp;out.mkdir(parents=True,exist_ok=False)
        contracts=query('SELECT * FROM trdstrm.wrds_contract_info WHERE contrcode = ANY(%s)',(codes,))
        contracts.to_parquet(out/'contracts_vendor.parquet',index=False)
        cols=', '.join(f'fc.{allowed[f]} AS {f}' for f in c['price_fields'])
        counts={}
        for code in codes:
            bars=query(f'''SELECT DISTINCT fc.futcode,fc.date_ AS date,{cols}
                FROM trdstrm.wrds_fut_contract fc JOIN trdstrm.wrds_contract_info ci
                ON fc.futcode=ci.futcode WHERE ci.contrcode=%s AND fc.date_ BETWEEN %s AND %s
                ORDER BY fc.futcode,fc.date_''',(code,c['start'],c['end']))
            if bars.duplicated(['futcode','date']).any():
                raise ValueError('Conflicting vendor rows; raw snapshot retained for audit')
            bars.to_parquet(out/f'{code}.parquet',index=False);counts[str(code)]=len(bars)
        (out/'manifest.json').write_text(json.dumps({'selection':c,'rows':counts,
            'source':'WRDS trdstrm','download_utc':stamp,'status':'raw_unadjusted'},indent=2))
        print(f'Saved raw snapshot: {out}')
    finally:
        conn.close()


if __name__=='__main__': main()
