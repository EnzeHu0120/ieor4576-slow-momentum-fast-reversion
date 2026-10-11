# Data choices — pending team discussion

No new WRDS history has been requested. The user will enter the password locally. The downloader uses a read-only SSL PostgreSQL connection; it does not save the password, create .pgpass, or put credentials on the command line.

Run `python scripts/01_get_data.py` to preview the empty shared template. Personal WRDS authentication/downloading is implemented only in the local Enze workspace (`scripts/wrds_download.py`), including a hidden password prompt, symbol/contrcode selection, date range, catalogue lookup and dated snapshots under `Enze/Data`. The shared main repository contains no personal authentication defaults. Actual live WRDS permissions remain untested until the user runs the downloader. Source tables/fields were checked against existing course source; credentials and raw data are never uploaded.

| Candidate | Access/evidence | Use and open issue |
|---|---|---|
| Pinnacle CLC | Commercial source in #67; no dataset bundled | Best fidelity if licensed 50-series history available; confirm purchase/access/roll adjustments first |
| WRDS Datastream `trdstrm.wrds_contract_info`, `wrds_fut_contract` | Course `futspread` source and existing ten-series daily cache | Most practical starting candidate; username alone does not establish current institutional entitlement or permitted redistribution |
| Course raw cache | Ten series, bars about Dec 2009–Dec 2025, mostly settlement+OI | Local pilot only; replace same-day selection/full-history liquidity filters with an auditable causal rule |
| Author CHRIS/Quandl example | README points to Nasdaq Data Link and 100 tickers | Current API entitlement/continuity not verified; do not assume the historical “free” README claim still works |
| CFTC COT | Weekly positioning data | Not daily settlement prices; any future use needs actual release dates and publication lag |
| Regime study data | WRDS selection pending | Reuse Regime Detection methods only; its old datasets are explicitly excluded by user instruction |
| IBKR | TWS/Gateway/API permissions and market subscriptions required | Recent read-only validation / later paper workflow; not a substitute for 1990–2020 futures archive |

## Decide together before download

- Fidelity target: original Pinnacle panel, or explicitly disclosed WRDS adaptation?
- Core universe and contrcodes; diversity across CM/EQ/FI/FX; USD units, delivery cutoffs and contract size.
- Longest defensible training history and an untouched test period. Pilot decade is not the final universe.
- Settlement/open/high/low/volume/OI, metadata, corporate/contract specification changes, historical holidays.
- Roll convention: fixed calendar versus lagged volume/OI, dates known at decision time, delivery exposure.
- Nonpositive price handling: contract-dollar P&L rather than naive percentage returns, especially CL in 2020.
- Storage, university rights and cost: licensed raw data remain local and untracked.

Public primary-source references checked 2026-10-11:
- [WRDS Datastream guide](https://wrds-www.wharton.upenn.edu/documents/1537/Thomson_Refinitiv_Datastream_and_ESG.pdf)
- [Pinnacle CLC](https://pinnacledata2.com/clc.html)
- [Official author repository](https://github.com/kieranjwood/trading-momentum-transformer)
- [CFTC historical reports](https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm)
- [IBKR futures API rules](https://www.interactivebrokers.com/docs/tws-api/contracts/futures): continuous futures are historical-only, cannot place orders, require empty endDateTime; expired futures historical access is limited to two years after expiry.
