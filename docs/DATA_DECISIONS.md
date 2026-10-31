# Data decisions

Source: WRDS LSEG/Datastream `trdstrm.wrds_contract_info` and `trdstrm.wrds_fut_contract`. Target: 1990–2020. The instrument mapping contains all 50 Appendix A entries and their published first-test dates; exchange-session/composite substitutions are explicit.

Raw contract-day observations: 4,321,070. Final feature rows: 346,218. Raw downloads are never changed by preprocessing and are not committed. Each instrument has source/query dates and SHA-256 manifests. Three pit-contract calendars supplement missing composite schedules by underlying/delivery month, with separate provenance hashes. They do not provide future-bar-inferred expiry dates.

The convenience continuous-series catalog omitted several required composite series, so the study uses actual individual-contract histories. A same-contract return index removes cross-contract settlement gaps. Calendar rolls and local-currency notional returns differ from Pinnacle's proprietary series. Futures roll principal is not a cash exchange; both trade legs incur fees in the extension.

Missing volume/OI is not a price-missing criterion. There are 138 causally carried settlement marks, counted by instrument; they are not assumed to be executable fills. Missing-market observations contribute cash under the fixed per-fold denominator. Discontinued NR is limited to the available source; it is not invented after 2015. Euro history starts in 1998, GSCI in 1992, and other later starts appear in `results/replication/tables/data_coverage.csv`. Earlier data in unrelated products is not claimed to be universally unavailable.

All source data are retrieved as a current historical vintage, not a point-in-time vendor archive. Scheduled notice/expiry metadata are assumed known before trading. Individual futures price changes use the same maturity at both endpoints, so the April 2020 WTI roll cannot book the near/far basis as P&L. A USD P&L conversion overlay and contract-lot capacity analysis are not included.

The author code winsorizes its price input and constructs labels from the resulting index. Original-style tables reproduce that convention; raw-return and capped/delayed scenarios disclose its tradability implications. Fixed costs are scenarios, including the professor's 2.5-bps reference, rather than measured bid/ask or impact estimates.
