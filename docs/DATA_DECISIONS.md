# Input data and output conventions

The five instruments are S&P 500 day-session futures (SP), US 10-year Treasury-note futures (TY), British-pound futures (BN), gold (ZG) and crude oil (ZU). Vendor identifiers and session mappings are in configs/universe.csv. Selection reflects coverage and asset classes, not test results.

WRDS LSEG/Datastream individual-contract histories are used. Calendar rolls occur five business days before the earliest scheduled notice/last-trade boundary; each return uses the same held contract at both endpoints. Cross-contract price gaps are not profit. Returns are compounded into a price index and passed through the preserved author feature functions. Missing settlements are marked causally and counted; fabricated historical prices are not supplied.

Feature/label observations retain contiguous next-observation targets and published first-test eligibility. Data coverage and selected vendor mappings are in report/tables/replication/data_panel_audit.csv. These differ from the Pinnacle continuous futures, session conventions and data vintages in the paper. Warmup means the usable feature history starts later than the requested raw history.

The main evaluation follows the author's winsorized price-index return labels. Raw-return and both-roll-leg cost accounting are separate diagnostics, not replacements for the gross result. No USD conversion overlay, integer-lot capacity model or measured spread/impact estimate is claimed. Fixed-bps costs are scenarios.

Only aggregate metrics and figures are public. Licensed observations, individual-contract quotes, fitted weights and credentials are excluded. File hashes identify the executed inputs without distributing them.
