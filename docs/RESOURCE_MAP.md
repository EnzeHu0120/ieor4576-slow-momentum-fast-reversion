# Sources and local resource organization

Public shared sources are recorded in UPSTREAM.json with exact commit and file hashes. Full upstream Git history remains in sibling `upstream_history/*.git`. Source snapshots retain licenses and authorship.

Ignored `resources/local/` contains the exact uploaded ProjectTemplate.pdf, primary/supporting papers, prior initial proposal, original course project guidelines/catalog/repository hierarchy. Keep course documents for reading; they are evidence of course requirements, not permission to publish data, change credentials or trade.

The Enze worktree adds an ignored `resources/personal/` collection:
- original classroom materials from “data-driven approaches used in modern finance”;
- selected IEORE4576 Week 1–4 resources and notebook source cells, with outputs removed;
- existing ten-series raw futures cache copied into project-owned data storage, leaving old assignments untouched;
- RegimeDetection source/docs/reports and a link to the original large CSV folder; no environment/build/caches copied.

Relevant concepts: futures roll construction and negative prices (Weeks 1–2); WRDS metadata and data provenance (Weeks 2–3); continuous futures returns (Week 3, independently rebuilt for this pilot); parquet/session handling/IC and mutual information (Week 4). CRSP equities, COT reports and Nasdaq ITCH do not substitute for the daily diversified futures panel. Heavy ITCH data, S3 credentials, browser cookies and temporary work folders are excluded.

Teammate screenshot/files not present in these local folders have not been invented. The previous proposal is retained as a preliminary plan, and the verified course catalog takes precedence for paper identity. The teammate's identity/branch can be added when provided.
