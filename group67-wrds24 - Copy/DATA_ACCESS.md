# Private ZIP revision

This ZIP includes the fixed snapshot and the three saved run archives.
Use START_HERE.md; external imports and the old author repository described
below are optional legacy routes for a code-only checkout.

# Data and experiment artifact access

No WRDS credentials are included. Licensed inputs are included only in the private course ZIP. An authorized WRDS
subscription to LSEG Datastream Futures Continuous Series Data is required to
obtain the underlying data. The project does not automate login or bypass access.

Original WRDS download: `gzh8fruv7i2jvqkj_csv.zip`, SHA-256
`96a13baca4b7d3bf49deb4c5f1a488079d5e3a3000a6007b7113285b01ef8a85`.
It contained 50 requested continuous series with expanded auxiliary metadata.
Duplicate (CalcSeriesCode, Date_) keys were collapsed after verifying agreement
of their core values and metadata. The 24 retained markets were selected by
validation/test coverage and provisional identity, not strategy performance.

In WRDS, use Datastream Continuous Series Info to inspect definitions, then
Future Series (`wrds_fut_series`) for daily observations. Use the codes in
`configs/config.yaml`. Request available history through 2025-12-31 and retain
roll metadata. Only 2010-2020 is consumed. Capture at least:

`CalcSeriesCode, Date_, Open_, High, Low, Volume, Settlement, OpenInterest,
LicFlag, ClsCode, DSMnem, CalcSeriesName, ISOCurrCode, ISOCurrDesc,
RollMethodCode, RollMethodDesc, PositionFwdCode, PositionFwdDesc, CalcMthCode,
trdmonths`.

Settlement/index levels are populated; OHLC, volume and open interest are not.
Roll method 4 is described as "As a price index", position 0 as "First", with
calculation method S. This does not establish equivalence to Pinnacle's
ratio-adjusted contracts or reconstruct executable rolls/dollar quantities.
Missing markets in the downloaded catalog are not proof of a licensing cause.

For the exact completed experiment, the immutable selected input snapshot and
universe metadata already reside in the team's author repository:

```bash
python scripts/01_get_data.py \
  --bundle-dir /mnt/c/Users/sunji/Downloads \
  --author-repo ~/paper67-author
```

This copies `data/raw/wrds_panel24/source_daily.csv` and `universe24.json` into
the new project without changing the source repository. `source_daily.csv` is
an immutable, deduplicated, 24-series input snapshot, not the untouched original
expanded WRDS query file. Source price levels have not been corrected or scaled.
In the organized project this derived snapshot is placed under
`data/interim/universe_snapshot/wrds_panel24/`, while untouched imported ZIP files
remain under `data/raw/artifacts/`. Full estimation exports analysis-ready feature
tables to `data/processed/wrds_panel24/`; generated compatibility-runtime copies
remain inside `data/interim/author_runtime/`.
Preserve the original WRDS ZIP separately for source inspection. An automated
reconstruction of this snapshot from an arbitrary future WRDS download is not
included; independent downloaders must reproduce the documented deduplication
and universe selection before estimation.

For report rebuilding, obtain the three exact saved experiment archives listed
in README.md through an authorized private course channel. Their hashes are
checked by `src/project/data.py`. They contain saved weights, feature inputs,
segmented index levels and prediction records; these remain private data-bearing
artifacts. A public code checkout alone cannot reproduce the numbers without
separate access to those inputs. This dependency is explicit.

Exclusions, conservative calendar restrictions and independent segment resets
are recorded in run configurations and `report/phase1_results.md`. These are
adaptation choices, not a verified universal exchange calendar.
