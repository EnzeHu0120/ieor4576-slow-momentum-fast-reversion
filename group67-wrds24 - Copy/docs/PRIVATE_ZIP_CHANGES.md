# Private ZIP packaging changes

This revision adds the fixed 24-series input snapshot and the three exact saved
experiment archives to the course submission. The small code-only archive and
its previous validation statements excluded private inputs; those statements
refer to that earlier artifact. The scientific report and fitted models are
unchanged. START_HERE.md is the authoritative entry point for this private ZIP.

The code now verifies snapshot hashes, the 24-market universe, unique market/date
keys, parsed dates and nonmissing index levels before preparing features.
`--stage prepare` exposes baseline feature preparation. Full estimation uses
project-relative included inputs instead of requiring a user's old checkout.

Remaining boundaries: historical selection from the original expanded WRDS
query is not reconstructed by this package; the fixed selection is supplied.
Full GPU training is not rerun for packaging. The five-asset full estimation
code is separately maintained in the linked team repository. No causal market-
count experiment or reserved-period extension is claimed.
