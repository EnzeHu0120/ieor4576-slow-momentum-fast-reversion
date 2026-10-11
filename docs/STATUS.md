# Initialization status — 2026-10-11

Implemented and checked: pinned complete author source; separate modern research environment; shared data-selection preview (personal WRDS download interface belongs only to local Enze); data contracts; past-only features; explicit fold/label purge; TSMOM exposure/cost engine; roll costs; synthetic end-to-end output; tests of timing, future perturbation, fees, missing held returns and contract returns.

Not yet completed: team data decision; exact Pinnacle dataset; live WRDS access; compatible author TF/GPflow training environment; LSTM/TFT fitting; CPD runs; paper tables; fitted-model robustness; actual broker execution or realized costs. The proposal is a planning artifact, not evidence these steps have happened.

## Internal milestones, not confirmed course deadlines

1. Week 1: agree data, inspect entitlement, freeze train/validation/test protocol, one-model smoke run.
2. Weeks 2–3: baseline LSTM and TFT fitting, correct preprocessing and record compute/seeds.
3. Weeks 4–5: CPD ablation, common-data net-cost comparisons, risk/regime diagnostics.
4. Weeks 6–7: optional cost-aware objective and post-2020 holdout if core passes.
5. Week 8: reproducibility run, results dashboard/framework, research slides and limitations.

The supplied Project_Guidelines.docx calls for a roughly four-slide midterm proposal deck and a working modular trading-system pipeline, including costs/risk/robustness. The separate ProjectTemplate.pdf governs the initial proposal. No final presentation due date was found in the local guidelines, so earlier proposal calendar dates remain provisional; use the live CourseWorks assignment dates when available.
