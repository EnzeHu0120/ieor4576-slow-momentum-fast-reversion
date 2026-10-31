"""Generate and execute the final research notebook from saved real experiments."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import nbformat as nbf
import pandas as pd
from .paths import ROOT,configuration

def experiment_inventory():
    rows=[]
    for path in sorted((ROOT/"results/replication/checkpoints").glob("*/*/receipt.json")):
        d=json.loads(path.read_text());t=d["trials"]
        rows.append({"strategy":d["strategy"],"fold":d["fold"],
            "instruments":len(d["eligible_instruments"]),"trials":len(t),
            "total_epochs":sum(x["epochs"] for x in t),"fit_seconds":sum(x["seconds"] for x in t),
            "best_validation_sharpe":-d["validation_loss"],"best_trial":d["best_trial"],
            "test_end_exclusive":d["test_end_exclusive"],"oos_asset_days":d["rows"]})
    return pd.DataFrame(rows)

def inventory_manifest(cfg,config_path="configs/config.yaml"):
    import pyarrow.parquet as pq
    experiments=experiment_inventory();experiments.to_csv(ROOT/"results/replication/tables/experiment_inventory.csv",index=False)
    cpd={str(l):sum(pq.read_metadata(p).num_rows for p in (ROOT/f"data/processed/cpd/{l}").glob('*.parquet')) for l in cfg["cpd_windows"]}
    cpd_diagnostics=[]
    for l in cfg["cpd_windows"]:
        paths=list((ROOT/f"data/processed/cpd/{l}").glob('*.parquet'))
        if paths:
            g=pd.concat([pd.read_parquet(p,columns=[f'cp_score_{l}',f'cp_rl_{l}','iterations','converged']) for p in paths],ignore_index=True)
            cpd_diagnostics.append({'window':l,'rows':len(g),'optimizer_converged_fraction':g.converged.mean(),
                'mean_iterations':g.iterations.mean(),'mean_severity':g[f'cp_score_{l}'].mean(),
                'mean_relative_location':g[f'cp_rl_{l}'].mean()})
    pd.DataFrame(cpd_diagnostics).to_csv(ROOT/'results/replication/tables/cpd_diagnostics.csv',index=False)
    expected={(name,y) for name in ["LSTM"]+[f"LSTM-CPD-{l}" for l in cfg["cpd_windows"]] for y in cfg["test_boundaries"]}
    compatible=experiments[experiments.trials.eq(cfg["random_search_trials"])] if len(experiments) else experiments
    actual=set(zip(compatible.strategy,compatible.fold)) if len(compatible) else set()
    selected=Path(config_path)
    if not selected.is_absolute():selected=ROOT/selected
    files=[selected,ROOT/"configs/universe.csv",ROOT/"configs/extension.yaml",ROOT/"data/processed/features.parquet"]
    result={"paper":"2105.13727v3","generated_utc":datetime.now(timezone.utc).isoformat(),
        "config":cfg,"completed_experiments":len(actual),"expected_experiments":len(expected),
        "missing_experiments":[{"strategy":s,"fold":y} for s,y in sorted(expected-actual)],
        "completed_trials":int(experiments.trials.sum()) if len(experiments) else 0,
        "completed_epochs":int(experiments.total_epochs.sum()) if len(experiments) else 0,
        "cpd_daily_rows":cpd,"hashes":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "numerical_replication":f"WRDS data adaptation; {cfg['random_search_trials']} random-search trials versus the published 50; PyTorch/Numba ports",
        "synthetic_results":False}
    (ROOT/"report/execution_manifest.json").write_text(json.dumps(result,indent=2))
    return result

def conclusion_text():
    comparison=pd.read_csv(ROOT/"results/replication/tables/published_comparison.csv").set_index("strategy")
    rows=[]
    if "LSTM" in comparison.index and "LSTM-CPD-Optimised" in comparison.index:
        a=comparison.loc["LSTM","sharpe_reproduced"];b=comparison.loc["LSTM-CPD-Optimised","sharpe_reproduced"]
        rows.append(f"The mean five-window gross Sharpe is {a:.3f} for LSTM and {b:.3f} for the validation-selected CPD model (difference {b-a:+.3f}). The published figures are 1.62 and 2.16. The datasets, roll conventions and numerical backends differ, so their numerical difference is not an implementation error estimate.")
    costs=pd.read_csv(ROOT/"results/replication/tables/transaction_costs.csv")
    selected=costs[costs.cost_bps.eq(2.5)&costs.strategy.isin(["TSMOM w=0","LSTM","LSTM-CPD-Optimised"])]
    if len(selected):rows.append("At 2.5 one-way bps, original-style turnover-adjusted mean Sharpe is "+", ".join(f"{r.strategy}: {r.sharpe:.3f}" for r in selected.itertuples())+". This is a cost scenario, not an estimate of broker fees or market impact.")
    extension_path=ROOT/'results/extension/tables/roll_costs.csv'
    if extension_path.exists():
        raw=pd.read_csv(extension_path)
        raw=raw[raw.cost_bps.eq(2.5)&raw.scenario.eq('raw returns plus rolls')&raw.strategy.isin(['TSMOM w=0','LSTM','LSTM-CPD-Optimised'])]
        if len(raw):rows.append('Using un-winsorized realized returns and charging both roll legs at 2.5 bps, mean net Sharpe is '+', '.join(f'{r.strategy}: {r.sharpe:.3f}' for r in raw.itertuples())+'. These figures assess tradability separately from the author-style return labels; daily quotes still do not establish executable spreads or capacity.')
    return "\n\n".join(rows)

METHODS="""The study is Wood, Roberts & Zohren (2022), *Slow Momentum with Fast Reversion* ([paper](https://arxiv.org/abs/2105.13727)), not the Momentum Transformer study. The reference list's item number is inconsistent with the requested title; this reproduction follows the title and arXiv identifier.

For each instrument, positions are selected at close t and evaluated against t-to-t+1 returns. The raw futures panel rolls five business days before the earliest vendor first-notice/last-trade date, selects the nearest eligible maturity with an observed positive settlement, and computes each day's price change within the previously selected contract. Missing historical volume/OI is not treated as a market inception date. For GI, ZW and ZZ, missing composite expiry metadata is supplemented from the vendor's pit-contract calendar for the identical underlying and delivery month. No expiration is inferred from the last observed price. Missing settlements are valued causally at the last observed mark and explicitly counted; these marks are not assumed to be executable fills. An index compounded from these real returns is the input to the authors' preserved feature code.

The features are 1/21/63/126/252-session volatility-normalized returns and MACD pairs (8,24), (16,48), (32,96). The classical MACD baseline maps the mean of the three normalized trend estimates through the author's separate `scale_signal` function, phi(y)=y exp(−y²/4)/0.89, as in equation (7) of the [foundational DMN reference](https://arxiv.org/html/1904.04912). The neural MACD inputs remain untransformed. The author implementation winsorizes the price index at five EWM standard deviations with half-life 252; daily volatility uses a 60-session EWM span. The original-style evaluation uses that implementation's winsorized return labels. The extension evaluates the positions on raw, un-winsorized same-contract returns as a separate tradability diagnostic.

For each trailing l+1 observation window, GP-CPD fits a stationary Matérn 3/2 covariance and a sigmoid-weighted pair of Matérn kernels by L-BFGS-B (maximum 200 iterations). Windows are reinitialized independently, with all stationary parameters set to one; the fitted stationary kernel initializes the changepoint model. Severity is sigmoid(NLML_M − NLML_CP). Relative location follows the author's code, (t−c)/l, which is the complement of the formula printed in equation (10). The Numba Kalman likelihood is algebraically equivalent to the dense GP covariance. Numerical stabilization adds 1e-8 diagonal jitter and bounded log parameters.

The stateless LSTM has 63 time steps, tanh position outputs, input and output dropout, and the pooled negative Sharpe objective on asset-time observations. Training uses non-overlapping sequences, shuffled sequence order, Adam, parameter-wise gradient norm clipping, a maximum output weight norm of three, up to 300 epochs, and 25-epoch validation patience. The PyTorch port freezes its redundant recurrent bias to match Keras' single bias. Backend initialization and optimizer arithmetic need not be bitwise identical to TensorFlow.

Training expands from 1990, with per-instrument chronological 90/10 train/validation splits. Labels crossing a split boundary are purged. An instrument needs at least one 63-step validation sequence and must satisfy Appendix A's first-test eligibility. Test windows are 1995–1999, 2000–2004, 2005–2009, 2010–2014, and 2015–2019; 2020 is reported as a separate frozen one-year test after a pre-2020 refit. This resolves the manuscript's ambiguous '1995–2020' endpoint explicitly; the 2020 window is excluded from the five-window exhibit averages.

All five CPD horizons (10,21,63,126,252) and the no-CPD LSTM are fitted. The main configuration requests 50 distinct random draws from the full published grid per model/window, using automatic CUDA/Metal/CPU selection and pre-test pooled validation Sharpe for selection. The fresh notebook run trains every candidate from random initialization; older model checkpoints are archived and are not used. The paper specifies 50 random-search iterations; the supporting later author repository defaults to diversified validation Sharpe. The inventory and manifest state the executed search count; a configuration file is not evidence that its experiments have run. One complete search repetition is conducted, with base seed 67 and per-candidate seeds 67+trial. Repeated-seed experiments are not represented as executed. Optimized CPD means selecting the lowest validation loss among the five completed CPD models, independently for each test window.

Exhibits 3 and 4 average each metric across the five main test windows. Annual return is compounded CAGR; Sharpe uses arithmetic mean excess P&L and sample volatility. The portfolio uses equal 1/N weights across the fold's eligible instruments, with no-return dates contributing cash. Exhibit 4 follows the later author aggregation code: use one factor, 15% divided by the mean annualized volatility of the five main windows, on the whole daily return series, then recompute each window's metrics. This targets 15% mean window volatility, not 15% in every window. The rounded printed paper is also consistent with linear scaling of averaged dimensional metrics; that display diagnostic is saved separately, because linear scaling of CAGR or drawdown is not identical to recomputation on compounded returns. These ex-post multipliers are not investable risk forecasts. No Sharpe is assumed to improve.

The data substitute is WRDS LSEG/Datastream individual-contract history, not the proprietary Pinnacle CLC panel. Session conventions, roll construction, starting dates and source revisions differ. Appendix A's NR and ZR series remain distinct; discontinued NR is not backfilled. The Euro series has no fabricated pre-euro/Deutsche-Mark extension. Several other markets start later than 1990; these differences and causal stale settlement marks are explicit in the coverage table. Raw futures P&L is expressed relative to each contract's local-currency price; a USD conversion overlay and integer-lot sizing are outside this reproduction. No unavailable Pinnacle values are imputed.
"""

def build_notebook(cfg,manifest,run_pipeline=True,config_path='configs/config.yaml'):
    n=nbf.v4.new_notebook();cells=[]
    def md(s):cells.append(nbf.v4.new_markdown_cell(s))
    def code(s):cells.append(nbf.v4.new_code_cell(s))
    md("# Paper 67 — Original Study Replication\n\n**Enze Hu · Jinrui Hu | IEOR 4576 | Wood, Roberts & Zohren (2022)**\n\nActual WRDS experiments; a documented data and compute adaptation. No synthetic performance estimates.\n\n"+f"Saved experiments: **{manifest['completed_experiments']}/{manifest['expected_experiments']}**; fitted trials: **{manifest['completed_trials']}**.\n\n"+"The published 1.62/2.16 Sharpe values are comparison targets, not reproduced results.")
    code("from pathlib import Path\nimport sys, json\nimport pandas as pd\nfrom IPython.display import display, Image, Markdown\nROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'src/momentum_lab').is_dir())\nsys.path.insert(0, str(ROOT/'src'))\nfrom momentum_lab.reporting import conclusion_text\npd.options.display.max_rows = 60\npd.options.display.max_columns = 15\npd.options.display.float_format = '{:.4f}'.format\nT = ROOT/'results/replication/tables'\nF = ROOT/'results/replication/figures'\nE = ROOT/'results/extension/tables'\nmanifest = json.loads((ROOT/'report/execution_manifest.json').read_text())\nprint('Real experiments:', manifest['completed_experiments'], 'Trials:', manifest['completed_trials'])\nprint('Daily CPD fits by window:', manifest['cpd_daily_rows'])")
    md("## Original methodology and implemented conventions\n\n"+METHODS)
    md("## Data coverage\n\nThe full Appendix A mapping is in `configs/universe.csv`. Raw downloads and manifests are local and excluded from version control. A later first date below describes the selected WRDS contract series, not proof that no earlier data exists in any WRDS product.")
    code("coverage = pd.read_csv(T/'data_coverage.csv')\ndisplay(coverage[['ticker','feature_days','first','last','rolls','stale_mark_days','calendar_reference_fills','equivalence']])")
    md("## Completed expanding-window experiments\n\nAll hyperparameter choices use pre-test validation observations. The final 2020-only test is separate from the five-window exhibits. `fit_seconds` is elapsed process time and includes user-requested pauses; it is not active compute time.")
    code("inventory = pd.read_csv(T/'experiment_inventory.csv')\ndisplay(inventory)\ndisplay(pd.read_csv(T/'cpd_diagnostics.csv'))\nassert not manifest['missing_experiments'], manifest['missing_experiments']")
    md("## Exhibit 3 — Raw signal output\n\nMetrics below come from saved out-of-sample positions. Percentage columns use fractional units (0.15 = 15%).")
    code("exhibit3 = pd.read_csv(T/'exhibit_3.csv').set_index('strategy')\ndisplay(exhibit3)")
    md("## Exhibit 4 — Rescaled to 15% volatility\n\nThis follows the author aggregation code: a single ex-post factor per strategy targets 15% mean main-window volatility; CAGR and drawdown are recomputed on the scaled daily returns. It is not a tradable volatility forecast.")
    code("display(pd.read_csv(T/'exhibit_4.csv').set_index('strategy'))\ndisplay(pd.read_csv(T/'volatility_rescaling.csv'))\nprint('Separate linear display approximation; these are transformed summary metrics, not recomputed strategy CAGR/MDD:')\ndisplay(pd.read_csv(T/'exhibit_4_linear_display.csv').set_index('strategy'))\ndisplay(Image(filename=str(F/'wealth_comparison.png')))")
    md("## Comparison with the published study\n\nThe 50-candidate search budget matches the paper. Data source, historical coverage, roll construction, random-search realization and numerical backend differ. A failure to recover the published improvement is a substantive empirical outcome.")
    code("comparison = pd.read_csv(T/'published_comparison.csv')\ndisplay(comparison[['strategy','sharpe_published','sharpe_reproduced','sharpe_difference','test_windows','status']])\ndisplay(Image(filename=str(F/'published_comparison.png')))\ndisplay(Image(filename=str(F/'cpd_window_sensitivity.png')))\ndisplay(pd.read_csv(T/'selected_cpd_windows.csv'))")
    md("## Appendix C — Transaction costs\n\nCosts are in one-way bps of traded notional. Position reversal, volatility resizing, initial entry and final exit are charged. As in Appendix C, the core table uses changes in scaled exposure; the extension adds both futures-roll legs. The vertical line marks the professor's 2.5-bps scenario. Costs are hypothetical stress levels, not measured liquidity or impact estimates.")
    code("costs = pd.read_csv(T/'transaction_costs.csv')\ndisplay(costs.pivot(index='strategy',columns='cost_bps',values='sharpe'))\ndisplay(Image(filename=str(F/'transaction_costs.png')))")
    md("## Futures rolls and execution sensitivity\n\nReturn endpoints always belong to the same contract. On a roll, the near/far settlement gap creates no cash flow. Both old and new traded exposures pay costs. The raw-return and delayed-signal scenarios below use the same already-fitted positions, without retraining on test data. Bid/ask spreads, market impact and integer-lot sizing are not identified by daily settlements.")
    code("roll = pd.read_csv(E/'roll_costs.csv')\ndisplay(roll[roll.cost_bps.eq(2.5)][['strategy','scenario','sharpe','annual_return','max_drawdown','annual_turnover','annual_roll_extra']])\ndisplay(pd.read_csv(E/'break_even_costs.csv'))\ndisplay(Image(filename=str(ROOT/'results/extension/figures/roll_and_execution_costs.png')))")
    md("## Risk and regime sensitivity\n\nThe 2020 crisis is displayed separately. Position traces use frozen OOS forecasts. A paired 63-session moving-block bootstrap is conditional on the fitted models and does not quantify retraining uncertainty.")
    code("display(Image(filename=str(F/'regime_sensitivity.png')))\ndisplay(Image(filename=str(F/'drawdowns.png')))\ndisplay(pd.read_csv(E/'paired_sharpe_bootstrap.csv'))\ndisplay(Image(filename=str(F/'position_diagnostics.png')))\ndisplay(pd.read_csv(T/'annual_metrics.csv').query('year == 2020'))")
    md("## Conclusions and limitations")
    code("display(Markdown(conclusion_text()))")
    budget_note=("The complete 50-trial original search has not been run in this local profile. " if cfg['random_search_trials']<50 else "")
    md(budget_note+"Repeated-seed significance has not been evaluated. Pinnacle's original histories and exact session/roll construction are unavailable in the connected sources. The daily close-t execution assumption is optimistic; the extension reports delayed-signal sensitivity. Fixed bps cannot establish implementable capacity, and MACD usefulness is not independently established by this reproduction. The five-window averaging convention and separate 2020 endpoint are fully specified above.")
    md("## Reproduction and provenance\n\nFrom the prepared repository environment: `make reproduce`. For a new model fit from random initialization: `make fresh`. The final notebook loads reusable module outputs; it is not the source of experiment logic.\n\nSources: [original paper](https://arxiv.org/abs/2105.13727), [author GP/feature code](https://github.com/kieranjwood/slow-momentum-fast-reversion), [supporting author LSTM code](https://github.com/kieranjwood/trading-momentum-transformer). Preserved MIT licenses and source hashes are recorded in `UPSTREAM.json`. Course requirements and graded feedback were read from the supplied reference folder.\n\nThe report manifest records completed runs and feature/configuration hashes. Licensed market data and local credentials are excluded from Git.")
    code("print(json.dumps({k:v for k,v in manifest.items() if k not in ['config']}, indent=2))")
    n.cells=cells;n.metadata={"kernelspec":{"name":"python3","display_name":"Python (replication environment)","language":"python"},
        "language_info":{"name":"python","version":sys.version.split()[0]}}
    for cell in n.cells:
        cell.metadata['slideshow']={'slide_type':'slide' if cell.cell_type=='markdown' else '-'}
    path=ROOT/"report/Paper67_Original_Study_Replication.ipynb"
    complete=manifest['completed_experiments']==manifest['expected_experiments']
    n.cells[0].source=("# Paper 67 — Original Study Replication\n\n"
        "**Enze Hu · Jinrui Hu | IEOR 4576 | Wood, Roberts & Zohren (2022)**\n\n"
        "完整研究流程已编写 / Complete runnable research workflow. Select the project's `.venv-replication/bin/python` kernel and **Run All**. "
        "The execution cell runs the full real-data study before the result cells run. It can remain busy for hours; no assistant session is needed.\n\n"
        f"Checkpoint snapshot when written: **{manifest['completed_experiments']}/{manifest['expected_experiments']}** searches; **{manifest['completed_trials']}** candidate fits. "
        +("The saved study is complete." if complete else "The remaining experiments have not yet run. This file does not claim final executed results.")+
        "\n\nNo synthetic performance estimates. Published Sharpe values 1.62/2.16 are comparison targets.")
    if run_pipeline:
        n.cells[0].source=("# Paper 67 — Original Study Replication\n\n"
            "**Enze Hu · Jinrui Hu | IEOR 4576 | Wood, Roberts & Zohren (2022)**\n\n"
            "完整原论文复现 / Full original-study reproduction. Select a Python environment installed from `pyproject.toml`, then **Run All**. "
            "默认 `FRESH_MODELS=True`：所有 LSTM / CPD-LSTM 候选从随机初始化训练，不加载旧权重；旧检查点自动归档。训练 cell 下实时更新训练/验证曲线。\n\n"
            "The notebook invokes reusable source modules. It does not read `Paper67_Progress_Executed.ipynb`. Raw data and deterministic causal feature caches may be reused. "
            "On a new clone, missing raw data are downloaded through your own authorized WRDS account and missing features are computed. "
            "No fitted checkpoints are required. No final results are claimed before execution completes.")
    n.cells[1].source=("from pathlib import Path\nimport sys, json\nimport pandas as pd\nfrom IPython.display import display, Image, Markdown\n"
        "ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'src/momentum_lab').is_dir())\n"
        "sys.path.insert(0, str(ROOT/'src'))\n"
        "from momentum_lab.paths import initialize_notebook_environment\ninitialize_notebook_environment()\n"
        "from momentum_lab.reporting import conclusion_text\n"
        "pd.options.display.max_rows = 60\npd.options.display.max_columns = 15\npd.options.display.float_format = '{:.4f}'.format\n"
        "T = ROOT/'results/replication/tables'\nF = ROOT/'results/replication/figures'\nE = ROOT/'results/extension/tables'\n"
        "print('Project:', ROOT)\nprint('Notebook kernel:', sys.executable)")
    n.cells[2:2]=[
        nbf.v4.new_markdown_cell("## Run the complete local reproduction\n\n"
            "设 `RUN_REPRODUCTION=True` 运行全部 50-candidate searches，再完成回测、交易成本、Exhibits 3/4、图和 PDF 报告。"
            "`FRESH_MODELS=True` 会归档旧权重并从零训练；仅在意外中断后想接着本次运行时改为 `False`。"
            "这一步使用本机 CPU/GPU，自动选择 CUDA、Metal 或 CPU；不调用模型 API，不需要助手在线。首次 clone 缺少数据时会使用你自己的 WRDS 授权下载。不要同时启动另一个训练命令。"
            "保持电脑唤醒；这一格长时间显示 `[*]` 是正常训练。日志保存在 `results/replication/logs/local_notebook_run.log`。"
            "训练中断后重新运行会保留已完成的 candidate；当时尚未完成的 candidate 会重跑。\n\n"
            "Once all outputs exist, set `RUN_REPRODUCTION=False` to read them without training. After Run All finishes, **save the notebook** to retain every output."),
        nbf.v4.new_code_cell(f"CONFIG = {str(config_path)!r}\nMODEL_WORKERS = 3\nRUN_REPRODUCTION = {run_pipeline!r}\nFRESH_MODELS = True\n\n"
            "if RUN_REPRODUCTION:\n"
            "    from momentum_lab.notebook_runner import run_prepared\n"
            "    manifest = run_prepared(CONFIG, model_workers=MODEL_WORKERS, fresh_models=FRESH_MODELS)\n"
            "else:\n"
            "    manifest = json.loads((ROOT/'report/execution_manifest.json').read_text())\n"
            "    assert not manifest['missing_experiments'], 'Experiments are incomplete: set RUN_REPRODUCTION=True.'\n"
            "print('Completed searches:', manifest['completed_experiments'], '/', manifest['expected_experiments'])\n"
            "print('Candidate fits:', manifest['completed_trials'])\n"
            "print('Daily CPD rows:', manifest['cpd_daily_rows'])")]
    n.metadata['replication_status']={'complete':complete,'runnable':True,'pipeline_enabled':run_pipeline}
    return path,n

def execute_notebook(path,notebook):
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    build=ROOT/"report/.build";build.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR",str(build/"matplotlib"))
    os.environ.setdefault("IPYTHONDIR",str(build/"ipython"))
    os.environ.setdefault("JUPYTER_RUNTIME_DIR",str(build/"jupyter"))
    manager=KernelManager(kernel_name="python3")
    manager.kernel_spec.argv=[sys.executable,"-m","ipykernel_launcher","-f","{connection_file}"]
    client=NotebookClient(notebook,timeout=600,kernel_name="python3",resources={"metadata":{"path":str(ROOT)}},km=manager)
    result=client.execute(cleanup_kc=True)
    temporary=path.with_suffix('.ipynb.tmp');nbf.write(result,temporary);temporary.replace(path)
    from nbconvert import HTMLExporter
    exporter=HTMLExporter();exporter.exclude_input=False
    body,_=exporter.from_notebook_node(result)
    temporary_html=path.with_suffix('.html.tmp');temporary_html.write_text(body);temporary_html.replace(path.with_suffix('.html'))
    return result

def generate(config_path="configs/config.yaml"):
    cfg=configuration(config_path);manifest=inventory_manifest(cfg,config_path)
    if manifest["missing_experiments"]:raise RuntimeError("Final notebook cannot be finalized while expected experiments are missing")
    path,notebook=build_notebook(cfg,manifest,run_pipeline=False,config_path=config_path)
    execute_notebook(path,notebook)
    (ROOT/"report/RESULTS.md").write_text("# Paper 67 — Executed results\n\n"+conclusion_text()+"\n\n"+METHODS)
    from .deliverables import deliver
    deliver()
    print("Executed",path.relative_to(ROOT),flush=True)
    return manifest

def prepare_runnable(config_path='configs/config.yaml'):
    """Write the entire workflow now; real results populate when the user runs it."""
    from .progress import status
    import shutil
    cfg=configuration(config_path);snapshot=status()
    manifest={'completed_experiments':snapshot['completed_searches'],
        'expected_experiments':snapshot['expected_searches'],'completed_trials':snapshot['completed_candidates']}
    path,notebook=build_notebook(cfg,manifest,run_pipeline=True,config_path=config_path)
    if path.exists():
        old=nbf.read(path,as_version=4)
        if old.metadata.get('replication_status',{}).get('runnable') is not True:
            shutil.copy2(path,path.with_name('Paper67_Progress_Executed.ipynb'))
            if path.with_suffix('.html').exists():
                shutil.copy2(path.with_suffix('.html'),path.with_name('Paper67_Progress_Executed.html'))
    nbf.write(notebook,path)
    from nbconvert import HTMLExporter
    body,_=HTMLExporter().from_notebook_node(notebook)
    path.with_suffix('.html').write_text(body)
    print('Complete runnable notebook prepared (pending cells intentionally unexecuted):',path,flush=True)
    return snapshot
