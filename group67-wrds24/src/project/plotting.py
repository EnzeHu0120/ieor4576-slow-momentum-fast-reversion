"""Publication figures based on recomputed conditional return series."""
from .analysis_math import plot_results

def build(raw,scaled,cost_table,output):
    # The validation-selected curve is identical to its chosen fixed-lookback
    # curve. Show it once in the figure; preserve every row in the tables.
    table=cost_table.copy()
    reference=table[table.strategy.eq('lstm_cpd_validated')].sort_values('cost_bps')
    for name,g in table.groupby('strategy'):
        if name.startswith('lstm_cpd_') and name!='lstm_cpd_validated':
            if g.sort_values('cost_bps').sharpe_ratio.tolist()==reference.sharpe_ratio.tolist():
                table=table[~table.strategy.eq(name)]
    plot_results(raw,scaled,table,output)
