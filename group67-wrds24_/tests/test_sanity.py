import numpy as np
import pandas as pd
import unittest
from project.analysis_math import net_on_common_intervals,rescale_ex_post,metrics
from project.data import extract_archive

class SanityTests(unittest.TestCase):
    def test_turnover_keeps_reversal_on_omitted_day(self):
        rows=[]
        for asset in range(24):
            for day,position in [(1,1.),(2,-1.),(3,1.)]:
                rows.append({'segment':str(asset),'target_date':pd.Timestamp(2020,1,day),
                             'daily_vol':.15/np.sqrt(252),'target_returns':.01,'example_position':position})
        full=pd.DataFrame(rows)
        common=full[full.target_date.dt.day.isin([1,3])][['segment','target_date']]
        _,turn,net,_=net_on_common_intervals(full,common,['example'])
        np.testing.assert_allclose(turn.example,[0,2])
        np.testing.assert_allclose(net.example_5bps,[.01,.009])

    def test_positive_risk_scaling_preserves_sharpe(self):
        raw=pd.DataFrame({'example':[.01,-.02,.015,-.005,.008]})
        scaled,_=rescale_ex_post(raw)
        self.assertAlmostEqual(scaled.example.std(ddof=1)*np.sqrt(252),.15)
        self.assertAlmostEqual(metrics(raw.example)['sharpe_ratio'],metrics(scaled.example)['sharpe_ratio'])

    def test_archive_cannot_escape_its_destination(self):
        import zipfile
        from tempfile import TemporaryDirectory
        from pathlib import Path
        with TemporaryDirectory() as folder:
            root=Path(folder);source=root/'invalid.zip'
            with zipfile.ZipFile(source,'w') as z:z.writestr('../escape.txt','bad')
            with self.assertRaises(ValueError):extract_archive(source,root/'inside')

if __name__=='__main__':unittest.main()
